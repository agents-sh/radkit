//! State management for agent tasks and skill handlers.
//!
//! This module provides state types that carry execution state across
//! skill handler invocations during agent task execution.
//!
//! # Overview
//!
//! - [`State`]: Unified state container with scoped accessors
//! - [`TaskState`]: Task-scoped state (per `task_id`) for multi-turn within a skill
//! - [`SessionState`]: Session-scoped state (per `context_id`) for cross-skill workflow
//! - [`ProgressSender`]: Streaming updates to client
//!
//! # Legacy Types (deprecated)
//!
//! - [`Context`]: Immutable execution context (use `runtime.current_user()` instead)
//! - [`TaskContext`]: Mutable context (use `State` instead)
//!
//! # Examples
//!
//! ```ignore
//! use radkit::runtime::state::{State, ProgressSender};
//!
//! // In a skill handler
//! async fn on_request(
//!     &self,
//!     state: &mut State,
//!     progress: &ProgressSender,
//!     runtime: &dyn AgentRuntime,
//!     content: Content,
//! ) -> Result<OnRequestResult, AgentError> {
//!     // Task-scoped state (for this skill's multi-turn)
//!     state.task().save("partial_data", &data)?;
//!
//!     // Session-scoped state (shared across skills)
//!     state.session().save("user_data", &user_data)?;
//!
//!     // Streaming updates
//!     progress.send_update("Processing...").await?;
//! }
//! ```

#[cfg(feature = "runtime")]
use crate::agent::Artifact;
use crate::errors::AgentError;
#[cfg(feature = "runtime")]
use crate::errors::AgentResult;
#[cfg(feature = "runtime")]
use crate::models::utils;
#[cfg(feature = "runtime")]
use crate::models::{Content, Role};
#[cfg(feature = "runtime")]
use crate::runtime::core::event_bus::TaskEventBus;
#[cfg(feature = "runtime")]
use crate::runtime::core::status_mapper;
#[cfg(feature = "runtime")]
use crate::runtime::task_manager::{TaskEvent, TaskManager};
#[cfg(feature = "runtime")]
use a2a_types::{TaskArtifactUpdateEvent, TaskState as A2ATaskState, TaskStatus};
#[cfg(feature = "runtime")]
use chrono::Utc;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::HashMap;
#[cfg(feature = "runtime")]
use std::sync::Arc;

// ============================================================================
// New State Types
// ============================================================================

/// Session-scoped state persisted per `context_id`.
///
/// This state is shared across all tasks within the same session/conversation.
/// Use this for cross-skill data sharing (e.g., user data extracted by one skill
/// and used by another).
///
/// # Examples
///
/// ```ignore
/// // In skill handler
/// state.session().save("user_data", &user_data)?;
///
/// // Later, in another skill
/// let user_data: UserData = state.session().load("user_data")?.unwrap();
/// ```
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SessionState {
    #[serde(default)]
    data: HashMap<String, serde_json::Value>,
}

impl SessionState {
    /// Creates a new empty session state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Saves data to the session state under the given key.
    ///
    /// # Errors
    ///
    /// Returns an error if the value cannot be serialized to JSON.
    pub fn save<T>(&mut self, key: &str, value: &T) -> Result<(), AgentError>
    where
        T: Serialize,
    {
        let serialized =
            serde_json::to_value(value).map_err(|e| AgentError::ContextError(e.to_string()))?;
        self.data.insert(key.to_string(), serialized);
        Ok(())
    }

    /// Loads data from the session state for the given key.
    ///
    /// Returns `Ok(None)` if the key doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored value cannot be deserialized into type T.
    pub fn load<T>(&self, key: &str) -> Result<Option<T>, AgentError>
    where
        T: DeserializeOwned,
    {
        match self.data.get(key) {
            Some(value) => {
                let deserialized = serde_json::from_value(value.clone())
                    .map_err(|e| AgentError::ContextError(e.to_string()))?;
                Ok(Some(deserialized))
            }
            None => Ok(None),
        }
    }

    /// Removes data from the session state for the given key.
    ///
    /// Returns the previously stored value, if any.
    pub fn remove(&mut self, key: &str) -> Option<serde_json::Value> {
        self.data.remove(key)
    }

    /// Returns true if the session state contains the given key.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Returns the number of keys in the session state.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the session state is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Task-scoped state persisted per `task_id`.
///
/// This state is scoped to a single task within a skill's multi-turn conversation.
/// Use this for partial data during multi-turn flows (e.g., storing intermediate
/// results while waiting for user input).
///
/// # Examples
///
/// ```ignore
/// // In on_request, save partial data before asking for input
/// state.task().save("partial_user", &user_data)?;
/// state.set_slot(MySlot::NeedEmail)?;
/// return Ok(OnRequestResult::InputRequired { ... });
///
/// // In on_input_received, load it back
/// let partial_user: UserData = state.task().load("partial_user")?.unwrap();
/// ```
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TaskState {
    #[serde(default)]
    data: HashMap<String, serde_json::Value>,
    #[serde(default)]
    slot: Option<serde_json::Value>,
}

impl TaskState {
    /// Creates a new empty task state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Saves data to the task state under the given key.
    ///
    /// # Errors
    ///
    /// Returns an error if the value cannot be serialized to JSON.
    pub fn save<T>(&mut self, key: &str, value: &T) -> Result<(), AgentError>
    where
        T: Serialize,
    {
        let serialized =
            serde_json::to_value(value).map_err(|e| AgentError::ContextError(e.to_string()))?;
        self.data.insert(key.to_string(), serialized);
        Ok(())
    }

    /// Loads data from the task state for the given key.
    ///
    /// Returns `Ok(None)` if the key doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored value cannot be deserialized into type T.
    pub fn load<T>(&self, key: &str) -> Result<Option<T>, AgentError>
    where
        T: DeserializeOwned,
    {
        match self.data.get(key) {
            Some(value) => {
                let deserialized = serde_json::from_value(value.clone())
                    .map_err(|e| AgentError::ContextError(e.to_string()))?;
                Ok(Some(deserialized))
            }
            None => Ok(None),
        }
    }

    /// Removes data from the task state for the given key.
    pub fn remove(&mut self, key: &str) -> Option<serde_json::Value> {
        self.data.remove(key)
    }

    /// Returns the current pending slot as a `SkillSlot`, if set.
    ///
    /// This is used by the runtime to check for pending input slots.
    #[must_use]
    pub fn current_slot(&self) -> Option<crate::agent::SkillSlot> {
        self.slot
            .clone()
            .map(crate::agent::SkillSlot::from_value_unchecked)
    }

    /// Loads and deserializes the currently expected input slot.
    ///
    /// # Errors
    ///
    /// Returns an error if the slot value cannot be deserialized into type T.
    pub fn slot<T>(&self) -> Result<Option<T>, AgentError>
    where
        T: DeserializeOwned,
    {
        match &self.slot {
            Some(value) => {
                let slot: T = serde_json::from_value(value.clone())
                    .map_err(|e| AgentError::SkillSlot(e.to_string()))?;
                Ok(Some(slot))
            }
            None => Ok(None),
        }
    }

    /// Stores the pending input slot.
    ///
    /// # Errors
    ///
    /// Returns an error if the slot cannot be serialized.
    pub fn set_slot<T>(&mut self, slot: T) -> Result<(), AgentError>
    where
        T: Serialize,
    {
        let serialized =
            serde_json::to_value(slot).map_err(|e| AgentError::SkillSlot(e.to_string()))?;
        self.slot = Some(serialized);
        Ok(())
    }

    /// Clears the pending input slot.
    pub fn clear_slot(&mut self) {
        self.slot = None;
    }
}

/// Unified state container with scoped accessors.
///
/// Provides access to both task-scoped and session-scoped state through
/// explicit accessors, making the scope of data storage clear.
///
/// # Examples
///
/// ```ignore
/// // Task-scoped (multi-turn within one skill)
/// state.task().save("partial", &data)?;
/// let partial: Data = state.task().load("partial")?.unwrap();
///
/// // Session-scoped (shared across skills in conversation)
/// state.session().save("user_data", &user_data)?;
/// let user_data: UserData = state.session().load("user_data")?.unwrap();
///
/// // Slot management
/// state.set_slot(MySlot::NeedEmail)?;
/// let slot: MySlot = state.slot()?.unwrap();
/// ```
#[derive(Debug, Default)]
pub struct State {
    task: TaskState,
    session: SessionState,
}

impl State {
    /// Creates a new empty state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a state with existing task and session state.
    #[must_use]
    pub const fn with_states(task: TaskState, session: SessionState) -> Self {
        Self { task, session }
    }

    /// Returns a mutable reference to the task-scoped state.
    ///
    /// Task state is scoped to `task_id` and used for multi-turn within a skill.
    pub const fn task(&mut self) -> &mut TaskState {
        &mut self.task
    }

    /// Returns a mutable reference to the session-scoped state.
    ///
    /// Session state is scoped to `context_id` and shared across skills.
    pub const fn session(&mut self) -> &mut SessionState {
        &mut self.session
    }

    /// Returns an immutable reference to the task state.
    #[must_use]
    pub const fn task_ref(&self) -> &TaskState {
        &self.task
    }

    /// Returns an immutable reference to the session state.
    #[must_use]
    pub const fn session_ref(&self) -> &SessionState {
        &self.session
    }

    /// Consumes the state and returns the inner task and session states.
    #[must_use]
    pub fn into_parts(self) -> (TaskState, SessionState) {
        (self.task, self.session)
    }

    /// Loads and deserializes the currently expected input slot.
    ///
    /// # Errors
    ///
    /// Returns an error if the slot value cannot be deserialized into type T.
    pub fn slot<T>(&self) -> Result<Option<T>, AgentError>
    where
        T: DeserializeOwned,
    {
        match &self.task.slot {
            Some(value) => {
                let slot: T = serde_json::from_value(value.clone())
                    .map_err(|e| AgentError::SkillSlot(e.to_string()))?;
                Ok(Some(slot))
            }
            None => Ok(None),
        }
    }

    /// Stores the pending input slot.
    ///
    /// # Errors
    ///
    /// Returns an error if the slot cannot be serialized.
    pub fn set_slot<T>(&mut self, slot: T) -> Result<(), AgentError>
    where
        T: Serialize,
    {
        let serialized =
            serde_json::to_value(slot).map_err(|e| AgentError::SkillSlot(e.to_string()))?;
        self.task.slot = Some(serialized);
        Ok(())
    }

    /// Clears the pending input slot.
    pub fn clear_slot(&mut self) {
        self.task.slot = None;
    }

    /// Returns the raw slot value as a `SkillSlot`, if set.
    pub fn current_slot(&self) -> Option<crate::agent::SkillSlot> {
        self.task
            .slot
            .clone()
            .map(crate::agent::SkillSlot::from_value_unchecked)
    }

    /// Internal: Sets the slot from a `SkillSlot` (used by executor).
    #[cfg(feature = "runtime")]
    pub(crate) fn set_pending_slot(&mut self, slot: crate::agent::SkillSlot) {
        self.task.slot = Some(slot.into_value());
    }

    /// Internal: Clears the pending slot (used by executor).
    #[cfg(feature = "runtime")]
    pub(crate) fn clear_pending_slot(&mut self) {
        self.task.slot = None;
    }
}

/// Sender for streaming progress updates to the client.
///
/// This type is separate from `State` to provide clear separation between
/// data storage and communication concerns.
///
/// # Examples
///
/// ```ignore
/// // Send intermediate status update
/// progress.send_update("Processing step 1...").await?;
///
/// // Send partial artifact
/// let artifact = Artifact::from_json("partial.json", &data)?;
/// progress.send_partial_artifact(artifact).await?;
/// ```
// Runtime implementation with full functionality
#[cfg(feature = "runtime")]
pub struct ProgressSender {
    auth: Option<AuthContext>,
    task_manager: Option<Arc<dyn TaskManager>>,
    task_id: String,
    context_id: String,
    event_bus: Option<Arc<TaskEventBus>>,
}

#[cfg(feature = "runtime")]
impl ProgressSender {
    /// Creates a new progress sender with the necessary handles.
    pub(crate) fn new(
        auth: AuthContext,
        task_manager: Arc<dyn TaskManager>,
        event_bus: Arc<TaskEventBus>,
        context_id: impl Into<String>,
        task_id: impl Into<String>,
    ) -> Self {
        Self {
            auth: Some(auth),
            task_manager: Some(task_manager),
            context_id: context_id.into(),
            task_id: task_id.into(),
            event_bus: Some(event_bus),
        }
    }

    /// Creates a no-op progress sender for testing purposes.
    ///
    /// All send methods will succeed but do nothing.
    #[must_use]
    pub fn noop() -> Self {
        Self {
            auth: None,
            task_manager: None,
            context_id: String::new(),
            task_id: String::new(),
            event_bus: None,
        }
    }

    /// Sends an intermediate status update (`TaskState::Working`, final=false).
    ///
    /// # Errors
    ///
    /// Returns an error if the task event cannot be added.
    pub async fn send_update(&self, message: impl Into<Content>) -> AgentResult<()> {
        // If in noop mode, return early
        let (Some(auth), Some(task_manager), Some(event_bus)) =
            (&self.auth, &self.task_manager, &self.event_bus)
        else {
            return Ok(());
        };

        let now = Utc::now();
        let status = TaskStatus {
            state: A2ATaskState::Working as i32,
            timestamp: Some(::pbjson_types::Timestamp {
                seconds: now.timestamp(),
                nanos: now.timestamp_subsec_nanos().cast_signed(),
            }),
            message: Some(utils::create_a2a_message(
                Some(&self.context_id),
                Some(&self.task_id),
                Role::Assistant,
                message.into(),
            )),
        };

        let event = status_mapper::create_status_update_event(
            &self.task_id,
            &self.context_id,
            status,
            false,
        );
        let task_event = TaskEvent::StatusUpdate(event);

        task_manager.add_task_event(auth, &task_event).await?;
        event_bus.publish(&task_event);
        Ok(())
    }

    /// Sends a partial artifact update (`is_final=false`).
    ///
    /// # Errors
    ///
    /// Returns an error if the task event cannot be added.
    pub async fn send_partial_artifact(&self, artifact: Artifact) -> AgentResult<()> {
        // If in noop mode, return early
        let (Some(auth), Some(task_manager), Some(event_bus)) =
            (&self.auth, &self.task_manager, &self.event_bus)
        else {
            return Ok(());
        };

        let a2a_artifact = utils::artifact_to_a2a(&artifact);
        let event = TaskArtifactUpdateEvent {
            task_id: self.task_id.clone(),
            context_id: self.context_id.clone(),
            artifact: Some(a2a_artifact),
            append: false,
            last_chunk: false,
            metadata: None,
        };

        let task_event = TaskEvent::ArtifactUpdate(event);
        task_manager.add_task_event(auth, &task_event).await?;
        event_bus.publish(&task_event);
        Ok(())
    }
}

// Non-runtime stub implementation (no-op for testing and non-server use cases)
#[cfg(not(feature = "runtime"))]
pub struct ProgressSender {
    _private: (),
}

#[cfg(not(feature = "runtime"))]
impl ProgressSender {
    /// Creates a no-op progress sender for non-runtime builds.
    #[must_use]
    pub const fn noop() -> Self {
        Self { _private: () }
    }

    /// No-op: sends nothing in non-runtime builds.
    ///
    /// # Errors
    /// Never errors in this stub implementation.
    #[allow(clippy::unused_async)] // async required for API parity with the runtime impl
    pub async fn send_update(
        &self,
        _message: impl Into<crate::models::Content>,
    ) -> Result<(), AgentError> {
        Ok(())
    }

    /// No-op: sends nothing in non-runtime builds.
    ///
    /// # Errors
    /// Never errors in this stub implementation.
    #[allow(clippy::unused_async)] // async required for API parity with the runtime impl
    pub async fn send_partial_artifact(
        &self,
        _artifact: crate::agent::Artifact,
    ) -> Result<(), AgentError> {
        Ok(())
    }
}

// ============================================================================
// Authentication Context
// ============================================================================

/// Authentication and tenancy context for the current execution.
///
/// This struct holds authentication and tenancy information that is used
/// to namespace operations in services like [`MemoryService`](crate::runtime::memory::MemoryService)
/// and [`TaskManager`](crate::runtime::task_manager::TaskManager).
///
/// # Multi-tenancy
///
/// All runtime services use `AuthContext` to ensure data isolation between
/// different applications and users. This guarantees that one user or application
/// cannot access another's data.
///
/// # Examples
///
/// ```
/// use radkit::runtime::context::AuthContext;
///
/// let auth_ctx = AuthContext {
///     app_name: "my-app".to_string(),
///     user_name: "alice".to_string(),
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AuthContext {
    /// The name of the application or agent.
    pub app_name: String,
    /// The name of the current user.
    pub user_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_state_save_load_roundtrip() {
        let mut session = SessionState::new();
        session.save("key", &42u32).expect("save");

        let value: Option<u32> = session.load("key").expect("load");
        assert_eq!(value, Some(42));

        let missing: Option<u32> = session.load("missing").expect("load");
        assert!(missing.is_none());
    }

    #[test]
    fn session_state_remove_and_contains() {
        let mut session = SessionState::new();
        assert!(!session.contains("key"));
        assert!(session.is_empty());

        session.save("key", &"value").expect("save");
        assert!(session.contains("key"));
        assert_eq!(session.len(), 1);

        session.remove("key");
        assert!(!session.contains("key"));
        assert!(session.is_empty());
    }

    #[test]
    fn task_state_save_load_roundtrip() {
        let mut task = TaskState::new();
        task.save("partial", &vec![1, 2, 3]).expect("save");

        let value: Option<Vec<i32>> = task.load("partial").expect("load");
        assert_eq!(value, Some(vec![1, 2, 3]));
    }

    #[test]
    fn task_state_slot_roundtrip() {
        #[derive(Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
        enum MySlot {
            NeedEmail,
            NeedPhone { name: String },
        }

        let mut task = TaskState::new();

        // Set slot
        task.set_slot(MySlot::NeedEmail).expect("set slot");
        let slot: Option<MySlot> = task.slot().expect("get slot");
        assert_eq!(slot, Some(MySlot::NeedEmail));

        // Update slot
        task.set_slot(MySlot::NeedPhone {
            name: "Alice".into(),
        })
        .expect("set slot");
        let slot: Option<MySlot> = task.slot().expect("get slot");
        assert_eq!(
            slot,
            Some(MySlot::NeedPhone {
                name: "Alice".into()
            })
        );

        // Clear slot
        task.clear_slot();
        let slot: Option<MySlot> = task.slot().expect("get slot");
        assert!(slot.is_none());
    }

    #[test]
    fn state_provides_scoped_access() {
        let mut state = State::new();

        // Task scope
        state.task().save("task_key", &"task_value").expect("save");
        let task_val: Option<String> = state.task().load("task_key").expect("load");
        assert_eq!(task_val, Some("task_value".to_string()));

        // Session scope
        state
            .session()
            .save("session_key", &"session_value")
            .expect("save");
        let session_val: Option<String> = state.session().load("session_key").expect("load");
        assert_eq!(session_val, Some("session_value".to_string()));

        // Verify scopes are separate
        let task_missing: Option<String> = state.task().load("session_key").expect("load");
        assert!(task_missing.is_none());
    }

    #[test]
    fn state_with_existing_states() {
        let mut task = TaskState::new();
        task.save("key", &1).expect("save");

        let mut session = SessionState::new();
        session.save("key", &2).expect("save");

        let state = State::with_states(task, session);

        let task_val: Option<i32> = state.task_ref().load("key").expect("load");
        let session_val: Option<i32> = state.session_ref().load("key").expect("load");

        assert_eq!(task_val, Some(1));
        assert_eq!(session_val, Some(2));
    }

    #[test]
    fn state_into_parts() {
        let mut state = State::new();
        state.task().save("t", &1).expect("save");
        state.session().save("s", &2).expect("save");

        let (task, session) = state.into_parts();

        let t: Option<i32> = task.load("t").expect("load");
        let s: Option<i32> = session.load("s").expect("load");
        assert_eq!(t, Some(1));
        assert_eq!(s, Some(2));
    }
}
