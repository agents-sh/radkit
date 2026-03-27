//! Task management service for agent execution.
//!
//! This module provides task persistence and event tracking functionality.
//! The [`TaskManager`] trait defines the interface for storing and retrieving
//! task state and events throughout the agent execution lifecycle.
//!
//! # Event Sourcing
//!
//! The task manager uses an event sourcing pattern where task state is stored
//! separately from its event history. This allows for efficient queries and
//! reconstruction of full task history when needed.
//!
//! # Multi-tenancy
//!
//! All operations are namespaced by [`AuthContext`](crate::runtime::context::AuthContext),
//! ensuring data isolation between different users and applications.

mod default;
pub mod in_memory;
#[cfg(all(
    feature = "task-store-sqlite",
    not(all(target_os = "wasi", target_env = "p1"))
))]
pub mod sqlite;

pub use default::DefaultTaskManager;
pub use in_memory::InMemoryTaskStore;
#[cfg(all(
    feature = "task-store-sqlite",
    not(all(target_os = "wasi", target_env = "p1"))
))]
pub use sqlite::SqliteTaskStore;
pub type InMemoryTaskManager = DefaultTaskManager;

use crate::compat::{MaybeSend, MaybeSync};
use crate::errors::AgentResult;
use crate::runtime::context::AuthContext;
use a2a_types::{Artifact, Message, TaskArtifactUpdateEvent, TaskStatus, TaskStatusUpdateEvent};
use serde::{Deserialize, Serialize};

// ============================================================================
// Data Structures
// ============================================================================

/// Represents the state of a single task, mirroring the A2A Task object, but
/// stored without the conversational history for efficiency.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Task {
    pub id: String,
    /// Corresponds to A2A `contextId`.
    pub context_id: String,
    pub status: TaskStatus,
    pub artifacts: Vec<Artifact>,
}

/// Represents a significant event that occurred during a task's lifecycle.
/// This enum can be converted from and to the various A2A event types.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum TaskEvent {
    StatusUpdate(TaskStatusUpdateEvent),
    ArtifactUpdate(TaskArtifactUpdateEvent),
    Message(Message),
}

/// Filter for listing tasks, enabling pagination.
#[derive(Debug, Default)]
pub struct ListTasksFilter<'a> {
    pub context_id: Option<&'a str>,
    pub page_size: Option<u32>,
    pub page_token: Option<&'a str>,
}

/// Represents a paginated result set.
#[derive(Debug)]
pub struct PaginatedResult<T> {
    pub items: Vec<T>,
    pub next_page_token: Option<String>,
}

pub(crate) const NEGOTIATION_PREFIX: &str = "_negotiation:";

/// Persistence interface for task data.
///
/// Higher-level task orchestration code delegates all storage operations through
/// this trait so callers can swap in database-backed stores without re-implementing
/// business logic.
#[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
pub trait TaskStore: MaybeSend + MaybeSync {
    /// Retrieves a single task by its ID, scoped to the `AuthContext`.
    async fn get_task(&self, auth_ctx: &AuthContext, task_id: &str) -> AgentResult<Option<Task>>;

    /// Lists tasks for the given user/app.
    async fn list_tasks(&self, auth_ctx: &AuthContext) -> AgentResult<Vec<Task>>;

    /// Stores or updates a task's state. This is an upsert operation.
    async fn save_task(&self, auth_ctx: &AuthContext, task: &Task) -> AgentResult<()>;

    /// Appends an event to a task's history keyed by `task_key`.
    async fn append_event(
        &self,
        auth_ctx: &AuthContext,
        task_key: &str,
        event: &TaskEvent,
    ) -> AgentResult<()>;

    /// Retrieves all events for a specific task key, ordered chronologically.
    async fn get_events(
        &self,
        auth_ctx: &AuthContext,
        task_key: &str,
    ) -> AgentResult<Vec<TaskEvent>>;

    /// Lists all task/event keys that contain events for the auth context.
    async fn list_event_task_keys(&self, auth_ctx: &AuthContext) -> AgentResult<Vec<String>>;

    /// Lists all task IDs for the given user/app.
    async fn list_task_ids(&self, auth_ctx: &AuthContext) -> AgentResult<Vec<String>>;

    /// Lists all known context IDs for the current auth context.
    async fn list_context_ids(&self, auth_ctx: &AuthContext) -> AgentResult<Vec<String>>;

    /// Saves the task state for multi-turn conversations.
    async fn save_task_state(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
        state: &crate::runtime::context::TaskState,
    ) -> AgentResult<()>;

    /// Loads the task state for a given task.
    ///
    /// Returns `None` if no state has been saved yet.
    async fn load_task_state(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
    ) -> AgentResult<Option<crate::runtime::context::TaskState>>;

    /// Associates a skill ID with a task for continuation purposes.
    async fn set_task_skill(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
        skill_id: &str,
    ) -> AgentResult<()>;

    /// Retrieves the skill ID associated with a task.
    ///
    /// Returns `None` if no skill has been associated with this task.
    async fn get_task_skill(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
    ) -> AgentResult<Option<String>>;

    /// Saves session state for cross-skill data sharing.
    ///
    /// Session state is scoped to `context_id` and shared across all tasks
    /// within the same conversation/session.
    async fn save_session_state(
        &self,
        auth_ctx: &AuthContext,
        context_id: &str,
        state: &crate::runtime::context::SessionState,
    ) -> AgentResult<()>;

    /// Loads session state for a given context.
    ///
    /// Returns `None` if no session state has been saved yet.
    async fn load_session_state(
        &self,
        auth_ctx: &AuthContext,
        context_id: &str,
    ) -> AgentResult<Option<crate::runtime::context::SessionState>>;
}

// ============================================================================
// TaskManager Trait
// ============================================================================

/// A stateful Data Access Object (DAO) for persisting and retrieving all data
/// related to agent interactions.
#[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
pub trait TaskManager: MaybeSend + MaybeSync {
    /// Retrieves a single task by its ID, scoped to the `AuthContext`.
    async fn get_task(&self, auth_ctx: &AuthContext, task_id: &str) -> AgentResult<Option<Task>>;

    /// Lists tasks for the given user/app, with optional filtering and pagination.
    /// This is the primary method for retrieving tasks associated with a "session" (contextId).
    async fn list_tasks(
        &self,
        auth_ctx: &AuthContext,
        filter: &ListTasksFilter<'_>,
    ) -> AgentResult<PaginatedResult<Task>>;

    /// Stores or updates a task's state. This is an upsert operation.
    async fn save_task(&self, auth_ctx: &AuthContext, task: &Task) -> AgentResult<()>;

    /// Appends an event to a task's history.
    ///
    /// The `task_id` and `context_id` are extracted from the event itself (Message,
    /// `StatusUpdate`, or `ArtifactUpdate` all contain these fields). For negotiation
    /// messages where `task_id` is None, events are stored under a synthetic key
    /// `_negotiation:{context_id}`.
    async fn add_task_event(&self, auth_ctx: &AuthContext, event: &TaskEvent) -> AgentResult<()>;

    /// Retrieves all events for a specific task, ordered chronologically.
    async fn get_task_events(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
    ) -> AgentResult<Vec<TaskEvent>>;

    /// Retrieves all Message events across all tasks within a context.
    /// This is useful for building the full conversation history for LLM context.
    /// Messages are returned in persisted append order.
    async fn get_negotiating_messages(
        &self,
        auth_ctx: &AuthContext,
        context_id: &str,
    ) -> AgentResult<Vec<Message>>;

    /// Lists all task IDs for the given user/app, optionally filtered by `context_id`.
    /// Returns all task IDs without pagination.
    async fn list_task_ids(
        &self,
        auth_ctx: &AuthContext,
        context_id: Option<&str>,
    ) -> AgentResult<Vec<String>>;

    /// Lists all known context IDs for the current auth context.
    ///
    /// This aggregates both contexts with persisted tasks and contexts that
    /// only contain negotiation messages (no tasks yet).
    async fn list_context_ids(&self, auth_ctx: &AuthContext) -> AgentResult<Vec<String>>;

    /// Saves the task state for multi-turn conversations.
    ///
    /// This allows skills to persist data between `on_request` and `on_input_received` calls.
    /// The state is namespaced by `AuthContext` to ensure tenant isolation.
    async fn save_task_state(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
        state: &crate::runtime::context::TaskState,
    ) -> AgentResult<()>;

    /// Loads the task state for a given task.
    ///
    /// Returns `None` if no state has been saved yet.
    async fn load_task_state(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
    ) -> AgentResult<Option<crate::runtime::context::TaskState>>;

    /// Associates a skill ID with a task for continuation purposes.
    ///
    /// This is stored when a task is created so we know which skill to call on continuation.
    async fn set_task_skill(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
        skill_id: &str,
    ) -> AgentResult<()>;

    /// Retrieves the skill ID associated with a task.
    ///
    /// Returns `None` if no skill has been associated with this task.
    async fn get_task_skill(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
    ) -> AgentResult<Option<String>>;

    /// Saves session state for cross-skill data sharing.
    ///
    /// Session state is scoped to `context_id` and shared across all tasks
    /// within the same conversation/session. This enables skills to pass data
    /// to each other (e.g., user data extracted by one skill used by another).
    async fn save_session_state(
        &self,
        auth_ctx: &AuthContext,
        context_id: &str,
        state: &crate::runtime::context::SessionState,
    ) -> AgentResult<()>;

    /// Loads session state for a given context.
    ///
    /// Returns `None` if no session state has been saved yet for this context.
    async fn load_session_state(
        &self,
        auth_ctx: &AuthContext,
        context_id: &str,
    ) -> AgentResult<Option<crate::runtime::context::SessionState>>;
}
