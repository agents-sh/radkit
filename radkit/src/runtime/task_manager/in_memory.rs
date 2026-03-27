//! In-memory implementation of the `TaskStore` trait.
//!
//! This module provides both native (thread-safe) and WASM (single-threaded)
//! implementations of the task manager using in-memory storage.

use crate::errors::AgentResult;
use crate::runtime::context::AuthContext;
use crate::runtime::task_manager::{Task, TaskEvent, TaskStore};

// ============================================================================
// Native Implementation (Thread-Safe)
// ============================================================================

#[cfg(not(all(target_os = "wasi", target_env = "p1")))]
mod native {
    use super::{AgentResult, AuthContext, Task, TaskEvent, TaskStore};
    use dashmap::DashMap;
    use std::collections::BTreeSet;
    use std::sync::Arc;

    /// An in-memory, thread-safe implementation of the [`TaskStore`].
    ///
    /// This implementation uses `DashMap` for concurrent access, making it suitable
    /// for the multi-threaded `tokio` runtime on native targets.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use radkit::runtime::task_manager::{DefaultTaskManager, Task, TaskManager};
    /// use radkit::runtime::InMemoryTaskStore;
    ///
    /// let manager = DefaultTaskManager::new(InMemoryTaskStore::new());
    /// let auth_ctx = AuthContext {
    ///     app_name: "my-app".to_string(),
    ///     user_name: "user1".to_string(),
    /// };
    ///
    /// let task = Task {
    ///     id: "task-1".to_string(),
    ///     context_id: "ctx-1".to_string(),
    ///     status: TaskStatus::default(),
    ///     artifacts: vec![],
    /// };
    ///
    /// manager.save_task(&auth_ctx, &task).await?;
    /// ```
    #[derive(Debug, Default)]
    pub struct InMemoryTaskStore {
        tasks: Arc<DashMap<String, Task>>,
        events: Arc<DashMap<String, Vec<TaskEvent>>>,
        /// Store `TaskState` for multi-turn conversations (task-scoped)
        task_states: Arc<DashMap<String, String>>, // Store as JSON string
        /// Store skill ID associations for task continuation
        task_skills: Arc<DashMap<String, String>>,
        /// Store `SessionState` for cross-skill data sharing (context-scoped)
        session_states: Arc<DashMap<String, String>>, // Store as JSON string
    }

    impl InMemoryTaskStore {
        /// Creates a new `InMemoryTaskStore`.
        #[must_use]
        pub fn new() -> Self {
            Self::default()
        }

        /// Creates a namespaced key from the auth context and the original key.
        fn get_namespaced_key(auth_ctx: &AuthContext, key: &str) -> String {
            format!("{}:{}:{}", auth_ctx.app_name, auth_ctx.user_name, key)
        }
    }

    #[async_trait::async_trait]
    impl TaskStore for InMemoryTaskStore {
        async fn get_task(
            &self,
            auth_ctx: &AuthContext,
            task_id: &str,
        ) -> AgentResult<Option<Task>> {
            let key = Self::get_namespaced_key(auth_ctx, task_id);
            Ok(self.tasks.get(&key).map(|t| t.value().clone()))
        }

        async fn list_tasks(&self, auth_ctx: &AuthContext) -> AgentResult<Vec<Task>> {
            let prefix = format!("{}:{}:", auth_ctx.app_name, auth_ctx.user_name);
            Ok(self
                .tasks
                .iter()
                .filter(|item| item.key().starts_with(&prefix))
                .map(|item| item.value().clone())
                .collect())
        }

        async fn save_task(&self, auth_ctx: &AuthContext, task: &Task) -> AgentResult<()> {
            let key = Self::get_namespaced_key(auth_ctx, &task.id);
            self.tasks.insert(key, task.clone());
            Ok(())
        }

        async fn append_event(
            &self,
            auth_ctx: &AuthContext,
            task_key: &str,
            event: &TaskEvent,
        ) -> AgentResult<()> {
            let key = Self::get_namespaced_key(auth_ctx, task_key);
            self.events.entry(key).or_default().push(event.clone());
            Ok(())
        }

        async fn get_events(
            &self,
            auth_ctx: &AuthContext,
            task_key: &str,
        ) -> AgentResult<Vec<TaskEvent>> {
            let key = Self::get_namespaced_key(auth_ctx, task_key);
            Ok(self
                .events
                .get(&key)
                .map_or_else(Vec::new, |v| v.value().clone()))
        }

        async fn list_event_task_keys(&self, auth_ctx: &AuthContext) -> AgentResult<Vec<String>> {
            let prefix = format!("{}:{}:", auth_ctx.app_name, auth_ctx.user_name);
            let keys = self
                .events
                .iter()
                .filter_map(|entry| {
                    entry
                        .key()
                        .strip_prefix(&prefix)
                        .map(std::string::ToString::to_string)
                })
                .collect();
            Ok(keys)
        }

        async fn list_task_ids(&self, auth_ctx: &AuthContext) -> AgentResult<Vec<String>> {
            let prefix = format!("{}:{}:", auth_ctx.app_name, auth_ctx.user_name);
            let ids = self
                .tasks
                .iter()
                .filter_map(|item| {
                    item.key()
                        .strip_prefix(&prefix)
                        .map(std::string::ToString::to_string)
                })
                .collect();
            Ok(ids)
        }

        async fn list_context_ids(&self, auth_ctx: &AuthContext) -> AgentResult<Vec<String>> {
            let prefix = format!("{}:{}:", auth_ctx.app_name, auth_ctx.user_name);
            let contexts: BTreeSet<String> = self
                .tasks
                .iter()
                .filter(|item| item.key().starts_with(&prefix))
                .map(|item| item.value().context_id.clone())
                .collect();

            Ok(contexts.into_iter().collect())
        }

        async fn save_task_state(
            &self,
            auth_ctx: &AuthContext,
            task_id: &str,
            state: &crate::runtime::context::TaskState,
        ) -> AgentResult<()> {
            let key = Self::get_namespaced_key(auth_ctx, task_id);
            let json_str = serde_json::to_string(state).map_err(|e| {
                crate::errors::AgentError::Serialization {
                    format: "json".to_string(),
                    reason: format!("Failed to serialize TaskState: {e}"),
                }
            })?;
            self.task_states.insert(key, json_str);
            Ok(())
        }

        async fn load_task_state(
            &self,
            auth_ctx: &AuthContext,
            task_id: &str,
        ) -> AgentResult<Option<crate::runtime::context::TaskState>> {
            let key = Self::get_namespaced_key(auth_ctx, task_id);
            match self.task_states.get(&key) {
                Some(json_ref) => {
                    let state = serde_json::from_str(json_ref.value()).map_err(|e| {
                        crate::errors::AgentError::Serialization {
                            format: "json".to_string(),
                            reason: format!("Failed to deserialize TaskState: {e}"),
                        }
                    })?;
                    Ok(Some(state))
                }
                None => Ok(None),
            }
        }

        async fn set_task_skill(
            &self,
            auth_ctx: &AuthContext,
            task_id: &str,
            skill_id: &str,
        ) -> AgentResult<()> {
            let key = Self::get_namespaced_key(auth_ctx, task_id);
            self.task_skills.insert(key, skill_id.to_string());
            Ok(())
        }

        async fn get_task_skill(
            &self,
            auth_ctx: &AuthContext,
            task_id: &str,
        ) -> AgentResult<Option<String>> {
            let key = Self::get_namespaced_key(auth_ctx, task_id);
            Ok(self.task_skills.get(&key).map(|s| s.value().clone()))
        }

        async fn save_session_state(
            &self,
            auth_ctx: &AuthContext,
            context_id: &str,
            state: &crate::runtime::context::SessionState,
        ) -> AgentResult<()> {
            let key = Self::get_namespaced_key(auth_ctx, context_id);
            let json_str = serde_json::to_string(state).map_err(|e| {
                crate::errors::AgentError::Serialization {
                    format: "json".to_string(),
                    reason: format!("Failed to serialize SessionState: {e}"),
                }
            })?;
            self.session_states.insert(key, json_str);
            Ok(())
        }

        async fn load_session_state(
            &self,
            auth_ctx: &AuthContext,
            context_id: &str,
        ) -> AgentResult<Option<crate::runtime::context::SessionState>> {
            let key = Self::get_namespaced_key(auth_ctx, context_id);
            match self.session_states.get(&key) {
                Some(json_ref) => {
                    let state = serde_json::from_str(json_ref.value()).map_err(|e| {
                        crate::errors::AgentError::Serialization {
                            format: "json".to_string(),
                            reason: format!("Failed to deserialize SessionState: {e}"),
                        }
                    })?;
                    Ok(Some(state))
                }
                None => Ok(None),
            }
        }
    }
}

#[cfg(not(all(target_os = "wasi", target_env = "p1")))]
pub use native::InMemoryTaskStore;

// ============================================================================
// WASM Implementation (Single-Threaded)
// ============================================================================

#[cfg(all(target_os = "wasi", target_env = "p1"))]
mod wasm {
    use super::{AgentResult, AuthContext, Task, TaskEvent, TaskStore};
    use std::cell::RefCell;
    use std::collections::{BTreeSet, HashMap};

    /// An in-memory, single-threaded implementation of the [`TaskStore`] for WASM.
    ///
    /// This implementation uses `RefCell<HashMap>` for interior mutability, suitable
    /// for the single-threaded WASM environment.
    #[derive(Debug, Default)]
    pub struct InMemoryTaskStore {
        tasks: RefCell<HashMap<String, Task>>,
        events: RefCell<HashMap<String, Vec<TaskEvent>>>,
        /// Store `TaskState` for multi-turn conversations (task-scoped)
        task_states: RefCell<HashMap<String, String>>, // Store as JSON string
        /// Store skill ID associations for task continuation
        task_skills: RefCell<HashMap<String, String>>,
        /// Store `SessionState` for cross-skill data sharing (context-scoped)
        session_states: RefCell<HashMap<String, String>>, // Store as JSON string
    }

    impl InMemoryTaskStore {
        /// Creates a new `InMemoryTaskStore`.
        #[must_use]
        pub fn new() -> Self {
            Self::default()
        }

        /// Creates a namespaced key from the auth context and the original key.
        fn get_namespaced_key(auth_ctx: &AuthContext, key: &str) -> String {
            format!("{}:{}:{}", auth_ctx.app_name, auth_ctx.user_name, key)
        }
    }

    #[async_trait::async_trait(?Send)]
    impl TaskStore for InMemoryTaskStore {
        async fn get_task(
            &self,
            auth_ctx: &AuthContext,
            task_id: &str,
        ) -> AgentResult<Option<Task>> {
            let key = Self::get_namespaced_key(auth_ctx, task_id);
            Ok(self.tasks.borrow().get(&key).cloned())
        }

        async fn list_tasks(&self, auth_ctx: &AuthContext) -> AgentResult<Vec<Task>> {
            let prefix = format!("{}:{}:", auth_ctx.app_name, auth_ctx.user_name);
            Ok(self
                .tasks
                .borrow()
                .iter()
                .filter(|(key, _)| key.starts_with(&prefix))
                .map(|(_, task)| task.clone())
                .collect())
        }

        async fn save_task(&self, auth_ctx: &AuthContext, task: &Task) -> AgentResult<()> {
            let key = Self::get_namespaced_key(auth_ctx, &task.id);
            self.tasks.borrow_mut().insert(key, task.clone());
            Ok(())
        }

        async fn append_event(
            &self,
            auth_ctx: &AuthContext,
            task_key: &str,
            event: &TaskEvent,
        ) -> AgentResult<()> {
            let key = Self::get_namespaced_key(auth_ctx, task_key);
            self.events
                .borrow_mut()
                .entry(key)
                .or_default()
                .push(event.clone());
            Ok(())
        }

        async fn get_events(
            &self,
            auth_ctx: &AuthContext,
            task_key: &str,
        ) -> AgentResult<Vec<TaskEvent>> {
            let key = Self::get_namespaced_key(auth_ctx, task_key);
            Ok(self.events.borrow().get(&key).cloned().unwrap_or_default())
        }

        async fn list_event_task_keys(&self, auth_ctx: &AuthContext) -> AgentResult<Vec<String>> {
            let prefix = format!("{}:{}:", auth_ctx.app_name, auth_ctx.user_name);
            let keys = self
                .events
                .borrow()
                .keys()
                .filter_map(|key| {
                    key.strip_prefix(&prefix)
                        .map(std::string::ToString::to_string)
                })
                .collect();
            Ok(keys)
        }

        async fn list_task_ids(&self, auth_ctx: &AuthContext) -> AgentResult<Vec<String>> {
            let prefix = format!("{}:{}:", auth_ctx.app_name, auth_ctx.user_name);
            let ids = self
                .tasks
                .borrow()
                .keys()
                .filter_map(|key| {
                    key.strip_prefix(&prefix)
                        .map(std::string::ToString::to_string)
                })
                .collect();
            Ok(ids)
        }

        async fn list_context_ids(&self, auth_ctx: &AuthContext) -> AgentResult<Vec<String>> {
            let prefix = format!("{}:{}:", auth_ctx.app_name, auth_ctx.user_name);
            let contexts: BTreeSet<String> = self
                .tasks
                .borrow()
                .iter()
                .filter(|(key, _)| key.starts_with(&prefix))
                .map(|(_, task)| task.context_id.clone())
                .collect();

            Ok(contexts.into_iter().collect())
        }

        async fn save_task_state(
            &self,
            auth_ctx: &AuthContext,
            task_id: &str,
            state: &crate::runtime::context::TaskState,
        ) -> AgentResult<()> {
            let key = Self::get_namespaced_key(auth_ctx, task_id);
            let json_str = serde_json::to_string(state).map_err(|e| {
                crate::errors::AgentError::Serialization {
                    format: "json".to_string(),
                    reason: format!("Failed to serialize TaskState: {e}"),
                }
            })?;
            self.task_states.borrow_mut().insert(key, json_str);
            Ok(())
        }

        async fn load_task_state(
            &self,
            auth_ctx: &AuthContext,
            task_id: &str,
        ) -> AgentResult<Option<crate::runtime::context::TaskState>> {
            let key = Self::get_namespaced_key(auth_ctx, task_id);
            match self.task_states.borrow().get(&key) {
                Some(json_str) => {
                    let state = serde_json::from_str(json_str).map_err(|e| {
                        crate::errors::AgentError::Serialization {
                            format: "json".to_string(),
                            reason: format!("Failed to deserialize TaskState: {e}"),
                        }
                    })?;
                    Ok(Some(state))
                }
                None => Ok(None),
            }
        }

        async fn set_task_skill(
            &self,
            auth_ctx: &AuthContext,
            task_id: &str,
            skill_id: &str,
        ) -> AgentResult<()> {
            let key = Self::get_namespaced_key(auth_ctx, task_id);
            self.task_skills
                .borrow_mut()
                .insert(key, skill_id.to_string());
            Ok(())
        }

        async fn get_task_skill(
            &self,
            auth_ctx: &AuthContext,
            task_id: &str,
        ) -> AgentResult<Option<String>> {
            let key = Self::get_namespaced_key(auth_ctx, task_id);
            Ok(self.task_skills.borrow().get(&key).cloned())
        }

        async fn save_session_state(
            &self,
            auth_ctx: &AuthContext,
            context_id: &str,
            state: &crate::runtime::context::SessionState,
        ) -> AgentResult<()> {
            let key = Self::get_namespaced_key(auth_ctx, context_id);
            let json_str = serde_json::to_string(state).map_err(|e| {
                crate::errors::AgentError::Serialization {
                    format: "json".to_string(),
                    reason: format!("Failed to serialize SessionState: {e}"),
                }
            })?;
            self.session_states.borrow_mut().insert(key, json_str);
            Ok(())
        }

        async fn load_session_state(
            &self,
            auth_ctx: &AuthContext,
            context_id: &str,
        ) -> AgentResult<Option<crate::runtime::context::SessionState>> {
            let key = Self::get_namespaced_key(auth_ctx, context_id);
            match self.session_states.borrow().get(&key) {
                Some(json_str) => {
                    let state = serde_json::from_str(json_str).map_err(|e| {
                        crate::errors::AgentError::Serialization {
                            format: "json".to_string(),
                            reason: format!("Failed to deserialize SessionState: {e}"),
                        }
                    })?;
                    Ok(Some(state))
                }
                None => Ok(None),
            }
        }
    }
}

#[cfg(all(target_os = "wasi", target_env = "p1"))]
pub use wasm::InMemoryTaskStore;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::context::{AuthContext, TaskState as RadkitTaskState};
    use crate::runtime::task_manager::{
        DefaultTaskManager, ListTasksFilter, TaskEvent, TaskManager,
    };
    use a2a_types::{
        Artifact, Message, TaskArtifactUpdateEvent, TaskState, TaskStatus, TaskStatusUpdateEvent,
    };

    fn auth() -> AuthContext {
        AuthContext {
            app_name: "app".into(),
            user_name: "user".into(),
        }
    }

    fn make_message(id: &str, context: &str) -> Message {
        Message {
            message_id: id.into(),
            role: a2a_types::Role::Agent as i32,
            parts: Vec::new(),
            context_id: context.into(),
            task_id: String::new(),
            reference_task_ids: Vec::new(),
            extensions: Vec::new(),
            metadata: None,
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn stores_tasks_events_and_context() {
        let manager = DefaultTaskManager::new(InMemoryTaskStore::new());
        let auth_ctx = auth();
        let task = Task {
            id: "task-1".into(),
            context_id: "ctx-1".into(),
            status: TaskStatus {
                state: TaskState::Submitted as i32,
                timestamp: None,
                message: None,
            },
            artifacts: Vec::new(),
        };

        manager
            .save_task(&auth_ctx, &task)
            .await
            .expect("save task");

        let retrieved = manager
            .get_task(&auth_ctx, "task-1")
            .await
            .expect("get task")
            .expect("task exists");
        assert_eq!(retrieved.id, task.id);

        // negotiation messages keep persisted append order
        let msg_a = make_message("b", "ctx-1");
        let msg_b = make_message("a", "ctx-1");
        manager
            .add_task_event(&auth_ctx, &TaskEvent::Message(msg_a.clone()))
            .await
            .expect("add message");
        manager
            .add_task_event(&auth_ctx, &TaskEvent::Message(msg_b.clone()))
            .await
            .expect("add message");

        let status_event = TaskStatusUpdateEvent {
            task_id: "task-1".into(),
            context_id: "ctx-1".into(),
            status: Some(TaskStatus {
                state: TaskState::Working as i32,
                timestamp: None,
                message: None,
            }),
            metadata: None,
        };
        manager
            .add_task_event(&auth_ctx, &TaskEvent::StatusUpdate(status_event))
            .await
            .expect("status");

        let artifact_event = TaskArtifactUpdateEvent {
            task_id: "task-1".into(),
            context_id: "ctx-1".into(),
            artifact: Some(Artifact {
                artifact_id: "artifact".into(),
                parts: Vec::new(),
                name: String::new(),
                description: String::new(),
                extensions: Vec::new(),
                metadata: None,
            }),
            append: false,
            last_chunk: false,
            metadata: None,
        };
        manager
            .add_task_event(&auth_ctx, &TaskEvent::ArtifactUpdate(artifact_event))
            .await
            .expect("artifact");

        let events = manager
            .get_task_events(&auth_ctx, "task-1")
            .await
            .expect("events");
        assert_eq!(events.len(), 2);

        let negotiation = manager
            .get_negotiating_messages(&auth_ctx, "ctx-1")
            .await
            .expect("negotiation");
        assert_eq!(negotiation.len(), 2);
        assert_eq!(negotiation[0].message_id, "b");
        assert_eq!(negotiation[1].message_id, "a");

        let ids = manager
            .list_task_ids(&auth_ctx, Some("ctx-1"))
            .await
            .expect("ids");
        assert_eq!(ids, vec!["task-1".to_string()]);

        let mut task_state = RadkitTaskState::new();
        task_state.save("flag", &true).expect("save flag");
        manager
            .save_task_state(&auth_ctx, "task-1", &task_state)
            .await
            .expect("save state");
        let restored = manager
            .load_task_state(&auth_ctx, "task-1")
            .await
            .expect("load state")
            .expect("state present");
        let flag: Option<bool> = restored.load("flag").expect("flag");
        assert_eq!(flag, Some(true));

        manager
            .set_task_skill(&auth_ctx, "task-1", "skill")
            .await
            .expect("set skill");
        let skill = manager
            .get_task_skill(&auth_ctx, "task-1")
            .await
            .expect("get skill");
        assert_eq!(skill.as_deref(), Some("skill"));

        let page = manager
            .list_tasks(
                &auth_ctx,
                &ListTasksFilter {
                    context_id: Some("ctx-1"),
                    page_size: Some(10),
                    page_token: None,
                },
            )
            .await
            .expect("list tasks");
        assert_eq!(page.items.len(), 1);
    }
}
