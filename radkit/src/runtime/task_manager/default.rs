use super::{
    ListTasksFilter, PaginatedResult, Task, TaskEvent, TaskManager, TaskStore, NEGOTIATION_PREFIX,
};
use crate::errors::AgentResult;
use crate::runtime::context::AuthContext;
use a2a_types::Message;
use std::collections::BTreeSet;
use std::sync::Arc;

/// Default task manager implementation that separates orchestration logic from storage.
#[derive(Clone)]
pub struct DefaultTaskManager {
    store: Arc<dyn TaskStore>,
}

impl DefaultTaskManager {
    /// Creates a new manager backed by the provided store.
    #[must_use]
    pub fn new(store: impl TaskStore + 'static) -> Self {
        Self {
            store: Arc::new(store),
        }
    }

    /// Creates a manager from a shared store handle.
    #[must_use]
    pub fn with_store(store: Arc<dyn TaskStore>) -> Self {
        Self { store }
    }

    /// Convenience constructor for the default in-memory store.
    #[must_use]
    pub fn in_memory() -> Self {
        Self::new(super::InMemoryTaskStore::new())
    }

    fn event_key(event: &TaskEvent) -> String {
        match event {
            TaskEvent::Message(msg) => msg.task_id.as_ref().map_or_else(
                || {
                    format!(
                        "{NEGOTIATION_PREFIX}{}",
                        msg.context_id.as_deref().unwrap_or("default")
                    )
                },
                std::clone::Clone::clone,
            ),
            TaskEvent::StatusUpdate(update) => update.task_id.clone(),
            TaskEvent::ArtifactUpdate(update) => update.task_id.clone(),
        }
    }

    fn paginate<T: Clone>(items: &[T], filter: &ListTasksFilter<'_>) -> PaginatedResult<T> {
        let page_size = filter.page_size.unwrap_or(100) as usize;
        let start_offset = filter
            .page_token
            .and_then(|token| token.parse::<usize>().ok())
            .unwrap_or(0);

        let total = items.len();
        let end_offset = (start_offset + page_size).min(total);
        let page_items = items[start_offset..end_offset].to_vec();

        let next_page_token = if end_offset < total {
            Some(end_offset.to_string())
        } else {
            None
        };

        PaginatedResult {
            items: page_items,
            next_page_token,
        }
    }
}

#[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
impl TaskManager for DefaultTaskManager {
    async fn get_task(&self, auth_ctx: &AuthContext, task_id: &str) -> AgentResult<Option<Task>> {
        self.store.get_task(auth_ctx, task_id).await
    }

    async fn list_tasks(
        &self,
        auth_ctx: &AuthContext,
        filter: &ListTasksFilter<'_>,
    ) -> AgentResult<PaginatedResult<Task>> {
        let mut tasks = self.store.list_tasks(auth_ctx).await?;

        if let Some(context_id) = filter.context_id {
            tasks.retain(|task| task.context_id == context_id);
        }

        tasks.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(Self::paginate(&tasks, filter))
    }

    async fn save_task(&self, auth_ctx: &AuthContext, task: &Task) -> AgentResult<()> {
        self.store.save_task(auth_ctx, task).await
    }

    async fn add_task_event(&self, auth_ctx: &AuthContext, event: &TaskEvent) -> AgentResult<()> {
        let task_key = Self::event_key(event);
        self.store.append_event(auth_ctx, &task_key, event).await
    }

    async fn get_task_events(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
    ) -> AgentResult<Vec<TaskEvent>> {
        self.store.get_events(auth_ctx, task_id).await
    }

    async fn get_negotiating_messages(
        &self,
        auth_ctx: &AuthContext,
        context_id: &str,
    ) -> AgentResult<Vec<Message>> {
        let negotiation_key = format!("{NEGOTIATION_PREFIX}{context_id}");
        let mut messages = Vec::new();
        for event in self.store.get_events(auth_ctx, &negotiation_key).await? {
            if let TaskEvent::Message(msg) = event {
                if msg.context_id.as_deref() == Some(context_id) && msg.task_id.is_none() {
                    messages.push(msg);
                }
            }
        }
        Ok(messages)
    }

    async fn list_task_ids(
        &self,
        auth_ctx: &AuthContext,
        context_id: Option<&str>,
    ) -> AgentResult<Vec<String>> {
        if let Some(target_context) = context_id {
            let mut tasks = self.store.list_tasks(auth_ctx).await?;
            tasks.retain(|task| task.context_id == target_context);
            let mut ids: Vec<String> = tasks.into_iter().map(|task| task.id).collect();
            ids.sort();
            Ok(ids)
        } else {
            let mut ids = self.store.list_task_ids(auth_ctx).await?;
            ids.sort();
            Ok(ids)
        }
    }

    async fn list_context_ids(&self, auth_ctx: &AuthContext) -> AgentResult<Vec<String>> {
        let mut contexts: BTreeSet<String> = self
            .store
            .list_context_ids(auth_ctx)
            .await?
            .into_iter()
            .collect();

        for key in self.store.list_event_task_keys(auth_ctx).await? {
            if let Some(context_id) = key.strip_prefix(NEGOTIATION_PREFIX) {
                contexts.insert(context_id.to_string());
            }
        }

        Ok(contexts.into_iter().collect())
    }

    async fn save_task_state(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
        state: &crate::runtime::context::TaskState,
    ) -> AgentResult<()> {
        self.store.save_task_state(auth_ctx, task_id, state).await
    }

    async fn load_task_state(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
    ) -> AgentResult<Option<crate::runtime::context::TaskState>> {
        self.store.load_task_state(auth_ctx, task_id).await
    }

    async fn set_task_skill(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
        skill_id: &str,
    ) -> AgentResult<()> {
        self.store.set_task_skill(auth_ctx, task_id, skill_id).await
    }

    async fn get_task_skill(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
    ) -> AgentResult<Option<String>> {
        self.store.get_task_skill(auth_ctx, task_id).await
    }

    async fn save_session_state(
        &self,
        auth_ctx: &AuthContext,
        context_id: &str,
        state: &crate::runtime::context::SessionState,
    ) -> AgentResult<()> {
        self.store
            .save_session_state(auth_ctx, context_id, state)
            .await
    }

    async fn load_session_state(
        &self,
        auth_ctx: &AuthContext,
        context_id: &str,
    ) -> AgentResult<Option<crate::runtime::context::SessionState>> {
        self.store.load_session_state(auth_ctx, context_id).await
    }
}
