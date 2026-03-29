//! `SQLite`-backed implementation of the [`TaskStore`] trait.
//!
//! This backend persists tasks, task events, task state, skill associations,
//! and session state in a local `SQLite` database on native targets.

use super::{Task, TaskEvent, TaskStore};
use crate::errors::{AgentError, AgentResult};
use crate::runtime::context::AuthContext;
use a2a_types::{Artifact, TaskStatus};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteRow};
use sqlx::{Row, SqlitePool};
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

const DEFAULT_BUSY_TIMEOUT: Duration = Duration::from_secs(5);
const DEFAULT_MAX_CONNECTIONS: u32 = 4;
const IN_MEMORY_MAX_CONNECTIONS: u32 = 1;

/// SQLite-backed task persistence for native runtimes.
#[derive(Clone, Debug)]
pub struct SqliteTaskStore {
    pool: SqlitePool,
}

impl SqliteTaskStore {
    /// Opens or creates a `SQLite`-backed store from a `SQLx` `SQLite` connection URL.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection URL is invalid, the database cannot be
    /// opened, or the schema initialization fails.
    pub async fn open(database_url: impl AsRef<str>) -> AgentResult<Self> {
        let database_url = database_url.as_ref();
        let options = SqliteConnectOptions::from_str(database_url)?
            .create_if_missing(true)
            .foreign_keys(true)
            .busy_timeout(DEFAULT_BUSY_TIMEOUT);
        let max_connections = if Self::is_in_memory_url(database_url) {
            IN_MEMORY_MAX_CONNECTIONS
        } else {
            DEFAULT_MAX_CONNECTIONS
        };

        Self::connect_with(options, max_connections).await
    }

    /// Opens or creates a `SQLite`-backed store from a filesystem path.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or the schema
    /// initialization fails.
    pub async fn from_path(path: impl AsRef<Path>) -> AgentResult<Self> {
        let options = SqliteConnectOptions::new()
            .filename(path.as_ref())
            .create_if_missing(true)
            .foreign_keys(true)
            .busy_timeout(DEFAULT_BUSY_TIMEOUT)
            .journal_mode(SqliteJournalMode::Wal);

        Self::connect_with(options, DEFAULT_MAX_CONNECTIONS).await
    }

    /// Wraps an existing pool and ensures the schema is present.
    ///
    /// # Errors
    ///
    /// Returns an error if schema initialization fails for the provided pool.
    pub async fn from_pool(pool: SqlitePool) -> AgentResult<Self> {
        let store = Self { pool };
        store.initialize().await?;
        Ok(store)
    }

    #[cfg(test)]
    async fn in_memory() -> AgentResult<Self> {
        Self::open("sqlite::memory:").await
    }

    async fn connect_with(
        options: SqliteConnectOptions,
        max_connections: u32,
    ) -> AgentResult<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(max_connections)
            .connect_with(options)
            .await?;
        Self::from_pool(pool).await
    }

    async fn initialize(&self) -> AgentResult<()> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS tasks (
                app_name TEXT NOT NULL,
                user_name TEXT NOT NULL,
                task_id TEXT NOT NULL,
                context_id TEXT NOT NULL,
                status_json TEXT NOT NULL,
                artifacts_json TEXT NOT NULL,
                PRIMARY KEY (app_name, user_name, task_id)
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_tasks_context
             ON tasks (app_name, user_name, context_id)",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS task_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                app_name TEXT NOT NULL,
                user_name TEXT NOT NULL,
                task_key TEXT NOT NULL,
                event_json TEXT NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_task_events_lookup
             ON task_events (app_name, user_name, task_key, id)",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS task_states (
                app_name TEXT NOT NULL,
                user_name TEXT NOT NULL,
                task_id TEXT NOT NULL,
                state_json TEXT NOT NULL,
                PRIMARY KEY (app_name, user_name, task_id)
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS task_skills (
                app_name TEXT NOT NULL,
                user_name TEXT NOT NULL,
                task_id TEXT NOT NULL,
                skill_id TEXT NOT NULL,
                PRIMARY KEY (app_name, user_name, task_id)
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS session_states (
                app_name TEXT NOT NULL,
                user_name TEXT NOT NULL,
                context_id TEXT NOT NULL,
                state_json TEXT NOT NULL,
                PRIMARY KEY (app_name, user_name, context_id)
            )",
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    fn is_in_memory_url(database_url: &str) -> bool {
        database_url.contains(":memory:") || database_url.contains("mode=memory")
    }

    fn serialize_json<T>(value: &T, label: &str) -> AgentResult<String>
    where
        T: serde::Serialize,
    {
        serde_json::to_string(value).map_err(|error| AgentError::Serialization {
            format: "json".to_string(),
            reason: format!("Failed to serialize {label}: {error}"),
        })
    }

    fn deserialize_json<T>(value: &str, label: &str) -> AgentResult<T>
    where
        T: serde::de::DeserializeOwned,
    {
        serde_json::from_str(value).map_err(|error| AgentError::Serialization {
            format: "json".to_string(),
            reason: format!("Failed to deserialize {label}: {error}"),
        })
    }

    fn task_from_row(row: &SqliteRow) -> AgentResult<Task> {
        let status_json: String = row.try_get("status_json")?;
        let artifacts_json: String = row.try_get("artifacts_json")?;

        Ok(Task {
            id: row.try_get("task_id")?,
            context_id: row.try_get("context_id")?,
            status: Self::deserialize_json::<TaskStatus>(&status_json, "task status")?,
            artifacts: Self::deserialize_json::<Vec<Artifact>>(&artifacts_json, "task artifacts")?,
        })
    }
}

#[async_trait::async_trait]
impl TaskStore for SqliteTaskStore {
    async fn get_task(&self, auth_ctx: &AuthContext, task_id: &str) -> AgentResult<Option<Task>> {
        let row = sqlx::query(
            "SELECT task_id, context_id, status_json, artifacts_json
             FROM tasks
             WHERE app_name = ?1 AND user_name = ?2 AND task_id = ?3",
        )
        .bind(&auth_ctx.app_name)
        .bind(&auth_ctx.user_name)
        .bind(task_id)
        .fetch_optional(&self.pool)
        .await?;

        row.as_ref().map(Self::task_from_row).transpose()
    }

    async fn list_tasks(&self, auth_ctx: &AuthContext) -> AgentResult<Vec<Task>> {
        let rows = sqlx::query(
            "SELECT task_id, context_id, status_json, artifacts_json
             FROM tasks
             WHERE app_name = ?1 AND user_name = ?2",
        )
        .bind(&auth_ctx.app_name)
        .bind(&auth_ctx.user_name)
        .fetch_all(&self.pool)
        .await?;

        rows.iter().map(Self::task_from_row).collect()
    }

    async fn save_task(&self, auth_ctx: &AuthContext, task: &Task) -> AgentResult<()> {
        let status_json = Self::serialize_json(&task.status, "task status")?;
        let artifacts_json = Self::serialize_json(&task.artifacts, "task artifacts")?;

        sqlx::query(
            "INSERT INTO tasks (
                app_name, user_name, task_id, context_id, status_json, artifacts_json
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(app_name, user_name, task_id) DO UPDATE SET
                context_id = excluded.context_id,
                status_json = excluded.status_json,
                artifacts_json = excluded.artifacts_json",
        )
        .bind(&auth_ctx.app_name)
        .bind(&auth_ctx.user_name)
        .bind(&task.id)
        .bind(&task.context_id)
        .bind(status_json)
        .bind(artifacts_json)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn append_event(
        &self,
        auth_ctx: &AuthContext,
        task_key: &str,
        event: &TaskEvent,
    ) -> AgentResult<()> {
        let event_json = Self::serialize_json(event, "task event")?;

        sqlx::query(
            "INSERT INTO task_events (app_name, user_name, task_key, event_json)
             VALUES (?1, ?2, ?3, ?4)",
        )
        .bind(&auth_ctx.app_name)
        .bind(&auth_ctx.user_name)
        .bind(task_key)
        .bind(event_json)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn get_events(
        &self,
        auth_ctx: &AuthContext,
        task_key: &str,
    ) -> AgentResult<Vec<TaskEvent>> {
        let rows = sqlx::query(
            "SELECT event_json
             FROM task_events
             WHERE app_name = ?1 AND user_name = ?2 AND task_key = ?3
             ORDER BY id ASC",
        )
        .bind(&auth_ctx.app_name)
        .bind(&auth_ctx.user_name)
        .bind(task_key)
        .fetch_all(&self.pool)
        .await?;

        rows.iter()
            .map(|row| {
                let event_json: String = row.try_get("event_json")?;
                Self::deserialize_json(&event_json, "task event")
            })
            .collect()
    }

    async fn list_event_task_keys(&self, auth_ctx: &AuthContext) -> AgentResult<Vec<String>> {
        let rows = sqlx::query(
            "SELECT DISTINCT task_key
             FROM task_events
             WHERE app_name = ?1 AND user_name = ?2
             ORDER BY task_key ASC",
        )
        .bind(&auth_ctx.app_name)
        .bind(&auth_ctx.user_name)
        .fetch_all(&self.pool)
        .await?;

        rows.iter()
            .map(|row| row.try_get("task_key").map_err(Into::into))
            .collect()
    }

    async fn list_task_ids(&self, auth_ctx: &AuthContext) -> AgentResult<Vec<String>> {
        let rows = sqlx::query(
            "SELECT task_id
             FROM tasks
             WHERE app_name = ?1 AND user_name = ?2
             ORDER BY task_id ASC",
        )
        .bind(&auth_ctx.app_name)
        .bind(&auth_ctx.user_name)
        .fetch_all(&self.pool)
        .await?;

        rows.iter()
            .map(|row| row.try_get("task_id").map_err(Into::into))
            .collect()
    }

    async fn list_context_ids(&self, auth_ctx: &AuthContext) -> AgentResult<Vec<String>> {
        let rows = sqlx::query(
            "SELECT DISTINCT context_id
             FROM tasks
             WHERE app_name = ?1 AND user_name = ?2
             ORDER BY context_id ASC",
        )
        .bind(&auth_ctx.app_name)
        .bind(&auth_ctx.user_name)
        .fetch_all(&self.pool)
        .await?;

        rows.iter()
            .map(|row| row.try_get("context_id").map_err(Into::into))
            .collect()
    }

    async fn save_task_state(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
        state: &crate::runtime::context::TaskState,
    ) -> AgentResult<()> {
        let state_json = Self::serialize_json(state, "task state")?;

        sqlx::query(
            "INSERT INTO task_states (app_name, user_name, task_id, state_json)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(app_name, user_name, task_id) DO UPDATE SET
                state_json = excluded.state_json",
        )
        .bind(&auth_ctx.app_name)
        .bind(&auth_ctx.user_name)
        .bind(task_id)
        .bind(state_json)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn load_task_state(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
    ) -> AgentResult<Option<crate::runtime::context::TaskState>> {
        let row = sqlx::query(
            "SELECT state_json
             FROM task_states
             WHERE app_name = ?1 AND user_name = ?2 AND task_id = ?3",
        )
        .bind(&auth_ctx.app_name)
        .bind(&auth_ctx.user_name)
        .bind(task_id)
        .fetch_optional(&self.pool)
        .await?;

        row.map(|row| {
            let state_json: String = row.try_get("state_json")?;
            Self::deserialize_json(&state_json, "task state")
        })
        .transpose()
    }

    async fn set_task_skill(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
        skill_id: &str,
    ) -> AgentResult<()> {
        sqlx::query(
            "INSERT INTO task_skills (app_name, user_name, task_id, skill_id)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(app_name, user_name, task_id) DO UPDATE SET
                skill_id = excluded.skill_id",
        )
        .bind(&auth_ctx.app_name)
        .bind(&auth_ctx.user_name)
        .bind(task_id)
        .bind(skill_id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn get_task_skill(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
    ) -> AgentResult<Option<String>> {
        let row = sqlx::query(
            "SELECT skill_id
             FROM task_skills
             WHERE app_name = ?1 AND user_name = ?2 AND task_id = ?3",
        )
        .bind(&auth_ctx.app_name)
        .bind(&auth_ctx.user_name)
        .bind(task_id)
        .fetch_optional(&self.pool)
        .await?;

        row.map(|row| row.try_get("skill_id").map_err(Into::into))
            .transpose()
    }

    async fn save_session_state(
        &self,
        auth_ctx: &AuthContext,
        context_id: &str,
        state: &crate::runtime::context::SessionState,
    ) -> AgentResult<()> {
        let state_json = Self::serialize_json(state, "session state")?;

        sqlx::query(
            "INSERT INTO session_states (app_name, user_name, context_id, state_json)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(app_name, user_name, context_id) DO UPDATE SET
                state_json = excluded.state_json",
        )
        .bind(&auth_ctx.app_name)
        .bind(&auth_ctx.user_name)
        .bind(context_id)
        .bind(state_json)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn load_session_state(
        &self,
        auth_ctx: &AuthContext,
        context_id: &str,
    ) -> AgentResult<Option<crate::runtime::context::SessionState>> {
        let row = sqlx::query(
            "SELECT state_json
             FROM session_states
             WHERE app_name = ?1 AND user_name = ?2 AND context_id = ?3",
        )
        .bind(&auth_ctx.app_name)
        .bind(&auth_ctx.user_name)
        .bind(context_id)
        .fetch_optional(&self.pool)
        .await?;

        row.map(|row| {
            let state_json: String = row.try_get("state_json")?;
            Self::deserialize_json(&state_json, "session state")
        })
        .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::context::{AuthContext, TaskState as RadkitTaskState};
    use crate::runtime::task_manager::{
        DefaultTaskManager, ListTasksFilter, TaskEvent, TaskManager,
    };
    use a2a_types::{
        Artifact, Message, Role, TaskArtifactUpdateEvent, TaskState, TaskStatus,
        TaskStatusUpdateEvent,
    };
    use uuid::Uuid;

    fn auth() -> AuthContext {
        AuthContext {
            app_name: "app".into(),
            user_name: "user".into(),
        }
    }

    fn make_message(id: &str, context: &str) -> Message {
        Message {
            message_id: id.into(),
            role: Role::Agent as i32,
            parts: Vec::new(),
            context_id: context.into(),
            task_id: String::new(),
            reference_task_ids: Vec::new(),
            extensions: Vec::new(),
            metadata: None,
        }
    }

    fn temp_db_path(test_name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("radkit-{test_name}-{}.sqlite", Uuid::new_v4()))
    }

    fn cleanup_db_files(path: &Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(format!("{}-wal", path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", path.display()));
    }

    // This integration-style test exercises the full TaskStore contract in one flow.
    #[allow(clippy::too_many_lines)]
    #[tokio::test(flavor = "current_thread")]
    async fn stores_tasks_events_and_context() {
        let store = SqliteTaskStore::in_memory().await.expect("store");
        let manager = DefaultTaskManager::new(store);
        let auth_ctx = auth();
        let task = Task {
            id: "task-1".into(),
            context_id: "ctx-1".into(),
            status: TaskStatus {
                state: TaskState::Submitted.into(),
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
                state: TaskState::Working.into(),
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

    #[tokio::test(flavor = "current_thread")]
    async fn persists_across_reopen() {
        let path = temp_db_path("sqlite-task-store");
        let auth_ctx = auth();

        {
            let store = SqliteTaskStore::from_path(&path).await.expect("store");
            let manager = DefaultTaskManager::new(store);

            let task = Task {
                id: "task-1".into(),
                context_id: "ctx-1".into(),
                status: TaskStatus {
                    state: TaskState::Working.into(),
                    timestamp: None,
                    message: None,
                },
                artifacts: Vec::new(),
            };

            manager
                .save_task(&auth_ctx, &task)
                .await
                .expect("save task");

            let mut session_state = crate::runtime::context::SessionState::new();
            session_state
                .save("user_name", &"alice")
                .expect("save session");
            manager
                .save_session_state(&auth_ctx, "ctx-1", &session_state)
                .await
                .expect("save session state");
        }

        {
            let store = SqliteTaskStore::from_path(&path).await.expect("store");
            let manager = DefaultTaskManager::new(store);

            let task = manager
                .get_task(&auth_ctx, "task-1")
                .await
                .expect("get task")
                .expect("task exists");
            assert_eq!(task.context_id, "ctx-1");

            let session_state = manager
                .load_session_state(&auth_ctx, "ctx-1")
                .await
                .expect("load session")
                .expect("session exists");
            let user_name: Option<String> =
                session_state.load("user_name").expect("session user name");
            assert_eq!(user_name.as_deref(), Some("alice"));
        }

        cleanup_db_files(&path);
    }
}
