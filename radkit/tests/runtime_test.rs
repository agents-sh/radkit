//! Integration tests for runtime services.
//!
//! These tests verify that the runtime services (TaskManager, MemoryService, LoggingService,
//! AuthService) work correctly together within a runtime handle.

use radkit::agent::Agent;
use radkit::runtime::context::AuthContext;
use radkit::runtime::memory::{ContentSource, MemoryContent, SearchOptions};
use radkit::runtime::task_manager::InMemoryTaskStore;
use radkit::runtime::{AgentRuntime, ListTasksFilter, LogLevel, Runtime};
use radkit::test_support::FakeLlm;
use std::collections::HashMap;

fn test_agent() -> radkit::agent::AgentDefinition {
    Agent::builder().with_name("Test Agent").build()
}

fn runtime_with_manager(llm: FakeLlm) -> Runtime {
    Runtime::builder(test_agent(), llm)
        .with_task_store(InMemoryTaskStore::new())
        .build()
}

#[tokio::test]
async fn test_auth_service_provides_context() {
    let llm = FakeLlm::with_responses("fake_llm", std::iter::empty());
    let runtime = Runtime::builder(test_agent(), llm).build();

    let auth_context = runtime.auth().get_auth_context();

    assert_eq!(auth_context.app_name, "default-app");
    assert_eq!(auth_context.user_name, "default-user");
}

#[tokio::test]
async fn test_task_manager_save_and_get() {
    let llm = FakeLlm::with_responses("fake_llm", std::iter::empty());
    let runtime = runtime_with_manager(llm);
    let task_manager = runtime.task_manager();

    let auth_context = runtime.auth().get_auth_context();

    use a2a_types::{TaskState, TaskStatus};
    use radkit::runtime::task_manager::Task;

    // Create a task
    let task = Task {
        id: "test-task-1".to_string(),
        context_id: "test-context-1".to_string(),
        status: TaskStatus {
            state: TaskState::Working,
            timestamp: Some(chrono::Utc::now().to_rfc3339()),
            message: None,
        },
        artifacts: vec![],
    };

    task_manager
        .save_task(&auth_context, &task)
        .await
        .expect("save task");

    // Retrieve the task
    let retrieved = task_manager
        .get_task(&auth_context, "test-task-1")
        .await
        .expect("get task")
        .expect("task should exist");

    assert_eq!(retrieved.id, "test-task-1");
    assert_eq!(retrieved.context_id, "test-context-1");
}

#[tokio::test]
async fn test_task_manager_auth_scoping() {
    let llm = FakeLlm::with_responses("fake_llm", std::iter::empty());
    let runtime = runtime_with_manager(llm).into_shared();
    let task_manager = runtime.task_manager();

    // Create two different auth contexts
    let auth_context_1 = AuthContext {
        app_name: "app1".to_string(),
        user_name: "user1".to_string(),
    };

    let auth_context_2 = AuthContext {
        app_name: "app2".to_string(),
        user_name: "user2".to_string(),
    };

    use a2a_types::{TaskState, TaskStatus};
    use radkit::runtime::task_manager::Task;

    // Save task for auth_context_1
    let task1 = Task {
        id: "task-1".to_string(),
        context_id: "context-1".to_string(),
        status: TaskStatus {
            state: TaskState::Working,
            timestamp: Some(chrono::Utc::now().to_rfc3339()),
            message: None,
        },
        artifacts: vec![],
    };

    task_manager
        .save_task(&auth_context_1, &task1)
        .await
        .expect("save task for auth1");

    // Save task for auth_context_2
    let task2 = Task {
        id: "task-2".to_string(),
        context_id: "context-2".to_string(),
        status: TaskStatus {
            state: TaskState::Working,
            timestamp: Some(chrono::Utc::now().to_rfc3339()),
            message: None,
        },
        artifacts: vec![],
    };

    task_manager
        .save_task(&auth_context_2, &task2)
        .await
        .expect("save task for auth2");

    // Verify auth_context_1 can only see its own task
    let retrieved = task_manager
        .get_task(&auth_context_1, "task-1")
        .await
        .expect("get task")
        .expect("task should exist");
    assert_eq!(retrieved.id, "task-1");

    let not_found = task_manager
        .get_task(&auth_context_1, "task-2")
        .await
        .expect("get task");
    assert!(not_found.is_none(), "auth_context_1 should not see task-2");

    // Verify auth_context_2 can only see its own task
    let retrieved = task_manager
        .get_task(&auth_context_2, "task-2")
        .await
        .expect("get task")
        .expect("task should exist");
    assert_eq!(retrieved.id, "task-2");

    let not_found = task_manager
        .get_task(&auth_context_2, "task-1")
        .await
        .expect("get task");
    assert!(not_found.is_none(), "auth_context_2 should not see task-1");
}

#[tokio::test]
async fn test_task_manager_list_tasks() {
    let llm = FakeLlm::with_responses("fake_llm", std::iter::empty());
    let runtime = runtime_with_manager(llm).into_shared();
    let task_manager = runtime.task_manager();
    let auth_context = runtime.auth().get_auth_context();

    use a2a_types::{TaskState, TaskStatus};
    use radkit::runtime::task_manager::Task;

    // Create multiple tasks
    for i in 1..=5 {
        let task = Task {
            id: format!("task-{i}"),
            context_id: format!("context-{i}"),
            status: TaskStatus {
                state: TaskState::Working,
                timestamp: Some(chrono::Utc::now().to_rfc3339()),
                message: None,
            },
            artifacts: vec![],
        };

        task_manager
            .save_task(&auth_context, &task)
            .await
            .expect("save task");
    }

    // List all tasks
    let filter = ListTasksFilter::default();
    let result = task_manager
        .list_tasks(&auth_context, &filter)
        .await
        .expect("list tasks");

    assert!(result.items.len() >= 5, "Should have at least 5 tasks");
}

#[tokio::test]
async fn test_memory_service_add_and_search() {
    let llm = FakeLlm::with_responses("fake_llm", std::iter::empty());
    let runtime = Runtime::builder(test_agent(), llm).build();

    let auth = runtime.auth();
    let auth_context = auth.get_auth_context();
    let memory = runtime.memory();

    // Add a user fact
    let content = MemoryContent {
        text: "User Alice prefers dark mode theme".to_string(),
        source: ContentSource::UserFact {
            category: Some("preferences".to_string()),
        },
        metadata: HashMap::new(),
    };

    let id = memory
        .add(&auth_context, content)
        .await
        .expect("add memory");

    assert!(id.starts_with("fact:preferences:"));

    // Search for the fact
    let results = memory
        .search(&auth_context, "dark mode", SearchOptions::history_only())
        .await
        .expect("search memory");

    assert_eq!(results.len(), 1);
    assert!(results[0].text.contains("dark mode"));
}

#[tokio::test]
async fn test_memory_service_auth_scoping() {
    let llm = FakeLlm::with_responses("fake_llm", std::iter::empty());
    let runtime = Runtime::builder(test_agent(), llm).build();

    let memory = runtime.memory();

    // Create two different auth contexts
    let auth_context_1 = AuthContext {
        app_name: "app1".to_string(),
        user_name: "user1".to_string(),
    };

    let auth_context_2 = AuthContext {
        app_name: "app2".to_string(),
        user_name: "user2".to_string(),
    };

    // Add fact for auth_context_1
    memory
        .add(
            &auth_context_1,
            MemoryContent {
                text: "User1 secret information".to_string(),
                source: ContentSource::UserFact { category: None },
                metadata: HashMap::new(),
            },
        )
        .await
        .expect("add for auth1");

    // Add fact for auth_context_2
    memory
        .add(
            &auth_context_2,
            MemoryContent {
                text: "User2 secret information".to_string(),
                source: ContentSource::UserFact { category: None },
                metadata: HashMap::new(),
            },
        )
        .await
        .expect("add for auth2");

    // Verify auth_context_1 only sees its own data
    let results1 = memory
        .search(&auth_context_1, "secret", SearchOptions::default())
        .await
        .expect("search auth1");
    assert_eq!(results1.len(), 1);
    assert!(results1[0].text.contains("User1"));

    // Verify auth_context_2 only sees its own data
    let results2 = memory
        .search(&auth_context_2, "secret", SearchOptions::default())
        .await
        .expect("search auth2");
    assert_eq!(results2.len(), 1);
    assert!(results2[0].text.contains("User2"));
}

#[tokio::test]
async fn test_logging_service() {
    let llm = FakeLlm::with_responses("fake_llm", std::iter::empty());
    let runtime = Runtime::builder(test_agent(), llm).build();

    let logging = runtime.logging();

    // Test logging at different levels (should not panic)
    logging.log(LogLevel::Info, "Test info message");
    logging.log(LogLevel::Warn, "Test warn message");
    logging.log(LogLevel::Error, "Test error message");
    logging.log(LogLevel::Debug, "Test debug message");
}

#[tokio::test]
async fn test_runtime_services_together() {
    // Test that all services can be used together in a workflow
    let llm = FakeLlm::with_responses("fake_llm", std::iter::empty());
    let runtime = runtime_with_manager(llm).into_shared();
    let task_manager = runtime.task_manager();

    let auth = runtime.auth();
    let auth_context = auth.get_auth_context();

    // Use logging
    runtime.logging().log(LogLevel::Info, "Starting workflow");

    // Use memory to store workflow state as a user fact
    let workflow_content = MemoryContent {
        text: "Workflow step 1: processing data".to_string(),
        source: ContentSource::UserFact {
            category: Some("workflow".to_string()),
        },
        metadata: HashMap::new(),
    };
    let _workflow_id = runtime
        .memory()
        .add(&auth_context, workflow_content)
        .await
        .expect("save workflow state");

    // Create a task
    use a2a_types::{TaskState, TaskStatus};
    use radkit::runtime::task_manager::Task;

    let task = Task {
        id: "workflow-task-1".to_string(),
        context_id: "workflow-context-1".to_string(),
        status: TaskStatus {
            state: TaskState::Working,
            timestamp: Some(chrono::Utc::now().to_rfc3339()),
            message: None,
        },
        artifacts: vec![],
    };

    task_manager
        .save_task(&auth_context, &task)
        .await
        .expect("save task");

    // Retrieve workflow state via search
    let workflow_results = runtime
        .memory()
        .search(
            &auth_context,
            "workflow processing",
            SearchOptions::default(),
        )
        .await
        .expect("search workflow state");

    assert!(!workflow_results.is_empty(), "Should find workflow state");
    assert!(workflow_results[0].text.contains("processing"));

    // Retrieve task
    let retrieved_task = task_manager
        .get_task(&auth_context, "workflow-task-1")
        .await
        .expect("get task")
        .expect("task should exist");

    assert_eq!(retrieved_task.id, "workflow-task-1");

    // Log completion
    runtime.logging().log(LogLevel::Info, "Workflow completed");
}

#[cfg(all(
    feature = "runtime",
    feature = "task-store-sqlite",
    not(all(target_os = "wasi", target_env = "p1"))
))]
mod sqlite_task_store_tests {
    use super::*;
    use a2a_types::{TaskState, TaskStatus};
    use radkit::runtime::task_manager::Task;
    use radkit::runtime::SqliteTaskStore;
    use std::path::Path;
    use uuid::Uuid;

    fn temp_db_path(test_name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("radkit-{test_name}-{}.sqlite", Uuid::new_v4()))
    }

    fn cleanup_db_files(path: &Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(format!("{}-wal", path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", path.display()));
    }

    #[tokio::test]
    async fn test_sqlite_task_store_persists_across_runtime_rebuild() {
        let path = temp_db_path("runtime-sqlite-store");

        {
            let llm = FakeLlm::with_responses("fake_llm", std::iter::empty());
            let store = SqliteTaskStore::from_path(&path)
                .await
                .expect("sqlite store");
            let runtime = Runtime::builder(test_agent(), llm)
                .with_task_store(store)
                .build();

            let auth_context = runtime.auth().get_auth_context();
            let task = Task {
                id: "persisted-task".to_string(),
                context_id: "persisted-context".to_string(),
                status: TaskStatus {
                    state: TaskState::Working,
                    timestamp: Some(chrono::Utc::now().to_rfc3339()),
                    message: None,
                },
                artifacts: vec![],
            };

            runtime
                .task_manager()
                .save_task(&auth_context, &task)
                .await
                .expect("save task");
        }

        {
            let llm = FakeLlm::with_responses("fake_llm", std::iter::empty());
            let store = SqliteTaskStore::from_path(&path)
                .await
                .expect("sqlite store");
            let runtime = Runtime::builder(test_agent(), llm)
                .with_task_store(store)
                .build();

            let auth_context = runtime.auth().get_auth_context();
            let task = runtime
                .task_manager()
                .get_task(&auth_context, "persisted-task")
                .await
                .expect("get task")
                .expect("task should exist");

            assert_eq!(task.context_id, "persisted-context");
        }

        cleanup_db_files(&path);
    }
}
