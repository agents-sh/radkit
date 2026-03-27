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
            state: TaskState::Working as i32,
            timestamp: None,
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
            state: TaskState::Working as i32,
            timestamp: None,
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
            state: TaskState::Working as i32,
            timestamp: None,
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
                state: TaskState::Working as i32,
                timestamp: None,
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
            state: TaskState::Working as i32,
            timestamp: None,
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

#[cfg(all(feature = "runtime", not(all(target_os = "wasi", target_env = "p1"))))]
mod a2a_v1_runtime_tests {
    use super::*;
    use a2a_client::A2AClient;
    use a2a_types::{self as v1, AgentCard, Message, Part};
    use futures::StreamExt;
    use radkit::agent::{
        OnInputResult, OnRequestResult, RegisteredSkill, SkillHandler, SkillMetadata,
    };
    use radkit::errors::{AgentError, AgentResult};
    use radkit::models::{Content, LlmResponse};
    use radkit::runtime::context::{ProgressSender, State as SkillState};
    use std::time::Duration;

    struct ImmediateSkill;

    #[async_trait::async_trait]
    impl SkillHandler for ImmediateSkill {
        async fn on_request(
            &self,
            _state: &mut SkillState,
            _progress: &ProgressSender,
            _runtime: &dyn AgentRuntime,
            _content: Content,
        ) -> Result<OnRequestResult, AgentError> {
            Ok(OnRequestResult::Completed {
                message: Some(Content::from_text("Done")),
                artifacts: Vec::new(),
            })
        }

        async fn on_input_received(
            &self,
            _state: &mut SkillState,
            _progress: &ProgressSender,
            _runtime: &dyn AgentRuntime,
            _input: Content,
        ) -> Result<OnInputResult, AgentError> {
            unreachable!("immediate skill never continues");
        }
    }

    impl RegisteredSkill for ImmediateSkill {
        fn metadata() -> std::sync::Arc<SkillMetadata> {
            std::sync::Arc::new(SkillMetadata::new(
                "immediate-skill",
                "Immediate Skill",
                "Completes immediately",
                &[],
                &[],
                &[],
                &[],
            ))
        }
    }

    fn negotiation_response(skill_id: &str) -> AgentResult<LlmResponse> {
        FakeLlm::text_response(
            serde_json::json!({
                "type": "start_task",
                "skill_id": skill_id,
                "reasoning": "selected in test"
            })
            .to_string(),
        )
    }

    fn test_agent() -> radkit::agent::AgentDefinition {
        Agent::builder()
            .with_name("Test Agent")
            .with_version("1.0.0")
            .with_skill(ImmediateSkill)
            .build()
    }

    fn free_local_address() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let address = listener.local_addr().expect("listener address");
        drop(listener);
        address.to_string()
    }

    async fn wait_for_server(base_url: &str) {
        let client = reqwest::Client::new();
        let card_url = format!("{base_url}/.well-known/agent-card.json");

        for _ in 0..50 {
            let ready = client
                .get(&card_url)
                .send()
                .await
                .map(|response| response.status().is_success())
                .unwrap_or(false);
            if ready {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        panic!("runtime server did not start at {card_url}");
    }

    fn create_message(text: &str) -> Message {
        Message {
            message_id: uuid::Uuid::new_v4().to_string(),
            context_id: String::new(),
            task_id: String::new(),
            role: v1::Role::User.into(),
            parts: vec![Part {
                content: Some(v1::part::Content::Text(text.to_string())),
                metadata: None,
                filename: String::new(),
                media_type: "text/plain".to_string(),
            }],
            metadata: None,
            extensions: Vec::new(),
            reference_task_ids: Vec::new(),
        }
    }

    fn http_json_client(base_url: &str) -> A2AClient {
        A2AClient::from_card(AgentCard {
            name: "Test Agent".to_string(),
            description: "desc".to_string(),
            supported_interfaces: vec![v1::AgentInterface {
                url: base_url.to_string(),
                protocol_binding: "HTTP+JSON".to_string(),
                tenant: String::new(),
                protocol_version: "1.0".to_string(),
            }],
            provider: None,
            version: "1.0.0".to_string(),
            documentation_url: None,
            capabilities: Some(v1::AgentCapabilities {
                streaming: Some(true),
                push_notifications: Some(false),
                extensions: Vec::new(),
                extended_agent_card: Some(false),
            }),
            security_schemes: std::collections::HashMap::new(),
            security_requirements: Vec::new(),
            default_input_modes: vec!["text/plain".to_string()],
            default_output_modes: vec!["text/plain".to_string()],
            skills: Vec::new(),
            signatures: Vec::new(),
            icon_url: None,
        })
        .expect("http client")
    }

    fn rpc_client(base_url: &str) -> A2AClient {
        A2AClient::from_card(AgentCard {
            name: "Test Agent".to_string(),
            description: "desc".to_string(),
            supported_interfaces: vec![v1::AgentInterface {
                url: format!("{base_url}/rpc"),
                protocol_binding: "JSONRPC".to_string(),
                tenant: String::new(),
                protocol_version: "1.0".to_string(),
            }],
            provider: None,
            version: "1.0.0".to_string(),
            documentation_url: None,
            capabilities: Some(v1::AgentCapabilities {
                streaming: Some(true),
                push_notifications: Some(false),
                extensions: Vec::new(),
                extended_agent_card: Some(false),
            }),
            security_schemes: std::collections::HashMap::new(),
            security_requirements: Vec::new(),
            default_input_modes: vec!["text/plain".to_string()],
            default_output_modes: vec!["text/plain".to_string()],
            skills: Vec::new(),
            signatures: Vec::new(),
            icon_url: None,
        })
        .expect("rpc client")
    }

    #[tokio::test]
    async fn a2a_client_v1_http_json_roundtrip() {
        let llm = FakeLlm::with_responses("fake-llm", [negotiation_response("immediate-skill")]);
        let address = free_local_address();
        let base_url = format!("http://{address}");
        let runtime = Runtime::builder(test_agent(), llm)
            .base_url(base_url.clone())
            .build();
        let server = tokio::spawn(async move {
            runtime.serve(address).await.expect("runtime server");
        });

        wait_for_server(&base_url).await;

        let client = http_json_client(&base_url);
        let send_response = client
            .send_message(v1::SendMessageRequest {
                tenant: String::new(),
                message: Some(create_message("hello")),
                configuration: None,
                metadata: None,
            })
            .await
            .expect("send message");

        let task = match send_response.payload {
            Some(v1::send_message_response::Payload::Task(task)) => task,
            other => panic!("expected task payload, got {other:?}"),
        };
        assert_eq!(
            v1::TaskState::try_from(task.status.as_ref().expect("status").state)
                .expect("task state"),
            v1::TaskState::Completed
        );

        let fetched = client
            .get_task(v1::GetTaskRequest {
                tenant: String::new(),
                id: task.id.clone(),
                history_length: Some(5),
            })
            .await
            .expect("get task");
        assert_eq!(fetched.id, task.id);

        let listed = client
            .list_tasks(v1::ListTasksRequest {
                tenant: String::new(),
                context_id: task.context_id.clone(),
                status: v1::TaskState::Unspecified.into(),
                page_size: Some(10),
                page_token: String::new(),
                history_length: Some(5),
                status_timestamp_after: None,
                include_artifacts: Some(true),
            })
            .await
            .expect("list tasks");
        assert!(listed
            .tasks
            .iter()
            .any(|listed_task| listed_task.id == task.id));

        let card = client
            .get_extended_agent_card(v1::GetExtendedAgentCardRequest {
                tenant: String::new(),
            })
            .await
            .expect("extended card");
        assert_eq!(card.name, "Test Agent");

        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn a2a_client_v1_jsonrpc_stream_roundtrip() {
        let llm = FakeLlm::with_responses("fake-llm", [negotiation_response("immediate-skill")]);
        let address = free_local_address();
        let base_url = format!("http://{address}");
        let runtime = Runtime::builder(test_agent(), llm)
            .base_url(base_url.clone())
            .build();
        let server = tokio::spawn(async move {
            runtime.serve(address).await.expect("runtime server");
        });

        wait_for_server(&base_url).await;

        let client = rpc_client(&base_url);
        let mut stream = client
            .send_streaming_message(v1::SendMessageRequest {
                tenant: String::new(),
                message: Some(create_message("hello")),
                configuration: None,
                metadata: None,
            })
            .await
            .expect("stream start");

        let mut last_event = None;
        while let Some(item) = stream.next().await {
            let event = item.expect("stream event");
            last_event = Some(event);
        }

        let last_event = last_event.expect("final event");
        match last_event.payload {
            Some(v1::stream_response::Payload::Task(task)) => {
                assert_eq!(
                    v1::TaskState::try_from(task.status.as_ref().expect("status").state)
                        .expect("task state"),
                    v1::TaskState::Completed
                );
            }
            Some(v1::stream_response::Payload::StatusUpdate(update)) => {
                assert_eq!(
                    v1::TaskState::try_from(update.status.as_ref().expect("status").state)
                        .expect("task state"),
                    v1::TaskState::Completed
                );
            }
            other => panic!("unexpected final payload: {other:?}"),
        }

        server.abort();
        let _ = server.await;
    }
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
                    state: TaskState::Working as i32,
                    timestamp: None,
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
