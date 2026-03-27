//! Integration tests for RequestExecutor orchestration.

#[cfg(all(
    feature = "runtime",
    feature = "test-support",
    not(all(target_os = "wasi", target_env = "p1"))
))]
mod tests {
    use a2a_types::{self as v1, part, Role, TaskState};
    use radkit::agent::{
        Agent, OnInputResult, OnRequestResult, RegisteredSkill, SkillHandler, SkillMetadata,
        SkillSlot,
    };
    use radkit::errors::AgentError;
    use radkit::models::{Content, LlmResponse, TokenUsage};
    use radkit::runtime::context::{ProgressSender, State};
    use radkit::runtime::core::executor::{ExecutorRuntime, RequestExecutor};
    use radkit::runtime::{AgentRuntime, Runtime};
    use radkit::test_support::FakeLlm;
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;
    use uuid::Uuid;

    fn negotiation_response(skill_id: &str) -> radkit::errors::AgentResult<LlmResponse> {
        let decision = serde_json::json!({
            "type": "start_task",
            "skill_id": skill_id,
            "reasoning": "Test selected this skill"
        });
        Ok(LlmResponse::new(
            Content::from_text(serde_json::to_string(&decision).expect("valid JSON")),
            TokenUsage::empty(),
        ))
    }

    fn create_send_request(
        text: &str,
        context_id: Option<String>,
        task_id: Option<String>,
    ) -> v1::SendMessageRequest {
        v1::SendMessageRequest {
            message: Some(v1::Message {
                message_id: Uuid::new_v4().to_string(),
                role: Role::User as i32,
                parts: vec![v1::Part {
                    content: Some(part::Content::Text(text.to_string())),
                    metadata: None,
                    filename: String::new(),
                    media_type: "text/plain".to_string(),
                }],
                context_id: context_id.unwrap_or_default(),
                task_id: task_id.unwrap_or_default(),
                reference_task_ids: vec![],
                extensions: vec![],
                metadata: None,
            }),
            configuration: None,
            metadata: None,
            tenant: String::new(),
        }
    }

    // ============================================================================
    // Test 1: New task creation and immediate completion
    // ============================================================================

    struct ImmediateSkill;

    static IMMEDIATE_METADATA: SkillMetadata = SkillMetadata::new(
        "immediate",
        "Immediate Skill",
        "Completes immediately",
        &[],
        &[],
        &[],
        &[],
    );

    #[cfg_attr(
        all(target_os = "wasi", target_env = "p1"),
        async_trait::async_trait(?Send)
    )]
    #[cfg_attr(
        not(all(target_os = "wasi", target_env = "p1")),
        async_trait::async_trait
    )]
    impl SkillHandler for ImmediateSkill {
        async fn on_request(
            &self,
            _state: &mut State,
            _progress: &ProgressSender,
            _runtime: &dyn AgentRuntime,
            _content: Content,
        ) -> Result<OnRequestResult, AgentError> {
            Ok(OnRequestResult::Completed {
                message: Some(Content::from_text("Task completed immediately!")),
                artifacts: vec![],
            })
        }

        async fn on_input_received(
            &self,
            _state: &mut State,
            _progress: &ProgressSender,
            _runtime: &dyn AgentRuntime,
            _input: Content,
        ) -> Result<OnInputResult, AgentError> {
            unreachable!("immediate skill should not receive input")
        }
    }

    impl RegisteredSkill for ImmediateSkill {
        fn metadata() -> &'static SkillMetadata {
            &IMMEDIATE_METADATA
        }
    }

    #[tokio::test]
    async fn test_new_task_immediate_completion() {
        let llm = FakeLlm::with_responses("fake-llm", [negotiation_response("immediate")]);
        let runtime = Runtime::builder(Agent::builder().with_skill(ImmediateSkill).build(), llm)
            .build()
            .into_shared();
        let executor_runtime: Arc<dyn ExecutorRuntime> = runtime.clone();
        let executor = RequestExecutor::new(executor_runtime);

        let result = executor
            .handle_send_message(create_send_request("Hello", None, None))
            .await;
        assert!(result.is_ok(), "send_message should succeed");

        match result.unwrap().payload {
            Some(v1::send_message_response::Payload::Task(task)) => {
                assert_eq!(
                    task.status.as_ref().unwrap().state,
                    TaskState::Completed as i32
                );
                assert!(!task.history.is_empty(), "should have messages in history");
            }
            _ => panic!("expected Task result"),
        }
    }

    // ============================================================================
    // Test 2: Task requires input and continuation
    // ============================================================================

    #[derive(Serialize, Deserialize, Clone, Debug)]
    enum GreetingSlot {
        AwaitingName,
    }

    struct GreetingSkill;

    static GREETING_METADATA: SkillMetadata = SkillMetadata::new(
        "greeting",
        "Greeting Skill",
        "Greets user by name",
        &[],
        &[],
        &[],
        &[],
    );

    #[cfg_attr(
        all(target_os = "wasi", target_env = "p1"),
        async_trait::async_trait(?Send)
    )]
    #[cfg_attr(
        not(all(target_os = "wasi", target_env = "p1")),
        async_trait::async_trait
    )]
    impl SkillHandler for GreetingSkill {
        async fn on_request(
            &self,
            _state: &mut State,
            _progress: &ProgressSender,
            _runtime: &dyn AgentRuntime,
            _content: Content,
        ) -> Result<OnRequestResult, AgentError> {
            Ok(OnRequestResult::InputRequired {
                message: Content::from_text("What is your name?"),
                slot: SkillSlot::new(GreetingSlot::AwaitingName),
            })
        }

        async fn on_input_received(
            &self,
            state: &mut State,
            _progress: &ProgressSender,
            _runtime: &dyn AgentRuntime,
            input: Content,
        ) -> Result<OnInputResult, AgentError> {
            let slot: GreetingSlot = state.slot()?.expect("slot should be available");
            match slot {
                GreetingSlot::AwaitingName => {
                    let name = input.first_text().unwrap_or("Friend");
                    Ok(OnInputResult::Completed {
                        message: Some(Content::from_text(format!("Hello, {}!", name))),
                        artifacts: vec![],
                    })
                }
            }
        }
    }

    impl RegisteredSkill for GreetingSkill {
        fn metadata() -> &'static SkillMetadata {
            &GREETING_METADATA
        }
    }

    #[tokio::test]
    async fn test_task_continuation_with_input() {
        let llm = FakeLlm::with_responses("fake-llm", [negotiation_response("greeting")]);
        let runtime = Runtime::builder(Agent::builder().with_skill(GreetingSkill).build(), llm)
            .build()
            .into_shared();
        let executor_runtime: Arc<dyn ExecutorRuntime> = runtime.clone();
        let executor = RequestExecutor::new(executor_runtime);

        let result1 = executor
            .handle_send_message(create_send_request("Greet me", None, None))
            .await
            .unwrap();
        let (context_id, task_id) = match result1.payload {
            Some(v1::send_message_response::Payload::Task(task)) => {
                assert_eq!(
                    task.status.as_ref().unwrap().state,
                    TaskState::InputRequired as i32
                );
                (task.context_id, task.id)
            }
            _ => panic!("expected Task result"),
        };

        let result2 = executor
            .handle_send_message(create_send_request(
                "Alice",
                Some(context_id),
                Some(task_id),
            ))
            .await
            .unwrap();
        match result2.payload {
            Some(v1::send_message_response::Payload::Task(task)) => {
                assert_eq!(
                    task.status.as_ref().unwrap().state,
                    TaskState::Completed as i32
                );
                let final_msg = task
                    .history
                    .iter()
                    .rfind(|msg| msg.role == Role::Agent as i32)
                    .expect("should have agent message");
                let text = match &final_msg.parts[0].content {
                    Some(part::Content::Text(t)) => t,
                    _ => panic!("expected text content"),
                };
                assert!(text.contains("Hello, Alice!"), "should greet by name");
            }
            _ => panic!("expected Task result"),
        }
    }

    // ============================================================================
    // Test 3: Task failure
    // ============================================================================

    struct FailingSkill;

    static FAILING_METADATA: SkillMetadata = SkillMetadata::new(
        "failing",
        "Failing Skill",
        "Always fails",
        &[],
        &[],
        &[],
        &[],
    );

    #[cfg_attr(
        all(target_os = "wasi", target_env = "p1"),
        async_trait::async_trait(?Send)
    )]
    #[cfg_attr(
        not(all(target_os = "wasi", target_env = "p1")),
        async_trait::async_trait
    )]
    impl SkillHandler for FailingSkill {
        async fn on_request(
            &self,
            _state: &mut State,
            _progress: &ProgressSender,
            _runtime: &dyn AgentRuntime,
            _content: Content,
        ) -> Result<OnRequestResult, AgentError> {
            Err(AgentError::Internal {
                component: "FailingSkill".to_string(),
                reason: "Intentional failure for testing".to_string(),
            })
        }

        async fn on_input_received(
            &self,
            _state: &mut State,
            _progress: &ProgressSender,
            _runtime: &dyn AgentRuntime,
            _input: Content,
        ) -> Result<OnInputResult, AgentError> {
            unreachable!()
        }
    }

    impl RegisteredSkill for FailingSkill {
        fn metadata() -> &'static SkillMetadata {
            &FAILING_METADATA
        }
    }

    #[tokio::test]
    async fn test_task_failure() {
        let llm = FakeLlm::with_responses("fake-llm", [negotiation_response("failing")]);
        let runtime = Runtime::builder(Agent::builder().with_skill(FailingSkill).build(), llm)
            .build()
            .into_shared();
        let executor_runtime: Arc<dyn ExecutorRuntime> = runtime.clone();
        let executor = RequestExecutor::new(executor_runtime);

        let result = executor
            .handle_send_message(create_send_request("Do something", None, None))
            .await;

        assert!(result.is_err(), "Should fail when skill throws error");
        match result.unwrap_err() {
            AgentError::Internal { component, reason } => {
                assert_eq!(component, "FailingSkill");
                assert!(reason.contains("Intentional failure"));
            }
            other => panic!("Expected Internal error, got {:?}", other),
        }
    }

    // ============================================================================
    // Test 4: Task retrieval
    // ============================================================================

    #[tokio::test]
    async fn test_get_task() {
        let llm = FakeLlm::with_responses("fake-llm", [negotiation_response("immediate")]);
        let runtime = Runtime::builder(Agent::builder().with_skill(ImmediateSkill).build(), llm)
            .build()
            .into_shared();
        let executor_runtime: Arc<dyn ExecutorRuntime> = runtime.clone();
        let executor = RequestExecutor::new(executor_runtime);

        let send_result = executor
            .handle_send_message(create_send_request("test", None, None))
            .await
            .unwrap();
        let task_id = match send_result.payload {
            Some(v1::send_message_response::Payload::Task(task)) => task.id,
            _ => panic!("expected Task"),
        };

        let get_result = executor
            .handle_get_task(v1::GetTaskRequest {
                id: task_id.clone(),
                history_length: None,
                tenant: String::new(),
            })
            .await;

        assert!(get_result.is_ok(), "should retrieve task");
        let retrieved_task = get_result.unwrap();
        assert_eq!(retrieved_task.id, task_id);
        assert_eq!(
            retrieved_task.status.as_ref().unwrap().state,
            TaskState::Completed as i32
        );
    }

    // ============================================================================
    // Test 5: Invalid task ID
    // ============================================================================

    #[tokio::test]
    async fn test_get_nonexistent_task() {
        let llm = FakeLlm::with_responses("fake-llm", [negotiation_response("immediate")]);
        let runtime = Runtime::builder(Agent::builder().with_skill(ImmediateSkill).build(), llm)
            .build()
            .into_shared();
        let executor_runtime: Arc<dyn ExecutorRuntime> = runtime.clone();
        let executor = RequestExecutor::new(executor_runtime);

        let result = executor
            .handle_get_task(v1::GetTaskRequest {
                id: "nonexistent-task-id".to_string(),
                history_length: None,
                tenant: String::new(),
            })
            .await;

        assert!(result.is_err(), "should fail for nonexistent task");
        match result.unwrap_err() {
            AgentError::TaskNotFound { task_id } => {
                assert_eq!(task_id, "nonexistent-task-id");
            }
            _ => panic!("expected TaskNotFound error"),
        }
    }

    // ============================================================================
    // Test 6: Continue with wrong context_id/task_id combination
    // ============================================================================

    #[tokio::test]
    async fn test_invalid_context_task_combination() {
        let llm = FakeLlm::with_responses("fake-llm", [negotiation_response("greeting")]);
        let runtime = Runtime::builder(Agent::builder().with_skill(GreetingSkill).build(), llm)
            .build()
            .into_shared();
        let executor_runtime: Arc<dyn ExecutorRuntime> = runtime.clone();
        let executor = RequestExecutor::new(executor_runtime);

        let result1 = executor
            .handle_send_message(create_send_request("Hello", None, None))
            .await
            .unwrap();
        let task1_id = match result1.payload {
            Some(v1::send_message_response::Payload::Task(task)) => task.id,
            _ => panic!("expected Task"),
        };

        let result2 = executor
            .handle_send_message(create_send_request(
                "Continue",
                Some("wrong-context-id".to_string()),
                Some(task1_id),
            ))
            .await;
        assert!(result2.is_err(), "should fail with mismatched context/task");
    }
}
