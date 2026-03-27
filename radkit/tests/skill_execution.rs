use radkit::agent::{
    Agent, AgentDefinition, Artifact, OnInputResult, OnRequestResult, RegisteredSkill,
    SkillHandler, SkillMetadata, SkillSlot,
};
use radkit::errors::AgentError;
use radkit::models::Content;
use radkit::runtime::context::{ProgressSender, State};
use radkit::runtime::{AgentRuntime, Runtime};
use radkit::test_support::FakeLlm;

// A test skill for verifying lifecycle behavior.
struct LifecycleSkill;

#[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
enum LifecycleSlot {
    AwaitingInput,
}

#[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
impl SkillHandler for LifecycleSkill {
    async fn on_request(
        &self,
        state: &mut State,
        _progress: &ProgressSender,
        _runtime: &dyn AgentRuntime,
        _content: Content,
    ) -> Result<OnRequestResult, AgentError> {
        state.task().save("request_seen", &true)?;
        Ok(OnRequestResult::InputRequired {
            message: Content::from_text("Please provide input."),
            slot: SkillSlot::new(LifecycleSlot::AwaitingInput),
        })
    }

    async fn on_input_received(
        &self,
        state: &mut State,
        _progress: &ProgressSender,
        _runtime: &dyn AgentRuntime,
        content: Content,
    ) -> Result<OnInputResult, AgentError> {
        let request_seen: bool = state.task().load("request_seen")?.unwrap_or(false);
        assert!(request_seen, "on_request should have been called first");

        let slot: LifecycleSlot = state.slot()?.expect("slot should be set");
        assert_eq!(slot, LifecycleSlot::AwaitingInput);

        let input_text = content.first_text().unwrap_or("");
        if input_text == "complete" {
            Ok(OnInputResult::Completed {
                message: Some(Content::from_text("Completed!")),
                artifacts: vec![],
            })
        } else {
            Ok(OnInputResult::Failed {
                error: Content::from_text("Invalid input"),
            })
        }
    }
}

impl RegisteredSkill for LifecycleSkill {
    fn metadata() -> std::sync::Arc<SkillMetadata> {
        std::sync::Arc::new(SkillMetadata::new(
            "lifecycle_skill",
            "Lifecycle Skill",
            "A skill for testing the execution lifecycle.",
            &[],
            &[],
            &[],
            &[],
        ))
    }
}

fn lifecycle_agent_definition() -> AgentDefinition {
    Agent::builder().with_skill(LifecycleSkill).build()
}

#[tokio::test]
async fn test_skill_lifecycle() {
    let llm = FakeLlm::with_responses("fake_llm", std::iter::empty());
    let agent = lifecycle_agent_definition();
    let runtime = Runtime::builder(lifecycle_agent_definition(), llm).build();

    let skill = agent.skills().first().unwrap();
    let mut state = State::new();
    let progress = ProgressSender::noop();

    // 1. Test on_request
    let request_result = skill
        .handler()
        .on_request(&mut state, &progress, &runtime, Content::from_text("start"))
        .await
        .unwrap();

    match request_result {
        OnRequestResult::InputRequired { message, slot } => {
            assert_eq!(message.first_text(), Some("Please provide input."));
            let slot_value: LifecycleSlot = slot.deserialize().unwrap();
            assert_eq!(slot_value, LifecycleSlot::AwaitingInput);
            // Simulate what the executor does: store the slot for continuation
            state.set_slot(slot).unwrap();
        }
        _ => panic!("Expected InputRequired"),
    }

    // 2. Test on_input_received
    let input_result = skill
        .handler()
        .on_input_received(
            &mut state,
            &progress,
            &runtime,
            Content::from_text("complete"),
        )
        .await
        .unwrap();

    match input_result {
        OnInputResult::Completed { message, .. } => {
            assert_eq!(message.unwrap().first_text(), Some("Completed!"));
        }
        _ => panic!("Expected Completed"),
    }
}

// Test skill that returns Failed on invalid input
#[tokio::test]
async fn test_skill_lifecycle_with_failure() {
    let llm = FakeLlm::with_responses("fake_llm", std::iter::empty());
    let agent = lifecycle_agent_definition();
    let runtime = Runtime::builder(lifecycle_agent_definition(), llm).build();

    let skill = agent.skills().first().unwrap();
    let mut state = State::new();
    let progress = ProgressSender::noop();

    // 1. Call on_request
    let request_result = skill
        .handler()
        .on_request(&mut state, &progress, &runtime, Content::from_text("start"))
        .await
        .unwrap();

    // 2. Extract and set the slot
    if let OnRequestResult::InputRequired { slot, .. } = request_result {
        state.set_slot(slot).unwrap();
    }

    // 3. Test on_input_received with invalid input
    let input_result = skill
        .handler()
        .on_input_received(
            &mut state,
            &progress,
            &runtime,
            Content::from_text("invalid"),
        )
        .await
        .unwrap();

    match input_result {
        OnInputResult::Failed { error } => {
            assert_eq!(error.first_text(), Some("Invalid input"));
        }
        _ => panic!("Expected Failed"),
    }
}

// Test skill that completes immediately without requiring input
struct ImmediateSkill;

#[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
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
        content: Content,
    ) -> Result<OnRequestResult, AgentError> {
        let text = content.first_text().unwrap_or("no input");
        Ok(OnRequestResult::Completed {
            message: Some(Content::from_text(format!("Processed: {text}"))),
            artifacts: vec![Artifact::from_text("result", "success")],
        })
    }
}

impl RegisteredSkill for ImmediateSkill {
    fn metadata() -> std::sync::Arc<SkillMetadata> {
        std::sync::Arc::new(SkillMetadata::new(
            "immediate_skill",
            "Immediate Skill",
            "A skill that completes immediately.",
            &[],
            &[],
            &[],
            &[],
        ))
    }
}

fn immediate_agent_definition() -> AgentDefinition {
    Agent::builder().with_skill(ImmediateSkill).build()
}

#[tokio::test]
async fn test_immediate_completion_skill() {
    let llm = FakeLlm::with_responses("fake_llm", std::iter::empty());
    let agent = immediate_agent_definition();
    let runtime = Runtime::builder(immediate_agent_definition(), llm).build();

    let skill = agent.skills().first().unwrap();
    let mut state = State::new();
    let progress = ProgressSender::noop();

    let request_result = skill
        .handler()
        .on_request(&mut state, &progress, &runtime, Content::from_text("test"))
        .await
        .unwrap();

    match request_result {
        OnRequestResult::Completed { message, artifacts } => {
            assert_eq!(message.unwrap().first_text(), Some("Processed: test"));
            assert_eq!(artifacts.len(), 1);
            assert_eq!(artifacts[0].name(), "result");
        }
        _ => panic!("Expected Completed"),
    }
}

// Test skill that rejects requests
struct RejectingSkill;

#[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
impl SkillHandler for RejectingSkill {
    async fn on_request(
        &self,
        _state: &mut State,
        _progress: &ProgressSender,
        _runtime: &dyn AgentRuntime,
        content: Content,
    ) -> Result<OnRequestResult, AgentError> {
        let text = content.first_text().unwrap_or("");
        if text.contains("forbidden") {
            Ok(OnRequestResult::Rejected {
                reason: Content::from_text("This request is forbidden"),
            })
        } else {
            Ok(OnRequestResult::Completed {
                message: Some(Content::from_text("Accepted")),
                artifacts: vec![],
            })
        }
    }
}

impl RegisteredSkill for RejectingSkill {
    fn metadata() -> std::sync::Arc<SkillMetadata> {
        std::sync::Arc::new(SkillMetadata::new(
            "rejecting_skill",
            "Rejecting Skill",
            "A skill that rejects certain requests.",
            &[],
            &[],
            &[],
            &[],
        ))
    }
}

fn rejecting_agent_definition() -> AgentDefinition {
    Agent::builder().with_skill(RejectingSkill).build()
}

#[tokio::test]
async fn test_rejecting_skill() {
    let llm = FakeLlm::with_responses("fake_llm", std::iter::empty());
    let agent = rejecting_agent_definition();
    let runtime = Runtime::builder(rejecting_agent_definition(), llm).build();

    let skill = agent.skills().first().unwrap();
    let mut state = State::new();
    let progress = ProgressSender::noop();

    // Test rejection
    let request_result = skill
        .handler()
        .on_request(
            &mut state,
            &progress,
            &runtime,
            Content::from_text("forbidden action"),
        )
        .await
        .unwrap();

    match request_result {
        OnRequestResult::Rejected { reason } => {
            assert_eq!(reason.first_text(), Some("This request is forbidden"));
        }
        _ => panic!("Expected Rejected"),
    }

    // Test acceptance
    let mut state = State::new();
    let request_result = skill
        .handler()
        .on_request(
            &mut state,
            &progress,
            &runtime,
            Content::from_text("allowed action"),
        )
        .await
        .unwrap();

    match request_result {
        OnRequestResult::Completed { .. } => {
            // Success
        }
        _ => panic!("Expected Completed"),
    }
}

// Test skill with multi-round input requests
struct MultiRoundSkill;

#[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
enum MultiRoundSlot {
    AwaitingName,
    AwaitingAge { name: String },
}

#[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
impl SkillHandler for MultiRoundSkill {
    async fn on_request(
        &self,
        _state: &mut State,
        _progress: &ProgressSender,
        _runtime: &dyn AgentRuntime,
        _content: Content,
    ) -> Result<OnRequestResult, AgentError> {
        Ok(OnRequestResult::InputRequired {
            message: Content::from_text("What is your name?"),
            slot: SkillSlot::new(MultiRoundSlot::AwaitingName),
        })
    }

    async fn on_input_received(
        &self,
        state: &mut State,
        _progress: &ProgressSender,
        _runtime: &dyn AgentRuntime,
        content: Content,
    ) -> Result<OnInputResult, AgentError> {
        let slot: MultiRoundSlot = state.slot()?.expect("slot should be set");

        match slot {
            MultiRoundSlot::AwaitingName => {
                let name = content.first_text().unwrap_or("Unknown").to_string();
                Ok(OnInputResult::InputRequired {
                    message: Content::from_text("What is your age?"),
                    slot: SkillSlot::new(MultiRoundSlot::AwaitingAge { name }),
                })
            }
            MultiRoundSlot::AwaitingAge { name } => {
                let age = content.first_text().unwrap_or("0");
                Ok(OnInputResult::Completed {
                    message: Some(Content::from_text(format!("Hello {name}, age {age}!"))),
                    artifacts: vec![],
                })
            }
        }
    }
}

impl RegisteredSkill for MultiRoundSkill {
    fn metadata() -> std::sync::Arc<SkillMetadata> {
        std::sync::Arc::new(SkillMetadata::new(
            "multi_round_skill",
            "Multi-Round Skill",
            "A skill that requires multiple rounds of input.",
            &[],
            &[],
            &[],
            &[],
        ))
    }
}

fn multi_round_agent_definition() -> AgentDefinition {
    Agent::builder().with_skill(MultiRoundSkill).build()
}

#[tokio::test]
async fn test_multi_round_input_skill() {
    let llm = FakeLlm::with_responses("fake_llm", std::iter::empty());
    let agent = multi_round_agent_definition();
    let runtime = Runtime::builder(multi_round_agent_definition(), llm).build();

    let skill = agent.skills().first().unwrap();
    let mut state = State::new();
    let progress = ProgressSender::noop();

    // Round 1: Initial request
    let request_result = skill
        .handler()
        .on_request(&mut state, &progress, &runtime, Content::from_text("start"))
        .await
        .unwrap();

    match request_result {
        OnRequestResult::InputRequired { message, slot } => {
            assert_eq!(message.first_text(), Some("What is your name?"));
            state.set_slot(slot).unwrap();
        }
        _ => panic!("Expected InputRequired"),
    }

    // Round 2: Provide name
    let input_result = skill
        .handler()
        .on_input_received(&mut state, &progress, &runtime, Content::from_text("Alice"))
        .await
        .unwrap();

    match input_result {
        OnInputResult::InputRequired { message, slot } => {
            assert_eq!(message.first_text(), Some("What is your age?"));
            state.set_slot(slot).unwrap();
        }
        _ => panic!("Expected InputRequired"),
    }

    // Round 3: Provide age
    let input_result = skill
        .handler()
        .on_input_received(&mut state, &progress, &runtime, Content::from_text("30"))
        .await
        .unwrap();

    match input_result {
        OnInputResult::Completed { message, .. } => {
            assert_eq!(message.unwrap().first_text(), Some("Hello Alice, age 30!"));
        }
        _ => panic!("Expected Completed"),
    }
}
