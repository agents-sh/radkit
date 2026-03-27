//! LLM-backed skill handler for `AgentSkills`.
//!
//! This module provides [`LlmSkillHandler`] — the internal [`SkillHandler`]
//! implementation that powers all `AgentSkills` loaded from `SKILL.md` files.
//! It is not part of the public API; users interact with `AgentSkills` through
//! [`AgentSkillDef`], [`include_skill!`], and [`AgentBuilder::with_skill_dir`].

// All items are used via `dyn SkillHandler` trait objects and `Arc<dyn SkillHandler>`,
// so the compiler cannot see direct usage.
#![allow(dead_code)]

use std::sync::Arc;

use crate::{
    agent::{
        skill::{OnInputResult, OnRequestResult, SkillSlot},
        LlmWorker, WorkStatus,
    },
    errors::AgentError,
    models::{BaseLlm, Content, Event, Thread},
    runtime::{
        context::{ProgressSender, State},
        AgentRuntime,
    },
};

/// A [`SkillHandler`] that executes an `AgentSkill` using an LLM.
///
/// The SKILL.md instructions become the system prompt. `LlmWorker<WorkStatus>`
/// drives the LLM call loop. Multi-turn conversations are supported by
/// serialising the full `Thread` into the [`SkillSlot`] between turns.
pub struct LlmSkillHandler {
    worker: LlmWorker<WorkStatus>,
}

impl LlmSkillHandler {
    /// Build from a shared LLM handle and the SKILL.md instruction body.
    pub(crate) fn new(llm: Arc<dyn BaseLlm>, instructions: impl Into<String>) -> Self {
        let worker = LlmWorker::<WorkStatus>::builder_shared(llm)
            .with_system_instructions(instructions)
            .build();
        Self { worker }
    }
}

#[cfg_attr(
    all(target_os = "wasi", target_env = "p1"),
    async_trait::async_trait(?Send)
)]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
impl crate::agent::skill::SkillHandler for LlmSkillHandler {
    async fn on_request(
        &self,
        _state: &mut State,
        _progress: &ProgressSender,
        _runtime: &dyn AgentRuntime,
        content: Content,
    ) -> Result<OnRequestResult, AgentError> {
        let (status, thread) = self.worker.run_and_continue(content).await?;
        Ok(work_status_to_request_result(status, thread))
    }

    async fn on_input_received(
        &self,
        state: &mut State,
        _progress: &ProgressSender,
        _runtime: &dyn AgentRuntime,
        content: Content,
    ) -> Result<OnInputResult, AgentError> {
        // Restore the conversation thread saved from the previous turn.
        let thread: Thread = state
            .slot::<Thread>()?
            .ok_or_else(|| AgentError::Internal {
                component: "LlmSkillHandler".into(),
                reason: "slot was missing when on_input_received was called".into(),
            })?;

        // Append the user's new message and continue the conversation.
        let updated = thread.add_event(Event::user(content));
        let (status, thread) = self.worker.run_and_continue(updated).await?;
        Ok(work_status_to_input_result(status, thread))
    }
}

// ── Conversion helpers ────────────────────────────────────────────────────────

fn work_status_to_request_result(status: WorkStatus, thread: Thread) -> OnRequestResult {
    match status {
        WorkStatus::Complete { message } => OnRequestResult::Completed {
            message: Some(Content::from_text(message)),
            artifacts: vec![],
        },
        WorkStatus::NeedsInput { message } => OnRequestResult::InputRequired {
            message: Content::from_text(message),
            slot: SkillSlot::new(thread),
        },
        WorkStatus::Failed { reason } => OnRequestResult::Failed {
            error: Content::from_text(reason),
        },
    }
}

fn work_status_to_input_result(status: WorkStatus, thread: Thread) -> OnInputResult {
    match status {
        WorkStatus::Complete { message } => OnInputResult::Completed {
            message: Some(Content::from_text(message)),
            artifacts: vec![],
        },
        WorkStatus::NeedsInput { message } => OnInputResult::InputRequired {
            message: Content::from_text(message),
            slot: SkillSlot::new(thread),
        },
        WorkStatus::Failed { reason } => OnInputResult::Failed {
            error: Content::from_text(reason),
        },
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "test-support"))]
mod tests {
    use super::*;
    use crate::{
        agent::skill::SkillHandler,
        models::{Content, LlmResponse, TokenUsage},
        runtime::{
            auth::StaticAuthService,
            context::{ProgressSender, SessionState, State, TaskState},
            logging::ConsoleLoggingService,
            memory::InMemoryMemoryService,
            AgentRuntime,
        },
        test_support::FakeLlm,
    };

    // ── Helpers ───────────────────────────────────────────────────────────

    fn make_state() -> State {
        State::with_states(TaskState::new(), SessionState::new())
    }

    /// Build a no-op `AgentRuntime` for tests — `LlmSkillHandler` doesn't use it.
    fn noop_runtime() -> Arc<dyn AgentRuntime> {
        struct NoopRuntime(Arc<dyn crate::models::BaseLlm>);
        impl AgentRuntime for NoopRuntime {
            fn auth(&self) -> Arc<dyn crate::runtime::AuthService> {
                Arc::new(StaticAuthService::default())
            }
            fn memory(&self) -> Arc<dyn crate::runtime::MemoryService> {
                Arc::new(InMemoryMemoryService::new())
            }
            fn logging(&self) -> Arc<dyn crate::runtime::LoggingService> {
                Arc::new(ConsoleLoggingService)
            }
            #[cfg(feature = "runtime")]
            fn default_llm(&self) -> Arc<dyn crate::models::BaseLlm> {
                self.0.clone()
            }
        }
        let llm = FakeLlm::with_responses("noop", std::iter::empty());
        Arc::new(NoopRuntime(Arc::new(llm)))
    }

    fn llm_responding(json: &str) -> Arc<dyn BaseLlm> {
        let resp = Ok(LlmResponse::new(
            Content::from_text(json.to_string()),
            TokenUsage::empty(),
        ));
        Arc::new(FakeLlm::with_responses("skill-llm", [resp]))
    }

    // ── on_request tests ──────────────────────────────────────────────────

    #[tokio::test]
    async fn complete_response_produces_completed_result() {
        let handler = LlmSkillHandler::new(
            llm_responding(r#"{"status":"complete","message":"Here is your answer."}"#),
            "You are a helper.",
        );

        let result = handler
            .on_request(
                &mut make_state(),
                &ProgressSender::noop(),
                &*noop_runtime(),
                Content::from_text("Do something"),
            )
            .await
            .expect("no error");

        match result {
            OnRequestResult::Completed { message, .. } => {
                assert_eq!(
                    message
                        .expect("message present")
                        .into_first_text()
                        .as_deref(),
                    Some("Here is your answer.")
                );
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn needs_input_produces_input_required_with_thread_in_slot() {
        let handler = LlmSkillHandler::new(
            llm_responding(
                r#"{"status":"needs_input","message":"Which language should I translate to?"}"#,
            ),
            "You are a translator.",
        );

        let result = handler
            .on_request(
                &mut make_state(),
                &ProgressSender::noop(),
                &*noop_runtime(),
                Content::from_text("Translate: Hello"),
            )
            .await
            .expect("no error");

        match result {
            OnRequestResult::InputRequired { message, slot } => {
                assert!(message.first_text().unwrap_or("").contains("language"));
                // The slot must contain a serialised Thread.
                let thread: Thread = slot.deserialize().expect("slot should be a Thread");
                assert!(!thread.events().is_empty(), "thread should have events");
            }
            other => panic!("expected InputRequired, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn failed_response_produces_failed_result() {
        let handler = LlmSkillHandler::new(
            llm_responding(r#"{"status":"failed","reason":"I cannot do that."}"#),
            "You are a helper.",
        );

        let result = handler
            .on_request(
                &mut make_state(),
                &ProgressSender::noop(),
                &*noop_runtime(),
                Content::from_text("Do something impossible"),
            )
            .await
            .expect("no error");

        match result {
            OnRequestResult::Failed { error } => {
                assert!(error.first_text().unwrap_or("").contains("cannot"));
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    // ── on_input_received tests ───────────────────────────────────────────

    #[tokio::test]
    async fn on_input_received_missing_slot_returns_internal_error() {
        let handler = LlmSkillHandler::new(
            llm_responding(r#"{"status":"complete","message":"Done"}"#),
            "You are a helper.",
        );

        // State with no slot set — simulates a bug.
        let err = handler
            .on_input_received(
                &mut make_state(),
                &ProgressSender::noop(),
                &*noop_runtime(),
                Content::from_text("More info"),
            )
            .await
            .expect_err("should fail when slot is absent");

        assert!(matches!(err, AgentError::Internal { .. }));
    }

    #[tokio::test]
    async fn on_input_received_restores_thread_and_completes() {
        // Turn 1: LLM asks for more info.
        let turn1_handler = LlmSkillHandler::new(
            llm_responding(r#"{"status":"needs_input","message":"What language?"}"#),
            "You are a translator.",
        );

        let mut state = make_state();
        let first = turn1_handler
            .on_request(
                &mut state,
                &ProgressSender::noop(),
                &*noop_runtime(),
                Content::from_text("Translate: Hello"),
            )
            .await
            .expect("turn 1 ok");

        // Load the thread slot into state.
        let slot = match first {
            OnRequestResult::InputRequired { slot, .. } => slot,
            other => panic!("expected InputRequired, got {other:?}"),
        };
        state.set_pending_slot(slot);

        // Turn 2: LLM now completes.
        let turn2_handler = LlmSkillHandler::new(
            llm_responding(r#"{"status":"complete","message":"Hola"}"#),
            "You are a translator.",
        );

        let second = turn2_handler
            .on_input_received(
                &mut state,
                &ProgressSender::noop(),
                &*noop_runtime(),
                Content::from_text("Spanish"),
            )
            .await
            .expect("turn 2 ok");

        match second {
            OnInputResult::Completed { message, .. } => {
                let text = message.expect("message present").into_first_text();
                assert_eq!(text.as_deref(), Some("Hola"));
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn on_input_received_can_chain_multiple_needs_input_rounds() {
        let ask_again_handler = LlmSkillHandler::new(
            llm_responding(r#"{"status":"needs_input","message":"And the target dialect?"}"#),
            "You are a translator.",
        );

        // Simulate a state that already has a thread slot from a previous turn.
        let initial_thread = Thread::from_user("Translate: Hello");
        let mut state = make_state();
        let dummy_slot = crate::agent::skill::SkillSlot::new(initial_thread);
        state.set_pending_slot(dummy_slot);

        let result = ask_again_handler
            .on_input_received(
                &mut state,
                &ProgressSender::noop(),
                &*noop_runtime(),
                Content::from_text("Spanish"),
            )
            .await
            .expect("no error");

        // Should be InputRequired again with a new (longer) thread in the slot.
        match result {
            OnInputResult::InputRequired { slot, .. } => {
                let thread: Thread = slot.deserialize().expect("slot is Thread");
                // Should have at least 2 events (original + new user message).
                assert!(
                    thread.events().len() >= 2,
                    "thread should accumulate events: got {}",
                    thread.events().len()
                );
            }
            other => panic!("expected InputRequired again, got {other:?}"),
        }
    }
}
