//! Default implementation of the Negotiator trait.

use std::sync::Arc;

use a2a_types::Message;

use crate::agent::{AgentDefinition, LlmFunction};
use crate::errors::AgentResult;
use crate::models::{BaseLlm, Content, Event, Thread};
use crate::runtime::context::AuthContext;
use crate::runtime::core::negotiator::{NegotiationDecision, Negotiator};

/// Returns the default negotiation prompt.
///
/// This prompt instructs the LLM to:
/// - Understand the user's intent
/// - Match it against available skills
/// - Ask clarifying questions when needed
/// - Decide when to proceed with task creation
#[must_use]
pub fn default_negotiation_prompt() -> Content {
    Content::from_text(
        r"You are an agent negotiator. Your role is to understand the user's intent and determine which skill should handle their request.

Available information:
- Agent skills: A list of available skills with their descriptions
- User message: The user's current message
- Context: Previous messages in this conversation (if any)
- Related tasks: IDs of related tasks in this context (if any)

Your responsibilities:
1. Analyze the user's message to understand their intent
2. Match the intent against the available skills
3. If a skill is clearly identified and you have all necessary information, respond with the skill ID
4. If the intent is ambiguous or you need more information, ask clarifying questions

Response format:
- If ready to start a task: Return the skill_id and any processed content
- If more information is needed: Return a clarifying question for the user

Be helpful, concise, and professional in your responses.",
    )
}

/// Default negotiator implementation.
///
/// This implementation provides basic negotiation logic using an LLM:
/// - Uses structured LLM outputs to determine intent
/// - Asks clarifying questions when skill selection is ambiguous
/// - Transitions to task creation when a skill is clearly identified
/// - Rejects requests that are out of scope for available skills
///
/// # Examples
///
/// ```ignore
/// use radkit::runtime::DefaultNegotiator;
/// use radkit::models::providers::AnthropicLlm;
/// use radkit::models::Content;
///
/// let llm = Arc::new(AnthropicLlm::from_env("claude-sonnet-4")?);
///
/// // Use default prompt
/// let negotiator = DefaultNegotiator::new(llm.clone());
///
/// // Override the prompt
/// let negotiator = DefaultNegotiator::new(llm)
///     .with_prompt(Content::from_text("Custom negotiation prompt"));
/// ```
#[derive(Clone)]
pub struct DefaultNegotiator {
    llm: Arc<dyn BaseLlm>,
    custom_prompt: Option<Content>,
}

impl DefaultNegotiator {
    /// Creates a new `DefaultNegotiator` with the given LLM provider.
    ///
    /// # Arguments
    ///
    /// * `llm` - The LLM provider to use for negotiation decisions
    pub fn new(llm: Arc<dyn BaseLlm>) -> Self {
        Self {
            llm,
            custom_prompt: None,
        }
    }

    /// Sets a custom negotiation prompt.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Custom system prompt to override the default
    #[must_use]
    pub fn with_prompt(mut self, prompt: Content) -> Self {
        self.custom_prompt = Some(prompt);
        self
    }

    /// Builds the negotiation system prompt from the agent definition.
    ///
    /// Creates a comprehensive prompt that includes:
    /// - Agent metadata (name, description)
    /// - Available skills with descriptions
    /// - Decision instructions
    fn build_negotiation_prompt(agent_def: &AgentDefinition) -> String {
        let mut prompt = format!(
            r#"You are the negotiator for the "{}" agent.

Agent Description: {}

Your role is to analyze user requests and determine the appropriate action."#,
            agent_def.name(),
            agent_def.description().unwrap_or("A helpful AI agent")
        );

        // Add available skills section
        prompt.push_str("\n\n## Available Skills\n\n");
        if agent_def.skills().is_empty() {
            prompt.push_str("No skills are currently available.\n");
        } else {
            use std::fmt::Write;
            for skill in agent_def.skills() {
                let metadata = skill.metadata();
                let _ = writeln!(
                    prompt,
                    "- **{}** (`{}`): {}",
                    metadata.name, metadata.id, metadata.description
                );
                if !metadata.examples.is_empty() {
                    prompt.push_str("  Examples:\n");
                    for example in metadata.examples {
                        let _ = writeln!(prompt, "    - {example}");
                    }
                }
            }
        }

        // Add decision instructions
        prompt.push_str(
            r"

## Your Task

Analyze the user's message and decide on ONE of the following actions:

1. **START_TASK** - If you can clearly identify which skill should handle this request and you have sufficient information to proceed.
   - Return the skill_id of the matching skill
   - Provide reasoning for your selection

2. **ASK_CLARIFICATION** - If the user's intent is unclear, ambiguous, or missing critical information.
   - Ask specific, helpful questions
   - Guide the user toward providing the information needed
   - Be concise but thorough

3. **REJECT** - If the request is outside the scope of all available skills.
   - Politely explain why this cannot be handled
   - Suggest what the agent CAN do (mention available skills)
   - Be helpful and professional

## Guidelines

- Be decisive: Don't ask for clarification if the intent is clear
- Be specific: When asking questions, explain exactly what information you need
- Be helpful: When rejecting, explain what the agent can actually help with
- Consider context: If this is a follow-up message, take the conversation history into account
- Match accurately: Only suggest a skill if it genuinely matches the user's intent

Respond with a structured decision following the NegotiationDecision schema."
        );

        prompt
    }

    /// Converts A2A Messages to a Thread for LLM consumption.
    fn messages_to_thread(messages: Vec<Message>) -> Thread {
        use a2a_types::Role;

        let events: Vec<Event> = messages
            .into_iter()
            .map(|msg| {
                let role = msg.role;
                let content = Content::from(msg);

                if role == Role::User as i32 {
                    Event::user(content)
                } else {
                    Event::assistant(content)
                }
            })
            .collect();

        Thread::new(events)
    }

    /// Executes the LLM to make a negotiation decision.
    async fn execute_negotiation(
        &self,
        system_prompt: String,
        thread: Thread,
    ) -> AgentResult<NegotiationDecision> {
        // Create LlmFunction with shared model reference
        let llm_function = LlmFunction::<NegotiationDecision>::new_with_shared_model(
            Arc::clone(&self.llm),
            Some(system_prompt),
        );

        llm_function.run(thread).await
    }
}

#[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
impl Negotiator for DefaultNegotiator {
    async fn negotiate(
        &self,
        _auth_ctx: &AuthContext,
        agent_def: &AgentDefinition,
        _context_id: &str,
        content: Content,
        history: Vec<Message>,
    ) -> AgentResult<NegotiationDecision> {
        // Build the negotiation system prompt
        let system_prompt = Self::build_negotiation_prompt(agent_def);

        // Create thread from history + new user message
        let thread = if history.is_empty() {
            // New conversation - just the user message
            let user_event = Event::user(content);
            Thread::new(vec![user_event])
        } else {
            // Continuing conversation - history + new message
            let thread = Self::messages_to_thread(history);
            thread.add_event(Event::user(content))
        };

        // Execute LLM to get negotiation decision
        // Executor is responsible for storing all messages
        self.execute_negotiation(system_prompt, thread).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{Agent, OnRequestResult, RegisteredSkill, SkillHandler, SkillMetadata};
    use crate::errors::{AgentError, AgentResult};
    use crate::models::LlmResponse;
    use crate::runtime::context::{ProgressSender, State};
    use crate::runtime::AgentRuntime;
    use crate::test_support::FakeLlm;
    use std::sync::Arc;

    struct StubSkill;

    static STUB_METADATA: SkillMetadata = SkillMetadata::new(
        "stub-skill",
        "Stub Skill",
        "A skill used for negotiation tests",
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
    impl SkillHandler for StubSkill {
        async fn on_request(
            &self,
            _state: &mut State,
            _progress: &ProgressSender,
            _runtime: &dyn AgentRuntime,
            _content: Content,
        ) -> Result<OnRequestResult, AgentError> {
            Ok(OnRequestResult::Completed {
                message: None,
                artifacts: Vec::new(),
            })
        }
    }

    impl RegisteredSkill for StubSkill {
        fn metadata() -> &'static SkillMetadata {
            &STUB_METADATA
        }
    }

    fn negotiation_response(skill_id: &str) -> AgentResult<LlmResponse> {
        let json_str = format!(
            r#"{{"type": "start_task", "skill_id": "{skill_id}", "reasoning": "clear intent"}}"#
        );
        FakeLlm::content_response(Content::from_text(json_str))
    }

    #[tokio::test(flavor = "current_thread")]
    async fn negotiate_returns_start_task_decision() {
        let llm: Arc<dyn BaseLlm> = Arc::new(FakeLlm::with_responses(
            "negotiator",
            [negotiation_response("stub-skill")],
        ));
        let negotiator = DefaultNegotiator::new(llm);

        let agent = Agent::builder()
            .with_name("Agent")
            .with_skill(StubSkill)
            .build();

        let decision = negotiator
            .negotiate(
                &AuthContext {
                    app_name: "app".into(),
                    user_name: "user".into(),
                },
                &agent,
                "ctx",
                Content::from_text("perform stub-skill"),
                Vec::new(),
            )
            .await
            .expect("decision");

        match decision {
            NegotiationDecision::StartTask { skill_id, .. } => {
                assert_eq!(skill_id, "stub-skill");
            }
            other => panic!("unexpected decision: {other:?}"),
        }
    }
}
