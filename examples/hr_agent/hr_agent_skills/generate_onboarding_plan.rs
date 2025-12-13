// --- Skill: Generate Onboarding Plan ---

use super::summarize_resume::UserData;
use radkit::agent::{Artifact, LlmFunction, OnRequestResult, SkillHandler};
use radkit::errors::AgentError;
use radkit::macros::skill;
use radkit::models::providers::OpenRouterLlm;
use radkit::models::Content;
use radkit::runtime::context::{ProgressSender, State};
use radkit::runtime::AgentRuntime;

fn generate_onboarding_tasks() -> LlmFunction<Vec<String>> {
    let llm = OpenRouterLlm::from_env("anthropic/claude-3.5-sonnet")
        .expect("Failed to create LLM from environment");

    LlmFunction::new_with_system_instructions(
        llm,
        "Generate a comprehensive list of onboarding tasks for the provided role. \
         Include technical setup, documentation review, and team introductions.",
    )
}

#[skill(
    id = "generate_onboarding_plan",
    name = "Onboarding Plan Generator",
    description = "Generates personalized onboarding plans for new hires",
    tags = ["hr", "onboarding", "planning"],
    examples = [
        "Generate an onboarding plan for a Software Engineer",
        "Create onboarding tasks for the new hire"
    ],
    input_modes = ["text/plain", "application/json"],
    output_modes = ["application/json"]
)]
pub struct GenerateOnboardingPlanSkill;

#[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
impl SkillHandler for GenerateOnboardingPlanSkill {
    async fn on_request(
        &self,
        state: &mut State,
        progress: &ProgressSender,
        _runtime: &dyn AgentRuntime,
        _content: Content,
    ) -> Result<OnRequestResult, AgentError> {
        // Send intermediate update
        progress.send_update("Looking for user profile...").await?;

        // Load user data from session state (saved by previous skill in same session)
        let user_data: Option<UserData> = state.session().load("user_data")?;

        if let Some(user_data) = user_data {
            let role = "Software Engineer"; // Placeholder - could be extracted from context

            // Send intermediate update
            progress
                .send_update(format!("Generating onboarding tasks for {} role...", role))
                .await?;

            let tasks = generate_onboarding_tasks().run(role).await?;

            let plan_artifact = Artifact::from_json("onboarding_plan.json", &tasks)?;

            Ok(OnRequestResult::Completed {
                message: Some(Content::from_text(format!(
                    "Onboarding plan generated for {}.",
                    user_data.name
                ))),
                artifacts: vec![plan_artifact],
            })
        } else {
            Ok(OnRequestResult::Failed {
                error: Content::from_text(
                    "Could not find a summarized resume in the current session.",
                ),
            })
        }
    }
}
