// --- Skill: Summarize Resume (Attempt and Continue Pattern) ---

use radkit::agent::{
    Artifact, LlmFunction, LlmWorker, OnInputResult, OnRequestResult, SkillHandler, SkillSlot,
};
use radkit::errors::AgentError;
use radkit::macros::{skill, tool, LLMOutput};
use radkit::models::providers::OpenRouterLlm;
use radkit::models::Content;
use radkit::runtime::context::{ProgressSender, State};
use radkit::runtime::AgentRuntime;
use radkit::tools::ToolResult;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

// --- Data Structures, Enums, and Callable Helpers for this Skill ---

#[derive(Serialize, Deserialize, LLMOutput, JsonSchema, Debug, Clone)]
pub struct UserData {
    pub name: String,
    pub email: String,
    pub github_username: String,
}

#[derive(Serialize, Deserialize, LLMOutput, JsonSchema, Debug, Clone)]
pub struct GitHubRepository {
    pub name: String,
    pub url: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FinalUser {
    pub name: String,
    pub email: String,
    pub github_username: String,
    pub repos: Vec<GitHubRepository>,
}

/// This enum defines all the possible pieces of information this skill might
/// need to ask the user for. It provides compile-time safety for input handling.
#[derive(Serialize, Deserialize)]
pub enum ResumeInputSlot {
    CorrectedUsername,
}

// Defines the callable struct for extracting user data.
fn extract_user_data() -> LlmFunction<UserData> {
    // Create LLM locally for this function
    let llm = OpenRouterLlm::from_env("anthropic/claude-3.5-sonnet")
        .expect("Failed to create LLM from environment");

    LlmFunction::new_with_system_instructions(
        llm,
        "Extract the user's name, email, and github username from the text.",
    )
}

#[derive(Deserialize, JsonSchema)]
struct GetRepoArgs {
    // GitHub username
    username: String,
}
#[tool(description = "GitHub tool to inspect for public repositories")]
async fn get_repos(args: GetRepoArgs) -> ToolResult {
    // Read optional GitHub access token from environment
    // Token is optional but recommended for higher rate limits (60/hour without vs 5000/hour with)
    let token = std::env::var("GITHUB_ACCESS_TOKEN").ok();

    // Build the GitHub API URL
    let url = format!("https://api.github.com/users/{}/repos", args.username);

    // Create HTTP client
    let client = match reqwest::Client::builder()
        .user_agent("radkit-hr-agent")
        .build()
    {
        Ok(c) => c,
        Err(e) => return ToolResult::error(format!("Failed to create HTTP client: {}", e)),
    };

    // Build the request
    let mut request = client.get(&url);

    // Add authorization header if token is available for higher rate limits
    if let Some(token) = token {
        request = request.header("Authorization", format!("Bearer {}", token));
    }

    // Execute the request
    let response = match request.send().await {
        Ok(r) => r,
        Err(e) => return ToolResult::error(format!("Failed to fetch GitHub repos: {}", e)),
    };

    // Check if the request was successful
    if !response.status().is_success() {
        let status = response.status();
        return ToolResult::error(format!(
            "GitHub API returned error status {}: {}",
            status,
            response.text().await.unwrap_or_default()
        ));
    }

    // Parse the JSON response
    let repos: Vec<serde_json::Value> = match response.json().await {
        Ok(r) => r,
        Err(e) => return ToolResult::error(format!("Failed to parse GitHub response: {}", e)),
    };

    // Transform to our repository format
    let repos: Vec<GitHubRepository> = repos
        .iter()
        .filter_map(|repo| {
            let name = repo.get("name")?.as_str()?.to_string();
            let url = repo.get("html_url")?.as_str()?.to_string();
            Some(GitHubRepository { name, url })
        })
        .collect();

    ToolResult::success(json!({
        "repos": repos,
        "count": repos.len(),
    }))
}

// Defines the callable struct for extracting GitHub repos.
fn extract_user_github_repos() -> LlmWorker<Vec<GitHubRepository>> {
    // Create LLM locally for this worker
    let llm = OpenRouterLlm::from_env("anthropic/claude-3.5-sonnet")
        .expect("Failed to create LLM from environment");

    LlmWorker::builder(llm)
        .with_system_instructions("Get the user's public GitHub repositories.")
        .with_tools(vec![get_repos])
        .build()
}

// --- Skill Logic Implementation ---

#[skill(
    id = "summarize_resume",
    name = "Resume Summarizer",
    description = "Extracts structured data from resumes and validates GitHub profiles",
    tags = ["hr", "resume", "onboarding"],
    examples = [
        "Summarize this resume: John Doe, john@example.com, github.com/johndoe",
        "Extract information from the attached resume"
    ],
    input_modes = ["text/plain", "application/pdf"],
    output_modes = ["application/json"]
)]
pub struct SummarizeResumeSkill;

#[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
impl SkillHandler for SummarizeResumeSkill {
    /// The "Attempt" handler: tries the full happy path.
    async fn on_request(
        &self,
        state: &mut State,
        progress: &ProgressSender,
        _runtime: &dyn AgentRuntime,
        content: Content,
    ) -> Result<OnRequestResult, AgentError> {
        // Send intermediate status update
        progress.send_update("Analyzing resume...").await?;

        // Extract text from content
        let resume_text = content
            .first_text()
            .ok_or_else(|| AgentError::MissingInput("No text content in message".to_string()))?;

        // Extract user data
        let user_data = extract_user_data().run(resume_text).await?;

        // Save to session state for cross-skill sharing
        state.session().save("user_data", &user_data)?;

        // Send another intermediate update
        progress
            .send_update("Fetching GitHub repositories...")
            .await?;

        // Fetch GitHub repos
        let repos = extract_user_github_repos()
            .run(&user_data.github_username)
            .await?;

        if repos.is_empty() {
            // Happy path failed. Save partial data and request input.
            state.task().save("partial_user", &user_data)?;

            Ok(OnRequestResult::InputRequired {
                message: Content::from_text(
                    "No GitHub repos found. Please provide the correct username.",
                ),
                slot: SkillSlot::new(ResumeInputSlot::CorrectedUsername),
            })
        } else {
            // Happy path succeeded. Complete the task directly.
            let user = FinalUser {
                name: user_data.name,
                email: user_data.email,
                github_username: user_data.github_username,
                repos,
            };

            let artifact = Artifact::from_json("user_profile.json", &user)?;

            Ok(OnRequestResult::Completed {
                message: Some(Content::from_text(
                    "Successfully extracted user profile and repositories.",
                )),
                artifacts: vec![artifact],
            })
        }
    }

    /// The "Continue" handler: runs after user provides the missing input.
    async fn on_input_received(
        &self,
        state: &mut State,
        _progress: &ProgressSender,
        _runtime: &dyn AgentRuntime,
        content: Content,
    ) -> Result<OnInputResult, AgentError> {
        let slot: ResumeInputSlot = state
            .slot()?
            .ok_or_else(|| AgentError::SkillSlot("No slot found".to_string()))?;

        match slot {
            ResumeInputSlot::CorrectedUsername => {
                let partial_user: UserData =
                    state.task().load("partial_user")?.ok_or_else(|| {
                        AgentError::ContextError("No partial user data found".to_string())
                    })?;

                let corrected_username = content
                    .first_text()
                    .ok_or_else(|| AgentError::MissingInput("No username provided".to_string()))?;

                let repos = extract_user_github_repos().run(corrected_username).await?;

                if repos.is_empty() {
                    Ok(OnInputResult::Failed {
                        error: Content::from_text(
                            "Could not find repositories with the provided username.",
                        ),
                    })
                } else {
                    let user = FinalUser {
                        name: partial_user.name,
                        email: partial_user.email,
                        github_username: corrected_username.to_string(),
                        repos,
                    };

                    let artifact = Artifact::from_json("user_profile.json", &user)?;

                    Ok(OnInputResult::Completed {
                        message: Some(Content::from_text(
                            "Successfully extracted user profile with corrected username.",
                        )),
                        artifacts: vec![artifact],
                    })
                }
            }
        }
    }
}
