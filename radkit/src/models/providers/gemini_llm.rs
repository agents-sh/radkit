//! Google Gemini LLM provider implementation.
//!
//! API Documentation: <https://ai.google.dev/api/generate-content>
//! Model Names: <https://ai.google.dev/gemini-api/docs/models/gemini>
//! Pricing: <https://ai.google.dev/pricing>
//!
//! # Computer Use
//!
//! This module also provides [`GeminiComputerUseWorker`], a browser-control agent
//! built on Gemini's Computer Use capability. The worker drives an agentic loop:
//! screenshot → model → actions → screenshot → … until the model returns a final
//! text answer.
//!
//! See [`GeminiLlm::computer_use_worker`] and the `gemini_computer_use` example for
//! usage.

use std::sync::Arc;

use base64::Engine as _;
use serde_json::{json, Value};

use crate::errors::{AgentError, AgentResult};
use crate::models::{BaseLlm, Content, ContentPart, LlmResponse, Role, Thread, TokenUsage};
use crate::tools::{BaseToolset, ToolCall};

const GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/";

// ============================================================================
// Computer Use types
// ============================================================================

/// A single UI action requested by the Computer Use model.
///
/// The `name` and `args` fields mirror the `functionCall` payload returned by
/// Gemini. `safety_decision` is populated when the model's internal safety
/// system flags the action as requiring user confirmation before execution.
#[derive(Debug, Clone)]
pub struct ComputerUseAction {
    /// The action name, e.g. `"click_at"`, `"type_text_at"`, `"navigate"`.
    pub name: String,
    /// Arguments for the action as a JSON object.
    pub args: Value,
    /// Present when Gemini's safety system requires explicit confirmation.
    pub safety_decision: Option<SafetyDecision>,
}

/// Safety classification attached to a [`ComputerUseAction`] by Gemini's
/// internal safety system.
#[derive(Debug, Clone)]
pub struct SafetyDecision {
    /// The decision value. Currently `"require_confirmation"` is the only
    /// non-trivial value; anything else (or absence) means the action is
    /// allowed.
    pub decision: String,
    /// Human-readable explanation of why confirmation is required.
    pub explanation: String,
}

impl SafetyDecision {
    /// Returns `true` when the model requires explicit user confirmation before
    /// the action may be executed.
    #[must_use]
    pub fn requires_confirmation(&self) -> bool {
        self.decision == "require_confirmation"
    }
}

/// Outcome reported back to the worker after executing a [`ComputerUseAction`].
///
/// Wrap the result of calling your Playwright / Puppeteer / headless-browser
/// integration in one of these variants and return it from
/// [`ComputerUseHandler::execute`].
#[derive(Debug)]
pub enum ActionOutcome {
    /// The action completed successfully. `url` is the current page URL after
    /// execution (used to populate the `FunctionResponse`).
    Success { url: String },
    /// The action failed. The worker will surface the error message to the model
    /// as part of the `FunctionResponse` so it can attempt recovery.
    Error { message: String },
    /// The user (or your safety policy) declined to execute the action. The
    /// worker will stop the loop immediately and return
    /// [`AgentError::SecurityViolation`].
    Denied { reason: String },
}

/// Trait that bridges the Computer Use worker to your actual browser environment.
///
/// Implement this for your chosen browser automation library (Playwright via
/// [`playwright-rust`](https://crates.io/crates/playwright), `chromiumoxide`,
/// `fantoccini`, a headless-chrome crate, etc.) or for a remote service such as
/// [Browserbase](https://browserbase.com).
///
/// # Safety
///
/// The worker calls [`execute`](ComputerUseHandler::execute) for **every** action
/// the model requests. If [`ComputerUseAction::safety_decision`] indicates
/// [`SafetyDecision::requires_confirmation`], you **must** obtain explicit user
/// consent before proceeding and return [`ActionOutcome::Denied`] if the user
/// refuses. Per the [Gemini API Terms of Service](https://ai.google.dev/terms)
/// you are not allowed to bypass these confirmation requests programmatically.
///
/// # Example
///
/// ```ignore
/// use radkit::models::providers::{
///     ActionOutcome, ComputerUseAction, ComputerUseHandler,
/// };
///
/// struct PlaywrightHandler { /* page handle, screen dimensions, … */ }
///
/// #[async_trait::async_trait]
/// impl ComputerUseHandler for PlaywrightHandler {
///     async fn screenshot(&self) -> Result<Vec<u8>, String> {
///         // return PNG bytes from the current browser page
///         todo!()
///     }
///
///     async fn execute(&self, action: ComputerUseAction) -> ActionOutcome {
///         // check safety_decision, then dispatch on action.name
///         match action.name.as_str() {
///             "click_at" => { /* … */ }
///             "type_text_at" => { /* … */ }
///             _ => {}
///         }
///         ActionOutcome::Success { url: "https://example.com".into() }
///     }
/// }
/// ```
#[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
pub trait ComputerUseHandler: crate::compat::MaybeSend + crate::compat::MaybeSync {
    /// Capture the current state of the browser / screen.
    ///
    /// Returns raw PNG bytes. The worker encodes them as base64 and includes
    /// them in the `FunctionResponse` sent back to the model.
    async fn screenshot(&self) -> Result<Vec<u8>, String>;

    /// Execute a UI action and report the outcome.
    ///
    /// The worker calls this once per action in the model's response. When
    /// the action has a `safety_decision` that [`requires_confirmation`](SafetyDecision::requires_confirmation),
    /// you must ask the user before proceeding and return
    /// [`ActionOutcome::Denied`] if they decline.
    async fn execute(&self, action: ComputerUseAction) -> ActionOutcome;
}

// ============================================================================
// Worker
// ============================================================================

/// An agentic loop that drives Gemini's Computer Use capability.
///
/// Created via [`GeminiLlm::computer_use_worker`]. Call [`run`](Self::run) with
/// a plain-text goal; the worker will:
///
/// 1. Take an initial screenshot via the [`ComputerUseHandler`].
/// 2. Send the goal + screenshot to the Gemini Computer Use model.
/// 3. Receive a list of UI actions (`functionCall`s).
/// 4. Execute each action through the handler.
/// 5. Capture a fresh screenshot and feed it back as a `FunctionResponse`.
/// 6. Repeat until the model returns a text-only response (task complete) or
///    `max_turns` is reached.
///
/// # Models
///
/// The worker defaults to `gemini-2.5-computer-use-preview-10-2025`.
/// Override with [`with_model`](Self::with_model).
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use radkit::models::providers::{GeminiLlm, ComputerUseHandler, ActionOutcome, ComputerUseAction};
///
/// struct MyBrowser;
///
/// #[async_trait::async_trait]
/// impl ComputerUseHandler for MyBrowser {
///     async fn screenshot(&self) -> Result<Vec<u8>, String> { todo!() }
///     async fn execute(&self, action: ComputerUseAction) -> ActionOutcome { todo!() }
/// }
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let llm = Arc::new(GeminiLlm::from_env("gemini-2.5-computer-use-preview-10-2025")?);
///     let worker = llm.computer_use_worker(Arc::new(MyBrowser));
///     let answer = worker.run("Search Google for the Rust programming language").await?;
///     println!("{answer}");
///     Ok(())
/// }
/// ```
pub struct GeminiComputerUseWorker {
    llm: Arc<GeminiLlm>,
    handler: Arc<dyn ComputerUseHandler>,
    model: String,
    max_turns: usize,
}

impl GeminiComputerUseWorker {
    /// Default Computer Use model.
    pub const DEFAULT_MODEL: &'static str = "gemini-2.5-computer-use-preview-10-2025";

    /// Override the model used for Computer Use requests.
    ///
    /// The model must support the `computer_use` tool. Currently only
    /// `gemini-2.5-computer-use-preview-10-2025` and `gemini-3-flash-preview`
    /// are supported by Google.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the maximum number of model turns before the worker gives up.
    ///
    /// Defaults to `20`. Each turn = one model call + one round of action
    /// execution.
    #[must_use]
    pub const fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Run the Computer Use agent loop for the given `goal`.
    ///
    /// Returns the final text answer produced by the model once it has
    /// completed the task.
    ///
    /// # Errors
    ///
    /// - [`AgentError::SecurityViolation`] — an action was denied by safety policy or user refusal
    /// - [`AgentError::LlmAuthentication`] / [`AgentError::LlmRateLimit`] — HTTP auth/rate errors
    /// - [`AgentError::LlmProvider`] — any other Gemini API error or malformed response
    /// - [`AgentError::Network`] — transport-level HTTP failure
    /// - [`AgentError::ResourceExhausted`] — `max_turns` reached without a final text response
    #[allow(clippy::too_many_lines)]
    pub async fn run(&self, goal: &str) -> AgentResult<String> {
        // --- initial screenshot ---
        let initial_screenshot = self
            .handler
            .screenshot()
            .await
            .map_err(|e| AgentError::Internal {
                component: "computer_use_handler".to_string(),
                reason: format!("initial screenshot failed: {e}"),
            })?;

        let encoded = base64::engine::general_purpose::STANDARD.encode(&initial_screenshot);

        // Build the first user turn: goal text + screenshot
        let mut contents: Vec<Value> = vec![json!({
            "role": "user",
            "parts": [
                {"text": goal},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": encoded
                    }
                }
            ]
        })];

        for _turn in 0..self.max_turns {
            // --- call the model ---
            let response_body = self.call_api(&contents).await?;

            // extract the model's content object
            let model_content = response_body
                .get("candidates")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("content"))
                .cloned()
                .ok_or_else(|| AgentError::LlmProvider {
                    provider: "Gemini".to_string(),
                    message: "missing candidates[0].content in Computer Use response".to_string(),
                })?;

            // append model turn to history
            contents.push(model_content.clone());

            // parse actions from parts
            let actions = Self::parse_actions(&model_content)?;

            // if no function calls → model is done, extract text answer
            if actions.is_empty() {
                let answer = model_content
                    .get("parts")
                    .and_then(|p| p.as_array())
                    .map(|parts| {
                        parts
                            .iter()
                            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                            .collect::<Vec<_>>()
                            .join("")
                    })
                    .unwrap_or_default();
                return Ok(answer);
            }

            // --- execute actions and collect function responses ---
            let mut function_response_parts: Vec<Value> = Vec::new();

            for action in actions {
                // execute through the handler
                let outcome = self.handler.execute(action.clone()).await;

                match outcome {
                    ActionOutcome::Denied { reason } => {
                        return Err(AgentError::SecurityViolation {
                            policy: "computer_use".to_string(),
                            reason,
                        });
                    }
                    ActionOutcome::Error { message } => {
                        // Take a screenshot of the error state so the model can
                        // see what happened and attempt recovery.
                        let screenshot_bytes =
                            self.handler.screenshot().await.unwrap_or_default();
                        let screenshot_b64 = base64::engine::general_purpose::STANDARD
                            .encode(&screenshot_bytes);

                        function_response_parts.push(json!({
                            "functionResponse": {
                                "name": action.name,
                                "response": {
                                    "error": message
                                }
                            },
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": screenshot_b64
                            }
                        }));
                    }
                    ActionOutcome::Success { url } => {
                        // Capture fresh screenshot after successful action
                        let screenshot_bytes =
                            self.handler.screenshot().await.map_err(|e| {
                                AgentError::Internal {
                                    component: "computer_use_handler".to_string(),
                                    reason: format!("screenshot after action failed: {e}"),
                                }
                            })?;
                        let screenshot_b64 = base64::engine::general_purpose::STANDARD
                            .encode(&screenshot_bytes);

                        // Build safety acknowledgement if it was required
                        let mut response_payload = json!({ "url": url });
                        if let Some(ref sd) = action.safety_decision {
                            if sd.requires_confirmation() {
                                response_payload["safety_acknowledgement"] =
                                    json!("true");
                            }
                        }

                        function_response_parts.push(json!({
                            "functionResponse": {
                                "name": action.name,
                                "response": response_payload,
                                "parts": [{
                                    "inline_data": {
                                        "mime_type": "image/png",
                                        "data": screenshot_b64
                                    }
                                }]
                            }
                        }));
                    }
                }
            }

            // append user turn with all function responses
            contents.push(json!({
                "role": "user",
                "parts": function_response_parts
            }));
        }

        Err(AgentError::ResourceExhausted {
            resource: "computer_use_turns".to_string(),
            reason: format!(
                "reached max_turns ({}) without a final text response",
                self.max_turns
            ),
        })
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// POST to the Gemini generateContent endpoint with the Computer Use tool.
    async fn call_api(&self, contents: &[Value]) -> AgentResult<Value> {
        let url = format!(
            "{}models/{}:generateContent",
            self.llm.base_url, self.model
        );

        let payload = json!({
            "contents": contents,
            "tools": [{
                "computer_use": {
                    "environment": "ENVIRONMENT_BROWSER"
                }
            }]
        });

        let client = reqwest::Client::new();
        let response = client
            .post(&url)
            .header("x-goog-api-key", &self.llm.api_key)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());

            return Err(match status.as_u16() {
                401 | 403 => AgentError::LlmAuthentication {
                    provider: "Gemini".to_string(),
                },
                429 => AgentError::LlmRateLimit {
                    provider: "Gemini".to_string(),
                },
                _ => AgentError::LlmProvider {
                    provider: "Gemini".to_string(),
                    message: format!("HTTP {status}: {error_body}"),
                },
            });
        }

        let body: Value = response.json().await?;

        if let Some(error) = body.get("error") {
            return Err(AgentError::LlmProvider {
                provider: "Gemini".to_string(),
                message: format!("API error: {error}"),
            });
        }

        Ok(body)
    }

    /// Extract [`ComputerUseAction`]s from a model content object.
    fn parse_actions(model_content: &Value) -> AgentResult<Vec<ComputerUseAction>> {
        let Some(parts) = model_content
            .get("parts")
            .and_then(|p| p.as_array())
        else {
            return Ok(vec![]);
        };

        let mut actions = Vec::new();

        for part in parts {
            let Some(fc) = part.get("functionCall") else {
                continue;
            };

            let name = fc
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| AgentError::LlmProvider {
                    provider: "Gemini".to_string(),
                    message: "missing 'name' in Computer Use functionCall".to_string(),
                })?
                .to_string();

            let mut args = fc.get("args").cloned().unwrap_or_else(|| Value::Object(serde_json::Map::default()));

            // Extract and remove safety_decision from args before forwarding
            let safety_decision = args
                .as_object_mut()
                .and_then(|map| map.remove("safety_decision"))
                .and_then(|sd| {
                    let decision = sd
                        .get("decision")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let explanation = sd
                        .get("explanation")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if decision.is_empty() {
                        None
                    } else {
                        Some(SafetyDecision { decision, explanation })
                    }
                });

            actions.push(ComputerUseAction {
                name,
                args,
                safety_decision,
            });
        }

        Ok(actions)
    }
}

// ============================================================================
// GeminiLlm
// ============================================================================

/// Google Gemini LLM implementation.
///
/// Provides access to Gemini models through the Google AI API.
/// Supports text generation, multi-modal inputs (images, documents), and tool use.
///
/// # Authentication
///
/// The API key can be provided explicitly or loaded from the `GEMINI_API_KEY`
/// environment variable via [`from_env`](GeminiLlm::from_env).
///
/// # Model Selection
///
/// Common model names:
/// - `gemini-2.5-flash` - Gemini 2.5 Flash
/// - `gemini-2.5-pro` - Gemini 2.5 Pro
/// - `gemini-2.0-flash-exp` - Gemini 2.0 Flash Experimental
/// - `gemini-1.5-pro` - Gemini 1.5 Pro
/// - `gemini-1.5-flash` - Gemini 1.5 Flash
///
/// # Computer Use
///
/// Call [`computer_use_worker`](GeminiLlm::computer_use_worker) to build a
/// browser-control agent powered by Gemini's Computer Use capability.
/// The worker handles the full agentic loop: screenshot → model → actions →
/// screenshot → … → final answer.
///
/// Supported Computer Use models:
/// - `gemini-2.5-computer-use-preview-10-2025`
/// - `gemini-3-flash-preview`
///
/// # Examples
///
/// ## Text generation
///
/// ```ignore
/// use radkit::models::providers::GeminiLlm;
/// use radkit::models::{BaseLlm, Thread};
///
/// // From environment variable
/// let llm = GeminiLlm::from_env("gemini-2.5-flash")?;
///
/// // With explicit API key
/// let llm = GeminiLlm::new("gemini-2.5-flash", "api-key");
///
/// // Generate content
/// let thread = Thread::from_user("Explain quantum computing");
/// let response = llm.generate_content(thread, None).await?;
/// println!("{}", response.content().first_text().unwrap_or("No response"));
/// ```
///
/// ## Computer Use
///
/// ```ignore
/// use std::sync::Arc;
/// use radkit::models::providers::{
///     GeminiLlm, ComputerUseHandler, ComputerUseAction, ActionOutcome,
/// };
///
/// struct MyBrowser;
///
/// #[async_trait::async_trait]
/// impl ComputerUseHandler for MyBrowser {
///     async fn screenshot(&self) -> Result<Vec<u8>, String> { todo!() }
///     async fn execute(&self, action: ComputerUseAction) -> ActionOutcome { todo!() }
/// }
///
/// let llm = Arc::new(GeminiLlm::from_env("gemini-2.5-computer-use-preview-10-2025")?);
/// let worker = llm.computer_use_worker(Arc::new(MyBrowser));
/// let answer = worker.run("Search for the Rust programming language on Google").await?;
/// println!("{answer}");
/// ```
pub struct GeminiLlm {
    model_name: String,
    api_key: String,
    base_url: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
}

impl GeminiLlm {
    /// Environment variable name for the Gemini API key.
    pub const API_KEY_ENV: &str = "GEMINI_API_KEY";

    /// Creates a new Gemini LLM instance with explicit API key.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The model to use (e.g., "gemini-2.5-flash")
    /// * `api_key` - Google AI API key
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let llm = GeminiLlm::new("gemini-2.5-flash", "api-key");
    /// ```
    pub fn new(model_name: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            api_key: api_key.into(),
            base_url: GEMINI_BASE_URL.to_string(),
            max_tokens: None,
            temperature: None,
        }
    }

    /// Creates a new Gemini LLM instance loading API key from environment.
    ///
    /// Reads the API key from the `GEMINI_API_KEY` environment variable.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The model to use (e.g., "gemini-2.5-flash")
    ///
    /// # Errors
    ///
    /// Returns an error if the environment variable is not set or is empty.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let llm = GeminiLlm::from_env("gemini-2.5-flash")?;
    /// ```
    pub fn from_env(model_name: impl Into<String>) -> AgentResult<Self> {
        let api_key =
            std::env::var(Self::API_KEY_ENV).map_err(|_| AgentError::MissingConfiguration {
                field: Self::API_KEY_ENV.to_string(),
            })?;

        if api_key.is_empty() {
            return Err(AgentError::InvalidConfiguration {
                field: Self::API_KEY_ENV.to_string(),
                reason: "API key cannot be empty".to_string(),
            });
        }

        Ok(Self::new(model_name, api_key))
    }

    /// Sets a custom base URL for the API endpoint.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Sets the maximum number of tokens to generate.
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the temperature for generation (0.0 to 2.0).
    ///
    /// Higher values produce more random outputs.
    #[must_use]
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Creates a [`GeminiComputerUseWorker`] that drives Gemini's Computer Use
    /// capability using `self` for API authentication.
    ///
    /// The `llm` must be wrapped in an [`Arc`] so the worker can share it
    /// without cloning the API key.
    ///
    /// The worker defaults to the
    /// `gemini-2.5-computer-use-preview-10-2025` model. Override it with
    /// [`GeminiComputerUseWorker::with_model`].
    ///
    /// # Arguments
    ///
    /// * `handler` - Your implementation of [`ComputerUseHandler`] that
    ///   controls the target browser environment.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::sync::Arc;
    /// use radkit::models::providers::{GeminiLlm, ComputerUseHandler, ComputerUseAction, ActionOutcome};
    ///
    /// struct MyBrowser;
    ///
    /// #[async_trait::async_trait]
    /// impl ComputerUseHandler for MyBrowser {
    ///     async fn screenshot(&self) -> Result<Vec<u8>, String> { todo!() }
    ///     async fn execute(&self, action: ComputerUseAction) -> ActionOutcome { todo!() }
    /// }
    ///
    /// let llm = Arc::new(GeminiLlm::from_env("gemini-2.5-computer-use-preview-10-2025")?);
    /// let worker = llm.computer_use_worker(Arc::new(MyBrowser));
    /// let answer = worker.run("Find the cheapest laptop on Amazon under $500").await?;
    /// ```
    pub fn computer_use_worker(
        self: Arc<Self>,
        handler: Arc<dyn ComputerUseHandler>,
    ) -> GeminiComputerUseWorker {
        GeminiComputerUseWorker {
            model: GeminiComputerUseWorker::DEFAULT_MODEL.to_string(),
            llm: self,
            handler,
            max_turns: 20,
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers (generate_content path)
    // -----------------------------------------------------------------------

    /// Converts a Thread into Gemini API request format.
    async fn build_request_payload(
        &self,
        thread: Thread,
        toolset: Option<Arc<dyn BaseToolset>>,
    ) -> AgentResult<Value> {
        let (system_prompt, events) = thread.into_parts();

        // Build contents array (Gemini's message format)
        let mut contents = Vec::new();
        let mut system_parts = Vec::new();

        // Add system prompt if present
        if let Some(system) = system_prompt {
            system_parts.push(system);
        }

        Self::build_contents_from_events(events, &mut contents, &mut system_parts);

        // Build the base payload
        let mut payload = json!({
            "contents": contents
        });

        // Add systemInstruction if we have system messages
        if !system_parts.is_empty() {
            let system_text = system_parts.join("\n");
            payload["systemInstruction"] = json!({
                "parts": [{"text": system_text}]
            });
        }

        // Add generationConfig
        Self::add_generation_config(&mut payload, self.temperature, self.max_tokens);

        // Add tools if provided
        Self::add_tools_to_payload(&mut payload, toolset).await;

        Ok(payload)
    }

    /// Builds Gemini-formatted contents from thread events.
    fn build_contents_from_events(
        events: Vec<crate::models::Event>,
        contents: &mut Vec<Value>,
        system_parts: &mut Vec<String>,
    ) {
        for event in events {
            let role = *event.role();
            let content = event.into_content();

            match role {
                Role::System => {
                    Self::process_system_message(&content, system_parts);
                }
                Role::User => {
                    Self::process_user_message(content, contents);
                }
                Role::Assistant => {
                    Self::process_assistant_message(content, contents);
                }
                Role::Tool => {
                    Self::process_tool_message(content, contents);
                }
            }
        }
    }

    /// Processes system messages.
    fn process_system_message(content: &Content, system_parts: &mut Vec<String>) {
        if let Some(text) = content.joined_texts() {
            system_parts.push(text);
        }
    }

    /// Processes user messages.
    fn process_user_message(content: Content, contents: &mut Vec<Value>) {
        let mut parts = Vec::new();

        for part in content {
            match part {
                ContentPart::Text(text) => {
                    parts.push(json!({"text": text}));
                }
                ContentPart::Data(data) => {
                    let part = match data.source {
                        crate::models::DataSource::Base64(b64) => {
                            json!({
                                "inline_data": {
                                    "mime_type": data.content_type,
                                    "data": b64
                                }
                            })
                        }
                        crate::models::DataSource::Uri(uri) => {
                            json!({
                                "fileData": {
                                    "mime_type": data.content_type,
                                    "fileUri": uri
                                }
                            })
                        }
                    };
                    parts.push(part);
                }
                ContentPart::ToolCall(tool_call) => {
                    parts.push(json!({
                        "functionCall": {
                            "name": tool_call.name(),
                            "args": tool_call.arguments()
                        }
                    }));
                }
                ContentPart::ToolResponse(tool_response) => {
                    let result = tool_response.result();
                    let response_content = if result.is_success() {
                        result.data().clone()
                    } else {
                        json!({
                            "error": result.error_message().unwrap_or("Unknown error")
                        })
                    };

                    parts.push(json!({
                        "functionResponse": {
                            "name": tool_response.tool_call_id(),
                            "response": {
                                "name": tool_response.tool_call_id(),
                                "content": response_content
                            }
                        }
                    }));
                }
            }
        }

        if !parts.is_empty() {
            contents.push(json!({
                "role": "user",
                "parts": parts
            }));
        }
    }

    /// Processes assistant messages.
    fn process_assistant_message(content: Content, contents: &mut Vec<Value>) {
        let mut parts = Vec::new();

        for part in content {
            match part {
                ContentPart::Text(text) => {
                    parts.push(json!({"text": text}));
                }
                ContentPart::ToolCall(tool_call) => {
                    // If Gemini gave us a raw part (with thought_signature etc.),
                    // echo it back verbatim so the model's signature is preserved.
                    if let Some(raw) = tool_call.provider_metadata() {
                        parts.push(raw.clone());
                    } else {
                        parts.push(json!({
                            "functionCall": {
                                "name": tool_call.name(),
                                "args": tool_call.arguments()
                            }
                        }));
                    }
                }
                _ => {} // Gemini doesn't support other types in assistant messages
            }
        }

        if !parts.is_empty() {
            contents.push(json!({
                "role": "model",
                "parts": parts
            }));
        }
    }

    /// Processes tool messages.
    fn process_tool_message(content: Content, contents: &mut Vec<Value>) {
        let mut parts = Vec::new();

        for part in content {
            match part {
                ContentPart::ToolResponse(tool_response) => {
                    let result = tool_response.result();
                    let response_content = if result.is_success() {
                        result.data().clone()
                    } else {
                        json!({
                            "error": result.error_message().unwrap_or("Unknown error")
                        })
                    };

                    parts.push(json!({
                        "functionResponse": {
                            "name": tool_response.tool_call_id(),
                            "response": {
                                "name": tool_response.tool_call_id(),
                                "content": response_content
                            }
                        }
                    }));
                }
                ContentPart::ToolCall(tool_call) => {
                    // Echo raw part verbatim to preserve thought_signature if present.
                    if let Some(raw) = tool_call.provider_metadata() {
                        parts.push(raw.clone());
                    } else {
                        parts.push(json!({
                            "functionCall": {
                                "name": tool_call.name(),
                                "args": tool_call.arguments()
                            }
                        }));
                    }
                }
                _ => {}
            }
        }

        if !parts.is_empty() {
            contents.push(json!({
                "role": "user",
                "parts": parts
            }));
        }
    }

    /// Adds generation configuration to payload.
    fn add_generation_config(
        payload: &mut Value,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) {
        let mut generation_config = json!({});

        if let Some(temperature) = temperature {
            generation_config["temperature"] = json!(temperature);
        }

        if let Some(max_tokens) = max_tokens {
            generation_config["maxOutputTokens"] = json!(max_tokens);
        }

        if !generation_config.as_object().unwrap().is_empty() {
            payload["generationConfig"] = generation_config;
        }
    }

    /// Adds tool configuration to the payload if tools are available.
    async fn add_tools_to_payload(payload: &mut Value, toolset: Option<Arc<dyn BaseToolset>>) {
        if let Some(toolset) = toolset {
            let tools_list = toolset.get_tools().await;
            if !tools_list.is_empty() {
                let function_declarations: Vec<Value> = tools_list
                    .iter()
                    .map(|tool| {
                        let decl = tool.declaration();
                        json!({
                            "name": decl.name(),
                            "description": decl.description(),
                            "parameters": decl.parameters()
                        })
                    })
                    .collect();

                payload["tools"] = json!([{
                    "function_declarations": function_declarations
                }]);
            }
        }
    }

    /// Parses Gemini API response into Content.
    fn parse_response(response_body: &Value) -> AgentResult<Content> {
        let mut content = Content::default();

        // Extract first candidate
        let candidates = response_body
            .get("candidates")
            .and_then(|v| v.as_array())
            .ok_or_else(|| AgentError::LlmProvider {
                provider: "Gemini".to_string(),
                message: "Missing or invalid 'candidates' field in response".to_string(),
            })?;

        let first_candidate = candidates.first().ok_or_else(|| AgentError::LlmProvider {
            provider: "Gemini".to_string(),
            message: "Empty candidates array in response".to_string(),
        })?;

        // Extract parts from the candidate
        let parts = first_candidate
            .get("content")
            .and_then(|v| v.get("parts"))
            .and_then(|v| v.as_array())
            .ok_or_else(|| AgentError::LlmProvider {
                provider: "Gemini".to_string(),
                message: "Missing or invalid 'content.parts' in candidate".to_string(),
            })?;

        for part in parts {
            // Check for text content
            if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                if !text.trim().is_empty() {
                    content.push(ContentPart::Text(text.to_string()));
                }
            }

            // Check for function call
            if let Some(function_call) = part.get("functionCall") {
                let name = function_call
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::LlmProvider {
                        provider: "Gemini".to_string(),
                        message: "Missing 'name' in functionCall".to_string(),
                    })?;

                let args = function_call.get("args").cloned().unwrap_or(Value::Null);

                // Gemini doesn't provide call_id, so use name as id.
                // Preserve the entire raw part as provider_metadata so that
                // thought_signature (and any other fields Gemini may add) are
                // echoed back verbatim in subsequent turns.
                let tool_call =
                    ToolCall::new(name, name, args).with_provider_metadata(part.clone());

                content.push(ContentPart::ToolCall(tool_call));
            }
        }

        Ok(content)
    }

    /// Parses token usage from Gemini API response.
    ///
    /// **Important**: Gemini's `candidatesTokenCount` does NOT include `thoughtsTokenCount`.
    /// We must add them together to get total completion tokens (OpenAI-style normalization).
    fn parse_usage(response_body: &Value) -> TokenUsage {
        let Some(usage_obj) = response_body.get("usageMetadata") else {
            return TokenUsage::empty();
        };

        let prompt_tokens = usage_obj
            .get("promptTokenCount")
            .and_then(serde_json::Value::as_u64)
            .map(|v| u32::try_from(v).unwrap_or(u32::MAX));

        // Extract candidates tokens and thoughts tokens separately
        let candidates_tokens = usage_obj
            .get("candidatesTokenCount")
            .and_then(serde_json::Value::as_u64)
            .map(|v| u32::try_from(v).unwrap_or(u32::MAX));

        let thoughts_tokens = usage_obj
            .get("thoughtsTokenCount")
            .and_then(serde_json::Value::as_u64)
            .map(|v| u32::try_from(v).unwrap_or(u32::MAX));

        // IMPORTANT: For Gemini, thoughtsTokenCount is NOT included in candidatesTokenCount
        // We must add them together to normalize to OpenAI-style completion_tokens
        let completion_tokens = match (candidates_tokens, thoughts_tokens) {
            (Some(c), Some(t)) => Some(c + t),
            (Some(c), None) => Some(c),
            (None, Some(t)) => Some(t),
            (None, None) => None,
        };

        let total_tokens = usage_obj
            .get("totalTokenCount")
            .and_then(serde_json::Value::as_u64)
            .map(|v| u32::try_from(v).unwrap_or(u32::MAX));

        TokenUsage::partial(prompt_tokens, completion_tokens, total_tokens)
    }
}

#[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
impl BaseLlm for GeminiLlm {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    async fn generate_content(
        &self,
        thread: Thread,
        toolset: Option<Arc<dyn BaseToolset>>,
    ) -> AgentResult<LlmResponse> {
        // Build request payload
        let payload = self.build_request_payload(thread, toolset).await?;

        // Build URL with model name
        let url = format!(
            "{}models/{}:generateContent",
            self.base_url, self.model_name
        );

        // Create HTTP client
        let client = reqwest::Client::new();

        // Make request - Gemini uses x-goog-api-key header
        let response = client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        // Check for HTTP errors
        if !response.status().is_success() {
            let status = response.status();
            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            return Err(match status.as_u16() {
                401 | 403 => AgentError::LlmAuthentication {
                    provider: "Gemini".to_string(),
                },
                429 => AgentError::LlmRateLimit {
                    provider: "Gemini".to_string(),
                },
                _ => AgentError::LlmProvider {
                    provider: "Gemini".to_string(),
                    message: format!("HTTP {status}: {error_body}"),
                },
            });
        }

        // Parse response
        let response_body: Value = response.json().await?;

        // Check for error in response body
        if let Some(error) = response_body.get("error") {
            return Err(AgentError::LlmProvider {
                provider: "Gemini".to_string(),
                message: format!("API error: {error}"),
            });
        }

        // Parse content and usage
        let content = Self::parse_response(&response_body)?;
        let usage = Self::parse_usage(&response_body);

        Ok(LlmResponse::new(content, usage))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Event;
    use crate::tools::BaseTool;

    struct TestTool;

    #[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
    #[cfg_attr(
        not(all(target_os = "wasi", target_env = "p1")),
        async_trait::async_trait
    )]
    impl BaseTool for TestTool {
        fn name(&self) -> &'static str {
            "gemini_tool"
        }

        fn description(&self) -> &'static str {
            "Test tool"
        }

        fn declaration(&self) -> crate::tools::FunctionDeclaration {
            crate::tools::FunctionDeclaration::new(
                "gemini_tool",
                "Test tool",
                serde_json::json!({"type": "object"}),
            )
        }

        async fn run_async(
            &self,
            _args: std::collections::HashMap<String, Value>,
            _context: &crate::tools::ToolContext<'_>,
        ) -> crate::tools::ToolResult {
            crate::tools::ToolResult::success(serde_json::json!({}))
        }
    }

    struct SimpleToolset(Vec<Box<dyn BaseTool>>);

    #[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
    #[cfg_attr(
        not(all(target_os = "wasi", target_env = "p1")),
        async_trait::async_trait
    )]
    impl BaseToolset for SimpleToolset {
        async fn get_tools(&self) -> Vec<&dyn BaseTool> {
            self.0.iter().map(std::convert::AsRef::as_ref).collect()
        }

        async fn close(&self) {}
    }

    #[tokio::test(flavor = "current_thread")]
    async fn build_request_payload_includes_system_instruction() {
        let llm = GeminiLlm::new("gemini-2.5-flash", "api-key")
            .with_temperature(0.8)
            .with_max_tokens(1024);

        let thread = Thread::from_system("You are helpful")
            .add_event(Event::user("Hello"))
            .add_event(Event::assistant("Working"));

        let payload = llm
            .build_request_payload(
                thread,
                Some(Arc::new(SimpleToolset(vec![
                    Box::new(TestTool) as Box<dyn BaseTool>
                ]))),
            )
            .await
            .expect("payload");

        assert_eq!(
            payload["systemInstruction"]["parts"][0]["text"],
            json!("You are helpful")
        );

        let generation = payload["generationConfig"].as_object().expect("gen config");
        let temperature = generation
            .get("temperature")
            .and_then(serde_json::Value::as_f64)
            .expect("temperature");
        assert!((temperature - 0.8).abs() < 1e-6);
        assert_eq!(generation.get("maxOutputTokens"), Some(&json!(1024)));

        let tools = payload["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(
            tools[0]["function_declarations"][0]["name"],
            json!("gemini_tool")
        );
    }

    #[test]
    fn parse_response_extracts_text_and_tool_calls() {
        let _llm = GeminiLlm::new("gemini-2.5-flash", "api-key");
        let body = json!({
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "Hello user"},
                            {"functionCall": {"name": "lookup", "args": {"key": "value"}}}
                        ]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 4,
                "thoughtsTokenCount": 2,
                "totalTokenCount": 14
            }
        });

        let content = GeminiLlm::parse_response(&body).expect("content");
        assert_eq!(content.first_text(), Some("Hello user"));
        let calls = content.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name(), "lookup");

        let usage = GeminiLlm::parse_usage(&body);
        assert_eq!(usage.input_tokens_opt(), Some(8));
        assert_eq!(usage.output_tokens_opt(), Some(6)); // 4 + 2
        assert_eq!(usage.total_tokens_opt(), Some(14));
    }

    #[test]
    fn parse_response_missing_candidates_errors() {
        let _llm = GeminiLlm::new("gemini-2.5-flash", "api-key");
        let body = json!({});
        let err = GeminiLlm::parse_response(&body).expect_err("expected error");
        match err {
            AgentError::LlmProvider { provider, .. } => assert_eq!(provider, "Gemini"),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn from_env_validates_presence() {
        let original = std::env::var(GeminiLlm::API_KEY_ENV).ok();
        std::env::remove_var(GeminiLlm::API_KEY_ENV);

        let missing = GeminiLlm::from_env("model");
        assert!(matches!(
            missing,
            Err(AgentError::MissingConfiguration { .. })
        ));

        std::env::set_var(GeminiLlm::API_KEY_ENV, "");
        let empty = GeminiLlm::from_env("model");
        assert!(matches!(
            empty,
            Err(AgentError::InvalidConfiguration { .. })
        ));

        match original {
            Some(value) => std::env::set_var(GeminiLlm::API_KEY_ENV, value),
            None => std::env::remove_var(GeminiLlm::API_KEY_ENV),
        }
    }

    // -----------------------------------------------------------------------
    // Computer Use tests
    // -----------------------------------------------------------------------

    #[test]
    fn parse_actions_extracts_function_calls() {
        let model_content = json!({
            "role": "model",
            "parts": [
                {"text": "I will click the search bar."},
                {
                    "functionCall": {
                        "name": "click_at",
                        "args": {"x": 500, "y": 300}
                    }
                }
            ]
        });

        let actions = GeminiComputerUseWorker::parse_actions(&model_content).expect("actions");
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].name, "click_at");
        assert_eq!(actions[0].args["x"], json!(500));
        assert!(actions[0].safety_decision.is_none());
    }

    #[test]
    fn parse_actions_extracts_safety_decision() {
        let model_content = json!({
            "role": "model",
            "parts": [{
                "functionCall": {
                    "name": "click_at",
                    "args": {
                        "x": 60,
                        "y": 100,
                        "safety_decision": {
                            "decision": "require_confirmation",
                            "explanation": "This clicks an I'm-not-a-robot checkbox."
                        }
                    }
                }
            }]
        });

        let actions = GeminiComputerUseWorker::parse_actions(&model_content).expect("actions");
        assert_eq!(actions.len(), 1);
        // safety_decision must be removed from args
        assert!(actions[0].args.get("safety_decision").is_none());
        let sd = actions[0].safety_decision.as_ref().expect("safety_decision");
        assert!(sd.requires_confirmation());
        assert!(!sd.explanation.is_empty());
    }

    #[test]
    fn parse_actions_empty_when_no_function_calls() {
        let model_content = json!({
            "role": "model",
            "parts": [{"text": "The task is complete."}]
        });

        let actions = GeminiComputerUseWorker::parse_actions(&model_content).expect("actions");
        assert!(actions.is_empty());
    }

    #[test]
    fn safety_decision_requires_confirmation() {
        let sd = SafetyDecision {
            decision: "require_confirmation".to_string(),
            explanation: "Clicking accept terms".to_string(),
        };
        assert!(sd.requires_confirmation());

        let sd_allowed = SafetyDecision {
            decision: "regular".to_string(),
            explanation: String::new(),
        };
        assert!(!sd_allowed.requires_confirmation());
    }

    // Shared no-op handler used by the two worker-builder tests below.
    struct NoopHandler;

    #[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
    #[cfg_attr(not(all(target_os = "wasi", target_env = "p1")), async_trait::async_trait)]
    impl ComputerUseHandler for NoopHandler {
        async fn screenshot(&self) -> Result<Vec<u8>, String> {
            Ok(vec![])
        }
        async fn execute(&self, _: ComputerUseAction) -> ActionOutcome {
            ActionOutcome::Success {
                url: String::new(),
            }
        }
    }

    #[test]
    fn computer_use_worker_default_model() {
        let llm = Arc::new(GeminiLlm::new("gemini-2.5-flash", "key"));
        let worker = llm.computer_use_worker(Arc::new(NoopHandler));
        assert_eq!(worker.model, GeminiComputerUseWorker::DEFAULT_MODEL);
        assert_eq!(worker.max_turns, 20);
    }

    #[test]
    fn computer_use_worker_builder_overrides() {
        let llm = Arc::new(GeminiLlm::new("gemini-2.5-flash", "key"));
        let worker = llm
            .computer_use_worker(Arc::new(NoopHandler))
            .with_model("gemini-3-flash-preview")
            .with_max_turns(5);

        assert_eq!(worker.model, "gemini-3-flash-preview");
        assert_eq!(worker.max_turns, 5);
    }
}
