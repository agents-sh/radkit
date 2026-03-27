//! `DeepSeek` LLM provider implementation.
//!
//! API Documentation: <https://api-docs.deepseek.com/>
//! Model Names: <https://platform.deepseek.com/>

use std::sync::Arc;

use serde_json::{json, Value};

use crate::errors::{AgentError, AgentResult};
use crate::models::{BaseLlm, Content, ContentPart, LlmResponse, Role, Thread, TokenUsage};
use crate::tools::{BaseToolset, ToolCall};

const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com/v1/chat/completions";

/// `DeepSeek` LLM implementation.
///
/// Provides access to `DeepSeek` models through the `DeepSeek` API.
/// The API is OpenAI-compatible.
///
/// # Authentication
///
/// The API key can be provided explicitly or loaded from the `DEEPSEEK_API_KEY`
/// environment variable via [`from_env`](DeepSeekLlm::from_env).
///
/// # Model Selection
///
/// Common model names:
/// - `deepseek-chat` - `DeepSeek` Chat
/// - `deepseek-reasoner` - `DeepSeek` Reasoner (with reasoning capabilities)
///
/// # Examples
///
/// ```ignore
/// use radkit::models::providers::DeepSeekLlm;
/// use radkit::models::{BaseLlm, Thread};
///
/// // From environment variable
/// let llm = DeepSeekLlm::from_env("deepseek-chat")?;
///
/// // With explicit API key
/// let llm = DeepSeekLlm::new("deepseek-chat", "sk-...");
///
/// // Generate content
/// let thread = Thread::from_user("Explain quantum computing");
/// let response = llm.generate_content(thread, None).await?;
/// println!("{}", response.content().first_text().unwrap_or("No response"));
/// ```
pub struct DeepSeekLlm {
    model_name: String,
    api_key: String,
    base_url: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
}

impl DeepSeekLlm {
    /// Environment variable name for the `DeepSeek` API key.
    pub const API_KEY_ENV: &str = "DEEPSEEK_API_KEY";

    /// Creates a new `DeepSeek` LLM instance with explicit API key.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The model to use (e.g., "deepseek-chat", "deepseek-reasoner")
    /// * `api_key` - `DeepSeek` API key
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let llm = DeepSeekLlm::new("deepseek-chat", "sk-...");
    /// ```
    pub fn new(model_name: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            api_key: api_key.into(),
            base_url: DEEPSEEK_BASE_URL.to_string(),
            max_tokens: None,
            temperature: None,
        }
    }

    /// Creates a new `DeepSeek` LLM instance loading API key from environment.
    ///
    /// Reads the API key from the `DEEPSEEK_API_KEY` environment variable.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The model to use (e.g., "deepseek-chat")
    ///
    /// # Errors
    ///
    /// Returns an error if the environment variable is not set or is empty.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let llm = DeepSeekLlm::from_env("deepseek-chat")?;
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

    /// Converts a Thread into `DeepSeek` API request format (OpenAI-compatible).
    async fn build_request_payload(
        &self,
        thread: Thread,
        toolset: Option<Arc<dyn BaseToolset>>,
    ) -> AgentResult<Value> {
        let (system_prompt, events) = thread.into_parts();

        // Build messages array (OpenAI format)
        let mut messages = Vec::new();

        // Add system prompt as first message if present
        if let Some(system) = system_prompt {
            messages.push(json!({
                "role": "system",
                "content": system
            }));
        }

        Self::build_messages_from_events(events, &mut messages);

        // Build the base payload
        let mut payload = json!({
            "model": self.model_name,
            "messages": messages
        });

        // Add optional parameters
        if let Some(temperature) = self.temperature {
            payload["temperature"] = json!(temperature);
        }

        if let Some(max_tokens) = self.max_tokens {
            payload["max_tokens"] = json!(max_tokens);
        }

        // Add tools if provided
        Self::add_tools_to_payload(&mut payload, toolset).await;

        Ok(payload)
    }

    /// Builds OpenAI-formatted messages from thread events.
    fn build_messages_from_events(events: Vec<crate::models::Event>, messages: &mut Vec<Value>) {
        for event in events {
            let role = *event.role();
            let role_str = match &role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
                Role::System => "system",
            };

            let content = event.into_content();

            match role {
                Role::System | Role::User => {
                    Self::process_user_or_system_message(role_str, content, messages);
                }
                Role::Assistant => {
                    Self::process_assistant_message(content, messages);
                }
                Role::Tool => {
                    Self::process_tool_message(content, messages);
                }
            }
        }
    }

    /// Processes user or system messages.
    fn process_user_or_system_message(role_str: &str, content: Content, messages: &mut Vec<Value>) {
        if content.is_text_only() && !content.is_text_empty() {
            let text = content.joined_texts().unwrap_or_default();
            messages.push(json!({
                "role": role_str,
                "content": text
            }));
        } else {
            let mut content_parts = Vec::new();

            for part in content {
                match part {
                    ContentPart::Text(text) => {
                        content_parts.push(json!({
                            "type": "text",
                            "text": text
                        }));
                    }
                    ContentPart::Data(data) => {
                        if data.content_type.starts_with("image/") {
                            let image_url = match data.source {
                                crate::models::DataSource::Base64(b64) => {
                                    format!("data:{};base64,{}", data.content_type, b64)
                                }
                                crate::models::DataSource::Uri(uri) => uri,
                            };
                            content_parts.push(json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }));
                        }
                    }
                    ContentPart::ToolCall(_) | ContentPart::ToolResponse(_) => {}
                }
            }

            if !content_parts.is_empty() {
                messages.push(json!({
                    "role": role_str,
                    "content": content_parts
                }));
            }
        }
    }

    /// Processes assistant messages with optional tool calls.
    fn process_assistant_message(content: Content, messages: &mut Vec<Value>) {
        let mut texts = Vec::new();
        let mut tool_calls = Vec::new();

        for part in content {
            match part {
                ContentPart::Text(text) => texts.push(text),
                ContentPart::ToolCall(tool_call) => {
                    tool_calls.push(json!({
                        "type": "function",
                        "id": tool_call.id(),
                        "function": {
                            "name": tool_call.name(),
                            "arguments": tool_call.arguments().to_string()
                        }
                    }));
                }
                _ => {}
            }
        }

        let content_text = texts.join("\n\n");
        let mut message = json!({
            "role": "assistant",
            "content": content_text
        });

        if !tool_calls.is_empty() {
            message["tool_calls"] = json!(tool_calls);
        }

        messages.push(message);
    }

    /// Processes tool response messages.
    fn process_tool_message(content: Content, messages: &mut Vec<Value>) {
        for part in content {
            if let ContentPart::ToolResponse(tool_response) = part {
                let result = tool_response.result();
                let content_value = if result.is_success() {
                    result.data().to_string()
                } else {
                    json!({
                        "error": result.error_message().unwrap_or("Unknown error")
                    })
                    .to_string()
                };

                messages.push(json!({
                    "role": "tool",
                    "content": content_value,
                    "tool_call_id": tool_response.tool_call_id()
                }));
            }
        }
    }

    /// Adds tool configuration to the payload if tools are available.
    async fn add_tools_to_payload(payload: &mut Value, toolset: Option<Arc<dyn BaseToolset>>) {
        if let Some(toolset) = toolset {
            let tools_list = toolset.get_tools().await;
            if !tools_list.is_empty() {
                let tools: Vec<Value> = tools_list
                    .iter()
                    .map(|tool| {
                        let decl = tool.declaration();
                        json!({
                            "type": "function",
                            "function": {
                                "name": decl.name(),
                                "description": decl.description(),
                                "parameters": decl.parameters(),
                                "strict": false
                            }
                        })
                    })
                    .collect();

                payload["tools"] = json!(tools);
            }
        }
    }

    /// Parses `DeepSeek` API response into Content (`OpenAI` format).
    fn parse_response(response_body: &Value) -> AgentResult<Content> {
        let mut content = Content::default();

        let first_choice = response_body
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .ok_or_else(|| AgentError::LlmProvider {
                provider: "DeepSeek".to_string(),
                message: "Missing or invalid 'choices' field in response".to_string(),
            })?;

        let message = first_choice
            .get("message")
            .ok_or_else(|| AgentError::LlmProvider {
                provider: "DeepSeek".to_string(),
                message: "Missing 'message' field in choice".to_string(),
            })?;

        if let Some(text) = message.get("content").and_then(|v| v.as_str()) {
            if !text.trim().is_empty() {
                content.push(ContentPart::Text(text.trim().to_string()));
            }
        }

        if let Some(tool_calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
            for tool_call in tool_calls {
                let id = tool_call
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::LlmProvider {
                        provider: "DeepSeek".to_string(),
                        message: "Missing 'id' in tool call".to_string(),
                    })?;

                let function =
                    tool_call
                        .get("function")
                        .ok_or_else(|| AgentError::LlmProvider {
                            provider: "DeepSeek".to_string(),
                            message: "Missing 'function' in tool call".to_string(),
                        })?;

                let name = function
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::LlmProvider {
                        provider: "DeepSeek".to_string(),
                        message: "Missing 'name' in tool call function".to_string(),
                    })?;

                let arguments = function.get("arguments").cloned().unwrap_or(Value::Null);
                let arguments = match arguments {
                    Value::String(s) => serde_json::from_str(&s).unwrap_or(Value::Null),
                    other => other,
                };

                content.push(ContentPart::ToolCall(ToolCall::new(id, name, arguments)));
            }
        }

        Ok(content)
    }

    /// Parses token usage from `DeepSeek` API response.
    fn parse_usage(response_body: &Value) -> TokenUsage {
        let Some(usage_obj) = response_body.get("usage") else {
            return TokenUsage::empty();
        };

        let prompt_tokens = usage_obj
            .get("prompt_tokens")
            .and_then(serde_json::Value::as_u64)
            .map(|v| u32::try_from(v).unwrap_or(u32::MAX));

        let completion_tokens = usage_obj
            .get("completion_tokens")
            .and_then(serde_json::Value::as_u64)
            .map(|v| u32::try_from(v).unwrap_or(u32::MAX));

        let total_tokens = usage_obj
            .get("total_tokens")
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
impl BaseLlm for DeepSeekLlm {
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

        // Create HTTP client
        let client = reqwest::Client::new();

        // Make request
        let response = client
            .post(&self.base_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
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
                    provider: "DeepSeek".to_string(),
                },
                429 => AgentError::LlmRateLimit {
                    provider: "DeepSeek".to_string(),
                },
                _ => AgentError::LlmProvider {
                    provider: "DeepSeek".to_string(),
                    message: format!("HTTP {status}: {error_body}"),
                },
            });
        }

        // Parse response
        let response_body: Value = response.json().await?;

        // Parse content and usage
        let content = Self::parse_response(&response_body)?;
        let usage = Self::parse_usage(&response_body);

        Ok(LlmResponse::new(content, usage))
    }
}

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
            "deepseek_tool"
        }

        fn description(&self) -> &'static str {
            "A test tool"
        }

        fn declaration(&self) -> crate::tools::FunctionDeclaration {
            crate::tools::FunctionDeclaration::new(
                "deepseek_tool",
                "A test tool",
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
    async fn build_request_payload_includes_tools() {
        let llm = DeepSeekLlm::new("deepseek-chat", "api-key")
            .with_max_tokens(512)
            .with_temperature(0.3);

        let thread = Thread::from_system("System prompt")
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

        assert_eq!(payload["model"], json!("deepseek-chat"));
        assert_eq!(payload["max_tokens"], json!(512));
        let temp = payload["temperature"].as_f64().expect("temperature");
        assert!((temp - 0.3).abs() < 1e-6);

        let tools = payload["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], json!("deepseek_tool"));
    }

    #[test]
    fn parse_response_extracts_text_and_tool_calls() {
        let _llm = DeepSeekLlm::new("deepseek-chat", "api-key");
        let body = json!({
            "choices": [
                {
                    "message": {
                        "content": "Hello user",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "lookup",
                                    "arguments": "{\"key\":\"value\"}"
                                }
                            }
                        ]
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13
            }
        });

        let content = DeepSeekLlm::parse_response(&body).expect("content");
        assert_eq!(content.first_text(), Some("Hello user"));
        let calls = content.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name(), "lookup");

        let usage = DeepSeekLlm::parse_usage(&body);
        assert_eq!(usage.input_tokens_opt(), Some(10));
        assert_eq!(usage.output_tokens_opt(), Some(3));
        assert_eq!(usage.total_tokens_opt(), Some(13));
    }

    #[test]
    fn parse_response_missing_choices_errors() {
        let _llm = DeepSeekLlm::new("deepseek-chat", "api-key");
        let body = json!({});
        let err = DeepSeekLlm::parse_response(&body).expect_err("expected error");
        match err {
            AgentError::LlmProvider { provider, .. } => assert_eq!(provider, "DeepSeek"),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn from_env_validates_presence() {
        let original = std::env::var(DeepSeekLlm::API_KEY_ENV).ok();
        std::env::remove_var(DeepSeekLlm::API_KEY_ENV);

        let missing = DeepSeekLlm::from_env("model");
        assert!(matches!(
            missing,
            Err(AgentError::MissingConfiguration { .. })
        ));

        std::env::set_var(DeepSeekLlm::API_KEY_ENV, "");
        let empty = DeepSeekLlm::from_env("model");
        assert!(matches!(
            empty,
            Err(AgentError::InvalidConfiguration { .. })
        ));

        match original {
            Some(value) => std::env::set_var(DeepSeekLlm::API_KEY_ENV, value),
            None => std::env::remove_var(DeepSeekLlm::API_KEY_ENV),
        }
    }
}
