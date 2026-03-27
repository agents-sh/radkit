//! LLM response types with token usage tracking.
//!
//! This module provides response types returned by LLM providers, including
//! generated content and token usage information.

use serde::{Deserialize, Serialize};

use crate::models::Content;

/// Response from an LLM generation request.
///
/// Contains the generated content and metadata about token usage.
/// This is returned by all [`BaseLlm`](crate::models::BaseLlm) implementations.
///
/// # Examples
///
/// ```ignore
/// use radkit::models::{BaseLlm, Thread};
///
/// let llm = AnthropicLlm::from_env("claude-sonnet-4-5-20250929")?;
/// let thread = Thread::from_user("What is Rust?");
/// let response = llm.generate_content(thread, None).await?;
///
/// println!("Response: {}", response.content().first_text().unwrap_or(""));
/// println!("Tokens used: {}", response.usage().total_tokens());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    content: Content,
    usage: TokenUsage,
}

impl LlmResponse {
    /// Creates a new LLM response.
    ///
    /// # Arguments
    ///
    /// * `content` - The generated content
    /// * `usage` - Token usage information
    #[must_use]
    pub const fn new(content: Content, usage: TokenUsage) -> Self {
        Self { content, usage }
    }

    /// Returns a reference to the generated content.
    #[must_use]
    pub const fn content(&self) -> &Content {
        &self.content
    }

    /// Returns a reference to the token usage information.
    #[must_use]
    pub const fn usage(&self) -> &TokenUsage {
        &self.usage
    }

    /// Consumes the response and returns the content.
    #[must_use]
    pub fn into_content(self) -> Content {
        self.content
    }

    /// Consumes the response and returns both content and usage.
    #[must_use]
    pub fn into_parts(self) -> (Content, TokenUsage) {
        (self.content, self.usage)
    }
}

/// Token usage statistics from an LLM request.
///
/// Tracks the number of tokens consumed in the prompt, generated in the output,
/// and the total. All values are optional as some providers may not report all metrics.
///
/// # Examples
///
/// ```ignore
/// let usage = TokenUsage::new(100, 50, 150);
/// println!("Input: {}, Output: {}, Total: {}",
///     usage.input_tokens(),
///     usage.output_tokens(),
///     usage.total_tokens());
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    input: Option<u32>,
    output: Option<u32>,
    total: Option<u32>,
}

impl TokenUsage {
    /// Creates a new token usage record.
    ///
    /// # Arguments
    ///
    /// * `input_tokens` - Number of tokens in the input/prompt
    /// * `output_tokens` - Number of tokens generated in the output
    /// * `total_tokens` - Total tokens (typically input + output)
    #[must_use]
    pub const fn new(input_tokens: u32, output_tokens: u32, total_tokens: u32) -> Self {
        Self {
            input: Some(input_tokens),
            output: Some(output_tokens),
            total: Some(total_tokens),
        }
    }

    /// Creates an empty token usage (all fields None).
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Creates a token usage with only some fields populated.
    #[must_use]
    pub const fn partial(
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
        total_tokens: Option<u32>,
    ) -> Self {
        Self {
            input: input_tokens,
            output: output_tokens,
            total: total_tokens,
        }
    }

    /// Returns the number of input tokens, or 0 if not reported.
    #[must_use]
    pub fn input_tokens(&self) -> u32 {
        self.input.unwrap_or(0)
    }

    /// Returns the number of output tokens, or 0 if not reported.
    #[must_use]
    pub fn output_tokens(&self) -> u32 {
        self.output.unwrap_or(0)
    }

    /// Returns the total number of tokens, or 0 if not reported.
    #[must_use]
    pub fn total_tokens(&self) -> u32 {
        self.total.unwrap_or(0)
    }

    /// Returns the input tokens as an Option.
    #[must_use]
    pub const fn input_tokens_opt(&self) -> Option<u32> {
        self.input
    }

    /// Returns the output tokens as an Option.
    #[must_use]
    pub const fn output_tokens_opt(&self) -> Option<u32> {
        self.output
    }

    /// Returns the total tokens as an Option.
    #[must_use]
    pub const fn total_tokens_opt(&self) -> Option<u32> {
        self.total
    }

    /// Sets the input tokens.
    #[must_use]
    pub const fn with_input_tokens(mut self, tokens: u32) -> Self {
        self.input = Some(tokens);
        self
    }

    /// Sets the output tokens.
    #[must_use]
    pub const fn with_output_tokens(mut self, tokens: u32) -> Self {
        self.output = Some(tokens);
        self
    }

    /// Sets the total tokens.
    #[must_use]
    pub const fn with_total_tokens(mut self, tokens: u32) -> Self {
        self.total = Some(tokens);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ContentPart;

    #[test]
    fn llm_response_accessors_work() {
        let content = Content::from_parts(vec![ContentPart::Text("Hello".into())]);
        let usage = TokenUsage::new(1, 2, 3);
        let response = LlmResponse::new(content, usage);

        assert_eq!(response.content().first_text(), Some("Hello"));
        assert_eq!(response.usage().total_tokens(), 3);

        let (content_parts, metrics) = response.into_parts();
        assert_eq!(content_parts.first_text(), Some("Hello"));
        assert_eq!(metrics.total_tokens(), 3);
    }

    #[test]
    fn token_usage_option_helpers() {
        let usage = TokenUsage::empty()
            .with_input_tokens(5)
            .with_output_tokens(7)
            .with_total_tokens(12);

        assert_eq!(usage.input_tokens_opt(), Some(5));
        assert_eq!(usage.output_tokens(), 7);
        assert_eq!(usage.total_tokens(), 12);
    }
}
