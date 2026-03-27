//! LLM function abstraction for structured outputs.
//!
//! This module provides [`LlmFunction`], a high-level abstraction for calling
//! LLMs with structured, typed responses. `LlmFunction` wraps an LLM client and
//! handles thread management, system instructions, and response deserialization.
//!
//! # Overview
//!
//! - [`LlmFunction`]: Wrapper for LLM calls with typed responses
//!
//! # Examples
//!
//! ```ignore
//! use radkit::agent::LlmFunction;
//! use serde::Deserialize;
//!
//! #[derive(Deserialize)]
//! struct WeatherResponse {
//!     temperature: f64,
//!     condition: String,
//! }
//!
//! // Create an LLM function with system instructions
//! let weather_fn = LlmFunction::<WeatherResponse>::new_with_system_instructions(
//!     my_llm_client,
//!     "You are a weather assistant. Always respond with JSON."
//! );
//!
//! // Run the function with a thread
//! let response: WeatherResponse = weather_fn.run(thread).await?;
//! println!("Temperature: {}°F", response.temperature);
//!
//! // Or continue the conversation
//! let (response, continued_thread) = weather_fn.run_and_continue(thread).await?;
//! ```

use std::sync::Arc;

use schemars::JsonSchema;

use super::structured_parser::{build_structured_output_instructions, extract_structured_output};
use crate::errors::AgentResult;
use crate::models::LLMOutputTrait;
use crate::models::{BaseLlm, Content, Event, Thread};
use crate::{compat::MaybeSend, compat::MaybeSync};

pub struct LlmFunction<T> {
    model: Arc<dyn BaseLlm>,
    system_instructions: Option<String>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> LlmFunction<T>
where
    T: LLMOutputTrait + JsonSchema + MaybeSend + MaybeSync + 'static,
{
    /// Creates a new `LlmFunction` with a given `LlmClient`.
    pub fn new(model: impl BaseLlm + 'static) -> Self {
        Self::new_with_shared_model(Arc::new(model) as Arc<dyn BaseLlm>, None)
    }

    /// Creates a new `LlmFunction` with default system instructions applied to each call.
    pub fn new_with_system_instructions(
        model: impl BaseLlm + 'static,
        instructions: impl Into<String>,
    ) -> Self {
        Self::new_with_shared_model(
            Arc::new(model) as Arc<dyn BaseLlm>,
            Some(instructions.into()),
        )
    }

    pub(crate) fn new_with_shared_model(
        model: Arc<dyn BaseLlm>,
        system_instructions: Option<String>,
    ) -> Self {
        Self {
            model,
            system_instructions,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Runs the `LlmFunction` on a given input that can be converted into a `Thread`.
    ///
    /// This method executes the LLM with the provided thread and deserializes
    /// the response into type `T`.
    ///
    /// # Arguments
    ///
    /// * `input` - Thread or any type that can be converted into a Thread
    ///
    /// # Returns
    ///
    /// Returns the deserialized response of type `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The LLM call fails
    /// - Response deserialization fails
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let response: WeatherResponse = weather_fn.run(thread).await?;
    /// println!("Temperature: {}°F", response.temperature);
    /// ```
    pub async fn run<IT>(&self, input: IT) -> AgentResult<T>
    where
        IT: Into<Thread>,
    {
        let thread = self.apply_defaults(input.into())?;
        let outcome = self.invoke(&thread).await?;
        Ok(outcome.value)
    }

    /// Runs the `LlmFunction` and returns both the deserialized result and the thread for follow-up work.
    ///
    /// This is useful for multi-turn conversations where you need to continue the thread
    /// with additional messages.
    ///
    /// # Arguments
    ///
    /// * `input` - Thread or any type that can be converted into a Thread
    ///
    /// # Returns
    ///
    /// Returns a tuple of:
    /// - The deserialized response of type `T`
    /// - The updated `Thread` with the response included
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The LLM call fails
    /// - Response deserialization fails
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let (response, continued_thread) = weather_fn.run_and_continue(thread).await?;
    /// println!("Temperature: {}°F", response.temperature);
    ///
    /// // Continue the conversation
    /// let continued_thread = continued_thread
    ///     .with_user("What about tomorrow?");
    /// let (next_response, _) = weather_fn.run_and_continue(continued_thread).await?;
    /// ```
    pub async fn run_and_continue<IT>(&self, input: IT) -> AgentResult<(T, Thread)>
    where
        IT: Into<Thread>,
    {
        let thread = self.apply_defaults(input.into())?;
        let outcome = self.invoke(&thread).await?;

        let continued_thread = if let Some(content) = outcome.assistant_content {
            thread.add_event(Event::assistant(content))
        } else {
            thread
        };

        Ok((outcome.value, continued_thread))
    }

    fn apply_defaults(&self, mut thread: Thread) -> AgentResult<Thread> {
        // Build structured output instructions
        let structured_instructions = build_structured_output_instructions::<T>()?;

        // Combine with user system instructions if present
        let combined_instructions = if let Some(user_instructions) = &self.system_instructions {
            format!("{user_instructions}\n\n{structured_instructions}")
        } else {
            structured_instructions
        };

        thread = thread.with_system(combined_instructions);
        Ok(thread)
    }

    async fn invoke(&self, thread: &Thread) -> AgentResult<InvocationOutcome<T>> {
        // No toolset needed - LLM will respond with JSON directly in text
        let response = self.model.generate_content(thread.clone(), None).await?;

        let content = response.into_content();

        // Extract the structured output from the text response
        let value = extract_structured_output::<T>(&content)?;

        Ok(InvocationOutcome {
            value,
            assistant_content: Some(content),
        })
    }
}

struct InvocationOutcome<T> {
    value: T,
    assistant_content: Option<Content>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macros::LLMOutput;
    use crate::models::{LlmResponse, TokenUsage};
    use crate::test_support::FakeLlm;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, PartialEq, Deserialize, LLMOutput, Serialize, JsonSchema)]
    struct Sample {
        value: i32,
    }

    fn structured_response(value: i32, extra_text: Option<&str>) -> LlmResponse {
        let json_str = format!(r#"{{"value": {value}}}"#);

        let content_str = if let Some(text) = extra_text {
            format!("{text}\n```json\n{json_str}\n```")
        } else {
            json_str
        };

        LlmResponse::new(Content::from_text(content_str), TokenUsage::empty())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn run_returns_deserialized_value_and_applies_system_prompt() {
        let fake = Arc::new(FakeLlm::with_responses(
            "fake-model",
            [Ok(structured_response(10, None))],
        ));

        let shared: Arc<dyn BaseLlm> = fake.clone();
        let func = LlmFunction::<Sample>::new_with_shared_model(
            shared,
            Some("You are helpful".to_string()),
        );

        let input_thread = Thread::from_user("Calculate");
        let result = func.run(input_thread).await.expect("llm function");
        assert_eq!(result, Sample { value: 10 });

        let calls = fake.calls();
        assert_eq!(calls.len(), 1);
        // System prompt should contain both user instructions and schema instructions
        assert!(calls[0].system().unwrap().contains("You are helpful"));
        assert!(calls[0].system().unwrap().contains("JSON"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn run_and_continue_appends_assistant_content() {
        let fake = Arc::new(FakeLlm::with_responses(
            "fake-model",
            [Ok(structured_response(5, Some("Done")))],
        ));
        let shared: Arc<dyn BaseLlm> = fake.clone();
        let func = LlmFunction::<Sample>::new_with_shared_model(shared, None);

        let thread = Thread::from_user("Start");
        let (result, continued) = func.run_and_continue(thread).await.expect("llm function");

        assert_eq!(result, Sample { value: 5 });

        let events = continued.events();
        assert_eq!(events.len(), 2);
        assert!(events[1].content().joined_texts().unwrap().contains("Done"));

        assert_eq!(fake.calls().len(), 1);
    }
}
