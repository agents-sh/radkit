//! LLM provider implementations.
//!
//! This module contains concrete implementations of the [`BaseLlm`](crate::models::BaseLlm)
//! trait for various LLM providers including Anthropic, `OpenAI`, Gemini, and others.
//!
//! # Available Providers
//!
//! - [`AnthropicLlm`]: Claude models via Anthropic API
//! - [`OpenAILlm`]: GPT models via `OpenAI` API
//! - [`GeminiLlm`]: Gemini models via Google AI API
//! - [`GrokLlm`]: Grok models via XAI API (OpenAI-compatible)
//! - [`DeepSeekLlm`]: `DeepSeek` models via `DeepSeek` API (OpenAI-compatible)
//! - [`OpenRouterLlm`]: `OpenRouter` marketplace models via OpenAI-compatible API
//!
//! # Computer Use (Gemini)
//!
//! [`GeminiLlm`] also exposes a browser-control agent via
//! [`GeminiLlm::computer_use_worker`]. Implement [`ComputerUseHandler`] for your
//! browser automation library, then call [`GeminiComputerUseWorker::run`] with a
//! plain-text goal.
//!
//! Key types:
//! - [`ComputerUseHandler`] — trait to implement for your browser environment
//! - [`GeminiComputerUseWorker`] — the agentic loop
//! - [`ComputerUseAction`] — a single UI action requested by the model
//! - [`ActionOutcome`] — result you return after executing an action
//! - [`SafetyDecision`] — present when the model requires user confirmation
//!
//! # Examples
//!
//! ```ignore
//! use radkit::models::providers::{AnthropicLlm, OpenAILlm, GeminiLlm, OpenRouterLlm};
//! use radkit::models::{BaseLlm, Thread};
//!
//! // Anthropic
//! let llm = AnthropicLlm::from_env("claude-sonnet-4-5-20250929")?;
//!
//! // OpenAI
//! let llm = OpenAILlm::from_env("gpt-4o")?;
//!
//! // OpenRouter
//! let llm = OpenRouterLlm::from_env("anthropic/claude-3.5-sonnet")?;
//!
//! // Gemini – text generation
//! let llm = GeminiLlm::from_env("gemini-2.5-flash")?;
//! let thread = Thread::from_user("Hello!");
//! let response = llm.generate_content(thread, None).await?;
//!
//! // Gemini – Computer Use
//! use std::sync::Arc;
//! use radkit::models::providers::{ComputerUseHandler, ComputerUseAction, ActionOutcome};
//!
//! struct MyBrowser;
//!
//! #[async_trait::async_trait]
//! impl ComputerUseHandler for MyBrowser {
//!     async fn screenshot(&self) -> Result<Vec<u8>, String> { todo!() }
//!     async fn execute(&self, action: ComputerUseAction) -> ActionOutcome { todo!() }
//! }
//!
//! let llm = Arc::new(GeminiLlm::from_env("gemini-2.5-computer-use-preview-10-2025")?);
//! let answer = llm
//!     .computer_use_worker(Arc::new(MyBrowser))
//!     .run("Search Google for the Rust programming language")
//!     .await?;
//! ```

mod anthropic_llm;
mod deepseek_llm;
mod gemini_llm;
mod grok_llm;
mod openai_llm;
mod openrouter_llm;

pub use anthropic_llm::AnthropicLlm;
pub use deepseek_llm::DeepSeekLlm;
pub use gemini_llm::{
    ActionOutcome, ComputerUseAction, ComputerUseHandler, GeminiComputerUseWorker, GeminiLlm,
    SafetyDecision,
};
pub use grok_llm::GrokLlm;
pub use openai_llm::OpenAILlm;
pub use openrouter_llm::OpenRouterLlm;
