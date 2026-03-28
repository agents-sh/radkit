//! Base LLM trait for content generation.
//!
//! This module defines the [`BaseLlm`] trait, which provides a unified interface
//! for interacting with different Large Language Model providers (e.g., `OpenAI`, Anthropic).
//!
//! # Examples
//!
//! ```ignore
//! use radkit::models::{BaseLlm, Thread};
//!
//! async fn generate(llm: &dyn BaseLlm) {
//!     let thread = Thread::from_user("Hello, world!");
//!     let response = llm.generate_content(thread, None).await.unwrap();
//!     println!("Response: {:?}", response);
//! }
//! ```

use std::sync::Arc;

use crate::errors::AgentResult;
use crate::models::{LlmResponse, Thread};
use crate::tools::BaseToolset;
use crate::{MaybeSend, MaybeSync};

/// Base trait for Large Language Model implementations.
///
/// This trait provides a unified interface for generating content from LLM providers.
/// All implementations must be `Send + Sync` (or equivalent via `MaybeSend + MaybeSync`
/// for WASM compatibility) to support concurrent usage across async tasks.
///
/// # Thread Safety
///
/// Implementations must be safe to share across threads (when not targeting WASM).
/// The trait bounds `MaybeSend + MaybeSync` ensure this portability.
///
/// # Error Handling
///
/// The [`generate_content`](BaseLlm::generate_content) method returns [`AgentResult<LlmResponse>`],
/// which may contain errors including
/// - Network/API errors when communicating with the LLM provider
/// - Authentication/authorization failures
/// - Rate limiting errors
/// - Invalid request parameters
/// - Tool execution failures (when using toolsets)
///
/// Implementors should map provider-specific errors into appropriate [`AgentError`](crate::errors::AgentError) variants.
#[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
pub trait BaseLlm: MaybeSend + MaybeSync {
    /// Returns the model identifier for this LLM instance.
    ///
    /// This should return a stable string identifier for the model (e.g., "gpt-4", "claude-3-opus").
    /// The model name is useful for logging, debugging, and selecting model-specific behavior.
    fn model_name(&self) -> &str;

    /// Generates content in response to a conversation thread.
    ///
    /// This is the primary method for interacting with the LLM. It takes a conversation
    /// thread representing the interaction history and optionally a toolset that the
    /// LLM can use to perform actions.
    ///
    /// # Arguments
    ///
    /// * `thread` - The conversation history, including system prompts and user/assistant messages
    /// * `toolset` - Optional set of tools the LLM can invoke during generation
    ///
    /// # Returns
    ///
    /// Returns an [`LlmResponse`] containing:
    /// - Generated content (text, tool calls, etc.)
    /// - Token usage information (input, output, and total tokens)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The LLM provider API request fails
    /// - Authentication is invalid or expired
    /// - The request is rate-limited
    /// - Tool execution fails (implementation-specific)
    /// - The thread contains invalid or unsupported content
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use radkit::models::{BaseLlm, Thread};
    ///
    /// async fn example(llm: &impl BaseLlm) -> Result<(), Box<dyn std::error::Error>> {
    ///     let thread = Thread::from_user("What is 2+2?");
    ///     let response = llm.generate_content(thread, None).await?;
    ///     println!("Answer: {}", response.content().first_text().unwrap_or("No text"));
    ///     println!("Tokens used: {}", response.usage().total_tokens());
    ///     Ok(())
    /// }
    /// ```
    async fn generate_content(
        &self,
        thread: Thread,
        toolset: Option<Arc<dyn BaseToolset>>,
    ) -> AgentResult<LlmResponse>;
}

/// Extension trait providing ergonomic helpers for [`BaseLlm`].
///
/// This trait is automatically implemented for all types that implement [`BaseLlm`],
/// providing convenient methods that accept any type convertible to [`Thread`].
///
/// # Design Pattern
///
/// This follows the standard Rust extension trait pattern used throughout the ecosystem
/// (e.g., `Iterator` + `IteratorExt`, `AsyncRead` + `AsyncReadExt`). The core trait
/// remains object-safe while extension methods provide zero-cost ergonomic improvements.
///
/// # Examples
///
/// ```ignore
/// use radkit::models::{BaseLlm, BaseLlmExt};
///
/// async fn example(llm: &impl BaseLlm) -> Result<(), Box<dyn std::error::Error>> {
///     // All of these work thanks to Into<Thread> implementations:
///     let r1 = llm.generate("What is 2+2?", None).await?;
///     let r2 = llm.generate(String::from("Hello!"), None).await?;
///     let r3 = llm.generate(Thread::from_user("Explain"), None).await?;
///
///     println!("Answer: {}", r1.content().first_text().unwrap_or("No text"));
///     Ok(())
/// }
/// ```
#[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
pub trait BaseLlmExt: BaseLlm {
    /// Generates content from any type convertible to a [`Thread`].
    ///
    /// This method provides an ergonomic wrapper around [`BaseLlm::generate_content`]
    /// that automatically converts strings, events, and other types into threads.
    ///
    /// # Arguments
    ///
    /// * `thread` - Anything convertible to `Thread`: `String`, `&str`, `Event`, or `Thread`
    /// * `toolset` - Optional set of tools the LLM can invoke during generation
    ///
    /// # Returns
    ///
    /// Returns an [`LlmResponse`] containing generated content and token usage.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use radkit::models::{BaseLlm, BaseLlmExt};
    ///
    /// async fn demo(llm: &impl BaseLlm) {
    ///     // String slice
    ///     let response = llm.generate("Hello", None).await?;
    ///
    ///     // Owned String
    ///     let query = String::from("What is Rust?");
    ///     let response = llm.generate(query, None).await?;
    ///
    ///     // Thread directly
    ///     let thread = Thread::from_user("Explain quantum computing");
    ///     let response = llm.generate(thread, None).await?;
    /// }
    /// ```
    async fn generate<T: Into<Thread> + MaybeSend>(
        &self,
        thread: T,
        toolset: Option<Arc<dyn BaseToolset>>,
    ) -> AgentResult<LlmResponse> {
        self.generate_content(thread.into(), toolset).await
    }
}

/// Blanket implementation of [`BaseLlmExt`] for all [`BaseLlm`] implementors.
///
/// This ensures every type implementing `BaseLlm` automatically gains the ergonomic
/// `generate` method without any additional implementation work.
impl<T: BaseLlm + ?Sized> BaseLlmExt for T {}

/// Blanket [`BaseLlm`] implementation for `Arc<dyn BaseLlm>`.
///
/// Allows passing a shared LLM handle directly anywhere `impl BaseLlm` is expected —
/// most importantly to [`LlmWorker::builder`] — without having to unwrap or clone the
/// inner concrete type:
///
/// ```ignore
/// // runtime.default_llm() returns Arc<dyn BaseLlm>
/// let worker = LlmWorker::<MyOutput>::builder(runtime.default_llm()).build();
/// ```
#[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
impl BaseLlm for Arc<dyn BaseLlm> {
    fn model_name(&self) -> &str {
        self.as_ref().model_name()
    }

    async fn generate_content(
        &self,
        thread: Thread,
        toolset: Option<Arc<dyn BaseToolset>>,
    ) -> AgentResult<LlmResponse> {
        self.as_ref().generate_content(thread, toolset).await
    }
}
