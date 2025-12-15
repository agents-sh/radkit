//! # radkit
//!
//! A Rust SDK for building AI agents that run on native and WASM targets.
//!
//! ## Overview
//!
//! radkit provides:
//! - **Agent Runtime**: Execute agents with skills and tool calling
//! - **LLM Integration**: Unified interface for multiple LLM providers
//! - **Tool System**: Extensible tool/function calling framework
//! - **Cross-Platform**: Native (tokio) and WASM (single-threaded) support
//! - **Type Safety**: Strong types with comprehensive error handling
//!
//! ## Quick Start
//!
//! ```ignore
//! use radkit::agent::Agent;
//! use radkit::runtime::RuntimeHandle;
//! use radkit::tools::{FunctionTool, ToolResult};
//! use serde_json::json;
//!
//! // Create a simple agent
//! let agent = Agent::builder()
//!     .with_name("My Agent")
//!     .build();
//!
//! // Define tools
//! let tool = FunctionTool::new(
//!     "get_weather",
//!     "Get weather for a location",
//!     |args, _ctx| Box::pin(async move {
//!         ToolResult::success(json!({"temp": 72}))
//!     })
//! );
//!
//! // Create a runtime
//! # let llm = radkit::test_support::FakeLlm::with_responses("demo", std::iter::empty());
//! let runtime = RuntimeHandle::builder(agent, llm).build();
//! ```
//!
//! ## Feature Flags
//!
//! radkit uses feature flags to control optional dependencies:
//! - (Features will be documented as they're added)
//!
//! ## Platform Support
//!
//! - **Native** (Linux, macOS, Windows): Full async runtime with tokio
//! - **WASM** (wasm32-wasi, wasm32-unknown-unknown): Single-threaded execution
//!
//! ## Architecture
//!
//! - [`agent`]: Agent definitions and builders
//! - [`tools`]: Tool system for function calling
//! - [`runtime`]: Runtime services and execution
//! - [`errors`]: Typed error handling
//! - [`compat`]: Cross-platform compatibility layer

#![deny(unsafe_code, unreachable_patterns, unused_must_use)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![recursion_limit = "512"]

pub mod agent;
pub mod compat;
pub mod errors;
pub mod models;
pub mod runtime;
pub mod tools;

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;

#[cfg(feature = "macros")]
/// Re-exported procedural macros (e.g., `radkit::macros::skill!`).
pub mod macros {
    pub use radkit_macros::*;

    /// Derive macro for fuzzy deserialization of LLM outputs.
    ///
    /// This macro provides robust parsing for LLM-generated JSON with:
    /// - Fuzzy field matching: `user_name` matches `userName`, `UserName`, `user-name`, etc.
    /// - Fuzzy enum matching: Case-insensitive variant matching with edit-distance scoring
    /// - Type coercion: String numbers to integers, singles to arrays, etc.
    /// - Transformation tracking: Records all modifications for debugging
    ///
    /// Supports `#[llm(union)]` attribute on enums for score-based variant selection.
    ///
    /// # Important Requirements
    ///
    /// ## Must Derive Both `Deserialize` and `LLMOutput`
    ///
    /// The `LLMOutput` derive macro requires `serde::Deserialize` to be derived as well:
    ///
    /// ```ignore
    /// use radkit::macros::LLMOutput;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize, LLMOutput)]  // Both required!
    /// struct MyOutput {
    ///     field: String,
    /// }
    /// ```
    ///
    /// ## Supported Types
    ///
    /// All standard integer and float types are supported with fuzzy parsing:
    ///
    /// ```ignore
    /// #[derive(Deserialize, LLMOutput)]
    /// struct Output {
    ///     count: i32,   // ✅ All integer types
    ///     size: usize,  // ✅ usize/isize
    ///     score: f64,   // ✅ f32/f64
    /// }
    /// ```
    ///
    /// ## Trait vs Macro Import
    ///
    /// This module provides the **derive macro**. For trait bounds, import from `models`:
    ///
    /// ```ignore
    /// // Derive macro (for #[derive(...)])
    /// use radkit::macros::LLMOutput;
    ///
    /// // Trait (for bounds like T: LLMOutputTrait)
    /// use radkit::models::LLMOutputTrait;
    /// ```
    ///
    /// # Example
    ///
    /// ```ignore
    /// use radkit::macros::LLMOutput;
    /// use radkit::tryparse;
    /// use schemars::JsonSchema;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize, LLMOutput, JsonSchema)]
    /// struct UserData {
    ///     name: String,
    ///     github_username: String,  // Matches githubUsername, GitHub-Username, etc.
    ///     age: u32,                 // All integer types supported
    /// }
    ///
    /// // Parse with fuzzy matching
    /// let data: UserData = tryparse::parse_llm(llm_response)?;
    /// ```
    ///
    /// Implements the `radkit::models::LLMOutputTrait` trait (alias for `tryparse::deserializer::LlmDeserialize`).
    pub use tryparse_derive::LlmDeserialize as LLMOutput;
}

pub use crate::compat::{MaybeSend, MaybeSync};

// Re-export tryparse for derive macro path resolution.
// This allows external crates to use `radkit::tryparse` instead of adding
// tryparse as a direct dependency. The tryparse-derive macro uses
// proc-macro-crate to find this re-export.
#[doc(hidden)]
pub use tryparse as __private_tryparse;

/// Re-exported tryparse crate for parsing LLM outputs.
///
/// This module re-exports the `tryparse` crate, allowing users of radkit
/// to parse LLM outputs without adding `tryparse` as a direct dependency.
///
/// # Example
///
/// ```ignore
/// use radkit::macros::LLMOutput;
/// use radkit::tryparse;
/// use serde::Deserialize;
///
/// #[derive(Deserialize, LLMOutput)]
/// struct UserData {
///     name: String,
///     age: u32,
/// }
///
/// // Parse LLM output with fuzzy matching
/// let data: UserData = tryparse::parse_llm(llm_response)?;
/// ```
pub use tryparse;
