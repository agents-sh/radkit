//! Tools and toolsets for agent capabilities.
//!
//! This module provides the core abstractions for building tools that agents can use
//! to interact with external systems, APIs, and services. Tools are callable functions
//! that LLMs can invoke during execution.
//!
//! # Core Concepts
//!
//! - [`BaseTool`]: The fundamental trait for implementing a tool
//! - [`BaseToolset`]: Collections of related tools
//! - [`FunctionTool`]: Simple wrapper for Rust functions as tools
//! - [`A2AAgentTool`]: Tool for communicating with remote A2A agents
//! - [`ToolContext`]: Execution context passed to tools
//! - [`ExecutionState`]: Key-value storage for execution-scoped data
//!
//! # Examples
//!
//! ```ignore
//! use radkit::tools::{FunctionTool, SimpleToolset, ToolResult};
//! use serde_json::json;
//!
//! // Create a simple tool
//! let weather_tool = FunctionTool::new(
//!     "get_weather",
//!     "Get current weather for a location",
//!     |args, _ctx| Box::pin(async move {
//!         ToolResult::success(json!({"temp": 72, "condition": "sunny"}))
//!     })
//! );
//!
//! // Create a toolset
//! let toolset = SimpleToolset::new(vec![Box::new(weather_tool)]);
//! ```

mod a2a_agent_tool;
pub mod base_tool;
pub mod base_toolset;
mod execution_state;
pub mod function_tool;
#[cfg(all(feature = "mcp", not(all(target_os = "wasi", target_env = "p1"))))]
pub mod mcp;
pub mod memory;
#[cfg(feature = "openapi")]
pub mod openapi;
pub mod tool;
pub mod tool_context;

pub use a2a_agent_tool::A2AAgentTool;
pub use base_tool::BaseTool;
pub use base_toolset::{BaseToolset, CombinedToolset, SimpleToolset};
pub use execution_state::{DefaultExecutionState, ExecutionState};
pub use function_tool::FunctionTool;
#[cfg(all(feature = "mcp", not(all(target_os = "wasi", target_env = "p1"))))]
pub use mcp::{MCPConnectionParams, MCPSessionManager, MCPTool, MCPToolFilter, MCPToolset};
#[cfg(feature = "openapi")]
pub use openapi::{AuthConfig, HeaderOrQuery, OpenApiToolSet};
pub use tool::{FunctionDeclaration, ToolCall, ToolResponse, ToolResult};
pub use tool_context::{ToolContext, ToolContextBuilder};
