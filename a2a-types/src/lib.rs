//! # A2A (Agent2Agent) Protocol Types
//!
//! This crate provides the Rust data structures for the Agent2Agent (A2A) protocol,
//! generated from the canonical `proto/a2a.proto` definition via `prost` + `pbjson`.
//!
//! All types are available directly as `a2a_types::Foo`.

use serde::{Deserialize, Serialize};

// ============================================================================
// Proto-generated types — included directly at the crate root.
// No intermediate module; every generated type is a first-class member of
// `a2a_types`.
// ============================================================================

include!(concat!(env!("OUT_DIR"), "/lf.a2a.v1.rs"));
include!(concat!(env!("OUT_DIR"), "/lf.a2a.v1.serde.rs"));

// ============================================================================
// JSON-RPC 2.0 Wire Types
//
// JSON-RPC framing is not in the proto (it is a transport-level binding),
// so these three types remain hand-written. Used by `a2a-client` and the
// `web` layer in `radkit` to parse and build JSON-RPC envelopes.
// ============================================================================

/// Represents a JSON-RPC 2.0 identifier, which can be a string, number, or null.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum JSONRPCId {
    String(String),
    Integer(i64),
    Null,
}

/// Represents a JSON-RPC 2.0 Error Response object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSONRPCErrorResponse {
    /// The version of the JSON-RPC protocol. MUST be exactly "2.0".
    pub jsonrpc: String,
    /// An object describing the error that occurred.
    pub error: JSONRPCError,
    /// The identifier established by the client.
    pub id: Option<JSONRPCId>,
}

/// Represents a JSON-RPC 2.0 Error object, included in an error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSONRPCError {
    /// A number that indicates the error type that occurred.
    pub code: i32,
    /// A string providing a short description of the error.
    pub message: String,
    /// A primitive or structured value containing additional information about the error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

// ============================================================================
// A2A Error Types
//
// Thin wrappers carrying well-known JSON-RPC error codes for the A2A protocol.
// Used by `error_mapper.rs` to build `JSONRPCError` values from `AgentError`.
// ============================================================================

/// An error indicating that the JSON sent is not a valid Request object.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct InvalidRequestError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl InvalidRequestError {
    #[must_use]
    pub fn new() -> Self {
        Self {
            code: -32600,
            message: "Request payload validation error".to_string(),
            data: None,
        }
    }
}

/// An error indicating that the method parameters are invalid.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct InvalidParamsError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl InvalidParamsError {
    #[must_use]
    pub fn new() -> Self {
        Self {
            code: -32602,
            message: "Invalid parameters".to_string(),
            data: None,
        }
    }
}

/// An error indicating an internal error on the server.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct InternalError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl InternalError {
    #[must_use]
    pub fn new() -> Self {
        Self {
            code: -32603,
            message: "Internal error".to_string(),
            data: None,
        }
    }
}

/// An A2A-specific error indicating that the requested task ID was not found.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TaskNotFoundError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl TaskNotFoundError {
    #[must_use]
    pub fn new() -> Self {
        Self {
            code: -32001,
            message: "Task not found".to_string(),
            data: None,
        }
    }
}

/// An A2A-specific error indicating that the requested operation is not supported.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct UnsupportedOperationError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl UnsupportedOperationError {
    #[must_use]
    pub fn new() -> Self {
        Self {
            code: -32004,
            message: "This operation is not supported".to_string(),
            data: None,
        }
    }
}
