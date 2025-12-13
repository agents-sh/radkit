/// Main error type for the Agent SDK
use crate::compat::MaybeSendBoxError;
#[cfg(all(feature = "runtime", not(all(target_os = "wasi", target_env = "p1"))))]
use {
    axum::http::StatusCode,
    axum::response::{IntoResponse, Json, Response},
    serde_json::json,
};

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    // === LLM Provider Errors ===
    #[error("LLM provider error ({provider}): {message}")]
    LlmProvider { provider: String, message: String },

    #[error("LLM API authentication failed: {provider}")]
    LlmAuthentication { provider: String },

    #[error("LLM API rate limit exceeded: {provider}")]
    LlmRateLimit { provider: String },

    #[error("LLM content filtered: {reason}")]
    ContentFiltered { reason: String },

    #[error("LLM context length exceeded: {current_tokens} > {max_tokens}")]
    ContextLengthExceeded {
        current_tokens: u32,
        max_tokens: u32,
    },

    #[error("LLM error: {source}")]
    LlmError {
        #[source]
        source: MaybeSendBoxError,
    },

    #[error("Feature not implemented: {feature}")]
    NotImplemented { feature: String },

    // === Task Management Errors ===
    #[error("Skill not found: {skill_id}")]
    SkillNotFound { skill_id: String },

    #[error("Task not found: {task_id}")]
    TaskNotFound { task_id: String },

    #[error("Task already exists: {task_id}")]
    TaskAlreadyExists { task_id: String },

    #[error("Invalid task state transition: {from} -> {to}")]
    InvalidTaskStateTransition { from: String, to: String },

    #[error("Task operation failed: {operation} on {task_id}: {reason}")]
    TaskOperationFailed {
        operation: String,
        task_id: String,
        reason: String,
    },

    // === Session Management Errors ===
    #[error("Session not found: {session_id} for {app_name}/{user_id}")]
    SessionNotFound {
        session_id: String,
        app_name: String,
        user_id: String,
    },

    #[error("Session access denied: {session_id} for {app_name}/{user_id}")]
    SessionAccessDenied {
        session_id: String,
        app_name: String,
        user_id: String,
    },

    #[error("Session state error: {session_id}: {reason}")]
    SessionStateError { session_id: String, reason: String },

    // === Tool Execution Errors ===
    #[error("Tool not found: {tool_name}")]
    ToolNotFound { tool_name: String },

    #[error("Tool execution failed: {tool_name}: {reason}")]
    ToolExecutionFailed { tool_name: String, reason: String },

    #[error("Tool validation error: {tool_name}: {reason}")]
    ToolValidationError { tool_name: String, reason: String },

    #[error("Tool timeout: {tool_name} exceeded {timeout_ms}ms")]
    ToolTimeout { tool_name: String, timeout_ms: u64 },

    #[error("Tool setup failed: {tool_name}: {reason}")]
    ToolSetupFailed { tool_name: String, reason: String },

    // === Security Errors ===
    #[error("Access denied: insufficient permissions for {app_name}/{user_id}")]
    AccessDenied { app_name: String, user_id: String },

    #[error("Invalid credentials for {app_name}")]
    InvalidCredentials { app_name: String },

    #[error("Security policy violation: {policy}: {reason}")]
    SecurityViolation { policy: String, reason: String },

    // === Configuration Errors ===
    #[error("Invalid configuration: {field}: {reason}")]
    InvalidConfiguration { field: String, reason: String },

    #[error("Missing configuration: {field}")]
    MissingConfiguration { field: String },

    // === Network/IO Errors ===
    #[error("Network error: {operation}: {reason}")]
    Network { operation: String, reason: String },

    #[error("Serialization error: {format}: {reason}")]
    Serialization { format: String, reason: String },

    // === General System Errors ===
    #[error("Resource exhausted: {resource}: {reason}")]
    ResourceExhausted { resource: String, reason: String },

    #[error("Internal error: {component}: {reason}")]
    Internal { component: String, reason: String },

    #[error("Validation error: {field}: {reason}")]
    Validation { field: String, reason: String },

    // === Timeout Errors ===
    #[error("Operation timed out: {operation} after {duration_ms}ms")]
    Timeout { operation: String, duration_ms: u64 },

    // === Skill Execution Errors ===
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Missing input: {0}")]
    MissingInput(String),

    #[error("Skill slot error: {0}")]
    SkillSlot(String),

    // === Runtime Errors ===
    #[error("Server start failed: {0}")]
    ServerStartFailed(String),

    #[error("Blocking operations not supported on WASM")]
    BlockingNotSupported,

    // === Data Validation Errors ===
    #[error("Invalid MIME type: {0}")]
    InvalidMimeType(String),

    #[error("Invalid base64 encoding: {0}")]
    InvalidBase64(String),

    #[error("Invalid URI: {0}")]
    InvalidUri(String),

    // === Context Errors ===
    #[error("Context error: {0}")]
    ContextError(String),

    // === Memory Service Errors ===
    /// Memory service errors (embedding failures, backend errors, etc.)
    #[error("Memory error: {0}")]
    Memory(String),
}

/// Convenience type alias
pub type AgentResult<T> = std::result::Result<T, AgentError>;

#[cfg(all(feature = "runtime", not(all(target_os = "wasi", target_env = "p1"))))]
impl IntoResponse for AgentError {
    fn into_response(self) -> Response {
        let (status, error_message) = match &self {
            Self::SkillNotFound { .. }
            | Self::TaskNotFound { .. }
            | Self::SessionNotFound { .. } => (StatusCode::NOT_FOUND, self.to_string()),

            Self::InvalidConfiguration { .. }
            | Self::Validation { .. }
            | Self::InvalidTaskStateTransition { .. }
            | Self::InvalidInput(..)
            | Self::MissingInput(..)
            | Self::InvalidMimeType(..)
            | Self::InvalidBase64(..)
            | Self::InvalidUri(..) => (StatusCode::BAD_REQUEST, self.to_string()),

            Self::AccessDenied { .. } | Self::SessionAccessDenied { .. } => {
                (StatusCode::FORBIDDEN, self.to_string())
            }

            Self::InvalidCredentials { .. } | Self::LlmAuthentication { .. } => {
                (StatusCode::UNAUTHORIZED, self.to_string())
            }

            Self::NotImplemented { .. } => (StatusCode::NOT_IMPLEMENTED, self.to_string()),

            _ => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };

        let body = Json(json!({ "error": error_message }));
        (status, body).into_response()
    }
}

/// Convert `AgentError` to `ToolResult` for tool execution contexts
impl From<AgentError> for crate::tools::ToolResult {
    fn from(error: AgentError) -> Self {
        Self::error(error.to_string())
    }
}

/// Convert common std errors to `AgentError`
impl From<serde_json::Error> for AgentError {
    fn from(error: serde_json::Error) -> Self {
        Self::Serialization {
            format: "json".to_string(),
            reason: error.to_string(),
        }
    }
}

impl From<std::io::Error> for AgentError {
    fn from(error: std::io::Error) -> Self {
        Self::Internal {
            component: "io".to_string(),
            reason: error.to_string(),
        }
    }
}

impl From<tokio::task::JoinError> for AgentError {
    fn from(error: tokio::task::JoinError) -> Self {
        let reason = if error.is_cancelled() {
            "task cancelled".to_string()
        } else if error.is_panic() {
            "task panicked".to_string()
        } else {
            error.to_string()
        };

        Self::Internal {
            component: "task".to_string(),
            reason,
        }
    }
}

impl From<reqwest::Error> for AgentError {
    fn from(error: reqwest::Error) -> Self {
        Self::Network {
            operation: "http_request".to_string(),
            reason: error.to_string(),
        }
    }
}

impl From<std::num::ParseIntError> for AgentError {
    fn from(error: std::num::ParseIntError) -> Self {
        Self::InvalidInput(error.to_string())
    }
}

impl From<std::num::ParseFloatError> for AgentError {
    fn from(error: std::num::ParseFloatError) -> Self {
        Self::InvalidInput(error.to_string())
    }
}

#[cfg(feature = "openapi")]
impl From<serde_yaml::Error> for AgentError {
    fn from(error: serde_yaml::Error) -> Self {
        Self::Serialization {
            format: "yaml".to_string(),
            reason: error.to_string(),
        }
    }
}

#[cfg(feature = "openapi")]
impl From<url::ParseError> for AgentError {
    fn from(error: url::ParseError) -> Self {
        Self::InvalidUri(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_to_string_contains_context() {
        let err = AgentError::InvalidConfiguration {
            field: "api_key".into(),
            reason: "missing".into(),
        };
        let message = err.to_string();
        assert!(message.contains("api_key"));
        assert!(message.contains("missing"));
    }
}
