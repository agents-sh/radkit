//! # A2A Protocol Client
//!
//! This crate provides a client for calling remote A2A (Agent-to-Agent) protocol compliant agents.
//! It supports both streaming and non-streaming interactions over HTTP/HTTPS.
//!
//! ## Features
//!
//! - A2A v1 transport support over HTTP+JSON and JSON-RPC 2.0
//! - Non-streaming and streaming message support
//! - Task retrieval and listing
//! - Agent discovery via agent cards
//! - Authentication support (Bearer tokens)
//! - Error handling with detailed error types
//!
//! ## Example
//!
//! ```rust,no_run
//! use a2a_client::A2AClient;
//! use a2a_types::{Message, Part, Role, SendMessageRequest, part};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create client from agent card URL
//! let client = A2AClient::from_card_url("https://agent.example.com")
//!     .await?
//!     .with_auth_token("your_api_key");
//!
//! // Create message
//! let message = Message {
//!     message_id: "msg_123".to_string(),
//!     context_id: String::new(),
//!     task_id: String::new(),
//!     role: Role::User.into(),
//!     parts: vec![Part {
//!         content: Some(part::Content::Text("Hello!".to_string())),
//!         metadata: None,
//!         filename: String::new(),
//!         media_type: "text/plain".to_string(),
//!     }],
//!     metadata: None,
//!     extensions: Vec::new(),
//!     reference_task_ids: Vec::new(),
//! };
//!
//! // Send message
//! let result = client
//!     .send_message(SendMessageRequest {
//!         tenant: String::new(),
//!         message: Some(message),
//!         configuration: None,
//!         metadata: None,
//!     })
//!     .await?;
//! # let _ = result;
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod constants;
pub mod error;

pub use client::A2AClient;
pub use error::{A2AError, A2AResult};

/// Re-export A2A protocol types so downstream crates can ensure they use the
/// exact same type definitions as the client.
pub mod types {
    pub use a2a_types::{self as v1, JSONRPCError, JSONRPCErrorResponse, JSONRPCId};
}
