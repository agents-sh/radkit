//! Content parts for representing different types of message content.
//!
//! This module provides [`ContentPart`], an enum for representing the various
//! types of content that can appear in a message: text, binary data, tool calls,
//! and tool responses.
//!
//! # Examples
//!
//! ```ignore
//! use radkit::models::ContentPart;
//!
//! // Create text content
//! let text = ContentPart::Text("Hello".to_string());
//!
//! // Create data content
//! let data = ContentPart::from_base64("image/png", "iVBORw0KG...", None);
//! ```

use crate::errors::AgentError;
use crate::tools::{ToolCall, ToolResponse};
use base64::Engine;
use derive_more::From;
use serde::{Deserialize, Serialize};

/// A segment of content in a message.
///
/// Content parts represent the different types of content that can be included
/// in a message exchanged with an LLM. This includes text, binary data (like images),
/// tool calls made by the LLM, and tool responses.
///
/// This enum is marked `#[non_exhaustive]` to allow adding new content types
/// in the future without breaking changes.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, From)]
pub enum ContentPart {
    #[from(String, &String, &str)]
    Text(String),

    #[from]
    Data(Data),

    #[from]
    ToolCall(ToolCall),

    #[from]
    ToolResponse(ToolResponse),
}

/// The source of binary data, either inline or via a URI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    Base64(String),
    Uri(String),
}

/// A block of binary data.
///
/// Represents binary data (such as images) encoded in base64 format or located at a URI.
/// The data includes a MIME type for content identification.
///
/// # Invariants
///
/// - `content_type` must be a valid MIME type format (e.g., "image/png")
/// - `source` must contain valid data (e.g., valid base64 or a valid URI format)
///
/// Use [`Data::new`] to construct instances with validation, or use
/// [`Data::new_unchecked`] if you've already validated the inputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Data {
    /// MIME type (e.g., "image/png").
    pub content_type: String,

    /// The source of the binary data.
    pub source: DataSource,

    /// Optional display name.
    pub name: Option<String>,
}

impl ContentPart {
    /// Creates a new text content part.
    pub fn from_text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Creates a new data content part from base64 with validation.
    ///
    /// # Errors
    ///
    /// Returns an error if the content type is invalid or base64 data is empty.
    pub fn from_base64(
        content_type: impl Into<String>,
        base64: impl Into<String>,
        name: Option<String>,
    ) -> Result<Self, AgentError> {
        Ok(Self::Data(Data::new(
            content_type,
            DataSource::Base64(base64.into()),
            name,
        )?))
    }

    /// Creates a new data content part from a URI.
    ///
    /// # Errors
    ///
    /// Returns an error if the content type is invalid or URI is empty.
    pub fn from_uri(
        content_type: impl Into<String>,
        uri: impl Into<String>,
        name: Option<String>,
    ) -> Result<Self, AgentError> {
        Ok(Self::Data(Data::new(
            content_type,
            DataSource::Uri(uri.into()),
            name,
        )?))
    }

    /// Returns a reference to the inner text if this part is text.
    #[must_use]
    pub const fn as_text(&self) -> Option<&str> {
        if let Self::Text(content) = self {
            Some(content.as_str())
        } else {
            None
        }
    }

    /// Consumes the part and returns the inner text.
    #[must_use]
    pub fn into_text(self) -> Option<String> {
        if let Self::Text(content) = self {
            Some(content)
        } else {
            None
        }
    }

    /// Returns a reference to the inner tool call if present.
    #[must_use]
    pub const fn as_tool_call(&self) -> Option<&ToolCall> {
        if let Self::ToolCall(tool_call) = self {
            Some(tool_call)
        } else {
            None
        }
    }

    /// Consumes the part and returns the inner tool call.
    #[must_use]
    pub fn into_tool_call(self) -> Option<ToolCall> {
        if let Self::ToolCall(tool_call) = self {
            Some(tool_call)
        } else {
            None
        }
    }

    /// Returns a reference to the inner tool response if present.
    #[must_use]
    pub const fn as_tool_response(&self) -> Option<&ToolResponse> {
        if let Self::ToolResponse(tool_response) = self {
            Some(tool_response)
        } else {
            None
        }
    }

    /// Consumes the part and returns the inner tool response.
    #[must_use]
    pub fn into_tool_response(self) -> Option<ToolResponse> {
        if let Self::ToolResponse(tool_response) = self {
            Some(tool_response)
        } else {
            None
        }
    }

    /// Returns a reference to the inner data if present.
    #[must_use]
    pub const fn as_data(&self) -> Option<&Data> {
        if let Self::Data(data) = self {
            Some(data)
        } else {
            None
        }
    }

    /// Consumes the part and returns the inner data.
    #[must_use]
    pub fn into_data(self) -> Option<Data> {
        if let Self::Data(data) = self {
            Some(data)
        } else {
            None
        }
    }

    /// Converts this content part into an A2A protocol message part when possible.
    #[must_use]
    pub fn into_a2a_part(self) -> Option<a2a_types::Part> {
        match self {
            Self::Text(text) => Some(a2a_types::Part {
                content: Some(a2a_types::part::Content::Text(text)),
                metadata: None,
                filename: String::new(),
                media_type: "text/plain".to_string(),
            }),
            Self::Data(data) => match (&*data.content_type, &data.source) {
                ("application/json", DataSource::Base64(encoded)) => {
                    if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(encoded) {
                        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                            if let Ok(proto_val) =
                                serde_json::from_value::<pbjson_types::Value>(value)
                            {
                                return Some(a2a_types::Part {
                                    content: Some(a2a_types::part::Content::Data(proto_val)),
                                    metadata: None,
                                    filename: String::new(),
                                    media_type: "application/json".to_string(),
                                });
                            }
                        }
                    }
                    Some(a2a_types::Part {
                        content: Some(a2a_types::part::Content::Raw(
                            base64::engine::general_purpose::STANDARD
                                .decode(encoded)
                                .unwrap_or_default(),
                        )),
                        metadata: None,
                        filename: data.name.unwrap_or_default(),
                        media_type: data.content_type,
                    })
                }
                (_, DataSource::Base64(encoded)) => Some(a2a_types::Part {
                    content: Some(a2a_types::part::Content::Raw(
                        base64::engine::general_purpose::STANDARD
                            .decode(encoded)
                            .unwrap_or_default(),
                    )),
                    metadata: None,
                    filename: data.name.unwrap_or_default(),
                    media_type: data.content_type,
                }),
                (_, DataSource::Uri(uri)) => Some(a2a_types::Part {
                    content: Some(a2a_types::part::Content::Url(uri.clone())),
                    metadata: None,
                    filename: data.name.unwrap_or_default(),
                    media_type: data.content_type,
                }),
            },
            Self::ToolCall(_) | Self::ToolResponse(_) => None,
        }
    }
}

impl Data {
    /// Creates a new data block with validation.
    ///
    /// # Errors
    ///
    /// Returns an error if the MIME type format is invalid or data source is empty.
    pub fn new(
        content_type: impl Into<String>,
        source: DataSource,
        name: Option<String>,
    ) -> Result<Self, AgentError> {
        let content_type = content_type.into();

        // Validate MIME type format
        if content_type.is_empty() || !content_type.contains('/') {
            return Err(AgentError::InvalidMimeType(
                "MIME type must be in format 'type/subtype'".to_string(),
            ));
        }

        // Validate source
        match &source {
            DataSource::Base64(base64) => {
                if base64.is_empty() {
                    return Err(AgentError::InvalidBase64(
                        "Base64 string cannot be empty".to_string(),
                    ));
                }
                if !base64.chars().all(|c| {
                    c.is_ascii_alphanumeric()
                        || c == '+'
                        || c == '/'
                        || c == '='
                        || c.is_whitespace()
                }) {
                    return Err(AgentError::InvalidBase64(
                        "Base64 string contains invalid characters".to_string(),
                    ));
                }
            }
            DataSource::Uri(uri) => {
                if uri.is_empty() {
                    return Err(AgentError::InvalidUri("URI cannot be empty".to_string()));
                }
            }
        }

        Ok(Self {
            content_type,
            source,
            name,
        })
    }

    /// Creates a new data block without validation.
    pub fn new_unchecked(
        content_type: impl Into<String>,
        source: DataSource,
        name: Option<String>,
    ) -> Self {
        Self {
            name,
            content_type: content_type.into(),
            source,
        }
    }
}

// ============================================================================
// A2A Conversions
// ============================================================================

impl From<a2a_types::Part> for ContentPart {
    fn from(part: a2a_types::Part) -> Self {
        let filename = if part.filename.is_empty() {
            None
        } else {
            Some(part.filename)
        };
        let media_type = if part.media_type.is_empty() {
            None
        } else {
            Some(part.media_type)
        };

        match part.content {
            Some(a2a_types::part::Content::Text(text)) => Self::Text(text),
            Some(a2a_types::part::Content::Data(data)) => {
                // Serialize the proto Value back to JSON bytes, then base64-encode.
                let json_bytes = serde_json::to_vec(&data).unwrap_or_else(|_| b"null".to_vec());
                let base64_data = base64::engine::general_purpose::STANDARD.encode(json_bytes);
                Self::Data(Data::new_unchecked(
                    media_type.unwrap_or_else(|| "application/json".to_string()),
                    DataSource::Base64(base64_data),
                    filename,
                ))
            }
            Some(a2a_types::part::Content::Raw(raw)) => {
                let base64_data = base64::engine::general_purpose::STANDARD.encode(&raw);
                let content_type =
                    media_type.unwrap_or_else(|| "application/octet-stream".to_string());
                Self::Data(Data::new_unchecked(
                    content_type,
                    DataSource::Base64(base64_data),
                    filename,
                ))
            }
            Some(a2a_types::part::Content::Url(uri)) => {
                let content_type =
                    media_type.unwrap_or_else(|| "application/octet-stream".to_string());
                Self::Data(Data::new_unchecked(
                    content_type,
                    DataSource::Uri(uri),
                    filename,
                ))
            }
            None => Self::Text(String::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::AgentError;

    #[test]
    fn data_from_base64_validates_inputs() {
        let valid = ContentPart::from_base64("text/plain", "SGVsbG8=", None).unwrap();
        assert!(matches!(valid, ContentPart::Data(_)));

        let err = ContentPart::from_base64("invalid", "", None).unwrap_err();
        match err {
            AgentError::InvalidMimeType(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn data_from_uri_rejects_empty_uri() {
        let err = ContentPart::from_uri("text/plain", "", None).unwrap_err();
        match err {
            AgentError::InvalidUri(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
