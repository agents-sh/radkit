//! Types for the semantic memory service.
//!
//! This module defines the core types used by the `MemoryService` trait:
//! - [`MemoryContent`]: Content to be added to memory
//! - [`MemoryEntry`]: A memory entry returned from search
//! - [`ContentSource`]: Where the content originated from
//! - [`SourceType`]: Source type for filtering searches
//! - [`SourceCategory`]: High-level category (History vs Knowledge)
//! - [`SearchOptions`]: Options for search operations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Content to be added to memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContent {
    /// The text content (will be embedded for vector stores).
    pub text: String,

    /// Source of this content.
    pub source: ContentSource,

    /// Additional metadata for filtering.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Where the content originated from.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentSource {
    /// From a past conversation (searchable history).
    PastConversation {
        context_id: String,
        message_id: String,
        role: String, // "user" | "agent"
    },

    /// User-provided facts/preferences (agent-learned).
    UserFact { category: Option<String> },

    /// From an uploaded document (RAG).
    Document {
        document_id: String,
        name: String,
        chunk_index: usize,
        total_chunks: usize,
    },

    /// External data source.
    External {
        source_name: String,
        source_id: Option<String>,
    },
}

impl ContentSource {
    /// Generates a unique ID for this content.
    #[must_use]
    pub fn generate_id(&self) -> String {
        match self {
            Self::PastConversation {
                context_id,
                message_id,
                ..
            } => {
                format!("conv:{context_id}:{message_id}")
            }
            Self::UserFact { category } => {
                let uuid = uuid::Uuid::new_v4();
                match category {
                    Some(cat) => format!("fact:{cat}:{uuid}"),
                    None => format!("fact:{uuid}"),
                }
            }
            Self::Document {
                document_id,
                chunk_index,
                ..
            } => {
                format!("doc:{document_id}:chunk-{chunk_index}")
            }
            Self::External {
                source_name,
                source_id,
            } => {
                let id = source_id
                    .clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                format!("ext:{source_name}:{id}")
            }
        }
    }

    /// Returns the source type for filtering.
    #[must_use]
    pub fn source_type(&self) -> SourceType {
        match self {
            Self::PastConversation { .. } => SourceType::PastConversation,
            Self::UserFact { .. } => SourceType::UserFact,
            Self::Document { .. } => SourceType::Document,
            Self::External { .. } => SourceType::External,
        }
    }

    /// Returns the category (History or Knowledge).
    #[must_use]
    pub fn category(&self) -> SourceCategory {
        match self {
            Self::PastConversation { .. } | Self::UserFact { .. } => SourceCategory::History,
            Self::Document { .. } | Self::External { .. } => SourceCategory::Knowledge,
        }
    }
}

/// High-level category for filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceCategory {
    /// Past conversations and user facts.
    History,
    /// Documents and external sources.
    Knowledge,
}

/// Source type for filtering searches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    PastConversation,
    UserFact,
    Document,
    External,
}

/// A memory entry returned from search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier.
    pub id: String,

    /// The text content.
    pub text: String,

    /// Source information.
    pub source: ContentSource,

    /// Relevance score (0.0 to 1.0, higher = more relevant).
    pub score: f32,

    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Options for search operations.
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    /// Maximum number of results (default: 10).
    pub limit: Option<usize>,

    /// Minimum relevance score (0.0 to 1.0).
    pub min_score: Option<f32>,

    /// Filter by source types.
    pub source_types: Option<Vec<SourceType>>,

    /// Filter by metadata key-value pairs.
    pub metadata_filter: Option<HashMap<String, serde_json::Value>>,
}

impl SearchOptions {
    /// Set maximum number of results.
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set minimum relevance score.
    #[must_use]
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = Some(min_score);
        self
    }

    /// Filter by source types.
    #[must_use]
    pub fn with_source_types(mut self, types: Vec<SourceType>) -> Self {
        self.source_types = Some(types);
        self
    }

    /// Filter to History sources only (PastConversation + UserFact).
    #[must_use]
    pub fn history_only() -> Self {
        Self::default().with_source_types(vec![SourceType::PastConversation, SourceType::UserFact])
    }

    /// Filter to Knowledge sources only (Document + External).
    #[must_use]
    pub fn knowledge_only() -> Self {
        Self::default().with_source_types(vec![SourceType::Document, SourceType::External])
    }

    /// Filter to conversations only.
    #[must_use]
    pub fn conversations_only() -> Self {
        Self::default().with_source_types(vec![SourceType::PastConversation])
    }

    /// Filter to documents only.
    #[must_use]
    pub fn documents_only() -> Self {
        Self::default().with_source_types(vec![SourceType::Document])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_source_generates_conversation_id() {
        let source = ContentSource::PastConversation {
            context_id: "ctx-123".to_string(),
            message_id: "msg-456".to_string(),
            role: "user".to_string(),
        };
        let id = source.generate_id();
        assert_eq!(id, "conv:ctx-123:msg-456");
        assert_eq!(source.source_type(), SourceType::PastConversation);
        assert_eq!(source.category(), SourceCategory::History);
    }

    #[test]
    fn content_source_generates_user_fact_id() {
        let source = ContentSource::UserFact {
            category: Some("preferences".to_string()),
        };
        let id = source.generate_id();
        assert!(id.starts_with("fact:preferences:"));
        assert_eq!(source.source_type(), SourceType::UserFact);
        assert_eq!(source.category(), SourceCategory::History);
    }

    #[test]
    fn content_source_generates_document_id() {
        let source = ContentSource::Document {
            document_id: "handbook".to_string(),
            name: "Employee Handbook".to_string(),
            chunk_index: 5,
            total_chunks: 10,
        };
        let id = source.generate_id();
        assert_eq!(id, "doc:handbook:chunk-5");
        assert_eq!(source.source_type(), SourceType::Document);
        assert_eq!(source.category(), SourceCategory::Knowledge);
    }

    #[test]
    fn content_source_generates_external_id() {
        let source = ContentSource::External {
            source_name: "api".to_string(),
            source_id: Some("xyz".to_string()),
        };
        let id = source.generate_id();
        assert_eq!(id, "ext:api:xyz");
        assert_eq!(source.source_type(), SourceType::External);
        assert_eq!(source.category(), SourceCategory::Knowledge);
    }

    #[test]
    fn search_options_history_only() {
        let opts = SearchOptions::history_only();
        let types = opts.source_types.unwrap();
        assert!(types.contains(&SourceType::PastConversation));
        assert!(types.contains(&SourceType::UserFact));
        assert!(!types.contains(&SourceType::Document));
    }

    #[test]
    fn search_options_knowledge_only() {
        let opts = SearchOptions::knowledge_only();
        let types = opts.source_types.unwrap();
        assert!(types.contains(&SourceType::Document));
        assert!(types.contains(&SourceType::External));
        assert!(!types.contains(&SourceType::PastConversation));
    }
}
