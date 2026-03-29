//! Facades for convenient memory access.
//!
//! This module provides [`OwnedHistory`] and [`OwnedKnowledge`] facades that wrap the
//! [`MemoryService`](super::MemoryService) trait with filtered access:
//!
//! - [`OwnedHistory`]: Access to past conversations and user facts
//! - [`OwnedKnowledge`]: Access to documents and external sources
//!
//! Both facades use the same underlying `MemoryService` but filter by source type.

use std::collections::HashMap;
use std::sync::Arc;

use crate::errors::AgentResult;
use crate::runtime::context::AuthContext;

use super::{ContentSource, MemoryContent, MemoryEntry, MemoryService, SearchOptions};

/// History facade for accessing past conversations and user facts.
///
/// Wraps a `MemoryService` and filters searches to `PastConversation` and `UserFact` sources.
/// Use this to recall what was discussed in previous sessions or retrieve user preferences.
///
/// # Example
///
/// ```ignore
/// // From runtime
/// let history = runtime.history();
/// let memories = history.recall(&auth, "dark mode preferences", 5).await?;
/// ```
pub struct OwnedHistory {
    service: Arc<dyn MemoryService>,
}

impl OwnedHistory {
    /// Create a new `OwnedHistory` facade wrapping the given memory service.
    pub fn new(service: Arc<dyn MemoryService>) -> Self {
        Self { service }
    }

    /// Search past conversations and user facts.
    ///
    /// # Arguments
    ///
    /// * `auth` - Authentication context for namespacing
    /// * `query` - Natural language query describing what you're looking for
    /// * `limit` - Maximum number of results to return
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying memory service fails to search.
    pub async fn recall(
        &self,
        auth: &AuthContext,
        query: &str,
        limit: usize,
    ) -> AgentResult<Vec<MemoryEntry>> {
        self.service
            .search(auth, query, SearchOptions::history_only().with_limit(limit))
            .await
    }

    /// Save a user fact to long-term memory.
    ///
    /// Use this to remember user preferences, facts, or insights
    /// that should be recalled in future conversations.
    ///
    /// # Arguments
    ///
    /// * `auth` - Authentication context for namespacing
    /// * `text` - The fact to remember (be specific and include context)
    /// * `category` - Optional category for organization (e.g., "preferences", "facts")
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying memory service fails to store the fact.
    pub async fn save_fact(
        &self,
        auth: &AuthContext,
        text: String,
        category: Option<String>,
    ) -> AgentResult<String> {
        self.service
            .add(
                auth,
                MemoryContent {
                    text,
                    source: ContentSource::UserFact { category },
                    metadata: HashMap::new(),
                },
            )
            .await
    }
}

/// Knowledge facade for accessing documents and external sources.
///
/// Wraps a `MemoryService` and filters searches to `Document` and `External` sources.
/// Use this to find answers from uploaded documents, manuals, or reference material.
///
/// # Example
///
/// ```ignore
/// // From runtime
/// let knowledge = runtime.knowledge();
/// let results = knowledge.search(&auth, "vacation policy", 5).await?;
/// ```
pub struct OwnedKnowledge {
    service: Arc<dyn MemoryService>,
}

impl OwnedKnowledge {
    /// Create a new `OwnedKnowledge` facade wrapping the given memory service.
    pub fn new(service: Arc<dyn MemoryService>) -> Self {
        Self { service }
    }

    /// Search documents and external sources.
    ///
    /// # Arguments
    ///
    /// * `auth` - Authentication context for namespacing
    /// * `query` - Natural language query describing what you're looking for
    /// * `limit` - Maximum number of results to return
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying memory service fails to search.
    pub async fn search(
        &self,
        auth: &AuthContext,
        query: &str,
        limit: usize,
    ) -> AgentResult<Vec<MemoryEntry>> {
        self.service
            .search(
                auth,
                query,
                SearchOptions::knowledge_only().with_limit(limit),
            )
            .await
    }
}
