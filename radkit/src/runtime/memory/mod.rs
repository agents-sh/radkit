//! Long-term semantic memory service for agents.
//!
//! This module provides semantic search over past conversations, user facts, and documents.
//! Unlike the short-term context handled by `TaskManager`, the `MemoryService` persists
//! across sessions and enables semantic (meaning-based) retrieval.
//!
//! # Overview
//!
//! - [`MemoryService`]: Core trait for semantic memory operations
//! - [`OwnedHistory`]: Facade for accessing past conversations and user facts
//! - [`OwnedKnowledge`]: Facade for accessing documents and external sources
//! - [`Embedder`]: Trait for generating text embeddings (required by vector backends)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     CONCEPTUAL LAYER                         │
//! │  ┌─────────────────┐       ┌─────────────────┐              │
//! │  │ OwnedHistory    │       │ OwnedKnowledge  │              │
//! │  │ recall()        │       │ search()        │              │
//! │  │ save_fact()     │       │                 │              │
//! │  └────────┬────────┘       └────────┬────────┘              │
//! │           └──────────┬──────────────┘                       │
//! │                      ▼                                       │
//! │              ┌───────────────┐                               │
//! │              │ MemoryService │  Core trait                   │
//! │              └───────┬───────┘                               │
//! │                      │                                       │
//! │  ┌───────────────────┼───────────────────┐                   │
//! │  ▼                   ▼                   ▼                   │
//! │ InMemory           Qdrant            Custom                  │
//! │ (keyword)         (vector)          backends                 │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Multi-tenancy
//!
//! All operations are namespaced by [`AuthContext`](crate::runtime::context::AuthContext),
//! ensuring data isolation between users and applications.
//!
//! # Examples
//!
//! ```ignore
//! use radkit::runtime::memory::{MemoryService, InMemoryMemoryService, SearchOptions};
//! use radkit::runtime::context::AuthContext;
//!
//! let memory = InMemoryMemoryService::new();
//! let auth = AuthContext {
//!     app_name: "my-app".to_string(),
//!     user_name: "alice".to_string(),
//! };
//!
//! // Search past conversations
//! let results = memory.search(&auth, "dark mode preferences", SearchOptions::history_only()).await?;
//! ```

pub mod backends;
mod embedder;
mod extensions;
mod facades;
mod types;

pub use backends::InMemoryMemoryService;
pub use embedder::Embedder;
pub use extensions::{
    chunk_text, CompletedConversation, CompletedMessage, Document, MemoryServiceConversationExt,
    MemoryServiceDocumentExt,
};
pub use facades::{OwnedHistory, OwnedKnowledge};
pub use types::{
    ContentSource, MemoryContent, MemoryEntry, SearchOptions, SourceCategory, SourceType,
};

use crate::compat::{MaybeSend, MaybeSync};
use crate::errors::AgentResult;
use crate::runtime::context::AuthContext;

/// Long-term semantic memory store.
///
/// Provides semantic search over past conversations, user facts, and documents.
/// For current conversation context, use `TaskManager` instead.
///
/// # Multi-tenancy
///
/// All operations are namespaced by the [`AuthContext`], ensuring data isolation
/// between different users and applications.
///
/// # Implementations
///
/// - [`InMemoryMemoryService`]: Development backend using keyword matching (no embeddings)
/// - `QdrantMemoryService`: Production backend using vector search (requires `memory-qdrant` feature)
#[cfg_attr(
    all(target_os = "wasi", target_env = "p1"),
    async_trait::async_trait(?Send)
)]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
pub trait MemoryService: MaybeSend + MaybeSync {
    /// Add content to memory.
    ///
    /// Returns the assigned ID. Uses upsert semantics - if content with the
    /// same ID already exists, it will be replaced.
    ///
    /// # Arguments
    ///
    /// * `auth_ctx` - Authentication context for namespacing
    /// * `content` - Content to add to memory
    async fn add(&self, auth_ctx: &AuthContext, content: MemoryContent) -> AgentResult<String>;

    /// Add multiple contents in batch.
    ///
    /// More efficient than calling `add` repeatedly.
    /// Returns the assigned IDs in the same order as input.
    async fn add_batch(
        &self,
        auth_ctx: &AuthContext,
        contents: Vec<MemoryContent>,
    ) -> AgentResult<Vec<String>>;

    /// Search memory for relevant content.
    ///
    /// Returns entries ordered by relevance score (highest first).
    ///
    /// # Empty Query Behavior
    ///
    /// - **InMemory backend**: Returns all entries (filtered by source_type) with score 1.0
    /// - **Vector backends**: Behavior depends on the embedding model
    async fn search(
        &self,
        auth_ctx: &AuthContext,
        query: &str,
        options: SearchOptions,
    ) -> AgentResult<Vec<MemoryEntry>>;

    /// Delete a specific entry by ID.
    ///
    /// Returns `true` if the entry existed and was deleted.
    async fn delete(&self, auth_ctx: &AuthContext, id: &str) -> AgentResult<bool>;

    /// Delete multiple entries by IDs.
    ///
    /// Returns the number of entries that were deleted.
    async fn delete_batch(&self, auth_ctx: &AuthContext, ids: &[String]) -> AgentResult<usize>;
}
