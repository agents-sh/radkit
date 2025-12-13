//! Embedder trait for text-to-vector conversion.
//!
//! This module defines the [`Embedder`] trait for generating text embeddings.
//! Vector-based memory backends (like Qdrant) require an embedder to convert
//! text into dense vectors for semantic similarity search.
//!
//! # Built-in Embedders
//!
//! - `OpenAIEmbedder`: Uses OpenAI's embedding API (requires `memory-qdrant` feature)
//!
//! # Custom Embedders
//!
//! You can implement the `Embedder` trait for any embedding provider:
//!
//! ```ignore
//! use radkit::runtime::memory::Embedder;
//! use radkit::errors::AgentResult;
//!
//! pub struct LocalEmbedder {
//!     model: SomeLocalModel,
//! }
//!
//! #[async_trait::async_trait]
//! impl Embedder for LocalEmbedder {
//!     async fn embed(&self, text: &str) -> AgentResult<Vec<f32>> {
//!         Ok(self.model.encode(text))
//!     }
//!
//!     fn dimension(&self) -> usize {
//!         384  // e.g., all-MiniLM-L6-v2
//!     }
//!
//!     fn model_id(&self) -> &str {
//!         "local:all-MiniLM-L6-v2"
//!     }
//! }
//! ```

use crate::compat::{MaybeSend, MaybeSync};
use crate::errors::AgentResult;

/// Trait for generating text embeddings.
///
/// Embedders convert text into dense vectors that capture semantic meaning.
/// These vectors can then be used for similarity search in vector databases.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` (via `MaybeSend + MaybeSync`) to
/// support concurrent access across async tasks.
#[cfg_attr(
    all(target_os = "wasi", target_env = "p1"),
    async_trait::async_trait(?Send)
)]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
pub trait Embedder: MaybeSend + MaybeSync {
    /// Generate embedding for text.
    ///
    /// Returns a dense vector of floating-point values representing the
    /// semantic content of the text.
    async fn embed(&self, text: &str) -> AgentResult<Vec<f32>>;

    /// Generate embeddings for multiple texts (batch).
    ///
    /// Default implementation calls `embed` in sequence.
    /// Override for providers that support batch APIs for better performance.
    async fn embed_batch(&self, texts: &[&str]) -> AgentResult<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    /// Embedding dimension (e.g., 1536 for OpenAI text-embedding-3-small).
    ///
    /// This is used to configure vector database collections.
    fn dimension(&self) -> usize;

    /// Model identifier for versioning (e.g., "openai:text-embedding-3-small").
    ///
    /// Stored in metadata to detect when the embedding model changes,
    /// which may require re-embedding existing content.
    fn model_id(&self) -> &str;
}
