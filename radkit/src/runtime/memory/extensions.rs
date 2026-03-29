//! Extension traits for domain-specific ingestion.
//!
//! This module provides extension traits that transform domain objects into
//! [`MemoryContent`](super::MemoryContent) and delegate to the core
//! [`MemoryService`](super::MemoryService) trait:
//!
//! - [`MemoryServiceConversationExt`]: For ingesting completed conversations
//! - [`MemoryServiceDocumentExt`]: For ingesting and managing documents (RAG)

use std::collections::HashMap;

use crate::errors::AgentResult;
use crate::runtime::context::AuthContext;

use super::{ContentSource, MemoryContent, MemoryService};

/// A completed conversation to be ingested into long-term memory.
///
/// Note: Named "`CompletedConversation`" to distinguish from A2A protocol's
/// conversation concept. This represents a finished task's messages being
/// archived for future semantic search.
#[derive(Debug, Clone)]
pub struct CompletedConversation {
    /// The context ID this conversation belongs to.
    pub context_id: String,
    /// Messages from the conversation.
    pub messages: Vec<CompletedMessage>,
}

/// A message from a completed conversation.
#[derive(Debug, Clone)]
pub struct CompletedMessage {
    /// Unique identifier for this message.
    pub message_id: String,
    /// Role of the message sender ("user" or "agent").
    pub role: String,
    /// Text content of the message.
    pub text: String,
    /// Optional timestamp in RFC 3339 format.
    pub timestamp: Option<String>,
}

impl CompletedConversation {
    /// Converts to memory contents for ingestion.
    #[must_use]
    pub fn into_memory_contents(self) -> Vec<MemoryContent> {
        self.messages
            .into_iter()
            .map(|msg| MemoryContent {
                text: msg.text,
                source: ContentSource::PastConversation {
                    context_id: self.context_id.clone(),
                    message_id: msg.message_id,
                    role: msg.role,
                },
                metadata: msg
                    .timestamp
                    .map(|ts| std::iter::once(("timestamp".to_string(), ts.into())).collect())
                    .unwrap_or_default(),
            })
            .collect()
    }
}

/// A document to be ingested (RAG).
#[derive(Debug, Clone)]
pub struct Document {
    /// Unique identifier for this document.
    pub id: String,
    /// Human-readable name of the document.
    pub name: String,
    /// Full text content of the document.
    pub content: String,
    /// Additional metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Document {
    /// Create a new document.
    pub fn new(id: impl Into<String>, name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            content: content.into(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the document.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Chunks the document and converts to memory contents.
    #[must_use]
    pub fn into_memory_contents(self, chunk_size: usize) -> Vec<MemoryContent> {
        let chunks = chunk_text(&self.content, chunk_size);
        let total_chunks = chunks.len();

        chunks
            .into_iter()
            .enumerate()
            .map(|(i, text)| MemoryContent {
                text,
                source: ContentSource::Document {
                    document_id: self.id.clone(),
                    name: self.name.clone(),
                    chunk_index: i,
                    total_chunks,
                },
                metadata: self.metadata.clone(),
            })
            .collect()
    }
}

/// Sentence-aware text chunking.
///
/// Splits text into chunks of approximately `chunk_size` characters,
/// respecting sentence boundaries where possible.
#[must_use]
pub fn chunk_text(text: &str, chunk_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for sentence in text.split_inclusive(['.', '!', '?']) {
        if current.len() + sentence.len() > chunk_size && !current.is_empty() {
            chunks.push(std::mem::take(&mut current));
        }
        current.push_str(sentence);
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    if chunks.is_empty() && !text.is_empty() {
        chunks.push(text.to_string());
    }

    chunks
}

/// Extension trait for conversation ingestion.
#[cfg_attr(
    all(target_os = "wasi", target_env = "p1"),
    async_trait::async_trait(?Send)
)]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
pub trait MemoryServiceConversationExt: MemoryService {
    /// Ingest a completed conversation into long-term memory.
    ///
    /// This is typically called automatically when a task completes,
    /// but can also be called manually for specific conversations.
    async fn add_conversation(
        &self,
        auth_ctx: &AuthContext,
        conversation: CompletedConversation,
    ) -> AgentResult<Vec<String>> {
        let contents = conversation.into_memory_contents();
        self.add_batch(auth_ctx, contents).await
    }
}

impl<T: MemoryService + ?Sized> MemoryServiceConversationExt for T {}

/// Extension trait for document ingestion (RAG).
#[cfg_attr(
    all(target_os = "wasi", target_env = "p1"),
    async_trait::async_trait(?Send)
)]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
pub trait MemoryServiceDocumentExt: MemoryService {
    /// Ingest a document into memory.
    ///
    /// Automatically chunks the document and stores each chunk.
    /// Uses upsert semantics - re-adding a document replaces existing chunks.
    /// Any chunks from a previous version beyond the new chunk count are deleted.
    ///
    /// # Arguments
    ///
    /// * `auth_ctx` - Authentication context for namespacing
    /// * `document` - The document to ingest
    /// * `chunk_size` - Optional chunk size (default: 1000 characters)
    async fn add_document(
        &self,
        auth_ctx: &AuthContext,
        document: Document,
        chunk_size: Option<usize>,
    ) -> AgentResult<Vec<String>> {
        let document_id = document.id.clone();
        let contents = document.into_memory_contents(chunk_size.unwrap_or(1000));
        let new_chunk_count = contents.len();

        // Add the new chunks (will upsert existing ones with same IDs)
        let ids = self.add_batch(auth_ctx, contents).await?;

        // Delete any stale chunks beyond the new chunk count
        // This handles the case where a re-ingested document has fewer chunks
        self.delete_stale_chunks(auth_ctx, &document_id, new_chunk_count)
            .await?;

        Ok(ids)
    }

    /// Delete chunks starting from a given index.
    ///
    /// Used internally to clean up stale chunks after re-ingestion.
    async fn delete_stale_chunks(
        &self,
        auth_ctx: &AuthContext,
        document_id: &str,
        start_index: usize,
    ) -> AgentResult<usize> {
        const MAX_STALE_CHUNKS: usize = 1000;
        const CONSECUTIVE_MISS_THRESHOLD: usize = 10;

        let mut deleted = 0;
        let mut consecutive_misses = 0;

        for i in start_index..(start_index + MAX_STALE_CHUNKS) {
            let chunk_id = format!("doc:{document_id}:chunk-{i}");
            if self.delete(auth_ctx, &chunk_id).await? {
                deleted += 1;
                consecutive_misses = 0;
            } else {
                consecutive_misses += 1;
                // Stop after threshold consecutive misses
                if consecutive_misses >= CONSECUTIVE_MISS_THRESHOLD {
                    break;
                }
            }
        }

        Ok(deleted)
    }

    /// Delete all chunks of a document.
    ///
    /// Note: This uses the `document_id` to construct chunk IDs directly rather
    /// than searching, which avoids issues with empty queries on vector backends.
    async fn delete_document(
        &self,
        auth_ctx: &AuthContext,
        document_id: &str,
    ) -> AgentResult<usize> {
        // Construct IDs based on the known ID pattern: doc:{document_id}:chunk-{n}
        // We try to delete chunks 0 through a reasonable max, counting successes.
        const MAX_CHUNKS: usize = 10000;
        const CONSECUTIVE_MISS_THRESHOLD: usize = 10;

        let mut deleted = 0;
        let mut consecutive_misses = 0;

        for i in 0..MAX_CHUNKS {
            let chunk_id = format!("doc:{document_id}:chunk-{i}");
            if self.delete(auth_ctx, &chunk_id).await? {
                deleted += 1;
                consecutive_misses = 0;
            } else {
                consecutive_misses += 1;
                // Stop after threshold consecutive misses
                // This prevents hammering the backend when doc doesn't exist
                if consecutive_misses >= CONSECUTIVE_MISS_THRESHOLD {
                    break;
                }
            }
        }

        Ok(deleted)
    }
}

impl<T: MemoryService + ?Sized> MemoryServiceDocumentExt for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_text_splits_by_sentences() {
        let text = "First sentence. Second sentence. Third sentence.";
        let chunks = chunk_text(text, 30);
        // 15 + 17 = 32 > 30, so first chunk pushed; 17 + 16 = 33 > 30, so second pushed
        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].ends_with('.'));
        assert!(chunks[1].ends_with('.'));
        assert!(chunks[2].ends_with('.'));
    }

    #[test]
    fn chunk_text_handles_no_sentences() {
        let text = "No sentence delimiters here";
        let chunks = chunk_text(text, 10);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn chunk_text_handles_empty() {
        let chunks = chunk_text("", 100);
        assert!(chunks.is_empty());
    }

    #[test]
    fn document_into_memory_contents() {
        let doc = Document::new("test-doc", "Test Document", "First. Second. Third.");
        let contents = doc.into_memory_contents(10);

        assert!(!contents.is_empty());
        for content in &contents {
            if let ContentSource::Document {
                document_id, name, ..
            } = &content.source
            {
                assert_eq!(document_id, "test-doc");
                assert_eq!(name, "Test Document");
            } else {
                panic!("Expected Document source");
            }
        }
    }

    #[test]
    fn completed_conversation_into_memory_contents() {
        let conv = CompletedConversation {
            context_id: "ctx-123".to_string(),
            messages: vec![
                CompletedMessage {
                    message_id: "msg-1".to_string(),
                    role: "user".to_string(),
                    text: "Hello".to_string(),
                    timestamp: Some("2024-01-01T00:00:00Z".to_string()),
                },
                CompletedMessage {
                    message_id: "msg-2".to_string(),
                    role: "agent".to_string(),
                    text: "Hi there!".to_string(),
                    timestamp: None,
                },
            ],
        };

        let contents = conv.into_memory_contents();
        assert_eq!(contents.len(), 2);

        if let ContentSource::PastConversation {
            context_id,
            message_id,
            role,
        } = &contents[0].source
        {
            assert_eq!(context_id, "ctx-123");
            assert_eq!(message_id, "msg-1");
            assert_eq!(role, "user");
        } else {
            panic!("Expected PastConversation source");
        }
    }
}

#[cfg(all(test, feature = "runtime"))]
mod async_tests {
    use super::*;
    use crate::runtime::memory::{InMemoryMemoryService, SearchOptions};

    fn test_auth() -> AuthContext {
        AuthContext {
            app_name: "test-app".to_string(),
            user_name: "test-user".to_string(),
        }
    }

    /// Test that re-ingesting a document with fewer chunks removes stale chunks.
    #[tokio::test]
    async fn add_document_removes_stale_chunks_on_reingest() {
        let memory = InMemoryMemoryService::new();
        let auth = test_auth();

        // First: ingest a document that will produce multiple chunks
        let doc_v1 = Document::new(
            "shrinking-doc",
            "Shrinking Document",
            "First sentence is here. Second sentence is here. Third sentence is here. Fourth sentence is here.",
        );
        let ids_v1 = memory.add_document(&auth, doc_v1, Some(30)).await.unwrap();
        assert!(
            ids_v1.len() >= 3,
            "Should have at least 3 chunks, got {} with IDs: {:?}",
            ids_v1.len(),
            ids_v1
        );
        let original_chunk_count = ids_v1.len();

        // Verify all chunks are searchable (use min_score to filter non-matches)
        let results_v1 = memory
            .search(
                &auth,
                "sentence",
                SearchOptions::documents_only().with_min_score(0.1),
            )
            .await
            .unwrap();
        assert_eq!(
            results_v1.len(),
            original_chunk_count,
            "All chunks should contain 'sentence'"
        );

        // Second: re-ingest with smaller content (fewer chunks)
        let doc_v2 = Document::new("shrinking-doc", "Shrinking Document", "Only one chunk now.");
        let ids_v2 = memory
            .add_document(&auth, doc_v2, Some(1000))
            .await
            .unwrap();
        assert_eq!(ids_v2.len(), 1, "Should have only 1 chunk now");

        // Verify stale chunks were removed - old content should not be found
        let results_v2 = memory
            .search(
                &auth,
                "sentence",
                SearchOptions::documents_only().with_min_score(0.1),
            )
            .await
            .unwrap();

        assert!(
            results_v2.is_empty(),
            "Old content should be gone after re-ingest, but found {} results",
            results_v2.len()
        );

        // Verify new content is searchable
        let results_new = memory
            .search(
                &auth,
                "Only one chunk",
                SearchOptions::documents_only().with_min_score(0.1),
            )
            .await
            .unwrap();
        assert_eq!(results_new.len(), 1, "New content should be searchable");
    }

    /// Test that `delete_document` stops quickly when the document doesn't exist.
    #[tokio::test]
    async fn delete_document_stops_early_for_missing_doc() {
        let memory = InMemoryMemoryService::new();
        let auth = test_auth();

        // Delete a document that was never added
        // This should stop after ~10 consecutive misses, not 10,000 iterations
        let deleted = memory
            .delete_document(&auth, "nonexistent-document")
            .await
            .unwrap();

        assert_eq!(deleted, 0, "Should not delete anything");
        // The test passing quickly (not timing out) proves the fix works
    }

    /// Test that `delete_document` handles sparse chunk IDs correctly.
    #[tokio::test]
    async fn delete_document_handles_normal_case() {
        let memory = InMemoryMemoryService::new();
        let auth = test_auth();

        // Add a document
        let doc = Document::new(
            "to-delete",
            "Document to Delete",
            "First part. Second part. Third part.",
        );
        let ids = memory.add_document(&auth, doc, Some(15)).await.unwrap();
        let chunk_count = ids.len();
        assert!(chunk_count >= 2, "Should have multiple chunks");

        // Verify it exists
        let before = memory
            .search(&auth, "part", SearchOptions::documents_only())
            .await
            .unwrap();
        assert_eq!(before.len(), chunk_count);

        // Delete it
        let deleted = memory.delete_document(&auth, "to-delete").await.unwrap();
        assert_eq!(deleted, chunk_count, "Should delete all chunks");

        // Verify it's gone
        let after = memory
            .search(&auth, "part", SearchOptions::documents_only())
            .await
            .unwrap();
        assert!(after.is_empty(), "All chunks should be deleted");
    }
}
