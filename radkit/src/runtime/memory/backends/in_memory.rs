//! In-memory implementation of the `MemoryService` trait.
//!
//! This module provides both native (thread-safe) and WASM (single-threaded)
//! implementations of the memory service using keyword matching.
//!
//! # Usage
//!
//! This backend is intended for **development and testing only**. It does not
//! persist data and uses simple keyword matching instead of semantic search.
//!
//! For production use cases requiring semantic search, use a vector backend
//! like `QdrantMemoryService`.
//!
//! # Platform Differences
//!
//! - **Native**: Uses `DashMap` for thread-safe concurrent access
//! - **WASM**: Uses `RefCell<HashMap>` for single-threaded interior mutability

use std::collections::{HashMap, HashSet};

use crate::errors::AgentResult;
use crate::runtime::context::AuthContext;
use crate::runtime::memory::{
    ContentSource, MemoryContent, MemoryEntry, MemoryService, SearchOptions,
};

/// Stored entry in the in-memory backend.
#[derive(Debug, Clone)]
struct StoredEntry {
    id: String,
    text: String,
    source: ContentSource,
    metadata: HashMap<String, serde_json::Value>,
}

// ============================================================================
// Native Implementation (Thread-Safe)
// ============================================================================

#[cfg(not(all(target_os = "wasi", target_env = "p1")))]
mod native {
    use super::*;
    use dashmap::DashMap;
    use std::sync::Arc;

    /// In-memory implementation using keyword matching.
    ///
    /// For development and testing only. No embeddings, no persistence.
    ///
    /// # Platform Notes
    ///
    /// Uses `DashMap` for thread-safe concurrent access.
    #[derive(Debug)]
    pub struct InMemoryMemoryService {
        store: Arc<DashMap<String, DashMap<String, StoredEntry>>>,
    }

    impl Default for InMemoryMemoryService {
        fn default() -> Self {
            Self {
                store: Arc::new(DashMap::new()),
            }
        }
    }

    impl InMemoryMemoryService {
        /// Creates a new `InMemoryMemoryService`.
        #[must_use]
        pub fn new() -> Self {
            Self::default()
        }

        fn namespace(auth_ctx: &AuthContext) -> String {
            format!("{}/{}", auth_ctx.app_name, auth_ctx.user_name)
        }

        fn extract_words(text: &str) -> HashSet<String> {
            // Simple word extraction without regex for WASM compatibility
            text.split(|c: char| !c.is_alphanumeric())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_lowercase())
                .collect()
        }

        fn keyword_score(query_words: &HashSet<String>, text: &str) -> f32 {
            let text_words = Self::extract_words(text);
            if text_words.is_empty() || query_words.is_empty() {
                return 0.0;
            }
            let matches = query_words.intersection(&text_words).count();
            matches as f32 / query_words.len() as f32
        }
    }

    #[async_trait::async_trait]
    impl MemoryService for InMemoryMemoryService {
        async fn add(&self, auth_ctx: &AuthContext, content: MemoryContent) -> AgentResult<String> {
            let ns = Self::namespace(auth_ctx);
            let id = content.source.generate_id();

            let entry = StoredEntry {
                id: id.clone(),
                text: content.text,
                source: content.source,
                metadata: content.metadata,
            };

            self.store.entry(ns).or_default().insert(id.clone(), entry);
            Ok(id)
        }

        async fn add_batch(
            &self,
            auth_ctx: &AuthContext,
            contents: Vec<MemoryContent>,
        ) -> AgentResult<Vec<String>> {
            let ns = Self::namespace(auth_ctx);
            let store = self.store.entry(ns).or_default();

            let mut ids = Vec::with_capacity(contents.len());
            for content in contents {
                let id = content.source.generate_id();
                let entry = StoredEntry {
                    id: id.clone(),
                    text: content.text,
                    source: content.source,
                    metadata: content.metadata,
                };
                store.insert(id.clone(), entry);
                ids.push(id);
            }
            Ok(ids)
        }

        async fn search(
            &self,
            auth_ctx: &AuthContext,
            query: &str,
            options: SearchOptions,
        ) -> AgentResult<Vec<MemoryEntry>> {
            let ns = Self::namespace(auth_ctx);
            let limit = options.limit.unwrap_or(10);
            let min_score = options.min_score.unwrap_or(0.0);
            let query_words = Self::extract_words(query);

            let mut results = Vec::new();

            if let Some(store) = self.store.get(&ns) {
                for entry_ref in store.iter() {
                    let entry = entry_ref.value();

                    // Apply source type filter
                    if let Some(ref types) = options.source_types {
                        if !types.contains(&entry.source.source_type()) {
                            continue;
                        }
                    }

                    // Apply metadata filter
                    if let Some(ref filter) = options.metadata_filter {
                        let mut matches = true;
                        for (key, value) in filter {
                            if entry.metadata.get(key) != Some(value) {
                                matches = false;
                                break;
                            }
                        }
                        if !matches {
                            continue;
                        }
                    }

                    // Calculate relevance score
                    // NOTE: Empty query returns all matching entries with score 1.0
                    // This is intentional for listing/browsing use cases
                    let score = if query.is_empty() {
                        1.0
                    } else {
                        Self::keyword_score(&query_words, &entry.text)
                    };

                    if score >= min_score {
                        results.push((entry.clone(), score));
                    }
                }
            }

            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(limit);

            Ok(results
                .into_iter()
                .map(|(entry, score)| MemoryEntry {
                    id: entry.id,
                    text: entry.text,
                    source: entry.source,
                    score,
                    metadata: entry.metadata,
                })
                .collect())
        }

        async fn delete(&self, auth_ctx: &AuthContext, id: &str) -> AgentResult<bool> {
            let ns = Self::namespace(auth_ctx);
            Ok(self
                .store
                .get(&ns)
                .map(|store| store.remove(id).is_some())
                .unwrap_or(false))
        }

        async fn delete_batch(&self, auth_ctx: &AuthContext, ids: &[String]) -> AgentResult<usize> {
            let ns = Self::namespace(auth_ctx);
            let mut count = 0;
            if let Some(store) = self.store.get(&ns) {
                for id in ids {
                    if store.remove(id).is_some() {
                        count += 1;
                    }
                }
            }
            Ok(count)
        }
    }
}

#[cfg(not(all(target_os = "wasi", target_env = "p1")))]
pub use native::InMemoryMemoryService;

// ============================================================================
// WASM Implementation (Single-Threaded)
// ============================================================================

#[cfg(all(target_os = "wasi", target_env = "p1"))]
mod wasm {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// In-memory implementation using keyword matching.
    ///
    /// For development and testing only. No embeddings, no persistence.
    ///
    /// # Platform Notes
    ///
    /// Uses `RefCell<HashMap>` for single-threaded access on WASM.
    #[derive(Debug)]
    pub struct InMemoryMemoryService {
        store: Rc<RefCell<HashMap<String, HashMap<String, StoredEntry>>>>,
    }

    impl Default for InMemoryMemoryService {
        fn default() -> Self {
            Self {
                store: Rc::new(RefCell::new(HashMap::new())),
            }
        }
    }

    impl InMemoryMemoryService {
        /// Creates a new `InMemoryMemoryService`.
        #[must_use]
        pub fn new() -> Self {
            Self::default()
        }

        fn namespace(auth_ctx: &AuthContext) -> String {
            format!("{}/{}", auth_ctx.app_name, auth_ctx.user_name)
        }

        fn extract_words(text: &str) -> HashSet<String> {
            text.split(|c: char| !c.is_alphanumeric())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_lowercase())
                .collect()
        }

        fn keyword_score(query_words: &HashSet<String>, text: &str) -> f32 {
            let text_words = Self::extract_words(text);
            if text_words.is_empty() || query_words.is_empty() {
                return 0.0;
            }
            let matches = query_words.intersection(&text_words).count();
            matches as f32 / query_words.len() as f32
        }
    }

    #[async_trait::async_trait(?Send)]
    impl MemoryService for InMemoryMemoryService {
        async fn add(&self, auth_ctx: &AuthContext, content: MemoryContent) -> AgentResult<String> {
            let ns = Self::namespace(auth_ctx);
            let id = content.source.generate_id();

            let entry = StoredEntry {
                id: id.clone(),
                text: content.text,
                source: content.source,
                metadata: content.metadata,
            };

            self.store
                .borrow_mut()
                .entry(ns)
                .or_default()
                .insert(id.clone(), entry);
            Ok(id)
        }

        async fn add_batch(
            &self,
            auth_ctx: &AuthContext,
            contents: Vec<MemoryContent>,
        ) -> AgentResult<Vec<String>> {
            let ns = Self::namespace(auth_ctx);
            let mut store = self.store.borrow_mut();
            let ns_store = store.entry(ns).or_default();

            let mut ids = Vec::with_capacity(contents.len());
            for content in contents {
                let id = content.source.generate_id();
                let entry = StoredEntry {
                    id: id.clone(),
                    text: content.text,
                    source: content.source,
                    metadata: content.metadata,
                };
                ns_store.insert(id.clone(), entry);
                ids.push(id);
            }
            Ok(ids)
        }

        async fn search(
            &self,
            auth_ctx: &AuthContext,
            query: &str,
            options: SearchOptions,
        ) -> AgentResult<Vec<MemoryEntry>> {
            let ns = Self::namespace(auth_ctx);
            let limit = options.limit.unwrap_or(10);
            let min_score = options.min_score.unwrap_or(0.0);
            let query_words = Self::extract_words(query);

            let mut results = Vec::new();
            let store = self.store.borrow();

            if let Some(ns_store) = store.get(&ns) {
                for entry in ns_store.values() {
                    // Apply source type filter
                    if let Some(ref types) = options.source_types {
                        if !types.contains(&entry.source.source_type()) {
                            continue;
                        }
                    }

                    // Apply metadata filter
                    if let Some(ref filter) = options.metadata_filter {
                        let mut matches = true;
                        for (key, value) in filter {
                            if entry.metadata.get(key) != Some(value) {
                                matches = false;
                                break;
                            }
                        }
                        if !matches {
                            continue;
                        }
                    }

                    // Calculate relevance score
                    // NOTE: Empty query returns all matching entries with score 1.0
                    // This is intentional for listing/browsing use cases
                    let score = if query.is_empty() {
                        1.0
                    } else {
                        Self::keyword_score(&query_words, &entry.text)
                    };

                    if score >= min_score {
                        results.push((entry.clone(), score));
                    }
                }
            }

            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(limit);

            Ok(results
                .into_iter()
                .map(|(entry, score)| MemoryEntry {
                    id: entry.id,
                    text: entry.text,
                    source: entry.source,
                    score,
                    metadata: entry.metadata,
                })
                .collect())
        }

        async fn delete(&self, auth_ctx: &AuthContext, id: &str) -> AgentResult<bool> {
            let ns = Self::namespace(auth_ctx);
            let mut store = self.store.borrow_mut();
            Ok(store
                .get_mut(&ns)
                .map(|ns_store| ns_store.remove(id).is_some())
                .unwrap_or(false))
        }

        async fn delete_batch(&self, auth_ctx: &AuthContext, ids: &[String]) -> AgentResult<usize> {
            let ns = Self::namespace(auth_ctx);
            let mut store = self.store.borrow_mut();
            let mut count = 0;
            if let Some(ns_store) = store.get_mut(&ns) {
                for id in ids {
                    if ns_store.remove(id).is_some() {
                        count += 1;
                    }
                }
            }
            Ok(count)
        }
    }
}

#[cfg(all(target_os = "wasi", target_env = "p1"))]
pub use wasm::InMemoryMemoryService;

#[cfg(test)]
mod tests {
    use super::*;

    fn test_auth() -> AuthContext {
        AuthContext {
            app_name: "test-app".into(),
            user_name: "alice".into(),
        }
    }

    fn other_auth() -> AuthContext {
        AuthContext {
            app_name: "test-app".into(),
            user_name: "bob".into(),
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn add_and_search_returns_results() {
        let service = InMemoryMemoryService::new();
        let auth = test_auth();

        let content = MemoryContent {
            text: "User prefers dark mode theme".to_string(),
            source: ContentSource::UserFact {
                category: Some("preferences".to_string()),
            },
            metadata: HashMap::new(),
        };

        let id = service.add(&auth, content).await.expect("add failed");
        assert!(id.starts_with("fact:preferences:"));

        let results = service
            .search(&auth, "dark mode", SearchOptions::default())
            .await
            .expect("search failed");

        assert_eq!(results.len(), 1);
        assert!(results[0].text.contains("dark mode"));
        assert!(results[0].score > 0.0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn search_filters_by_source_type() {
        let service = InMemoryMemoryService::new();
        let auth = test_auth();

        // Add a user fact
        service
            .add(
                &auth,
                MemoryContent {
                    text: "User likes cats".to_string(),
                    source: ContentSource::UserFact { category: None },
                    metadata: HashMap::new(),
                },
            )
            .await
            .expect("add fact");

        // Add a document
        service
            .add(
                &auth,
                MemoryContent {
                    text: "Cats are mammals".to_string(),
                    source: ContentSource::Document {
                        document_id: "doc1".to_string(),
                        name: "Animals".to_string(),
                        chunk_index: 0,
                        total_chunks: 1,
                    },
                    metadata: HashMap::new(),
                },
            )
            .await
            .expect("add doc");

        // Search only user facts
        let results = service
            .search(&auth, "cats", SearchOptions::history_only())
            .await
            .expect("search");
        assert_eq!(results.len(), 1);
        assert!(results[0].text.contains("likes cats"));

        // Search only documents
        let results = service
            .search(&auth, "cats", SearchOptions::knowledge_only())
            .await
            .expect("search");
        assert_eq!(results.len(), 1);
        assert!(results[0].text.contains("mammals"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn empty_query_returns_all_entries() {
        let service = InMemoryMemoryService::new();
        let auth = test_auth();

        service
            .add(
                &auth,
                MemoryContent {
                    text: "First entry".to_string(),
                    source: ContentSource::UserFact { category: None },
                    metadata: HashMap::new(),
                },
            )
            .await
            .expect("add 1");

        service
            .add(
                &auth,
                MemoryContent {
                    text: "Second entry".to_string(),
                    source: ContentSource::UserFact { category: None },
                    metadata: HashMap::new(),
                },
            )
            .await
            .expect("add 2");

        let results = service
            .search(&auth, "", SearchOptions::default())
            .await
            .expect("search");

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.score == 1.0));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn multi_tenant_isolation() {
        let service = InMemoryMemoryService::new();
        let alice = test_auth();
        let bob = other_auth();

        // Alice adds data
        service
            .add(
                &alice,
                MemoryContent {
                    text: "Alice secret".to_string(),
                    source: ContentSource::UserFact { category: None },
                    metadata: HashMap::new(),
                },
            )
            .await
            .expect("alice add");

        // Bob adds data
        service
            .add(
                &bob,
                MemoryContent {
                    text: "Bob secret".to_string(),
                    source: ContentSource::UserFact { category: None },
                    metadata: HashMap::new(),
                },
            )
            .await
            .expect("bob add");

        // Alice can only see her data
        let alice_results = service
            .search(&alice, "secret", SearchOptions::default())
            .await
            .expect("alice search");
        assert_eq!(alice_results.len(), 1);
        assert!(alice_results[0].text.contains("Alice"));

        // Bob can only see his data
        let bob_results = service
            .search(&bob, "secret", SearchOptions::default())
            .await
            .expect("bob search");
        assert_eq!(bob_results.len(), 1);
        assert!(bob_results[0].text.contains("Bob"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn delete_removes_entry() {
        let service = InMemoryMemoryService::new();
        let auth = test_auth();

        let content = MemoryContent {
            text: "To be deleted".to_string(),
            source: ContentSource::UserFact { category: None },
            metadata: HashMap::new(),
        };

        let id = service.add(&auth, content).await.expect("add");

        // Verify it exists
        let results = service
            .search(&auth, "deleted", SearchOptions::default())
            .await
            .expect("search");
        assert_eq!(results.len(), 1);

        // Delete it
        let deleted = service.delete(&auth, &id).await.expect("delete");
        assert!(deleted);

        // Verify it's gone
        let results = service
            .search(&auth, "deleted", SearchOptions::default())
            .await
            .expect("search after delete");
        assert_eq!(results.len(), 0);

        // Deleting again returns false
        let deleted_again = service.delete(&auth, &id).await.expect("delete again");
        assert!(!deleted_again);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn add_batch_adds_multiple() {
        let service = InMemoryMemoryService::new();
        let auth = test_auth();

        let contents = vec![
            MemoryContent {
                text: "First batch item".to_string(),
                source: ContentSource::UserFact { category: None },
                metadata: HashMap::new(),
            },
            MemoryContent {
                text: "Second batch item".to_string(),
                source: ContentSource::UserFact { category: None },
                metadata: HashMap::new(),
            },
            MemoryContent {
                text: "Third batch item".to_string(),
                source: ContentSource::UserFact { category: None },
                metadata: HashMap::new(),
            },
        ];

        let ids = service.add_batch(&auth, contents).await.expect("add batch");
        assert_eq!(ids.len(), 3);

        let results = service
            .search(&auth, "batch item", SearchOptions::default())
            .await
            .expect("search");
        assert_eq!(results.len(), 3);
    }
}
