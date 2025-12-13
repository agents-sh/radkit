//! Memory service backend implementations.
//!
//! This module contains the built-in implementations of the [`MemoryService`](super::MemoryService) trait.
//!
//! # Available Backends
//!
//! | Backend | Use Case | Embeddings | Feature Flag |
//! |---------|----------|------------|--------------|
//! | [`InMemoryMemoryService`] | Development, testing | No (keyword) | default |
//! | `QdrantMemoryService` | Production, self-hosted | Yes | `memory-qdrant` |
//!
//! # Custom Backends
//!
//! You can implement your own backend by implementing the `MemoryService` trait:
//!
//! ```ignore
//! use radkit::runtime::memory::{MemoryService, MemoryContent, MemoryEntry, SearchOptions};
//! use radkit::runtime::context::AuthContext;
//! use radkit::errors::AgentResult;
//!
//! pub struct MyCustomBackend { /* ... */ }
//!
//! #[async_trait::async_trait]
//! impl MemoryService for MyCustomBackend {
//!     // Implement the trait methods...
//! }
//! ```

mod in_memory;

pub use in_memory::InMemoryMemoryService;
