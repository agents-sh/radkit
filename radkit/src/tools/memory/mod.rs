//! Memory tools for agent access to long-term memory.
//!
//! This module provides tools that agents can use to interact with the
//! [`MemoryService`](crate::runtime::memory::MemoryService):
//!
//! - [`LoadMemoryTool`]: Search past conversations and user facts
//! - [`SaveMemoryTool`]: Store important user information
//! - [`SearchKnowledgeTool`]: Search documents and knowledge base
//! - [`MemoryToolset`]: Convenience toolset containing all memory tools
//!
//! # Usage
//!
//! ```ignore
//! use radkit::tools::memory::MemoryToolset;
//! use radkit::runtime::memory::InMemoryMemoryService;
//! use radkit::runtime::context::AuthContext;
//! use std::sync::Arc;
//!
//! let memory_service = Arc::new(InMemoryMemoryService::new());
//! let auth = AuthContext::new("my-app", "user123");
//!
//! let toolset = MemoryToolset::new(memory_service, auth);
//!
//! // Add toolset to agent...
//! ```

mod load_memory;
mod save_memory;
mod search_knowledge;

pub use load_memory::LoadMemoryTool;
pub use save_memory::SaveMemoryTool;
pub use search_knowledge::SearchKnowledgeTool;

use crate::runtime::context::AuthContext;
use crate::runtime::memory::MemoryService;
use crate::tools::{BaseTool, BaseToolset};
use std::sync::Arc;

/// Toolset providing memory and knowledge access to agents.
///
/// Contains all three memory tools:
/// - `load_memory`: Search past conversations and user facts
/// - `save_memory`: Store user facts and preferences
/// - `search_knowledge`: Search documents and external sources
pub struct MemoryToolset {
    load_memory: LoadMemoryTool,
    save_memory: SaveMemoryTool,
    search_knowledge: SearchKnowledgeTool,
}

impl MemoryToolset {
    /// Creates a new memory toolset with the given memory service and auth context.
    ///
    /// The auth context is captured at construction time and used for all tool
    /// invocations.
    pub fn new(memory_service: Arc<dyn MemoryService>, auth_context: AuthContext) -> Self {
        Self {
            load_memory: LoadMemoryTool::new(Arc::clone(&memory_service), auth_context.clone()),
            save_memory: SaveMemoryTool::new(Arc::clone(&memory_service), auth_context.clone()),
            search_knowledge: SearchKnowledgeTool::new(memory_service, auth_context),
        }
    }
}

#[cfg_attr(
    all(target_os = "wasi", target_env = "p1"),
    async_trait::async_trait(?Send)
)]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
impl BaseToolset for MemoryToolset {
    async fn get_tools(&self) -> Vec<&dyn BaseTool> {
        vec![&self.load_memory, &self.save_memory, &self.search_knowledge]
    }

    async fn close(&self) {
        // No cleanup needed
    }
}
