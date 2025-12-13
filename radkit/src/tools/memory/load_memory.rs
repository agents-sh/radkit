//! Load memory tool for searching past conversations and user facts.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

use crate::runtime::context::AuthContext;
use crate::runtime::memory::{MemoryService, SearchOptions};
use crate::tools::{BaseTool, FunctionDeclaration, ToolContext, ToolResult};

/// Tool for agents to search long-term memory (past conversations and user facts).
///
/// This tool allows agents to recall what was discussed in previous sessions
/// or retrieve user preferences.
pub struct LoadMemoryTool {
    memory_service: Arc<dyn MemoryService>,
    auth_context: AuthContext,
}

impl LoadMemoryTool {
    /// Creates a new load memory tool with the given memory service and auth context.
    pub fn new(memory_service: Arc<dyn MemoryService>, auth_context: AuthContext) -> Self {
        Self {
            memory_service,
            auth_context,
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
impl BaseTool for LoadMemoryTool {
    fn name(&self) -> &str {
        "load_memory"
    }

    fn description(&self) -> &str {
        "Search long-term memory for past conversations and user facts. \
         Use this to recall what was discussed in previous sessions or \
         retrieve user preferences."
    }

    fn declaration(&self) -> FunctionDeclaration {
        FunctionDeclaration::new(
            self.name(),
            self.description(),
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query describing what you're looking for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5, max: 10)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }),
        )
    }

    async fn run_async(
        &self,
        args: HashMap<String, Value>,
        _context: &ToolContext<'_>,
    ) -> ToolResult {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => return ToolResult::error("Missing required argument: query"),
        };

        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v.min(10) as usize)
            .unwrap_or(5);

        let options = SearchOptions::history_only().with_limit(limit);

        let entries = match self
            .memory_service
            .search(&self.auth_context, query, options)
            .await
        {
            Ok(e) => e,
            Err(e) => return ToolResult::error(format!("Memory search failed: {e}")),
        };

        if entries.is_empty() {
            return ToolResult::success(json!({
                "found": false,
                "message": "No relevant memories found."
            }));
        }

        let memories: Vec<_> = entries
            .iter()
            .map(|e| {
                json!({
                    "text": e.text,
                    "source": e.source,
                    "relevance": e.score,
                })
            })
            .collect();

        ToolResult::success(json!({
            "found": true,
            "count": memories.len(),
            "memories": memories
        }))
    }
}
