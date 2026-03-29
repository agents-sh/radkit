//! Search knowledge tool for querying documents and external sources.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

use crate::runtime::context::AuthContext;
use crate::runtime::memory::{MemoryService, SearchOptions};
use crate::tools::{BaseTool, FunctionDeclaration, ToolContext, ToolResult};

/// Tool for agents to search documents and external knowledge.
///
/// This tool allows agents to find answers from uploaded documents,
/// manuals, or reference material.
pub struct SearchKnowledgeTool {
    memory_service: Arc<dyn MemoryService>,
    auth_context: AuthContext,
}

impl SearchKnowledgeTool {
    /// Creates a new search knowledge tool with the given memory service and auth context.
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
impl BaseTool for SearchKnowledgeTool {
    fn name(&self) -> &'static str {
        "search_knowledge"
    }

    fn description(&self) -> &'static str {
        "Search documents and knowledge base for relevant information. \
         Use this to find answers from uploaded documents, manuals, or reference material."
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
        let Some(query) = args.get("query").and_then(|v| v.as_str()) else {
            return ToolResult::error("Missing required argument: query");
        };

        let limit = args
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .map_or(5, |v| v.min(10) as usize);

        let options = SearchOptions::knowledge_only().with_limit(limit);

        let entries = match self
            .memory_service
            .search(&self.auth_context, query, options)
            .await
        {
            Ok(e) => e,
            Err(e) => return ToolResult::error(format!("Knowledge search failed: {e}")),
        };

        if entries.is_empty() {
            return ToolResult::success(json!({
                "found": false,
                "message": "No relevant documents found."
            }));
        }

        let results: Vec<_> = entries
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
            "count": results.len(),
            "results": results
        }))
    }
}
