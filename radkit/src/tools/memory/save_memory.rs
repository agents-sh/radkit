//! Save memory tool for storing user facts.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

use crate::runtime::context::AuthContext;
use crate::runtime::memory::{ContentSource, MemoryContent, MemoryService};
use crate::tools::{BaseTool, FunctionDeclaration, ToolContext, ToolResult};

const MAX_CONTENT_LENGTH: usize = 4000;

/// Tool for agents to save important facts to long-term memory.
///
/// This tool allows agents to remember user preferences, facts, or insights
/// that should be recalled in future conversations.
pub struct SaveMemoryTool {
    memory_service: Arc<dyn MemoryService>,
    auth_context: AuthContext,
}

impl SaveMemoryTool {
    /// Creates a new save memory tool with the given memory service and auth context.
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
impl BaseTool for SaveMemoryTool {
    fn name(&self) -> &'static str {
        "save_memory"
    }

    fn description(&self) -> &'static str {
        "Save important user information to long-term memory. \
         Use this to remember user preferences, facts, or insights \
         that should be recalled in future conversations."
    }

    fn declaration(&self) -> FunctionDeclaration {
        FunctionDeclaration::new(
            self.name(),
            self.description(),
            json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The information to remember. Be specific and include context."
                    },
                    "category": {
                        "type": "string",
                        "description": "Category for organization (e.g., 'preferences', 'facts')"
                    }
                },
                "required": ["content"]
            }),
        )
    }

    async fn run_async(
        &self,
        args: HashMap<String, Value>,
        _context: &ToolContext<'_>,
    ) -> ToolResult {
        let text_content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => return ToolResult::error("Missing required argument: content"),
        };

        if text_content.len() > MAX_CONTENT_LENGTH {
            return ToolResult::error(format!(
                "Content too long. Maximum {MAX_CONTENT_LENGTH} characters."
            ));
        }

        let category = args
            .get("category")
            .and_then(|v| v.as_str())
            .map(std::string::ToString::to_string);

        let memory_content = MemoryContent {
            text: text_content,
            source: ContentSource::UserFact {
                category: category.clone(),
            },
            metadata: HashMap::new(),
        };

        let id = match self
            .memory_service
            .add(&self.auth_context, memory_content)
            .await
        {
            Ok(id) => id,
            Err(e) => return ToolResult::error(format!("Failed to save: {e}")),
        };

        ToolResult::success(json!({
            "saved": true,
            "id": id,
            "category": category
        }))
    }
}
