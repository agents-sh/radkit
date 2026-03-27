//! Procedural macros for the radkit agent framework.
//!
//! This crate provides attribute macros for defining A2A-compliant skills and tools,
//! and the `include_skill!` function macro for embedding `AgentSkills` at compile time.

#![deny(unsafe_code, unreachable_patterns, unused_must_use)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)] // Common pattern in proc macro crates

mod agentskill;
mod skill;
mod tool;
mod validation;

use proc_macro::TokenStream;
use syn::parse_macro_input;

/// Attribute macro for defining A2A skills with metadata.
///
/// This macro generates the `SkillMetadata` and implements the `RegisteredSkill` trait
/// for the annotated struct, making it usable with the radkit agent builder.
///
/// # Required Parameters
///
/// - `id`: A unique identifier for the skill (String)
/// - `name`: A human-readable name for the skill (String)
/// - `description`: A detailed description of what the skill does (String)
///
/// # Optional Parameters
///
/// - `tags`: Array of keywords describing the skill's capabilities (default: [])
/// - `examples`: Array of example prompts or scenarios (default: [])
/// - `input_modes`: Array of supported input MIME types (default: [])
/// - `output_modes`: Array of supported output MIME types (default: [])
///
/// # MIME Type Validation
///
/// The macro validates `input_modes` and `output_modes` against a list of common MIME types.
/// If an invalid type is provided, a compile error will be generated with suggestions.
///
/// # Example
///
/// ```ignore
/// use radkit::prelude::*;
///
/// #[skill(
///     id = "summarize_text",
///     name = "Text Summarizer",
///     description = "Summarizes long text documents into concise summaries",
///     tags = ["text", "summarization", "nlp"],
///     examples = [
///         "Summarize this article",
///         "Give me a brief summary of this document"
///     ],
///     input_modes = ["text/plain", "text/markdown"],
///     output_modes = ["text/plain", "application/json"]
/// )]
/// pub struct SummarizeTextSkill;
///
/// #[async_trait]
/// impl SkillHandler for SummarizeTextSkill {
///     async fn on_request(
///         &self,
///         task_context: &mut TaskContext,
///         context: &Context,
///         runtime: &dyn Runtime,
///         content: Content,
///     ) -> Result<OnRequestResult, AgentError> {
///         // Implementation here
///         Ok(OnRequestResult::Completed {
///             message: Some(Content::text("Summary here")),
///             artifacts: vec![],
///         })
///     }
/// }
/// ```
///
/// # Generated Code
///
/// The macro generates:
/// 1. A static `SkillMetadata` constant named `{STRUCT_NAME}_METADATA`
/// 2. An implementation of `RegisteredSkill` trait for the struct
///
/// This allows the skill to be registered with an agent using `.with_skill()`:
///
/// ```ignore
/// let agent = AgentBuilder::new()
///     .with_skill(SummarizeTextSkill)
///     .build(runtime)?;
/// ```
#[proc_macro_attribute]
pub fn skill(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as skill::SkillArgs);
    let item = proc_macro2::TokenStream::from(item);

    skill::generate_skill_impl(args, item).into()
}

/// Attribute macro for defining tools with automatic parameter extraction.
///
/// This macro generates a zero-sized struct with the function name and implements
/// the `BaseTool` trait directly, eliminating manual parameter extraction and JSON schema construction.
///
/// The function name is used as the tool name, so choose function names that accurately
/// describe the tool's purpose.
///
/// # Required Parameters
///
/// - `description`: A detailed description of what the tool does (String)
///
/// # Example
///
/// ```ignore
/// use radkit::tools::{ToolResult, ToolContext};
/// use radkit_macros::tool;
/// use serde::{Deserialize};
/// use schemars::JsonSchema;
/// use serde_json::json;
///
/// #[derive(Deserialize, JsonSchema)]
/// struct AddArgs {
///     a: i64,
///     b: i64,
/// }
///
/// #[tool(description = "Add two numbers")]
/// async fn add(args: AddArgs) -> ToolResult {
///     ToolResult::success(json!({"sum": args.a + args.b}))
/// }
///
/// // With ToolContext
/// #[derive(Deserialize, JsonSchema)]
/// struct SaveArgs {
///     key: String,
///     value: String,
/// }
///
/// #[tool(description = "Save state")]
/// async fn save_state(args: SaveArgs, ctx: &ToolContext<'_>) -> ToolResult {
///     ctx.state().set_state(&args.key, json!(args.value));
///     ToolResult::success(json!({"saved": true}))
/// }
/// ```
///
/// # Generated Code
///
/// The macro transforms the async function into a zero-sized struct that implements
/// `BaseTool`. Parameters are automatically deserialized using serde and the JSON
/// schema is generated using schemars. The function name becomes both the struct
/// name and the tool name visible to the LLM.
///
/// # Usage
///
/// ```ignore
/// // Pass the tool struct directly to with_tool() - no function call!
/// let worker = LlmWorker::builder(llm)
///     .with_tool(add)         // ← Not add()
///     .with_tool(save_state)  // ← Not save_state()
///     .build();
/// ```
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as tool::ToolArgs);
    let item = proc_macro2::TokenStream::from(item);

    tool::generate_tool_impl(args, item).into()
}

/// Embeds an `AgentSkill` directory into the binary at compile time.
///
/// Reads the `SKILL.md` file from the given path (relative to the crate root),
/// validates its frontmatter structure, and returns an [`AgentSkillDef`] value
/// that can be passed to [`AgentBuilder::with_skill_def`].
///
/// Because `include_str!` is used internally, the `SKILL.md` content is compiled
/// into the binary — no filesystem I/O happens at runtime, and this works on
/// WASM targets.
///
/// [`AgentSkillDef`]: radkit::agent::AgentSkillDef
/// [`AgentBuilder::with_skill_def`]: radkit::agent::AgentBuilder::with_skill_def
///
/// # Panics (at startup, not compile time)
///
/// If the embedded SKILL.md fails full validation (e.g. the `name` field is
/// invalid), the process will panic at startup with a clear message.
///
/// # Compile errors
///
/// - `SKILL.md` does not exist at the given path
/// - `SKILL.md` does not begin with `---`
/// - `SKILL.md` frontmatter is not closed with `---`
///
/// # Examples
///
/// ```ignore
/// use radkit::{agent::Agent, include_skill};
///
/// let agent = Agent::builder()
///     .with_name("My Agent")
///     .with_skill_def(include_skill!("./skills/pdf-processing"))
///     .build();
/// ```
#[proc_macro]
pub fn include_skill(input: TokenStream) -> TokenStream {
    agentskill::generate_include_skill(input)
}
