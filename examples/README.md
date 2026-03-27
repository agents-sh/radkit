# Radkit Examples

This directory contains practical examples demonstrating radkit's capabilities.

## Running Examples

All examples require an API key for your chosen LLM provider:

```bash
# For Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# For OpenAI
export OPENAI_API_KEY="sk-..."

# For Google Gemini
export GEMINI_API_KEY="..."
```

Then run any example with:

```bash
cargo run --example <example_name>
```

## Available Examples

### 1. Simple Function (`simple_function.rs`)

**Demonstrates:** Basic `LlmFunction` usage for structured outputs.

```bash
cargo run --example simple_function
```

Shows how to:
- Create an LLM client
- Define response schema with JSON Schema
- Get type-safe structured responses

**Key concepts:** `LlmFunction`, structured outputs, `JsonSchema`

---

### 2. Worker with Tools (`worker_with_tools.rs`)

**Demonstrates:** `LlmWorker` with automatic tool calling.

```bash
cargo run --example worker_with_tools
```

Shows how to:
- Create custom tools with `FunctionTool`
- Build a worker with tools
- Let the LLM automatically decide when to use tools

**Key concepts:** `LlmWorker`, `FunctionTool`, tool execution

---

### 3. Weather Agent (`weather_agent.rs`)

**Demonstrates:** Complete agent workflow with realistic tool usage.

```bash
cargo run --example weather_agent
```

Shows how to:
- Create a tool that simulates API calls
- Build a domain-specific agent
- Handle multiple queries with the same agent
- Configure max iterations to prevent loops

**Key concepts:** Agent patterns, tool design, error handling

---

### 4. Multi-Turn Conversation (`multi_turn_conversation.rs`)

**Demonstrates:** Maintaining conversation context across multiple exchanges.

```bash
cargo run --example multi_turn_conversation
```

Shows how to:
- Use `run_and_continue()` to preserve context
- Build multi-turn conversations
- Add events to the thread
- Track conversation history

**Key concepts:** `Thread`, `Event`, conversation state

---

### 5. Stateful Shopping Cart (`stateful_shopping_cart.rs`)

**Demonstrates:** Tools with persistent state using `ExecutionState`.

```bash
cargo run --example stateful_shopping_cart
```

Shows how to:
- Create stateful tools using `ToolContext`
- Maintain state across multiple tool calls
- Build complex multi-step workflows
- Handle tool errors gracefully

**Key concepts:** `ExecutionState`, `ToolContext`, stateful workflows

---

### 6. AgentSkill Agent (`agentskill_agent/agent.rs`)

**Demonstrates:** Registering LLM-driven skills defined in `SKILL.md` files — no Rust skill code required.

```bash
cargo run --example agentskill_agent --features agentskill
```

Shows how to:
- Write a skill as a `SKILL.md` file following the [AgentSkills spec](https://agentskills.io/specification)
- Embed a skill at compile time with `include_skill!` (no I/O at startup)
- Load a skill from the filesystem at runtime with `with_skill_dir`
- Mix file-based AgentSkills with programmatic Rust skills

**Key concepts:** `AgentSkillDef`, `include_skill!`, `with_skill_dir`, `SKILL.md`, multi-turn AgentSkills

---

## Example Categories

### For Beginners
Start with these examples to learn the basics:
1. `simple_function.rs` - Understand structured outputs
2. `worker_with_tools.rs` - Learn tool execution

### For Intermediate Users
Explore more complex patterns:
3. `weather_agent.rs` - See realistic agent design
4. `multi_turn_conversation.rs` - Master conversation management

### For Advanced Users
Tackle sophisticated workflows:
5. `stateful_shopping_cart.rs` - Build stateful agents
6. `agentskill_agent/` - Define skills in SKILL.md files

---

## Common Patterns

### Creating an LLM Client

```rust
use radkit::models::providers::AnthropicLlm;

// From environment variable
let llm = AnthropicLlm::from_env("claude-sonnet-4-5-20250929")?;

// With explicit API key
let llm = AnthropicLlm::new("claude-sonnet-4-5-20250929", "sk-ant-...");
```

### Defining Response Schema

```rust
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct MyResponse {
    field1: String,
    field2: i32,
}
```

### Creating a Tool

```rust
use radkit::tools::{FunctionTool, ToolResult};
use serde_json::json;

let tool = FunctionTool::new(
    "tool_name",
    "Tool description",
    |args, ctx| {
        Box::pin(async move {
            // Tool logic here
            ToolResult::success(json!({"result": "value"}))
        })
    },
);
```

### Building a Worker

```rust
use radkit::agent::LlmWorker;

let worker = LlmWorker::<ResponseType>::builder(llm)
    .with_system_instructions("System prompt here")
    .with_tool(tool)
    .with_max_iterations(5)
    .build();

let response = worker.run("User query").await?;
```

---

## Next Steps

After exploring these examples:

1. Read the [main README](../a2a-client/README.md) for comprehensive API documentation
2. Check the [test suite](../tests/) for more usage patterns
3. Review [TESTING_DOCUMENTATION.md](../../TESTING_DOCUMENTATION.md) for detailed API coverage
4. Build your own agent using these patterns as templates

---

## Troubleshooting

**Missing API Key:**
```
Error: Missing ANTHROPIC_API_KEY environment variable
```
→ Set your API key as shown above

**Compilation Errors:**
```
cargo build --example <name>
```
→ Check that you're using the correct imports and types

**Runtime Errors:**
```
Max iterations exceeded
```
→ Increase `with_max_iterations()` or check your tool logic

---

## Contributing

To add a new example:

1. Create a new file in this directory: `my_example.rs`
2. Add documentation at the top explaining what it demonstrates
3. Include a `main()` function that runs the example
4. Update this README with a description
5. Ensure it compiles: `cargo build --example my_example`
6. Test it works: `cargo run --example my_example`

Keep examples:
- ✅ Focused on one concept
- ✅ Well-documented with comments
- ✅ Practical and runnable
- ✅ Under 200 lines when possible
