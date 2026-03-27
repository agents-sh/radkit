# radkit-macros

Procedural macros for the radkit agent framework.

## Macros

### `#[skill]` — Programmatic skill registration

Annotates a struct to make it a registered A2A skill. Generates an implementation of the `RegisteredSkill` trait that returns `Arc<SkillMetadata>` with the provided fields.

```rust
use radkit::macros::skill;
use radkit::agent::SkillHandler;

#[skill(
    id = "weather-checker",
    name = "Weather Checker",
    description = "Fetches weather information for any location",
    tags = ["weather", "api"],
    examples = ["What's the weather in London?"],
    input_modes = ["text/plain"],
    output_modes = ["application/json"]
)]
pub struct WeatherSkill;

#[async_trait::async_trait]
impl SkillHandler for WeatherSkill {
    async fn on_request(
        &self,
        _state: &mut radkit::runtime::context::State,
        _progress: &radkit::runtime::context::ProgressSender,
        _runtime: &dyn radkit::runtime::AgentRuntime,
        content: radkit::models::Content,
    ) -> radkit::errors::AgentResult<radkit::agent::OnRequestResult> {
        Ok(radkit::agent::OnRequestResult::Completed {
            message: Some(content),
            artifacts: vec![],
        })
    }
}
```

Register it with the agent builder:

```rust
Agent::builder()
    .with_skill(WeatherSkill)
    .build();
```

#### Required parameters

| Parameter | Description |
|---|---|
| `id` | Unique identifier. Used for routing and task association. |
| `name` | Human-readable display name. |
| `description` | What the skill does. Shown to the negotiator LLM. |

#### Optional parameters

| Parameter | Default | Description |
|---|---|---|
| `tags` | `[]` | Keyword tags for discovery. |
| `examples` | `[]` | Example prompts shown in the agent card. |
| `input_modes` | `[]` | Accepted input MIME types (validated at compile time). |
| `output_modes` | `[]` | Produced output MIME types (validated at compile time). |

#### MIME type validation

`input_modes` and `output_modes` are validated against a list of common MIME types at compile time. An invalid type produces a compile error with suggestions:

```
error: Invalid MIME type: 'text/mark'. Did you mean one of: text/markdown?
```

---

### `include_skill!` — Compile-time AgentSkill embedding

Reads a `SKILL.md` file at compile time, validates its frontmatter, and returns an `AgentSkillDef` value ready to pass to `AgentBuilder::with_skill_def`.

The `SKILL.md` content is embedded in the binary using `include_str!` — no filesystem I/O happens at startup, and it works on WASM targets.

```rust
use radkit::{agent::Agent, include_skill};

let agent = Agent::builder()
    .with_name("My Agent")
    .with_skill_def(include_skill!("./skills/text-summariser"))
    .build();
```

The path is relative to the crate root (`CARGO_MANIFEST_DIR`), the same as `include_str!`.

**Compile errors** are produced if:
- The `SKILL.md` file does not exist at the given path
- The file does not begin with `---`
- The frontmatter is not closed with `---`

**Startup panics** (not compile errors) if full validation fails (e.g. the `name` field has uppercase letters or the description is empty).

For a description of the `SKILL.md` format, see the [AgentSkills specification](https://agentskills.io/specification) or the [Skills documentation](https://radkit.rs/docs/a2a/skills).

---

### `#[tool]` — Tool definition

Transforms an `async fn` into a zero-sized struct implementing `BaseTool`. The function name becomes the tool name visible to the LLM.

```rust
use radkit::macros::tool;
use radkit::tools::{ToolContext, ToolResult};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize, JsonSchema)]
struct AddArgs {
    a: i64,
    b: i64,
}

#[tool(description = "Add two numbers and return their sum")]
async fn add(args: AddArgs) -> ToolResult {
    ToolResult::success(json!({ "sum": args.a + args.b }))
}

// Pass the struct, not a function call
let worker = radkit::agent::LlmWorker::<MyResponse>::builder(llm)
    .with_tool(add)   // ← not add()
    .build();
```

With `ToolContext` for state access:

```rust
#[tool(description = "Save a value to session state")]
async fn save(args: SaveArgs, ctx: &ToolContext<'_>) -> ToolResult {
    ctx.state().set_state(&args.key, json!(args.value));
    ToolResult::success(json!({ "saved": true }))
}
```

## License

MIT
