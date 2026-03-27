---
title: Skills
description: Building agent capabilities with programmatic Rust skills and file-based AgentSkills.
---

In Radkit, the fundamental unit of capability for an A2A agent is the **Skill**. A skill handles a specific type of task. An agent is a collection of one or more skills.

There are two kinds of skills:

- **Programmatic Rust skills** — logic implemented in Rust, annotated with `#[skill]`.
- **AgentSkills** — LLM-driven skills defined in a `SKILL.md` file, no Rust code required.

Both register with the same `AgentBuilder` API and are indistinguishable at runtime.

---

## Programmatic Rust Skills

### The `#[skill]` macro

Annotate your struct with `#[skill]` to provide A2A metadata. This metadata populates the agent card and guides the negotiator LLM when routing incoming messages to the right skill.

```rust
use radkit::macros::skill;

#[skill(
    id = "extract_profile",
    name = "Profile Extractor",
    description = "Extracts structured user profiles from text",
    tags = ["extraction", "profiles"],
    examples = [
        "Extract a profile from: John Doe, john@example.com",
        "Parse this resume into a profile"
    ],
    input_modes = ["text/plain"],
    output_modes = ["application/json"]
)]
pub struct ProfileExtractorSkill;
```

### The `SkillHandler` trait

Implement `SkillHandler` to define the skill's logic. The only required method is `on_request`, called for every new task assigned to the skill.

```rust
use radkit::agent::{Artifact, LlmFunction, OnRequestResult, SkillHandler};
use radkit::errors::AgentResult;
use radkit::models::Content;
use radkit::runtime::context::{ProgressSender, State};
use radkit::runtime::AgentRuntime;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
struct UserProfile {
    name: String,
    email: String,
    role: String,
}

#[async_trait::async_trait]
impl SkillHandler for ProfileExtractorSkill {
    async fn on_request(
        &self,
        _state: &mut State,
        _progress: &ProgressSender,
        runtime: &dyn AgentRuntime,
        content: Content,
    ) -> AgentResult<OnRequestResult> {
        let llm = runtime.default_llm();

        let profile = LlmFunction::<UserProfile>::new_with_shared_model(
            llm,
            Some("Extract the user's name, email, and role from the text.".into()),
        )
        .run(content)
        .await?;

        let artifact = Artifact::from_json("user_profile.json", &profile)?;

        Ok(OnRequestResult::Completed {
            message: Some(Content::from_text("Profile extracted successfully.")),
            artifacts: vec![artifact],
        })
    }
}
```

### `OnRequestResult` variants

| Variant | A2A task state | When to use |
|---|---|---|
| `Completed { message, artifacts }` | `completed` | Task finished successfully |
| `InputRequired { message, slot }` | `input-required` | Need more information from the user |
| `Failed { error }` | `failed` | Unrecoverable error |
| `Rejected { reason }` | `rejected` | Skill cannot handle this request |

---

## AgentSkills — File-Based Skills

AgentSkills let you define a skill entirely in a `SKILL.md` file. The LLM reads the instructions and drives the task — no Rust implementation required. This follows the [AgentSkills specification](https://agentskills.io/specification).

### Directory structure

```
skills/
└── text-summariser/
    └── SKILL.md
```

The directory name must match the `name` field in the frontmatter.

### SKILL.md format

The file must start with YAML frontmatter followed by Markdown instructions:

```markdown
---
name: text-summariser
description: Summarises text into a concise overview. Use when the user asks to summarise or condense text.
license: MIT
allowed-tools: Bash(python3:*)
---

You are a precise text summariser.

## Instructions

1. Read the provided text carefully.
2. Write a concise summary that captures the essential information.

## Output format

Respond with:
{ "status": "complete", "message": "Your summary here." }

If no text has been provided:
{ "status": "needs_input", "message": "Please provide the text to summarise." }
```

#### Frontmatter fields

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Lowercase letters, numbers, hyphens only. Must match directory name. Max 64 chars. |
| `description` | Yes | What the skill does and when to use it. Shown to the negotiator. Max 1024 chars. |
| `license` | No | License name or path to a bundled license file. |
| `compatibility` | No | Environment requirements (packages, network access, etc.). |
| `allowed-tools` | No | Space-delimited list of pre-approved tool names. |
| `metadata` | No | Arbitrary key-value pairs for custom use. |

### Registering AgentSkills

**Compile-time embedding** — `SKILL.md` is baked into the binary at compile time (like `include_str!`). No filesystem I/O at startup. Works on WASM.

```rust
use radkit::{agent::Agent, include_skill};

let agent = Agent::builder()
    .with_name("My Agent")
    .with_skill_def(include_skill!("./skills/text-summariser"))
    .build();
```

**Runtime loading** — `SKILL.md` is read from disk at startup. Useful when you want to add or update skills without recompiling.

```rust
let agent = Agent::builder()
    .with_name("My Agent")
    .with_skill_dir("./skills/text-summariser")?
    .build();
```

**Mixing both kinds** — programmatic and file-based skills can coexist freely:

```rust
Agent::builder()
    .with_name("My Agent")
    .with_skill(ProfileExtractorSkill)                          // Rust skill
    .with_skill_def(include_skill!("./skills/summarise"))       // compile-time AgentSkill
    .with_skill_dir("./skills/translate")?                      // runtime AgentSkill
    .build()
```

### Multi-turn AgentSkills

AgentSkills support multi-turn conversations automatically. The LLM signals intent through the `status` field in its JSON response:

| LLM responds with | What happens |
|---|---|
| `{ "status": "complete", "message": "..." }` | Task completes, message returned to user |
| `{ "status": "needs_input", "message": "..." }` | Task pauses, user is asked the question, conversation continues |
| `{ "status": "failed", "reason": "..." }` | Task fails with the given reason |

The full conversation thread is preserved between turns — the LLM always has complete context when resuming.

### Feature flag

AgentSkill support requires the `agentskill` feature, which is included in the `macros` feature (enabled by default):

```toml
[dependencies]
# agentskill is included in macros (default)
radkit = { version = "0.0.4", features = ["runtime"] }

# Or enable explicitly
radkit = { version = "0.0.4", features = ["runtime", "agentskill"] }
```

---

## Registering skills with the runtime

Pass the agent builder directly to `Runtime::builder` — it injects the LLM into AgentSkill handlers automatically during `build()`.

```rust
use radkit::runtime::Runtime;

Runtime::builder(
    Agent::builder()
        .with_name("My Agent")
        .with_skill(MyRustSkill)
        .with_skill_def(include_skill!("./skills/summarise")),
    llm,
)
.build()
.serve("0.0.0.0:8080")
.await?;
```

---

## Further reading

- [Multi-turn Conversations](./multi-turn-conversations.md) — how `InputRequired` and `SkillSlot` work in detail
- [Progress Updates](./progress-updates.md) — streaming intermediate results during skill execution
