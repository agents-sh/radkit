//! # AgentSkill Agent Example
//!
//! Demonstrates how to register AgentSkills — LLM-driven skills defined in
//! `SKILL.md` files — alongside programmatic Rust skills.
//!
//! There are two ways to load an AgentSkill:
//!
//! 1. **Compile-time** via `include_skill!("path")` — the `SKILL.md` is
//!    embedded in the binary at compile time (like `include_str!`). No
//!    filesystem I/O at startup. Works on WASM.
//!
//! 2. **Runtime** via `Agent::builder().with_skill_dir("path")?` — the
//!    `SKILL.md` is read from disk at startup. Useful for hot-swappable skills
//!    without recompiling.
//!
//! Both produce identical `SkillRegistration`s at runtime.
//!
//! # Running with Dev UI
//!
//! ```bash
//! OPENROUTER_API_KEY=... cargo run --example agentskill_agent --features dev-ui
//! ```
//!
//! Then open http://localhost:8080 in your browser.
//! Send messages like "Summarise this text: ..." or "Translate hello to Spanish".

use radkit::{
    agent::Agent,
    include_skill,
    models::providers::OpenRouterLlm,
    runtime::Runtime,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load LLM via OPENROUTER_API_KEY environment variable.
    let llm = OpenRouterLlm::from_env("anthropic/claude-3.5-sonnet")?;

    // Build the agent.
    //
    // Programmatic Rust skills and AgentSkills are registered via the same
    // builder API. The LLM is injected into AgentSkill handlers automatically
    // when `Runtime::builder(...).build()` is called.
    //
    // Paths for include_skill! are relative to the `radkit` crate root
    // (CARGO_MANIFEST_DIR), which is radkit/radkit/.
    let agent = Agent::builder()
        .with_name("Text Processing Agent")
        .with_description(
            "Summarises and translates text. \
             Ask me to summarise any text, or to translate something into another language.",
        )
        // Compile-time embedded — SKILL.md baked into the binary, zero I/O at startup.
        .with_skill_def(include_skill!("../examples/agentskill_agent/skills/text-summariser"))
        // Runtime loaded — SKILL.md read from disk when the binary starts.
        // Path is relative to the current working directory at runtime.
        .with_skill_dir("examples/agentskill_agent/skills/translate")?;

    println!("AgentSkill agent running.");
    println!("Dev UI  → http://localhost:8080");
    println!("A2A RPC → http://localhost:8080/rpc");
    println!();
    println!("Try asking:");
    println!("  • Summarise this: <paste any text>");
    println!("  • Translate 'hello world' to Spanish");

    Runtime::builder(agent, llm)
        .base_url("http://localhost:8080")
        .build()
        .serve("0.0.0.0:8080")
        .await?;

    Ok(())
}
