<div style="text-align: center;">
  <div class="centered-logo-text-group">
    <img src="docs/src/assets/logo.svg" alt="RadKit Logo" width="100">
    <h1>Radkit - Rust Agent Development Kit</h1>
  </div>
</div>

**A Rust SDK for building reliable AI agent systems with first-class [A2A protocol](https://a2a-protocol.org) support.**

Radkit prioritizes developer experience and control above all else. 
Developers maintain complete control over agent behavior, execution flow, context management, and state. 

While the library provides abstractions, developers can always drop down to lower-level APIs when needed.


[![Crates.io](https://img.shields.io/crates/v/radkit.svg)](https://crates.io/crates/radkit)
[![Documentation](https://docs.rs/radkit/badge.svg)](https://docs.rs/radkit)
[![License](https://img.shields.io/crates/l/radkit.svg)](LICENSE)

---

## Features

- **A2A Protocol First** - Native support for Agent-to-Agent communication standard
- **Unified LLM Interface** - Single API for Anthropic, OpenAI, Gemini, Grok, DeepSeek
- **Tool Execution** - Automatic tool calling with multi-turn loops and state management
- **Structured Outputs** - Type-safe response deserialization with JSON Schema
- **Type Safety** - Leverage Rust's type system for reliability and correctness

---

## Installation

Add `radkit` to your `Cargo.toml`.

#### Default (Minimal)

For using core types and helpers like `LlmFunction` and `LlmWorker` without the agent server runtime:

```toml
[dependencies]
radkit = "0.0.4"
tokio = { version = "1", features = ["rt-multi-thread", "sync", "net", "process", "macros"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
schemars = "1"
```

#### With Agent Server Runtime

To include the runtime server handle and enable the full A2A agent server capabilities (on native targets), enable the `runtime` feature:

```toml
[dependencies]
radkit = { version = "0.0.4", features = ["runtime"] }
tokio = { version = "1", features = ["rt-multi-thread", "sync", "net", "process", "macros"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
schemars = "1"
```

## Feature Flags

Radkit ships optional capabilities that you can opt into per target:

- `runtime`: Enables the native runtime handle, HTTP server, tracing, and other dependencies required to run A2A-compliant agents locally.
- `dev-ui`: Builds on top of `runtime` and serves an interactive UI (native-only) where you can trigger tasks, and inspect streaming output.
- `task-store-sqlite`: Enables a native SQLite-backed `TaskStore` for persistent task, event, and state storage.

## Core Concepts

### Thread - Conversation Context

A `Thread` represents the complete conversation history with the LLM, including system prompts and message exchanges.

```rust
use radkit::models::{Thread, Event};

// Simple thread from user message
let thread = Thread::from_user("Hello, world!");

// Thread with system prompt
let thread = Thread::from_system("You are a helpful coding assistant")
    .add_event(Event::user("Explain Rust ownership"));

// Multi-turn conversation
let thread = Thread::new(vec![
    Event::user("What is 2+2?"),
    Event::assistant("2+2 equals 4."),
    Event::user("What about 3+3?"),
]);

// Builder pattern
let thread = Thread::new(vec![])
    .with_system("You are an expert in mathematics")
    .add_event(Event::user("Calculate the area of a circle with radius 5"));
```

**Type Conversions:**

```rust
// From string slice
let thread: Thread = "Hello".into();

// From String
let thread: Thread = String::from("World").into();

// From Event
let thread: Thread = Event::user("Question").into();

// From Vec<Event>
let thread: Thread = vec![
    Event::user("First"),
    Event::assistant("Response"),
].into();
```

---

### Content - Multi-Modal Messages

`Content` represents the payload of a message, supporting text, images, documents, tool calls, and tool responses.

```rust
use radkit::models::{Content, ContentPart};
use serde_json::json;

// Simple text content
let content = Content::from_text("Hello!");

// Multi-part content
let content = Content::from_parts(vec![
    ContentPart::Text("Check this image:".to_string()),
    ContentPart::from_data(
        "image/png",
        "base64_encoded_image_data_here",
        Some("photo.png".to_string())
    )?,
]);

// Access text parts
for text in content.texts() {
    println!("{}", text);
}

// Query content
if content.has_text() {
    println!("First text: {}", content.first_text().unwrap());
}

if content.has_tool_calls() {
    println!("Tool calls: {}", content.tool_calls().len());
}

// Join all text parts
if let Some(combined) = content.joined_texts() {
    println!("Combined: {}", combined);
}
```

---

### Event - Conversation Messages

`Event` represents a single message in a conversation with an associated role.

```rust
use radkit::models::{Event, Role};

// Create events with different roles
let system_event = Event::system("You are a helpful assistant");
let user_event = Event::user("What is Rust?");
let assistant_event = Event::assistant("Rust is a systems programming language...");

// Access event properties
match event.role() {
    Role::System => println!("System message"),
    Role::User => println!("User message"),
    Role::Assistant => println!("Assistant message"),
    Role::Tool => println!("Tool response"),
}

let content = event.content();
println!("Message: {}", content.first_text().unwrap_or(""));
```

---

## LLM Providers

Radkit supports multiple LLM providers with a unified interface.

### Anthropic (Claude)

```rust
use radkit::models::providers::AnthropicLlm;
use radkit::models::{BaseLlm, Thread};

// From environment variable (ANTHROPIC_API_KEY)
let llm = AnthropicLlm::from_env("claude-sonnet-4-5-20250929")?;

// With explicit API key
let llm = AnthropicLlm::new("claude-sonnet-4-5-20250929", "sk-ant-...");

// With configuration
let llm = AnthropicLlm::from_env("claude-opus-4-1-20250805")?
    .with_max_tokens(8192)
    .with_temperature(0.7);

// Generate content
let thread = Thread::from_user("Explain quantum computing");
let response = llm.generate_content(thread, None).await?;

println!("Response: {}", response.content().first_text().unwrap());
println!("Tokens used: {}", response.usage().total_tokens());
```

### OpenAI (GPT)

```rust
use radkit::models::providers::OpenAILlm;

// From environment variable (OPENAI_API_KEY)
let llm = OpenAILlm::from_env("gpt-4o")?;

// With configuration
let llm = OpenAILlm::from_env("gpt-4o-mini")?
    .with_max_tokens(2000)
    .with_temperature(0.5);

let response = llm.generate("What is machine learning?", None).await?;
```

### OpenRouter

OpenRouter exposes an OpenAI-compatible endpoint that can route calls to hosted Anthropic, Google, Cohere, and other marketplace models behind a single API key.

```rust
use radkit::models::providers::OpenRouterLlm;

// From environment variable (OPENROUTER_API_KEY)
let llm = OpenRouterLlm::from_env("anthropic/claude-3.5-sonnet")?
    .with_site_url("https://example.com") // optional attribution headers
    .with_app_name("My Radkit Agent");

let response = llm.generate("Summarize the latest release notes", None).await?;
```

### Google Gemini

```rust
use radkit::models::providers::GeminiLlm;

// From environment variable (GEMINI_API_KEY)
let llm = GeminiLlm::from_env("gemini-2.0-flash-exp")?;

let response = llm.generate("Explain neural networks", None).await?;
```

### Grok (xAI)

```rust
use radkit::models::providers::GrokLlm;

// From environment variable (XAI_API_KEY)
let llm = GrokLlm::from_env("grok-2-latest")?;

let response = llm.generate("What is the meaning of life?", None).await?;
```

### DeepSeek

```rust
use radkit::models::providers::DeepSeekLlm;

// From environment variable (DEEPSEEK_API_KEY)
let llm = DeepSeekLlm::from_env("deepseek-chat")?;

let response = llm.generate("Code review best practices", None).await?;
```

---

## LlmFunction - Simple Structured Outputs

`LlmFunction<T>` is perfect for when you want structured, typed responses without tool execution.

### Basic Usage

```rust
use radkit::agent::LlmFunction;
use radkit::models::providers::AnthropicLlm;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct MovieRecommendation {
    title: String,
    year: u16,
    genre: String,
    rating: f32,
    reason: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = AnthropicLlm::from_env("claude-sonnet-4-5-20250929")?;
    let movie_fn = LlmFunction::<MovieRecommendation>::new(llm);

    let recommendation = movie_fn
        .run("Recommend a sci-fi movie for someone who loves The Matrix")
        .await?;

    println!("🎬 {}", recommendation.title);
    println!("📅 Year: {}", recommendation.year);
    println!("🎭 Genre: {}", recommendation.genre);
    println!("⭐ Rating: {}/10", recommendation.rating);
    println!("💡 {}", recommendation.reason);

    Ok(())
}
```

### With System Instructions

```rust
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct CodeReview {
    issues: Vec<String>,
    suggestions: Vec<String>,
    severity: String,
}

let llm = AnthropicLlm::from_env("claude-sonnet-4-5-20250929")?;

let review_fn = LlmFunction::<CodeReview>::new_with_system_instructions(
    llm,
    "You are a senior code reviewer. Be thorough but constructive."
);

let code = r#"
    fn divide(a: i32, b: i32) -> i32 {
        a / b
    }
"#;

let review = review_fn.run(format!("Review this code:\n{}", code)).await?;

println!("Severity: {}", review.severity);
println!("\nIssues:");
for issue in review.issues {
    println!("  - {}", issue);
}
println!("\nSuggestions:");
for suggestion in review.suggestions {
    println!("  - {}", suggestion);
}
```

### Multi-Turn Conversations

```rust
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Answer {
    response: String,
    confidence: f32,
}

let llm = AnthropicLlm::from_env("claude-sonnet-4-5-20250929")?;
let qa_fn = LlmFunction::<Answer>::new(llm);

// First question
let (answer1, thread) = qa_fn
    .run_and_continue("What is Rust?")
    .await?;

println!("Q1: {}", answer1.response);

// Follow-up question (continues conversation)
let (answer2, thread) = qa_fn
    .run_and_continue(
        thread.add_event(Event::user("What are its main benefits?"))
    )
    .await?;

println!("Q2: {}", answer2.response);

// Another follow-up
let (answer3, _) = qa_fn
    .run_and_continue(
        thread.add_event(Event::user("Give me a code example"))
    )
    .await?;

println!("Q3: {}", answer3.response);
```

### Complex Data Structures

```rust
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Recipe {
    name: String,
    prep_time_minutes: u32,
    cook_time_minutes: u32,
    servings: u8,
    ingredients: Vec<Ingredient>,
    instructions: Vec<String>,
    tags: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Ingredient {
    name: String,
    amount: String,
    unit: String,
}

let llm = AnthropicLlm::from_env("claude-sonnet-4-5-20250929")?;
let recipe_fn = LlmFunction::<Recipe>::new_with_system_instructions(
    llm,
    "You are a professional chef. Provide detailed, accurate recipes."
);

let recipe = recipe_fn
    .run("Create a recipe for chocolate chip cookies")
    .await?;

println!("🍪 {}", recipe.name);
println!("⏱️  Prep: {}min, Cook: {}min", recipe.prep_time_minutes, recipe.cook_time_minutes);
println!("👥 Servings: {}", recipe.servings);
println!("\n📋 Ingredients:");
for ingredient in recipe.ingredients {
    println!("  - {} {} {}", ingredient.amount, ingredient.unit, ingredient.name);
}
println!("\n👨‍🍳 Instructions:");
for (i, instruction) in recipe.instructions.iter().enumerate() {
    println!("  {}. {}", i + 1, instruction);
}
```

---

## LlmWorker - Tool Execution

`LlmWorker<T>` adds automatic tool calling and multi-turn execution loops to `LlmFunction`.

```rust
use radkit::agent::LlmWorker;
use radkit::models::providers::AnthropicLlm;
use radkit::tools::{tool, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct WeatherReport {
    location: String,
    temperature: f64,
    condition: String,
    forecast: String,
}

// Define tool arguments
#[derive(Deserialize, JsonSchema)]
struct GetWeatherArgs {
    /// City name or location
    location: String,
}

// Define the weather tool using the #[tool] macro
#[tool(description = "Get current weather for a location")]
async fn get_weather(args: GetWeatherArgs) -> ToolResult {
    // In real app, call weather API here
    let weather_data = json!({
        "temperature": 72.5,
        "condition": "Sunny",
        "humidity": 65,
        "location": args.location,
    });

    ToolResult::success(weather_data)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create worker with tool
    let llm = AnthropicLlm::from_env("claude-sonnet-4-5-20250929")?;
    let worker = LlmWorker::<WeatherReport>::builder(llm)
        .with_system_instructions("You are a weather assistant")
        .with_tool(get_weather)  // Pass the tool directly
        .build();

    // Run - LLM will automatically call the weather tool
    let report = worker.run("What's the weather in San Francisco?").await?;

    println!("📍 Location: {}", report.location);
    println!("🌡️  Temperature: {}°F", report.temperature);
    println!("☀️  Condition: {}", report.condition);
    println!("📅 {}", report.forecast);

    Ok(())
}
```

### Multiple Tools

```rust
use radkit::tools::tool;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct TravelPlan {
    destination: String,
    weather_summary: String,
    hotel_recommendation: String,
    estimated_cost: f64,
}

// Define tool argument structs
#[derive(Deserialize, JsonSchema)]
struct WeatherArgs {
    /// Location to get weather for
    location: String,
}

#[derive(Deserialize, JsonSchema)]
struct HotelArgs {
    /// Location to search hotels in
    location: String,
}

#[derive(Deserialize, JsonSchema)]
struct TripCostArgs {
    /// Hotel price per night
    hotel_price: f64,
    /// Number of nights
    nights: i64,
}

// Define tools using the #[tool] macro
#[tool(description = "Get weather forecast")]
async fn get_weather(args: WeatherArgs) -> ToolResult {
    ToolResult::success(json!({
        "forecast": format!("Sunny and 75°F in {}", args.location)
    }))
}

#[tool(description = "Search for hotels in a location")]
async fn search_hotels(args: HotelArgs) -> ToolResult {
    ToolResult::success(json!({
        "hotels": [{
            "name": "Grand Hotel",
            "price": 150,
            "rating": 4.5,
            "location": args.location
        }]
    }))
}

#[tool(description = "Calculate estimated trip cost")]
async fn calculate_trip_cost(args: TripCostArgs) -> ToolResult {
    let total = args.hotel_price * args.nights as f64 + 500.0; // +flight estimate
    ToolResult::success(json!({"total": total}))
}

let llm = AnthropicLlm::from_env("claude-sonnet-4-5-20250929")?;
let worker = LlmWorker::<TravelPlan>::builder(llm)
    .with_system_instructions("You are a travel planning assistant")
    .with_tools(vec![get_weather, search_hotels, calculate_trip_cost])
    .build();

let plan = worker.run("Plan a 3-day trip to Tokyo").await?;

println!("🗺️  {}", plan.destination);
println!("🌤️  {}", plan.weather_summary);
println!("🏨 {}", plan.hotel_recommendation);
println!("💰 Estimated cost: ${:.2}", plan.estimated_cost);
```

### Stateful Tools

Tools can maintain state across calls using `ToolContext`.

```rust
use radkit::tools::{tool, ToolContext};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ShoppingCart {
    items: Vec<String>,
    total_items: u32,
    estimated_total: f64,
}

// Define tool arguments
#[derive(Deserialize, JsonSchema)]
struct AddToCartArgs {
    /// Item name to add
    item: String,
    /// Price of the item
    price: f64,
}

// Add to cart tool with state management
#[tool(description = "Add an item to the shopping cart")]
async fn add_to_cart(args: AddToCartArgs, ctx: ToolContext) -> ToolResult {
    // Get current cart
    let mut items: Vec<String> = ctx
        .state()
        .get_state("items")
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default();

    let total_price: f64 = ctx
        .state()
        .get_state("total_price")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    // Add item
    items.push(args.item.clone());
    let new_total = total_price + args.price;

    // Update state
    ctx.state().set_state("items", json!(items));
    ctx.state().set_state("total_price", json!(new_total));

    ToolResult::success(json!({
        "item_added": args.item,
        "cart_size": items.len(),
        "total": new_total
    }))
}

let llm = AnthropicLlm::from_env("claude-sonnet-4-5-20250929")?;
let worker = LlmWorker::<ShoppingCart>::builder(llm)
    .with_tool(add_to_cart)
    .build();

// The worker can call add_to_cart multiple times, maintaining state
let cart = worker.run("Add a laptop for $999 and a mouse for $25").await?;

println!("🛒 Cart:");
for item in cart.items {
    println!("  - {}", item);
}
println!("📦 Total items: {}", cart.total_items);
println!("💵 Total: ${:.2}", cart.estimated_total);
```

---

## A2A Agents

Radkit provides first-class support for building [Agent-to-Agent (A2A) protocol](https://a2a-protocol.org) compliant agents. The framework ensures that if your code compiles, it's automatically A2A compliant.

### What is A2A?

The A2A protocol is an open standard that enables seamless communication and collaboration between AI agents. It provides:
- Standardized agent discovery via Agent Cards
- Task lifecycle management (submitted, working, completed, etc.)
- Multi-turn conversations with input-required states
- Streaming support for long-running operations
- Artifact generation for tangible outputs

### Building A2A Agents

Agents in radkit are composed of **skills**. Each skill handles a specific capability and is annotated with the `#[skill]` macro to provide A2A metadata.

#### Defining a Skill

```rust
use radkit::agent::{Artifact, LlmFunction, OnRequestResult, SkillHandler};
use radkit::errors::AgentError;
use radkit::macros::skill;
use radkit::models::{BaseLlm, Content};
use radkit::runtime::context::{ProgressSender, State};
use radkit::runtime::Runtime;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// Define your output types
#[derive(Serialize, Deserialize, JsonSchema)]
struct UserProfile {
    name: String,
    email: String,
    role: String,
}

// Annotate with A2A metadata
#[skill(
    id = "extract_profile",
    name = "Profile Extractor",
    description = "Extracts structured user profiles from text",
    tags = ["extraction", "profiles"],
    examples = [
        "Extract profile: John Doe, john@example.com, Software Engineer",
        "Parse this resume into a profile"
    ],
    input_modes = ["text/plain", "application/pdf"],
    output_modes = ["application/json"]
)]
pub struct ProfileExtractorSkill;

// Implement the SkillHandler trait
#[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
impl SkillHandler for ProfileExtractorSkill {
    async fn on_request(
        &self,
        state: &mut State,
        progress: &ProgressSender,
        runtime: &dyn Runtime,
        content: Content,
    ) -> Result<OnRequestResult, AgentError> {
        // Get LLM from runtime
        let llm = runtime.llm_provider().default_llm()?;

        // Send intermediate update (A2A TaskState::Working)
        progress.send_update("Analyzing text...").await?;

        // Use LLM function for extraction
        let profile = extract_profile_data(llm)
            .run(content.first_text().unwrap())
            .await?;

        // Create artifact (automatically becomes A2A Artifact)
        let artifact = Artifact::from_json("user_profile.json", &profile)?;

        // Return completion (A2A TaskState::Completed)
        Ok(OnRequestResult::Completed {
            message: Some(Content::from_text("Profile extracted successfully")),
            artifacts: vec![artifact],
        })
    }
}

fn extract_profile_data(llm: impl BaseLlm + 'static) -> LlmFunction<UserProfile> {
    LlmFunction::new_with_system_instructions(
        llm,
        "Extract name, email, and role from the provided text."
    )
}
```

#### Multi-Turn Conversations

Skills can request additional input from users when needed. Use **slot enums** to track different input states:

```rust
use serde::{Deserialize, Serialize};

// Define slot enum to track different input requirements
#[derive(Serialize, Deserialize)]
enum ProfileSlot {
    Email,
    PhoneNumber,
    Department,
}

#[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
impl SkillHandler for ProfileExtractorSkill {
    async fn on_request(
        &self,
        state: &mut State,
        progress: &ProgressSender,
        runtime: &dyn Runtime,
        content: Content,
    ) -> Result<OnRequestResult, AgentError> {
        let llm = runtime.llm_provider().default_llm()?;
        let profile = extract_profile_data(llm)
            .run(content.first_text().unwrap())
            .await?;

        // Check what information is missing
        if profile.email.is_empty() {
            state.task().save("partial_profile", &profile)?;

            // Request email - track with slot
            state.set_slot(ProfileSlot::Email)?;
            return Ok(OnRequestResult::InputRequired {
                message: Content::from_text("Please provide the user's email address"),
            });
        }

        if profile.phone.is_empty() {
            state.task().save("partial_profile", &profile)?;

            // Request phone - different slot
            state.set_slot(ProfileSlot::PhoneNumber)?;
            return Ok(OnRequestResult::InputRequired {
                message: Content::from_text("Please provide the user's phone number"),
            });
        }

        let artifact = Artifact::from_json("user_profile.json", &profile)?;
        Ok(OnRequestResult::Completed {
            message: Some(Content::from_text("Profile complete!")),
            artifacts: vec![artifact],
        })
    }

    // Handle the follow-up input based on which slot was requested
    async fn on_input_received(
        &self,
        state: &mut State,
        progress: &ProgressSender,
        runtime: &dyn Runtime,
        content: Content,
    ) -> Result<OnInputResult, AgentError> {
        // Get the slot to know which input we're continuing from
        let slot: ProfileSlot = state.slot()?.unwrap();

        // Load saved state
        let mut profile: UserProfile = state.task()
            .load("partial_profile")?
            .ok_or_else(|| anyhow!("No partial profile found"))?;

        // Handle different continuation paths based on slot
        match slot {
            ProfileSlot::Email => {
                profile.email = content.first_text().unwrap().to_string();

                // Check if we need phone number next
                if profile.phone.is_empty() {
                    state.task().save("partial_profile", &profile)?;
                    state.set_slot(ProfileSlot::PhoneNumber)?;
                    return Ok(OnInputResult::InputRequired {
                        message: Content::from_text("Please provide your phone number"),
                    });
                }
            }
            ProfileSlot::PhoneNumber => {
                profile.phone = content.first_text().unwrap().to_string();

                // Validate phone format
                if !is_valid_phone(&profile.phone) {
                    return Ok(OnInputResult::Failed {
                        error: "Invalid phone number format".to_string(),
                    });
                }
            }
            ProfileSlot::Department => {
                profile.department = content.first_text().unwrap().to_string();
            }
        }

        // Profile is complete
        state.clear_slot();
        let artifact = Artifact::from_json("user_profile.json", &profile)?;
        Ok(OnInputResult::Completed {
            message: Some(Content::from_text("Profile completed!")),
            artifacts: vec![artifact],
        })
    }
}
```

#### Intermediate Updates and Partial Artifacts

For long-running operations, send progress updates and partial results:

```rust
#[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
impl SkillHandler for ReportGeneratorSkill {
    async fn on_request(
        &self,
        state: &mut State,
        progress: &ProgressSender,
        runtime: &dyn Runtime,
        content: Content,
    ) -> Result<OnRequestResult, AgentError> {
        let llm = runtime.llm_provider().default_llm()?;

        // Send intermediate status (A2A TaskStatusUpdateEvent with state=working)
        progress.send_update("Analyzing data...").await?;

        let analysis = analyze_data(llm.clone())
            .run(content.first_text().unwrap())
            .await?;

        // Send partial artifact (A2A TaskArtifactUpdateEvent)
        let partial = Artifact::from_json("analysis.json", &analysis)?;
        progress.send_artifact(partial).await?;

        // Another update
        progress.send_update("Generating visualizations...").await?;

        let charts = generate_charts(llm.clone())
            .run(&analysis)
            .await?;

        // Another partial artifact
        let charts_artifact = Artifact::from_json("charts.json", &charts)?;
        progress.send_artifact(charts_artifact).await?;

        // Final compilation
        progress.send_update("Compiling final report...").await?;

        let report = compile_report(llm)
            .run(&analysis, &charts)
            .await?;

        // Return final state with final artifact
        let final_artifact = Artifact::from_json("report.json", &report)?;
        Ok(OnRequestResult::Completed {
            message: Some(Content::from_text("Report complete!")),
            artifacts: vec![final_artifact],
        })
    }
}
```

#### Composing an Agent

```rust
use radkit::agent::{Agent, AgentDefinition};
use radkit::models::providers::AnthropicLlm;
use radkit::runtime::Runtime;

pub fn configure_agent() -> AgentDefinition {
    Agent::builder()
        .with_name("My A2A Agent")
        .with_description("An intelligent agent with multiple skills")
        // Skills automatically provide metadata from #[skill] macro
        .with_skill(ProfileExtractorSkill)
        .with_skill(ReportGeneratorSkill)
        .with_skill(DataAnalysisSkill)
        .build()
}

// Local development
#[cfg(not(all(target_os = "wasi", target_env = "p1")))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = AnthropicLlm::from_env("claude-sonnet-4-5-20250929")?;
    Runtime::builder(configure_agent(), llm)
        .build()
        .serve("127.0.0.1:8080")
        .await?;

    Ok(())
}
```

### How Radkit Guarantees A2A Compliance

Radkit ensures A2A compliance through **compile-time guarantees** and automatic protocol mapping:

#### 1. Typed State Management

```rust
pub enum OnRequestResult {
    InputRequired { message: Content, slot: SkillSlot },    // → A2A: state=input-required
    Completed { message: Option<Content>, artifacts: Vec<Artifact> }, // → A2A: state=completed
    Failed { error: String },                                // → A2A: state=failed
    Rejected { reason: String },                             // → A2A: state=rejected
}
```

**Guarantee:** You can only return valid A2A task states. Invalid states won't compile.

#### 2. Intermediate Updates

```rust
// Always maps to A2A TaskState::Working with final=false
progress.send_update("Processing...").await?;

// Always creates A2A TaskArtifactUpdateEvent
progress.send_artifact(artifact).await?;
```

**Guarantee:** You cannot accidentally send terminal states or mark intermediate updates as final.

#### 3. Automatic Metadata Generation

The `#[skill]` macro automatically generates:
- A2A `AgentSkill` entries for the Agent Card
- MIME type validation based on `input_modes`/`output_modes`
- Proper skill discovery metadata

**Guarantee:** Your Agent Card is always consistent with your skill implementations.

#### 4. Protocol Type Mapping

The framework automatically converts between radkit types and A2A protocol types:

| Radkit Type | A2A Protocol Type |
|-------------|-------------------|
| `Content` | `Message` with `Part[]` |
| `Artifact::from_json()` | `Artifact` with `DataPart` |
| `Artifact::from_text()` | `Artifact` with `TextPart` |
| `OnRequestResult::Completed` | `Task` with `state=completed` |
| `OnRequestResult::InputRequired` | `Task` with `state=input-required` |

**Guarantee:** You never handle A2A protocol types directly. The framework ensures correct serialization.

#### 5. Lifecycle Enforcement

```rust
// ✅ Allowed: Send intermediate updates during execution
progress.send_update("Working...").await?;

// ✅ Allowed: Send partial artifacts any time
progress.send_artifact(artifact).await?;

// ✅ Allowed: Return terminal state with final artifacts
Ok(OnRequestResult::Completed {
    artifacts: vec![final_artifact],
    ..
})

// ❌ Not possible: Can't send "completed" state during execution
// ❌ Not possible: Can't mark intermediate update as final
// ❌ Not possible: Can't send invalid task states
```

**Guarantee:** The type system prevents protocol violations at compile time.

#### How These Guarantees Work

Radkit enforces A2A compliance through several type-level mechanisms:

**1. Unrepresentable Invalid States**

The `OnRequestResult` and `OnInputResult` enums only expose valid A2A states as variants. There's no way to construct an invalid state because the type system doesn't allow it:

```rust
// ✅ This compiles - valid A2A state
Ok(OnRequestResult::Completed { message: None, artifacts: vec![] })

// ❌ This doesn't compile - InvalidState doesn't exist
Ok(OnRequestResult::InvalidState { ... })  // Compilation error!
```

**2. Restricted Method APIs**

Methods like `progress.send_update()` are internally hardcoded to use `TaskState::Working` with `final=false`. The API doesn't expose parameters that would allow setting invalid combinations:

```rust
// Implementation detail (in radkit internals):
pub async fn send_update(&self, message: impl Into<Content>) -> Result<()> {
    // Always sends TaskState::Working with final=false
    // No way for developers to override this behavior
}
```

**3. Separation of Concerns via Return Types**

Intermediate updates go through `ProgressSender` methods, while final states are only set via return values from `on_request()` and `on_input_received()`. This architectural separation, enforced by Rust's type system, makes it impossible to accidentally mark an intermediate update as final or send a terminal state mid-execution:

```rust
// During execution: Only intermediate methods available via ProgressSender
progress.send_update("Working...").await?;  // Always non-final

// At completion: Only way to set final state is via return
Ok(OnRequestResult::Completed { ... })  // Compiler ensures this ends execution
```

**4. Compile-Time WASM Compatibility**

The library uses conditional compilation and the `compat` module to ensure WASM portability while maintaining the same API surface. The `?Send` trait bound is conditionally applied based on target:

```rust
#[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait(?Send))]
#[cfg_attr(not(all(target_os = "wasi", target_env = "p1")), async_trait)]
```

This means WASM compatibility is verified at compile time - if your agent compiles for native targets, it will compile for WASM without code changes.

### Example: Complete A2A Agent

See the [hr_agent example](examples/hr_agent/) for a complete multi-skill A2A agent with:
- Resume processing with multi-turn input handling
- Onboarding plan generation with intermediate updates
- IT account creation via remote agent delegation
- Full A2A protocol compliance

---

## Contributing

Contributions welcome!

> We love agentic coding. We use Claude-Code, Gemini, Codex.
> That doesn't mean this is a random vibe-coded project. Everything in this project is carefully crafted.
> And we expect your contributions to be well-thought-out and have reasons for the changes you submit.

1. Follow the [AGENTS.md](radkit/AGENTS.md)
2. Add tests for new features
3. Update documentation
4. Ensure `cargo fmt` and `cargo clippy` pass

---

## License

MIT

---
