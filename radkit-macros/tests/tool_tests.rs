use radkit::tools::{BaseTool, DefaultExecutionState, ToolContext, ToolResult};
use radkit_macros::tool;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;

// Test 1: Basic tool with required fields only
#[derive(Deserialize, JsonSchema)]
struct AddArgs {
    a: i64,
    b: i64,
}

#[tool(description = "Add two numbers")]
async fn add(args: AddArgs) -> ToolResult {
    ToolResult::success(json!({"sum": args.a + args.b}))
}

#[tokio::test]
async fn test_basic_tool_execution() {
    let tool = &add as &dyn BaseTool;
    assert_eq!(tool.name(), "add");
    assert_eq!(tool.description(), "Add two numbers");

    let state = DefaultExecutionState::new();
    let ctx = ToolContext::builder()
        .with_state(&state)
        .build()
        .expect("context");

    let result = tool
        .run_async(
            HashMap::from([("a".to_string(), json!(5)), ("b".to_string(), json!(3))]),
            &ctx,
        )
        .await;

    assert!(result.is_success());
    assert_eq!(result.data(), &json!({"sum": 8}));
}

// Test 2: Tool with ToolContext
#[derive(Deserialize, JsonSchema)]
struct SaveArgs {
    key: String,
    value: String,
}

#[tool(description = "Save state")]
async fn save_state(args: SaveArgs, ctx: &ToolContext<'_>) -> ToolResult {
    ctx.state().set_state(&args.key, json!(args.value));
    ToolResult::success(json!({"saved": true}))
}

#[tokio::test]
async fn test_tool_with_context() {
    let tool = &save_state as &dyn BaseTool;
    assert_eq!(tool.name(), "save_state");
    assert_eq!(tool.description(), "Save state");

    let state = DefaultExecutionState::new();
    let ctx = ToolContext::builder()
        .with_state(&state)
        .build()
        .expect("context");

    let result = tool
        .run_async(
            HashMap::from([
                ("key".to_string(), json!("test_key")),
                ("value".to_string(), json!("test_value")),
            ]),
            &ctx,
        )
        .await;

    assert!(result.is_success());
    assert_eq!(result.data(), &json!({"saved": true}));

    // Verify state was set
    let saved_value = ctx.state().get_state("test_key");
    assert_eq!(saved_value, Some(json!("test_value")));
}

// Test 3: Function name becomes tool name
#[derive(Deserialize, JsonSchema)]
struct GetWeatherArgs {
    location: String,
}

#[tool(description = "Get weather")]
async fn weather_api(args: GetWeatherArgs) -> ToolResult {
    ToolResult::success(json!({"temp": 72, "location": args.location}))
}

#[test]
fn test_function_name_as_tool_name() {
    let tool = &weather_api as &dyn BaseTool;
    assert_eq!(tool.name(), "weather_api");
    assert_eq!(tool.description(), "Get weather");
}

// Test 4: Schema generation
#[test]
fn test_schema_generation() {
    let tool = &add as &dyn BaseTool;
    let declaration = tool.declaration();
    let schema = declaration.parameters();

    // Verify schema has correct structure
    assert!(schema.get("type").is_some());
    assert_eq!(schema.get("type").and_then(|v| v.as_str()), Some("object"));

    let properties = schema.get("properties");
    assert!(properties.is_some());

    // Verify properties exist for both parameters
    if let Some(props) = properties.and_then(|v| v.as_object()) {
        assert!(props.contains_key("a"));
        assert!(props.contains_key("b"));
    }
}

// Test 5: Optional parameters with serde defaults
fn default_limit() -> usize {
    10
}

#[derive(Deserialize, JsonSchema)]
struct SearchArgs {
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
}

#[tool(description = "Search with optional limit")]
async fn search(args: SearchArgs) -> ToolResult {
    ToolResult::success(json!({"query": args.query, "limit": args.limit}))
}

#[tokio::test]
async fn test_optional_parameters_with_default() {
    let tool = &search as &dyn BaseTool;

    let state = DefaultExecutionState::new();
    let ctx = ToolContext::builder()
        .with_state(&state)
        .build()
        .expect("context");

    // Test with explicit limit
    let result = tool
        .run_async(
            HashMap::from([
                ("query".to_string(), json!("rust")),
                ("limit".to_string(), json!(5)),
            ]),
            &ctx,
        )
        .await;

    assert!(result.is_success());
    assert_eq!(result.data(), &json!({"query": "rust", "limit": 5}));

    // Test with default limit
    let result = tool
        .run_async(HashMap::from([("query".to_string(), json!("rust"))]), &ctx)
        .await;

    assert!(result.is_success());
    assert_eq!(result.data(), &json!({"query": "rust", "limit": 10}));
}

// Test 6: Invalid arguments (missing required field)
#[tokio::test]
async fn test_missing_required_parameter() {
    let tool = &add as &dyn BaseTool;

    let state = DefaultExecutionState::new();
    let ctx = ToolContext::builder()
        .with_state(&state)
        .build()
        .expect("context");

    // Missing 'b' parameter
    let result = tool
        .run_async(HashMap::from([("a".to_string(), json!(5))]), &ctx)
        .await;

    assert!(!result.is_success());
    assert!(result.error_message().is_some());
    assert!(result
        .error_message()
        .unwrap()
        .contains("missing field `b`"));
}

// Test 7: Wrong type for parameter
#[tokio::test]
async fn test_wrong_parameter_type() {
    let tool = &add as &dyn BaseTool;

    let state = DefaultExecutionState::new();
    let ctx = ToolContext::builder()
        .with_state(&state)
        .build()
        .expect("context");

    // String instead of integer
    let result = tool
        .run_async(
            HashMap::from([
                ("a".to_string(), json!(5)),
                ("b".to_string(), json!("not a number")),
            ]),
            &ctx,
        )
        .await;

    assert!(!result.is_success());
    assert!(result.error_message().is_some());
}

// Test 8: Complex nested struct
#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Address {
    street: String,
    city: String,
}

#[derive(Deserialize, JsonSchema)]
struct UserArgs {
    name: String,
    age: u32,
    address: Address,
}

#[tool(description = "Create user with nested address")]
async fn create_user(args: UserArgs) -> ToolResult {
    ToolResult::success(json!({
        "name": args.name,
        "age": args.age,
        "city": args.address.city
    }))
}

#[tokio::test]
async fn test_nested_struct_deserialization() {
    let tool = &create_user as &dyn BaseTool;

    let state = DefaultExecutionState::new();
    let ctx = ToolContext::builder()
        .with_state(&state)
        .build()
        .expect("context");

    let result = tool
        .run_async(
            HashMap::from([
                ("name".to_string(), json!("Alice")),
                ("age".to_string(), json!(30)),
                (
                    "address".to_string(),
                    json!({
                        "street": "123 Main St",
                        "city": "New York"
                    }),
                ),
            ]),
            &ctx,
        )
        .await;

    assert!(result.is_success());
    assert_eq!(
        result.data(),
        &json!({"name": "Alice", "age": 30, "city": "New York"})
    );
}

// ============================================================================
// Test 9: Visibility and attributes preservation
// ============================================================================

#[derive(Deserialize, JsonSchema)]
struct SimpleArgs {
    value: i32,
}

/// Test that pub(crate) visibility is preserved
#[tool(description = "Crate-visible tool")]
pub(crate) async fn crate_tool(args: SimpleArgs) -> ToolResult {
    ToolResult::success(json!(args.value))
}

/// Test that private visibility is preserved
#[tool(description = "Private tool")]
async fn private_tool(args: SimpleArgs) -> ToolResult {
    ToolResult::success(json!(args.value))
}

/// Test that cfg attributes are preserved
#[cfg(test)]
#[tool(description = "Conditionally compiled tool")]
pub async fn cfg_tool(args: SimpleArgs) -> ToolResult {
    ToolResult::success(json!(args.value))
}

/// Test that doc comments are preserved
#[tool(description = "Documented tool")]
/// This is a documented tool
/// with multiple lines
pub async fn documented_tool(args: SimpleArgs) -> ToolResult {
    ToolResult::success(json!(args.value))
}

#[test]
fn test_visibility_preservation() {
    // These should compile with their respective visibilities
    let _crate_tool = &crate_tool as &dyn BaseTool;
    let _private_tool = &private_tool as &dyn BaseTool;
    let _cfg_tool = &cfg_tool as &dyn BaseTool;
    let _documented_tool = &documented_tool as &dyn BaseTool;

    // Verify tools are callable
    assert_eq!(crate_tool.name(), "crate_tool");
    assert_eq!(private_tool.name(), "private_tool");
    assert_eq!(cfg_tool.name(), "cfg_tool");
    assert_eq!(documented_tool.name(), "documented_tool");
}
