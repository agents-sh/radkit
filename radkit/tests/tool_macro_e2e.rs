//! End-to-end integration tests for tool macro with Agent and LLM.
//!
//! These tests demonstrate complete workflows using the #[tool] macro
//! in realistic agent scenarios.

use radkit::agent::LlmWorker;
use radkit::macros::{tool, LLMOutput};
use radkit::models::{Content, ContentPart, LlmResponse, Thread, TokenUsage};
use radkit::test_support::{structured_response, FakeLlm};
use radkit::tools::{BaseTool, BaseToolset, ToolCall, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

// ============================================================================
// E2E Test 1: Customer Service Agent with macro tools
// ============================================================================

#[derive(Debug, Deserialize, LLMOutput, Serialize, JsonSchema)]
struct CustomerInfo {
    customer_id: String,
    name: String,
    email: String,
}

#[derive(Deserialize, LLMOutput, JsonSchema)]
struct GetCustomerArgs {
    customer_id: String,
}

#[tool(description = "Get customer information by ID")]
async fn get_customer(args: GetCustomerArgs) -> ToolResult {
    // Simulate database lookup
    ToolResult::success(json!({
        "customer_id": args.customer_id,
        "name": "John Doe",
        "email": "john@example.com"
    }))
}

#[derive(Deserialize, LLMOutput, JsonSchema)]
struct UpdateEmailArgs {
    customer_id: String,
    new_email: String,
}

#[tool(description = "Update customer email address")]
async fn update_email(args: UpdateEmailArgs, ctx: &ToolContext<'_>) -> ToolResult {
    // Log the update in context state
    ctx.state()
        .set_state("last_update", json!({"customer": args.customer_id}));

    ToolResult::success(json!({
        "success": true,
        "customer_id": args.customer_id,
        "new_email": args.new_email
    }))
}

#[derive(Debug, Deserialize, LLMOutput, Serialize, JsonSchema)]
struct ServiceResult {
    action: String,
    customer_name: String,
    status: String,
}

#[tokio::test]
async fn test_customer_service_agent_with_macro_tools() {
    // Simulate conversation:
    // 1. LLM calls get_customer
    // 2. LLM calls update_email
    // 3. LLM returns final result

    let get_customer_call = LlmResponse::new(
        Content::from_parts(vec![ContentPart::ToolCall(ToolCall::new(
            "call-1",
            "get_customer",
            json!({"customer_id": "cust_123"}),
        ))]),
        TokenUsage::empty(),
    );

    let update_email_call = LlmResponse::new(
        Content::from_parts(vec![ContentPart::ToolCall(ToolCall::new(
            "call-2",
            "update_email",
            json!({
                "customer_id": "cust_123",
                "new_email": "newemail@example.com"
            }),
        ))]),
        TokenUsage::empty(),
    );

    let final_response = ServiceResult {
        action: "email_updated".to_string(),
        customer_name: "John Doe".to_string(),
        status: "success".to_string(),
    };

    let final_llm_response = structured_response(&final_response);

    let llm = FakeLlm::with_responses(
        "fake-llm",
        [
            Ok(get_customer_call),
            Ok(update_email_call),
            Ok(final_llm_response),
        ],
    );

    let worker = LlmWorker::<ServiceResult>::builder(llm)
        .with_tool(get_customer)
        .with_tool(update_email)
        .build();

    let thread =
        Thread::from_user("Update the email for customer cust_123 to newemail@example.com");
    let result = worker.run(thread).await.unwrap();

    assert_eq!(result.action, "email_updated");
    assert_eq!(result.customer_name, "John Doe");
    assert_eq!(result.status, "success");
}

// ============================================================================
// E2E Test 2: Data Analysis Agent with multiple tools
// ============================================================================

#[derive(Deserialize, LLMOutput, JsonSchema)]
struct LoadDataArgs {
    dataset_name: String,
}

#[tool(description = "Load dataset from storage")]
async fn load_dataset(args: LoadDataArgs) -> ToolResult {
    // Simulate loading data
    ToolResult::success(json!({
        "dataset": args.dataset_name,
        "rows": 1000,
        "columns": ["id", "value", "timestamp"]
    }))
}

#[derive(Deserialize, LLMOutput, JsonSchema)]
struct FilterDataArgs {
    dataset: String,
    condition: String,
}

#[tool(description = "Filter dataset based on condition")]
async fn filter_data(args: FilterDataArgs) -> ToolResult {
    ToolResult::success(json!({
        "dataset": args.dataset,
        "filtered_rows": 250,
        "condition": args.condition
    }))
}

#[derive(Deserialize, LLMOutput, JsonSchema)]
struct AggregateArgs {
    #[allow(dead_code)] // Required for JSON deserialization but not used in mock implementation
    dataset: String,
    operation: String,
}

#[tool(description = "Aggregate data with operation")]
async fn aggregate(args: AggregateArgs) -> ToolResult {
    ToolResult::success(json!({
        "operation": args.operation,
        "result": 42.5
    }))
}

#[derive(Debug, Deserialize, LLMOutput, Serialize, JsonSchema)]
struct AnalysisResult {
    dataset: String,
    filtered_count: usize,
    average_value: f64,
}

#[tokio::test]
async fn test_data_analysis_workflow() {
    // Multi-step workflow:
    // 1. Load dataset
    // 2. Filter data
    // 3. Aggregate
    // 4. Return result

    let load_call = LlmResponse::new(
        Content::from_parts(vec![ContentPart::ToolCall(ToolCall::new(
            "call-1",
            "load_dataset",
            json!({"dataset_name": "sales_data"}),
        ))]),
        TokenUsage::empty(),
    );

    let filter_call = LlmResponse::new(
        Content::from_parts(vec![ContentPart::ToolCall(ToolCall::new(
            "call-2",
            "filter_data",
            json!({
                "dataset": "sales_data",
                "condition": "value > 100"
            }),
        ))]),
        TokenUsage::empty(),
    );

    let aggregate_call = LlmResponse::new(
        Content::from_parts(vec![ContentPart::ToolCall(ToolCall::new(
            "call-3",
            "aggregate",
            json!({
                "dataset": "sales_data",
                "operation": "average"
            }),
        ))]),
        TokenUsage::empty(),
    );

    let final_result = AnalysisResult {
        dataset: "sales_data".to_string(),
        filtered_count: 250,
        average_value: 42.5,
    };

    let final_response = structured_response(&final_result);

    let llm = FakeLlm::with_responses(
        "fake-llm",
        [
            Ok(load_call),
            Ok(filter_call),
            Ok(aggregate_call),
            Ok(final_response),
        ],
    );

    let worker = LlmWorker::<AnalysisResult>::builder(llm)
        .with_tool(load_dataset)
        .with_tool(filter_data)
        .with_tool(aggregate)
        .build();

    let thread = Thread::from_user("Load sales_data, filter for values > 100, and compute average");
    let result = worker.run(thread).await.unwrap();

    assert_eq!(result.dataset, "sales_data");
    assert_eq!(result.filtered_count, 250);
    assert_eq!(result.average_value, 42.5);
}

// ============================================================================
// E2E Test 3: Agent with Runtime and macro tools
// ============================================================================

#[derive(Deserialize, LLMOutput, JsonSchema)]
struct SendNotificationArgs {
    recipient: String,
    message: String,
}

#[tool(description = "Send notification to user")]
async fn send_notification(args: SendNotificationArgs, ctx: &ToolContext<'_>) -> ToolResult {
    // Track notifications in state
    let mut notifications: Vec<String> = ctx
        .state()
        .get_state("notifications")
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default();

    notifications.push(format!("{}: {}", args.recipient, args.message));
    ctx.state().set_state("notifications", json!(notifications));

    ToolResult::success(json!({
        "sent": true,
        "recipient": args.recipient
    }))
}

#[derive(Deserialize, LLMOutput, JsonSchema)]
struct LogEventArgs {
    event: String,
    level: String,
}

#[tool(description = "Log system event")]
async fn log_event(args: LogEventArgs, ctx: &ToolContext<'_>) -> ToolResult {
    let mut logs: Vec<String> = ctx
        .state()
        .get_state("logs")
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default();

    logs.push(format!("[{}] {}", args.level, args.event));
    ctx.state().set_state("logs", json!(logs));

    ToolResult::success(json!({"logged": true}))
}

#[derive(Debug, Deserialize, LLMOutput, Serialize, JsonSchema)]
struct NotificationResult {
    notifications_sent: usize,
    logs_created: usize,
}

#[tokio::test]
async fn test_agent_with_stateful_macro_tools() {
    // Test tools that maintain state across calls
    let notify_call = LlmResponse::new(
        Content::from_parts(vec![ContentPart::ToolCall(ToolCall::new(
            "call-1",
            "send_notification",
            json!({
                "recipient": "user@example.com",
                "message": "Welcome!"
            }),
        ))]),
        TokenUsage::empty(),
    );

    let log_call = LlmResponse::new(
        Content::from_parts(vec![ContentPart::ToolCall(ToolCall::new(
            "call-2",
            "log_event",
            json!({
                "event": "Notification sent",
                "level": "INFO"
            }),
        ))]),
        TokenUsage::empty(),
    );

    let final_result = NotificationResult {
        notifications_sent: 1,
        logs_created: 1,
    };

    let final_response = structured_response(&final_result);

    let llm = FakeLlm::with_responses(
        "fake-llm",
        [Ok(notify_call), Ok(log_call), Ok(final_response)],
    );

    let worker = LlmWorker::<NotificationResult>::builder(llm)
        .with_tool(send_notification)
        .with_tool(log_event)
        .build();

    let thread = Thread::from_user("Send welcome notification and log it");
    let result = worker.run(thread).await.unwrap();

    assert_eq!(result.notifications_sent, 1);
    assert_eq!(result.logs_created, 1);
}

// ============================================================================
// E2E Test 4: Agent with custom toolset using macro tools
// ============================================================================

struct MacroToolset {
    tools: Vec<Box<dyn BaseTool>>,
}

impl MacroToolset {
    fn new() -> Self {
        Self {
            tools: vec![
                Box::new(get_customer),
                Box::new(update_email),
                Box::new(send_notification),
                Box::new(log_event),
            ],
        }
    }
}

#[cfg_attr(all(target_os = "wasi", target_env = "p1"), async_trait::async_trait(?Send))]
#[cfg_attr(
    not(all(target_os = "wasi", target_env = "p1")),
    async_trait::async_trait
)]
impl BaseToolset for MacroToolset {
    async fn get_tools(&self) -> Vec<&dyn BaseTool> {
        self.tools.iter().map(|b| b.as_ref()).collect()
    }

    async fn close(&self) {
        // No cleanup needed for this simple toolset
    }
}

#[tokio::test]
async fn test_agent_with_macro_toolset() {
    // Create toolset with macro tools
    let toolset = MacroToolset::new();
    let tools = toolset.get_tools().await;

    // Verify all tools are present
    assert_eq!(tools.len(), 4);

    let tool_names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
    assert!(tool_names.contains(&"get_customer"));
    assert!(tool_names.contains(&"update_email"));
    assert!(tool_names.contains(&"send_notification"));
    assert!(tool_names.contains(&"log_event"));

    // Verify declarations
    for tool in tools {
        let declaration = tool.declaration();
        assert!(!declaration.name().is_empty());
        assert!(!declaration.description().is_empty());
        assert!(declaration.parameters().is_object());
    }
}

// ============================================================================
// E2E Test 5: Real-world scenario - Order Processing Agent
// ============================================================================

#[derive(Deserialize, LLMOutput, JsonSchema)]
struct ValidateOrderArgs {
    order_id: String,
}

#[tool(description = "Validate order exists and is processable")]
async fn validate_order(args: ValidateOrderArgs) -> ToolResult {
    ToolResult::success(json!({
        "valid": true,
        "order_id": args.order_id,
        "total": 99.99,
        "items": 3
    }))
}

#[derive(Deserialize, LLMOutput, JsonSchema)]
struct ChargePaymentArgs {
    #[allow(dead_code)] // Required for JSON deserialization but not used in mock implementation
    order_id: String,
    amount: f64,
}

#[tool(description = "Charge payment for order")]
async fn charge_payment(args: ChargePaymentArgs) -> ToolResult {
    ToolResult::success(json!({
        "charged": true,
        "transaction_id": "txn_123",
        "amount": args.amount
    }))
}

#[derive(Deserialize, LLMOutput, JsonSchema)]
struct ShipOrderArgs {
    order_id: String,
}

#[tool(description = "Ship the order")]
async fn ship_order(args: ShipOrderArgs, ctx: &ToolContext<'_>) -> ToolResult {
    ctx.state().set_state("shipped_order", json!(args.order_id));

    ToolResult::success(json!({
        "shipped": true,
        "tracking_number": "TRACK123",
        "order_id": args.order_id
    }))
}

#[derive(Debug, Deserialize, LLMOutput, Serialize, JsonSchema)]
struct OrderProcessingResult {
    order_id: String,
    status: String,
    transaction_id: String,
    tracking_number: String,
}

#[tokio::test]
async fn test_order_processing_workflow() {
    // Complete order processing workflow
    let validate_call = LlmResponse::new(
        Content::from_parts(vec![ContentPart::ToolCall(ToolCall::new(
            "call-1",
            "validate_order",
            json!({"order_id": "ORD-001"}),
        ))]),
        TokenUsage::empty(),
    );

    let charge_call = LlmResponse::new(
        Content::from_parts(vec![ContentPart::ToolCall(ToolCall::new(
            "call-2",
            "charge_payment",
            json!({"order_id": "ORD-001", "amount": 99.99}),
        ))]),
        TokenUsage::empty(),
    );

    let ship_call = LlmResponse::new(
        Content::from_parts(vec![ContentPart::ToolCall(ToolCall::new(
            "call-3",
            "ship_order",
            json!({"order_id": "ORD-001"}),
        ))]),
        TokenUsage::empty(),
    );

    let final_result = OrderProcessingResult {
        order_id: "ORD-001".to_string(),
        status: "completed".to_string(),
        transaction_id: "txn_123".to_string(),
        tracking_number: "TRACK123".to_string(),
    };

    let final_response = structured_response(&final_result);

    let llm = FakeLlm::with_responses(
        "fake-llm",
        [
            Ok(validate_call),
            Ok(charge_call),
            Ok(ship_call),
            Ok(final_response),
        ],
    );

    let worker = LlmWorker::<OrderProcessingResult>::builder(llm)
        .with_tool(validate_order)
        .with_tool(charge_payment)
        .with_tool(ship_order)
        .build();

    let thread = Thread::from_user("Process order ORD-001");
    let result = worker.run(thread).await.unwrap();

    assert_eq!(result.order_id, "ORD-001");
    assert_eq!(result.status, "completed");
    assert_eq!(result.transaction_id, "txn_123");
    assert_eq!(result.tracking_number, "TRACK123");
}
