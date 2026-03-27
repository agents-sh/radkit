//! Integration tests for LlmWorker and LlmFunction execution within skills.
//!
//! These tests verify that LlmWorker and LlmFunction work correctly when used
//! within actual skill implementations, demonstrating realistic usage patterns.

use radkit::agent::{LlmFunction, LlmWorker};
use radkit::macros::LLMOutput;
use radkit::models::{Content, ContentPart, Event, LlmResponse, Thread, TokenUsage};
use radkit::test_support::{structured_response, FakeLlm, RecordingTool};
use radkit::tools::{FunctionTool, ToolCall, ToolContext, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, VecDeque};

// ============================================================================
// Test 1: LlmFunction basic usage in skill-like scenario
// ============================================================================

#[derive(Debug, PartialEq, Deserialize, LLMOutput, Serialize, JsonSchema)]
struct SentimentAnalysis {
    sentiment: String,
    confidence: f64,
}

#[tokio::test]
async fn test_llm_function_with_system_instructions() {
    let response = SentimentAnalysis {
        sentiment: "positive".to_string(),
        confidence: 0.95,
    };

    let llm = FakeLlm::with_responses("fake-llm", [Ok(structured_response(&response))]);

    // Create LlmFunction with system instructions (typical skill pattern)
    let sentiment_fn = LlmFunction::<SentimentAnalysis>::new_with_system_instructions(
        llm.clone(),
        "You are a sentiment analyzer.",
    );

    let thread = Thread::from_user("Analyze: I love this!");
    let result = sentiment_fn.run(thread).await.unwrap();

    assert_eq!(result.sentiment, "positive");
    assert_eq!(result.confidence, 0.95);

    // Verify system instructions were applied (includes both custom and schema instructions)
    let calls = llm.calls();
    assert_eq!(calls.len(), 1);
    let system_instructions = calls[0].system().expect("system instructions present");
    assert!(
        system_instructions.contains("You are a sentiment analyzer."),
        "Expected system instructions to contain custom instructions"
    );
    assert!(
        system_instructions.contains("JSON object matching the following schema"),
        "Expected system instructions to contain schema instructions"
    );
}

// ============================================================================
// Test 2: LlmFunction with multi-turn conversation
// ============================================================================

#[derive(Debug, PartialEq, Deserialize, LLMOutput, Serialize, JsonSchema)]
struct Translation {
    translated_text: String,
    source_language: String,
}

#[tokio::test]
async fn test_llm_function_multi_turn_conversation() {
    let response1 = Translation {
        translated_text: "Hola mundo".to_string(),
        source_language: "English".to_string(),
    };

    let response2 = Translation {
        translated_text: "Buenos días".to_string(),
        source_language: "English".to_string(),
    };

    let llm = FakeLlm::with_responses(
        "fake-llm",
        [
            Ok(structured_response(&response1)),
            Ok(structured_response(&response2)),
        ],
    );

    let translate_fn = LlmFunction::<Translation>::new(llm.clone());

    // First translation
    let thread = Thread::from_user("Translate to Spanish: Hello world");
    let (result1, continued_thread) = translate_fn.run_and_continue(thread).await.unwrap();

    assert_eq!(result1.translated_text, "Hola mundo");

    // Continue with more context
    let thread2 = continued_thread.add_event(Event::user("Now translate: Good morning"));
    let result2 = translate_fn.run(thread2).await.unwrap();

    assert_eq!(result2.translated_text, "Buenos días");

    // Verify both calls were made
    assert_eq!(llm.calls().len(), 2);
}

// ============================================================================
// Test 3: LlmWorker with a single tool
// ============================================================================

#[derive(Debug, PartialEq, Deserialize, LLMOutput, Serialize, JsonSchema)]
struct WeatherReport {
    location: String,
    temperature: f64,
    condition: String,
}

#[tokio::test]
async fn test_llm_worker_with_tool() {
    // First response: LLM calls the weather tool
    let tool_call_response = LlmResponse::new(
        Content::from_parts(vec![ContentPart::ToolCall(ToolCall::new(
            "call-1",
            "get_weather",
            json!({"location": "Seattle"}),
        ))]),
        TokenUsage::empty(),
    );

    // Second response: LLM returns structured output after seeing tool result
    let final_response = WeatherReport {
        location: "Seattle".to_string(),
        temperature: 72.5,
        condition: "sunny".to_string(),
    };

    // Create weather tool
    let weather_tool = FunctionTool::new(
        "get_weather",
        "Get current weather",
        |args: HashMap<String, serde_json::Value>, _ctx: &ToolContext| {
            Box::pin(async move {
                let location = args
                    .get("location")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown");
                ToolResult::success(json!({
                    "temperature": 72.5,
                    "condition": "sunny",
                    "location": location
                }))
            })
        },
    );

    // Create worker with tool
    let worker_llm = FakeLlm::with_responses(
        "fake-llm",
        [
            Ok(tool_call_response),
            Ok(structured_response(&final_response)),
        ],
    );

    let worker = LlmWorker::<WeatherReport>::builder(worker_llm)
        .with_tool(weather_tool)
        .build();

    let thread = Thread::from_user("What's the weather in Seattle?");
    let report = worker.run(thread).await.unwrap();

    assert_eq!(report.location, "Seattle");
    assert_eq!(report.temperature, 72.5);
    assert_eq!(report.condition, "sunny");
}

// ============================================================================
// Test 4: LlmWorker with multiple tools and iterations
// ============================================================================

#[derive(Debug, PartialEq, Deserialize, LLMOutput, Serialize, JsonSchema)]
struct CalculationResult {
    result: f64,
    steps: Vec<String>,
}

#[tokio::test]
async fn test_llm_worker_multiple_tool_calls() {
    // Simulate: (2 + 3) * 4 = 20
    // First call: add(2, 3) = 5
    let response1 = LlmResponse::new(
        Content::from_parts(vec![ContentPart::ToolCall(ToolCall::new(
            "call-1",
            "add",
            json!({"a": 2, "b": 3}),
        ))]),
        TokenUsage::empty(),
    );

    // Second call: multiply(5, 4) = 20
    let response2 = LlmResponse::new(
        Content::from_parts(vec![ContentPart::ToolCall(ToolCall::new(
            "call-2",
            "multiply",
            json!({"a": 5, "b": 4}),
        ))]),
        TokenUsage::empty(),
    );

    // Final response: structured output
    let final_result = CalculationResult {
        result: 20.0,
        steps: vec![
            "add(2, 3) = 5".to_string(),
            "multiply(5, 4) = 20".to_string(),
        ],
    };

    let llm = FakeLlm::with_responses(
        "fake-llm",
        [
            Ok(response1),
            Ok(response2),
            Ok(structured_response(&final_result)),
        ],
    );

    let add_tool = FunctionTool::new(
        "add",
        "Add two numbers",
        |args: HashMap<String, serde_json::Value>, _ctx: &ToolContext| {
            Box::pin(async move {
                let a = args.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let b = args.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
                ToolResult::success(json!(a + b))
            })
        },
    );

    let multiply_tool = FunctionTool::new(
        "multiply",
        "Multiply two numbers",
        |args: HashMap<String, serde_json::Value>, _ctx: &ToolContext| {
            Box::pin(async move {
                let a = args.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let b = args.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
                ToolResult::success(json!(a * b))
            })
        },
    );

    let worker = LlmWorker::<CalculationResult>::builder(llm)
        .with_tool(add_tool)
        .with_tool(multiply_tool)
        .build();

    let thread = Thread::from_user("Calculate: (2 + 3) * 4");
    let result = worker.run(thread).await.unwrap();

    assert_eq!(result.result, 20.0);
    assert_eq!(result.steps.len(), 2);
}

// ============================================================================
// Test 5: LlmWorker with RecordingTool to verify tool execution
// ============================================================================

#[derive(Debug, Deserialize, LLMOutput, Serialize, JsonSchema)]
struct SearchResult {
    query: String,
    found: bool,
}

#[tokio::test]
async fn test_llm_worker_tool_execution_verification() {
    // Create a recording tool to verify it's called
    let recording_tool = RecordingTool::new("search", "Search for information", {
        let mut results = VecDeque::new();
        results.push_back(ToolResult::success(json!({"results": ["item1", "item2"]})));
        results
    });

    // First response: call the search tool
    let tool_call_response = LlmResponse::new(
        Content::from_parts(vec![ContentPart::ToolCall(ToolCall::new(
            "call-1",
            "search",
            json!({"query": "rust async"}),
        ))]),
        TokenUsage::empty(),
    );

    // Second response: structured output
    let final_response = SearchResult {
        query: "rust async".to_string(),
        found: true,
    };

    let llm = FakeLlm::with_responses(
        "fake-llm",
        [
            Ok(tool_call_response),
            Ok(structured_response(&final_response)),
        ],
    );

    let worker = LlmWorker::<SearchResult>::builder(llm)
        .with_tool(recording_tool.clone())
        .build();

    let thread = Thread::from_user("Search for rust async");
    let result = worker.run(thread).await.unwrap();

    assert_eq!(result.query, "rust async");
    assert!(result.found);

    // Verify the tool was actually called
    assert_eq!(recording_tool.call_count(), 1);
    let calls = recording_tool.calls();
    assert_eq!(calls[0]["query"], "rust async");
}

// ============================================================================
// Test 6: LlmWorker iteration limit protection
// ============================================================================

#[tokio::test]
async fn test_llm_worker_respects_max_iterations() {
    // Create an LLM that keeps calling tools infinitely
    let endless_tool_call = LlmResponse::new(
        Content::from_parts(vec![ContentPart::ToolCall(ToolCall::new(
            "call-1",
            "infinite_tool",
            json!({}),
        ))]),
        TokenUsage::empty(),
    );

    // Create 25 responses (more than default max of 20)
    let mut responses = Vec::new();
    for _ in 0..25 {
        responses.push(Ok(endless_tool_call.clone()));
    }

    let llm = FakeLlm::with_responses("fake-llm", responses);

    let infinite_tool = FunctionTool::new(
        "infinite_tool",
        "Tool that never ends",
        |_args: HashMap<String, serde_json::Value>, _ctx: &ToolContext| {
            Box::pin(async move { ToolResult::success(json!({"status": "continue"})) })
        },
    );

    #[derive(Debug, Deserialize, LLMOutput, Serialize, JsonSchema)]
    struct SimpleResponse {
        done: bool,
    }

    let worker = LlmWorker::<SimpleResponse>::builder(llm)
        .with_tool(infinite_tool)
        .with_max_iterations(5) // Set low limit for testing
        .build();

    let thread = Thread::from_user("Start infinite loop");
    let result = worker.run(thread).await;

    // Should error due to hitting iteration limit
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("iteration") || err.to_string().contains("limit"));
}
