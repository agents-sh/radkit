//! End-to-end tests for the Memory Service.
//!
//! These tests demonstrate the full memory service capabilities including:
//! - History and Knowledge facades
//! - Memory tools with LLM workers
//! - Document and conversation ingestion
//! - Auth context auto-wiring

use std::collections::HashMap;

use radkit::agent::Agent;
use radkit::runtime::context::AuthContext;
use radkit::runtime::memory::{
    CompletedConversation, CompletedMessage, ContentSource, Document, MemoryContent,
    MemoryServiceConversationExt, MemoryServiceDocumentExt, SearchOptions, SourceType,
};
use radkit::runtime::{AgentRuntime, Runtime};
use radkit::test_support::FakeLlm;

fn test_agent() -> radkit::agent::AgentDefinition {
    Agent::builder().with_name("Memory Test Agent").build()
}

fn test_runtime() -> Runtime {
    let llm = FakeLlm::with_responses("memory_test", std::iter::empty());
    Runtime::builder(test_agent(), llm).build()
}

// =============================================================================
// History Facade Tests
// =============================================================================

#[tokio::test]
async fn test_history_facade_save_and_recall() {
    let runtime = test_runtime();
    let auth = AuthContext {
        app_name: "test-app".to_string(),
        user_name: "alice".to_string(),
    };

    // Use the history facade
    let history = runtime.history();

    // Save a user fact
    let id = history
        .save_fact(
            &auth,
            "Alice prefers dark mode for all applications".to_string(),
            Some("preferences".to_string()),
        )
        .await
        .expect("save fact");

    assert!(id.starts_with("fact:preferences:"));

    // Recall the fact
    let results = history.recall(&auth, "dark mode", 5).await.expect("recall");

    assert_eq!(results.len(), 1);
    assert!(results[0].text.contains("dark mode"));
    assert!(results[0].score > 0.0);
}

#[tokio::test]
async fn test_history_facade_filters_to_history_sources() {
    let runtime = test_runtime();
    let auth = AuthContext {
        app_name: "test-app".to_string(),
        user_name: "bob".to_string(),
    };
    let memory = runtime.memory();

    // Add a user fact (History)
    memory
        .add(
            &auth,
            MemoryContent {
                text: "Bob likes Python programming".to_string(),
                source: ContentSource::UserFact {
                    category: Some("skills".to_string()),
                },
                metadata: HashMap::new(),
            },
        )
        .await
        .expect("add user fact");

    // Add a document (Knowledge - should NOT appear in history)
    memory
        .add(
            &auth,
            MemoryContent {
                text: "Python is a programming language".to_string(),
                source: ContentSource::Document {
                    document_id: "doc-1".to_string(),
                    name: "Python Guide".to_string(),
                    chunk_index: 0,
                    total_chunks: 1,
                },
                metadata: HashMap::new(),
            },
        )
        .await
        .expect("add document");

    // History facade should only return user facts, not documents
    let history = runtime.history();
    let results = history.recall(&auth, "Python", 10).await.expect("recall");

    assert_eq!(results.len(), 1);
    assert!(results[0].text.contains("Bob likes"));

    // Verify the document is NOT in history results
    for result in &results {
        match &result.source {
            ContentSource::Document { .. } => panic!("Document should not appear in history"),
            _ => {}
        }
    }
}

// =============================================================================
// Knowledge Facade Tests
// =============================================================================

#[tokio::test]
async fn test_knowledge_facade_search() {
    let runtime = test_runtime();
    let auth = AuthContext {
        app_name: "test-app".to_string(),
        user_name: "charlie".to_string(),
    };
    let memory = runtime.memory();

    // Add a document
    memory
        .add(
            &auth,
            MemoryContent {
                text: "The vacation policy allows 20 days of paid time off".to_string(),
                source: ContentSource::Document {
                    document_id: "hr-handbook".to_string(),
                    name: "HR Handbook".to_string(),
                    chunk_index: 0,
                    total_chunks: 1,
                },
                metadata: HashMap::new(),
            },
        )
        .await
        .expect("add document");

    // Use knowledge facade to search
    let knowledge = runtime.knowledge();
    let results = knowledge
        .search(&auth, "vacation policy", 5)
        .await
        .expect("search");

    assert_eq!(results.len(), 1);
    assert!(results[0].text.contains("20 days"));
}

#[tokio::test]
async fn test_knowledge_facade_filters_to_knowledge_sources() {
    let runtime = test_runtime();
    let auth = AuthContext {
        app_name: "test-app".to_string(),
        user_name: "diana".to_string(),
    };
    let memory = runtime.memory();

    // Add a user fact (History - should NOT appear in knowledge)
    memory
        .add(
            &auth,
            MemoryContent {
                text: "Diana is interested in machine learning".to_string(),
                source: ContentSource::UserFact { category: None },
                metadata: HashMap::new(),
            },
        )
        .await
        .expect("add user fact");

    // Add an external source (Knowledge)
    memory
        .add(
            &auth,
            MemoryContent {
                text: "Machine learning is a subset of artificial intelligence".to_string(),
                source: ContentSource::External {
                    source_name: "wikipedia".to_string(),
                    source_id: Some("ml-article".to_string()),
                },
                metadata: HashMap::new(),
            },
        )
        .await
        .expect("add external");

    // Knowledge facade should only return documents/external, not user facts
    let knowledge = runtime.knowledge();
    let results = knowledge
        .search(&auth, "machine learning", 10)
        .await
        .expect("search");

    assert_eq!(results.len(), 1);
    assert!(results[0].text.contains("artificial intelligence"));

    // Verify user facts are NOT in knowledge results
    for result in &results {
        match &result.source {
            ContentSource::UserFact { .. } => {
                panic!("User fact should not appear in knowledge")
            }
            ContentSource::PastConversation { .. } => {
                panic!("Conversation should not appear in knowledge")
            }
            _ => {}
        }
    }
}

// =============================================================================
// Document Ingestion Tests
// =============================================================================

#[tokio::test]
async fn test_document_ingestion_and_search() {
    let runtime = test_runtime();
    let auth = AuthContext {
        app_name: "test-app".to_string(),
        user_name: "eve".to_string(),
    };
    let memory = runtime.memory();

    // Create a document
    let document = Document {
        id: "employee-handbook-v1".to_string(),
        name: "Employee Handbook".to_string(),
        content: "Welcome to our company! This handbook covers policies and procedures. \
                  Section 1: Work Hours. Employees should work 8 hours per day. \
                  Section 2: Leave Policy. You get 15 vacation days per year."
            .to_string(),
        metadata: HashMap::from([("version".to_string(), serde_json::json!("1.0"))]),
    };

    // Ingest the document (will be chunked)
    let ids = memory
        .add_document(&auth, document, Some(100))
        .await
        .expect("add document");

    assert!(!ids.is_empty(), "Document should be chunked");

    // All IDs should have the document prefix
    for id in &ids {
        assert!(
            id.starts_with("doc:employee-handbook-v1:chunk-"),
            "ID should have doc prefix: {}",
            id
        );
    }

    // Search for content
    let results = memory
        .search(&auth, "vacation days", SearchOptions::documents_only())
        .await
        .expect("search");

    assert!(!results.is_empty(), "Should find vacation content");
    assert!(results[0].text.contains("vacation") || results[0].text.contains("leave"));
}

#[tokio::test]
async fn test_document_deletion() {
    let runtime = test_runtime();
    let auth = AuthContext {
        app_name: "test-app".to_string(),
        user_name: "frank".to_string(),
    };
    let memory = runtime.memory();

    // Create and ingest a document
    let document = Document {
        id: "temp-doc".to_string(),
        name: "Temporary Document".to_string(),
        content: "This is temporary content that will be deleted.".to_string(),
        metadata: HashMap::new(),
    };

    let ids = memory
        .add_document(&auth, document, Some(1000))
        .await
        .expect("add document");

    assert!(!ids.is_empty());

    // Verify it exists
    let before = memory
        .search(&auth, "temporary", SearchOptions::documents_only())
        .await
        .expect("search before");
    assert!(!before.is_empty());

    // Delete the document
    let deleted_count = memory
        .delete_document(&auth, "temp-doc")
        .await
        .expect("delete document");

    assert!(deleted_count > 0, "Should have deleted at least one chunk");

    // Verify it's gone
    let after = memory
        .search(&auth, "temporary", SearchOptions::documents_only())
        .await
        .expect("search after");
    assert!(after.is_empty(), "Document should be deleted");
}

// =============================================================================
// Conversation Ingestion Tests
// =============================================================================

#[tokio::test]
async fn test_conversation_ingestion() {
    let runtime = test_runtime();
    let auth = AuthContext {
        app_name: "test-app".to_string(),
        user_name: "grace".to_string(),
    };
    let memory = runtime.memory();

    // Create a completed conversation
    let conversation = CompletedConversation {
        context_id: "session-123".to_string(),
        messages: vec![
            CompletedMessage {
                message_id: "msg-1".to_string(),
                role: "user".to_string(),
                text: "How do I reset my password?".to_string(),
                timestamp: None,
            },
            CompletedMessage {
                message_id: "msg-2".to_string(),
                role: "assistant".to_string(),
                text: "You can reset your password by clicking the 'Forgot Password' link on the login page."
                    .to_string(),
                timestamp: None,
            },
        ],
    };

    // Ingest the conversation
    let ids = memory
        .add_conversation(&auth, conversation)
        .await
        .expect("add conversation");

    assert_eq!(ids.len(), 2, "Should have 2 message IDs");

    // Search for the conversation
    let results = memory
        .search(&auth, "reset password", SearchOptions::conversations_only())
        .await
        .expect("search");

    assert!(!results.is_empty(), "Should find conversation");

    // Verify the results are from the conversation
    for result in &results {
        match &result.source {
            ContentSource::PastConversation {
                context_id,
                role,
                message_id,
            } => {
                assert_eq!(context_id, "session-123");
                assert!(role == "user" || role == "assistant");
                assert!(message_id == "msg-1" || message_id == "msg-2");
            }
            _ => panic!("Expected PastConversation source"),
        }
    }
}

// =============================================================================
// Search Options Tests
// =============================================================================

#[tokio::test]
async fn test_search_options_filtering() {
    let runtime = test_runtime();
    let auth = AuthContext {
        app_name: "test-app".to_string(),
        user_name: "henry".to_string(),
    };
    let memory = runtime.memory();

    // Add different types of content all mentioning "project"
    memory
        .add(
            &auth,
            MemoryContent {
                text: "Henry is working on project Alpha".to_string(),
                source: ContentSource::UserFact { category: None },
                metadata: HashMap::new(),
            },
        )
        .await
        .expect("add fact");

    memory
        .add(
            &auth,
            MemoryContent {
                text: "Project management best practices guide".to_string(),
                source: ContentSource::Document {
                    document_id: "pm-guide".to_string(),
                    name: "PM Guide".to_string(),
                    chunk_index: 0,
                    total_chunks: 1,
                },
                metadata: HashMap::new(),
            },
        )
        .await
        .expect("add doc");

    memory
        .add(
            &auth,
            MemoryContent {
                text: "User asked about project status".to_string(),
                source: ContentSource::PastConversation {
                    context_id: "ctx-1".to_string(),
                    message_id: "msg-1".to_string(),
                    role: "user".to_string(),
                },
                metadata: HashMap::new(),
            },
        )
        .await
        .expect("add conversation");

    // Test: Search all (no filter)
    let all = memory
        .search(&auth, "project", SearchOptions::default())
        .await
        .expect("search all");
    assert_eq!(all.len(), 3, "Should find all 3 items");

    // Test: Search history only (UserFact + PastConversation)
    let history = memory
        .search(&auth, "project", SearchOptions::history_only())
        .await
        .expect("search history");
    assert_eq!(history.len(), 2, "Should find 2 history items");

    // Test: Search knowledge only (Document)
    let knowledge = memory
        .search(&auth, "project", SearchOptions::knowledge_only())
        .await
        .expect("search knowledge");
    assert_eq!(knowledge.len(), 1, "Should find 1 knowledge item");

    // Test: Search with specific source type
    let facts_only = memory
        .search(
            &auth,
            "project",
            SearchOptions::default().with_source_types(vec![SourceType::UserFact]),
        )
        .await
        .expect("search facts");
    assert_eq!(facts_only.len(), 1, "Should find 1 fact");

    // Test: Search with limit
    let limited = memory
        .search(&auth, "project", SearchOptions::default().with_limit(1))
        .await
        .expect("search limited");
    assert_eq!(limited.len(), 1, "Should be limited to 1 result");
}

// =============================================================================
// Multi-tenancy Tests
// =============================================================================

#[tokio::test]
async fn test_memory_multi_tenancy() {
    let runtime = test_runtime();

    let alice = AuthContext {
        app_name: "app-a".to_string(),
        user_name: "alice".to_string(),
    };

    let bob = AuthContext {
        app_name: "app-b".to_string(),
        user_name: "bob".to_string(),
    };

    let memory = runtime.memory();

    // Alice saves a secret
    memory
        .add(
            &alice,
            MemoryContent {
                text: "Alice's secret project details".to_string(),
                source: ContentSource::UserFact { category: None },
                metadata: HashMap::new(),
            },
        )
        .await
        .expect("alice add");

    // Bob saves a secret
    memory
        .add(
            &bob,
            MemoryContent {
                text: "Bob's secret project details".to_string(),
                source: ContentSource::UserFact { category: None },
                metadata: HashMap::new(),
            },
        )
        .await
        .expect("bob add");

    // Alice can only see her own data
    let alice_results = memory
        .search(&alice, "secret project", SearchOptions::default())
        .await
        .expect("alice search");
    assert_eq!(alice_results.len(), 1);
    assert!(alice_results[0].text.contains("Alice"));

    // Bob can only see his own data
    let bob_results = memory
        .search(&bob, "secret project", SearchOptions::default())
        .await
        .expect("bob search");
    assert_eq!(bob_results.len(), 1);
    assert!(bob_results[0].text.contains("Bob"));
}

// =============================================================================
// Runtime Integration Tests
// =============================================================================

#[tokio::test]
async fn test_runtime_facades_share_memory_service() {
    let runtime = test_runtime();
    let auth = AuthContext {
        app_name: "test-app".to_string(),
        user_name: "ivy".to_string(),
    };

    // Use history facade to save a fact
    let history = runtime.history();
    history
        .save_fact(
            &auth,
            "Ivy prefers email notifications".to_string(),
            Some("settings".to_string()),
        )
        .await
        .expect("save via history");

    // Use the raw memory service to search - should find the same data
    let memory = runtime.memory();
    let results = memory
        .search(&auth, "email notifications", SearchOptions::default())
        .await
        .expect("search via memory");

    assert_eq!(results.len(), 1);
    assert!(results[0].text.contains("Ivy"));
}

#[tokio::test]
async fn test_memory_tools_creation() {
    let runtime = test_runtime();

    // Verify memory_tools can be created without panic
    let toolset = runtime.memory_tools();

    // Get the tool declarations
    use radkit::tools::BaseToolset;
    let tools = toolset.get_tools().await;

    // Should have 3 tools
    assert_eq!(tools.len(), 3);

    // Verify tool names
    let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
    assert!(names.contains(&"load_memory"));
    assert!(names.contains(&"save_memory"));
    assert!(names.contains(&"search_knowledge"));
}

#[tokio::test]
async fn test_memory_batch_operations() {
    let runtime = test_runtime();
    let auth = AuthContext {
        app_name: "test-app".to_string(),
        user_name: "jack".to_string(),
    };
    let memory = runtime.memory();

    // Add batch of facts
    let contents = vec![
        MemoryContent {
            text: "Jack likes coffee".to_string(),
            source: ContentSource::UserFact {
                category: Some("preferences".to_string()),
            },
            metadata: HashMap::new(),
        },
        MemoryContent {
            text: "Jack works in engineering".to_string(),
            source: ContentSource::UserFact {
                category: Some("work".to_string()),
            },
            metadata: HashMap::new(),
        },
        MemoryContent {
            text: "Jack prefers morning meetings".to_string(),
            source: ContentSource::UserFact {
                category: Some("schedule".to_string()),
            },
            metadata: HashMap::new(),
        },
    ];

    let ids = memory.add_batch(&auth, contents).await.expect("add batch");
    assert_eq!(ids.len(), 3);

    // Search should find all
    let results = memory
        .search(&auth, "Jack", SearchOptions::default().with_limit(10))
        .await
        .expect("search");
    assert_eq!(results.len(), 3);

    // Delete batch
    let deleted = memory
        .delete_batch(&auth, &ids)
        .await
        .expect("delete batch");
    assert_eq!(deleted, 3);

    // Verify deleted
    let after = memory
        .search(&auth, "Jack", SearchOptions::default())
        .await
        .expect("search after");
    assert!(after.is_empty());
}

// =============================================================================
// Memory Tools with LlmWorker Tests
// =============================================================================

/// This test verifies that memory tools work out-of-the-box with LlmWorker
/// when obtained via `runtime.memory_tools()`. The auth context is automatically
/// captured at construction time, so no manual wiring is needed.
#[tokio::test]
async fn test_memory_tools_with_llm_worker() {
    use radkit::agent::LlmWorker;
    use radkit::macros::LLMOutput;
    use radkit::models::{Content, ContentPart, LlmResponse, Thread, TokenUsage};
    use radkit::test_support::{structured_response, FakeLlm};
    use radkit::tools::ToolCall;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Deserialize, LLMOutput, Serialize, JsonSchema)]
    struct MemoryResponse {
        answer: String,
    }

    // Create runtime - this uses the default auth service which provides
    // (default-app, default-user) as the auth context
    let runtime = test_runtime();

    // Get memory_tools from runtime - this now captures auth context automatically
    let toolset = runtime.memory_tools();

    // Pre-populate some data that the load_memory tool should find
    let auth = runtime.current_user();
    runtime
        .memory()
        .add(
            &auth,
            MemoryContent {
                text: "User prefers dark mode in all applications".to_string(),
                source: ContentSource::UserFact {
                    category: Some("preferences".to_string()),
                },
                metadata: HashMap::new(),
            },
        )
        .await
        .expect("pre-populate data");

    // First response: LLM calls load_memory tool
    let tool_call_response = LlmResponse::new(
        Content::from_parts(vec![ContentPart::ToolCall(ToolCall::new(
            "call-1",
            "load_memory",
            serde_json::json!({ "query": "dark mode" }),
        ))]),
        TokenUsage::empty(),
    );

    // Second response: LLM provides final answer
    let final_response = MemoryResponse {
        answer: "Based on your preferences, you like dark mode.".to_string(),
    };

    let llm = FakeLlm::with_responses(
        "memory_tools_test",
        [
            Ok(tool_call_response),
            Ok(structured_response(&final_response)),
        ],
    );

    // Create LlmWorker with the memory toolset
    let worker = LlmWorker::<MemoryResponse>::builder(llm)
        .with_toolset(std::sync::Arc::new(toolset))
        .build();

    // Run the worker - this should NOT fail with "No auth context available"
    let thread = Thread::from("What are my preferences?");
    let result = worker.run(thread).await;

    // Verify it succeeded (the fix ensures auth context is available)
    assert!(
        result.is_ok(),
        "Worker should succeed with captured auth context: {:?}",
        result.err()
    );
}

/// This test verifies that save_memory tool works with captured auth context
#[tokio::test]
async fn test_save_memory_tool_with_llm_worker() {
    use radkit::agent::LlmWorker;
    use radkit::macros::LLMOutput;
    use radkit::models::{Content, ContentPart, LlmResponse, Thread, TokenUsage};
    use radkit::test_support::{structured_response, FakeLlm};
    use radkit::tools::ToolCall;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Deserialize, LLMOutput, Serialize, JsonSchema)]
    struct SaveResponse {
        saved: bool,
    }

    let runtime = test_runtime();
    let toolset = runtime.memory_tools();
    let auth = runtime.current_user();

    // First response: LLM calls save_memory
    let tool_call_response = LlmResponse::new(
        Content::from_parts(vec![ContentPart::ToolCall(ToolCall::new(
            "call-1",
            "save_memory",
            serde_json::json!({
                "content": "User mentioned they love hiking",
                "category": "hobbies"
            }),
        ))]),
        TokenUsage::empty(),
    );

    // Second response: structured output
    let final_response = SaveResponse { saved: true };

    let llm = FakeLlm::with_responses(
        "save_memory_test",
        [
            Ok(tool_call_response),
            Ok(structured_response(&final_response)),
        ],
    );

    let worker = LlmWorker::<SaveResponse>::builder(llm)
        .with_toolset(std::sync::Arc::new(toolset))
        .build();

    let thread = Thread::from("Remember that I love hiking");
    let result = worker.run(thread).await;

    assert!(
        result.is_ok(),
        "save_memory should work with captured auth: {:?}",
        result.err()
    );

    // Verify the fact was actually saved
    let history = runtime.history();
    let results = history.recall(&auth, "hiking", 5).await.expect("recall");
    assert!(!results.is_empty(), "Fact should be saved to memory");
    assert!(results[0].text.contains("hiking"));
}

/// This test verifies that search_knowledge tool works with captured auth context
#[tokio::test]
async fn test_search_knowledge_tool_with_llm_worker() {
    use radkit::agent::LlmWorker;
    use radkit::macros::LLMOutput;
    use radkit::models::{Content, ContentPart, LlmResponse, Thread, TokenUsage};
    use radkit::test_support::{structured_response, FakeLlm};
    use radkit::tools::ToolCall;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Deserialize, LLMOutput, Serialize, JsonSchema)]
    struct KnowledgeResponse {
        answer: String,
    }

    let runtime = test_runtime();
    let toolset = runtime.memory_tools();
    let auth = runtime.current_user();

    // Pre-populate a document
    runtime
        .memory()
        .add(
            &auth,
            MemoryContent {
                text: "The company vacation policy provides 20 days PTO".to_string(),
                source: ContentSource::Document {
                    document_id: "hr-handbook".to_string(),
                    name: "HR Handbook".to_string(),
                    chunk_index: 0,
                    total_chunks: 1,
                },
                metadata: HashMap::new(),
            },
        )
        .await
        .expect("add document");

    // First response: LLM calls search_knowledge
    let tool_call_response = LlmResponse::new(
        Content::from_parts(vec![ContentPart::ToolCall(ToolCall::new(
            "call-1",
            "search_knowledge",
            serde_json::json!({ "query": "vacation policy" }),
        ))]),
        TokenUsage::empty(),
    );

    // Second response: structured output
    let final_response = KnowledgeResponse {
        answer: "According to the HR handbook, you get 20 days PTO.".to_string(),
    };

    let llm = FakeLlm::with_responses(
        "search_knowledge_test",
        [
            Ok(tool_call_response),
            Ok(structured_response(&final_response)),
        ],
    );

    let worker = LlmWorker::<KnowledgeResponse>::builder(llm)
        .with_toolset(std::sync::Arc::new(toolset))
        .build();

    let thread = Thread::from("What's our vacation policy?");
    let result = worker.run(thread).await;

    assert!(
        result.is_ok(),
        "search_knowledge should work with captured auth: {:?}",
        result.err()
    );
}
