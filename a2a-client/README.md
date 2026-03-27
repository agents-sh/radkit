# A2A Client

Rust client for A2A v1.0 agents over HTTP+JSON and JSON-RPC.

## Version Compatibility

| Crate Version | A2A Protocol Version | Notes |
|---------------|---------------------|-------|
| 0.1.x | 1.0 | v1 transport surface with generated protobuf-backed types |

## Installation

```toml
[dependencies]
a2a-client = "0.1.2"
a2a-types = "0.1.3"
```

## Quick Start

```rust
use a2a_client::A2AClient;
use a2a_types::v1;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = A2AClient::from_card_url("https://agent.example.com")
        .await?
        .with_auth_token("your_api_key");

    let message = v1::Message {
        message_id: "msg_123".to_string(),
        context_id: String::new(),
        task_id: String::new(),
        role: v1::Role::User.into(),
        parts: vec![v1::Part {
            content: Some(v1::part::Content::Text("Hello, agent!".to_string())),
            metadata: None,
            filename: String::new(),
            media_type: "text/plain".to_string(),
        }],
        metadata: None,
        extensions: Vec::new(),
        reference_task_ids: Vec::new(),
    };

    let response = client
        .send_message(v1::SendMessageRequest {
            tenant: String::new(),
            message: Some(message),
            configuration: None,
            metadata: None,
        })
        .await?;

    println!("{response:?}");
    Ok(())
}
```

## Core Methods

- `send_message(v1::SendMessageRequest) -> v1::SendMessageResponse`
- `send_streaming_message(v1::SendMessageRequest) -> Stream<Item = Result<v1::StreamResponse, A2AError>>`
- `get_task(v1::GetTaskRequest) -> v1::Task`
- `list_tasks(v1::ListTasksRequest) -> v1::ListTasksResponse`
- `cancel_task(v1::CancelTaskRequest) -> v1::Task`
- `subscribe_to_task(v1::SubscribeToTaskRequest) -> Stream<Item = Result<v1::StreamResponse, A2AError>>`
- `get_extended_agent_card(v1::GetExtendedAgentCardRequest) -> v1::AgentCard`

## Streaming Example

```rust
use a2a_client::A2AClient;
use a2a_types::v1;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = A2AClient::from_card_url("https://agent.example.com").await?;

    let request = v1::SendMessageRequest {
        tenant: String::new(),
        message: Some(v1::Message {
            message_id: "msg_123".to_string(),
            context_id: String::new(),
            task_id: String::new(),
            role: v1::Role::User.into(),
            parts: vec![v1::Part {
                content: Some(v1::part::Content::Text("Hello, agent!".to_string())),
                metadata: None,
                filename: String::new(),
                media_type: "text/plain".to_string(),
            }],
            metadata: None,
            extensions: Vec::new(),
            reference_task_ids: Vec::new(),
        }),
        configuration: None,
        metadata: None,
    };

    let mut stream = client.send_streaming_message(request).await?;

    while let Some(event) = stream.next().await {
        println!("{:?}", event?);
    }

    Ok(())
}
```

## Notes

- The client prefers HTTP+JSON when the agent card advertises it, and falls back to JSON-RPC when needed.
- `A2AClient::agent_card()` returns the cached `a2a_types::v1::AgentCard`.
- `a2a_client::types` re-exports the generated `v1` module plus JSON-RPC envelope helpers.

## License

MIT
