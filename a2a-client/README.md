# a2a-client

Rust HTTP client for calling remote [A2A v1.0](https://a2a-protocol.org/latest/specification/) agents. Supports both HTTP+JSON and JSON-RPC 2.0 transports, streaming via SSE, and agent discovery from an agent card.

## Installation

```toml
[dependencies]
a2a-client = "0.2.0"
a2a-types  = "0.2.0"
```

## Quick start

```rust
use a2a_client::A2AClient;
use a2a_types::{Message, Part, Role, SendMessageRequest, part};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = A2AClient::from_card_url("https://agent.example.com")
        .await?
        .with_auth_token("your_api_key");

    let response = client
        .send_message(SendMessageRequest {
            tenant: String::new(),
            message: Some(Message {
                message_id: "msg_1".to_string(),
                context_id: String::new(),
                task_id: String::new(),
                role: Role::User.into(),
                parts: vec![Part {
                    content: Some(part::Content::Text("Hello!".to_string())),
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
        })
        .await?;

    println!("{response:?}");
    Ok(())
}
```

## Streaming

```rust
use a2a_client::A2AClient;
use a2a_types::{Message, Part, Role, SendMessageRequest, part};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = A2AClient::from_card_url("https://agent.example.com").await?;

    let mut stream = client
        .send_streaming_message(SendMessageRequest {
            tenant: String::new(),
            message: Some(Message {
                message_id: "msg_1".to_string(),
                context_id: String::new(),
                task_id: String::new(),
                role: Role::User.into(),
                parts: vec![Part {
                    content: Some(part::Content::Text("Hello!".to_string())),
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
        })
        .await?;

    while let Some(event) = stream.next().await {
        println!("{:?}", event?);
    }
    Ok(())
}
```

## API

### Constructors

| Method | Description |
|---|---|
| `from_card_url(url)` | Fetch agent card from `/.well-known/agent-card.json` and build client |
| `from_card_url_with_client(url, client)` | Same, with a custom `reqwest::Client` |
| `from_card(card)` | Build from an already-fetched `AgentCard` |
| `from_card_with_client(card, client)` | Same, with a custom `reqwest::Client` |
| `from_card_with_headers(card, headers)` | Build with custom default headers (e.g. API keys) |
| `.with_auth_token(token)` | Attach a Bearer token (builder pattern) |

### Core methods

| Method | Returns |
|---|---|
| `send_message(SendMessageRequest)` | `SendMessageResponse` |
| `send_streaming_message(SendMessageRequest)` | `Stream<Item = Result<StreamResponse>>` |
| `get_task(GetTaskRequest)` | `Task` |
| `list_tasks(ListTasksRequest)` | `ListTasksResponse` |
| `cancel_task(CancelTaskRequest)` | `Task` |
| `subscribe_to_task(SubscribeToTaskRequest)` | `Stream<Item = Result<StreamResponse>>` |
| `get_extended_agent_card(GetExtendedAgentCardRequest)` | `AgentCard` |
| `fetch_extended_agent_card_if_available()` | `Option<AgentCard>` — checks capability flag first, no-ops if not advertised |

### Push notification methods

`create_task_push_notification_config`, `get_task_push_notification_config`, `list_task_push_notification_configs`, `delete_task_push_notification_config` — all guarded by `capabilities.push_notifications`.

## Transport behaviour

- Prefers **HTTP+JSON** when the agent card advertises it (`protocol_binding: "HTTP+JSON"`), falls back to **JSON-RPC 2.0** otherwise.
- Non-empty `tenant` on a request is sent as `X-A2A-Tenant` header, not as a URL path segment.
- SSE streams validate `Content-Type: text/event-stream`; JSON responses validate `Content-Type: application/json`.
- Default timeout: **30 seconds** (override by passing a custom `reqwest::Client`).
- JSON-RPC response IDs are validated against the request ID; a mismatch is a hard error.
- W3C `traceparent`/`tracestate` headers are injected automatically for distributed tracing.

## Version compatibility

| Crate version | A2A protocol version |
|---|---|
| 0.1.x | 1.0 |
| 0.2.x | 1.0 — flat `a2a_types::*` namespace, protocol compliance fixes |

## License

MIT
