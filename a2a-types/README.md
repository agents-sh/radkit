# a2a-types

Rust types for the [A2A v1.0 protocol](https://a2a-protocol.org/latest/specification/), generated from the canonical `a2a.proto` definition vendored in this repository.

## Source of truth

All types are generated at build time by [`prost`](https://github.com/tokio-rs/prost) and [`pbjson`](https://github.com/influxdata/pbjson) from `proto/a2a.proto`. There is no hand-written domain model — the proto file is the single source of truth for every field name, type, and serialization format.

## Installation

```toml
[dependencies]
a2a-types = "0.2.0"
```

## Usage

Every generated type is available directly at the crate root — no sub-module prefix required.

```rust
use a2a_types::{Message, Part, Role, SendMessageRequest, part};

let message = Message {
    message_id: "msg_123".to_string(),
    context_id: String::new(),
    task_id: String::new(),
    role: Role::User.into(),
    parts: vec![Part {
        content: Some(part::Content::Text("Hello, agent!".to_string())),
        metadata: None,
        filename: String::new(),
        media_type: "text/plain".to_string(),
    }],
    metadata: None,
    extensions: Vec::new(),
    reference_task_ids: Vec::new(),
};

let request = SendMessageRequest {
    tenant: String::new(),
    message: Some(message),
    configuration: None,
    metadata: None,
};
```

## What is included

| Category | Types |
|---|---|
| Core protocol | `Task`, `TaskStatus`, `TaskState`, `Message`, `Part`, `Artifact`, `Role` |
| Streaming events | `TaskStatusUpdateEvent`, `TaskArtifactUpdateEvent`, `StreamResponse` |
| Request / response | `SendMessageRequest`, `SendMessageResponse`, `GetTaskRequest`, `ListTasksRequest`, `CancelTaskRequest`, `SubscribeToTaskRequest` |
| Agent card | `AgentCard`, `AgentCapabilities`, `AgentSkill`, `AgentInterface`, `AgentProvider` |
| Security schemes | `SecurityScheme`, `ApiKeySecurityScheme`, `HttpAuthSecurityScheme`, `OAuth2SecurityScheme`, `OAuthFlows`, and all OAuth flow variants |
| JSON-RPC framing | `JSONRPCId`, `JSONRPCError`, `JSONRPCErrorResponse` (hand-written; JSON-RPC is not in the proto) |
| A2A error codes | `InvalidRequestError`, `InvalidParamsError`, `InternalError`, `TaskNotFoundError`, `UnsupportedOperationError` |

`oneof` payload variants live in sub-modules mirroring the proto structure:

```rust
use a2a_types::{part, send_message_response, stream_response};

// Construct a text part
let content = part::Content::Text("hello".to_string());

// Match a stream event
match response.payload {
    Some(stream_response::Payload::StatusUpdate(update)) => { /* ... */ }
    Some(stream_response::Payload::ArtifactUpdate(update)) => { /* ... */ }
    Some(stream_response::Payload::Message(msg)) => { /* ... */ }
    Some(stream_response::Payload::Task(task)) => { /* ... */ }
    None => {}
}
```

## Version compatibility

| Crate version | A2A protocol version |
|---|---|
| 0.1.x | 1.0 — hand-written types, `a2a_types::v1` namespace |
| 0.2.x | 1.0 — proto-generated types, flat `a2a_types::*` namespace |

## License

MIT
