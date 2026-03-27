#![cfg(all(feature = "runtime", not(all(target_os = "wasi", target_env = "p1"))))]

//! Web server handlers for the runtime handle.

use crate::agent::AgentDefinition;
use crate::errors::{AgentError, AgentResult};
use crate::runtime::context::AuthContext;
use crate::runtime::core::error_mapper;
use crate::runtime::core::executor::{
    ExecutorRuntime, PreparedSendMessage, RequestExecutor, TaskStream,
};
use crate::runtime::task_manager::{Task, TaskEvent};
use crate::runtime::Runtime;
use a2a_types::{
    A2ARequestPayload, AgentCard, AgentSkill, CancelTaskResponse, CancelTaskSuccessResponse,
    GetTaskResponse, GetTaskSuccessResponse, JSONRPCErrorResponse, JSONRPCId, MessageSendParams,
    SendMessageResponse, SendMessageResult, SendMessageSuccessResponse,
    SendStreamingMessageResponse, SendStreamingMessageResult, SendStreamingMessageSuccessResponse,
    TaskIdParams, TransportProtocol,
};
use async_stream::stream;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use futures::Stream;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

/// Infers a base URL from a bind address.
///
/// This handles common bind address patterns:
/// - `0.0.0.0:PORT` → `http://localhost:PORT`
/// - `127.0.0.1:PORT` → `http://localhost:PORT`
/// - `localhost:PORT` → `http://localhost:PORT`
/// - `HOST:PORT` → `http://HOST:PORT`
/// - `PORT` → `http://localhost:PORT`
fn infer_base_url(bind_address: &str) -> String {
    // Extract port from address
    let port = bind_address
        .split(':')
        .next_back()
        .and_then(|p| p.parse::<u16>().ok());

    match (bind_address, port) {
        // 0.0.0.0:PORT → localhost:PORT
        (addr, Some(port)) if addr.starts_with("0.0.0.0:") => {
            format!("http://localhost:{port}")
        }
        // 127.0.0.1:PORT → localhost:PORT
        (addr, Some(port)) if addr.starts_with("127.0.0.1:") => {
            format!("http://localhost:{port}")
        }
        // localhost:PORT → http://localhost:PORT
        (addr, Some(port)) if addr.starts_with("localhost:") => {
            format!("http://localhost:{port}")
        }
        // Just a port number → localhost:PORT
        (addr, Some(port)) if addr == port.to_string() => {
            format!("http://localhost:{port}")
        }
        // HOST:PORT → http://HOST:PORT
        (_, Some(port)) => {
            let host = bind_address
                .rsplit_once(':')
                .map_or("localhost", |(h, _)| h);
            format!("http://{host}:{port}")
        }
        // No port found → just localhost
        _ => "http://localhost".to_string(),
    }
}

pub(crate) fn build_agent_card(runtime: &Runtime, agent: &AgentDefinition) -> AgentCard {
    let base_url = runtime.configured_base_url().map_or_else(
        || {
            runtime
                .bind_address()
                .map_or_else(|| "http://localhost".to_string(), infer_base_url)
        },
        str::to_owned,
    );

    let normalized_base = base_url.trim_end_matches('/');
    let version = agent.version();

    let mut card = AgentCard::new(
        agent.name(),
        agent.description().unwrap_or_default(),
        version,
        format!("{normalized_base}/rpc"),
    );

    card.capabilities.streaming = Some(true);
    card = card.add_interface(TransportProtocol::HttpJson, normalized_base.to_string());

    card.skills = agent
        .skills()
        .iter()
        .map(|skill| AgentSkill {
            id: skill.id().to_string(),
            name: skill.name().to_string(),
            description: skill.description().to_string(),
            tags: skill
                .metadata()
                .tags
                .iter()
                .map(|tag| (*tag).to_string())
                .collect(),
            examples: skill
                .metadata()
                .examples
                .iter()
                .map(|example| (*example).to_string())
                .collect(),
            input_modes: skill
                .metadata()
                .input_modes
                .iter()
                .map(|mode| (*mode).to_string())
                .collect(),
            output_modes: skill
                .metadata()
                .output_modes
                .iter()
                .map(|mode| (*mode).to_string())
                .collect(),
            security: Vec::new(),
        })
        .collect();

    card
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_localhost_variants() {
        assert_eq!(infer_base_url("0.0.0.0:8080"), "http://localhost:8080");
        assert_eq!(infer_base_url("127.0.0.1:3000"), "http://localhost:3000");
        assert_eq!(infer_base_url("localhost:9000"), "http://localhost:9000");
        assert_eq!(infer_base_url("8080"), "http://localhost:8080");
    }

    #[test]
    fn keeps_host_when_present() {
        assert_eq!(
            infer_base_url("example.com:7000"),
            "http://example.com:7000"
        );
        assert_eq!(
            infer_base_url("api.internal:443"),
            "http://api.internal:443"
        );
    }

    #[cfg(feature = "test-support")]
    mod handler_tests {
        use super::*;
        use crate::agent::{
            Agent, OnInputResult, OnRequestResult, RegisteredSkill, SkillHandler, SkillMetadata,
        };
        use crate::errors::{AgentError, AgentResult};
        use crate::models::{Content, LlmResponse};
        use crate::runtime::context::{ProgressSender, State as SkillState};
        use crate::runtime::{AgentRuntime, RuntimeBuilder};
        use crate::test_support::FakeLlm;
        use a2a_types::{
            A2ARequest, A2ARequestPayload, JSONRPCId, Message, MessageRole, MessageSendParams,
            Part, SendMessageResponse, SendStreamingMessageResponse, SendStreamingMessageResult,
            TaskState,
        };
        use axum::extract::State;
        use axum::http::StatusCode;
        use axum::response::Response;
        use axum::Json;
        use serde_json::json;
        use std::sync::Arc;

        async fn response_bytes(response: Response) -> Vec<u8> {
            use http_body_util::BodyExt;

            let (_, body) = response.into_parts();
            let collected = body.collect().await.expect("collect body");
            collected.to_bytes().to_vec()
        }

        fn negotiation_response(skill_id: &str) -> AgentResult<LlmResponse> {
            let decision = json!({
                "type": "start_task",
                "skill_id": skill_id,
                "reasoning": "selected in test"
            });
            FakeLlm::text_response(serde_json::to_string(&decision).expect("valid JSON"))
        }

        fn create_message(
            text: &str,
            context_id: Option<String>,
            task_id: Option<String>,
        ) -> Message {
            Message {
                kind: "message".to_string(),
                message_id: uuid::Uuid::new_v4().to_string(),
                role: MessageRole::User,
                parts: vec![Part::Text {
                    text: text.to_string(),
                    metadata: None,
                }],
                context_id,
                task_id,
                reference_task_ids: Vec::new(),
                extensions: Vec::new(),
                metadata: None,
            }
        }

        struct ImmediateSkill;

        static IMMEDIATE_METADATA: SkillMetadata = SkillMetadata::new(
            "immediate-skill",
            "Immediate Skill",
            "Completes immediately",
            &[],
            &[],
            &[],
            &[],
        );

        #[cfg_attr(
            all(target_os = "wasi", target_env = "p1"),
            async_trait::async_trait(?Send)
        )]
        #[cfg_attr(
            not(all(target_os = "wasi", target_env = "p1")),
            async_trait::async_trait
        )]
        impl SkillHandler for ImmediateSkill {
            async fn on_request(
                &self,
                _state: &mut SkillState,
                _progress: &ProgressSender,
                _runtime: &dyn AgentRuntime,
                _content: Content,
            ) -> Result<OnRequestResult, AgentError> {
                Ok(OnRequestResult::Completed {
                    message: Some(Content::from_text("Done")),
                    artifacts: Vec::new(),
                })
            }

            async fn on_input_received(
                &self,
                _state: &mut SkillState,
                _progress: &ProgressSender,
                _runtime: &dyn AgentRuntime,
                _input: Content,
            ) -> Result<OnInputResult, AgentError> {
                unreachable!("immediate skill never continues");
            }
        }

        impl RegisteredSkill for ImmediateSkill {
            fn metadata() -> &'static SkillMetadata {
                &IMMEDIATE_METADATA
            }
        }

        #[tokio::test(flavor = "current_thread")]
        async fn agent_card_handler_returns_agent_card() {
            let llm =
                FakeLlm::with_responses("fake-llm", [negotiation_response(IMMEDIATE_METADATA.id)]);
            let agent = Agent::builder()
                .with_version("1.0.0")
                .with_name("Test Agent")
                .with_skill(ImmediateSkill)
                .build();
            let runtime = RuntimeBuilder::new(agent, llm)
                .base_url("http://localhost:3000")
                .build()
                .into_shared();

            let response = agent_card_handler(State(Arc::clone(&runtime))).await;

            assert_eq!(response.status(), StatusCode::OK);
            let bytes = response_bytes(response).await;
            let card: a2a_types::AgentCard = serde_json::from_slice(&bytes).expect("agent card");
            assert_eq!(card.capabilities.streaming, Some(true));
            assert_eq!(card.skills.len(), 1);
            assert!(card.url.ends_with("/rpc"));
            assert!(card
                .additional_interfaces
                .iter()
                .any(|iface| iface.url == "http://localhost:3000"));
        }

        #[tokio::test(flavor = "current_thread")]
        async fn json_rpc_handler_send_message_returns_task() {
            let llm =
                FakeLlm::with_responses("fake-llm", [negotiation_response(IMMEDIATE_METADATA.id)]);
            let agent = Agent::builder()
                .with_version("1.0.0")
                .with_name("Test Agent")
                .with_skill(ImmediateSkill)
                .build();
            let runtime = RuntimeBuilder::new(agent, llm)
                .base_url("http://localhost:3000")
                .build()
                .into_shared();

            let params = MessageSendParams {
                message: create_message("hello", None, None),
                configuration: None,
                metadata: None,
            };

            let payload = A2ARequest {
                jsonrpc: "2.0".to_string(),
                id: Some(JSONRPCId::String("req-1".into())),
                payload: A2ARequestPayload::SendMessage { params },
            };

            let response = json_rpc_handler(State(Arc::clone(&runtime)), Json(payload)).await;

            assert_eq!(response.status(), StatusCode::OK);
            let body = response_bytes(response).await;
            let parsed: SendMessageResponse =
                serde_json::from_slice(&body).expect("JSON-RPC send response");

            match parsed {
                SendMessageResponse::Success(success) => {
                    assert_eq!(success.id, Some(JSONRPCId::String("req-1".into())));
                    match &success.result {
                        a2a_types::SendMessageResult::Task(task) => {
                            assert_eq!(task.status.state, TaskState::Completed);
                            assert!(!task.history.is_empty());
                        }
                        other => panic!("expected task, got {other:?}"),
                    }
                }
                SendMessageResponse::Error(err) => {
                    panic!("unexpected error response: {err:?}");
                }
            }
        }

        #[tokio::test(flavor = "current_thread")]
        async fn message_stream_handler_emits_terminal_task_event() {
            let llm =
                FakeLlm::with_responses("fake-llm", [negotiation_response(IMMEDIATE_METADATA.id)]);
            let agent = Agent::builder()
                .with_version("1.0.0")
                .with_name("Test Agent")
                .with_skill(ImmediateSkill)
                .build();
            let runtime = Arc::new(
                RuntimeBuilder::new(agent, llm)
                    .base_url("http://localhost:3000")
                    .build(),
            );

            let params = MessageSendParams {
                message: create_message("hello", None, None),
                configuration: None,
                metadata: None,
            };

            let response = message_stream_handler(State(Arc::clone(&runtime)), Json(params)).await;

            assert_eq!(response.status(), StatusCode::OK);
            let body = response_bytes(response).await;
            let body_str = String::from_utf8(body).expect("utf8");
            #[cfg(test)]
            dbg!(&body_str);
            let mut events: Vec<SendStreamingMessageResponse> = Vec::new();
            for chunk in body_str
                .split("\n\n")
                .filter(|chunk| !chunk.trim().is_empty())
            {
                let data = chunk
                    .trim()
                    .strip_prefix("data:")
                    .map(str::trim)
                    .filter(|s| !s.is_empty());
                if let Some(json) = data {
                    events.push(
                        serde_json::from_str::<SendStreamingMessageResponse>(json).expect("event"),
                    );
                }
            }
            assert!(
                !events.is_empty(),
                "expected at least one SSE event, got none"
            );
            let last = events.last().expect("final event");
            match last {
                SendStreamingMessageResponse::Success(success) => match &success.result {
                    SendStreamingMessageResult::Task(task) => {
                        assert_eq!(task.status.state, TaskState::Completed);
                    }
                    SendStreamingMessageResult::TaskStatusUpdate(status_update) => {
                        assert_eq!(status_update.status.state, TaskState::Completed);
                        assert!(status_update.is_final, "expected final status update");
                    }
                    other => panic!("expected final task or status update event, got {other:?}"),
                },
                other @ SendStreamingMessageResponse::Error(_) => {
                    panic!("unexpected error event: {other:?}")
                }
            }
        }

        #[tokio::test(flavor = "current_thread")]
        async fn task_resubscribe_handler_returns_error_for_missing_task() {
            let llm =
                FakeLlm::with_responses("fake-llm", [negotiation_response(IMMEDIATE_METADATA.id)]);
            let agent = Agent::builder()
                .with_version("1.0.0")
                .with_name("Test Agent")
                .with_skill(ImmediateSkill)
                .build();
            let runtime = Arc::new(
                RuntimeBuilder::new(agent, llm)
                    .base_url("http://localhost:3000")
                    .build(),
            );

            let response = task_resubscribe_handler(
                State(Arc::clone(&runtime)),
                axum::extract::Path("task-123".to_string()),
            )
            .await;

            assert_eq!(response.status(), StatusCode::OK);
            let body = response_bytes(response).await;
            let parsed: SendStreamingMessageResponse =
                serde_json::from_slice(&body).expect("json error response");
            match parsed {
                SendStreamingMessageResponse::Error(err) => {
                    assert!(
                        err.error.message.contains("not found"),
                        "unexpected message: {}",
                        err.error.message
                    );
                }
                other => panic!("expected error response, got {other:?}"),
            }
        }

        #[tokio::test(flavor = "current_thread")]
        async fn build_task_with_history_collects_event_messages() {
            use crate::runtime::task_manager::{Task, TaskEvent};

            let llm = FakeLlm::with_responses("fake-llm", std::iter::empty());
            let agent = Agent::builder()
                .with_version("1.0.0")
                .with_name("Test Agent")
                .with_skill(ImmediateSkill)
                .build();
            let runtime = RuntimeBuilder::new(agent, llm).build().into_shared();
            let auth_ctx = runtime.auth().get_auth_context();
            let task_manager = runtime.task_manager();

            let mut stored_task = Task {
                id: "task-42".to_string(),
                context_id: "ctx-99".to_string(),
                status: a2a_types::TaskStatus {
                    state: TaskState::Working,
                    timestamp: Some(chrono::Utc::now().to_rfc3339()),
                    message: None,
                },
                artifacts: Vec::new(),
            };

            task_manager
                .save_task(&auth_ctx, &stored_task)
                .await
                .expect("store task");

            let user_message = Message {
                kind: a2a_types::MESSAGE_KIND.to_string(),
                message_id: "msg-1".to_string(),
                role: MessageRole::User,
                parts: vec![Part::Text {
                    text: "Hello runtime".to_string(),
                    metadata: None,
                }],
                context_id: Some(stored_task.context_id.clone()),
                task_id: Some(stored_task.id.clone()),
                reference_task_ids: Vec::new(),
                extensions: Vec::new(),
                metadata: None,
            };

            task_manager
                .add_task_event(&auth_ctx, &TaskEvent::Message(user_message.clone()))
                .await
                .expect("store user message");

            let agent_message = Message {
                kind: a2a_types::MESSAGE_KIND.to_string(),
                message_id: "msg-2".to_string(),
                role: MessageRole::Agent,
                parts: vec![Part::Text {
                    text: "Finished".to_string(),
                    metadata: None,
                }],
                context_id: Some(stored_task.context_id.clone()),
                task_id: Some(stored_task.id.clone()),
                reference_task_ids: Vec::new(),
                extensions: Vec::new(),
                metadata: None,
            };

            let status_update = a2a_types::TaskStatusUpdateEvent {
                kind: a2a_types::STATUS_UPDATE_KIND.to_string(),
                task_id: stored_task.id.clone(),
                context_id: stored_task.context_id.clone(),
                status: a2a_types::TaskStatus {
                    state: TaskState::Completed,
                    timestamp: Some(chrono::Utc::now().to_rfc3339()),
                    message: Some(agent_message.clone()),
                },
                is_final: true,
                metadata: None,
            };

            task_manager
                .add_task_event(&auth_ctx, &TaskEvent::StatusUpdate(status_update))
                .await
                .expect("store status update");

            stored_task.status.state = TaskState::Completed;
            task_manager
                .save_task(&auth_ctx, &stored_task)
                .await
                .expect("update task status");

            let task = build_task_with_history(runtime.as_ref(), &auth_ctx, &stored_task)
                .await
                .expect("task reconstruction");

            assert_eq!(task.id, stored_task.id);
            assert_eq!(task.history.len(), 2);
            assert!(task
                .history
                .iter()
                .any(|msg| msg.role == MessageRole::User && msg.parts.len() == 1));
            assert!(task
                .history
                .iter()
                .any(|msg| msg.role == MessageRole::Agent && msg.parts.len() == 1));
            assert_eq!(task.status.state, TaskState::Completed);
        }

        #[tokio::test(flavor = "current_thread")]
        async fn build_streaming_sse_sends_only_status_events() {
            use crate::runtime::core::executor::TaskStream;
            use crate::runtime::task_manager::TaskEvent;
            use axum::response::IntoResponse;

            let status_event = a2a_types::TaskStatusUpdateEvent {
                kind: a2a_types::STATUS_UPDATE_KIND.to_string(),
                task_id: "task-1".into(),
                context_id: "ctx-1".into(),
                status: a2a_types::TaskStatus {
                    state: TaskState::Completed,
                    timestamp: Some(chrono::Utc::now().to_rfc3339()),
                    message: None,
                },
                is_final: true,
                metadata: None,
            };

            let stream = TaskStream {
                initial_events: vec![TaskEvent::StatusUpdate(status_event)],
                receiver: None,
            };

            let response = build_streaming_sse(None, stream).into_response();
            assert_eq!(response.status(), StatusCode::OK);
            let body = response_bytes(response).await;
            let body_str = String::from_utf8(body).expect("utf8");
            let events: Vec<SendStreamingMessageResponse> = body_str
                .split("\n\n")
                .filter_map(|chunk| {
                    chunk
                        .trim()
                        .strip_prefix("data:")
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                })
                .map(|json| {
                    serde_json::from_str::<SendStreamingMessageResponse>(json).expect("event json")
                })
                .collect();

            assert_eq!(events.len(), 1);
            match &events[0] {
                SendStreamingMessageResponse::Success(success) => match &success.result {
                    SendStreamingMessageResult::TaskStatusUpdate(update) => {
                        assert!(update.is_final);
                    }
                    other => panic!("unexpected streaming payload: {other:?}"),
                },
                other @ SendStreamingMessageResponse::Error(_) => {
                    panic!("unexpected response: {other:?}")
                }
            }
        }
    }
}

/// Axum handler for serving the runtime's `AgentCard`.
///
/// This function handles `GET /.well-known/agent-card.json` requests.
pub async fn agent_card_handler(State(runtime): State<Arc<Runtime>>) -> Response {
    let card = build_agent_card(&runtime, runtime.agent());
    (StatusCode::OK, Json(card)).into_response()
}

/// Axum handler for JSON-RPC requests for the single configured agent.
///
/// This function handles `POST /rpc` requests.
pub async fn json_rpc_handler(
    State(runtime): State<Arc<Runtime>>,
    Json(payload): Json<a2a_types::A2ARequest>,
) -> Response {
    let executor = {
        let exec_runtime: Arc<dyn ExecutorRuntime> = runtime.clone();
        RequestExecutor::new(exec_runtime)
    };

    let request_id = payload.id.clone();

    match payload.payload {
        A2ARequestPayload::SendStreamingMessage { params } => {
            match executor.handle_message_stream(params).await {
                Ok(PreparedSendMessage::Task(stream)) => {
                    build_streaming_sse(request_id.clone(), stream).into_response()
                }
                Ok(PreparedSendMessage::Message(message)) => {
                    build_message_sse(request_id.clone(), message).into_response()
                }
                Err(error) => build_streaming_error_response(request_id.clone(), error),
            }
        }
        A2ARequestPayload::TaskResubscription { params } => {
            match executor.handle_task_resubscribe(params).await {
                Ok(stream) => build_streaming_sse(request_id.clone(), stream).into_response(),
                Err(error) => build_streaming_error_response(request_id.clone(), error),
            }
        }
        A2ARequestPayload::SendMessage { params } => {
            match executor.handle_send_message(params).await {
                Ok(result) => build_send_message_success_response(request_id.clone(), result),
                Err(error) => build_send_message_error_response(request_id.clone(), error),
            }
        }
        A2ARequestPayload::GetTask { params } => match executor.handle_get_task(params).await {
            Ok(task) => build_get_task_success_response(request_id.clone(), task),
            Err(error) => build_get_task_error_response(request_id.clone(), error),
        },
        A2ARequestPayload::CancelTask { params } => {
            match executor.handle_cancel_task(params).await {
                Ok(task) => build_cancel_task_success_response(request_id.clone(), task),
                Err(error) => build_cancel_task_error_response(request_id.clone(), error),
            }
        }
        _ => AgentError::NotImplemented {
            feature: "This RPC method".to_string(),
        }
        .into_response(),
    }
}

/// Axum handler for the `message/stream` endpoint.
pub async fn message_stream_handler(
    State(runtime): State<Arc<Runtime>>,
    Json(params): Json<MessageSendParams>,
) -> Response {
    let executor = {
        let exec_runtime: Arc<dyn ExecutorRuntime> = runtime.clone();
        RequestExecutor::new(exec_runtime)
    };

    match executor.handle_message_stream(params).await {
        Ok(PreparedSendMessage::Task(stream)) => build_streaming_sse(None, stream).into_response(),
        Ok(PreparedSendMessage::Message(message)) => {
            build_message_sse(None, message).into_response()
        }
        Err(error) => build_streaming_error_response(None, error),
    }
}

/// Axum handler for the `tasks/resubscribe` endpoint.
pub async fn task_resubscribe_handler(
    State(runtime): State<Arc<Runtime>>,
    Path(task_id): Path<String>,
) -> Response {
    let executor = {
        let exec_runtime: Arc<dyn ExecutorRuntime> = runtime.clone();
        RequestExecutor::new(exec_runtime)
    };
    let params = TaskIdParams {
        id: task_id,
        metadata: None,
    };

    match executor.handle_task_resubscribe(params).await {
        Ok(stream) => build_streaming_sse(None, stream).into_response(),
        Err(error) => build_streaming_error_response(None, error),
    }
}
fn build_streaming_sse(
    request_id: Option<JSONRPCId>,
    stream_state: TaskStream,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    #[cfg(test)]
    let init_event_len = stream_state.initial_events.len();
    #[cfg(test)]
    let has_receiver = stream_state.receiver.is_some();
    let mut initial_events = stream_state.initial_events.into_iter();
    #[cfg(test)]
    eprintln!("stream init events={init_event_len} receiver={has_receiver}");
    let id_clone = request_id.clone();
    let receiver = stream_state.receiver;
    let stream = stream! {
        let mut final_seen = false;

        for event in initial_events.by_ref() {
            if let Some((evt, is_final)) = task_event_to_event(request_id.as_ref(), event) {
                yield Ok(evt);
                if is_final {
                    final_seen = true;
                    break;
                }
            }
        }

        if !final_seen {
            if let Some(mut rx) = receiver {
                while let Some(event) = rx.recv().await {
                    if let Some((evt, is_final)) = task_event_to_event(id_clone.as_ref(), event) {
                        yield Ok(evt);
                        if is_final {
                            break;
                        }
                    }
                }
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

fn build_message_sse(
    request_id: Option<JSONRPCId>,
    message: a2a_types::Message,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = stream! {
        let response = SendStreamingMessageResponse::Success(Box::new(
            SendStreamingMessageSuccessResponse {
                jsonrpc: "2.0".to_string(),
                result: SendStreamingMessageResult::Message(message),
                id: request_id.clone(),
            },
        ));

        if let Ok(data) = serde_json::to_string(&response) {
            yield Ok(Event::default().data(data));
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

fn build_send_message_success_response(
    request_id: Option<JSONRPCId>,
    result: SendMessageResult,
) -> Response {
    let response = SendMessageResponse::Success(Box::new(SendMessageSuccessResponse {
        jsonrpc: "2.0".to_string(),
        result,
        id: request_id,
    }));

    Json(response).into_response()
}

fn build_send_message_error_response(request_id: Option<JSONRPCId>, error: AgentError) -> Response {
    let response = SendMessageResponse::Error(JSONRPCErrorResponse {
        jsonrpc: "2.0".to_string(),
        error: error_mapper::to_jsonrpc_error(error),
        id: request_id,
    });

    Json(response).into_response()
}

fn build_get_task_success_response(
    request_id: Option<JSONRPCId>,
    task: a2a_types::Task,
) -> Response {
    let response = GetTaskResponse::Success(Box::new(GetTaskSuccessResponse {
        jsonrpc: "2.0".to_string(),
        result: task,
        id: request_id,
    }));

    Json(response).into_response()
}

fn build_get_task_error_response(request_id: Option<JSONRPCId>, error: AgentError) -> Response {
    let response = GetTaskResponse::Error(JSONRPCErrorResponse {
        jsonrpc: "2.0".to_string(),
        error: error_mapper::to_jsonrpc_error(error),
        id: request_id,
    });

    Json(response).into_response()
}

fn build_cancel_task_success_response(
    request_id: Option<JSONRPCId>,
    task: a2a_types::Task,
) -> Response {
    let response = CancelTaskResponse::Success(Box::new(CancelTaskSuccessResponse {
        jsonrpc: "2.0".to_string(),
        result: task,
        id: request_id,
    }));

    Json(response).into_response()
}

fn build_cancel_task_error_response(request_id: Option<JSONRPCId>, error: AgentError) -> Response {
    let response = CancelTaskResponse::Error(JSONRPCErrorResponse {
        jsonrpc: "2.0".to_string(),
        error: error_mapper::to_jsonrpc_error(error),
        id: request_id,
    });

    Json(response).into_response()
}

fn build_streaming_error_response(request_id: Option<JSONRPCId>, error: AgentError) -> Response {
    let response = SendStreamingMessageResponse::Error(JSONRPCErrorResponse {
        jsonrpc: "2.0".to_string(),
        error: error_mapper::to_jsonrpc_error(error),
        id: request_id,
    });

    Json(response).into_response()
}

fn task_event_to_event(
    request_id: Option<&JSONRPCId>,
    event: crate::runtime::task_manager::TaskEvent,
) -> Option<(Event, bool)> {
    let (result, is_final) = convert_task_event(event);

    let response =
        SendStreamingMessageResponse::Success(Box::new(SendStreamingMessageSuccessResponse {
            jsonrpc: "2.0".to_string(),
            result,
            id: request_id.cloned(),
        }));

    serde_json::to_string(&response)
        .ok()
        .map(|data| (Event::default().data(data), is_final))
}

fn convert_task_event(event: TaskEvent) -> (SendStreamingMessageResult, bool) {
    match event {
        TaskEvent::StatusUpdate(update) => {
            let is_final = update.is_final;
            (
                SendStreamingMessageResult::TaskStatusUpdate(update),
                is_final,
            )
        }
        TaskEvent::ArtifactUpdate(update) => (
            SendStreamingMessageResult::TaskArtifactUpdate(update),
            false,
        ),
        TaskEvent::Message(message) => (SendStreamingMessageResult::Message(message), false),
    }
}

/// Agent information returned to the development UI.
///
/// This struct contains essential agent metadata including the ID,
/// which is not part of the A2A `AgentCard` specification.
#[cfg(feature = "dev-ui")]
#[derive(serde::Serialize, ts_rs::TS)]
#[ts(export, export_to = "../ui/src/types/")]
pub struct AgentInfo {
    /// Human-readable agent name
    pub name: String,
    /// Agent version string
    pub version: String,
    /// Optional description of agent capabilities
    #[ts(optional)]
    pub description: Option<String>,
    /// Number of skills registered with this agent
    pub skill_count: usize,
}

#[cfg(feature = "dev-ui")]
impl AgentInfo {
    /// Create an `AgentInfo` from an `AgentDefinition`
    fn from_agent_definition(agent: &crate::agent::AgentDefinition) -> Self {
        Self {
            name: agent.name().to_string(),
            version: agent.version().to_string(),
            description: agent.description().map(String::from),
            skill_count: agent.skills.len(),
        }
    }
}

/// Handler for returning the single registered agent (dev-ui only).
#[cfg(feature = "dev-ui")]
pub async fn agent_info_handler(State(runtime): State<Arc<Runtime>>) -> Json<AgentInfo> {
    Json(AgentInfo::from_agent_definition(runtime.agent()))
}

/// Handler for listing context IDs for an agent (dev-ui only)
///
/// Returns a JSON array of context IDs that have tasks associated with them.
#[cfg(feature = "dev-ui")]
pub async fn list_contexts_handler(State(runtime): State<Arc<Runtime>>) -> Response {
    let auth_ctx = runtime.auth().get_auth_context();
    match runtime.task_manager().list_context_ids(&auth_ctx).await {
        Ok(context_ids) => Json::<Vec<String>>(context_ids).into_response(),
        Err(err) => err.into_response(),
    }
}

#[cfg(feature = "dev-ui")]
#[derive(serde::Serialize)]
struct UiTaskSummary {
    task: a2a_types::Task,
    #[serde(skip_serializing_if = "Option::is_none")]
    skill_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pending_slot: Option<serde_json::Value>,
}

#[cfg(feature = "dev-ui")]
#[derive(serde::Serialize)]
struct UiTaskEvent {
    result: SendStreamingMessageResult,
    is_final: bool,
}

#[cfg(feature = "dev-ui")]
#[derive(serde::Serialize)]
struct UiTaskEventsResponse {
    events: Vec<UiTaskEvent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    task: Option<a2a_types::Task>,
}

#[cfg(feature = "dev-ui")]
pub async fn context_tasks_handler(
    State(runtime): State<Arc<Runtime>>,
    Path(context_id): Path<String>,
) -> Response {
    match fetch_context_tasks(&runtime, &context_id).await {
        Ok(tasks) => Json(tasks).into_response(),
        Err(err) => err.into_response(),
    }
}

#[cfg(feature = "dev-ui")]
pub async fn task_events_handler(
    State(runtime): State<Arc<Runtime>>,
    Path(task_id): Path<String>,
) -> Response {
    match fetch_task_events(&runtime, &task_id).await {
        Ok(body) => Json(body).into_response(),
        Err(err) => err.into_response(),
    }
}

#[cfg(feature = "dev-ui")]
#[derive(serde::Serialize)]
#[cfg_attr(feature = "dev-ui", derive(ts_rs::TS))]
#[cfg_attr(feature = "dev-ui", ts(export, export_to = "../ui/src/types/"))]
pub struct StateTransition {
    from_state: Option<String>,
    to_state: String,
    timestamp: String,
    trigger: String,
}

#[cfg(feature = "dev-ui")]
#[derive(serde::Serialize)]
#[cfg_attr(feature = "dev-ui", derive(ts_rs::TS))]
#[cfg_attr(feature = "dev-ui", ts(export, export_to = "../ui/src/types/"))]
pub struct TaskTransitionsResponse {
    transitions: Vec<StateTransition>,
}

#[cfg(feature = "dev-ui")]
pub async fn task_transitions_handler(
    State(runtime): State<Arc<Runtime>>,
    Path(task_id): Path<String>,
) -> Response {
    match fetch_task_transitions(&runtime, &task_id).await {
        Ok(transitions) => Json(transitions).into_response(),
        Err(err) => err.into_response(),
    }
}

#[cfg(feature = "dev-ui")]
async fn fetch_task_transitions(
    runtime: &Arc<Runtime>,
    task_id: &str,
) -> AgentResult<TaskTransitionsResponse> {
    use a2a_types::TaskState;

    let auth_ctx = runtime.auth().get_auth_context();
    let events = runtime
        .task_manager()
        .get_task_events(&auth_ctx, task_id)
        .await?;

    let mut transitions = Vec::new();
    let mut prev_state: Option<TaskState> = None;

    for event in events {
        if let TaskEvent::StatusUpdate(update) = event {
            let current_state = update.status.state.clone();

            // Determine trigger type based on state transition
            let trigger = match (&prev_state, &current_state) {
                // Initial transition to Submitted or Working
                (None, TaskState::Submitted | TaskState::Working) => "on_request",
                // Transition from InputRequired back to Working
                (Some(TaskState::InputRequired), TaskState::Working) => "on_input_received",
                // All other transitions
                _ => "status_update",
            };

            transitions.push(StateTransition {
                from_state: prev_state.as_ref().map(|s| format!("{s:?}")),
                to_state: format!("{current_state:?}"),
                timestamp: update
                    .status
                    .timestamp
                    .unwrap_or_else(|| chrono::Utc::now().to_rfc3339()),
                trigger: trigger.to_string(),
            });

            prev_state = Some(current_state);
        }
    }

    Ok(TaskTransitionsResponse { transitions })
}

#[cfg(feature = "dev-ui")]
async fn fetch_context_tasks(
    runtime: &Arc<Runtime>,
    context_id: &str,
) -> AgentResult<Vec<UiTaskSummary>> {
    let auth_ctx = runtime.auth().get_auth_context();
    let task_ids = runtime
        .task_manager()
        .list_task_ids(&auth_ctx, Some(context_id))
        .await?;

    let mut summaries = Vec::new();
    for task_id in task_ids {
        if let Some(stored_task) = runtime.task_manager().get_task(&auth_ctx, &task_id).await? {
            let task = build_task_with_history(runtime.as_ref(), &auth_ctx, &stored_task).await?;
            let skill_id = runtime
                .task_manager()
                .get_task_skill(&auth_ctx, &task_id)
                .await?;
            let pending_slot = runtime
                .task_manager()
                .load_task_state(&auth_ctx, &task_id)
                .await?
                .and_then(|state| {
                    state
                        .current_slot()
                        .and_then(|slot| slot.deserialize::<serde_json::Value>().ok())
                });

            summaries.push(UiTaskSummary {
                task,
                skill_id,
                pending_slot,
            });
        }
    }

    summaries.sort_by(|a, b| b.task.status.timestamp.cmp(&a.task.status.timestamp));

    Ok(summaries)
}

#[cfg(feature = "dev-ui")]
async fn fetch_task_events(
    runtime: &Arc<Runtime>,
    task_id: &str,
) -> AgentResult<UiTaskEventsResponse> {
    let auth_ctx = runtime.auth().get_auth_context();
    let events = runtime
        .task_manager()
        .get_task_events(&auth_ctx, task_id)
        .await?;

    let mut serialized_events = Vec::new();
    let mut final_seen = false;
    for event in events {
        let (result, is_final) = convert_task_event(event);
        if is_final {
            final_seen = true;
        }
        serialized_events.push(UiTaskEvent { result, is_final });
    }

    let stored_task = runtime.task_manager().get_task(&auth_ctx, task_id).await?;

    let task_snapshot = if let Some(task) = stored_task {
        Some(build_task_with_history(runtime.as_ref(), &auth_ctx, &task).await?)
    } else {
        None
    };

    if final_seen {
        if let Some(task) = &task_snapshot {
            serialized_events.push(UiTaskEvent {
                result: SendStreamingMessageResult::Task(task.clone()),
                is_final: true,
            });
        }
    }

    Ok(UiTaskEventsResponse {
        events: serialized_events,
        task: task_snapshot,
    })
}

#[cfg_attr(not(feature = "dev-ui"), allow(dead_code))]
async fn build_task_with_history(
    runtime: &Runtime,
    auth_ctx: &AuthContext,
    stored_task: &Task,
) -> AgentResult<a2a_types::Task> {
    let events = runtime
        .task_manager()
        .get_task_events(auth_ctx, &stored_task.id)
        .await?;

    let history: Vec<a2a_types::Message> = events
        .into_iter()
        .filter_map(|event| match event {
            TaskEvent::Message(msg) => Some(msg),
            TaskEvent::StatusUpdate(update) => update.status.message,
            TaskEvent::ArtifactUpdate(_) => None,
        })
        .collect();

    Ok(a2a_types::Task {
        id: stored_task.id.clone(),
        context_id: stored_task.context_id.clone(),
        status: stored_task.status.clone(),
        artifacts: stored_task.artifacts.clone(),
        history,
        kind: a2a_types::TASK_KIND.to_string(),
        metadata: None,
    })
}
