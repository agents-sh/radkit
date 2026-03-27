#![cfg(all(feature = "runtime", not(all(target_os = "wasi", target_env = "p1"))))]

//! Web server handlers for the runtime handle.

use crate::agent::AgentDefinition;
use crate::errors::{AgentError, AgentResult};
use crate::runtime::context::AuthContext;
use crate::runtime::core::error_mapper;
use crate::runtime::core::executor::{
    ExecutorRuntime, PreparedSendMessage, RequestExecutor, TaskStream,
};
use crate::runtime::task_manager::{ListTasksFilter, PaginatedResult, Task, TaskEvent};
use crate::runtime::{AgentRuntime, Runtime};
use a2a_types::{self as v1, JSONRPCErrorResponse, JSONRPCId};
use async_stream::stream;
use axum::{
    body::Bytes,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use futures::Stream;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
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

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    #[serde(default)]
    id: Option<JSONRPCId>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

#[derive(Debug, Deserialize, Serialize)]
struct JsonRpcSuccessResponse<T> {
    jsonrpc: String,
    result: T,
    id: Option<JSONRPCId>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GetTaskQuery {
    #[serde(default, alias = "historyLength")]
    history_length: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ListTasksQuery {
    #[serde(default, alias = "contextId")]
    context_id: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default, alias = "pageSize")]
    page_size: Option<i32>,
    #[serde(default, alias = "pageToken")]
    page_token: Option<String>,
    #[serde(default, alias = "historyLength")]
    history_length: Option<i32>,
    #[serde(default, alias = "statusTimestampAfter")]
    status_timestamp_after: Option<String>,
    #[serde(default, alias = "includeArtifacts")]
    include_artifacts: Option<bool>,
}

fn parse_json_body<T: serde::de::DeserializeOwned>(
    body: &Bytes,
    context: &'static str,
) -> AgentResult<T> {
    serde_json::from_slice(body).map_err(|error| AgentError::Serialization {
        format: "json".to_string(),
        reason: format!("failed to parse {context}: {error}"),
    })
}

fn parse_send_message_request(body: &Bytes) -> AgentResult<v1::SendMessageRequest> {
    parse_json_body(body, "SendMessage request")
}

fn task_state_matches_filter(task: &v1::Task, status: &str) -> Result<bool, AgentError> {
    use a2a_types::TaskState;
    let normalized = status.trim().to_ascii_uppercase();
    let expected = match normalized.as_str() {
        "TASK_STATE_SUBMITTED" | "SUBMITTED" => TaskState::Submitted,
        "TASK_STATE_WORKING" | "WORKING" => TaskState::Working,
        "TASK_STATE_COMPLETED" | "COMPLETED" => TaskState::Completed,
        "TASK_STATE_FAILED" | "FAILED" => TaskState::Failed,
        "TASK_STATE_CANCELED" | "CANCELED" => TaskState::Canceled,
        "TASK_STATE_INPUT_REQUIRED" | "INPUT_REQUIRED" => TaskState::InputRequired,
        "TASK_STATE_REJECTED" | "REJECTED" => TaskState::Rejected,
        "TASK_STATE_AUTH_REQUIRED" | "AUTH_REQUIRED" => TaskState::AuthRequired,
        "TASK_STATE_UNSPECIFIED" | "UNKNOWN" => TaskState::Unspecified,
        other => {
            return Err(AgentError::Validation {
                field: "status".to_string(),
                reason: format!("unsupported task state `{other}`"),
            });
        }
    };

    Ok(task
        .status
        .as_ref()
        .is_some_and(|s| s.state == expected as i32))
}

fn timestamp_after_filter(task: &v1::Task, after: &str) -> Result<bool, AgentError> {
    let Some(task_timestamp) = task.status.as_ref().and_then(|s| s.timestamp.as_ref()) else {
        return Ok(false);
    };
    let task_dt = chrono::DateTime::from_timestamp(
        task_timestamp.seconds,
        task_timestamp.nanos.cast_unsigned(),
    )
    .ok_or_else(|| AgentError::Validation {
        field: "task.status.timestamp".to_string(),
        reason: "invalid timestamp".to_string(),
    })?;
    let after_dt =
        chrono::DateTime::parse_from_rfc3339(after).map_err(|error| AgentError::Validation {
            field: "status_timestamp_after".to_string(),
            reason: error.to_string(),
        })?;
    Ok(task_dt.timestamp() >= after_dt.timestamp())
}

fn truncate_history(task: &mut v1::Task, history_length: Option<i32>) {
    if let Some(history_length) = history_length {
        let history_length = usize::try_from(history_length.max(0)).unwrap_or_default();
        if task.history.len() > history_length {
            let split_index = task.history.len().saturating_sub(history_length);
            task.history = task.history.split_off(split_index);
        }
    }
}

pub(crate) fn build_agent_card(runtime: &Runtime, agent: &AgentDefinition) -> v1::AgentCard {
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

    v1::AgentCard {
        name: agent.name().to_string(),
        description: agent.description().unwrap_or_default().to_string(),
        supported_interfaces: vec![
            v1::AgentInterface {
                url: format!("{normalized_base}/rpc"),
                protocol_binding: "JSONRPC".to_string(),
                tenant: String::new(),
                protocol_version: "1.0".to_string(),
            },
            v1::AgentInterface {
                url: normalized_base.to_string(),
                protocol_binding: "HTTP+JSON".to_string(),
                tenant: String::new(),
                protocol_version: "1.0".to_string(),
            },
        ],
        provider: None,
        version: version.to_string(),
        documentation_url: None,
        capabilities: Some(v1::AgentCapabilities {
            streaming: Some(true),
            push_notifications: Some(false),
            extensions: Vec::new(),
            extended_agent_card: Some(true),
        }),
        security_schemes: HashMap::new(),
        security_requirements: Vec::new(),
        default_input_modes: vec!["text/plain".to_string()],
        default_output_modes: vec!["text/plain".to_string()],
        skills: agent
            .skills()
            .iter()
            .map(|skill| v1::AgentSkill {
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
                security_requirements: Vec::new(),
            })
            .collect(),
        signatures: Vec::new(),
        icon_url: None,
    }
}

/// Axum handler for serving the runtime's `AgentCard`.
///
/// This function handles `GET /.well-known/agent-card.json` requests.
pub(crate) async fn agent_card_handler(State(runtime): State<Arc<Runtime>>) -> Response {
    let card = build_agent_card(&runtime, runtime.agent());
    (StatusCode::OK, Json(card)).into_response()
}

/// Axum handler for the `POST /message:send` endpoint.
pub(crate) async fn message_send_handler(
    State(runtime): State<Arc<Runtime>>,
    body: Bytes,
) -> Response {
    let executor = {
        let exec_runtime: Arc<dyn ExecutorRuntime> = runtime.clone();
        RequestExecutor::new(exec_runtime)
    };

    let params = match parse_send_message_request(&body) {
        Ok(params) => params,
        Err(error) => return error.into_response(),
    };

    match executor.handle_send_message(params).await {
        Ok(result) => build_v1_send_message_http_response(result),
        Err(error) => error.into_response(),
    }
}

/// Axum handler for the `GET /tasks/{id}` endpoint.
pub(crate) async fn get_task_handler(
    State(runtime): State<Arc<Runtime>>,
    Path(task_id): Path<String>,
    Query(query): Query<GetTaskQuery>,
) -> Response {
    let executor = {
        let exec_runtime: Arc<dyn ExecutorRuntime> = runtime.clone();
        RequestExecutor::new(exec_runtime)
    };

    let params = v1::GetTaskRequest {
        id: task_id,
        history_length: query.history_length,
        tenant: String::new(),
    };

    match executor.handle_get_task(params).await {
        Ok(task) => build_v1_task_http_response(task),
        Err(error) => error.into_response(),
    }
}

/// Router entrypoint for task-related GET endpoints under `/tasks/*`.
pub(crate) async fn task_get_route_handler(
    State(runtime): State<Arc<Runtime>>,
    Path(task_path): Path<String>,
    query: Query<GetTaskQuery>,
) -> Response {
    if let Some(task_id) = task_path.strip_suffix(":subscribe") {
        return subscribe_task_handler(State(runtime), Path(task_id.to_string())).await;
    }

    get_task_handler(State(runtime), Path(task_path), query).await
}

/// Axum handler for the `GET /tasks` endpoint.
pub(crate) async fn list_tasks_handler(
    State(runtime): State<Arc<Runtime>>,
    Query(query): Query<ListTasksQuery>,
) -> Response {
    match fetch_v1_tasks(&runtime, query).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

/// Axum handler for the `POST /tasks/{id}:cancel` endpoint.
pub(crate) async fn cancel_task_handler(
    State(runtime): State<Arc<Runtime>>,
    Path(task_id): Path<String>,
) -> Response {
    let executor = {
        let exec_runtime: Arc<dyn ExecutorRuntime> = runtime.clone();
        RequestExecutor::new(exec_runtime)
    };
    let params = v1::CancelTaskRequest {
        id: task_id,
        metadata: None,
        tenant: String::new(),
    };

    match executor.handle_cancel_task(params).await {
        Ok(task) => build_v1_task_http_response(task),
        Err(error) => error.into_response(),
    }
}

/// Router entrypoint for task-related POST endpoints under `/tasks/*`.
pub(crate) async fn task_post_route_handler(
    State(runtime): State<Arc<Runtime>>,
    Path(task_path): Path<String>,
) -> Response {
    if let Some(task_id) = task_path.strip_suffix(":cancel") {
        return cancel_task_handler(State(runtime), Path(task_id.to_string())).await;
    }

    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({
            "error": format!("unsupported task route `/tasks/{task_path}`"),
        })),
    )
        .into_response()
}

/// Axum handler for the `GET /extendedAgentCard` endpoint.
pub(crate) async fn extended_agent_card_handler(State(runtime): State<Arc<Runtime>>) -> Response {
    agent_card_handler(State(runtime)).await
}

/// Axum handler for JSON-RPC requests for the single configured agent.
///
/// This function handles `POST /rpc` requests.
// Centralized dispatch keeps the protocol boundary readable even though the match is long.
#[allow(clippy::too_many_lines)]
pub(crate) async fn json_rpc_handler(State(runtime): State<Arc<Runtime>>, body: Bytes) -> Response {
    let executor = {
        let exec_runtime: Arc<dyn ExecutorRuntime> = runtime.clone();
        RequestExecutor::new(exec_runtime)
    };

    let payload: JsonRpcRequest = match parse_json_body(&body, "JSON-RPC request") {
        Ok(payload) => payload,
        Err(error) => return build_jsonrpc_error_response(None, error),
    };

    if payload.jsonrpc != "2.0" {
        return build_jsonrpc_error_response(
            payload.id.clone(),
            AgentError::Validation {
                field: "jsonrpc".to_string(),
                reason: format!("expected `2.0`, got `{}`", payload.jsonrpc),
            },
        );
    }

    let request_id = payload.id.clone();

    match payload.method.as_str() {
        "SendMessage" => {
            let request =
                match parse_rpc_params::<v1::SendMessageRequest>(payload.params, "SendMessage") {
                    Ok(request) => request,
                    Err(error) => return build_jsonrpc_error_response(request_id.clone(), error),
                };

            match executor.handle_send_message(request).await {
                Ok(result) => build_v1_send_message_rpc_response(request_id.clone(), result),
                Err(error) => build_jsonrpc_error_response(request_id.clone(), error),
            }
        }
        "SendStreamingMessage" => {
            let request = match parse_rpc_params::<v1::SendMessageRequest>(
                payload.params,
                "SendStreamingMessage",
            ) {
                Ok(request) => request,
                Err(error) => return build_jsonrpc_error_response(request_id.clone(), error),
            };

            match executor.handle_message_stream(request).await {
                Ok(PreparedSendMessage::Task(stream)) => {
                    build_v1_jsonrpc_streaming_sse(request_id.clone(), stream).into_response()
                }
                Ok(PreparedSendMessage::Message(message)) => {
                    build_v1_jsonrpc_message_sse(request_id.clone(), message).into_response()
                }
                Err(error) => build_jsonrpc_error_response(request_id.clone(), error),
            }
        }
        "GetTask" => {
            let request = match parse_rpc_params::<v1::GetTaskRequest>(payload.params, "GetTask") {
                Ok(request) => request,
                Err(error) => return build_jsonrpc_error_response(request_id.clone(), error),
            };

            match executor.handle_get_task(request).await {
                Ok(task) => build_v1_task_rpc_response(request_id.clone(), task),
                Err(error) => build_jsonrpc_error_response(request_id.clone(), error),
            }
        }
        "CancelTask" => {
            let request =
                match parse_rpc_params::<v1::CancelTaskRequest>(payload.params, "CancelTask") {
                    Ok(request) => request,
                    Err(error) => return build_jsonrpc_error_response(request_id.clone(), error),
                };

            match executor.handle_cancel_task(request).await {
                Ok(task) => build_v1_task_rpc_response(request_id.clone(), task),
                Err(error) => build_jsonrpc_error_response(request_id.clone(), error),
            }
        }
        "SubscribeToTask" => {
            let request = match parse_rpc_params::<v1::SubscribeToTaskRequest>(
                payload.params,
                "SubscribeToTask",
            ) {
                Ok(request) => request,
                Err(error) => return build_jsonrpc_error_response(request_id.clone(), error),
            };

            match executor.handle_task_resubscribe(request).await {
                Ok(stream) => {
                    build_v1_jsonrpc_streaming_sse(request_id.clone(), stream).into_response()
                }
                Err(error) => build_jsonrpc_error_response(request_id.clone(), error),
            }
        }
        "GetExtendedAgentCard" => {
            let card = build_agent_card(&runtime, runtime.agent());
            build_jsonrpc_success_response(request_id.clone(), card)
        }
        _ => build_jsonrpc_error_response(
            request_id.clone(),
            AgentError::NotImplemented {
                feature: format!("JSON-RPC method `{}`", payload.method),
            },
        ),
    }
}

/// Axum handler for the `message/stream` endpoint.
pub(crate) async fn message_stream_handler(
    State(runtime): State<Arc<Runtime>>,
    body: Bytes,
) -> Response {
    let executor = {
        let exec_runtime: Arc<dyn ExecutorRuntime> = runtime.clone();
        RequestExecutor::new(exec_runtime)
    };

    let params = match parse_send_message_request(&body) {
        Ok(params) => params,
        Err(error) => return error.into_response(),
    };

    match executor.handle_message_stream(params).await {
        Ok(PreparedSendMessage::Task(stream)) => {
            build_v1_http_streaming_sse(stream).into_response()
        }
        Ok(PreparedSendMessage::Message(message)) => {
            build_v1_http_message_sse(message).into_response()
        }
        Err(error) => error.into_response(),
    }
}

/// Axum handler for the `tasks/{id}:subscribe` endpoint.
pub(crate) async fn subscribe_task_handler(
    State(runtime): State<Arc<Runtime>>,
    Path(task_id): Path<String>,
) -> Response {
    let executor = {
        let exec_runtime: Arc<dyn ExecutorRuntime> = runtime.clone();
        RequestExecutor::new(exec_runtime)
    };
    let params = v1::SubscribeToTaskRequest {
        id: task_id,
        tenant: String::new(),
    };

    match executor.handle_task_resubscribe(params).await {
        Ok(stream) => build_v1_http_streaming_sse(stream).into_response(),
        Err(error) => error.into_response(),
    }
}

fn parse_rpc_params<T: serde::de::DeserializeOwned>(
    params: Option<Value>,
    method: &'static str,
) -> AgentResult<T> {
    serde_json::from_value(params.unwrap_or(Value::Null)).map_err(|error| {
        AgentError::Serialization {
            format: "json".to_string(),
            reason: format!("failed to parse params for {method}: {error}"),
        }
    })
}

fn build_jsonrpc_success_response<T: Serialize>(
    request_id: Option<JSONRPCId>,
    result: T,
) -> Response {
    Json(JsonRpcSuccessResponse {
        jsonrpc: "2.0".to_string(),
        result,
        id: request_id,
    })
    .into_response()
}

fn build_jsonrpc_error_response(request_id: Option<JSONRPCId>, error: AgentError) -> Response {
    Json(JSONRPCErrorResponse {
        jsonrpc: "2.0".to_string(),
        error: error_mapper::to_jsonrpc_error(error),
        id: request_id,
    })
    .into_response()
}

fn build_v1_send_message_http_response(result: v1::SendMessageResponse) -> Response {
    Json(result).into_response()
}

fn build_v1_send_message_rpc_response(
    request_id: Option<JSONRPCId>,
    result: v1::SendMessageResponse,
) -> Response {
    build_jsonrpc_success_response(request_id, result)
}

fn build_v1_task_http_response(task: v1::Task) -> Response {
    Json(task).into_response()
}

fn build_v1_task_rpc_response(request_id: Option<JSONRPCId>, task: v1::Task) -> Response {
    build_jsonrpc_success_response::<v1::Task>(request_id, task)
}

fn task_event_to_v1_response(event: TaskEvent) -> (v1::StreamResponse, bool) {
    let (payload, is_final) = match event {
        TaskEvent::StatusUpdate(update) => {
            let is_final = update.status.as_ref().is_some_and(|s| {
                use a2a_types::TaskState;
                matches!(
                    TaskState::try_from(s.state),
                    Ok(TaskState::Completed
                        | TaskState::Failed
                        | TaskState::Rejected
                        | TaskState::Canceled
                        | TaskState::Unspecified)
                )
            });
            (v1::stream_response::Payload::StatusUpdate(update), is_final)
        }
        TaskEvent::ArtifactUpdate(update) => {
            (v1::stream_response::Payload::ArtifactUpdate(update), false)
        }
        TaskEvent::Message(message) => (v1::stream_response::Payload::Message(message), false),
    };
    (
        v1::StreamResponse {
            payload: Some(payload),
        },
        is_final,
    )
}

fn build_v1_jsonrpc_streaming_sse(
    request_id: Option<JSONRPCId>,
    stream_state: TaskStream,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    build_v1_streaming_sse(stream_state, move |result| {
        serde_json::to_string(&JsonRpcSuccessResponse {
            jsonrpc: "2.0".to_string(),
            result,
            id: request_id.clone(),
        })
        .ok()
    })
}

fn build_v1_http_streaming_sse(
    stream_state: TaskStream,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    build_v1_streaming_sse(stream_state, |result| serde_json::to_string(&result).ok())
}

fn build_v1_jsonrpc_message_sse(
    request_id: Option<JSONRPCId>,
    message: v1::Message,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let response = v1::StreamResponse {
        payload: Some(v1::stream_response::Payload::Message(message)),
    };
    let payload = serde_json::to_string(&JsonRpcSuccessResponse {
        jsonrpc: "2.0".to_string(),
        result: response,
        id: request_id,
    })
    .ok();
    let stream = stream! {
        if let Some(data) = payload {
            yield Ok(Event::default().data(data));
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

fn build_v1_http_message_sse(
    message: v1::Message,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let response = v1::StreamResponse {
        payload: Some(v1::stream_response::Payload::Message(message)),
    };
    let payload = serde_json::to_string(&response).ok();
    let stream = stream! {
        if let Some(data) = payload {
            yield Ok(Event::default().data(data));
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

fn build_v1_streaming_sse<F>(
    stream_state: TaskStream,
    serializer: F,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>>
where
    F: Fn(v1::StreamResponse) -> Option<String> + Send + Sync + 'static,
{
    let mut initial_events = stream_state.initial_events.into_iter();
    let receiver = stream_state.receiver;
    let serializer = Arc::new(serializer);
    let stream = stream! {
        let mut final_seen = false;

        for event in initial_events.by_ref() {
            let (result, is_final) = task_event_to_v1_response(event);
            if let Some(data) = serializer(result) {
                yield Ok(Event::default().data(data));
            }
            if is_final {
                final_seen = true;
                break;
            }
        }

        if !final_seen {
            if let Some(mut rx) = receiver {
                while let Some(event) = rx.recv().await {
                    let (result, is_final) = task_event_to_v1_response(event);
                    if let Some(data) = serializer(result) {
                        yield Ok(Event::default().data(data));
                    }
                    if is_final {
                        break;
                    }
                }
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

async fn fetch_v1_tasks(
    runtime: &Arc<Runtime>,
    query: ListTasksQuery,
) -> AgentResult<v1::ListTasksResponse> {
    let auth_ctx = runtime.auth().get_auth_context();

    // Fetch broadly, then apply v1-specific filters/projections locally.
    let all_tasks: PaginatedResult<Task> = runtime
        .task_manager()
        .list_tasks(
            &auth_ctx,
            &ListTasksFilter {
                context_id: query.context_id.as_deref(),
                page_size: Some(10_000),
                page_token: None,
            },
        )
        .await?;

    let include_artifacts = query.include_artifacts.unwrap_or(false);
    let mut tasks = Vec::new();

    for stored_task in all_tasks.items {
        let mut task = build_task_with_history(runtime.as_ref(), &auth_ctx, &stored_task).await?;

        if let Some(status) = query.status.as_deref() {
            if !task_state_matches_filter(&task, status)? {
                continue;
            }
        }

        if let Some(after) = query.status_timestamp_after.as_deref() {
            if !timestamp_after_filter(&task, after)? {
                continue;
            }
        }

        truncate_history(&mut task, query.history_length);
        if !include_artifacts {
            task.artifacts.clear();
        }

        tasks.push(task);
    }

    let total_size = i32::try_from(tasks.len()).unwrap_or(i32::MAX);
    let page_size_i32 = query.page_size.unwrap_or(50).clamp(1, 100);
    let page_size = usize::try_from(page_size_i32).unwrap_or(100);
    let start_offset = query
        .page_token
        .as_deref()
        .and_then(|token| token.parse::<usize>().ok())
        .unwrap_or(0);
    let end_offset = (start_offset + page_size).min(tasks.len());
    let has_more = end_offset < tasks.len();

    Ok(v1::ListTasksResponse {
        tasks: tasks
            .into_iter()
            .skip(start_offset)
            .take(end_offset.saturating_sub(start_offset))
            .collect::<Vec<_>>(),
        next_page_token: if has_more {
            end_offset.to_string()
        } else {
            String::new()
        },
        page_size: page_size_i32,
        total_size,
    })
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
    task: v1::Task,
    #[serde(skip_serializing_if = "Option::is_none")]
    skill_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pending_slot: Option<serde_json::Value>,
}

#[cfg(feature = "dev-ui")]
#[derive(serde::Serialize)]
struct UiTaskEvent {
    result: v1::StreamResponse,
    is_final: bool,
}

#[cfg(feature = "dev-ui")]
#[derive(serde::Serialize)]
struct UiTaskEventsResponse {
    events: Vec<UiTaskEvent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    task: Option<v1::Task>,
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
    let mut prev_state: Option<i32> = None;

    for event in events {
        if let TaskEvent::StatusUpdate(update) = event {
            let Some(status) = update.status.as_ref() else {
                continue;
            };
            let current_state = status.state;

            let current = TaskState::try_from(current_state).unwrap_or(TaskState::Unspecified);
            let prev = prev_state.and_then(|s| TaskState::try_from(s).ok());

            let trigger = match (&prev, &current) {
                (None, TaskState::Submitted | TaskState::Working) => "on_request",
                (Some(TaskState::InputRequired), TaskState::Working) => "on_input_received",
                _ => "status_update",
            };

            let timestamp = status.timestamp.as_ref().map_or_else(
                || chrono::Utc::now().to_rfc3339(),
                |ts| {
                    chrono::DateTime::from_timestamp(ts.seconds, ts.nanos as u32)
                        .map(|dt| dt.to_rfc3339())
                        .unwrap_or_else(|| chrono::Utc::now().to_rfc3339())
                },
            );

            transitions.push(StateTransition {
                from_state: prev.as_ref().map(|s| format!("{s:?}")),
                to_state: format!("{current:?}"),
                timestamp,
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

    summaries.sort_by(|a, b| {
        let a_ts = a
            .task
            .status
            .as_ref()
            .and_then(|s| s.timestamp.as_ref())
            .map(|ts| ts.seconds);
        let b_ts = b
            .task
            .status
            .as_ref()
            .and_then(|s| s.timestamp.as_ref())
            .map(|ts| ts.seconds);
        b_ts.cmp(&a_ts)
    });

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
        let (response, is_final) = task_event_to_v1_response(event);
        if is_final {
            final_seen = true;
        }
        serialized_events.push(UiTaskEvent {
            result: response,
            is_final,
        });
    }

    let stored_task = runtime.task_manager().get_task(&auth_ctx, task_id).await?;

    let task_snapshot = if let Some(task) = stored_task {
        Some(build_task_with_history(runtime.as_ref(), &auth_ctx, &task).await?)
    } else {
        None
    };

    if final_seen {
        if let Some(task) = &task_snapshot {
            let response = v1::StreamResponse {
                payload: Some(v1::stream_response::Payload::Task(task.clone())),
            };
            serialized_events.push(UiTaskEvent {
                result: response,
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
) -> AgentResult<v1::Task> {
    let events = runtime
        .task_manager()
        .get_task_events(auth_ctx, &stored_task.id)
        .await?;

    let history: Vec<v1::Message> = events
        .into_iter()
        .filter_map(|event| match event {
            TaskEvent::Message(msg) => Some(msg),
            TaskEvent::StatusUpdate(update) => update.status.and_then(|s| s.message),
            TaskEvent::ArtifactUpdate(_) => None,
        })
        .collect();

    Ok(v1::Task {
        id: stored_task.id.clone(),
        context_id: stored_task.context_id.clone(),
        status: Some(stored_task.status.clone()),
        artifacts: stored_task.artifacts.clone(),
        history,
        metadata: None,
    })
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
        use a2a_types::{self as v1, JSONRPCId, TaskState};
        use axum::body::Bytes;
        use axum::extract::State;
        use axum::http::StatusCode;
        use axum::response::Response;
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
        ) -> v1::Message {
            v1::Message {
                message_id: uuid::Uuid::new_v4().to_string(),
                context_id: context_id.unwrap_or_default(),
                task_id: task_id.unwrap_or_default(),
                role: v1::Role::User.into(),
                parts: vec![v1::Part {
                    content: Some(v1::part::Content::Text(text.to_string())),
                    metadata: None,
                    filename: String::new(),
                    media_type: "text/plain".to_string(),
                }],
                metadata: None,
                reference_task_ids: Vec::new(),
                extensions: Vec::new(),
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
        async fn agent_card_handler_returns_v1_agent_card() {
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
            let card: v1::AgentCard = serde_json::from_slice(&bytes).expect("agent card");
            assert_eq!(
                card.capabilities.as_ref().and_then(|caps| caps.streaming),
                Some(true)
            );
            assert_eq!(card.skills.len(), 1);
            assert_eq!(card.supported_interfaces.len(), 2);
            assert_eq!(card.supported_interfaces[0].protocol_binding, "JSONRPC");
            assert_eq!(
                card.supported_interfaces[0].url,
                "http://localhost:3000/rpc"
            );
            assert_eq!(card.supported_interfaces[1].protocol_binding, "HTTP+JSON");
            assert_eq!(card.supported_interfaces[1].url, "http://localhost:3000");
        }

        #[tokio::test(flavor = "current_thread")]
        async fn json_rpc_handler_accepts_v1_send_message() {
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

            let request = v1::SendMessageRequest {
                tenant: String::new(),
                message: Some(create_message("hello", None, None)),
                configuration: None,
                metadata: None,
            };
            let payload = json!({
                "jsonrpc": "2.0",
                "id": "req-v1",
                "method": "SendMessage",
                "params": request,
            });

            let response = json_rpc_handler(
                State(Arc::clone(&runtime)),
                Bytes::from(serde_json::to_vec(&payload).expect("payload json")),
            )
            .await;

            assert_eq!(response.status(), StatusCode::OK);
            let body = response_bytes(response).await;
            let parsed: JsonRpcSuccessResponse<v1::SendMessageResponse> =
                serde_json::from_slice(&body).expect("JSON-RPC v1 send response");

            assert_eq!(parsed.id, Some(JSONRPCId::String("req-v1".into())));
            match parsed.result.payload {
                Some(v1::send_message_response::Payload::Task(task)) => {
                    assert_eq!(
                        v1::TaskState::try_from(task.status.as_ref().expect("status").state)
                            .expect("task state"),
                        v1::TaskState::Completed
                    );
                }
                other => panic!("expected task payload, got {other:?}"),
            }
        }

        #[tokio::test(flavor = "current_thread")]
        async fn message_send_handler_returns_v1_task() {
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

            let request = v1::SendMessageRequest {
                tenant: String::new(),
                message: Some(create_message("hello", None, None)),
                configuration: None,
                metadata: None,
            };

            let response = message_send_handler(
                State(Arc::clone(&runtime)),
                Bytes::from(serde_json::to_vec(&request).expect("request json")),
            )
            .await;

            assert_eq!(response.status(), StatusCode::OK);
            let body = response_bytes(response).await;
            let parsed: v1::SendMessageResponse =
                serde_json::from_slice(&body).expect("send response");

            match parsed.payload {
                Some(v1::send_message_response::Payload::Task(task)) => {
                    assert_eq!(
                        v1::TaskState::try_from(task.status.as_ref().expect("status").state)
                            .expect("task state"),
                        v1::TaskState::Completed
                    );
                }
                other => panic!("expected task payload, got {other:?}"),
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

            let request = v1::SendMessageRequest {
                tenant: String::new(),
                message: Some(create_message("hello", None, None)),
                configuration: None,
                metadata: None,
            };

            let response = message_stream_handler(
                State(Arc::clone(&runtime)),
                Bytes::from(serde_json::to_vec(&request).expect("request json")),
            )
            .await;

            assert_eq!(response.status(), StatusCode::OK);
            let body = response_bytes(response).await;
            let body_str = String::from_utf8(body).expect("utf8");
            #[cfg(test)]
            dbg!(&body_str);
            let mut events: Vec<v1::StreamResponse> = Vec::new();
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
                    events.push(serde_json::from_str::<v1::StreamResponse>(json).expect("event"));
                }
            }
            assert!(
                !events.is_empty(),
                "expected at least one SSE event, got none"
            );
            let last = events.last().expect("final event");
            match &last.payload {
                Some(v1::stream_response::Payload::Task(task)) => {
                    assert_eq!(
                        v1::TaskState::try_from(task.status.as_ref().expect("status").state)
                            .expect("task state"),
                        v1::TaskState::Completed
                    );
                }
                Some(v1::stream_response::Payload::StatusUpdate(status_update)) => {
                    assert_eq!(
                        v1::TaskState::try_from(
                            status_update.status.as_ref().expect("status").state
                        )
                        .expect("task state"),
                        v1::TaskState::Completed
                    );
                }
                other => panic!("expected final task or status update event, got {other:?}"),
            }
        }

        #[tokio::test(flavor = "current_thread")]
        async fn subscribe_task_handler_returns_not_found_for_missing_task() {
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

            let response = subscribe_task_handler(
                State(Arc::clone(&runtime)),
                axum::extract::Path("task-123".to_string()),
            )
            .await;

            assert_eq!(response.status(), StatusCode::NOT_FOUND);
            let body = response_bytes(response).await;
            let parsed: serde_json::Value = serde_json::from_slice(&body).expect("json error");
            assert!(parsed["error"]
                .as_str()
                .expect("error string")
                .contains("not found"));
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
                    state: TaskState::Working as i32,
                    timestamp: None,
                    message: None,
                },
                artifacts: Vec::new(),
            };

            task_manager
                .save_task(&auth_ctx, &stored_task)
                .await
                .expect("store task");

            let user_message = v1::Message {
                message_id: "msg-1".to_string(),
                role: v1::Role::User as i32,
                parts: vec![v1::Part {
                    content: Some(v1::part::Content::Text("Hello runtime".to_string())),
                    metadata: None,
                    filename: String::new(),
                    media_type: "text/plain".to_string(),
                }],
                context_id: stored_task.context_id.clone(),
                task_id: stored_task.id.clone(),
                reference_task_ids: Vec::new(),
                extensions: Vec::new(),
                metadata: None,
            };

            task_manager
                .add_task_event(&auth_ctx, &TaskEvent::Message(user_message.clone()))
                .await
                .expect("store user message");

            let agent_message = v1::Message {
                message_id: "msg-2".to_string(),
                role: v1::Role::Agent as i32,
                parts: vec![v1::Part {
                    content: Some(v1::part::Content::Text("Finished".to_string())),
                    metadata: None,
                    filename: String::new(),
                    media_type: "text/plain".to_string(),
                }],
                context_id: stored_task.context_id.clone(),
                task_id: stored_task.id.clone(),
                reference_task_ids: Vec::new(),
                extensions: Vec::new(),
                metadata: None,
            };

            let status_update = a2a_types::TaskStatusUpdateEvent {
                task_id: stored_task.id.clone(),
                context_id: stored_task.context_id.clone(),
                status: Some(a2a_types::TaskStatus {
                    state: TaskState::Completed as i32,
                    timestamp: None,
                    message: Some(agent_message.clone()),
                }),
                metadata: None,
            };

            task_manager
                .add_task_event(&auth_ctx, &TaskEvent::StatusUpdate(status_update))
                .await
                .expect("store status update");

            stored_task.status.state = TaskState::Completed as i32;
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
                .any(|msg| msg.role == v1::Role::User as i32 && msg.parts.len() == 1));
            assert!(task
                .history
                .iter()
                .any(|msg| msg.role == v1::Role::Agent as i32 && msg.parts.len() == 1));
            assert_eq!(
                task.status.as_ref().unwrap().state,
                TaskState::Completed as i32
            );
        }

        #[tokio::test(flavor = "current_thread")]
        async fn build_v1_streaming_sse_sends_only_status_events() {
            use crate::runtime::core::executor::TaskStream;
            use crate::runtime::task_manager::TaskEvent;
            use axum::response::IntoResponse;

            let status_event = a2a_types::TaskStatusUpdateEvent {
                task_id: "task-1".into(),
                context_id: "ctx-1".into(),
                status: Some(a2a_types::TaskStatus {
                    state: TaskState::Completed as i32,
                    timestamp: None,
                    message: None,
                }),
                metadata: None,
            };

            let stream = TaskStream {
                initial_events: vec![TaskEvent::StatusUpdate(status_event)],
                receiver: None,
            };

            let response = build_v1_http_streaming_sse(stream).into_response();
            assert_eq!(response.status(), StatusCode::OK);
            let body = response_bytes(response).await;
            let body_str = String::from_utf8(body).expect("utf8");
            let events: Vec<v1::StreamResponse> = body_str
                .split("\n\n")
                .filter_map(|chunk| {
                    chunk
                        .trim()
                        .strip_prefix("data:")
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                })
                .map(|json| serde_json::from_str::<v1::StreamResponse>(json).expect("event json"))
                .collect();

            assert_eq!(events.len(), 1);
            match &events[0].payload {
                Some(v1::stream_response::Payload::StatusUpdate(update)) => {
                    assert_eq!(
                        v1::TaskState::try_from(update.status.as_ref().expect("status").state)
                            .expect("task state"),
                        v1::TaskState::Completed
                    );
                }
                other => panic!("unexpected response payload: {other:?}"),
            }
        }
    }
}
