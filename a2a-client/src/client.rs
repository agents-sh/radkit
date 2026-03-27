//! A2A Client for calling remote A2A agents
//!
//! This module provides a client for making A2A protocol calls to remote agents.
//! It supports both streaming and non-streaming interactions.

use self::sse_parser::SseParser;
use crate::constants::{AGENT_CARD_PATH, JSONRPC_VERSION};
use crate::error::{A2AError, A2AResult};
use a2a_types::{self as v1, JSONRPCErrorResponse, JSONRPCId};
use futures_core::Stream;
use reqwest::{Client, RequestBuilder, Url};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(not(target_arch = "wasm32"))]
type BoxedResultStream<T> = Pin<Box<dyn Stream<Item = A2AResult<T>> + Send>>;
#[cfg(target_arch = "wasm32")]
type BoxedResultStream<T> = Pin<Box<dyn Stream<Item = A2AResult<T>>>>;

type SseStream = BoxedResultStream<v1::StreamResponse>;

/// A2A client for communicating with remote agents
#[derive(Clone)]
pub struct A2AClient {
    /// HTTP client for making requests
    client: Client,
    /// JSON-RPC service endpoint URL from the agent card, if available.
    rpc_endpoint_url: Option<String>,
    /// HTTP+JSON base URL from the agent card, if available.
    http_json_endpoint_url: Option<String>,
    /// Optional authentication token
    auth_token: Option<String>,
    /// Request ID counter for JSON-RPC requests
    request_id_counter: Arc<AtomicU64>,
    /// Cached agent card
    agent_card: Arc<v1::AgentCard>,
}

/// JSON-RPC 2.0 request structure
#[derive(Debug, Serialize)]
struct JsonRpcRequest<T> {
    jsonrpc: String,
    id: JSONRPCId,
    method: String,
    params: T,
}

/// JSON-RPC 2.0 response structure
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum JsonRpcResponse<T> {
    Success { id: Option<JSONRPCId>, result: T },
    Error(JSONRPCErrorResponse),
}

fn parse_agent_card_bytes(bytes: &[u8]) -> A2AResult<v1::AgentCard> {
    serde_json::from_slice(bytes).map_err(|error| A2AError::SerializationError {
        message: format!("Failed to parse agent card: {error}"),
    })
}

fn normalize_endpoint_url(url: &str) -> Option<String> {
    let trimmed = url.trim().trim_end_matches('/');
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn record_endpoint(slot: &mut Option<String>, url: &str) {
    if slot.is_none() {
        *slot = normalize_endpoint_url(url);
    }
}

fn resolve_endpoints(agent_card: &v1::AgentCard) -> A2AResult<(Option<String>, Option<String>)> {
    let mut rpc_endpoint_url = None;
    let mut http_json_endpoint_url = None;

    for interface in &agent_card.supported_interfaces {
        match interface.protocol_binding.as_str() {
            "JSONRPC" => record_endpoint(&mut rpc_endpoint_url, &interface.url),
            "HTTP+JSON" => record_endpoint(&mut http_json_endpoint_url, &interface.url),
            _ => {}
        }
    }

    if rpc_endpoint_url.is_none() && http_json_endpoint_url.is_none() {
        return Err(A2AError::InvalidParameter {
            message: "Agent card does not contain a supported JSON-RPC or HTTP+JSON endpoint"
                .to_string(),
        });
    }

    Ok((rpc_endpoint_url, http_json_endpoint_url))
}

fn timestamp_to_rfc3339<T>(value: T) -> A2AResult<String>
where
    T: Serialize,
{
    match serde_json::to_value(value).map_err(|error| A2AError::SerializationError {
        message: format!("Failed to serialize timestamp: {error}"),
    })? {
        serde_json::Value::String(value) => Ok(value),
        other => Err(A2AError::SerializationError {
            message: format!("Expected RFC3339 timestamp string, got {other}"),
        }),
    }
}

fn task_state_query_value(value: i32) -> A2AResult<Option<String>> {
    let state = v1::TaskState::try_from(value).map_err(|_| A2AError::InvalidParameter {
        message: format!("Unknown task state enum value {value}"),
    })?;

    match state {
        v1::TaskState::Unspecified => Ok(None),
        other => Ok(Some(other.as_str_name().to_string())),
    }
}

/// Handles parsing of Server-Sent Events (SSE) streams, accommodating both WASM and native targets.
mod sse_parser {
    use super::{A2AError, A2AResult, JsonRpcResponse};
    use futures_core::Stream;
    use serde::de::DeserializeOwned;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    // Define a trait that abstracts over the `Send` bound, which is required for non-WASM targets.
    #[cfg(not(target_arch = "wasm32"))]
    pub trait ByteStreamTrait: Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send {}
    #[cfg(not(target_arch = "wasm32"))]
    impl<T: Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send> ByteStreamTrait for T {}

    #[cfg(target_arch = "wasm32")]
    pub trait ByteStreamTrait: Stream<Item = Result<bytes::Bytes, reqwest::Error>> {}
    #[cfg(target_arch = "wasm32")]
    impl<T: Stream<Item = Result<bytes::Bytes, reqwest::Error>>> ByteStreamTrait for T {}

    // Define a type alias for the pinned byte stream to avoid repetition.
    #[cfg(not(target_arch = "wasm32"))]
    type PinnedByteStream =
        Pin<Box<dyn Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send>>;
    #[cfg(target_arch = "wasm32")]
    type PinnedByteStream = Pin<Box<dyn Stream<Item = Result<bytes::Bytes, reqwest::Error>>>>;

    /// A parser for Server-Sent Events (SSE) streams.
    pub struct SseParser<T> {
        inner: PinnedByteStream,
        buffer: String,
        event_data_buffer: String,
        pending_results: Vec<A2AResult<T>>,
        parser: fn(&str) -> A2AResult<T>,
    }

    impl<T> SseParser<T> {
        /// Creates a new SSE parser from a byte stream.
        pub fn new(
            inner: impl ByteStreamTrait + 'static,
            parser: fn(&str) -> A2AResult<T>,
        ) -> Self {
            Self {
                inner: Box::pin(inner),
                buffer: String::new(),
                event_data_buffer: String::new(),
                pending_results: Vec::new(),
                parser,
            }
        }

        /// Processes a chunk of bytes from the stream, parsing full SSE events.
        fn process_chunk(&mut self, chunk: bytes::Bytes) -> Vec<A2AResult<T>> {
            self.buffer.push_str(&String::from_utf8_lossy(&chunk));
            let mut results = Vec::new();

            // Process buffer line by line.
            while let Some(newline_pos) = self.buffer.find('\n') {
                let line = self.buffer[..newline_pos]
                    .trim_end_matches('\r')
                    .to_string();
                self.buffer = self.buffer[newline_pos + 1..].to_string();

                if line.is_empty() {
                    // An empty line signifies the end of an event.
                    if !self.event_data_buffer.is_empty() {
                        match (self.parser)(&self.event_data_buffer) {
                            Ok(result) => results.push(Ok(result)),
                            Err(e) => results.push(Err(e)),
                        }
                        self.event_data_buffer.clear();
                    }
                } else if let Some(data) = line.strip_prefix("data:") {
                    // Accumulate data lines for a single event.
                    if !self.event_data_buffer.is_empty() {
                        self.event_data_buffer.push('\n');
                    }
                    self.event_data_buffer.push_str(data.trim_start());
                } else if line.starts_with(':') {
                    // Ignore comment lines.
                }
            }
            results
        }
    }

    impl<T: Unpin> Stream for SseParser<T> {
        type Item = A2AResult<T>;

        fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let this = self.get_mut();
            // Drain any pending results from the last chunk processing.
            if let Some(result) = this.pending_results.pop() {
                return Poll::Ready(Some(result));
            }

            // Poll the underlying stream for more data.
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    let mut results = this.process_chunk(chunk);
                    if results.is_empty() {
                        // If no full events were parsed, wait for more data.
                        cx.waker().wake_by_ref();
                        Poll::Pending
                    } else {
                        // Reverse results to return them in the correct order.
                        results.reverse();
                        this.pending_results = results;
                        Poll::Ready(this.pending_results.pop())
                    }
                }
                Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(A2AError::NetworkError {
                    message: format!("Stream error: {}", e),
                }))),
                Poll::Ready(None) => Poll::Ready(None),
                Poll::Pending => Poll::Pending,
            }
        }
    }

    /// Processes the data part of a single SSE event carrying a JSON-RPC success envelope.
    pub(super) fn process_jsonrpc_sse_event<T>(json_data: &str) -> A2AResult<T>
    where
        T: DeserializeOwned,
    {
        if json_data.trim().is_empty() {
            return Err(A2AError::SerializationError {
                message: "Empty SSE event data".to_string(),
            });
        }

        let json_response: JsonRpcResponse<T> =
            serde_json::from_str(json_data).map_err(|e| A2AError::SerializationError {
                message: format!("Failed to parse SSE event data: {}", e),
            })?;

        match json_response {
            JsonRpcResponse::Success { result, .. } => Ok(result),
            JsonRpcResponse::Error(err) => Err(A2AError::RemoteAgentError {
                message: format!("SSE event contained an error: {}", err.error.message),
                code: Some(err.error.code),
            }),
        }
    }

    /// Processes the data part of a single SSE event carrying a direct JSON payload.
    pub(super) fn process_direct_sse_event<T>(json_data: &str) -> A2AResult<T>
    where
        T: DeserializeOwned,
    {
        if json_data.trim().is_empty() {
            return Err(A2AError::SerializationError {
                message: "Empty SSE event data".to_string(),
            });
        }

        serde_json::from_str(json_data).map_err(|e| A2AError::SerializationError {
            message: format!("Failed to parse SSE event data: {}", e),
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use a2a_types::{self as v1, JSONRPCError, JSONRPCErrorResponse, JSONRPCId};
        use bytes::Bytes;
        use futures_util::{StreamExt, stream};

        fn sample_message(text: &str) -> v1::Message {
            v1::Message {
                message_id: format!("msg-{text}"),
                context_id: "ctx-1".into(),
                task_id: "task-1".into(),
                role: v1::Role::Agent.into(),
                parts: vec![v1::Part {
                    content: Some(v1::part::Content::Text(text.to_string())),
                    metadata: None,
                    filename: String::new(),
                    media_type: "text/plain".into(),
                }],
                metadata: None,
                reference_task_ids: Vec::new(),
                extensions: Vec::new(),
            }
        }

        #[tokio::test]
        async fn sse_parser_emits_multiple_events_in_order() {
            let first = JsonRpcResponse::Success {
                id: Some(JSONRPCId::Integer(1)),
                result: v1::StreamResponse {
                    payload: Some(v1::stream_response::Payload::Message(sample_message("one"))),
                },
            };
            let second = JsonRpcResponse::Success {
                id: Some(JSONRPCId::Integer(2)),
                result: v1::StreamResponse {
                    payload: Some(v1::stream_response::Payload::Message(sample_message("two"))),
                },
            };
            let payload = format!(
                "data: {}\n\ndata: {}\n\n",
                serde_json::to_string(&first).expect("json"),
                serde_json::to_string(&second).expect("json")
            );
            let byte_stream = stream::iter(vec![Ok::<Bytes, reqwest::Error>(Bytes::from(payload))]);

            let mut parser =
                SseParser::new(byte_stream, process_jsonrpc_sse_event::<v1::StreamResponse>);
            let first_item: v1::StreamResponse =
                parser.next().await.expect("first event").expect("ok");
            let second_item: v1::StreamResponse =
                parser.next().await.expect("second event").expect("ok");

            match first_item.payload {
                Some(v1::stream_response::Payload::Message(msg)) => {
                    assert!(
                        msg.parts.iter().any(|part| {
                            matches!(part.content, Some(v1::part::Content::Text(_)))
                        })
                    );
                }
                other => panic!("expected message, got {other:?}"),
            }

            match second_item.payload {
                Some(v1::stream_response::Payload::Message(msg)) => {
                    assert!(msg.message_id.contains("two"));
                }
                other => panic!("expected message, got {other:?}"),
            }
        }

        #[test]
        fn process_sse_event_returns_error_for_remote_failure() {
            let error = JsonRpcResponse::<v1::StreamResponse>::Error(JSONRPCErrorResponse {
                jsonrpc: "2.0".into(),
                error: JSONRPCError {
                    code: -1,
                    message: "boom".into(),
                    data: None,
                },
                id: Some(JSONRPCId::Integer(1)),
            });
            let json = serde_json::to_string(&error).expect("json");
            let result = process_jsonrpc_sse_event::<v1::StreamResponse>(&json);
            assert!(matches!(result, Err(A2AError::RemoteAgentError { .. })));
        }
    }
}

impl A2AClient {
    /// Create a new A2A client from an agent card URL
    ///
    /// This will fetch the agent card from the specified URL and use the
    /// advertised v1 endpoints from the card for all subsequent requests.
    ///
    /// Uses a default `reqwest::Client` for HTTP requests. For custom HTTP
    /// configuration, use `from_card_url_with_client()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use a2a_client::A2AClient;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = A2AClient::from_card_url("https://agent.example.com").await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the agent card cannot be fetched, parsed, or does not advertise
    /// a supported `JSONRPC` or `HTTP+JSON` interface.
    pub async fn from_card_url(base_url: impl AsRef<str>) -> A2AResult<Self> {
        Self::from_card_url_with_client(base_url, Client::new()).await
    }

    /// Create a new A2A client from an agent card URL with a custom HTTP client
    ///
    /// This allows you to provide a pre-configured `reqwest::Client` with
    /// custom settings like timeouts, proxies, TLS config, default headers, etc.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(not(target_family = "wasm"))]
    /// # {
    /// use a2a_client::A2AClient;
    /// use reqwest::Client;
    /// use std::time::Duration;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let http_client = Client::builder()
    ///     .timeout(Duration::from_secs(30))
    ///     .build()?;
    ///
    /// let client = A2AClient::from_card_url_with_client(
    ///     "https://agent.example.com",
    ///     http_client
    /// ).await?;
    /// # Ok(())
    /// # }
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the agent card cannot be fetched, the response status is not successful,
    /// JSON parsing fails, or the card does not advertise a supported interface.
    pub async fn from_card_url_with_client(
        base_url: impl AsRef<str>,
        http_client: Client,
    ) -> A2AResult<Self> {
        let base_url = base_url.as_ref().trim_end_matches('/');
        let card_url = format!("{}/{}", base_url, AGENT_CARD_PATH);

        let response = http_client
            .get(&card_url)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| A2AError::NetworkError {
                message: format!("Failed to fetch agent card from {}: {}", card_url, e),
            })?;

        if !response.status().is_success() {
            return Err(A2AError::NetworkError {
                message: format!("Failed to fetch agent card: HTTP {}", response.status()),
            });
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| A2AError::SerializationError {
                message: format!("Failed to read agent card response: {}", e),
            })?;
        let agent_card = parse_agent_card_bytes(&bytes)?;
        let (rpc_endpoint_url, http_json_endpoint_url) = resolve_endpoints(&agent_card)?;

        Ok(Self {
            client: http_client,
            rpc_endpoint_url,
            http_json_endpoint_url,
            auth_token: None,
            request_id_counter: Arc::new(AtomicU64::new(1)),
            agent_card: Arc::new(agent_card),
        })
    }

    /// Create a new A2A client directly from an agent card
    ///
    /// This is useful when you already have an agent card and don't need to fetch it.
    /// Uses a default `reqwest::Client`. For custom HTTP configuration, use `from_card_with_client()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use a2a_client::A2AClient;
    /// use a2a_types::AgentCard;
    ///
    /// # fn example(agent_card: AgentCard) -> Result<(), Box<dyn std::error::Error>> {
    /// let client = A2AClient::from_card(agent_card)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the agent card does not advertise a supported `JSONRPC` or `HTTP+JSON` interface.
    pub fn from_card(agent_card: v1::AgentCard) -> A2AResult<Self> {
        Self::from_card_with_client(agent_card, Client::new())
    }

    /// Create a new A2A client from an agent card with a custom HTTP client
    ///
    /// This allows you to provide a pre-configured `reqwest::Client` with
    /// custom settings like timeouts, proxies, TLS config, default headers, etc.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(not(target_family = "wasm"))]
    /// # {
    /// use a2a_client::A2AClient;
    /// use a2a_types::AgentCard;
    /// use reqwest::Client;
    /// use std::time::Duration;
    ///
    /// # fn example(agent_card: AgentCard) -> Result<(), Box<dyn std::error::Error>> {
    /// let http_client = Client::builder()
    ///     .timeout(Duration::from_secs(30))
    ///     .default_headers({
    ///         let mut headers = reqwest::header::HeaderMap::new();
    ///         headers.insert("X-Custom-Header", "value".parse()?);
    ///         headers
    ///     })
    ///     .build()?;
    ///
    /// let client = A2AClient::from_card_with_client(agent_card, http_client)?;
    /// # Ok(())
    /// # }
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the agent card does not advertise a supported `JSONRPC` or `HTTP+JSON` interface.
    pub fn from_card_with_client(
        agent_card: v1::AgentCard,
        http_client: Client,
    ) -> A2AResult<Self> {
        let (rpc_endpoint_url, http_json_endpoint_url) = resolve_endpoints(&agent_card)?;

        Ok(Self {
            client: http_client,
            rpc_endpoint_url,
            http_json_endpoint_url,
            auth_token: None,
            request_id_counter: Arc::new(AtomicU64::new(1)),
            agent_card: Arc::new(agent_card),
        })
    }

    /// Create a new A2A client from an agent card with custom headers
    ///
    /// This is a convenience method that builds a reqwest::Client with the provided
    /// headers and uses it to create the A2AClient.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use a2a_client::A2AClient;
    /// use a2a_types::AgentCard;
    /// use std::collections::HashMap;
    ///
    /// # fn example(agent_card: AgentCard) -> Result<(), Box<dyn std::error::Error>> {
    /// let mut headers = HashMap::new();
    /// headers.insert("Authorization".to_string(), "Bearer token123".to_string());
    /// headers.insert("X-API-Key".to_string(), "my-api-key".to_string());
    ///
    /// let client = A2AClient::from_card_with_headers(agent_card, headers)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the agent card is invalid, headers cannot be parsed, or the HTTP client cannot be built.
    pub fn from_card_with_headers(
        agent_card: v1::AgentCard,
        headers: std::collections::HashMap<String, String>,
    ) -> A2AResult<Self> {
        use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
        use std::str::FromStr;

        let mut header_map = HeaderMap::new();
        for (key, value) in headers {
            let header_name =
                HeaderName::from_str(&key).map_err(|e| A2AError::InvalidParameter {
                    message: format!("Invalid header name '{}': {}", key, e),
                })?;
            let header_value =
                HeaderValue::from_str(&value).map_err(|e| A2AError::InvalidParameter {
                    message: format!("Invalid header value for '{}': {}", key, e),
                })?;
            header_map.insert(header_name, header_value);
        }

        let http_client = Client::builder()
            .default_headers(header_map)
            .build()
            .map_err(|e| A2AError::NetworkError {
                message: format!("Failed to build HTTP client with headers: {}", e),
            })?;

        Self::from_card_with_client(agent_card, http_client)
    }

    /// Set authentication token (builder pattern)
    pub fn with_auth_token(mut self, token: impl Into<String>) -> Self {
        self.auth_token = Some(token.into());
        self
    }

    /// Get the cached agent card
    pub fn agent_card(&self) -> &v1::AgentCard {
        &self.agent_card
    }

    /// Fetch a fresh agent card from the base URL
    ///
    /// # Errors
    ///
    /// Returns an error if the network request fails, the response status is not successful, or JSON parsing fails.
    pub async fn fetch_agent_card(&self, base_url: impl AsRef<str>) -> A2AResult<v1::AgentCard> {
        let base_url = base_url.as_ref().trim_end_matches('/');
        let card_url = format!("{}/{}", base_url, AGENT_CARD_PATH);

        let mut req = self
            .client
            .get(&card_url)
            .header("Accept", "application/json");

        if let Some(token) = &self.auth_token {
            req = req.bearer_auth(token);
        }

        let response = req.send().await.map_err(|e| A2AError::NetworkError {
            message: format!("Failed to fetch agent card from {}: {}", card_url, e),
        })?;

        if !response.status().is_success() {
            return Err(A2AError::NetworkError {
                message: format!("Failed to fetch agent card: HTTP {}", response.status()),
            });
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| A2AError::SerializationError {
                message: format!("Failed to read agent card response: {}", e),
            })?;

        parse_agent_card_bytes(&bytes)
    }

    /// Get the next request ID
    fn next_request_id(&self) -> JSONRPCId {
        let id = self.request_id_counter.fetch_add(1, Ordering::SeqCst);
        JSONRPCId::Integer(id as i64)
    }

    fn rpc_endpoint(&self) -> A2AResult<&str> {
        self.rpc_endpoint_url
            .as_deref()
            .ok_or_else(|| A2AError::InvalidParameter {
                message: "Agent does not advertise a JSON-RPC endpoint".to_string(),
            })
    }

    fn http_json_base_url(&self) -> Option<&str> {
        self.http_json_endpoint_url.as_deref()
    }

    fn build_http_json_url(&self, tenant: &str, segments: &[&str]) -> A2AResult<String> {
        let base = self
            .http_json_base_url()
            .ok_or_else(|| A2AError::InvalidParameter {
                message: "Agent does not advertise an HTTP+JSON endpoint".to_string(),
            })?;
        let mut url = Url::parse(base).map_err(|error| A2AError::InvalidParameter {
            message: format!("Invalid HTTP+JSON base URL `{base}`: {error}"),
        })?;
        {
            let mut path_segments =
                url.path_segments_mut()
                    .map_err(|()| A2AError::InvalidParameter {
                        message: format!("HTTP+JSON base URL `{base}` cannot accept path segments"),
                    })?;
            if !tenant.is_empty() {
                path_segments.push(tenant);
            }
            for segment in segments {
                path_segments.push(segment);
            }
        }
        Ok(url.to_string())
    }

    fn prepare_request(&self, mut request: RequestBuilder) -> RequestBuilder {
        for (key, value) in Self::inject_trace_context() {
            request = request.header(key, value);
        }

        if let Some(token) = &self.auth_token {
            request = request.bearer_auth(token);
        }

        request
    }

    async fn send_json_request<TResponse>(
        &self,
        request: RequestBuilder,
        context: &str,
    ) -> A2AResult<TResponse>
    where
        TResponse: for<'de> Deserialize<'de>,
    {
        let response =
            self.prepare_request(request)
                .send()
                .await
                .map_err(|e| A2AError::NetworkError {
                    message: format!("Failed to send {context} request: {e}"),
                })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            if let Ok(error_json) = serde_json::from_str::<JSONRPCErrorResponse>(&error_text) {
                return Err(A2AError::RemoteAgentError {
                    message: error_json.error.message,
                    code: Some(error_json.error.code),
                });
            }
            return Err(A2AError::NetworkError {
                message: format!("HTTP error {status}: {error_text}"),
            });
        }

        response
            .json()
            .await
            .map_err(|e| A2AError::SerializationError {
                message: format!("Failed to parse {context} response: {e}"),
            })
    }

    async fn start_sse_request(
        &self,
        request: RequestBuilder,
        context: &str,
    ) -> A2AResult<reqwest::Response> {
        let response =
            self.prepare_request(request)
                .send()
                .await
                .map_err(|e| A2AError::NetworkError {
                    message: format!("Failed to send {context} request: {e}"),
                })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(A2AError::NetworkError {
                message: format!("HTTP error {status}: {error_text}"),
            });
        }

        let content_type = response
            .headers()
            .get("Content-Type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        if !content_type.starts_with("text/event-stream") {
            return Err(A2AError::NetworkError {
                message: format!(
                    "Invalid response Content-Type for SSE stream. Expected 'text/event-stream', got '{}'",
                    content_type
                ),
            });
        }

        Ok(response)
    }

    /// Inject W3C Trace Context into HTTP headers for distributed tracing
    ///
    /// Extracts the OpenTelemetry context from the current tracing span and
    /// injects it into a carrier (HashMap) that can be used as HTTP headers.
    /// This enables trace propagation across service boundaries.
    fn inject_trace_context() -> std::collections::HashMap<String, String> {
        use opentelemetry::global;
        use tracing_opentelemetry::OpenTelemetrySpanExt;

        let mut carrier = std::collections::HashMap::new();

        // Get the OpenTelemetry context from the current tracing span
        let context = tracing::Span::current().context();

        // Inject the context into the carrier (adds traceparent, tracestate headers)
        // OpenTelemetry 0.31+ uses a closure-based API
        global::get_text_map_propagator(|propagator| {
            propagator.inject_context(&context, &mut carrier);
        });

        carrier
    }

    /// Helper method to make a generic JSON-RPC POST request
    async fn post_rpc_request<TParams, TResponse>(
        &self,
        method: &str,
        params: TParams,
    ) -> A2AResult<JsonRpcResponse<TResponse>>
    where
        TParams: Serialize,
        TResponse: for<'de> Deserialize<'de>,
    {
        let request_id = self.next_request_id();
        let rpc_request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            method: method.to_string(),
            params,
            id: request_id.clone(),
        };

        let req = self
            .client
            .post(self.rpc_endpoint()?)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(&rpc_request);

        let response =
            self.prepare_request(req)
                .send()
                .await
                .map_err(|e| A2AError::NetworkError {
                    message: format!("Failed to send {method} request: {e}"),
                })?;

        if !response.status().is_success() {
            // Try to parse error response
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            if let Ok(error_json) = serde_json::from_str::<JSONRPCErrorResponse>(&error_text) {
                return Ok(JsonRpcResponse::Error(error_json));
            }
            return Err(A2AError::NetworkError {
                message: format!("HTTP error {}: {}", status, error_text),
            });
        }

        let json_response: JsonRpcResponse<TResponse> =
            response
                .json()
                .await
                .map_err(|e| A2AError::SerializationError {
                    message: format!("Failed to parse {} response: {}", method, e),
                })?;

        // Validate response ID matches request ID
        if let JsonRpcResponse::Success {
            id: Some(resp_id), ..
        } = &json_response
            && resp_id != &request_id
        {
            eprintln!(
                "WARNING: RPC response ID mismatch for method {}. Expected {:?}, got {:?}",
                method, request_id, resp_id
            );
        }

        Ok(json_response)
    }

    fn unwrap_rpc_response<T>(&self, response: JsonRpcResponse<T>) -> A2AResult<T> {
        match response {
            JsonRpcResponse::Success { result, .. } => Ok(result),
            JsonRpcResponse::Error(err) => Err(A2AError::RemoteAgentError {
                message: format!("Remote agent error: {}", err.error.message),
                code: Some(err.error.code),
            }),
        }
    }

    fn ensure_streaming_enabled(&self, action: &str) -> A2AResult<()> {
        if self
            .agent_card
            .capabilities
            .as_ref()
            .and_then(|capabilities| capabilities.streaming)
            .unwrap_or(false)
        {
            Ok(())
        } else {
            Err(A2AError::InvalidParameter {
                message: format!("Agent does not support streaming (required for {action})"),
            })
        }
    }

    fn ensure_push_notifications_enabled(&self) -> A2AResult<()> {
        if self
            .agent_card
            .capabilities
            .as_ref()
            .and_then(|capabilities| capabilities.push_notifications)
            .unwrap_or(false)
        {
            Ok(())
        } else {
            Err(A2AError::InvalidParameter {
                message: "Agent does not support push notifications (capabilities.pushNotifications is not true)"
                    .to_string(),
            })
        }
    }

    /// Send a message using the A2A v1 surface.
    pub async fn send_message(
        &self,
        request: v1::SendMessageRequest,
    ) -> A2AResult<v1::SendMessageResponse> {
        if self.http_json_base_url().is_some() {
            let url = self.build_http_json_url(&request.tenant, &["message:send"])?;
            return self
                .send_json_request(
                    self.client
                        .post(url)
                        .header("Content-Type", "application/json")
                        .header("Accept", "application/json")
                        .json(&request),
                    "SendMessage",
                )
                .await;
        }

        self.unwrap_rpc_response(self.post_rpc_request("SendMessage", request).await?)
    }

    /// Send a streaming message using the A2A v1 surface.
    pub async fn send_streaming_message(
        &self,
        request: v1::SendMessageRequest,
    ) -> A2AResult<SseStream> {
        self.ensure_streaming_enabled("SendStreamingMessage")?;

        if self.http_json_base_url().is_some() {
            let url = self.build_http_json_url(&request.tenant, &["message:stream"])?;
            let response = self
                .start_sse_request(
                    self.client
                        .post(url)
                        .header("Content-Type", "application/json")
                        .header("Accept", "text/event-stream")
                        .json(&request),
                    "SendStreamingMessage",
                )
                .await?;
            return Ok(Box::pin(SseParser::new(
                response.bytes_stream(),
                sse_parser::process_direct_sse_event::<v1::StreamResponse>,
            )));
        }

        let rpc_request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            method: "SendStreamingMessage".to_string(),
            params: request,
            id: self.next_request_id(),
        };
        let response = self
            .start_sse_request(
                self.client
                    .post(self.rpc_endpoint()?)
                    .header("Content-Type", "application/json")
                    .header("Accept", "text/event-stream")
                    .json(&rpc_request),
                "SendStreamingMessage",
            )
            .await?;

        Ok(Box::pin(SseParser::new(
            response.bytes_stream(),
            sse_parser::process_jsonrpc_sse_event::<v1::StreamResponse>,
        )))
    }

    /// Retrieve a task using the A2A v1 surface.
    pub async fn get_task(&self, request: v1::GetTaskRequest) -> A2AResult<v1::Task> {
        if self.http_json_base_url().is_some() {
            #[derive(Serialize)]
            struct GetTaskQuery {
                #[serde(skip_serializing_if = "Option::is_none", rename = "historyLength")]
                history_length: Option<i32>,
            }

            let url = self.build_http_json_url(&request.tenant, &["tasks", &request.id])?;
            return self
                .send_json_request(
                    self.client
                        .get(url)
                        .header("Accept", "application/json")
                        .query(&GetTaskQuery {
                            history_length: request.history_length,
                        }),
                    "GetTask",
                )
                .await;
        }

        self.unwrap_rpc_response(self.post_rpc_request("GetTask", request).await?)
    }

    /// List tasks using the A2A v1 surface.
    pub async fn list_tasks(
        &self,
        request: v1::ListTasksRequest,
    ) -> A2AResult<v1::ListTasksResponse> {
        if self.http_json_base_url().is_some() {
            #[derive(Serialize)]
            struct ListTasksQuery {
                #[serde(skip_serializing_if = "String::is_empty", rename = "contextId")]
                context_id: String,
                #[serde(skip_serializing_if = "Option::is_none")]
                status: Option<String>,
                #[serde(skip_serializing_if = "Option::is_none", rename = "pageSize")]
                page_size: Option<i32>,
                #[serde(skip_serializing_if = "String::is_empty", rename = "pageToken")]
                page_token: String,
                #[serde(skip_serializing_if = "Option::is_none", rename = "historyLength")]
                history_length: Option<i32>,
                #[serde(
                    skip_serializing_if = "Option::is_none",
                    rename = "statusTimestampAfter"
                )]
                status_timestamp_after: Option<String>,
                #[serde(skip_serializing_if = "Option::is_none", rename = "includeArtifacts")]
                include_artifacts: Option<bool>,
            }

            let url = self.build_http_json_url(&request.tenant, &["tasks"])?;
            let status = task_state_query_value(request.status)?;
            let status_timestamp_after = request
                .status_timestamp_after
                .map(timestamp_to_rfc3339)
                .transpose()?;
            return self
                .send_json_request(
                    self.client
                        .get(url)
                        .header("Accept", "application/json")
                        .query(&ListTasksQuery {
                            context_id: request.context_id,
                            status,
                            page_size: request.page_size,
                            page_token: request.page_token,
                            history_length: request.history_length,
                            status_timestamp_after,
                            include_artifacts: request.include_artifacts,
                        }),
                    "ListTasks",
                )
                .await;
        }

        self.unwrap_rpc_response(self.post_rpc_request("ListTasks", request).await?)
    }

    /// Cancel a task using the A2A v1 surface.
    pub async fn cancel_task(&self, request: v1::CancelTaskRequest) -> A2AResult<v1::Task> {
        if self.http_json_base_url().is_some() {
            let cancel_segment = format!("{}:cancel", request.id);
            let url = self.build_http_json_url(&request.tenant, &["tasks", &cancel_segment])?;
            return self
                .send_json_request(
                    self.client
                        .post(url)
                        .header("Content-Type", "application/json")
                        .header("Accept", "application/json")
                        .json(&request),
                    "CancelTask",
                )
                .await;
        }

        self.unwrap_rpc_response(self.post_rpc_request("CancelTask", request).await?)
    }

    /// Subscribe to a task stream using the A2A v1 surface.
    pub async fn subscribe_to_task(
        &self,
        request: v1::SubscribeToTaskRequest,
    ) -> A2AResult<SseStream> {
        self.ensure_streaming_enabled("SubscribeToTask")?;

        if self.http_json_base_url().is_some() {
            let subscribe_segment = format!("{}:subscribe", request.id);
            let url = self.build_http_json_url(&request.tenant, &["tasks", &subscribe_segment])?;
            let response = self
                .start_sse_request(
                    self.client.get(url).header("Accept", "text/event-stream"),
                    "SubscribeToTask",
                )
                .await?;
            return Ok(Box::pin(SseParser::new(
                response.bytes_stream(),
                sse_parser::process_direct_sse_event::<v1::StreamResponse>,
            )));
        }

        let rpc_request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            method: "SubscribeToTask".to_string(),
            params: request,
            id: self.next_request_id(),
        };
        let response = self
            .start_sse_request(
                self.client
                    .post(self.rpc_endpoint()?)
                    .header("Content-Type", "application/json")
                    .header("Accept", "text/event-stream")
                    .json(&rpc_request),
                "SubscribeToTask",
            )
            .await?;

        Ok(Box::pin(SseParser::new(
            response.bytes_stream(),
            sse_parser::process_jsonrpc_sse_event::<v1::StreamResponse>,
        )))
    }

    /// Fetch the extended agent card using the A2A v1 surface.
    pub async fn get_extended_agent_card(
        &self,
        request: v1::GetExtendedAgentCardRequest,
    ) -> A2AResult<v1::AgentCard> {
        if self.http_json_base_url().is_some() {
            let url = self.build_http_json_url(&request.tenant, &["extendedAgentCard"])?;
            return self
                .send_json_request(
                    self.client.get(url).header("Accept", "application/json"),
                    "GetExtendedAgentCard",
                )
                .await;
        }

        self.unwrap_rpc_response(
            self.post_rpc_request("GetExtendedAgentCard", request)
                .await?,
        )
    }

    /// Create or replace a task push notification config using the A2A v1 surface.
    pub async fn create_task_push_notification_config(
        &self,
        request: v1::TaskPushNotificationConfig,
    ) -> A2AResult<v1::TaskPushNotificationConfig> {
        self.ensure_push_notifications_enabled()?;

        if self.http_json_base_url().is_some() {
            let url = self.build_http_json_url(
                &request.tenant,
                &["tasks", &request.task_id, "pushNotificationConfigs"],
            )?;
            return self
                .send_json_request(
                    self.client
                        .post(url)
                        .header("Content-Type", "application/json")
                        .header("Accept", "application/json")
                        .json(&request),
                    "CreateTaskPushNotificationConfig",
                )
                .await;
        }

        self.unwrap_rpc_response(
            self.post_rpc_request("CreateTaskPushNotificationConfig", request)
                .await?,
        )
    }

    /// Fetch a task push notification config using the A2A v1 surface.
    pub async fn get_task_push_notification_config(
        &self,
        request: v1::GetTaskPushNotificationConfigRequest,
    ) -> A2AResult<v1::TaskPushNotificationConfig> {
        if self.http_json_base_url().is_some() {
            let url = self.build_http_json_url(
                &request.tenant,
                &[
                    "tasks",
                    &request.task_id,
                    "pushNotificationConfigs",
                    &request.id,
                ],
            )?;
            return self
                .send_json_request(
                    self.client.get(url).header("Accept", "application/json"),
                    "GetTaskPushNotificationConfig",
                )
                .await;
        }

        self.unwrap_rpc_response(
            self.post_rpc_request("GetTaskPushNotificationConfig", request)
                .await?,
        )
    }

    /// List task push notification configs using the A2A v1 surface.
    pub async fn list_task_push_notification_configs(
        &self,
        request: v1::ListTaskPushNotificationConfigsRequest,
    ) -> A2AResult<v1::ListTaskPushNotificationConfigsResponse> {
        if self.http_json_base_url().is_some() {
            #[derive(Serialize)]
            struct ListConfigsQuery {
                #[serde(rename = "pageSize")]
                page_size: i32,
                #[serde(skip_serializing_if = "String::is_empty", rename = "pageToken")]
                page_token: String,
            }

            let url = self.build_http_json_url(
                &request.tenant,
                &["tasks", &request.task_id, "pushNotificationConfigs"],
            )?;
            return self
                .send_json_request(
                    self.client
                        .get(url)
                        .header("Accept", "application/json")
                        .query(&ListConfigsQuery {
                            page_size: request.page_size,
                            page_token: request.page_token,
                        }),
                    "ListTaskPushNotificationConfigs",
                )
                .await;
        }

        self.unwrap_rpc_response(
            self.post_rpc_request("ListTaskPushNotificationConfigs", request)
                .await?,
        )
    }

    /// Delete a task push notification config using the A2A v1 surface.
    pub async fn delete_task_push_notification_config(
        &self,
        request: v1::DeleteTaskPushNotificationConfigRequest,
    ) -> A2AResult<()> {
        if self.http_json_base_url().is_some() {
            let url = self.build_http_json_url(
                &request.tenant,
                &[
                    "tasks",
                    &request.task_id,
                    "pushNotificationConfigs",
                    &request.id,
                ],
            )?;
            let _: serde_json::Value = self
                .send_json_request(
                    self.client.delete(url).header("Accept", "application/json"),
                    "DeleteTaskPushNotificationConfig",
                )
                .await?;
            return Ok(());
        }

        let _: serde_json::Value = self.unwrap_rpc_response(
            self.post_rpc_request("DeleteTaskPushNotificationConfig", request)
                .await?,
        )?;
        Ok(())
    }

    /// Call a custom extension method
    ///
    /// This allows calling custom JSON-RPC methods defined by agent extensions.
    ///
    /// # Errors
    ///
    /// Returns an error if the RPC request fails or the remote agent returns an error response.
    pub async fn call_extension_method<TParams, TResponse>(
        &self,
        method: &str,
        params: TParams,
    ) -> A2AResult<TResponse>
    where
        TParams: Serialize,
        TResponse: for<'de> Deserialize<'de>,
    {
        match self.post_rpc_request(method, params).await? {
            JsonRpcResponse::Success { result, .. } => Ok(result),
            JsonRpcResponse::Error(err) => Err(A2AError::RemoteAgentError {
                message: format!("Remote agent error: {}", err.error.message),
                code: Some(err.error.code),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_requires_valid_card_url() {
        let card_without_url = v1::AgentCard {
            name: "Test".to_string(),
            description: "Test".to_string(),
            supported_interfaces: vec![],
            provider: None,
            version: "1.0.0".to_string(),
            documentation_url: None,
            capabilities: Some(v1::AgentCapabilities::default()),
            security_schemes: std::collections::HashMap::new(),
            security_requirements: Vec::new(),
            default_input_modes: vec![],
            default_output_modes: vec![],
            skills: vec![],
            signatures: vec![],
            icon_url: None,
        };

        assert!(A2AClient::from_card(card_without_url).is_err());
    }

    #[test]
    fn test_from_card_with_headers() {
        let mut headers = std::collections::HashMap::new();
        headers.insert("Authorization".to_string(), "Bearer token123".to_string());
        headers.insert("X-API-Key".to_string(), "my-api-key".to_string());

        let card = v1::AgentCard {
            name: "Test".to_string(),
            description: "Test agent".to_string(),
            supported_interfaces: vec![v1::AgentInterface {
                url: "https://example.com".to_string(),
                protocol_binding: "JSONRPC".to_string(),
                tenant: String::new(),
                protocol_version: "1.0".to_string(),
            }],
            provider: None,
            version: "1.0.0".to_string(),
            documentation_url: None,
            capabilities: Some(v1::AgentCapabilities::default()),
            security_schemes: std::collections::HashMap::new(),
            security_requirements: Vec::new(),
            default_input_modes: vec![],
            default_output_modes: vec![],
            skills: vec![],
            signatures: vec![],
            icon_url: None,
        };

        let result = A2AClient::from_card_with_headers(card, headers);
        assert!(result.is_ok());

        let client = result.unwrap();
        assert_eq!(
            client.rpc_endpoint_url.as_deref(),
            Some("https://example.com")
        );
        assert_eq!(client.http_json_endpoint_url, None);
    }

    #[test]
    fn test_from_card_with_invalid_header_name() {
        let mut headers = std::collections::HashMap::new();
        headers.insert("Invalid Header Name!".to_string(), "value".to_string());

        let card = v1::AgentCard {
            name: "Test".to_string(),
            description: "Test agent".to_string(),
            supported_interfaces: vec![v1::AgentInterface {
                url: "https://example.com".to_string(),
                protocol_binding: "JSONRPC".to_string(),
                tenant: String::new(),
                protocol_version: "1.0".to_string(),
            }],
            provider: None,
            version: "1.0.0".to_string(),
            documentation_url: None,
            capabilities: Some(v1::AgentCapabilities::default()),
            security_schemes: std::collections::HashMap::new(),
            security_requirements: Vec::new(),
            default_input_modes: vec![],
            default_output_modes: vec![],
            skills: vec![],
            signatures: vec![],
            icon_url: None,
        };

        let result = A2AClient::from_card_with_headers(card, headers);
        assert!(result.is_err());
        if let Err(err) = result {
            assert!(matches!(err, A2AError::InvalidParameter { .. }));
        }
    }

    #[test]
    fn next_request_id_is_monotonic() {
        let client = A2AClient::from_card(v1::AgentCard {
            name: "Test".to_string(),
            description: "desc".to_string(),
            supported_interfaces: vec![v1::AgentInterface {
                url: "https://example.com".to_string(),
                protocol_binding: "JSONRPC".to_string(),
                tenant: String::new(),
                protocol_version: "1.0".to_string(),
            }],
            provider: None,
            version: "1.0.0".to_string(),
            documentation_url: None,
            capabilities: Some(v1::AgentCapabilities::default()),
            security_schemes: std::collections::HashMap::new(),
            security_requirements: Vec::new(),
            default_input_modes: vec![],
            default_output_modes: vec![],
            skills: vec![],
            signatures: vec![],
            icon_url: None,
        })
        .expect("valid card");

        let first = match client.next_request_id() {
            JSONRPCId::Integer(value) => value,
            other => panic!("unexpected id variant: {other:?}"),
        };
        let second = match client.next_request_id() {
            JSONRPCId::Integer(value) => value,
            other => panic!("unexpected id variant: {other:?}"),
        };

        assert_eq!(first, 1);
        assert_eq!(second, 2);
    }

    #[test]
    fn parses_v1_agent_card_bytes() {
        let card = v1::AgentCard {
            name: "V1 Agent".to_string(),
            description: "Latest schema".to_string(),
            supported_interfaces: vec![
                v1::AgentInterface {
                    url: "https://example.com/rpc".to_string(),
                    protocol_binding: "JSONRPC".to_string(),
                    tenant: String::new(),
                    protocol_version: "1.0".to_string(),
                },
                v1::AgentInterface {
                    url: "https://example.com".to_string(),
                    protocol_binding: "HTTP+JSON".to_string(),
                    tenant: String::new(),
                    protocol_version: "1.0".to_string(),
                },
            ],
            provider: None,
            version: "1.2.3".to_string(),
            documentation_url: None,
            capabilities: Some(v1::AgentCapabilities {
                streaming: Some(true),
                push_notifications: Some(false),
                extensions: Vec::new(),
                extended_agent_card: Some(true),
            }),
            security_schemes: std::collections::HashMap::new(),
            security_requirements: Vec::new(),
            default_input_modes: vec!["text/plain".to_string()],
            default_output_modes: vec!["text/plain".to_string()],
            skills: Vec::new(),
            signatures: Vec::new(),
            icon_url: None,
        };

        let json = serde_json::to_vec(&card).expect("v1 card json");
        let parsed = parse_agent_card_bytes(&json).expect("parsed card");

        assert_eq!(parsed.name, "V1 Agent");
        assert_eq!(parsed.supported_interfaces[0].protocol_version, "1.0");
        assert_eq!(
            parsed.supported_interfaces[0].url,
            "https://example.com/rpc"
        );
        assert_eq!(
            parsed.capabilities.as_ref().and_then(|caps| caps.streaming),
            Some(true)
        );
        assert_eq!(
            parsed
                .capabilities
                .as_ref()
                .and_then(|caps| caps.extended_agent_card),
            Some(true)
        );
        assert_eq!(parsed.supported_interfaces.len(), 2);
        assert_eq!(parsed.supported_interfaces[1].protocol_binding, "HTTP+JSON");
        assert_eq!(parsed.supported_interfaces[1].url, "https://example.com");
    }

    #[test]
    fn resolves_http_json_endpoint_from_additional_interfaces() {
        let client = A2AClient::from_card(v1::AgentCard {
            name: "Test".to_string(),
            description: "desc".to_string(),
            supported_interfaces: vec![
                v1::AgentInterface {
                    url: "https://example.com/rpc".to_string(),
                    protocol_binding: "JSONRPC".to_string(),
                    tenant: String::new(),
                    protocol_version: "1.0".to_string(),
                },
                v1::AgentInterface {
                    url: "https://example.com".to_string(),
                    protocol_binding: "HTTP+JSON".to_string(),
                    tenant: String::new(),
                    protocol_version: "1.0".to_string(),
                },
            ],
            provider: None,
            version: "1.0.0".to_string(),
            documentation_url: None,
            capabilities: Some(v1::AgentCapabilities::default()),
            security_schemes: std::collections::HashMap::new(),
            security_requirements: Vec::new(),
            default_input_modes: vec![],
            default_output_modes: vec![],
            skills: vec![],
            signatures: vec![],
            icon_url: None,
        })
        .expect("valid card");

        assert_eq!(
            client.rpc_endpoint_url.as_deref(),
            Some("https://example.com/rpc")
        );
        assert_eq!(
            client.http_json_endpoint_url.as_deref(),
            Some("https://example.com")
        );
    }
}
