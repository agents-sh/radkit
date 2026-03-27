#![allow(clippy::needless_update)]

// Core context types
pub mod context;

// User-facing services
pub mod auth;
pub mod logging;
pub mod memory;
pub mod task_manager;

// Core framework components (advanced API)
pub mod core;

// Web server (native only)
#[cfg(all(feature = "runtime", not(all(target_os = "wasi", target_env = "p1"))))]
pub mod web;

// Startup banner (native only)
#[cfg(all(feature = "runtime", not(all(target_os = "wasi", target_env = "p1"))))]
mod banner;

// Re-export service traits for convenience
pub use auth::AuthService;
pub use logging::{LogLevel, LoggingService};
pub use memory::MemoryService;
#[cfg(all(
    feature = "task-store-sqlite",
    not(all(target_os = "wasi", target_env = "p1"))
))]
pub use task_manager::SqliteTaskStore;
pub use task_manager::{
    DefaultTaskManager, ListTasksFilter, PaginatedResult, Task, TaskEvent, TaskManager, TaskStore,
};

// Re-export default implementations for convenience
#[cfg(feature = "runtime")]
pub use auth::StaticAuthService;
#[cfg(feature = "runtime")]
pub use core::negotiator::DefaultNegotiator;
#[cfg(feature = "runtime")]
pub use logging::ConsoleLoggingService;
#[cfg(feature = "runtime")]
pub use memory::InMemoryMemoryService;
#[cfg(feature = "runtime")]
pub use task_manager::{InMemoryTaskManager, InMemoryTaskStore};

#[cfg(feature = "runtime")]
use crate::agent::AgentDefinition;
use crate::{MaybeSend, MaybeSync};
use std::sync::Arc;

#[cfg(feature = "runtime")]
use {
    crate::errors::{AgentError, AgentResult},
    crate::models::BaseLlm,
    crate::runtime::core::event_bus::TaskEventBus,
};

// Conditional imports for the native-only server implementation
#[cfg(all(feature = "runtime", not(all(target_os = "wasi", target_env = "p1"))))]
use {
    axum::{
        routing::{get, post},
        Router,
    },
    tower_http::trace::TraceLayer,
    tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt},
};

// Conditional imports for dev-ui feature
#[cfg(all(feature = "dev-ui", not(all(target_os = "wasi", target_env = "p1"))))]
use tower_http::services::{ServeDir, ServeFile};

/// Core trait for runtime implementations exposed to skill authors.
///
/// The trait intentionally exposes only the services that handlers should use
/// directly. Infrastructure components like the negotiator, task manager, or
/// event bus remain internal to the runtime so advanced orchestration can
/// evolve without affecting the public API.
pub trait AgentRuntime: MaybeSend + MaybeSync {
    /// Returns the current user's authentication context.
    ///
    /// This provides access to the app name and user name for the current
    /// request, useful for multi-tenant scenarios.
    fn current_user(&self) -> context::AuthContext {
        self.auth().get_auth_context()
    }

    /// Returns the authentication service.
    fn auth(&self) -> Arc<dyn AuthService>;

    /// Returns the memory service for storing and retrieving data.
    fn memory(&self) -> Arc<dyn MemoryService>;

    /// Returns the logging service for structured logging.
    fn logging(&self) -> Arc<dyn LoggingService>;

    /// Returns the default LLM for this runtime.
    #[cfg(feature = "runtime")]
    fn default_llm(&self) -> Arc<dyn BaseLlm>;

    /// Returns a History facade for searching past conversations and user facts.
    ///
    /// This is a convenience method that wraps the memory service with
    /// pre-configured filtering for history-related content.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let history = runtime.history();
    /// let memories = history.recall(&auth, "user preferences", 5).await?;
    /// ```
    fn history(&self) -> memory::OwnedHistory {
        memory::OwnedHistory::new(self.memory())
    }

    /// Returns a Knowledge facade for searching documents and external sources.
    ///
    /// This is a convenience method that wraps the memory service with
    /// pre-configured filtering for knowledge-related content.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let knowledge = runtime.knowledge();
    /// let results = knowledge.search(&auth, "vacation policy", 5).await?;
    /// ```
    fn knowledge(&self) -> memory::OwnedKnowledge {
        memory::OwnedKnowledge::new(self.memory())
    }

    /// Returns a pre-configured MemoryToolset for agent use.
    ///
    /// The toolset provides these tools:
    /// - `load_memory`: Search past conversations and user facts
    /// - `save_memory`: Store user facts and preferences
    /// - `search_knowledge`: Search documents and external sources
    ///
    /// The toolset is pre-configured with the current user's auth context.
    #[cfg(feature = "runtime")]
    fn memory_tools(&self) -> crate::tools::memory::MemoryToolset {
        crate::tools::memory::MemoryToolset::new(self.memory(), self.current_user())
    }
}

/// Default runtime implementation for native targets.
///
/// Each runtime instance is responsible for exactly one [`AgentDefinition`].
/// Use [`RuntimeBuilder`] to configure custom services or to override the
/// public base URL before calling [`Runtime::serve`].
#[cfg(feature = "runtime")]
#[derive(Clone)]
pub struct Runtime {
    auth_service: Arc<dyn AuthService>,
    task_manager: Arc<dyn TaskManager>,
    memory_service: Arc<dyn MemoryService>,
    logging_service: Arc<dyn LoggingService>,
    base_llm: Arc<dyn BaseLlm>,
    negotiator: Arc<dyn core::negotiator::Negotiator>,
    event_bus: Arc<TaskEventBus>,
    agent: Arc<AgentDefinition>,
    #[cfg_attr(all(target_os = "wasi", target_env = "p1"), allow(dead_code))]
    base_url: Option<String>,
    #[cfg_attr(all(target_os = "wasi", target_env = "p1"), allow(dead_code))]
    bind_address: Option<String>,
}

/// Builder for configuring [`Runtime`] instances.
#[cfg(feature = "runtime")]
pub struct RuntimeBuilder {
    agent: AgentDefinition,
    auth_service: Arc<dyn AuthService>,
    task_store: Arc<dyn TaskStore>,
    memory_service: Arc<dyn MemoryService>,
    logging_service: Arc<dyn LoggingService>,
    base_llm: Arc<dyn BaseLlm>,
    base_url: Option<String>,
}

#[cfg(feature = "runtime")]
impl Runtime {
    /// Creates a builder for the provided agent and LLM provider.
    pub fn builder(agent: AgentDefinition, llm: impl BaseLlm + 'static) -> RuntimeBuilder {
        RuntimeBuilder::new(agent, llm)
    }

    /// Returns the agent definition owned by this runtime.
    #[cfg_attr(all(target_os = "wasi", target_env = "p1"), allow(dead_code))]
    pub(crate) fn agent(&self) -> &AgentDefinition {
        &self.agent
    }

    #[cfg_attr(all(target_os = "wasi", target_env = "p1"), allow(dead_code))]
    pub(crate) fn configured_base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
    }

    #[cfg_attr(all(target_os = "wasi", target_env = "p1"), allow(dead_code))]
    pub(crate) fn bind_address(&self) -> Option<&str> {
        self.bind_address.as_deref()
    }

    /// Exposes the task manager for integration tests.
    #[cfg(any(test, feature = "test-support"))]
    #[must_use]
    pub fn task_manager(&self) -> Arc<dyn TaskManager> {
        self.task_manager.clone()
    }

    /// Converts this runtime into a reference-counted handle suitable for sharing.
    #[must_use]
    pub fn into_shared(self) -> Arc<Self> {
        Arc::new(self)
    }

    /// Starts the local development server at the given address.
    ///
    /// This is an async method that runs until the server is stopped.
    ///
    /// # Errors
    ///
    /// Returns an error if the server cannot bind to the address or fails during operation.
    #[cfg(not(all(target_os = "wasi", target_env = "p1")))]
    pub async fn serve(mut self, address: impl AsRef<str>) -> AgentResult<()> {
        // Initialize tracing subscriber for logging
        tracing_subscriber::registry()
            .with(tracing_subscriber::EnvFilter::new(
                std::env::var("RUST_LOG")
                    .unwrap_or_else(|_| "radkit=debug,tower_http=debug".into()),
            ))
            .with(tracing_subscriber::fmt::layer())
            .init();

        let address = address.as_ref();

        // Store bind address for fallback URL generation
        self.bind_address = Some(address.to_string());

        if self.base_url.is_none() {
            tracing::warn!(
                "base_url not configured - agent cards will infer from bind address. \
                 For production, call .base_url(\"https://your-domain.com\") before .serve()"
            );
        }

        banner::display_banner(address, self.base_url.as_deref(), &self.agent);

        // Share the runtime state with the handlers
        let shared_runtime = Arc::new(self);

        // Build the Axum router with A2A API routes
        let api_routes = Router::new()
            .route("/.well-known/agent-card.json", get(web::agent_card_handler))
            .route("/rpc", post(web::json_rpc_handler))
            .route("/message:stream", post(web::message_stream_handler))
            .route(
                "/tasks/{task_id}/subscribe",
                post(web::task_resubscribe_handler),
            )
            .with_state(Arc::clone(&shared_runtime));

        // Serve the React UI when dev-ui feature is enabled
        #[cfg(feature = "dev-ui")]
        let app = {
            // Path to the UI dist directory relative to the radkit crate location
            let ui_dist_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("ui")
                .join("dist");

            // Serve static files from ui/dist, with index.html as fallback for client-side routing
            let serve_dir = ServeDir::new(&ui_dist_path)
                .not_found_service(ServeFile::new(ui_dist_path.join("index.html")));

            // UI-specific API routes under /ui/*
            let ui_api_routes = Router::new()
                .route("/ui/agent", get(web::agent_info_handler))
                .route("/ui/contexts", get(web::list_contexts_handler))
                .route(
                    "/ui/contexts/{context_id}/tasks",
                    get(web::context_tasks_handler),
                )
                .route("/ui/tasks/{task_id}/events", get(web::task_events_handler))
                .route(
                    "/ui/tasks/{task_id}/transitions",
                    get(web::task_transitions_handler),
                )
                .with_state(Arc::clone(&shared_runtime));

            // Priority: A2A routes → UI API routes → static files
            api_routes
                .merge(ui_api_routes)
                .fallback_service(serve_dir)
                .layer(TraceLayer::new_for_http())
        };

        #[cfg(not(feature = "dev-ui"))]
        let app = api_routes.layer(TraceLayer::new_for_http());

        tracing::debug!("starting server on {}", address);

        // Run the server
        let listener = tokio::net::TcpListener::bind(address)
            .await
            .map_err(|e| AgentError::ServerStartFailed(e.to_string()))?;

        axum::serve(listener, app.into_make_service())
            .await
            .map_err(|e| AgentError::ServerStartFailed(e.to_string()))
    }

    /// Placeholder for WASM targets where serve is not available.
    #[cfg(all(target_os = "wasi", target_env = "p1"))]
    pub async fn serve(self, _address: impl AsRef<str>) -> AgentResult<()> {
        Err(AgentError::NotImplemented {
            feature: "serve is not available for WASM targets".to_string(),
        })
    }
}

#[cfg(feature = "runtime")]
impl AgentRuntime for Runtime {
    fn auth(&self) -> Arc<dyn AuthService> {
        self.auth_service.clone()
    }

    fn memory(&self) -> Arc<dyn MemoryService> {
        self.memory_service.clone()
    }

    fn logging(&self) -> Arc<dyn LoggingService> {
        self.logging_service.clone()
    }

    fn default_llm(&self) -> Arc<dyn BaseLlm> {
        self.base_llm.clone()
    }
}

#[cfg(feature = "runtime")]
impl crate::runtime::core::executor::ExecutorRuntime for Runtime {
    fn agent(&self) -> &AgentDefinition {
        &self.agent
    }

    fn task_manager(&self) -> Arc<dyn TaskManager> {
        self.task_manager.clone()
    }

    fn event_bus(&self) -> Arc<TaskEventBus> {
        self.event_bus.clone()
    }

    fn negotiator(&self) -> Arc<dyn core::negotiator::Negotiator> {
        self.negotiator.clone()
    }
}

#[cfg(feature = "runtime")]
impl RuntimeBuilder {
    pub fn new(agent: AgentDefinition, llm: impl BaseLlm + 'static) -> Self {
        let base_llm: Arc<dyn BaseLlm> = Arc::new(llm);
        Self {
            agent,
            auth_service: Arc::new(StaticAuthService::default()),
            task_store: Arc::new(InMemoryTaskStore::new()),
            memory_service: Arc::new(InMemoryMemoryService::new()),
            logging_service: Arc::new(ConsoleLoggingService),
            base_llm,
            base_url: None,
        }
    }

    /// Overrides the authentication service used by the runtime.
    #[must_use]
    pub fn with_auth_service(mut self, service: impl AuthService + 'static) -> Self {
        self.auth_service = Arc::new(service);
        self
    }

    /// Overrides the memory service implementation.
    #[must_use]
    pub fn with_memory_service(mut self, service: impl MemoryService + 'static) -> Self {
        self.memory_service = Arc::new(service);
        self
    }

    /// Overrides the logging service implementation.
    #[must_use]
    pub fn with_logging_service(mut self, service: impl LoggingService + 'static) -> Self {
        self.logging_service = Arc::new(service);
        self
    }

    /// Overrides the task store implementation (persistence layer).
    #[must_use]
    pub fn with_task_store(mut self, store: impl TaskStore + 'static) -> Self {
        self.task_store = Arc::new(store);
        self
    }

    /// Sets the public-facing base URL for the runtime.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Builds the runtime with the configured services.
    #[must_use]
    pub fn build(self) -> Runtime {
        let negotiator = Arc::new(DefaultNegotiator::new(self.base_llm.clone()));
        let task_manager = Arc::new(DefaultTaskManager::with_store(self.task_store));
        Runtime {
            auth_service: self.auth_service,
            task_manager,
            memory_service: self.memory_service,
            logging_service: self.logging_service,
            base_llm: self.base_llm,
            negotiator,
            event_bus: Arc::new(TaskEventBus::new()),
            agent: Arc::new(self.agent),
            base_url: self.base_url,
            bind_address: None,
        }
    }
}

#[cfg(all(test, feature = "runtime"))]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use crate::test_support::FakeLlm;

    fn test_agent() -> AgentDefinition {
        Agent::builder().with_name("Test Agent").build()
    }

    #[test]
    fn builder_provides_default_services() {
        let llm = FakeLlm::with_responses("runtime", std::iter::empty());
        let runtime = Runtime::builder(test_agent(), llm).build();

        let auth_ctx = runtime.auth().get_auth_context();
        assert_eq!(auth_ctx.app_name, "default-app");
        assert_eq!(auth_ctx.user_name, "default-user");
    }
}
