//! Task execution logic for the runtime.
//!
//! This module is only available on native targets with the `runtime` feature enabled.
//! It handles the execution of A2A protocol requests and coordinates with services.
//!
//! # Slot Handling
//!
//! When skills return `InputRequired`, they provide a `SkillSlot` that describes what
//! type of input is expected. This slot information is used for:
//! - Input validation when the user provides a response
//! - UI hints for clients about what to collect
//! - Type-safe multi-turn conversations
//!
//! Skill state (slots and multi-turn data) is stored in `TaskState` and persisted
//! through the `TaskManager`. Session state for cross-skill data sharing is stored
//! in `SessionState` and scoped to the `context_id`.
//!
//! **Future implementation:**
//! ```rust,ignore
//! // When InputRequired is returned:
//! task_context.save_data("__radkit_current_slot", &slot)?;
//!
//! // When continuing:
//! let slot: SkillSlot = task_context.load_data("__radkit_current_slot")?
//!     .ok_or(...)?;
//! // Use slot for input validation
//! ```

use crate::agent::{
    builder::SkillRegistration, skill::SkillHandler, AgentDefinition, Artifact, OnInputResult,
    OnRequestResult, SkillSlot,
};
use crate::compat;
use crate::errors::{AgentError, AgentResult};
use crate::models::{utils, Content, Role};
use crate::runtime::context::{AuthContext, ProgressSender, State, TaskState as RuntimeTaskState};
use crate::runtime::core::{
    negotiator::{NegotiationDecision, Negotiator},
    status_mapper, TaskEventBus, TaskEventReceiver,
};
use crate::runtime::task_manager::{Task, TaskEvent, TaskManager};
use crate::runtime::AgentRuntime;
use a2a_types::TaskStatus;
use a2a_types::{self as v1, TaskArtifactUpdateEvent, TaskState};
use std::sync::Arc;
use tracing::error;
use uuid::Uuid;

pub trait ExecutorRuntime: AgentRuntime {
    fn agent(&self) -> &AgentDefinition;
    fn task_manager(&self) -> Arc<dyn TaskManager>;
    fn event_bus(&self) -> Arc<TaskEventBus>;
    fn negotiator(&self) -> Arc<dyn Negotiator>;
}

pub struct RequestExecutor {
    runtime: Arc<dyn ExecutorRuntime>,
}

/// Buffered task events plus optional event-bus subscription used to power
/// the `message/stream` and `tasks/resubscribe` SSE responses defined in the
/// A2A specification (§7.2 and §7.10 respectively).
#[cfg_attr(all(target_os = "wasi", target_env = "p1"), allow(dead_code))]
#[derive(Debug)]
pub struct TaskStream {
    pub initial_events: Vec<TaskEvent>,
    pub receiver: Option<TaskEventReceiver>,
}

/// Shared event writer to keep persistence + publish ordering consistent.
#[derive(Clone)]
struct TaskEventWriter {
    task_manager: Arc<dyn TaskManager>,
    event_bus: Arc<TaskEventBus>,
}

impl TaskEventWriter {
    fn new(runtime: &Arc<dyn ExecutorRuntime>) -> Self {
        Self {
            task_manager: runtime.task_manager(),
            event_bus: runtime.event_bus(),
        }
    }

    async fn push(&self, auth_ctx: &AuthContext, event: TaskEvent) -> AgentResult<()> {
        self.task_manager.add_task_event(auth_ctx, &event).await?;
        self.event_bus.publish(&event);
        Ok(())
    }

    async fn message(
        &self,
        auth_ctx: &AuthContext,
        context_id: &str,
        task_id: Option<&str>,
        role: Role,
        content: Content,
    ) -> AgentResult<v1::Message> {
        let message = utils::create_a2a_message(Some(context_id), task_id, role, content);
        self.push(auth_ctx, TaskEvent::Message(message.clone()))
            .await?;
        Ok(message)
    }

    async fn status(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
        context_id: &str,
        final_status: TaskStatus,
        is_final: bool,
    ) -> AgentResult<()> {
        let status_event =
            status_mapper::create_status_update_event(task_id, context_id, final_status, is_final);
        self.push(auth_ctx, TaskEvent::StatusUpdate(status_event))
            .await
    }

    async fn artifacts(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
        context_id: &str,
        artifacts: &[Artifact],
    ) -> AgentResult<()> {
        for artifact in artifacts {
            let event = TaskArtifactUpdateEvent {
                task_id: task_id.to_string(),
                context_id: context_id.to_string(),
                artifact: Some(utils::artifact_to_a2a(artifact)),
                append: false,
                last_chunk: true,
                metadata: None,
            };
            self.push(auth_ctx, TaskEvent::ArtifactUpdate(event))
                .await?;
        }
        Ok(())
    }
}

#[derive(Clone)]
struct TaskIdentifiers {
    context_id: String,
    task_id: String,
}

impl TaskIdentifiers {
    #[allow(clippy::missing_const_for_fn)]
    fn new(context_id: String, task_id: String) -> Self {
        Self {
            context_id,
            task_id,
        }
    }
}

#[cfg_attr(all(target_os = "wasi", target_env = "p1"), allow(dead_code))]
#[derive(Debug)]
pub enum PreparedSendMessage {
    Task(TaskStream),
    Message(v1::Message),
}

#[cfg_attr(all(target_os = "wasi", target_env = "p1"), allow(dead_code))]
#[derive(Copy, Clone, Debug)]
enum DeliveryMode {
    Blocking,
    Streaming,
}

#[cfg_attr(all(target_os = "wasi", target_env = "p1"), allow(dead_code))]
#[derive(Debug)]
enum DeliveryOutcome {
    Task(v1::Task),
    Message(v1::Message),
    Stream(TaskStream),
}

/// Indicates whether the inbound user payload is targeting an existing task,
/// an established context, or a brand-new context (per spec §2 and §7.1).
#[derive(Debug)]
enum MessageRoute {
    ExistingTask { context_id: String, task_id: String },
    ExistingContext { context_id: String },
    NewContext,
}

#[derive(Debug)]
enum ExecutionPhase {
    Initial(Content),
    Input(Content),
}

#[derive(Debug)]
enum ExecutionResult {
    Initial(OnRequestResult),
    Input(OnInputResult),
}

struct InitializedTask {
    task: Task,
    handler: Arc<dyn SkillHandler>,
    task_state: RuntimeTaskState,
    identifiers: TaskIdentifiers,
}

struct PreparedContinuation {
    task: Task,
    task_state: RuntimeTaskState,
    handler: Arc<dyn SkillHandler>,
    identifiers: TaskIdentifiers,
}

impl ExecutionResult {
    fn status(&self) -> TaskStatus {
        match self {
            Self::Initial(result) => status_mapper::on_request_to_status(result),
            Self::Input(result) => status_mapper::on_input_to_status(result),
        }
    }

    const fn message_content(&self) -> Option<&Content> {
        match self {
            Self::Initial(OnRequestResult::Completed { message, .. })
            | Self::Input(OnInputResult::Completed { message, .. }) => message.as_ref(),
            Self::Initial(OnRequestResult::InputRequired { message, .. })
            | Self::Input(OnInputResult::InputRequired { message, .. }) => Some(message),
            Self::Initial(OnRequestResult::Failed { error })
            | Self::Input(OnInputResult::Failed { error }) => Some(error),
            Self::Initial(OnRequestResult::Rejected { reason }) => Some(reason),
        }
    }

    const fn slot(&self) -> Option<&SkillSlot> {
        match self {
            Self::Initial(OnRequestResult::InputRequired { slot, .. })
            | Self::Input(OnInputResult::InputRequired { slot, .. }) => Some(slot),
            _ => None,
        }
    }

    fn artifacts(&self) -> Option<&[Artifact]> {
        match self {
            Self::Initial(OnRequestResult::Completed { artifacts, .. })
            | Self::Input(OnInputResult::Completed { artifacts, .. }) => Some(artifacts),
            _ => None,
        }
    }
}

impl MessageRoute {
    fn from_params(params: &v1::SendMessageRequest) -> AgentResult<Self> {
        let msg = params
            .message
            .as_ref()
            .ok_or_else(|| AgentError::InvalidInput("message is required".to_string()))?;
        let context_id = if msg.context_id.is_empty() {
            None
        } else {
            Some(msg.context_id.clone())
        };
        let task_id = if msg.task_id.is_empty() {
            None
        } else {
            Some(msg.task_id.clone())
        };
        match (context_id, task_id) {
            (Some(context_id), Some(task_id)) => Ok(Self::ExistingTask {
                context_id,
                task_id,
            }),
            (Some(context_id), None) => Ok(Self::ExistingContext { context_id }),
            (None, Some(_)) => Err(AgentError::InvalidInput(
                "context_id is required when passing task_id".to_string(),
            )),
            (None, None) => Ok(Self::NewContext),
        }
    }
}

impl RequestExecutor {
    #[must_use]
    pub fn new(runtime: Arc<dyn ExecutorRuntime>) -> Self {
        Self { runtime }
    }

    #[cfg_attr(all(target_os = "wasi", target_env = "p1"), allow(dead_code))]
    /// # Errors
    /// Returns an error if the message cannot be processed or the task cannot be created.
    pub async fn handle_send_message(
        &self,
        params: v1::SendMessageRequest,
    ) -> AgentResult<v1::SendMessageResponse> {
        self.handle_send_message_internal(params).await
    }

    /// Prepares and dispatches a streaming send-message request.
    ///
    /// # Errors
    ///
    /// Returns an [`AgentError`] if task creation, agent dispatch, or event-bus
    /// subscription fails.
    #[cfg_attr(all(target_os = "wasi", target_env = "p1"), allow(dead_code))]
    pub async fn handle_message_stream(
        &self,
        params: v1::SendMessageRequest,
    ) -> AgentResult<PreparedSendMessage> {
        self.handle_message_stream_internal(params).await
    }

    #[cfg_attr(all(target_os = "wasi", target_env = "p1"), allow(dead_code))]
    async fn handle_send_message_internal(
        &self,
        params: v1::SendMessageRequest,
    ) -> AgentResult<v1::SendMessageResponse> {
        match self
            .dispatch_message(params, DeliveryMode::Blocking)
            .await?
        {
            DeliveryOutcome::Task(task) => Ok(v1::SendMessageResponse {
                payload: Some(v1::send_message_response::Payload::Task(task)),
            }),
            DeliveryOutcome::Message(message) => Ok(v1::SendMessageResponse {
                payload: Some(v1::send_message_response::Payload::Message(message)),
            }),
            DeliveryOutcome::Stream(_) => Err(AgentError::Internal {
                component: "runtime".into(),
                reason: "streaming response produced for blocking message/send".into(),
            }),
        }
    }

    #[cfg_attr(all(target_os = "wasi", target_env = "p1"), allow(dead_code))]
    async fn handle_message_stream_internal(
        &self,
        params: v1::SendMessageRequest,
    ) -> AgentResult<PreparedSendMessage> {
        match self
            .dispatch_message(params, DeliveryMode::Streaming)
            .await?
        {
            DeliveryOutcome::Stream(stream) => Ok(PreparedSendMessage::Task(stream)),
            DeliveryOutcome::Message(message) => Ok(PreparedSendMessage::Message(message)),
            DeliveryOutcome::Task(_) => Err(AgentError::Internal {
                component: "runtime".into(),
                reason: "blocking task returned while handling message/stream".into(),
            }),
        }
    }

    async fn dispatch_message(
        &self,
        params: v1::SendMessageRequest,
        mode: DeliveryMode,
    ) -> AgentResult<DeliveryOutcome> {
        let auth_context = self.runtime.auth().get_auth_context();
        match MessageRoute::from_params(&params)? {
            MessageRoute::ExistingTask {
                context_id,
                task_id,
            } => {
                self.handle_existing_task(&auth_context, context_id, task_id, params, mode)
                    .await
            }
            MessageRoute::ExistingContext { context_id } => {
                self.handle_existing_context(&auth_context, context_id, params, mode)
                    .await
            }
            MessageRoute::NewContext => self.handle_new_context(&auth_context, params, mode).await,
        }
    }

    async fn handle_existing_task(
        &self,
        auth_ctx: &AuthContext,
        context_id: String,
        task_id: String,
        params: v1::SendMessageRequest,
        mode: DeliveryMode,
    ) -> AgentResult<DeliveryOutcome> {
        let stored_task = self
            .runtime
            .task_manager()
            .get_task(auth_ctx, &task_id)
            .await?
            .ok_or_else(|| AgentError::TaskNotFound {
                task_id: task_id.clone(),
            })?;

        if stored_task.context_id != context_id {
            return Err(AgentError::InvalidInput(format!(
                "Task id {task_id} does not belong to context id {context_id}"
            )));
        }

        let baseline_len = self
            .runtime
            .task_manager()
            .get_task_events(auth_ctx, &task_id)
            .await?
            .len();

        let content = Content::from(params.message.unwrap_or_default());
        self.append_message_event(
            auth_ctx,
            &context_id,
            Some(&task_id),
            Role::User,
            content.clone(),
        )
        .await?;

        match mode {
            DeliveryMode::Blocking => {
                let updated_task = self
                    .continue_task_blocking(auth_ctx, stored_task, content)
                    .await?;
                Ok(DeliveryOutcome::Task(updated_task))
            }
            DeliveryMode::Streaming => {
                let stream = self
                    .continue_task_streaming(auth_ctx, stored_task, content, baseline_len)
                    .await?;
                Ok(DeliveryOutcome::Stream(stream))
            }
        }
    }

    async fn handle_existing_context(
        &self,
        auth_ctx: &AuthContext,
        context_id: String,
        params: v1::SendMessageRequest,
        mode: DeliveryMode,
    ) -> AgentResult<DeliveryOutcome> {
        let related_tasks = self
            .runtime
            .task_manager()
            .list_task_ids(auth_ctx, Some(&context_id))
            .await?;

        let negotiation_messages = self
            .runtime
            .task_manager()
            .get_negotiating_messages(auth_ctx, &context_id)
            .await?;

        if related_tasks.is_empty() && negotiation_messages.is_empty() {
            return Err(AgentError::InvalidInput(format!(
                "Invalid context id {context_id}"
            )));
        }

        let content = Content::from(params.message.unwrap_or_default());
        self.append_message_event(auth_ctx, &context_id, None, Role::User, content.clone())
            .await?;

        let decision = self
            .runtime
            .negotiator()
            .negotiate(
                auth_ctx,
                self.runtime.agent(),
                &context_id,
                content.clone(),
                negotiation_messages,
            )
            .await?;

        self.deliver_negotiation_decision(auth_ctx, &context_id, content, decision, mode)
            .await
    }

    async fn handle_new_context(
        &self,
        auth_ctx: &AuthContext,
        params: v1::SendMessageRequest,
        mode: DeliveryMode,
    ) -> AgentResult<DeliveryOutcome> {
        let content = Content::from(params.message.unwrap_or_default());
        let context_id = Uuid::new_v4().to_string();

        self.append_message_event(auth_ctx, &context_id, None, Role::User, content.clone())
            .await?;

        let decision = self
            .runtime
            .negotiator()
            .negotiate(
                auth_ctx,
                self.runtime.agent(),
                &context_id,
                content.clone(),
                Vec::new(),
            )
            .await?;

        self.deliver_negotiation_decision(auth_ctx, &context_id, content, decision, mode)
            .await
    }

    #[cfg_attr(all(target_os = "wasi", target_env = "p1"), allow(dead_code))]
    pub(crate) async fn handle_task_resubscribe(
        &self,
        params: v1::SubscribeToTaskRequest,
    ) -> AgentResult<TaskStream> {
        self.handle_task_resubscribe_internal(params).await
    }

    #[cfg_attr(all(target_os = "wasi", target_env = "p1"), allow(dead_code))]
    async fn handle_task_resubscribe_internal(
        &self,
        params: v1::SubscribeToTaskRequest,
    ) -> AgentResult<TaskStream> {
        let auth_ctx = self.runtime.auth().get_auth_context();
        let stored_task = self
            .runtime
            .task_manager()
            .get_task(&auth_ctx, &params.id)
            .await?
            .ok_or_else(|| AgentError::TaskNotFound {
                task_id: params.id.clone(),
            })?;

        let events = self
            .runtime
            .task_manager()
            .get_task_events(&auth_ctx, &params.id)
            .await?;

        let is_final = status_mapper::is_terminal_status(&stored_task.status);

        let receiver = if is_final {
            None
        } else {
            Some(self.runtime.event_bus().subscribe(&params.id))
        };

        Ok(TaskStream {
            initial_events: events,
            receiver,
        })
    }

    /// # Errors
    /// Returns an error if the task is not found or retrieval fails.
    pub async fn handle_get_task(&self, params: v1::GetTaskRequest) -> AgentResult<v1::Task> {
        self.handle_get_task_internal(params).await
    }

    async fn handle_get_task_internal(&self, params: v1::GetTaskRequest) -> AgentResult<v1::Task> {
        let auth_context = self.runtime.auth().get_auth_context();
        let task_id = params.id;

        let stored_task = self
            .runtime
            .task_manager()
            .get_task(&auth_context, &task_id)
            .await?
            .ok_or_else(|| AgentError::TaskNotFound {
                task_id: task_id.clone(),
            })?;

        self.reconstruct_a2a_task(&auth_context, &stored_task).await
    }

    /// # Errors
    /// Always returns `NotImplemented` — cancel is not yet supported.
    #[allow(clippy::unused_async)] // async required for API parity with other handle_* methods
    pub async fn handle_cancel_task(&self, params: v1::CancelTaskRequest) -> AgentResult<v1::Task> {
        Self::handle_cancel_task_internal(&params)
    }

    fn handle_cancel_task_internal(params: &v1::CancelTaskRequest) -> AgentResult<v1::Task> {
        Err(AgentError::NotImplemented {
            feature: format!("tasks/cancel for task_id {}", params.id),
        })
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    async fn initialize_new_task(
        &self,
        auth_ctx: &AuthContext,
        context_id: &str,
        skill_id: &str,
        initial_message: Content,
    ) -> AgentResult<InitializedTask> {
        let task_id = Uuid::new_v4().to_string();
        let skill_reg = self.find_skill_by_id(skill_id)?;

        let task = Task {
            id: task_id.clone(),
            context_id: context_id.to_string(),
            status: status_mapper::working_status(),
            artifacts: vec![],
        };

        self.runtime
            .task_manager()
            .save_task(auth_ctx, &task)
            .await?;

        self.runtime
            .task_manager()
            .set_task_skill(auth_ctx, &task_id, skill_id)
            .await?;

        let writer = TaskEventWriter::new(&self.runtime);
        writer
            .message(
                auth_ctx,
                context_id,
                Some(&task_id),
                Role::User,
                initial_message.clone(),
            )
            .await?;

        Ok(InitializedTask {
            task,
            handler: skill_reg.handler_arc(),
            task_state: RuntimeTaskState::new(),
            identifiers: TaskIdentifiers::new(context_id.to_string(), task_id),
        })
    }

    async fn prepare_task_for_continuation(
        &self,
        auth_ctx: &AuthContext,
        mut task: Task,
    ) -> AgentResult<PreparedContinuation> {
        if !status_mapper::can_continue(&task.status) {
            return Err(AgentError::InvalidTaskStateTransition {
                from: format!("{:?}", task.status.state),
                to: "continuation".to_string(),
            });
        }

        let skill_id = self
            .runtime
            .task_manager()
            .get_task_skill(auth_ctx, &task.id)
            .await?
            .ok_or_else(|| AgentError::Internal {
                component: "task_manager".to_string(),
                reason: format!("No skill associated with task {}", task.id),
            })?;
        let skill_reg = self.find_skill_by_id(&skill_id)?;

        let task_state = self
            .runtime
            .task_manager()
            .load_task_state(auth_ctx, &task.id)
            .await?
            .unwrap_or_default();

        task.status = status_mapper::working_status();
        self.runtime
            .task_manager()
            .save_task(auth_ctx, &task)
            .await?;

        Ok(PreparedContinuation {
            handler: skill_reg.handler_arc(),
            identifiers: TaskIdentifiers::new(task.context_id.clone(), task.id.clone()),
            task,
            task_state,
        })
    }

    /// Finds a skill handler by its ID in the agent definition.
    ///
    /// # Arguments
    /// * `skill_id` - The unique identifier of the skill to find
    ///
    /// # Returns
    /// Reference to the skill registration or `SkillNotFound` error
    fn find_skill_by_id(&self, skill_id: &str) -> AgentResult<&SkillRegistration> {
        self.runtime
            .agent()
            .skills
            .iter()
            .find(|skill| skill.id() == skill_id)
            .ok_or_else(|| AgentError::SkillNotFound {
                skill_id: skill_id.to_string(),
            })
    }

    /// Reconstructs a full A2A Task from the stored Task + events.
    ///
    /// This method retrieves the task's event history and extracts message
    /// events to populate the task's history field per A2A protocol requirements.
    ///
    /// # Arguments
    /// * `auth_ctx` - Authentication context for access control
    /// * `task` - The stored task to reconstruct
    ///
    /// # Returns
    /// A complete A2A Task with populated history
    async fn reconstruct_a2a_task(
        &self,
        auth_ctx: &AuthContext,
        task: &Task,
    ) -> AgentResult<v1::Task> {
        let events = self
            .runtime
            .task_manager()
            .get_task_events(auth_ctx, &task.id)
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
            id: task.id.clone(),
            context_id: task.context_id.clone(),
            status: Some(task.status.clone()),
            history,
            artifacts: task.artifacts.clone(),
            metadata: None,
        })
    }

    // ========================================================================
    // Task Execution Methods
    // ========================================================================

    async fn start_task_blocking(
        &self,
        auth_ctx: &AuthContext,
        context_id: &str,
        skill_id: &str,
        initial_message: Content,
    ) -> AgentResult<v1::Task> {
        let InitializedTask {
            task,
            handler,
            task_state,
            identifiers,
        } = self
            .initialize_new_task(auth_ctx, context_id, skill_id, initial_message.clone())
            .await?;

        let task = drive_task(
            Arc::clone(&self.runtime),
            handler,
            task_state,
            auth_ctx.clone(),
            identifiers,
            ExecutionPhase::Initial(initial_message),
            task,
        )
        .await?;

        self.reconstruct_a2a_task(auth_ctx, &task).await
    }

    async fn continue_task_blocking(
        &self,
        auth_ctx: &AuthContext,
        task: Task,
        user_message: Content,
    ) -> AgentResult<v1::Task> {
        let PreparedContinuation {
            task,
            task_state,
            handler,
            identifiers,
        } = self.prepare_task_for_continuation(auth_ctx, task).await?;

        let task = drive_task(
            Arc::clone(&self.runtime),
            handler,
            task_state,
            auth_ctx.clone(),
            identifiers,
            ExecutionPhase::Input(user_message),
            task,
        )
        .await?;

        self.reconstruct_a2a_task(auth_ctx, &task).await
    }

    /// Persists the negotiator's assistant-facing response (reasoning,
    /// clarification, or rejection) and maps the `NegotiationDecision` into
    /// either a concrete task execution or a plain `Message`, depending on
    /// whether we're servicing `message/send` or `message/stream`.
    async fn deliver_negotiation_decision(
        &self,
        auth_ctx: &AuthContext,
        context_id: &str,
        content: Content,
        decision: NegotiationDecision,
        mode: DeliveryMode,
    ) -> AgentResult<DeliveryOutcome> {
        match decision {
            NegotiationDecision::StartTask {
                skill_id,
                reasoning,
            } => {
                self.append_message_event(
                    auth_ctx,
                    context_id,
                    None,
                    Role::Assistant,
                    Content::from_text(&reasoning),
                )
                .await?;

                match mode {
                    DeliveryMode::Blocking => {
                        let task = self
                            .start_task_blocking(auth_ctx, context_id, &skill_id, content)
                            .await?;
                        Ok(DeliveryOutcome::Task(task))
                    }
                    DeliveryMode::Streaming => {
                        let stream = self
                            .start_task_streaming(auth_ctx, context_id, &skill_id, content)
                            .await?;
                        Ok(DeliveryOutcome::Stream(stream))
                    }
                }
            }
            NegotiationDecision::AskClarification { message } => {
                let assistant_message = self
                    .append_message_event(
                        auth_ctx,
                        context_id,
                        None,
                        Role::Assistant,
                        Content::from_text(&message),
                    )
                    .await?;
                Ok(DeliveryOutcome::Message(assistant_message))
            }
            NegotiationDecision::Reject { reason } => {
                let assistant_message = self
                    .append_message_event(
                        auth_ctx,
                        context_id,
                        None,
                        Role::Assistant,
                        Content::from_text(&reason),
                    )
                    .await?;
                Ok(DeliveryOutcome::Message(assistant_message))
            }
        }
    }

    /// Starts a task for `message/stream`, immediately returning the buffered
    /// events (e.g., the user message) plus an event-bus subscription so the
    /// SSE layer can emit `SendStreamingMessageResponse` objects as work
    /// progresses.
    async fn start_task_streaming(
        &self,
        auth_ctx: &AuthContext,
        context_id: &str,
        skill_id: &str,
        initial_message: Content,
    ) -> AgentResult<TaskStream> {
        let InitializedTask {
            task,
            handler,
            task_state,
            identifiers,
        } = self
            .initialize_new_task(auth_ctx, context_id, skill_id, initial_message.clone())
            .await?;

        let initial_events = self
            .collect_new_events(auth_ctx, &identifiers.task_id, 0)
            .await?;

        let runtime = Arc::clone(&self.runtime);
        let auth_clone = auth_ctx.clone();
        let receiver = Some(self.runtime.event_bus().subscribe(&identifiers.task_id));

        compat::spawn({
            async move {
                if let Err(err) = drive_task(
                    runtime,
                    handler,
                    task_state,
                    auth_clone,
                    identifiers.clone(),
                    ExecutionPhase::Initial(initial_message),
                    task,
                )
                .await
                {
                    error!(
                        task_id = %identifiers.task_id,
                        error = %err,
                        "failed to execute streaming task"
                    );
                }
            }
        });

        Ok(TaskStream {
            initial_events,
            receiver,
        })
    }

    /// Continues a task in streaming mode by emitting the buffered user turn
    /// immediately and spawning background execution using `compat::spawn`,
    /// ensuring portability across native and WASI targets.
    async fn continue_task_streaming(
        &self,
        auth_ctx: &AuthContext,
        task: Task,
        user_message: Content,
        baseline_len: usize,
    ) -> AgentResult<TaskStream> {
        let initial_events = self
            .collect_new_events(auth_ctx, &task.id, baseline_len)
            .await?;

        let PreparedContinuation {
            task,
            task_state,
            handler,
            identifiers,
        } = self.prepare_task_for_continuation(auth_ctx, task).await?;

        let runtime = Arc::clone(&self.runtime);
        let auth_clone = auth_ctx.clone();
        let receiver = Some(self.runtime.event_bus().subscribe(&identifiers.task_id));
        let phase = ExecutionPhase::Input(user_message);

        compat::spawn({
            async move {
                if let Err(err) = drive_task(
                    runtime,
                    handler,
                    task_state,
                    auth_clone,
                    identifiers.clone(),
                    phase,
                    task,
                )
                .await
                {
                    error!(
                        task_id = %identifiers.task_id,
                        error = %err,
                        "failed to execute streaming continuation"
                    );
                }
            }
        });

        Ok(TaskStream {
            initial_events,
            receiver,
        })
    }

    /// Collects task events recorded after a given baseline index so we can
    /// replay them to SSE clients before live streaming begins.
    async fn collect_new_events(
        &self,
        auth_ctx: &AuthContext,
        task_id: &str,
        baseline_len: usize,
    ) -> AgentResult<Vec<TaskEvent>> {
        let events = self
            .runtime
            .task_manager()
            .get_task_events(auth_ctx, task_id)
            .await?;
        Ok(events.into_iter().skip(baseline_len).collect())
    }

    /// Persists a user/assistant message turn as a `TaskEvent::Message` and
    /// returns the generated A2A `Message` for downstream use.
    async fn append_message_event(
        &self,
        auth_ctx: &AuthContext,
        context_id: &str,
        task_id: Option<&str>,
        role: Role,
        content: Content,
    ) -> AgentResult<v1::Message> {
        let writer = TaskEventWriter::new(&self.runtime);
        writer
            .message(auth_ctx, context_id, task_id, role, content)
            .await
    }
}

async fn drive_task(
    runtime: Arc<dyn ExecutorRuntime>,
    handler: Arc<dyn SkillHandler>,
    task_state: RuntimeTaskState,
    auth_ctx: AuthContext,
    identifiers: TaskIdentifiers,
    phase: ExecutionPhase,
    mut task: Task,
) -> AgentResult<Task> {
    let TaskIdentifiers {
        context_id,
        task_id,
    } = identifiers;

    let session_state = runtime
        .task_manager()
        .load_session_state(&auth_ctx, &context_id)
        .await?
        .unwrap_or_default();

    let mut state = State::with_states(task_state, session_state);

    let progress = ProgressSender::new(
        auth_ctx.clone(),
        runtime.task_manager(),
        runtime.event_bus(),
        &context_id,
        &task_id,
    );

    let runtime_for_handlers: &dyn AgentRuntime = runtime.as_ref();
    let result = match phase {
        ExecutionPhase::Initial(content) => ExecutionResult::Initial(
            handler
                .on_request(&mut state, &progress, runtime_for_handlers, content)
                .await?,
        ),
        ExecutionPhase::Input(content) => ExecutionResult::Input(
            handler
                .on_input_received(&mut state, &progress, runtime_for_handlers, content)
                .await?,
        ),
    };

    let writer = TaskEventWriter::new(&runtime);
    let final_status = result.status();
    task.status = final_status.clone();

    if let Some(artifacts) = result.artifacts() {
        task.artifacts = utils::artifacts_to_a2a(artifacts);
        writer
            .artifacts(&auth_ctx, &task_id, &context_id, artifacts)
            .await?;
    }

    if let Some(slot) = result.slot() {
        state.set_pending_slot(slot.clone());
    } else {
        state.clear_pending_slot();
    }

    let (task_state, session_state) = state.into_parts();

    runtime
        .task_manager()
        .save_task_state(&auth_ctx, &task_id, &task_state)
        .await?;

    runtime
        .task_manager()
        .save_session_state(&auth_ctx, &context_id, &session_state)
        .await?;

    if let Some(content) = result.message_content() {
        writer
            .message(
                &auth_ctx,
                &context_id,
                Some(&task_id),
                Role::Assistant,
                content.clone(),
            )
            .await?;
    }

    let is_final = status_mapper::is_terminal_status(&final_status)
        || final_status.state == TaskState::InputRequired as i32;
    writer
        .status(&auth_ctx, &task_id, &context_id, final_status, is_final)
        .await?;

    runtime.task_manager().save_task(&auth_ctx, &task).await?;

    Ok(task)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{Agent, RegisteredSkill, SkillHandler, SkillMetadata, SkillSlot};
    use crate::models::{Content, LlmResponse};
    use crate::runtime::RuntimeBuilder;
    use crate::test_support::FakeLlm;
    use a2a_types::{self as v1, part, Role};

    fn negotiation_response(skill_id: &str, reasoning: &str) -> AgentResult<LlmResponse> {
        let decision = serde_json::json!({
            "type": "start_task",
            "skill_id": skill_id,
            "reasoning": reasoning,
        });
        FakeLlm::text_response(serde_json::to_string(&decision).expect("valid JSON"))
    }

    fn clarification_response(message: &str) -> AgentResult<LlmResponse> {
        let decision = serde_json::json!({
            "type": "ask_clarification",
            "message": message,
        });
        FakeLlm::text_response(serde_json::to_string(&decision).expect("valid JSON"))
    }

    fn rejection_response(reason: &str) -> AgentResult<LlmResponse> {
        let decision = serde_json::json!({
            "type": "reject",
            "reason": reason,
        });
        FakeLlm::text_response(serde_json::to_string(&decision).expect("valid JSON"))
    }

    fn make_message_params(
        text: &str,
        context_id: Option<&str>,
        task_id: Option<&str>,
    ) -> v1::SendMessageRequest {
        v1::SendMessageRequest {
            message: Some(v1::Message {
                message_id: uuid::Uuid::new_v4().to_string(),
                role: Role::User as i32,
                parts: vec![v1::Part {
                    content: Some(part::Content::Text(text.to_string())),
                    metadata: None,
                    filename: String::new(),
                    media_type: "text/plain".to_string(),
                }],
                context_id: context_id.unwrap_or_default().to_string(),
                task_id: task_id.unwrap_or_default().to_string(),
                reference_task_ids: Vec::new(),
                extensions: Vec::new(),
                metadata: None,
            }),
            configuration: None,
            metadata: None,
            tenant: String::new(),
        }
    }

    struct SimpleSkill;

    #[cfg_attr(
        all(target_os = "wasi", target_env = "p1"),
        async_trait::async_trait(?Send)
    )]
    #[cfg_attr(
        not(all(target_os = "wasi", target_env = "p1")),
        async_trait::async_trait
    )]
    impl SkillHandler for SimpleSkill {
        async fn on_request(
            &self,
            _state: &mut State,
            _progress: &ProgressSender,
            _runtime: &dyn AgentRuntime,
            _content: Content,
        ) -> Result<OnRequestResult, AgentError> {
            Ok(OnRequestResult::Completed {
                message: Some(Content::from_text("done")),
                artifacts: Vec::new(),
            })
        }
    }

    impl RegisteredSkill for SimpleSkill {
        fn metadata() -> std::sync::Arc<SkillMetadata> {
            std::sync::Arc::new(SkillMetadata::new(
                "simple-skill",
                "Simple Skill",
                "Completes immediately",
                &[],
                &[],
                &[],
                &[],
            ))
        }
    }

    struct MultiTurnSkill;

    #[cfg_attr(
        all(target_os = "wasi", target_env = "p1"),
        async_trait::async_trait(?Send)
    )]
    #[cfg_attr(
        not(all(target_os = "wasi", target_env = "p1")),
        async_trait::async_trait
    )]
    impl SkillHandler for MultiTurnSkill {
        async fn on_request(
            &self,
            state: &mut State,
            _progress: &ProgressSender,
            _runtime: &dyn AgentRuntime,
            _content: Content,
        ) -> Result<OnRequestResult, AgentError> {
            state.task().save("asked", &true)?;
            Ok(OnRequestResult::InputRequired {
                message: Content::from_text("Need more info"),
                slot: SkillSlot::new("details"),
            })
        }

        async fn on_input_received(
            &self,
            _state: &mut State,
            _progress: &ProgressSender,
            _runtime: &dyn AgentRuntime,
            content: Content,
        ) -> Result<OnInputResult, AgentError> {
            let response = format!("Received: {}", content.joined_texts().unwrap_or_default());
            Ok(OnInputResult::Completed {
                message: Some(Content::from_text(response)),
                artifacts: Vec::new(),
            })
        }
    }

    impl RegisteredSkill for MultiTurnSkill {
        fn metadata() -> std::sync::Arc<SkillMetadata> {
            std::sync::Arc::new(SkillMetadata::new(
                "multi-skill",
                "Multi-turn Skill",
                "Requests additional input",
                &[],
                &[],
                &[],
                &[],
            ))
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn new_context_creates_task_and_records_events() {
        let llm = FakeLlm::with_responses(
            "negotiator",
            [negotiation_response("simple-skill", "Using simple skill")],
        );
        let agent = Agent::builder()
            .with_name("Agent")
            .with_skill(SimpleSkill)
            .build();
        let runtime = RuntimeBuilder::new(agent, llm).build().into_shared();
        let exec_runtime: Arc<dyn ExecutorRuntime> = runtime.clone();
        let executor = RequestExecutor::new(exec_runtime);

        let params = make_message_params("Do the task", None, None);
        let result = executor
            .handle_send_message(params)
            .await
            .expect("send message result");

        let Some(v1::send_message_response::Payload::Task(task)) = result.payload else {
            panic!("expected task result")
        };

        assert_eq!(
            task.status.as_ref().unwrap().state,
            TaskState::Completed as i32
        );
        let auth = runtime.auth().get_auth_context();
        let stored = runtime
            .task_manager()
            .get_task(&auth, &task.id)
            .await
            .expect("get task")
            .expect("task exists");
        assert_eq!(stored.status.state, TaskState::Completed as i32);

        let events = runtime
            .task_manager()
            .get_task_events(&auth, &task.id)
            .await
            .expect("events");
        assert!(events.iter().any(
            |event| matches!(event, TaskEvent::Message(msg) if msg.role == Role::Agent as i32)
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn continuation_flow_updates_task() {
        let llm = FakeLlm::with_responses(
            "negotiator",
            [negotiation_response("multi-skill", "Need more info")],
        );
        let agent = Agent::builder()
            .with_name("Agent")
            .with_skill(MultiTurnSkill)
            .build();
        let runtime = RuntimeBuilder::new(agent, llm).build().into_shared();
        let exec_runtime: Arc<dyn ExecutorRuntime> = runtime.clone();
        let executor = RequestExecutor::new(exec_runtime);

        let first_result = executor
            .handle_send_message(make_message_params("Start", None, None))
            .await
            .expect("first result");
        let Some(v1::send_message_response::Payload::Task(task)) = first_result.payload else {
            panic!("expected task")
        };
        assert_eq!(
            task.status.as_ref().unwrap().state,
            TaskState::InputRequired as i32
        );

        let auth = runtime.auth().get_auth_context();

        let continue_params =
            make_message_params("Provide details", Some(&task.context_id), Some(&task.id));
        let continued = executor
            .handle_send_message(continue_params)
            .await
            .expect("continuation");
        let Some(v1::send_message_response::Payload::Task(updated)) = continued.payload else {
            panic!("expected task")
        };
        assert_eq!(
            updated.status.as_ref().unwrap().state,
            TaskState::Completed as i32
        );

        let task_state = runtime
            .task_manager()
            .load_task_state(&auth, &updated.id)
            .await
            .expect("load state")
            .expect("state present");
        let slot: Option<String> = task_state.slot().expect("slot load");
        assert!(slot.is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn negotiation_clarification_yields_message() {
        let llm =
            FakeLlm::with_responses("negotiator", [clarification_response("Need more details")]);
        let agent = Agent::builder()
            .with_name("Agent")
            .with_skill(SimpleSkill)
            .build();
        let runtime = RuntimeBuilder::new(agent, llm).build().into_shared();
        let exec_runtime: Arc<dyn ExecutorRuntime> = runtime.clone();
        let executor = RequestExecutor::new(exec_runtime);

        let result = executor
            .handle_send_message(make_message_params("Clarify", None, None))
            .await
            .expect("clarification");

        match result.payload {
            Some(v1::send_message_response::Payload::Message(msg)) => {
                assert_eq!(msg.role, Role::Agent as i32);
                assert!(!msg.context_id.is_empty());
            }
            other => panic!("expected message result, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn negotiation_reject_returns_message() {
        let llm = FakeLlm::with_responses("negotiator", [rejection_response("Out of scope")]);
        let agent = Agent::builder()
            .with_name("Agent")
            .with_skill(SimpleSkill)
            .build();
        let runtime = RuntimeBuilder::new(agent, llm).build().into_shared();
        let exec_runtime: Arc<dyn ExecutorRuntime> = runtime.clone();
        let executor = RequestExecutor::new(exec_runtime);

        let result = executor
            .handle_send_message(make_message_params("Reject", None, None))
            .await
            .expect("rejection");

        match result.payload {
            Some(v1::send_message_response::Payload::Message(msg)) => {
                assert_eq!(msg.role, Role::Agent as i32);
                assert!(!msg.context_id.is_empty());
            }
            other => panic!("expected message result, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancel_task_returns_not_implemented() {
        let llm = FakeLlm::with_responses("negotiator", [clarification_response("Need more info")]);
        let agent = Agent::builder()
            .with_name("Agent")
            .with_skill(SimpleSkill)
            .build();
        let runtime = RuntimeBuilder::new(agent, llm).build().into_shared();
        let exec_runtime: Arc<dyn ExecutorRuntime> = runtime.clone();
        let executor = RequestExecutor::new(exec_runtime);

        let err = executor
            .handle_cancel_task(v1::CancelTaskRequest {
                id: "task-1".into(),
                metadata: None,
                tenant: String::new(),
            })
            .await
            .expect_err("expected not implemented");
        match err {
            AgentError::NotImplemented { feature } => {
                assert!(feature.contains("tasks/cancel"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
