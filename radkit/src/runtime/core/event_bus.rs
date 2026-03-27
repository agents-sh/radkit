//! Per-task fan-out for task events.
//!
//! The event bus allows multiple subscribers (e.g., SSE streams) to receive
//! updates for a given task while keeping persistence as the source of truth.

use crate::compat::channel::{self, UnboundedReceiver, UnboundedSender};
use crate::runtime::task_manager::TaskEvent;

#[cfg(not(all(target_os = "wasi", target_env = "p1")))]
use dashmap::DashMap;

#[cfg(all(target_os = "wasi", target_env = "p1"))]
use std::cell::RefCell;
#[cfg(all(target_os = "wasi", target_env = "p1"))]
use std::collections::HashMap;

/// Receiver for task event stream.
pub type TaskEventReceiver = UnboundedReceiver<TaskEvent>;

type TaskId = String;

#[derive(Default)]
struct Subscribers {
    senders: Vec<UnboundedSender<TaskEvent>>,
}

impl Subscribers {
    fn add(&mut self, sender: UnboundedSender<TaskEvent>) {
        self.senders.push(sender);
    }

    fn broadcast(&mut self, event: &TaskEvent) {
        self.senders
            .retain(|sender| sender.send(event.clone()).is_ok());
    }

    const fn is_empty(&self) -> bool {
        self.senders.is_empty()
    }
}

/// Multiplexes [`TaskEvent`]s to subscribers per task.
pub struct TaskEventBus {
    #[cfg(not(all(target_os = "wasi", target_env = "p1")))]
    inner: DashMap<TaskId, Subscribers>,

    #[cfg(all(target_os = "wasi", target_env = "p1"))]
    inner: RefCell<HashMap<TaskId, Subscribers>>,
}

impl TaskEventBus {
    /// Creates a new, empty bus.
    #[must_use]
    pub fn new() -> Self {
        Self {
            #[cfg(not(all(target_os = "wasi", target_env = "p1")))]
            inner: DashMap::new(),

            #[cfg(all(target_os = "wasi", target_env = "p1"))]
            inner: RefCell::new(HashMap::new()),
        }
    }

    /// Subscribes to events for the provided task identifier.
    #[must_use]
    pub fn subscribe(&self, task_id: &str) -> TaskEventReceiver {
        let (tx, rx) = channel::unbounded_channel();
        self.add_sender(task_id.to_string(), tx);
        rx
    }

    /// Publishes an event to all subscribers interested in the task.
    pub fn publish(&self, event: &TaskEvent) {
        if let Some(task_id) = extract_task_id(event) {
            self.broadcast(&task_id, event);
        }
    }

    fn add_sender(&self, task_id: TaskId, sender: UnboundedSender<TaskEvent>) {
        #[cfg(not(all(target_os = "wasi", target_env = "p1")))]
        {
            let mut entry = self.inner.entry(task_id).or_default();
            entry.add(sender);
        }

        #[cfg(all(target_os = "wasi", target_env = "p1"))]
        {
            let mut map = self.inner.borrow_mut();
            let subs = map.entry(task_id).or_default();
            subs.add(sender);
        }
    }

    fn broadcast(&self, task_id: &str, event: &TaskEvent) {
        #[cfg(not(all(target_os = "wasi", target_env = "p1")))]
        {
            if let Some(mut entry) = self.inner.get_mut(task_id) {
                entry.broadcast(event);
                if entry.is_empty() {
                    drop(entry);
                    self.inner.remove(task_id);
                }
            }
        }

        #[cfg(all(target_os = "wasi", target_env = "p1"))]
        {
            let mut map = self.inner.borrow_mut();
            if let Some(subs) = map.get_mut(task_id) {
                subs.broadcast(event);
                if subs.is_empty() {
                    map.remove(task_id);
                }
            }
        }
    }
}

fn extract_task_id(event: &TaskEvent) -> Option<TaskId> {
    match event {
        TaskEvent::StatusUpdate(update) => Some(update.task_id.clone()),
        TaskEvent::ArtifactUpdate(update) => Some(update.task_id.clone()),
        TaskEvent::Message(message) => {
            if message.task_id.is_empty() {
                None
            } else {
                Some(message.task_id.clone())
            }
        }
    }
}

impl Default for TaskEventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::task_manager::TaskEvent;
    use a2a_types::{Message, TaskState, TaskStatus, TaskStatusUpdateEvent};

    fn message_event(task_id: &str) -> TaskEvent {
        TaskEvent::Message(Message {
            message_id: "msg".into(),
            role: a2a_types::Role::Agent as i32,
            parts: Vec::new(),
            context_id: "ctx".into(),
            task_id: task_id.into(),
            reference_task_ids: Vec::new(),
            extensions: Vec::new(),
            metadata: None,
        })
    }

    #[tokio::test(flavor = "current_thread")]
    async fn publishes_events_to_subscribers() {
        let bus = TaskEventBus::new();
        let mut rx = bus.subscribe("task-123");

        let status = TaskStatusUpdateEvent {
            task_id: "task-123".into(),
            context_id: "ctx".into(),
            status: Some(TaskStatus {
                state: TaskState::Working as i32,
                timestamp: None,
                message: None,
            }),
            metadata: None,
        };
        bus.publish(&TaskEvent::StatusUpdate(status));

        let received = rx.recv().await.expect("event");
        match received {
            TaskEvent::StatusUpdate(update) => assert_eq!(update.task_id, "task-123"),
            _ => panic!("unexpected event"),
        }

        drop(rx);
        // publishing after subscriber drop should not panic
        bus.publish(&message_event("task-123"));
    }
}
