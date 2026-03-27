//! Status mapping utilities for converting skill results to A2A protocol types.
//!
//! This module centralizes all conversion logic between radkit's internal skill
//! handler results and the A2A protocol's task status types. Keeping these conversions
//! isolated ensures we maintain a clean boundary between our internal types and the
//! external protocol.

use crate::agent::{OnInputResult, OnRequestResult};
use a2a_types::{TaskState, TaskStatus, TaskStatusUpdateEvent};
use pbjson_types::Timestamp;

/// Returns the current UTC time as a proto `Timestamp`.
fn now() -> Timestamp {
    let now = chrono::Utc::now();
    Timestamp {
        seconds: now.timestamp(),
        nanos: now.timestamp_subsec_nanos().cast_signed(),
    }
}

/// Creates a `TaskStatus` representing the Working state.
#[must_use]
pub fn working_status() -> TaskStatus {
    TaskStatus {
        state: TaskState::Working as i32,
        timestamp: Some(now()),
        message: None,
    }
}

/// Converts an `OnRequestResult` to an A2A `TaskStatus`.
#[must_use]
pub fn on_request_to_status(result: &OnRequestResult) -> TaskStatus {
    let state = match result {
        OnRequestResult::InputRequired { .. } => TaskState::InputRequired,
        OnRequestResult::Completed { .. } => TaskState::Completed,
        OnRequestResult::Failed { .. } => TaskState::Failed,
        OnRequestResult::Rejected { .. } => TaskState::Rejected,
    };

    TaskStatus {
        state: state as i32,
        timestamp: Some(now()),
        message: None,
    }
}

/// Converts an `OnInputResult` to an A2A `TaskStatus`.
#[must_use]
pub fn on_input_to_status(result: &OnInputResult) -> TaskStatus {
    let state = match result {
        OnInputResult::InputRequired { .. } => TaskState::InputRequired,
        OnInputResult::Completed { .. } => TaskState::Completed,
        OnInputResult::Failed { .. } => TaskState::Failed,
    };

    TaskStatus {
        state: state as i32,
        timestamp: Some(now()),
        message: None,
    }
}

/// Creates a `TaskStatusUpdateEvent` from a task status.
#[must_use]
pub fn create_status_update_event(
    task_id: &str,
    context_id: &str,
    status: TaskStatus,
    _is_final: bool,
) -> TaskStatusUpdateEvent {
    // Note: v1::TaskStatusUpdateEvent has no `kind` or `is_final` field.
    // The `_is_final` parameter is kept for call-site compatibility during
    // migration; callers can be updated to omit it once fully migrated.
    TaskStatusUpdateEvent {
        task_id: task_id.to_string(),
        context_id: context_id.to_string(),
        status: Some(status),
        metadata: None,
    }
}

/// Checks if a `TaskState` is terminal (cannot transition further).
#[must_use]
pub const fn is_terminal_state(state: &TaskState) -> bool {
    matches!(
        state,
        TaskState::Completed
            | TaskState::Failed
            | TaskState::Rejected
            | TaskState::Canceled
            | TaskState::Unspecified
    )
}

/// Checks if a `TaskStatus` (by its i32 state field) is terminal.
#[must_use]
pub fn is_terminal_status(status: &TaskStatus) -> bool {
    TaskState::try_from(status.state)
        .map(|s| is_terminal_state(&s))
        .unwrap_or(true)
}

/// Checks if a task can be continued (resumed with new input).
#[must_use]
pub fn can_continue(status: &TaskStatus) -> bool {
    TaskState::try_from(status.state)
        .map(|s| matches!(s, TaskState::InputRequired))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Content;

    #[test]
    fn test_working_status() {
        let status = working_status();
        assert_eq!(status.state, TaskState::Working as i32);
        assert!(status.timestamp.is_some());
        assert!(status.message.is_none());
    }

    #[test]
    fn test_on_request_to_status_input_required() {
        let result = OnRequestResult::InputRequired {
            message: Content::from_text("Need more info"),
            slot: crate::agent::SkillSlot::new(()),
        };
        let status = on_request_to_status(&result);
        assert_eq!(status.state, TaskState::InputRequired as i32);
        assert!(status.timestamp.is_some());
    }

    #[test]
    fn test_on_request_to_status_completed() {
        let result = OnRequestResult::Completed {
            message: Some(Content::from_text("Done")),
            artifacts: vec![],
        };
        let status = on_request_to_status(&result);
        assert_eq!(status.state, TaskState::Completed as i32);
        assert!(status.timestamp.is_some());
    }

    #[test]
    fn test_on_request_to_status_failed() {
        let result = OnRequestResult::Failed {
            error: Content::from_text("Something went wrong"),
        };
        let status = on_request_to_status(&result);
        assert_eq!(status.state, TaskState::Failed as i32);
        assert!(status.timestamp.is_some());
    }

    #[test]
    fn test_on_request_to_status_rejected() {
        let result = OnRequestResult::Rejected {
            reason: Content::from_text("Out of scope"),
        };
        let status = on_request_to_status(&result);
        assert_eq!(status.state, TaskState::Rejected as i32);
        assert!(status.timestamp.is_some());
    }

    #[test]
    fn test_is_terminal_state() {
        assert!(is_terminal_state(&TaskState::Completed));
        assert!(is_terminal_state(&TaskState::Failed));
        assert!(is_terminal_state(&TaskState::Rejected));
        assert!(is_terminal_state(&TaskState::Canceled));
        assert!(is_terminal_state(&TaskState::Unspecified));

        assert!(!is_terminal_state(&TaskState::Working));
        assert!(!is_terminal_state(&TaskState::InputRequired));
        assert!(!is_terminal_state(&TaskState::Submitted));
    }

    #[test]
    fn test_can_continue() {
        assert!(can_continue(&TaskStatus {
            state: TaskState::InputRequired as i32,
            timestamp: None,
            message: None,
        }));

        assert!(!can_continue(&TaskStatus {
            state: TaskState::Working as i32,
            timestamp: None,
            message: None,
        }));
        assert!(!can_continue(&TaskStatus {
            state: TaskState::Completed as i32,
            timestamp: None,
            message: None,
        }));
    }

    #[test]
    fn test_create_status_update_event() {
        let status = working_status();
        let event = create_status_update_event("task-123", "ctx-456", status, false);

        assert_eq!(event.task_id, "task-123");
        assert_eq!(event.context_id, "ctx-456");
        assert_eq!(
            event.status.as_ref().unwrap().state,
            TaskState::Working as i32
        );
    }
}
