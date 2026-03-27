use a2a_types::{self as types, part, stream_response};
use serde_json::json;

#[test]
fn generated_v1_part_serializes_without_legacy_kind() {
    let part = types::Part {
        content: Some(part::Content::Text("hello".to_string())),
        metadata: None,
        filename: String::new(),
        media_type: String::new(),
    };

    let value = serde_json::to_value(&part).expect("part json");

    assert_eq!(value, json!({ "text": "hello" }));
}

#[test]
fn generated_v1_stream_response_uses_wrapper_fields() {
    let response = types::StreamResponse {
        payload: Some(stream_response::Payload::StatusUpdate(
            types::TaskStatusUpdateEvent {
                task_id: "task-1".to_string(),
                context_id: "ctx-1".to_string(),
                status: Some(types::TaskStatus {
                    state: types::TaskState::Working.into(),
                    message: None,
                    timestamp: None,
                }),
                metadata: None,
            },
        )),
    };

    let value = serde_json::to_value(&response).expect("stream response json");

    assert_eq!(
        value,
        json!({
            "statusUpdate": {
                "taskId": "task-1",
                "contextId": "ctx-1",
                "status": {
                    "state": "TASK_STATE_WORKING"
                }
            }
        })
    );
}
