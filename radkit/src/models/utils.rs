//! Utility functions for working with models and A2A protocol types.

use a2a_types::{Artifact as A2AArtifact, Message as A2AMessage, Part as A2APart, Role as A2ARole};
use uuid::Uuid;

use crate::agent::Artifact;
use crate::models::{Content, ContentPart, Role};

/// Creates an A2A `Message` from radkit `Content`.
pub fn create_a2a_message(
    context_id: Option<&str>,
    task_id: Option<&str>,
    role: Role,
    content: Content,
) -> A2AMessage {
    let parts: Vec<A2APart> = content
        .into_parts()
        .into_iter()
        .filter_map(ContentPart::into_a2a_part)
        .collect();

    let proto_role = match role {
        Role::User => A2ARole::User,
        Role::Assistant | Role::System | Role::Tool => A2ARole::Agent,
    };

    A2AMessage {
        message_id: Uuid::new_v4().to_string(),
        role: proto_role as i32,
        context_id: context_id.unwrap_or_default().to_string(),
        task_id: task_id.unwrap_or_default().to_string(),
        parts,
        reference_task_ids: Vec::new(),
        extensions: Vec::new(),
        metadata: None,
    }
}

/// Converts a radkit `Artifact` to `a2a_types::Artifact`.
pub fn artifact_to_a2a(artifact: &Artifact) -> A2AArtifact {
    let parts: Vec<A2APart> = artifact
        .content()
        .clone()
        .into_parts()
        .into_iter()
        .filter_map(ContentPart::into_a2a_part)
        .collect();

    A2AArtifact {
        artifact_id: artifact.name().to_string(),
        parts,
        name: artifact.name().to_string(),
        description: String::new(),
        extensions: Vec::new(),
        metadata: None,
    }
}

/// Converts a slice of radkit `Artifact`s to `a2a_types::Artifact`s.
pub fn artifacts_to_a2a(artifacts: &[Artifact]) -> Vec<A2AArtifact> {
    artifacts.iter().map(artifact_to_a2a).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Artifact;

    #[test]
    fn create_a2a_message_maps_roles_and_parts() {
        let content = Content::from_text("Hello");
        let message =
            create_a2a_message(Some("ctx-1"), Some("task-2"), Role::User, content.clone());

        assert_eq!(message.context_id, "ctx-1");
        assert_eq!(message.task_id, "task-2");
        assert_eq!(message.role, A2ARole::User as i32);
        assert_eq!(message.parts.len(), content.parts().len());
    }

    #[test]
    fn artifact_conversion_preserves_name_and_parts() {
        let artifact = Artifact::from_text("notes.txt", "example");
        let converted = artifact_to_a2a(&artifact);
        assert_eq!(converted.artifact_id, "notes.txt");
        assert_eq!(converted.name, "notes.txt");
        assert_eq!(converted.parts.len(), artifact.content().parts().len());

        let multiple = artifacts_to_a2a(&[artifact.clone(), artifact]);
        assert_eq!(multiple.len(), 2);
    }
}
