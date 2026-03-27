//! Agent builder and definition types.
//!
//! This module provides a fluent builder API for constructing agent definitions
//! that can be deployed to the runtime. Agents are composed of metadata (name,
//! description) and registered skills — either programmatic Rust skills or
//! file-based `AgentSkills` loaded from `SKILL.md` directories.
//!
//! # Overview
//!
//! - [`Agent`]: Entry point for the builder API
//! - [`AgentBuilder`]: Fluent builder for agent definitions
//! - [`AgentDefinition`]: Complete agent specification
//! - [`SkillRegistration`]: Internal representation of a registered skill
//!
//! # Examples
//!
//! ```ignore
//! use radkit::agent::Agent;
//!
//! let agent = Agent::builder()
//!     .with_name("Weather Assistant")
//!     .with_description("Provides weather information")
//!     .with_skill(MyForecastSkill)
//!     // AgentSkill — compile-time embedded from SKILL.md
//!     .with_skill_def(include_skill!("./skills/summarise"))
//!     // AgentSkill — runtime-loaded from directory
//!     .with_skill_dir("./skills/translate")?
//!     .build();
//! ```

use crate::agent::skill::{RegisteredSkill, SkillHandler, SkillMetadata};
use std::sync::Arc;

#[cfg(feature = "agentskill")]
use crate::agent::agentskill::AgentSkillDef;

const DEFAULT_AGENT_VERSION: &str = "0.0.1";

/// Declarative definition for an agent that can be deployed to the runtime.
pub struct AgentDefinition {
    pub(crate) name: String,
    pub(crate) version: String,
    pub(crate) description: Option<String>,
    pub(crate) dispatcher_prompt: Option<String>,
    pub(crate) skills: Vec<SkillRegistration>,
}

/// Fluent builder for constructing [`AgentDefinition`] instances.
pub struct AgentBuilder {
    inner: AgentDefinition,
    /// `AgentSkill` definitions waiting for an LLM to be injected.
    /// Resolved in `RuntimeBuilder::build()`.
    #[cfg(feature = "agentskill")]
    pub(crate) pending_skill_defs: Vec<AgentSkillDef>,
}

/// Marker struct providing the static entry point [`Agent::builder()`].
pub struct Agent;

/// Internal representation of a skill registered against an agent.
///
/// Both programmatic Rust skills and LLM-backed `AgentSkills` are represented
/// by this type after builder resolution.
pub struct SkillRegistration {
    pub(crate) metadata: Arc<SkillMetadata>,
    pub(crate) handler: Arc<dyn SkillHandler>,
}

impl Agent {
    /// Creates a new agent builder.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let agent = Agent::builder()
    ///     .with_name("My Agent")
    ///     .build();
    /// ```
    #[must_use]
    pub fn builder() -> AgentBuilder {
        AgentBuilder {
            inner: AgentDefinition {
                name: String::new(),
                version: DEFAULT_AGENT_VERSION.to_string(),
                description: None,
                dispatcher_prompt: None,
                skills: Vec::new(),
            },
            #[cfg(feature = "agentskill")]
            pending_skill_defs: Vec::new(),
        }
    }
}

impl AgentBuilder {
    /// Sets the agent display name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.inner.name = name.into();
        self
    }

    /// Sets the agent version string used for versioned transport routes.
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.inner.version = version.into();
        self
    }

    /// Sets the agent description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.inner.description = Some(description.into());
        self
    }

    /// Sets the dispatcher prompt used by the runtime negotiator.
    #[must_use]
    pub fn with_dispatcher_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.inner.dispatcher_prompt = Some(prompt.into());
        self
    }

    /// Registers a programmatic Rust skill with the agent.
    ///
    /// The skill must implement [`RegisteredSkill`], typically derived by the
    /// `#[skill]` macro.
    #[must_use]
    pub fn with_skill<T>(mut self, skill: T) -> Self
    where
        T: RegisteredSkill + 'static,
    {
        self.inner.skills.push(SkillRegistration {
            metadata: T::metadata(),
            handler: Arc::new(skill),
        });
        self
    }

    /// Registers an `AgentSkill` from an [`AgentSkillDef`].
    ///
    /// Use [`include_skill!`] to create one at compile time (SKILL.md embedded
    /// into the binary), or [`AgentSkillDef::from_dir`] for runtime loading.
    ///
    /// The LLM handler is injected when [`Runtime::builder`] calls `build()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Compile-time — skill embedded in binary, no I/O at startup
    /// Agent::builder()
    ///     .with_skill_def(include_skill!("./skills/pdf-processing"))
    ///     .build()
    /// ```
    #[cfg(feature = "agentskill")]
    #[must_use]
    pub fn with_skill_def(mut self, def: AgentSkillDef) -> Self {
        self.pending_skill_defs.push(def);
        self
    }

    /// Load and register an `AgentSkill` from a directory at runtime.
    ///
    /// Reads `SKILL.md` from the given directory, validates it against the
    /// `AgentSkills` specification, and queues it for handler injection when
    /// `Runtime::builder(...).build()` is called.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read, `SKILL.md` is missing,
    /// the frontmatter is invalid, or the `name` field doesn't match the
    /// directory name.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// Agent::builder()
    ///     .with_skill_dir("./skills/translate")?
    ///     .build()
    /// ```
    #[cfg(feature = "agentskill")]
    pub fn with_skill_dir(
        mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, crate::errors::AgentError> {
        let def = AgentSkillDef::from_dir(path)?;
        self.pending_skill_defs.push(def);
        Ok(self)
    }

    /// Finalizes and returns the agent definition.
    ///
    /// Note: if you have registered `AgentSkills` via [`with_skill_def`] or
    /// [`with_skill_dir`], pass this builder to [`Runtime::builder`] rather
    /// than calling `build()` directly — the runtime injects the LLM into
    /// `AgentSkill` handlers during its own `build()`.
    #[must_use]
    pub fn build(mut self) -> AgentDefinition {
        if self.inner.version.trim().is_empty() {
            self.inner.version = DEFAULT_AGENT_VERSION.to_string();
        }
        self.inner
    }

    /// Splits this builder into its `AgentDefinition` and pending `AgentSkillDef`s.
    ///
    /// Used internally by `RuntimeBuilder::new`.
    #[cfg(feature = "agentskill")]
    pub(crate) fn into_parts(
        mut self,
    ) -> (
        AgentDefinition,
        Vec<crate::agent::agentskill::AgentSkillDef>,
    ) {
        if self.inner.version.trim().is_empty() {
            self.inner.version = DEFAULT_AGENT_VERSION.to_string();
        }
        (self.inner, self.pending_skill_defs)
    }
}

impl From<AgentDefinition> for AgentBuilder {
    /// Wraps a pre-built `AgentDefinition` in an `AgentBuilder`.
    fn from(def: AgentDefinition) -> Self {
        Self {
            inner: def,
            #[cfg(feature = "agentskill")]
            pending_skill_defs: Vec::new(),
        }
    }
}

impl AgentDefinition {
    /// Returns the agent name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the agent version string.
    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Returns the agent description, if set.
    #[must_use]
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Returns the dispatcher prompt, if set.
    #[must_use]
    pub fn dispatcher_prompt(&self) -> Option<&str> {
        self.dispatcher_prompt.as_deref()
    }

    /// Returns a slice of registered skills.
    #[must_use]
    pub fn skills(&self) -> &[SkillRegistration] {
        &self.skills
    }
}

impl SkillRegistration {
    /// Returns the skill name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.metadata.name
    }

    /// Returns the skill identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.metadata.id
    }

    /// Returns the skill description.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.metadata.description
    }

    /// Returns a reference to the skill handler.
    #[must_use]
    pub fn handler(&self) -> &dyn SkillHandler {
        &*self.handler
    }

    /// Returns a shareable handle to the skill handler.
    #[must_use]
    pub fn handler_arc(&self) -> Arc<dyn SkillHandler> {
        Arc::clone(&self.handler)
    }

    /// Returns the metadata associated with the skill.
    #[must_use]
    pub fn metadata(&self) -> &SkillMetadata {
        &self.metadata
    }

    /// Returns a cloned `Arc` to the skill metadata.
    #[must_use]
    pub fn metadata_arc(&self) -> Arc<SkillMetadata> {
        Arc::clone(&self.metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::skill::{OnInputResult, OnRequestResult, SkillSlot};
    use crate::errors::AgentError;
    use crate::models::Content;

    #[test]
    fn build_applies_defaults() {
        let agent = Agent::builder().with_name("Test").build();

        assert_eq!(agent.name(), "Test");
        assert_eq!(agent.version(), DEFAULT_AGENT_VERSION);
        assert!(agent.description().is_none());
        assert!(agent.dispatcher_prompt().is_none());
        assert!(agent.skills().is_empty());
    }

    #[test]
    fn build_preserves_all_fields() {
        let agent = Agent::builder()
            .with_name("Custom Agent")
            .with_version("1.2.3")
            .with_description("A helpful description")
            .with_dispatcher_prompt("Route wisely")
            .with_skill(DummySkill)
            .build();

        assert_eq!(agent.name(), "Custom Agent");
        assert_eq!(agent.version(), "1.2.3");
        assert_eq!(agent.description(), Some("A helpful description"));
        assert_eq!(agent.dispatcher_prompt(), Some("Route wisely"));

        let skills = agent.skills();
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].id(), DummySkill::metadata().id);
        assert_eq!(skills[0].name(), DummySkill::metadata().name);
    }

    #[test]
    fn skills_maintain_registration_order() {
        let agent = Agent::builder()
            .with_skill(DummySkill)
            .with_skill(SecondarySkill)
            .build();

        let skills = agent.skills();
        assert_eq!(skills.len(), 2);
        assert_eq!(skills[0].id(), DummySkill::metadata().id);
        assert_eq!(skills[1].id(), SecondarySkill::metadata().id);
    }

    struct DummySkill;

    #[cfg_attr(
        all(target_os = "wasi", target_env = "p1"),
        async_trait::async_trait(?Send)
    )]
    #[cfg_attr(
        not(all(target_os = "wasi", target_env = "p1")),
        async_trait::async_trait
    )]
    impl SkillHandler for DummySkill {
        async fn on_request(
            &self,
            _state: &mut crate::runtime::context::State,
            _progress: &crate::runtime::context::ProgressSender,
            _runtime: &dyn crate::runtime::AgentRuntime,
            _content: Content,
        ) -> Result<OnRequestResult, AgentError> {
            Ok(OnRequestResult::Completed {
                message: None,
                artifacts: Vec::new(),
            })
        }

        async fn on_input_received(
            &self,
            _state: &mut crate::runtime::context::State,
            _progress: &crate::runtime::context::ProgressSender,
            _runtime: &dyn crate::runtime::AgentRuntime,
            _content: Content,
        ) -> Result<OnInputResult, AgentError> {
            Ok(OnInputResult::InputRequired {
                message: Content::from_text("Need more input"),
                slot: SkillSlot::new("dummy"),
            })
        }
    }

    impl RegisteredSkill for DummySkill {
        fn metadata() -> Arc<SkillMetadata> {
            Arc::new(SkillMetadata::new(
                "dummy_skill",
                "Dummy Skill",
                "A test skill used for verifying registration behaviour.",
                &[],
                &[],
                &[],
                &[],
            ))
        }
    }

    struct SecondarySkill;

    #[cfg_attr(
        all(target_os = "wasi", target_env = "p1"),
        async_trait::async_trait(?Send)
    )]
    #[cfg_attr(
        not(all(target_os = "wasi", target_env = "p1")),
        async_trait::async_trait
    )]
    impl SkillHandler for SecondarySkill {
        async fn on_request(
            &self,
            _state: &mut crate::runtime::context::State,
            _progress: &crate::runtime::context::ProgressSender,
            _runtime: &dyn crate::runtime::AgentRuntime,
            _content: Content,
        ) -> Result<OnRequestResult, AgentError> {
            Ok(OnRequestResult::Rejected {
                reason: Content::from_text("Not supported"),
            })
        }
    }

    impl RegisteredSkill for SecondarySkill {
        fn metadata() -> Arc<SkillMetadata> {
            Arc::new(SkillMetadata::new(
                "secondary_skill",
                "Secondary Skill",
                "A second skill for ordering tests.",
                &[],
                &[],
                &[],
                &[],
            ))
        }
    }

    // ── AgentSkill builder tests ──────────────────────────────────────────

    #[cfg(feature = "agentskill")]
    mod agentskill_builder_tests {
        use super::*;
        use crate::agent::agentskill::AgentSkillDef;

        const VALID_SKILL_MD: &str = "\
---
name: test-skill
description: A skill used in builder tests.
---

## Instructions

Do the thing.
";

        #[test]
        fn with_skill_def_queues_pending_def() {
            let def = AgentSkillDef::from_skill_md_str(VALID_SKILL_MD, "").expect("valid skill");

            let builder = Agent::builder().with_name("Test").with_skill_def(def);

            assert_eq!(builder.pending_skill_defs.len(), 1);
            assert_eq!(builder.pending_skill_defs[0].id(), "test-skill");
        }

        #[test]
        fn multiple_skill_defs_maintain_order() {
            let def1 = AgentSkillDef::from_skill_md_str(VALID_SKILL_MD, "").expect("valid");
            let def2 = AgentSkillDef::from_skill_md_str(
                "---\nname: second-skill\ndescription: Another skill.\n---\nbody",
                "",
            )
            .expect("valid second skill");

            let builder = Agent::builder().with_skill_def(def1).with_skill_def(def2);

            assert_eq!(builder.pending_skill_defs.len(), 2);
            assert_eq!(builder.pending_skill_defs[0].id(), "test-skill");
            assert_eq!(builder.pending_skill_defs[1].id(), "second-skill");
        }

        #[test]
        fn build_splits_correctly_via_into_parts() {
            // into_parts: programmatic skills go into AgentDefinition,
            // AgentSkills stay pending until RuntimeBuilder injects the LLM.
            let def = AgentSkillDef::from_skill_md_str(VALID_SKILL_MD, "").expect("valid skill");
            let (agent_def, pending) = Agent::builder()
                .with_name("Test")
                .with_skill_def(def)
                .into_parts();

            assert_eq!(agent_def.skills().len(), 0);
            assert_eq!(pending.len(), 1);
            assert_eq!(pending[0].id(), "test-skill");
        }

        #[test]
        fn with_skill_dir_returns_error_for_nonexistent_path() {
            let result = Agent::builder().with_skill_dir("/does/not/exist");
            assert!(result.is_err());
        }

        #[test]
        fn agentskill_and_rust_skill_can_coexist() {
            let def = AgentSkillDef::from_skill_md_str(VALID_SKILL_MD, "").expect("valid skill");
            let builder = Agent::builder().with_skill(DummySkill).with_skill_def(def);

            assert_eq!(builder.inner.skills.len(), 1);
            assert_eq!(builder.pending_skill_defs.len(), 1);
        }

        #[test]
        fn from_agentdefinition_produces_empty_pending_defs() {
            let def = Agent::builder().with_name("Test").build();
            let builder: AgentBuilder = def.into();
            assert_eq!(builder.pending_skill_defs.len(), 0);
        }
    }
}
