//! Agent definition primitives.

pub mod builder;
pub mod llm_function;
pub mod llm_worker;
pub mod skill;
pub mod structured_parser;

#[cfg(feature = "agentskill")]
pub mod agentskill;
#[cfg(feature = "agentskill")]
pub(crate) mod llm_skill;

pub use builder::{Agent, AgentBuilder, AgentDefinition, SkillRegistration};
pub use llm_function::LlmFunction;
pub use llm_worker::{LlmWorker, LlmWorkerBuilder};
pub use skill::{
    Artifact, OnInputResult, OnRequestResult, RegisteredSkill, SkillHandler, SkillMetadata,
    SkillSlot, WorkStatus,
};

#[cfg(feature = "agentskill")]
pub use agentskill::AgentSkillDef;
