//! `AgentSkill` loading and registration.
//!
//! This module provides [`AgentSkillDef`] — the runtime representation of an
//! `AgentSkill` parsed from a `SKILL.md` file. `AgentSkills` can be loaded either
//! at compile time via the [`include_skill!`] macro or at runtime via
//! [`AgentSkillDef::from_dir`].
//!
//! # `AgentSkills` specification
//!
//! An `AgentSkill` is a directory containing a `SKILL.md` file:
//!
//! ```text
//! skill-name/
//! ├── SKILL.md          # Required: YAML frontmatter + Markdown instructions
//! ├── scripts/          # Optional: executable code
//! └── references/       # Optional: documentation
//! ```
//!
//! The `SKILL.md` file must start with YAML frontmatter:
//!
//! ```markdown
//! ---
//! name: skill-name
//! description: What this skill does and when to use it.
//! ---
//!
//! # Instructions
//!
//! Step-by-step instructions for the LLM...
//! ```
//!
//! # Examples
//!
//! ```ignore
//! // Compile-time — embedded in binary, zero I/O at startup
//! Agent::builder()
//!     .with_skill_def(include_skill!("./skills/summarise"))
//!     .build()
//!
//! // Runtime — loaded from the filesystem at startup
//! Agent::builder()
//!     .with_skill_dir("./skills/summarise")?
//!     .build()
//! ```

use std::{collections::HashMap, path::Path, sync::Arc};

use serde::Deserialize;

use crate::{
    agent::{builder::SkillRegistration, llm_skill::LlmSkillHandler, skill::SkillMetadata},
    errors::AgentError,
    models::BaseLlm,
};

// ── Public type ───────────────────────────────────────────────────────────────

/// A parsed `AgentSkill` ready to be registered with an agent.
///
/// Create via [`AgentSkillDef::from_dir`] for runtime loading or via
/// [`include_skill!`] for compile-time embedding. Pass to
/// [`AgentBuilder::with_skill_def`].
#[derive(Debug)]
#[allow(dead_code)] // fields are used by RuntimeBuilder behind #[cfg(feature = "runtime")]
pub struct AgentSkillDef {
    pub(crate) metadata: Arc<SkillMetadata>,
    /// SKILL.md Markdown body (everything after the YAML frontmatter).
    pub(crate) instructions: String,
}

impl AgentSkillDef {
    /// Load an `AgentSkill` from a directory containing `SKILL.md`.
    ///
    /// Reads and validates the `SKILL.md` file, parses the YAML frontmatter,
    /// and verifies that the `name` field matches the directory name.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The directory or `SKILL.md` cannot be read
    /// - The frontmatter is missing, malformed, or fails validation
    /// - The `name` field doesn't match the directory name
    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self, AgentError> {
        let path = path.as_ref();
        let skill_md = path.join("SKILL.md");

        let content = std::fs::read_to_string(&skill_md).map_err(|e| {
            AgentError::InvalidInput(format!("Cannot read {}: {e}", skill_md.display()))
        })?;

        let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        Self::from_skill_md_str(&content, dir_name)
    }

    /// Parse a `SKILL.md` string directly.
    ///
    /// Used internally by [`include_skill!`] (which embeds the string at
    /// compile time via `include_str!`) and by [`from_dir`].
    ///
    /// `dir_name` is the parent directory name used to validate that the
    /// frontmatter `name` field matches. Pass an empty string to skip this
    /// validation (useful in tests).
    ///
    /// # Errors
    ///
    /// Returns an error if frontmatter is missing/invalid or validation fails.
    pub fn from_skill_md_str(content: &str, dir_name: &str) -> Result<Self, AgentError> {
        let (frontmatter, body) = split_frontmatter(content)?;

        // Validate name matches directory when dir_name is provided.
        if !dir_name.is_empty() && frontmatter.name != dir_name {
            return Err(AgentError::InvalidInput(format!(
                "AgentSkill `name` field '{}' must match the directory name '{}'",
                frontmatter.name, dir_name
            )));
        }

        validate_skill_name(&frontmatter.name)?;
        validate_description(&frontmatter.description)?;

        let allowed_tools: Vec<String> = frontmatter
            .allowed_tools
            .as_deref()
            .unwrap_or("")
            .split_whitespace()
            .map(str::to_string)
            .collect();

        let metadata = Arc::new(SkillMetadata {
            id: frontmatter.name.clone(),
            name: to_display_name(&frontmatter.name),
            description: frontmatter.description,
            tags: Vec::new(),
            examples: Vec::new(),
            input_modes: vec!["text/plain".to_string()],
            output_modes: vec!["text/plain".to_string()],
            instructions: Some(body.clone()),
            license: frontmatter.license,
            compatibility: frontmatter.compatibility,
            allowed_tools,
        });

        Ok(Self {
            metadata,
            instructions: body,
        })
    }

    /// Returns the skill id (the `name` field from frontmatter).
    #[must_use]
    pub fn id(&self) -> &str {
        &self.metadata.id
    }

    /// Returns the parsed [`SkillMetadata`].
    #[must_use]
    pub fn metadata(&self) -> &SkillMetadata {
        &self.metadata
    }

    /// Converts this definition into a [`SkillRegistration`] by attaching an LLM.
    ///
    /// Called by `RuntimeBuilder::build()` which injects the shared LLM.
    // Used from `runtime::RuntimeBuilder::build`; cross-cfg dead_code lint is a
    // false positive here.
    #[allow(dead_code)]
    pub(crate) fn into_registration(self, llm: Arc<dyn BaseLlm>) -> SkillRegistration {
        let handler = Arc::new(LlmSkillHandler::new(llm, &self.instructions));
        SkillRegistration {
            metadata: self.metadata,
            handler,
        }
    }
}

// ── YAML frontmatter ──────────────────────────────────────────────────────────

/// Parsed YAML frontmatter fields as defined by the `AgentSkills` specification.
#[derive(Debug, Deserialize)]
struct SkillFrontmatter {
    name: String,
    description: String,
    #[serde(default)]
    license: Option<String>,
    #[serde(default)]
    compatibility: Option<String>,
    /// Arbitrary key-value metadata (ignored for now, preserved for future use).
    #[serde(default)]
    #[allow(dead_code)]
    metadata: HashMap<String, String>,
    /// Space-delimited list of pre-approved tool names.
    #[serde(default, rename = "allowed-tools")]
    allowed_tools: Option<String>,
}

/// Split `SKILL.md` content into (frontmatter, body).
///
/// The file must start with `---`, have a closing `---`, and the body
/// is everything after the closing delimiter.
fn split_frontmatter(content: &str) -> Result<(SkillFrontmatter, String), AgentError> {
    let rest = content.strip_prefix("---").ok_or_else(|| {
        AgentError::InvalidInput("SKILL.md must begin with YAML frontmatter (---)".to_string())
    })?;

    // Find the closing ---
    let end_offset = rest.find("\n---").ok_or_else(|| {
        AgentError::InvalidInput("SKILL.md frontmatter is not closed with ---".to_string())
    })?;

    let yaml = &rest[..end_offset];
    // Body: everything after "\n---", skipping the optional leading newline.
    let body = rest[end_offset + 4..].trim_start_matches('\n').to_string();

    let frontmatter: SkillFrontmatter = serde_yaml::from_str(yaml)
        .map_err(|e| AgentError::InvalidInput(format!("Invalid SKILL.md frontmatter: {e}")))?;

    Ok((frontmatter, body))
}

// ── Validation ────────────────────────────────────────────────────────────────

fn validate_skill_name(name: &str) -> Result<(), AgentError> {
    if name.is_empty() || name.len() > 64 {
        return Err(AgentError::InvalidInput(
            "Skill name must be 1–64 characters".to_string(),
        ));
    }
    if name.starts_with('-') || name.ends_with('-') {
        return Err(AgentError::InvalidInput(
            "Skill name must not start or end with a hyphen".to_string(),
        ));
    }
    if name.contains("--") {
        return Err(AgentError::InvalidInput(
            "Skill name must not contain consecutive hyphens (--)".to_string(),
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return Err(AgentError::InvalidInput(
            "Skill name may only contain lowercase letters (a-z), digits (0-9), and hyphens (-)"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_description(desc: &str) -> Result<(), AgentError> {
    if desc.is_empty() || desc.len() > 1024 {
        return Err(AgentError::InvalidInput(
            "Skill description must be 1–1024 characters".to_string(),
        ));
    }
    Ok(())
}

/// Convert a kebab-case skill id to a Title Case display name.
///
/// `"pdf-processing"` → `"Pdf Processing"`
fn to_display_name(id: &str) -> String {
    id.split('-')
        .map(|word| {
            let mut chars = word.chars();
            chars.next().map_or_else(String::new, |first| {
                first.to_uppercase().collect::<String>() + chars.as_str()
            })
        })
        .collect::<Vec<_>>()
        .join(" ")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_SKILL_MD: &str = "\
---
name: my-skill
description: Does something useful when you need it.
---

# Instructions

Step 1: do the thing.
";

    const FULL_SKILL_MD: &str = "\
---
name: pdf-processing
description: Extracts text from PDFs. Use when working with PDF files.
license: MIT
compatibility: Requires no special packages.
allowed-tools: Bash(python3:*) Read Write
metadata:
  author: example-org
  version: \"1.0\"
---

Extract text, tables, and metadata from PDF documents.
";

    #[test]
    fn parses_minimal_frontmatter() {
        let def =
            AgentSkillDef::from_skill_md_str(MINIMAL_SKILL_MD, "my-skill").expect("valid skill");

        assert_eq!(def.id(), "my-skill");
        assert_eq!(def.metadata().name, "My Skill");
        assert_eq!(
            def.metadata().description,
            "Does something useful when you need it."
        );
        assert!(def.metadata().license.is_none());
        assert!(def.metadata().allowed_tools.is_empty());
        assert!(def.instructions.contains("Step 1: do the thing."));
    }

    #[test]
    fn parses_full_frontmatter() {
        let def =
            AgentSkillDef::from_skill_md_str(FULL_SKILL_MD, "pdf-processing").expect("valid skill");

        assert_eq!(def.id(), "pdf-processing");
        assert_eq!(def.metadata().name, "Pdf Processing");
        assert_eq!(def.metadata().license.as_deref(), Some("MIT"));
        assert_eq!(
            def.metadata().compatibility.as_deref(),
            Some("Requires no special packages.")
        );
        assert_eq!(
            def.metadata().allowed_tools,
            vec!["Bash(python3:*)", "Read", "Write"]
        );
        assert!(def.metadata().instructions.is_some());
    }

    #[test]
    fn skips_name_validation_when_dir_name_empty() {
        // dir_name="" disables the name-must-match-dir check
        let def = AgentSkillDef::from_skill_md_str(MINIMAL_SKILL_MD, "")
            .expect("should succeed without dir check");
        assert_eq!(def.id(), "my-skill");
    }

    #[test]
    fn rejects_name_mismatch() {
        let err = AgentSkillDef::from_skill_md_str(MINIMAL_SKILL_MD, "wrong-dir")
            .expect_err("name mismatch should fail");
        assert!(err.to_string().contains("must match"));
    }

    #[test]
    fn rejects_uppercase_name() {
        let content = "---\nname: MySkill\ndescription: A skill.\n---\n\nbody";
        let err = AgentSkillDef::from_skill_md_str(content, "").expect_err("uppercase rejected");
        assert!(err.to_string().contains("lowercase"));
    }

    #[test]
    fn rejects_leading_hyphen() {
        let content = "---\nname: -skill\ndescription: A skill.\n---\n\nbody";
        let err =
            AgentSkillDef::from_skill_md_str(content, "").expect_err("leading hyphen rejected");
        assert!(err.to_string().contains("hyphen"));
    }

    #[test]
    fn rejects_trailing_hyphen() {
        let content = "---\nname: skill-\ndescription: A skill.\n---\n\nbody";
        let err =
            AgentSkillDef::from_skill_md_str(content, "").expect_err("trailing hyphen rejected");
        assert!(err.to_string().contains("hyphen"));
    }

    #[test]
    fn rejects_consecutive_hyphens() {
        let content = "---\nname: my--skill\ndescription: A skill.\n---\n\nbody";
        let err = AgentSkillDef::from_skill_md_str(content, "")
            .expect_err("consecutive hyphens rejected");
        assert!(err.to_string().contains("consecutive"));
    }

    #[test]
    fn rejects_empty_description() {
        let content = "---\nname: my-skill\ndescription: \"\"\n---\n\nbody";
        let err =
            AgentSkillDef::from_skill_md_str(content, "").expect_err("empty description rejected");
        assert!(err.to_string().contains("description"));
    }

    #[test]
    fn rejects_missing_frontmatter_delimiter() {
        let content = "name: my-skill\ndescription: test\n\nbody";
        let err = AgentSkillDef::from_skill_md_str(content, "").expect_err("missing --- rejected");
        assert!(err.to_string().contains("frontmatter"));
    }

    #[test]
    fn rejects_unclosed_frontmatter() {
        let content = "---\nname: my-skill\ndescription: test\n\nbody";
        let err = AgentSkillDef::from_skill_md_str(content, "").expect_err("unclosed --- rejected");
        assert!(err.to_string().contains("not closed"));
    }

    #[test]
    fn display_name_conversion() {
        assert_eq!(to_display_name("pdf-processing"), "Pdf Processing");
        assert_eq!(to_display_name("my-skill"), "My Skill");
        assert_eq!(to_display_name("single"), "Single");
        assert_eq!(to_display_name("a-b-c"), "A B C");
    }

    #[test]
    fn allowed_tools_parsed_from_space_delimited_string() {
        let content = "\
---
name: my-skill
description: A skill.
allowed-tools: Bash(git:*) Read Write
---
body";
        let def = AgentSkillDef::from_skill_md_str(content, "").expect("valid");
        assert_eq!(
            def.metadata().allowed_tools,
            vec!["Bash(git:*)", "Read", "Write"]
        );
    }
}
