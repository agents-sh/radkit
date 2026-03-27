use proc_macro::TokenStream;
use quote::quote;
use syn::LitStr;

/// Implementation for the `include_skill!` macro.
///
/// Reads the `SKILL.md` at compile time, validates the frontmatter structure,
/// and emits an expression that constructs an `AgentSkillDef` at runtime from
/// the embedded `&'static str` constants.
///
/// This means:
/// - The `SKILL.md` content is in the binary (like `include_str!`)
/// - No filesystem I/O happens at startup
/// - Works on WASM targets
/// - If `SKILL.md` is missing or has a broken `---` header, it's a compile error
pub fn generate_include_skill(input: TokenStream) -> TokenStream {
    let path_lit = syn::parse_macro_input!(input as LitStr);
    let path_str = path_lit.value();

    // Resolve path relative to the crate being compiled (same semantics as include_str!)
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR is always set by cargo during proc-macro expansion");

    let skill_dir = std::path::Path::new(&manifest_dir).join(&path_str);
    let skill_md_path = skill_dir.join("SKILL.md");

    // Read the file at compile time so missing files are a compile error.
    let skill_md_content = match std::fs::read_to_string(&skill_md_path) {
        Ok(c) => c,
        Err(e) => {
            return syn::Error::new(
                path_lit.span(),
                format!(
                    "include_skill!: cannot read {}: {e}",
                    skill_md_path.display()
                ),
            )
            .to_compile_error()
            .into();
        }
    };

    // Minimal structural validation at compile time.
    if !skill_md_content.starts_with("---") {
        return syn::Error::new(
            path_lit.span(),
            format!(
                "include_skill!: {} must begin with YAML frontmatter (---)",
                skill_md_path.display()
            ),
        )
        .to_compile_error()
        .into();
    }

    if !skill_md_content.contains("\n---") {
        return syn::Error::new(
            path_lit.span(),
            format!(
                "include_skill!: {} frontmatter is not closed with ---",
                skill_md_path.display()
            ),
        )
        .to_compile_error()
        .into();
    }

    // The directory name is used to validate the `name` field at runtime.
    let dir_name = skill_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();

    // Emit the absolute path as a string literal for include_str!.
    // include_str! requires a string literal known at macro expansion time.
    let skill_md_path_str = skill_md_path.to_string_lossy().to_string();

    let expanded = quote! {
        {
            // Embed the SKILL.md contents into the binary at compile time.
            const SKILL_MD_CONTENT: &str = include_str!(#skill_md_path_str);

            // Parse and validate the frontmatter at runtime (startup).
            // Panics with a clear message if the embedded content is invalid —
            // this is intentional: a bad SKILL.md should be caught early.
            ::radkit::agent::AgentSkillDef::from_skill_md_str(SKILL_MD_CONTENT, #dir_name)
                .unwrap_or_else(|e| panic!(
                    "include_skill!(\"{}\") — invalid SKILL.md: {}",
                    #path_str, e
                ))
        }
    };

    TokenStream::from(expanded)
}
