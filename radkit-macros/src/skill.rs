use proc_macro2::TokenStream;
use quote::quote;
use syn::{parse::Parse, parse::ParseStream, Expr, Ident, LitStr, Token};

use crate::validation::{is_valid_mime_type, suggest_mime_type};

/// Parsed skill macro attributes
pub struct SkillArgs {
    pub id: String,
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
    pub examples: Vec<String>,
    pub input_modes: Vec<String>,
    pub output_modes: Vec<String>,
}

impl Parse for SkillArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut id = None;
        let mut name = None;
        let mut description = None;
        let mut tags = Vec::new();
        let mut examples = Vec::new();
        let mut input_modes = Vec::new();
        let mut output_modes = Vec::new();

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            let _: Token![=] = input.parse()?;

            match key.to_string().as_str() {
                "id" => {
                    let value: LitStr = input.parse()?;
                    id = Some(value.value());
                }
                "name" => {
                    let value: LitStr = input.parse()?;
                    name = Some(value.value());
                }
                "description" => {
                    let value: LitStr = input.parse()?;
                    description = Some(value.value());
                }
                "tags" => {
                    tags = parse_string_array(input)?;
                }
                "examples" => {
                    examples = parse_string_array(input)?;
                }
                "input_modes" => {
                    input_modes = parse_string_array(input)?;
                }
                "output_modes" => {
                    output_modes = parse_string_array(input)?;
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("Unknown skill attribute: {key}"),
                    ));
                }
            }

            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

        // Validate required fields
        let id = id.ok_or_else(|| syn::Error::new(input.span(), "Missing required field: id"))?;
        let name =
            name.ok_or_else(|| syn::Error::new(input.span(), "Missing required field: name"))?;
        let description = description
            .ok_or_else(|| syn::Error::new(input.span(), "Missing required field: description"))?;

        Ok(Self {
            id,
            name,
            description,
            tags,
            examples,
            input_modes,
            output_modes,
        })
    }
}

/// Parse a string array like `["tag1", "tag2"]`
fn parse_string_array(input: ParseStream) -> syn::Result<Vec<String>> {
    let content;
    syn::bracketed!(content in input);
    let exprs = content.parse_terminated(Expr::parse, Token![,])?;

    exprs
        .into_iter()
        .map(|expr| {
            if let Expr::Lit(lit) = expr {
                if let syn::Lit::Str(s) = lit.lit {
                    Ok(s.value())
                } else {
                    Err(syn::Error::new_spanned(
                        lit,
                        "Expected string literal in array",
                    ))
                }
            } else {
                Err(syn::Error::new_spanned(expr, "Expected string literal"))
            }
        })
        .collect()
}

/// Validate MIME types and return error if any are invalid
pub fn validate_mime_types(mime_types: &[String]) -> syn::Result<()> {
    for mime_type in mime_types {
        if !is_valid_mime_type(mime_type) {
            let suggestions = suggest_mime_type(mime_type);
            let error_msg = if suggestions.is_empty() {
                format!("Invalid MIME type: '{mime_type}'. See https://www.iana.org/assignments/media-types/media-types.xhtml for valid types.")
            } else {
                let suggested = suggestions.join(", ");
                format!("Invalid MIME type: '{mime_type}'. Did you mean one of: {suggested}?")
            };
            return Err(syn::Error::new(proc_macro2::Span::call_site(), error_msg));
        }
    }
    Ok(())
}

/// Generate the skill implementation
#[allow(clippy::needless_pass_by_value)] // We consume args and clone item, this is correct for proc macros
pub fn generate_skill_impl(args: SkillArgs, item: TokenStream) -> TokenStream {
    // Validate MIME types
    if let Err(e) = validate_mime_types(&args.input_modes) {
        return e.to_compile_error();
    }
    if let Err(e) = validate_mime_types(&args.output_modes) {
        return e.to_compile_error();
    }

    // Extract struct name from the item
    let struct_item: syn::ItemStruct = match syn::parse2(item.clone()) {
        Ok(s) => s,
        Err(e) => return e.to_compile_error(),
    };
    let struct_name = &struct_item.ident;

    // Convert Rust strings to token streams
    let id = &args.id;
    let name = &args.name;
    let description = &args.description;

    let tags: Vec<&str> = args.tags.iter().map(String::as_str).collect();
    let examples: Vec<&str> = args.examples.iter().map(String::as_str).collect();
    let input_modes: Vec<&str> = args.input_modes.iter().map(String::as_str).collect();
    let output_modes: Vec<&str> = args.output_modes.iter().map(String::as_str).collect();

    // Generate the implementation
    quote! {
        #item

        impl ::radkit::agent::RegisteredSkill for #struct_name {
            fn metadata() -> ::std::sync::Arc<::radkit::agent::SkillMetadata> {
                ::std::sync::Arc::new(::radkit::agent::SkillMetadata::new(
                    #id,
                    #name,
                    #description,
                    &[#(#tags),*],
                    &[#(#examples),*],
                    &[#(#input_modes),*],
                    &[#(#output_modes),*],
                ))
            }
        }
    }
}
