//! Startup banner display for the runtime server.
//!
//! This module provides Spring Boot-style startup information
//! including agent details, endpoints, and server configuration.

use crate::agent::AgentDefinition;

/// Displays a startup banner with agent and server information.
///
/// # Arguments
///
/// * `address` - The bind address the server is listening on
/// * `base_url` - The public-facing base URL (if configured)
/// * `agent` - The single agent served by this runtime
pub fn display_banner(address: &str, base_url: Option<&str>, agent: &AgentDefinition) {
    let banner = r"
  ____            _ _  ___ _
 |  _ \ __ _   __| | |/ (_) |_
 | |_) / _` | / _` | ' /| | __|
 |  _ < (_| || (_| | . \| | |_
 |_| \_\__,_| \__,_|_|\_\_|\__|
                                ";

    println!("{banner}");
    println!("  Agentic SDK for Rust");
    let effective_base_url = base_url.map_or_else(|| infer_base_url(address), String::from);

    #[cfg(feature = "dev-ui")]
    println!(
        "  Dev UI:     {}/",
        effective_base_url.trim_end_matches('/')
    );

    #[cfg(not(feature = "dev-ui"))]
    println!("  API URL:    {}", effective_base_url.trim_end_matches('/'));

    println!();

    let normalized_base = effective_base_url.trim_end_matches('/');

    // Server info
    println!("Server Configuration:");
    println!("  Bind Address: {address}");

    if let Some(url) = base_url {
        println!("  Base URL:     {url}");
    } else {
        println!("  Base URL:     <inferred from bind address>");
    }

    println!();

    let version = agent.version();
    let skill_count = agent.skills().len();

    println!("Agent:");
    println!("  \u{2022} {} (v{})", agent.name(), version);
    println!("    Skills:      {skill_count}");
    println!("    Agent Card:  {normalized_base}/.well-known/agent-card.json");
    println!("    RPC:         {normalized_base}/rpc");
    println!("    Send:        {normalized_base}/message:send");
    println!("    Streaming:   {normalized_base}/message:stream");
    println!("    Tasks:       {normalized_base}/tasks/{{id}}");
    println!("    Subscribe:   {normalized_base}/tasks/{{id}}:subscribe");
    println!();

    println!("Ready to accept connections!");
    println!();
}

/// Infers a base URL from a bind address.
///
/// This handles common bind address patterns:
/// - `0.0.0.0:PORT` → `http://localhost:PORT`
/// - `127.0.0.1:PORT` → `http://localhost:PORT`
/// - `localhost:PORT` → `http://localhost:PORT`
/// - `HOST:PORT` → `http://HOST:PORT`
/// - `PORT` → `http://localhost:PORT`
fn infer_base_url(bind_address: &str) -> String {
    // Extract port from address
    let port = bind_address
        .split(':')
        .next_back()
        .and_then(|p| p.parse::<u16>().ok());

    match (bind_address, port) {
        // 0.0.0.0:PORT → localhost:PORT
        (addr, Some(port)) if addr.starts_with("0.0.0.0:") => {
            format!("http://localhost:{port}")
        }
        // 127.0.0.1:PORT → localhost:PORT
        (addr, Some(port)) if addr.starts_with("127.0.0.1:") => {
            format!("http://localhost:{port}")
        }
        // localhost:PORT → http://localhost:PORT
        (addr, Some(port)) if addr.starts_with("localhost:") => {
            format!("http://localhost:{port}")
        }
        // Just a port number → localhost:PORT
        (addr, Some(port)) if addr == port.to_string() => {
            format!("http://localhost:{port}")
        }
        // HOST:PORT → http://HOST:PORT
        (_, Some(port)) => {
            let host = bind_address
                .rsplit_once(':')
                .map_or("localhost", |(h, _)| h);
            format!("http://{host}:{port}")
        }
        // No port found → just localhost
        _ => "http://localhost".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_base_url() {
        assert_eq!(infer_base_url("0.0.0.0:8080"), "http://localhost:8080");
        assert_eq!(infer_base_url("127.0.0.1:3000"), "http://localhost:3000");
        assert_eq!(infer_base_url("localhost:9000"), "http://localhost:9000");
        assert_eq!(infer_base_url("8080"), "http://localhost:8080");
        assert_eq!(
            infer_base_url("example.com:7000"),
            "http://example.com:7000"
        );
    }
}
