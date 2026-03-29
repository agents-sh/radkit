//! Gemini Computer Use example.
//!
//! Demonstrates how to build a browser-control agent with
//! [`GeminiComputerUseWorker`]. The example uses a stub [`ComputerUseHandler`]
//! that records every action and returns a blank 1×1 PNG screenshot so the
//! example compiles and runs offline without a real browser.
//!
//! To connect to a real browser, replace `StubBrowser` with your own
//! implementation backed by a library such as `chromiumoxide`, `fantoccini`,
//! or the Browserbase remote-browser API.
//!
//! # Running
//!
//! ```sh
//! GEMINI_API_KEY=<your-key> cargo run --example gemini_computer_use
//! ```
//!
//! # Safety note
//!
//! When the Gemini model attaches a `safety_decision` with
//! `decision = "require_confirmation"` to an action, you **must** obtain
//! explicit user consent before executing it. This example demonstrates that
//! pattern in [`StubBrowser::execute`]. Per the Gemini API Terms of Service
//! you are not allowed to bypass these confirmation requests automatically.

use std::sync::Arc;

use radkit::models::providers::{
    ActionOutcome, ComputerUseAction, ComputerUseHandler, GeminiLlm,
};

// ============================================================================
// Stub browser implementation
// ============================================================================

/// A minimal [`ComputerUseHandler`] that logs actions and returns a blank
/// screenshot. Replace this with your real browser integration.
struct StubBrowser {
    /// Current simulated URL – updated when the model calls `navigate`.
    current_url: std::sync::Mutex<String>,
}

impl StubBrowser {
    fn new(start_url: impl Into<String>) -> Self {
        Self {
            current_url: std::sync::Mutex::new(start_url.into()),
        }
    }

    fn url(&self) -> String {
        self.current_url.lock().unwrap().clone()
    }

    /// Produce a minimal valid 1×1 white PNG so the example runs without a
    /// real browser process.
    fn blank_png() -> Vec<u8> {
        // Pre-encoded 1×1 white PNG (67 bytes).
        vec![
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, // PNG signature
            0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41, // IDAT chunk
            0x54, 0x08, 0xd7, 0x63, 0xf8, 0xcf, 0xc0, 0x00,
            0x00, 0x00, 0x02, 0x00, 0x01, 0xe2, 0x21, 0xbc,
            0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, // IEND chunk
            0x44, 0xae, 0x42, 0x60, 0x82,
        ]
    }
}

#[async_trait::async_trait]
impl ComputerUseHandler for StubBrowser {
    async fn screenshot(&self) -> Result<Vec<u8>, String> {
        println!("  [browser] screenshot captured (stub)");
        Ok(Self::blank_png())
    }

    async fn execute(&self, action: ComputerUseAction) -> ActionOutcome {
        // ----------------------------------------------------------------
        // Safety gate — MUST be implemented in production code
        // ----------------------------------------------------------------
        if let Some(ref sd) = action.safety_decision {
            if sd.requires_confirmation() {
                // In a real application, prompt the user here.
                // For this example we auto-confirm so the loop can proceed,
                // but in production you must obtain explicit human consent.
                println!(
                    "  [safety] confirmation required: {}",
                    sd.explanation
                );
                println!("  [safety] auto-confirming for stub example (replace with real prompt)");
                // If the user refuses: return ActionOutcome::Denied { reason: "user declined".into() }
            }
        }

        // ----------------------------------------------------------------
        // Dispatch on action name
        // ----------------------------------------------------------------
        println!("  [browser] executing: {} args={}", action.name, action.args);

        match action.name.as_str() {
            "open_web_browser" => {
                println!("  [browser] browser already open");
            }
            "navigate" => {
                if let Some(url) = action.args.get("url").and_then(|v| v.as_str()) {
                    *self.current_url.lock().unwrap() = url.to_string();
                    println!("  [browser] navigated to {url}");
                }
            }
            "click_at" => {
                let x = action.args.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
                let y = action.args.get("y").and_then(|v| v.as_i64()).unwrap_or(0);
                println!("  [browser] click at ({x}, {y})");
            }
            "type_text_at" => {
                let text = action
                    .args
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                println!("  [browser] type: \"{text}\"");
            }
            "scroll_document" => {
                let dir = action
                    .args
                    .get("direction")
                    .and_then(|v| v.as_str())
                    .unwrap_or("down");
                println!("  [browser] scroll {dir}");
            }
            "wait_5_seconds" => {
                println!("  [browser] waiting 5 seconds (stubbed, skipping actual sleep)");
            }
            "go_back" => {
                println!("  [browser] go back");
            }
            "go_forward" => {
                println!("  [browser] go forward");
            }
            "search" => {
                *self.current_url.lock().unwrap() = "https://www.google.com".to_string();
                println!("  [browser] navigated to search engine");
            }
            "hover_at" => {
                let x = action.args.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
                let y = action.args.get("y").and_then(|v| v.as_i64()).unwrap_or(0);
                println!("  [browser] hover at ({x}, {y})");
            }
            "key_combination" => {
                let keys = action
                    .args
                    .get("keys")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                println!("  [browser] key combination: {keys}");
            }
            "scroll_at" => {
                let dir = action
                    .args
                    .get("direction")
                    .and_then(|v| v.as_str())
                    .unwrap_or("down");
                println!("  [browser] scroll_at {dir}");
            }
            "drag_and_drop" => {
                println!("  [browser] drag and drop");
            }
            other => {
                println!("  [browser] unrecognised action: {other} (ignored)");
            }
        }

        ActionOutcome::Success { url: self.url() }
    }
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Gemini Computer Use Example ===\n");

    // Load API key from environment.
    // Set GEMINI_API_KEY before running.
    let llm = Arc::new(GeminiLlm::from_env(
        "gemini-2.5-computer-use-preview-10-2025",
    )?);

    let browser = Arc::new(StubBrowser::new("https://www.google.com"));

    let worker = llm
        .computer_use_worker(Arc::clone(&browser) as Arc<dyn ComputerUseHandler>)
        // Limit turns so the stub doesn't run forever during CI / quick tests
        .with_max_turns(10);

    let goal = "Go to https://www.google.com and search for 'Rust programming language'. \
                List the top 3 result titles you can see.";

    println!("Goal: {goal}\n");
    println!("--- agent loop ---");

    match worker.run(goal).await {
        Ok(answer) => {
            println!("\n--- final answer ---");
            println!("{answer}");
        }
        Err(e) => {
            eprintln!("\nAgent error: {e}");
        }
    }

    Ok(())
}
