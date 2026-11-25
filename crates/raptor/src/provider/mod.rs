//! LLM provider abstraction.
//!
//! Offline provider selection:
//!
//! 1. `RAPTOR_PROVIDER=sim|ollama` (explicit; errors loudly if misconfigured)
//! 2. a local Ollama server (`OLLAMA_HOST`, default `127.0.0.1:11434`)
//! 3. the built-in deterministic **simulator** (always available, clearly labelled everywhere)
//!
//! Model ids are discovered from the local Ollama server or set via `RAPTOR_MODEL`.

pub mod http;
pub mod sim;

use std::sync::Arc;

pub use http::HttpProvider;
pub use sim::{Chaos, SimConfig, SimProvider};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Planner,
    Executor,
    Critic,
    Synthesizer,
}

/// A finished upstream result handed to the synthesizer.
#[derive(Clone, Debug, Default)]
pub struct NodeOutput {
    pub id: String,
    pub output: String,
    pub is_tool: bool,
    pub low_confidence: bool,
}

/// Structured context. Real providers use the rendered `system`/`user` prompts;
/// the simulator reads this struct directly (so it never has to parse prompts).
#[derive(Clone, Debug, Default)]
pub struct Ctx {
    pub request: String,
    pub task_id: String,
    pub task: String,
    pub deps: Vec<(String, String)>,
    /// 1-based attempt number within the transient-retry loop.
    pub attempt: u32,
    /// Critic feedback when this call is a repair round.
    pub feedback: Option<String>,
    pub memory: Vec<String>,
    pub tools: Vec<String>,
    /// Output under review (critic role).
    pub output: Option<String>,
    /// Completed outputs (synthesizer role).
    pub outputs: Vec<NodeOutput>,
    /// Tasks that failed or were skipped (synthesizer role).
    pub incomplete: Vec<String>,
    pub word_limit: usize,
}

#[derive(Clone, Debug)]
pub struct LlmRequest {
    pub role: Role,
    pub system: String,
    pub user: String,
    pub ctx: Ctx,
}

#[derive(Clone, Debug)]
pub struct LlmError {
    /// Worth retrying (rate limit, 5xx, network).
    pub transient: bool,
    /// The per-run call budget was exhausted.
    pub budget: bool,
    pub msg: String,
}

impl LlmError {
    pub fn transient(msg: impl Into<String>) -> Self {
        LlmError {
            transient: true,
            budget: false,
            msg: msg.into(),
        }
    }
    pub fn permanent(msg: impl Into<String>) -> Self {
        LlmError {
            transient: false,
            budget: false,
            msg: msg.into(),
        }
    }
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.msg)
    }
}

pub trait Provider: Send + Sync {
    fn name(&self) -> String;
    /// Resolved model id, or a placeholder until first use.
    fn model(&self) -> String;
    fn simulated(&self) -> bool {
        false
    }
    /// Validate the provider and resolve the model up front, so errors are clear and early.
    fn warm_up(&self) -> Result<(), String> {
        Ok(())
    }
    fn available_models(&self) -> Result<Vec<String>, String> {
        Err(format!("{} does not support model discovery", self.name()))
    }
    fn complete(&self, req: &LlmRequest) -> Result<String, LlmError>;
}

pub fn clean_key(v: Option<String>) -> Option<String> {
    let v = v?
        .trim()
        .trim_matches(|c| c == '"' || c == '\'')
        .trim()
        .to_string();
    if v.is_empty() {
        None
    } else {
        Some(v)
    }
}

/// Build one of the offline providers. Remote endpoints and API-key providers are disabled.
pub fn build_named(
    name: &str,
    env: &dyn Fn(&str) -> Option<String>,
    sim: SimConfig,
) -> Result<Arc<dyn Provider>, String> {
    let model = clean_key(env("RAPTOR_MODEL"));
    match name {
        "sim" | "simulator" => Ok(Arc::new(SimProvider::new(sim))),
        "ollama" => {
            let base = ollama_base(env);
            if !http::is_local_ollama_url(&base) {
                return Err("offline mode only supports Ollama on this computer (127.0.0.1); remote hosts are disabled".into());
            }
            let timeout = clean_key(env("RAPTOR_TIMEOUT_SECS"))
                .and_then(|v| v.parse().ok())
                .unwrap_or(300);
            Ok(Arc::new(
                HttpProvider::openai_compatible("ollama", base, model)
                    .with_hint("Is Ollama running? Start it with `ollama serve`, then install a model, e.g. `ollama pull llama3.2`.")
                    .with_timeout(timeout),
            ))
        }
        other => Err(format!(
            "unknown provider '{other}' (offline mode supports only: sim, ollama)"
        )),
    }
}

/// `OLLAMA_HOST` may be `0.0.0.0:11434` (a *bind* address, unusable as a client target),
/// `localhost` (can resolve to IPv6 while Ollama listens on IPv4) or lack a scheme/port.
pub fn ollama_base(env: &dyn Fn(&str) -> Option<String>) -> String {
    let host = clean_key(env("OLLAMA_HOST")).unwrap_or_else(|| "127.0.0.1:11434".into());
    let host = host
        .trim_end_matches('/')
        .replace("0.0.0.0", "127.0.0.1")
        .replace("localhost", "127.0.0.1")
        .replace("[::]", "127.0.0.1");
    let host = if host
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .contains(':')
    {
        host
    } else {
        format!("{host}:11434")
    };
    if host.starts_with("http") {
        format!("{host}/v1")
    } else {
        format!("http://{host}/v1")
    }
}

/// Auto-detect the best available provider. Never fails to find *something*: the
/// simulator is the guaranteed last resort.
pub fn detect(
    env: &dyn Fn(&str) -> Option<String>,
    sim: SimConfig,
    probe_ollama: bool,
) -> Result<Arc<dyn Provider>, String> {
    if let Some(explicit) = clean_key(env("RAPTOR_PROVIDER")) {
        return build_named(&explicit.to_lowercase(), env, sim);
    }
    if probe_ollama {
        let base = ollama_base(env);
        if !http::is_local_ollama_url(&base) {
            return Err("offline mode only supports Ollama on this computer (127.0.0.1); remote hosts are disabled".into());
        }
        if http::ollama_alive(&base) {
            return build_named("ollama", env, sim);
        }
    }
    Ok(Arc::new(SimProvider::new(sim)))
}

/// Local Ollama endpoint probed during auto-detection.
pub fn local_candidates(env: &dyn Fn(&str) -> Option<String>) -> Vec<(&'static str, String)> {
    vec![("ollama", ollama_base(env))]
}

/// Pure model-picking heuristic used when no model is configured: prefer small/fast chat models.
pub fn pick_model(ids: &[String]) -> Option<String> {
    const SKIP: &[&str] = &[
        "embed",
        "tts",
        "whisper",
        "image",
        "audio",
        "moderation",
        "realtime",
        "dall",
        "transcribe",
        "rerank",
        "guard",
        "vision-preview",
        "search",
        "instruct-vl",
    ];
    const PREFER: &[&str] = &[
        "0.5b", "1b", "mini", "flash", "lite", "small", "nano", "3b", "8b", "chat",
    ];
    let mut chat: Vec<&String> = ids
        .iter()
        .filter(|i| !SKIP.iter().any(|s| i.to_lowercase().contains(s)))
        .collect();
    chat.sort();
    for p in PREFER {
        if let Some(m) = chat.iter().find(|i| i.to_lowercase().contains(p)) {
            return Some((*m).clone());
        }
    }
    chat.first().map(|s| (*s).clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let m: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k| m.get(k).cloned()
    }

    #[test]
    fn no_keys_falls_back_to_simulator_instead_of_crashing() {
        let p = detect(&env_of(&[]), SimConfig::default(), false).unwrap();
        assert!(p.simulated());
        assert_eq!(p.name(), "simulator");
    }

    #[test]
    fn api_keys_do_not_enable_online_providers() {
        let p = detect(
            &env_of(&[("OPENAI_API_KEY", "test-key")]),
            SimConfig::default(),
            false,
        )
        .unwrap();
        assert!(p.simulated(), "API keys must not select an online provider");
    }

    #[test]
    fn online_and_unsupported_provider_names_are_rejected() {
        let e = env_of(&[("RAPTOR_PROVIDER", "openai")]);
        let err = detect(&e, SimConfig::default(), false).err().unwrap();
        assert!(err.contains("offline mode"), "{err}");
        let e = env_of(&[("RAPTOR_PROVIDER", "bogus")]);
        assert!(detect(&e, SimConfig::default(), false)
            .err()
            .unwrap()
            .contains("unknown provider"));
    }

    #[test]
    fn ollama_host_variants_are_normalized_to_a_reachable_client_url() {
        let b = |h: &str| ollama_base(&env_of(&[("OLLAMA_HOST", h)]));
        assert_eq!(b("0.0.0.0:11434"), "http://127.0.0.1:11434/v1");
        assert_eq!(b("localhost"), "http://127.0.0.1:11434/v1");
        assert_eq!(b("http://localhost:11434/"), "http://127.0.0.1:11434/v1");
        assert_eq!(ollama_base(&env_of(&[])), "http://127.0.0.1:11434/v1");
        assert!(build_named(
            "ollama",
            &env_of(&[("OLLAMA_HOST", "192.168.1.20:11434")]),
            SimConfig::default()
        )
        .err()
        .unwrap()
        .contains("remote hosts are disabled"));
    }

    #[test]
    fn custom_or_remote_endpoints_are_not_selected() {
        let e = env_of(&[("RAPTOR_BASE_URL", "http://example.com/v1")]);
        assert!(detect(&e, SimConfig::default(), false).unwrap().simulated());
        assert!(build_named("local", &e, SimConfig::default())
            .err()
            .unwrap()
            .contains("offline mode supports only"));
    }

    #[test]
    fn ollama_urls_must_be_loopback_without_userinfo() {
        assert!(http::is_local_ollama_url("http://127.0.0.1:11434/v1"));
        assert!(http::is_local_ollama_url("http://127.0.0.1/v1"));
        assert!(!http::is_local_ollama_url("https://127.0.0.1:11434/v1"));
        assert!(!http::is_local_ollama_url("http://192.168.1.4:11434/v1"));
        assert!(!http::is_local_ollama_url(
            "http://127.0.0.1:80@evil.example/v1"
        ));
    }

    #[test]
    fn model_picker_prefers_small_chat_models_and_skips_non_chat() {
        let ids: Vec<String> = [
            "nomic-embed-text",
            "llama3.2",
            "qwen2.5:3b",
            "whisper",
            "tts",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(pick_model(&ids).unwrap(), "qwen2.5:3b");
        let ids: Vec<String> = ["llama3.3:70b", "qwen2.5:0.5b"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(pick_model(&ids).unwrap(), "qwen2.5:0.5b");
        assert_eq!(
            pick_model(&["only-model".to_string()]).unwrap(),
            "only-model"
        );
        assert!(pick_model(&["text-embedding-x".to_string()]).is_none());
    }
}
