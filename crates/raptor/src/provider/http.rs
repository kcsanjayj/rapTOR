//! Local Ollama provider over its OpenAI-compatible HTTP endpoint.

use super::{pick_model, LlmError, LlmRequest, Provider};
use serde_json::{json, Value};
use std::sync::Mutex;
use std::time::Duration;

pub struct HttpProvider {
    name: String,
    base_url: String,
    model_override: Option<String>,
    resolved: Mutex<Option<String>>,
    agent: ureq::Agent,
    hint: Option<String>,
}

fn agent(timeout: Duration) -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(5))
        .timeout(timeout)
        .redirects(0)
        .build()
}

impl HttpProvider {
    pub fn openai_compatible(name: &str, base_url: String, model: Option<String>) -> HttpProvider {
        HttpProvider {
            name: name.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model_override: model,
            resolved: Mutex::new(None),
            agent: agent(Duration::from_secs(120)),
            hint: None,
        }
    }

    /// Extra advice appended to connection errors (e.g. "start `ollama serve`").
    pub fn with_hint(mut self, hint: &str) -> HttpProvider {
        self.hint = Some(hint.to_string());
        self
    }

    /// Overall request timeout. Local models can take minutes to load and answer.
    pub fn with_timeout(mut self, secs: u64) -> HttpProvider {
        self.agent = agent(Duration::from_secs(secs));
        self
    }

    fn ensure_loopback(&self) -> Result<(), LlmError> {
        if is_local_ollama_url(&self.base_url) {
            Ok(())
        } else {
            Err(LlmError::permanent(
                "offline mode only permits HTTP requests to local Ollama on 127.0.0.1",
            ))
        }
    }

    fn finish(&self, r: Result<ureq::Response, ureq::Error>) -> Result<Value, LlmError> {
        match r {
            Ok(resp) => resp.into_json::<Value>().map_err(|e| {
                LlmError::permanent(format!("{}: invalid JSON in response: {e}", self.name))
            }),
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                Err(map_status(&self.name, code, &body))
            }
            Err(ureq::Error::Transport(t)) => Err(LlmError::transient(format!(
                "{}: cannot reach {} ({t}){}",
                self.name,
                self.base_url,
                self.hint
                    .as_ref()
                    .map(|h| format!(". {h}"))
                    .unwrap_or_default()
            ))),
        }
    }

    fn get_json(&self, path: &str) -> Result<Value, LlmError> {
        self.ensure_loopback()?;
        let url = format!("{}{}", self.base_url, path);
        self.finish(self.agent.get(&url).call())
    }

    fn post_json(&self, path: &str, body: &Value) -> Result<Value, LlmError> {
        self.ensure_loopback()?;
        let url = format!("{}{}", self.base_url, path);
        self.finish(self.agent.post(&url).send_json(body.clone()))
    }

    pub fn list_models(&self) -> Result<Vec<String>, LlmError> {
        let v = self.get_json("/models")?;
        let arr = v
            .get("data")
            .or_else(|| v.get("models"))
            .and_then(|d| d.as_array())
            .cloned()
            .unwrap_or_default();
        Ok(arr
            .iter()
            .filter_map(|m| {
                m.get("id")
                    .or_else(|| m.get("name"))
                    .and_then(|x| x.as_str())
            })
            .map(|s| s.trim_start_matches("models/").to_string())
            .collect())
    }

    fn resolve_model(&self) -> Result<String, LlmError> {
        if let Some(m) = &self.model_override {
            return Ok(m.clone());
        }
        if let Some(m) = self.resolved.lock().unwrap().clone() {
            return Ok(m);
        }
        let picked = match self.list_models() {
            Ok(ids) => pick_model(&ids).ok_or_else(|| {
                LlmError::permanent(format!(
                    "{}: no usable chat model found (listed: {}). Install/load one, or set RAPTOR_MODEL.{}",
                    self.name,
                    ids.join(", "),
                    self.hint.as_ref().map(|h| format!(" {h}")).unwrap_or_default()
                ))
            })?,
            Err(e) => {
                return Err(LlmError {
                    msg: format!("{} (could not discover a model; set RAPTOR_MODEL to skip discovery)", e.msg),
                    ..e
                })
            }
        };
        *self.resolved.lock().unwrap() = Some(picked.clone());
        Ok(picked)
    }
}

fn api_message(body: &str) -> String {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    let msg = parsed
        .as_ref()
        .and_then(|v| {
            v.pointer("/error/message")
                .or_else(|| v.get("message"))
                .or_else(|| v.get("error"))
        })
        .and_then(|m| m.as_str().map(|s| s.to_string()))
        .unwrap_or_else(|| body.chars().take(200).collect());
    msg.chars().take(240).collect()
}

pub(crate) fn is_local_ollama_url(base_url: &str) -> bool {
    let Some(rest) = base_url.strip_prefix("http://") else {
        return false;
    };
    let authority = rest.split('/').next().unwrap_or_default();
    if authority.contains('@') {
        return false;
    }
    let mut parts = authority.split(':');
    let host = parts.next().unwrap_or_default();
    let port = parts.next();
    host == "127.0.0.1"
        && parts.next().is_none()
        && port
            .map(|p| !p.is_empty() && p.parse::<u16>().is_ok())
            .unwrap_or(true)
}

pub fn map_status(provider: &str, code: u16, body: &str) -> LlmError {
    let detail = api_message(body);
    match code {
        401 | 403 => LlmError::permanent(format!(
            "{provider}: local server rejected the request (HTTP {code}). Offline mode expects a local Ollama server. {detail}"
        )),
        404 => LlmError::permanent(format!("{provider}: not found (HTTP 404). Wrong model or base URL? {detail}")),
        408 | 409 | 425 | 429 => LlmError::transient(format!("{provider}: rate limited / busy (HTTP {code}). {detail}")),
        500..=599 => LlmError::transient(format!("{provider}: server error (HTTP {code}). {detail}")),
        _ => LlmError::permanent(format!("{provider}: request rejected (HTTP {code}). {detail}")),
    }
}

impl Provider for HttpProvider {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn model(&self) -> String {
        if let Some(m) = &self.model_override {
            return m.clone();
        }
        self.resolved
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_else(|| "auto".into())
    }

    fn warm_up(&self) -> Result<(), String> {
        self.resolve_model().map(|_| ()).map_err(|e| e.msg)
    }

    fn available_models(&self) -> Result<Vec<String>, String> {
        self.list_models().map_err(|e| e.msg)
    }

    fn complete(&self, req: &LlmRequest) -> Result<String, LlmError> {
        let model = self.resolve_model()?;
        let body = json!({
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user}
            ]
        });
        let v = self.post_json("/chat/completions", &body)?;
        let content = v.pointer("/choices/0/message/content");
        match content {
            Some(Value::String(s)) if !s.trim().is_empty() => Ok(s.clone()),
            Some(Value::Array(parts)) => {
                let t: String = parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("");
                if t.trim().is_empty() {
                    Err(LlmError::transient(format!(
                        "{}: empty completion",
                        self.name
                    )))
                } else {
                    Ok(t)
                }
            }
            _ => Err(LlmError::transient(format!(
                "{}: response had no message content",
                self.name
            ))),
        }
    }
}

/// Quick liveness probe for a local Ollama server (400 ms budget).
pub fn ollama_alive(base_url: &str) -> bool {
    probe(base_url).map(|m| !m.is_empty()).unwrap_or(false)
}

/// List models from the local Ollama endpoint with a short timeout.
pub fn probe(base_url: &str) -> Result<Vec<String>, String> {
    let p = HttpProvider::openai_compatible("ollama", base_url.to_string(), None).with_timeout(2);
    p.list_models().map_err(|e| e.msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_mapping_classifies_retryability() {
        assert!(!map_status("x", 401, "{}").transient);
        assert!(!map_status("x", 404, "{}").transient);
        assert!(map_status("x", 429, "{}").transient);
        assert!(map_status("x", 503, "{}").transient);
        assert!(!map_status("x", 400, r#"{"error":{"message":"bad field"}}"#).transient);
        assert!(map_status("x", 400, r#"{"error":{"message":"bad field"}}"#)
            .msg
            .contains("bad field"));
    }
}
