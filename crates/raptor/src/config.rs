//! Run-level configuration (limits, retries, budgets).

use std::path::PathBuf;

pub fn load_dotenv(path: &std::path::Path) -> Result<(), String> {
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };

    for (key, value) in parse_dotenv(&content)? {
        if std::env::var_os(&key).is_none() {
            std::env::set_var(key, value);
        }
    }
    Ok(())
}

pub fn save_dotenv_values(path: &std::path::Path, values: &[(&str, &str)]) -> Result<(), String> {
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    parse_dotenv(&content)?;

    let mut lines: Vec<String> = content.lines().map(str::to_string).collect();
    for (key, value) in values {
        let replacement = format!(
            "{key}=\"{}\"",
            value.replace('\\', "\\\\").replace('"', "\\\"")
        );
        let mut found = false;
        for line in &mut lines {
            if dotenv_key(line).as_deref() == Some(key) {
                *line = replacement.clone();
                found = true;
            }
        }
        if !found {
            lines.push(replacement);
        }
    }

    let output = if lines.is_empty() {
        String::new()
    } else {
        format!("{}\n", lines.join("\n"))
    };
    std::fs::write(path, output).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

fn dotenv_key(line: &str) -> Option<String> {
    let line = line.trim();
    let line = line.strip_prefix("export ").unwrap_or(line);
    let (key, _) = line.split_once('=')?;
    let key = key.trim();
    if key.is_empty() || key.starts_with('#') {
        None
    } else {
        Some(key.to_string())
    }
}

fn parse_dotenv(content: &str) -> Result<Vec<(String, String)>, String> {
    let mut entries = Vec::new();
    for (index, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let (key, raw_value) = line
            .split_once('=')
            .ok_or_else(|| format!(".env line {} is missing '='", index + 1))?;
        let key = key.trim();
        if key.is_empty() || !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(format!(
                ".env line {} has an invalid variable name",
                index + 1
            ));
        }
        let raw_value = raw_value.trim();
        let value = if raw_value.starts_with('"')
            && raw_value.ends_with('"')
            && raw_value.len() >= 2
        {
            raw_value[1..raw_value.len() - 1]
                .replace("\\\"", "\"")
                .replace("\\\\", "\\")
        } else if raw_value.starts_with('\'') && raw_value.ends_with('\'') && raw_value.len() >= 2 {
            raw_value[1..raw_value.len() - 1].to_string()
        } else {
            raw_value.to_string()
        };
        entries.push((key.to_string(), value));
    }
    Ok(entries)
}

#[derive(Clone, Debug)]
pub struct RunConfig {
    /// Parallel worker threads.
    pub max_workers: usize,
    /// Max tasks in a plan.
    pub max_tasks: usize,
    /// Transient-failure retries per LLM call (so `retries + 1` attempts).
    pub retries: u32,
    /// Critic-guided repair rounds per task.
    pub max_repairs: u32,
    /// Critic score (0-10) at or above which a task is accepted.
    pub pass_score: u8,
    /// Hard cap on provider calls in one run (planner + executors + critics + synthesizer).
    pub max_llm_calls: usize,
    /// Base for exponential backoff between retries.
    pub backoff_base_ms: u64,
    /// Stop *starting* new tasks after this many seconds.
    pub deadline_secs: u64,
    /// Soft word limit passed to the synthesizer.
    pub answer_words: usize,
}

impl Default for RunConfig {
    fn default() -> Self {
        RunConfig {
            max_workers: 4,
            max_tasks: 12,
            retries: 2,
            max_repairs: 2,
            pass_score: 7,
            max_llm_calls: 60,
            backoff_base_ms: 150,
            deadline_secs: 300,
            answer_words: 220,
        }
    }
}

/// Where traces and memory live. `RAPTOR_DATA_DIR` or `./data`.
pub fn data_dir_from_env() -> PathBuf {
    std::env::var("RAPTOR_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dotenv_parser_handles_comments_quotes_and_equals_in_values() {
        let parsed = parse_dotenv(
            "# config\nOPENAI_API_KEY=\"key=with\\\\slash\\\"quote\"\nRAPTOR_MODEL='model-id'\n",
        )
        .unwrap();
        assert_eq!(
            parsed,
            vec![
                ("OPENAI_API_KEY".into(), "key=with\\slash\"quote".into()),
                ("RAPTOR_MODEL".into(), "model-id".into())
            ]
        );
    }

    #[test]
    fn dotenv_parser_rejects_malformed_entries() {
        assert!(parse_dotenv("not-an-assignment")
            .unwrap_err()
            .contains("line 1"));
    }

    #[test]
    fn saving_dotenv_values_preserves_other_settings_and_quotes_values() {
        let path = std::env::temp_dir().join(format!("raptor-dotenv-{}.env", std::process::id()));
        std::fs::write(
            &path,
            "# keep this comment\nOPENAI_API_KEY=old\nOTHER_SETTING=value\n",
        )
        .unwrap();

        save_dotenv_values(
            &path,
            &[
                ("OPENAI_API_KEY", "new=key\\value"),
                ("RAPTOR_MODEL", "model-id"),
            ],
        )
        .unwrap();
        let saved = std::fs::read_to_string(&path).unwrap();
        assert!(saved.contains("# keep this comment"));
        assert!(saved.contains("OPENAI_API_KEY=\"new=key\\\\value\""));
        assert!(saved.contains("OTHER_SETTING=value"));
        assert!(saved.contains("RAPTOR_MODEL=\"model-id\""));

        std::fs::remove_file(path).unwrap();
    }
}
