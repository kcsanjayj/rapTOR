//! The critic reviews **every** completed LLM task (v1 skipped the final wave) and returns a
//! 0-10 score plus actionable feedback. A score below the pass mark triggers a repair round.
//!
//! Honest limit: an LLM critic is a heuristic, not a ground-truth verifier. If the critic is
//! unavailable the task is accepted *and the fact is recorded* (`critic_unavailable` event).

use crate::llm::Llm;
use crate::model::Task;
use crate::provider::{Ctx, LlmError, LlmRequest, Role};
use crate::util::{extract_json_object, truncate};

#[derive(Clone, Debug, PartialEq)]
pub struct Review {
    pub score: u8,
    pub issues: Vec<String>,
    pub feedback: String,
}

const SYSTEM: &str = "You are the quality reviewer in a multi-agent pipeline. Judge ONE task output for correctness, \
relevance, completeness and use of the upstream results. Be strict but fair: do not demand sources or extra research \
unless the task asks for them. Reply with ONLY a JSON object: \
{\"score\": <integer 0-10>, \"issues\": [<short strings>], \"feedback\": \"<one actionable instruction for the author, or empty>\"}. \
Scoring: 9-10 excellent; 7-8 good enough; 4-6 meaningful problems; 0-3 unusable.";

pub fn review(
    llm: &Llm,
    request: &str,
    task: &Task,
    deps: &[(String, String)],
    output: &str,
    on_fail: &dyn Fn(u32, &LlmError, bool),
) -> Result<Review, LlmError> {
    let upstream: String = deps
        .iter()
        .map(|(id, o)| format!("### {id}\n{}\n", truncate(o, 600)))
        .collect();
    let user = format!(
        "ORIGINAL USER REQUEST:\n{request}\n\nTASK ({}):\n{}\n\nUPSTREAM RESULTS AVAILABLE TO THE AUTHOR:\n{}\nOUTPUT TO REVIEW:\n{}",
        task.id,
        task.description,
        if upstream.is_empty() { "(none)\n".to_string() } else { upstream },
        truncate(output, 3000)
    );
    let ctx = Ctx {
        request: request.into(),
        task_id: task.id.clone(),
        task: task.description.clone(),
        deps: deps.to_vec(),
        output: Some(output.to_string()),
        ..Default::default()
    };
    let raw = llm.call(
        LlmRequest {
            role: Role::Critic,
            system: SYSTEM.into(),
            user,
            ctx,
        },
        on_fail,
    )?;
    parse_review(&raw).map_err(LlmError::permanent)
}

pub fn parse_review(raw: &str) -> Result<Review, String> {
    let json = extract_json_object(raw).ok_or("critic returned no JSON object")?;
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("critic JSON invalid: {e}"))?;
    let score = match v.get("score") {
        Some(serde_json::Value::Number(n)) => n.as_f64().unwrap_or(0.0),
        Some(serde_json::Value::String(s)) => s
            .trim()
            .parse::<f64>()
            .map_err(|_| "critic score is not a number")?,
        _ => return Err("critic JSON has no 'score'".into()),
    };
    let issues = v
        .get("issues")
        .and_then(|i| i.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| match x {
                    serde_json::Value::String(s) => Some(s.clone()),
                    other => other
                        .get("description")
                        .and_then(|d| d.as_str())
                        .map(|s| s.to_string()),
                })
                .collect()
        })
        .unwrap_or_default();
    let feedback = v
        .get("feedback")
        .and_then(|f| f.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    Ok(Review {
        score: score.round().clamp(0.0, 10.0) as u8,
        issues,
        feedback,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fenced_json_and_clamps() {
        let r = parse_review("```json\n{\"score\": 14, \"issues\": [\"a\", {\"description\": \"b\"}], \"feedback\": \" fix \"}\n```").unwrap();
        assert_eq!(
            r,
            Review {
                score: 10,
                issues: vec!["a".into(), "b".into()],
                feedback: "fix".into()
            }
        );
        assert_eq!(parse_review("{\"score\": \"6.4\"}").unwrap().score, 6);
        assert_eq!(parse_review("{\"score\": -3}").unwrap().score, 0);
    }

    #[test]
    fn rejects_garbage_instead_of_guessing() {
        assert!(parse_review("looks fine to me!").is_err());
        assert!(parse_review("{\"issues\": []}").is_err());
        assert!(parse_review("{\"score\": \"high\"}").is_err());
    }
}
