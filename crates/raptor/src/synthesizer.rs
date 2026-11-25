//! Synthesizer: turns completed task outputs into one user-facing answer.
//!
//! If some tasks failed or were skipped the answer must say so (no silent partials).
//! If the model call itself fails, a deterministic fallback is used and flagged.

use crate::llm::Llm;
use crate::provider::{Ctx, LlmError, LlmRequest, NodeOutput, Role};
use crate::util::truncate;

const SYSTEM: &str = "You write the final answer for the user. Use ONLY the task results provided; do not add facts \
that are not supported by them. Lead with the conclusion. Be concise. Do not mention tasks, agents, DAGs or the pipeline. \
If some work is listed as incomplete, state plainly what could not be done.";

pub fn synthesize(
    llm: &Llm,
    request: &str,
    outputs: &[NodeOutput],
    incomplete: &[String],
    word_limit: usize,
    on_fail: &dyn Fn(u32, &LlmError, bool),
) -> Result<String, LlmError> {
    let mut user = format!("USER REQUEST:\n{request}\n\nTASK RESULTS:\n");
    for o in outputs {
        let flag = if o.low_confidence {
            " (reviewer had low confidence in this result)"
        } else {
            ""
        };
        user.push_str(&format!(
            "### {}{flag}\n{}\n\n",
            o.id,
            truncate(&o.output, 2500)
        ));
    }
    if !incomplete.is_empty() {
        user.push_str(&format!(
            "INCOMPLETE (failed or skipped): {}\n\n",
            incomplete.join(", ")
        ));
    }
    user.push_str(&format!(
        "Write the final answer in at most {word_limit} words."
    ));
    let ctx = Ctx {
        request: request.into(),
        outputs: outputs.to_vec(),
        incomplete: incomplete.to_vec(),
        word_limit,
        ..Default::default()
    };
    let raw = llm.call(
        LlmRequest {
            role: Role::Synthesizer,
            system: SYSTEM.into(),
            user,
            ctx,
        },
        on_fail,
    )?;
    Ok(limit_words(
        raw.trim(),
        word_limit.saturating_mul(3) / 2 + 20,
    ))
}

/// Deterministic answer used when the synthesizer call fails.
pub fn fallback_answer(outputs: &[NodeOutput], incomplete: &[String]) -> String {
    let mut s = String::from("(Automatic summary: the final-answer step was unavailable.)\n");
    if let Some(last) = outputs.iter().rev().find(|o| !o.output.trim().is_empty()) {
        s.push_str(last.output.trim());
    }
    if !incomplete.is_empty() {
        s.push_str(&format!("\nIncomplete tasks: {}.", incomplete.join(", ")));
    }
    s
}

/// Cut to at most `n` words **preserving whitespace/newlines** (v1 flattened them).
pub fn limit_words(text: &str, n: usize) -> String {
    let mut words = 0usize;
    let mut in_word = false;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            in_word = false;
        } else if !in_word {
            in_word = true;
            words += 1;
            if words > n {
                return format!("{}…", text[..i].trim_end());
            }
        }
    }
    text.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_words_preserves_newlines_and_cuts_on_word_boundary() {
        let t = "one two\nthree four\n- five six";
        assert_eq!(limit_words(t, 4), "one two\nthree four…");
        assert_eq!(limit_words(t, 99), t);
    }

    #[test]
    fn fallback_mentions_incomplete_work() {
        let o = vec![NodeOutput {
            id: "a".into(),
            output: "result A".into(),
            ..Default::default()
        }];
        let s = fallback_answer(&o, &["b".into()]);
        assert!(s.contains("result A") && s.contains("Incomplete tasks: b"));
    }
}
