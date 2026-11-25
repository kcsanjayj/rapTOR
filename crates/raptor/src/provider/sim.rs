//! Deterministic offline provider.
//!
//! **What it is:** a rule-based stand-in that lets the *entire* pipeline (plan, parallel
//! execution, critic, repair, synthesis) run with no API key, no network and no cost.
//! It is what makes the engine testable and the evals reproducible in CI.
//!
//! **What it is not:** an AI. Its text is template output and is labelled `[simulated]`.
//! It never invents facts about the world. The UI and CLI always show when it is active.
//!
//! Fault injection ([`Chaos`]) is seeded and deterministic, so "the engine recovered from a
//! 503" is a repeatable test, not an anecdote.

use super::{Ctx, LlmError, LlmRequest, Provider, Role};
use crate::dag::normalize_id;
use crate::util::{hash_unit, truncate};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Chaos {
    /// Probability that an executor call fails transiently (never more than twice in a row).
    pub transient_rate: f64,
    /// Probability that a task's first draft is deliberately low quality (critic should catch it).
    pub degrade_rate: f64,
    pub seed: u64,
    /// Task ids that always fail (simulates a permanent outage for that task).
    pub fail_tasks: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SimConfig {
    /// Average simulated model latency per call, in ms. 0 disables sleeping (tests/evals).
    pub latency_ms: u64,
    pub chaos: Chaos,
}

pub struct SimProvider {
    cfg: SimConfig,
}

impl SimProvider {
    pub fn new(cfg: SimConfig) -> SimProvider {
        SimProvider { cfg }
    }

    fn pause(&self, ctx_key: &str, salt: u64, weight: f64) {
        if self.cfg.latency_ms == 0 {
            return;
        }
        let jitter = 0.6 + 0.8 * hash_unit(self.cfg.chaos.seed, ctx_key, salt);
        let ms = (self.cfg.latency_ms as f64 * weight * jitter) as u64;
        std::thread::sleep(Duration::from_millis(ms));
    }
}

impl Provider for SimProvider {
    fn name(&self) -> String {
        "simulator".into()
    }
    fn model(&self) -> String {
        "raptor-sim-1".into()
    }
    fn simulated(&self) -> bool {
        true
    }

    fn complete(&self, req: &LlmRequest) -> Result<String, LlmError> {
        let c = &req.ctx;
        match req.role {
            Role::Planner => {
                self.pause(&c.request, 1, 1.0);
                Ok(plan_json(&c.request, &c.memory))
            }
            Role::Executor => {
                self.pause(&c.task_id, 2 + c.attempt as u64, 1.4);
                let ch = &self.cfg.chaos;
                if ch.fail_tasks.iter().any(|t| normalize_id(t) == c.task_id) {
                    return Err(LlmError::transient(format!(
                        "simulated persistent outage for task '{}'",
                        c.task_id
                    )));
                }
                if c.attempt < 3
                    && ch.transient_rate > 0.0
                    && hash_unit(ch.seed, &c.task_id, 100 + c.attempt as u64) < ch.transient_rate
                {
                    return Err(LlmError::transient("simulated 503 from provider"));
                }
                let degrade = c.feedback.is_none()
                    && ch.degrade_rate > 0.0
                    && hash_unit(ch.seed, &c.task_id, 7) < ch.degrade_rate;
                Ok(exec_output(c, degrade))
            }
            Role::Critic => {
                self.pause(&c.task_id, 50, 0.6);
                Ok(critic_json(c))
            }
            Role::Synthesizer => {
                self.pause(&c.request, 90, 1.0);
                Ok(synth_text(c))
            }
        }
    }
}

// ------------------------------------------------------------------ planner

/// Extract `(A, B, context)` from phrasings like "Compare A and B for X", "A vs B", "which is better, A or B".
pub fn extract_pair(request: &str) -> Option<(String, String, String)> {
    let mut s = request
        .trim()
        .trim_end_matches(['?', '.', '!'])
        .trim()
        .to_string();
    let lower = s.to_lowercase();
    let prefixes = [
        "compare ",
        "comparison of ",
        "comparison between ",
        "differences between ",
        "difference between ",
        "contrast ",
        "which is better, ",
        "which is better: ",
        "which is better ",
        "what is better, ",
        "pros and cons of ",
    ];
    let mut had_prefix = false;
    for p in prefixes {
        if lower.starts_with(p) {
            s = s[p.len()..].trim().to_string();
            had_prefix = true;
            break;
        }
    }
    let lower = s.to_lowercase();
    let seps: &[(&str, bool)] = &[
        (" vs. ", false),
        (" vs ", false),
        (" versus ", false),
        (" compared to ", false),
        (" compared with ", false),
        (" and ", true),
        (" or ", true),
    ];
    for (sep, needs_prefix) in seps {
        if *needs_prefix && !had_prefix {
            continue;
        }
        if let Some(pos) = lower.find(sep) {
            let left = s[..pos].trim();
            let right_full = s[pos + sep.len()..].trim();
            let rl = right_full.to_lowercase();
            let cut = [
                " for ", " in ", " when ", " if ", " on ", " with ", " to ", ",", " - ", " — ",
                " which ", " as ", " at ",
            ]
            .iter()
            .filter_map(|m| rl.find(m))
            .min();
            let (b, ctx) = match cut {
                Some(i) => (
                    right_full[..i].trim(),
                    right_full[i..].trim().trim_start_matches(',').trim(),
                ),
                None => (right_full, ""),
            };
            let clean = |x: &str| {
                x.trim()
                    .trim_matches(|c: char| c == ',' || c == ':' || c == '"')
                    .trim()
                    .to_string()
            };
            let (a, b) = (clean(left), clean(b));
            let ok = |x: &str| !x.is_empty() && x.len() <= 40 && x.split_whitespace().count() <= 4;
            if ok(&a) && ok(&b) && a.to_lowercase() != b.to_lowercase() {
                return Some((a, b, ctx.to_string()));
            }
        }
    }
    None
}

/// Longest arithmetic-looking substring, if the request asks for a calculation.
pub fn extract_expression(request: &str) -> Option<String> {
    let lower = request.to_lowercase();
    let wants_calc = [
        "calculate",
        "compute",
        "evaluate",
        "what is ",
        "what's ",
        "solve",
        "how much is",
    ]
    .iter()
    .any(|k| lower.contains(k));
    if !wants_calc {
        return None;
    }
    let ok = |c: char| c.is_ascii_digit() || "+-*/^%(). ".contains(c);
    let mut best = String::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, best: &mut String| {
        let t = cur.trim().to_string();
        let starts = t
            .chars()
            .next()
            .map(|c| c.is_ascii_digit() || c == '(' || c == '-')
            .unwrap_or(false);
        let ends = t
            .chars()
            .last()
            .map(|c| c.is_ascii_digit() || c == ')')
            .unwrap_or(false);
        let has_op = t.chars().any(|c| "+-*/^%".contains(c));
        if starts && ends && has_op && t.len() > best.len() {
            *best = t;
        }
        cur.clear();
    };
    for c in request.chars() {
        if ok(c) {
            cur.push(c);
        } else {
            flush(&mut cur, &mut best);
        }
    }
    flush(&mut cur, &mut best);
    if best.is_empty() {
        None
    } else {
        Some(best)
    }
}

fn plan_json(request: &str, memory: &[String]) -> String {
    let goal = truncate(request.trim(), 120);
    let note = if memory.is_empty() {
        String::new()
    } else {
        format!(" (informed by {} similar past run(s))", memory.len())
    };
    let tasks: Vec<Value> = if let Some(expr) = extract_expression(request) {
        vec![
            json!({"id":"calculate","description":format!("Evaluate the expression {expr} exactly"),"dependencies":[],"kind":"tool","tool":"calculator","args":{"expression":expr}}),
            json!({"id":"explain_result","description":format!("Explain the computed result of {expr} in one sentence"),"dependencies":["calculate"],"kind":"llm"}),
        ]
    } else if let Some((a, b, ctx)) = extract_pair(request) {
        let (mut ia, mut ib) = (normalize_id(&a), normalize_id(&b));
        if ia.is_empty() {
            ia = "option_a".into();
        }
        if ib.is_empty() || ib == ia {
            ib = "option_b".into();
        }
        let (ra, rb) = (format!("research_{ia}"), format!("research_{ib}"));
        let ctx_txt = if ctx.is_empty() {
            String::new()
        } else {
            format!(" {ctx}")
        };
        vec![
            json!({"id":ra,"description":format!("Research {a}: strengths, weaknesses and fit{ctx_txt}"),"dependencies":[],"kind":"llm"}),
            json!({"id":rb,"description":format!("Research {b}: strengths, weaknesses and fit{ctx_txt}"),"dependencies":[],"kind":"llm"}),
            json!({"id":"compare","description":format!("Compare {a} and {b} against the research results"),"dependencies":[ra,rb],"kind":"llm"}),
            json!({"id":"recommend","description":format!("Recommend {a} or {b} and justify the choice"),"dependencies":["compare"],"kind":"llm"}),
        ]
    } else {
        vec![
            json!({"id":"clarify_goal","description":format!("Restate the goal and success criteria for: {goal}"),"dependencies":[],"kind":"llm"}),
            json!({"id":"gather_background","description":format!("Collect the background knowledge needed for: {goal}"),"dependencies":[],"kind":"llm"}),
            json!({"id":"analyze","description":"Analyze the goal against the background and list the key points","dependencies":["clarify_goal","gather_background"],"kind":"llm"}),
            json!({"id":"final_answer","description":"Write the final answer from the analysis","dependencies":["analyze"],"kind":"llm"}),
        ]
    };
    json!({"goal": format!("{goal}{note}"), "tasks": tasks}).to_string()
}

// ----------------------------------------------------------------- executor

fn exec_output(c: &Ctx, degrade: bool) -> String {
    if degrade {
        return "[simulated] rough draft, needs work.".to_string();
    }
    let builds_on = if c.deps.is_empty() {
        "nothing (root task)".to_string()
    } else {
        c.deps
            .iter()
            .map(|(id, _)| format!("<- {id}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let signal = if c.deps.is_empty() {
        String::new()
    } else {
        let parts: Vec<String> = c
            .deps
            .iter()
            .map(|(id, o)| format!("{id}: \"{}\"", truncate(first_line(o), 70)))
            .collect();
        format!("\nUpstream signal: {}", parts.join(" | "))
    };
    let revised = match &c.feedback {
        Some(f) => format!("\nRevised after review: {}", truncate(f, 90)),
        None => String::new(),
    };
    format!(
        "[simulated] {}\nBuilds on: {builds_on}{signal}\nNote: deterministic placeholder text; no language model was called.{revised}",
        c.task
    )
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or("")
}

// ------------------------------------------------------------------- critic

/// A real (if simple) heuristic critic: length, upstream references, placeholders.
fn critic_json(c: &Ctx) -> String {
    let out = c.output.clone().unwrap_or_default();
    let words = out.split_whitespace().count();
    let mut score: i32 = 9;
    let mut issues: Vec<String> = vec![];
    let mut fixes: Vec<String> = vec![];
    if words < 12 {
        score -= 5;
        issues.push(format!("too short / vague ({words} words)"));
        fixes.push("expand the answer with concrete, task-specific detail".into());
    }
    for (id, _) in &c.deps {
        if !out.contains(id.as_str()) {
            score -= 3;
            issues.push(format!("does not use upstream result '{id}'"));
            fixes.push(format!("explicitly build on the result of '{id}'"));
        }
    }
    if out.contains("TODO") {
        score -= 3;
        issues.push("contains TODO placeholder".into());
    }
    json!({"score": score.clamp(0, 10), "issues": issues, "feedback": fixes.join("; ")}).to_string()
}

// -------------------------------------------------------------- synthesizer

fn synth_text(c: &Ctx) -> String {
    let mut s = String::from("Simulated answer (no language model connected).\n");
    s.push_str(&format!("Request: {}\n", truncate(c.request.trim(), 160)));
    for o in c.outputs.iter().filter(|o| o.is_tool) {
        s.push_str(&format!(
            "Computed result: {} (from the deterministic {} tool)\n",
            o.output.trim(),
            o.id
        ));
    }
    let prose: Vec<_> = c.outputs.iter().filter(|o| !o.is_tool).collect();
    if !prose.is_empty() {
        s.push_str("Pipeline findings:\n");
        for o in prose {
            let flag = if o.low_confidence {
                " (low confidence)"
            } else {
                ""
            };
            s.push_str(&format!(
                "- {}{flag}: {}\n",
                o.id,
                truncate(
                    first_line(&o.output).trim_start_matches("[simulated] "),
                    100
                )
            ));
        }
    }
    if !c.incomplete.is_empty() {
        s.push_str(&format!(
            "Incomplete tasks: {}. This answer is partial.\n",
            c.incomplete.join(", ")
        ));
    }
    s.push_str("Select local Ollama and install a model for real content.");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(s: &str) -> Option<(String, String, String)> {
        extract_pair(s)
    }

    #[test]
    fn pair_extraction_variants() {
        assert_eq!(
            pair("Compare Rust and Go for a CLI tool").unwrap(),
            ("Rust".into(), "Go".into(), "for a CLI tool".into())
        );
        assert_eq!(
            pair("Python vs Rust for web APIs?").unwrap(),
            ("Python".into(), "Rust".into(), "for web APIs".into())
        );
        assert_eq!(
            pair("Which is better, tea or coffee?").unwrap(),
            ("tea".into(), "coffee".into(), "".into())
        );
        assert_eq!(pair("PostgreSQL versus MySQL").unwrap().1, "MySQL");
    }

    #[test]
    fn plain_and_or_do_not_trigger_comparison() {
        assert!(pair("Plan a 3-day trip to Rome and Paris").is_none());
        assert!(pair("Write tests or docs").is_none());
        assert!(pair("Explain recursion").is_none());
    }

    #[test]
    fn expression_extraction() {
        assert_eq!(
            extract_expression("Calculate 12*(3+4)").unwrap(),
            "12*(3+4)"
        );
        assert_eq!(extract_expression("What is 2^10 + 5?").unwrap(), "2^10 + 5");
        assert_eq!(extract_expression("calculate 9**9**9").unwrap(), "9**9**9");
        assert!(extract_expression("Plan a 3-day trip to Rome").is_none()); // no calc verb
        assert!(extract_expression("What is love").is_none());
    }

    #[test]
    fn planner_output_is_valid_json_for_each_shape() {
        for r in [
            "Compare Rust and Go",
            "Calculate 1+2",
            "Plan a trip to Rome",
        ] {
            let v: Value = serde_json::from_str(&plan_json(r, &[])).unwrap();
            assert!(v["tasks"].as_array().unwrap().len() >= 2, "{r}");
        }
    }

    #[test]
    fn critic_flags_vague_and_unreferenced_output() {
        let mut c = Ctx {
            deps: vec![("a".into(), "x".into())],
            output: Some("[simulated] rough draft, needs work.".into()),
            ..Default::default()
        };
        let v: Value = serde_json::from_str(&critic_json(&c)).unwrap();
        assert!(v["score"].as_i64().unwrap() < 7);
        c.output = Some(exec_output(
            &Ctx {
                task: "do x".into(),
                deps: vec![("a".into(), "x".into())],
                ..Default::default()
            },
            false,
        ));
        let v: Value = serde_json::from_str(&critic_json(&c)).unwrap();
        assert!(v["score"].as_i64().unwrap() >= 7, "{v}");
    }
}
