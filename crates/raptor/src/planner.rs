//! Planner: request -> validated DAG.
//!
//! Flow: ask the model for JSON -> parse -> validate (cycles, unknown deps, limits, tools).
//! If rejected, tell the model *why* and ask once more. If it still fails, fall back to a
//! **request-agnostic** 3-step plan (v1 fell back to a hard-coded "Python vs Rust" plan).

use crate::dag::{self, PlanLimits};
use crate::events::{EventKind, EventLog};
use crate::llm::Llm;
use crate::model::{Kind, Plan, Task};
use crate::provider::{Ctx, LlmRequest, Role};
use crate::tools::Tools;
use crate::util::{extract_json_object, truncate};
use serde_json::json;

const SYSTEM: &str =
    "You are the planner of a multi-agent system. Decompose the user's request into a SMALL \
directed acyclic graph of tasks. Return ONLY a JSON object, no prose.";

pub fn prompt(
    request: &str,
    memory: &[String],
    tools: &Tools,
    limits: &PlanLimits,
    rejection: Option<&str>,
) -> String {
    let mem = if memory.is_empty() {
        String::new()
    } else {
        format!(
            "\nRELEVANT PAST RUNS (context only, may be stale):\n{}\n",
            memory
                .iter()
                .map(|m| format!("- {m}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };
    let retry = rejection
        .map(|r| format!("\nYOUR PREVIOUS PLAN WAS REJECTED: {r}\nFix exactly that problem and return the corrected JSON.\n"))
        .unwrap_or_default();
    format!(
        "USER REQUEST:\n{request}\n{mem}\nAVAILABLE TOOLS (use kind \"tool\" only for deterministic work such as arithmetic):\n{tools}\n\n\
RULES:\n\
1. At most {max} tasks. Prefer 3-6.\n\
2. Independent tasks must have no dependency on each other so they can run in parallel.\n\
3. Every id in \"dependencies\" must be the id of another task. No cycles.\n\
4. Research, comparison, reasoning and writing are kind \"llm\". Never invent a tool.\n\
5. The last task should produce the final deliverable and depend on what it needs.\n\
6. A tool task may reference an upstream result in a string argument as {{{{task_id}}}}.\n\n\
SCHEMA:\n{{\"goal\": \"...\", \"tasks\": [{{\"id\": \"snake_case_id\", \"description\": \"what to do\", \"dependencies\": [\"other_id\"], \"kind\": \"llm\"}}, \
{{\"id\": \"calc\", \"description\": \"compute\", \"dependencies\": [], \"kind\": \"tool\", \"tool\": \"calculator\", \"args\": {{\"expression\": \"2+2\"}}}}]}}\n{retry}",
        tools = tools.describe(),
        max = limits.max_tasks,
    )
}

pub fn parse_plan(raw: &str) -> Result<Plan, String> {
    let json = extract_json_object(raw).ok_or("response contained no JSON object")?;
    serde_json::from_str::<Plan>(json)
        .map_err(|e| format!("JSON does not match the plan schema: {e}"))
}

/// Request-agnostic fallback: nothing in it depends on the topic.
pub fn fallback_plan(request: &str) -> Plan {
    let goal = truncate(request.trim(), 160);
    Plan {
        goal: goal.clone(),
        tasks: vec![
            Task::llm(
                "understand_request",
                &format!("Restate what is being asked and what a good answer needs: {goal}"),
                &[],
            ),
            Task::llm(
                "analyze",
                "Work through the request step by step and list the key points",
                &["understand_request"],
            ),
            Task::llm(
                "final_answer",
                "Write the final answer for the user from the analysis",
                &["analyze"],
            ),
        ],
    }
}

/// Returns the validated plan and where it came from (`llm` | `fallback`).
pub fn plan(
    llm: &Llm,
    request: &str,
    memory: &[String],
    tools: &Tools,
    limits: &PlanLimits,
    log: &EventLog,
) -> Result<(Plan, &'static str), String> {
    let names = tools.names();
    let mut rejection: Option<String> = None;
    for attempt in 1..=2u32 {
        log.emit(
            EventKind::PlanRequested,
            "planner",
            None,
            format!("planning (attempt {attempt})"),
            json!({"attempt": attempt}),
        );
        let req = LlmRequest {
            role: Role::Planner,
            system: SYSTEM.into(),
            user: prompt(request, memory, tools, limits, rejection.as_deref()),
            ctx: Ctx {
                request: request.into(),
                memory: memory.to_vec(),
                tools: names.clone(),
                ..Default::default()
            },
        };
        let raw = llm
            .call(req, &|a, e, retry| {
                log.emit(
                    EventKind::AttemptFailed,
                    "planner",
                    None,
                    format!("planner call failed (attempt {a}): {}", e.msg),
                    json!({"attempt": a, "transient": e.transient, "will_retry": retry, "error": e.msg}),
                );
            })
            .map_err(|e| e.msg)?;
        let verdict = parse_plan(&raw).and_then(|mut p| {
            dag::validate(&mut p, limits, &names)
                .map(|_| p)
                .map_err(|e| e.0)
        });
        match verdict {
            Ok(p) => return Ok((p, "llm")),
            Err(reason) => {
                log.emit(
                    EventKind::PlanRejected,
                    "planner",
                    None,
                    format!("plan rejected: {reason}"),
                    json!({"attempt": attempt, "reason": reason}),
                );
                rejection = Some(reason);
            }
        }
    }
    let reason = rejection.unwrap_or_default();
    log.emit(
        EventKind::PlanFallback,
        "planner",
        None,
        "using request-agnostic fallback plan",
        json!({"reason": reason}),
    );
    let mut p = fallback_plan(request);
    dag::validate(&mut p, limits, &names).map_err(|e| e.0)?;
    let _ = Kind::Llm;
    Ok((p, "fallback"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{LlmError, Provider};
    use std::sync::{Arc, Mutex};

    type PlanRunResult = (
        Result<(Plan, &'static str), String>,
        Vec<String>,
        Vec<EventKind>,
    );

    struct Scripted(Mutex<Vec<String>>, Mutex<Vec<String>>);
    impl Provider for Scripted {
        fn name(&self) -> String {
            "s".into()
        }
        fn model(&self) -> String {
            "m".into()
        }
        fn complete(&self, r: &LlmRequest) -> Result<String, LlmError> {
            self.1.lock().unwrap().push(r.user.clone());
            Ok(self.0.lock().unwrap().remove(0))
        }
    }
    fn run(script: &[&str]) -> PlanRunResult {
        let prov = Arc::new(Scripted(
            Mutex::new(script.iter().map(|s| s.to_string()).collect()),
            Mutex::new(vec![]),
        ));
        let llm = Llm::new(prov.clone(), 10, 0, 0);
        let log = EventLog::new();
        let r = plan(
            &llm,
            "Plan a trip to Rome",
            &[],
            &Tools::builtin(),
            &PlanLimits::default(),
            &log,
        );
        let prompts = prov.1.lock().unwrap().clone();
        (r, prompts, log.all().iter().map(|e| e.kind).collect())
    }

    #[test]
    fn accepts_valid_plan_inside_prose_and_fences() {
        let good = "Here you go:\n```json\n{\"goal\":\"g\",\"tasks\":[{\"id\":\"A\",\"task\":\"do a\"},{\"id\":\"b\",\"description\":\"do b\",\"deps\":[\"a\"]}]}\n```";
        let (r, _, _) = run(&[good]);
        let (p, src) = r.unwrap();
        assert_eq!(src, "llm");
        assert_eq!(p.tasks[1].dependencies, vec!["a"]);
    }

    #[test]
    fn rejection_reason_is_fed_back_and_second_attempt_can_succeed() {
        let cyclic = r#"{"tasks":[{"id":"a","description":"x","dependencies":["b"]},{"id":"b","description":"y","dependencies":["a"]}]}"#;
        let good = r#"{"tasks":[{"id":"a","description":"x"}]}"#;
        let (r, prompts, kinds) = run(&[cyclic, good]);
        assert_eq!(r.unwrap().1, "llm");
        assert!(prompts[1].contains("PREVIOUS PLAN WAS REJECTED") && prompts[1].contains("cycle"));
        assert!(kinds.contains(&EventKind::PlanRejected));
    }

    #[test]
    fn falls_back_to_a_generic_plan_that_ignores_the_topic() {
        let (r, _, kinds) = run(&["not json", "still not json"]);
        let (p, src) = r.unwrap();
        assert_eq!(src, "fallback");
        assert!(kinds.contains(&EventKind::PlanFallback));
        let text = format!("{:?}", p);
        assert!(
            text.contains("Rome") && !text.contains("Python") && !text.contains("Rust"),
            "fallback must be topic-agnostic: {text}"
        );
    }
}
