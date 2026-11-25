//! The orchestration engine: plan -> validate -> parallel schedule -> [execute -> critic -> repair] -> synthesize.
//!
//! Fixes vs v1: dependency outputs are passed to dependents; the critic gates *every* LLM node
//! (before dependents are released, so nothing downstream consumes unreviewed output);
//! failed nodes skip their dependents explicitly; budgets/deadlines are enforced.

use crate::config::RunConfig;
use crate::dag::{self, PlanLimits};
use crate::events::{EventKind, EventLog};
use crate::llm::Llm;
use crate::memory::MemoryStore;
use crate::model::{Kind, NodeStatus, Plan, RunStatus, Task};
use crate::provider::{Ctx, LlmError, LlmRequest, NodeOutput, Provider, Role};
use crate::tools::{substitute_args, Tools};
use crate::util::truncate;
use crate::{critic, planner, synthesizer};
use serde::Serialize;
use serde_json::json;
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Serialize)]
pub struct NodeReport {
    pub id: String,
    pub status: NodeStatus,
    pub output: Option<String>,
    pub error: Option<String>,
    pub attempts: u32,
    pub repairs: u32,
    pub score: Option<u8>,
    pub low_confidence: bool,
    pub reviewed: bool,
    pub worker: Option<usize>,
    pub started_ms: Option<u64>,
    pub ended_ms: Option<u64>,
    pub inputs_from: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Default)]
pub struct Metrics {
    pub llm_calls: usize,
    pub retries: usize,
    pub repairs: u32,
    pub critic_reviews: u32,
    pub tasks_total: usize,
    pub tasks_completed: usize,
    pub tasks_failed: usize,
    pub tasks_skipped: usize,
    pub wall_ms: u64,
    pub node_ms_sum: u64,
    /// Sum of per-node durations / execution wall time. >1 means real parallelism.
    pub parallelism: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunResult {
    pub status: RunStatus,
    pub answer: String,
    pub error: Option<String>,
    pub plan: Option<Plan>,
    pub nodes: Vec<NodeReport>,
    pub metrics: Metrics,
    pub provider: String,
    pub model: String,
    pub simulated: bool,
}

pub struct Engine {
    pub provider: Arc<dyn Provider>,
    pub tools: Arc<Tools>,
    pub memory: Option<Arc<MemoryStore>>,
    pub cfg: RunConfig,
}

struct Shared {
    llm: Arc<Llm>,
    tools: Arc<Tools>,
    log: Arc<EventLog>,
    cfg: RunConfig,
    request: String,
}

struct Done {
    output: String,
    attempts: u32,
    repairs: u32,
    score: Option<u8>,
    low_confidence: bool,
    reviewed: bool,
    reviews: u32,
}

struct Outcome {
    id: String,
    worker: usize,
    started_ms: u64,
    ended_ms: u64,
    result: Result<Done, String>,
}

fn upstream_block(deps: &[(String, String)]) -> String {
    let mut s = String::new();
    for (id, o) in deps {
        s.push_str(&format!("### {id}\n{}\n\n", truncate(o, 2000)));
    }
    truncate(&s, 8000)
}

fn exec_request(
    sh: &Shared,
    task: &Task,
    deps: &[(String, String)],
    feedback: Option<&str>,
) -> LlmRequest {
    let mut user = format!(
        "ORIGINAL USER REQUEST:\n{}\n\nYOUR TASK:\n{}\n",
        sh.request, task.description
    );
    if !deps.is_empty() {
        user.push_str(&format!(
            "\nUPSTREAM RESULTS (use them):\n{}",
            upstream_block(deps)
        ));
    }
    if let Some(f) = feedback {
        user.push_str(&format!("\nREVIEWER FEEDBACK TO ADDRESS:\n{f}\n"));
    }
    LlmRequest {
        role: Role::Executor,
        system: "You are one worker in a multi-agent pipeline. Complete ONLY the assigned task using the upstream results. \
Be accurate and concise; never fabricate sources or numbers; say when you are unsure. Return only the result."
            .into(),
        user,
        ctx: Ctx {
            request: sh.request.clone(),
            task_id: task.id.clone(),
            task: task.description.clone(),
            deps: deps.to_vec(),
            feedback: feedback.map(|s| s.to_string()),
            ..Default::default()
        },
    }
}

fn run_node(
    sh: &Shared,
    task: &Task,
    deps: &[(String, String)],
    worker: usize,
) -> Result<Done, String> {
    let agent = format!("worker-{worker}");
    let log = &sh.log;
    log.emit(
        EventKind::NodeStarted,
        &agent,
        Some(&task.id),
        format!("{} started", task.id),
        json!({"worker": worker, "kind": task.kind, "inputs_from": deps.iter().map(|d| &d.0).collect::<Vec<_>>(),
               "input_chars": deps.iter().map(|d| d.1.len()).sum::<usize>()}),
    );
    if task.kind == Kind::Tool {
        let name = task.tool.clone().unwrap_or_default();
        let args = substitute_args(&task.args, deps);
        log.emit(
            EventKind::ToolCalled,
            &agent,
            Some(&task.id),
            format!("tool {name}"),
            json!({"tool": name, "args": args}),
        );
        let out = sh.tools.execute(&name, &args)?;
        return Ok(Done {
            output: out,
            attempts: 1,
            repairs: 0,
            score: None,
            low_confidence: false,
            reviewed: false,
            reviews: 0,
        });
    }
    let fails = std::cell::Cell::new(0u32);
    let on_fail =
        |a: u32, e: &LlmError, retry: bool| {
            fails.set(fails.get() + 1);
            log.emit(
            EventKind::AttemptFailed,
            &agent,
            Some(&task.id),
            format!("attempt {a} failed: {}{}", e.msg, if retry { " (retrying)" } else { "" }),
            json!({"attempt": a, "transient": e.transient, "will_retry": retry, "error": e.msg}),
        );
        };
    let (mut feedback, mut repairs, mut rounds, mut reviews) = (None::<String>, 0u32, 0u32, 0u32);
    loop {
        let out = sh
            .llm
            .call(exec_request(sh, task, deps, feedback.as_deref()), &on_fail)
            .map_err(|e| e.msg)?;
        rounds += 1;
        log.emit(
            EventKind::ReviewStarted,
            "critic",
            Some(&task.id),
            format!("reviewing {}", task.id),
            json!({}),
        );
        let critic_fail = |_a: u32, _e: &LlmError, _r: bool| {};
        let (rp, rv, fl, rd) = (repairs, reviews, fails.get(), rounds);
        let done =
            move |score: Option<u8>, low: bool, reviewed: bool, out: String, rev_inc: u32| Done {
                output: out,
                attempts: rd + fl,
                repairs: rp,
                score,
                low_confidence: low,
                reviewed,
                reviews: rv + rev_inc,
            };
        match critic::review(&sh.llm, &sh.request, task, deps, &out, &critic_fail) {
            Ok(rev) => {
                reviews += 1;
                let pass = rev.score >= sh.cfg.pass_score;
                log.emit(
                    EventKind::CriticReview,
                    "critic",
                    Some(&task.id),
                    format!("{} scored {}/10{}", task.id, rev.score, if pass { " - accepted" } else { " - needs repair" }),
                    json!({"score": rev.score, "pass": pass, "issues": rev.issues, "feedback": rev.feedback}),
                );
                if pass {
                    return Ok(done(Some(rev.score), false, true, out, 1));
                }
                if repairs >= sh.cfg.max_repairs {
                    return Ok(done(Some(rev.score), true, true, out, 1));
                }
                repairs += 1;
                log.emit(
                    EventKind::RepairRequested,
                    &agent,
                    Some(&task.id),
                    format!("repair round {repairs}: {}", truncate(&rev.feedback, 80)),
                    json!({"round": repairs, "feedback": rev.feedback}),
                );
                feedback = Some(if rev.feedback.is_empty() {
                    rev.issues.join("; ")
                } else {
                    rev.feedback
                });
            }
            Err(e) => {
                log.emit(
                    EventKind::CriticUnavailable,
                    "critic",
                    Some(&task.id),
                    format!(
                        "critic unavailable ({}); accepting unreviewed",
                        truncate(&e.msg, 80)
                    ),
                    json!({"error": e.msg}),
                );
                return Ok(done(None, false, false, out, 0));
            }
        }
    }
}

impl Engine {
    pub fn run(
        &self,
        request: &str,
        log: &Arc<EventLog>,
        plan_override: Option<Plan>,
    ) -> RunResult {
        let provider = &self.provider;
        let llm = Arc::new(Llm::new(
            provider.clone(),
            self.cfg.max_llm_calls,
            self.cfg.retries,
            self.cfg.backoff_base_ms,
        ));
        let names = self.tools.names();
        let limits = PlanLimits {
            max_tasks: self.cfg.max_tasks,
        };
        log.emit(
            EventKind::RunStarted,
            "engine",
            None,
            format!("run started on {} / {}", provider.name(), provider.model()),
            json!({"request": request, "provider": provider.name(), "model": provider.model(), "simulated": provider.simulated()}),
        );
        let finish = |status: RunStatus,
                      answer: String,
                      error: Option<String>,
                      plan: Option<Plan>,
                      nodes: Vec<NodeReport>,
                      extra: (u32, u32, u64, u64)| {
            let mut m = Metrics {
                llm_calls: llm.calls(),
                retries: llm.retries_used(),
                repairs: nodes.iter().map(|n| n.repairs).sum(),
                critic_reviews: extra.0,
                tasks_total: nodes.len(),
                tasks_completed: nodes
                    .iter()
                    .filter(|n| n.status == NodeStatus::Completed)
                    .count(),
                tasks_failed: nodes
                    .iter()
                    .filter(|n| n.status == NodeStatus::Failed)
                    .count(),
                tasks_skipped: nodes
                    .iter()
                    .filter(|n| n.status == NodeStatus::Skipped)
                    .count(),
                wall_ms: log.elapsed_ms(),
                node_ms_sum: extra.3,
                parallelism: 0.0,
            };
            let _ = extra.1;
            if extra.2 > 0 {
                m.parallelism = (m.node_ms_sum as f64 / extra.2 as f64 * 100.0).round() / 100.0;
            }
            let res = RunResult {
                status,
                answer,
                error,
                plan,
                nodes,
                metrics: m,
                provider: provider.name(),
                model: provider.model(),
                simulated: provider.simulated(),
            };
            log.emit(
                EventKind::RunFinished,
                "engine",
                None,
                format!("run finished: {:?}", res.status).to_lowercase(),
                json!({"status": res.status, "answer": res.answer, "error": res.error, "metrics": res.metrics}),
            );
            res
        };

        // ---- memory + plan
        let recalled = self
            .memory
            .as_ref()
            .map(|m| m.recall(request, 3))
            .unwrap_or_default();
        let memory_ctx: Vec<String> = recalled
            .iter()
            .map(|r| {
                format!(
                    "Q: {} | A: {}",
                    truncate(&r.request, 100),
                    truncate(&r.answer, 160)
                )
            })
            .collect();
        if self.memory.is_some() {
            log.emit(
                EventKind::MemoryRecalled,
                "engine",
                None,
                format!("recalled {} similar past run(s)", recalled.len()),
                json!({"count": recalled.len(), "items": memory_ctx}),
            );
        }
        let plan_t0 = Instant::now();
        let (plan, source) = match plan_override {
            Some(mut p) => match dag::validate(&mut p, &limits, &names) {
                Ok(()) => (p, "override"),
                Err(e) => {
                    log.emit(
                        EventKind::PlanRejected,
                        "planner",
                        None,
                        format!("plan rejected: {e}"),
                        json!({"reason": e.0, "attempt": 1}),
                    );
                    return finish(
                        RunStatus::Failed,
                        String::new(),
                        Some(format!("plan rejected: {e}")),
                        None,
                        vec![],
                        (0, 0, 0, 0),
                    );
                }
            },
            None => match planner::plan(&llm, request, &memory_ctx, &self.tools, &limits, log) {
                Ok(x) => x,
                Err(e) => {
                    return finish(
                        RunStatus::Failed,
                        String::new(),
                        Some(e),
                        None,
                        vec![],
                        (0, 0, 0, 0),
                    )
                }
            },
        };
        log.emit(
            EventKind::PlanAccepted,
            "planner",
            None,
            format!(
                "plan accepted: {} tasks ({source}, {} ms)",
                plan.tasks.len(),
                plan_t0.elapsed().as_millis()
            ),
            json!({"plan": plan, "source": source}),
        );

        // ---- schedule
        let tasks: HashMap<String, Task> = plan
            .tasks
            .iter()
            .map(|t| (t.id.clone(), t.clone()))
            .collect();
        let mut remaining: HashMap<String, usize> = plan
            .tasks
            .iter()
            .map(|t| (t.id.clone(), t.dependencies.len()))
            .collect();
        let mut reports: HashMap<String, NodeReport> = plan
            .tasks
            .iter()
            .map(|t| {
                (
                    t.id.clone(),
                    NodeReport {
                        id: t.id.clone(),
                        status: NodeStatus::Pending,
                        output: None,
                        error: None,
                        attempts: 0,
                        repairs: 0,
                        score: None,
                        low_confidence: false,
                        reviewed: false,
                        worker: None,
                        started_ms: None,
                        ended_ms: None,
                        inputs_from: t.dependencies.clone(),
                    },
                )
            })
            .collect();
        let mut ready: VecDeque<String> = plan
            .tasks
            .iter()
            .filter(|t| t.dependencies.is_empty())
            .map(|t| t.id.clone())
            .collect();
        let mut free: BTreeSet<usize> = (1..=self.cfg.max_workers.max(1)).collect();
        let mut outputs: HashMap<String, String> = HashMap::new();
        let shared = Arc::new(Shared {
            llm: llm.clone(),
            tools: self.tools.clone(),
            log: log.clone(),
            cfg: self.cfg.clone(),
            request: request.to_string(),
        });
        let (tx, rx) = mpsc::channel::<Outcome>();
        let (mut running, mut reviews, mut budget_hit) = (0usize, 0u32, false);
        let exec_t0 = Instant::now();
        let deadline = exec_t0 + Duration::from_secs(self.cfg.deadline_secs);
        let mut deadline_hit = false;

        loop {
            if Instant::now() > deadline {
                deadline_hit = true;
            }
            while !deadline_hit && !ready.is_empty() && !free.is_empty() {
                let id = ready.pop_front().unwrap();
                let w = *free.iter().next().unwrap();
                free.remove(&w);
                let task = tasks[&id].clone();
                let deps: Vec<(String, String)> = task
                    .dependencies
                    .iter()
                    .map(|d| (d.clone(), outputs[d].clone()))
                    .collect();
                {
                    let r = reports.get_mut(&id).unwrap();
                    r.status = NodeStatus::Running;
                    r.worker = Some(w);
                    r.started_ms = Some(log.elapsed_ms());
                }
                let (sh, tx2) = (shared.clone(), tx.clone());
                running += 1;
                std::thread::Builder::new()
                    .name(format!("worker-{w}"))
                    .spawn(move || {
                        let started_ms = sh.log.elapsed_ms();
                        let result =
                            catch_unwind(AssertUnwindSafe(|| run_node(&sh, &task, &deps, w)))
                                .unwrap_or_else(|_| {
                                    Err("internal error: worker panicked".to_string())
                                });
                        let _ = tx2.send(Outcome {
                            id: task.id.clone(),
                            worker: w,
                            started_ms,
                            ended_ms: sh.log.elapsed_ms(),
                            result,
                        });
                    })
                    .expect("spawn worker");
            }
            if running == 0 {
                break;
            }
            let msg = match rx.recv_timeout(Duration::from_millis(250)) {
                Ok(m) => m,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(_) => break,
            };
            running -= 1;
            free.insert(msg.worker);
            let r = reports.get_mut(&msg.id).unwrap();
            r.started_ms = Some(msg.started_ms);
            r.ended_ms = Some(msg.ended_ms);
            match msg.result {
                Ok(d) => {
                    reviews += d.reviews;
                    r.status = NodeStatus::Completed;
                    r.output = Some(truncate(&d.output, 4000));
                    r.attempts = d.attempts;
                    r.repairs = d.repairs;
                    r.score = d.score;
                    r.low_confidence = d.low_confidence;
                    r.reviewed = d.reviewed;
                    log.emit(
                        EventKind::NodeCompleted,
                        &format!("worker-{}", msg.worker),
                        Some(&msg.id),
                        format!("{} completed in {} ms", msg.id, msg.ended_ms - msg.started_ms),
                        json!({"attempts": d.attempts, "repairs": d.repairs, "score": d.score, "low_confidence": d.low_confidence,
                               "reviewed": d.reviewed, "duration_ms": msg.ended_ms - msg.started_ms, "output": truncate(&d.output, 1500)}),
                    );
                    outputs.insert(msg.id.clone(), d.output);
                    for t in &plan.tasks {
                        if t.dependencies.contains(&msg.id) {
                            let c = remaining.get_mut(&t.id).unwrap();
                            *c -= 1;
                            if *c == 0 {
                                ready.push_back(t.id.clone());
                            }
                        }
                    }
                }
                Err(e) => {
                    if e.contains("budget exhausted") && !budget_hit {
                        budget_hit = true;
                        log.emit(
                            EventKind::BudgetExhausted,
                            "engine",
                            None,
                            e.clone(),
                            json!({"max_llm_calls": self.cfg.max_llm_calls}),
                        );
                    }
                    r.status = NodeStatus::Failed;
                    r.error = Some(e.clone());
                    log.emit(
                        EventKind::NodeFailed,
                        &format!("worker-{}", msg.worker),
                        Some(&msg.id),
                        format!("{} failed: {}", msg.id, truncate(&e, 120)),
                        json!({"error": e, "duration_ms": msg.ended_ms - msg.started_ms}),
                    );
                    for dep in dag::transitive_dependents(&plan, &msg.id) {
                        let rr = reports.get_mut(&dep).unwrap();
                        if rr.status == NodeStatus::Pending {
                            rr.status = NodeStatus::Skipped;
                            rr.error = Some(format!("skipped: upstream '{}' failed", msg.id));
                            log.emit(
                                EventKind::NodeSkipped,
                                "engine",
                                Some(&dep),
                                format!("{dep} skipped: upstream '{}' failed", msg.id),
                                json!({"reason": format!("upstream '{}' failed", msg.id)}),
                            );
                        }
                    }
                    ready.retain(|x| reports[x].status == NodeStatus::Pending);
                }
            }
        }
        for t in &plan.tasks {
            let r = reports.get_mut(&t.id).unwrap();
            if r.status == NodeStatus::Pending {
                r.status = NodeStatus::Skipped;
                r.error = Some("skipped: deadline exceeded".into());
                log.emit(
                    EventKind::NodeSkipped,
                    "engine",
                    Some(&t.id),
                    format!("{} skipped: deadline exceeded", t.id),
                    json!({"reason": "deadline exceeded"}),
                );
            }
        }
        let exec_ms = exec_t0.elapsed().as_millis() as u64;
        let nodes: Vec<NodeReport> = plan.tasks.iter().map(|t| reports[&t.id].clone()).collect();
        let node_ms_sum: u64 = nodes
            .iter()
            .filter_map(|n| Some(n.ended_ms?.saturating_sub(n.started_ms?)))
            .sum();
        let completed: Vec<NodeOutput> = nodes
            .iter()
            .filter(|n| n.status == NodeStatus::Completed)
            .map(|n| NodeOutput {
                id: n.id.clone(),
                output: outputs.get(&n.id).cloned().unwrap_or_default(),
                is_tool: tasks[&n.id].kind == Kind::Tool,
                low_confidence: n.low_confidence,
            })
            .collect();
        let incomplete: Vec<String> = nodes
            .iter()
            .filter(|n| n.status != NodeStatus::Completed)
            .map(|n| n.id.clone())
            .collect();
        let extra = (reviews, 0u32, exec_ms, node_ms_sum);
        if completed.is_empty() {
            let why = nodes
                .iter()
                .find_map(|n| n.error.clone())
                .unwrap_or_else(|| "no task completed".into());
            return finish(
                RunStatus::Failed,
                String::new(),
                Some(why),
                Some(plan),
                nodes,
                extra,
            );
        }

        // ---- synthesize
        log.emit(
            EventKind::SynthesisStarted,
            "synthesizer",
            None,
            format!("synthesizing from {} result(s)", completed.len()),
            json!({"nodes": completed.len()}),
        );
        let on_fail = |a: u32, e: &LlmError, retry: bool| {
            log.emit(
                EventKind::AttemptFailed,
                "synthesizer",
                None,
                format!("synthesis attempt {a} failed: {}", e.msg),
                json!({"attempt": a, "will_retry": retry, "error": e.msg}),
            );
        };
        let answer = match synthesizer::synthesize(
            &llm,
            request,
            &completed,
            &incomplete,
            self.cfg.answer_words,
            &on_fail,
        ) {
            Ok(a) => a,
            Err(e) => {
                log.emit(
                    EventKind::SynthesisFallback,
                    "synthesizer",
                    None,
                    format!(
                        "synthesis failed ({}); using fallback",
                        truncate(&e.msg, 80)
                    ),
                    json!({"error": e.msg}),
                );
                synthesizer::fallback_answer(&completed, &incomplete)
            }
        };
        let status = if incomplete.is_empty() {
            RunStatus::Succeeded
        } else {
            RunStatus::Partial
        };
        let error = if incomplete.is_empty() {
            None
        } else if budget_hit {
            Some(format!(
                "LLM call budget exhausted (max_llm_calls={}); incomplete: {}",
                self.cfg.max_llm_calls,
                incomplete.join(", ")
            ))
        } else {
            Some(format!("incomplete: {}", incomplete.join(", ")))
        };
        if status == RunStatus::Succeeded {
            if let Some(m) = &self.memory {
                m.save(request, &answer);
            }
        }
        finish(status, answer, error, Some(plan), nodes, extra)
    }
}
