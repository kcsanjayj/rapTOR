//! End-to-end engine tests (simulator, no network) + a mock local Ollama-compatible server.
use raptor::config::RunConfig;
use raptor::engine::{Engine, RunResult};
use raptor::events::{EventKind, EventLog};
use raptor::model::{Kind, NodeStatus, Plan, RunStatus, Task};
use raptor::provider::{Chaos, HttpProvider, Provider, SimConfig, SimProvider};
use raptor::tools::Tools;
use serde_json::json;
use std::sync::Arc;

fn engine(sim: SimConfig, cfg: RunConfig) -> Engine {
    Engine {
        provider: Arc::new(SimProvider::new(sim)),
        tools: Arc::new(Tools::builtin()),
        memory: None,
        cfg,
    }
}
fn fast() -> RunConfig {
    RunConfig {
        backoff_base_ms: 1,
        ..Default::default()
    }
}
fn run(e: &Engine, req: &str, plan: Option<Plan>) -> (RunResult, Arc<EventLog>) {
    let log = Arc::new(EventLog::new());
    (e.run(req, &log, plan), log)
}
fn count(log: &EventLog, k: EventKind) -> usize {
    log.all().iter().filter(|e| e.kind == k).count()
}

#[test]
fn happy_path_runs_parallel_dag_with_data_flow_and_reviews_every_llm_node() {
    let (r, _log) = run(
        &engine(SimConfig::default(), fast()),
        "Compare Rust and Go for a CLI tool",
        None,
    );
    assert_eq!(r.status, RunStatus::Succeeded, "{:?}", r.error);
    assert_eq!(r.nodes.len(), 4);
    // data flow: compare's input came from both research nodes, and its output reflects them
    let compare = r.nodes.iter().find(|n| n.id == "compare").unwrap();
    assert_eq!(compare.inputs_from, vec!["research_rust", "research_go"]);
    let out = compare.output.clone().unwrap();
    assert!(
        out.contains("research_rust") && out.contains("research_go"),
        "{out}"
    );
    // the critic reviewed every LLM node, including the LAST one (v1 skipped it)
    assert!(r.nodes.iter().all(|n| n.reviewed && n.score.unwrap() >= 7));
    assert_eq!(r.metrics.critic_reviews as usize, 4);
}

#[test]
fn dependents_never_start_before_their_dependencies_finish() {
    let (_r, log) = run(
        &engine(
            SimConfig {
                latency_ms: 20,
                ..Default::default()
            },
            fast(),
        ),
        "Compare Rust and Go",
        None,
    );
    let ev = log.all();
    let t = |k: EventKind, n: &str| {
        ev.iter()
            .find(|e| e.kind == k && e.node.as_deref() == Some(n))
            .map(|e| e.seq)
            .unwrap()
    };
    assert!(t(EventKind::NodeCompleted, "research_rust") < t(EventKind::NodeStarted, "compare"));
    assert!(t(EventKind::NodeCompleted, "research_go") < t(EventKind::NodeStarted, "compare"));
    assert!(t(EventKind::NodeCompleted, "compare") < t(EventKind::NodeStarted, "recommend"));
}

#[test]
fn independent_tasks_really_run_in_parallel() {
    let (r, _) = run(
        &engine(
            SimConfig {
                latency_ms: 120,
                ..Default::default()
            },
            fast(),
        ),
        "Compare Rust and Go",
        None,
    );
    assert!(
        r.metrics.parallelism > 1.2,
        "parallelism {}",
        r.metrics.parallelism
    );
}

#[test]
fn transient_failures_are_retried_and_degraded_drafts_are_repaired() {
    let chaos = Chaos {
        transient_rate: 1.0,
        degrade_rate: 1.0,
        seed: 3,
        fail_tasks: vec![],
    };
    let (r, log) = run(
        &engine(
            SimConfig {
                latency_ms: 0,
                chaos,
            },
            fast(),
        ),
        "Compare Rust and Go",
        None,
    );
    assert_eq!(r.status, RunStatus::Succeeded);
    assert!(
        r.metrics.retries >= 4 && r.metrics.repairs >= 4,
        "{:?}",
        r.metrics
    );
    assert!(r
        .nodes
        .iter()
        .all(|n| n.score.unwrap() >= 7 && !n.low_confidence));
    assert!(count(&log, EventKind::RepairRequested) >= 4);
}

#[test]
fn permanent_failure_skips_dependents_and_returns_an_honest_partial() {
    let chaos = Chaos {
        fail_tasks: vec!["research_go".into()],
        ..Default::default()
    };
    let (r, _) = run(
        &engine(
            SimConfig {
                latency_ms: 0,
                chaos,
            },
            fast(),
        ),
        "Compare Rust and Go",
        None,
    );
    assert_eq!(r.status, RunStatus::Partial);
    let st = |id: &str| r.nodes.iter().find(|n| n.id == id).unwrap().status;
    assert_eq!(st("research_rust"), NodeStatus::Completed);
    assert_eq!(st("research_go"), NodeStatus::Failed);
    assert_eq!(st("compare"), NodeStatus::Skipped);
    assert_eq!(st("recommend"), NodeStatus::Skipped);
    assert!(r.answer.contains("Incomplete tasks"), "{}", r.answer);
}

#[test]
fn cyclic_or_dangling_plans_are_rejected_before_any_work() {
    let cyc = Plan {
        goal: "g".into(),
        tasks: vec![Task::llm("a", "x", &["b"]), Task::llm("b", "y", &["a"])],
    };
    let (r, log) = run(&engine(SimConfig::default(), fast()), "x", Some(cyc));
    assert_eq!(r.status, RunStatus::Failed);
    assert!(r.error.unwrap().contains("cycle"));
    assert_eq!(count(&log, EventKind::NodeStarted), 0);
    let dangling = Plan {
        goal: "g".into(),
        tasks: vec![Task::llm("a", "x", &["ghost"])],
    };
    assert!(
        run(&engine(SimConfig::default(), fast()), "x", Some(dangling))
            .0
            .error
            .unwrap()
            .contains("unknown task")
    );
}

#[test]
fn budget_is_a_hard_cap_and_is_reported() {
    let cfg = RunConfig {
        max_llm_calls: 3,
        ..fast()
    };
    let (r, log) = run(
        &engine(SimConfig::default(), cfg),
        "Compare Rust and Go",
        None,
    );
    assert!(r.metrics.llm_calls <= 3, "{:?}", r.metrics);
    assert_ne!(r.status, RunStatus::Succeeded);
    assert!(count(&log, EventKind::BudgetExhausted) >= 1);
}

#[test]
fn calculator_tool_runs_and_dos_expression_fails_fast() {
    let (r, _) = run(
        &engine(SimConfig::default(), fast()),
        "Calculate 12*(3+4)",
        None,
    );
    assert_eq!(r.status, RunStatus::Succeeded);
    assert_eq!(r.nodes[0].output.as_deref(), Some("84"));
    assert!(r.answer.contains("84"));
    let t = std::time::Instant::now();
    let (r, _) = run(
        &engine(SimConfig::default(), fast()),
        "Calculate 9**9**9",
        None,
    );
    assert!(t.elapsed().as_secs() < 2);
    assert_eq!(r.nodes[0].status, NodeStatus::Failed);
    assert!(r.nodes[0].error.as_ref().unwrap().contains("exponent"));
}

#[test]
fn tool_args_can_consume_upstream_output_via_placeholders() {
    let mut calc = Task::llm("calc", "double it", &["base"]);
    calc.kind = Kind::Tool;
    calc.tool = Some("calculator".into());
    calc.args = json!({"expression": "{{base}} * 2"});
    let mut base = Task::llm("base", "compute 20+1", &[]);
    base.kind = Kind::Tool;
    base.tool = Some("calculator".into());
    base.args = json!({"expression": "20+1"});
    let (r, _) = run(
        &engine(SimConfig::default(), fast()),
        "x",
        Some(Plan {
            goal: "g".into(),
            tasks: vec![base, calc],
        }),
    );
    assert_eq!(r.nodes[1].output.as_deref(), Some("42"));
}

#[test]
fn unrelated_requests_never_leak_demo_topics() {
    let (r, _) = run(
        &engine(SimConfig::default(), fast()),
        "Plan a 3-day trip to Rome",
        None,
    );
    assert_eq!(r.status, RunStatus::Succeeded);
    let all = format!("{} {:?}", r.answer, r.plan);
    assert!(!all.contains("Python") && !all.contains("Rust"), "{all}");
}

#[test]
fn simulator_is_deterministic_for_a_given_seed() {
    let chaos = || Chaos {
        transient_rate: 0.5,
        degrade_rate: 0.5,
        seed: 42,
        fail_tasks: vec![],
    };
    let a = run(
        &engine(
            SimConfig {
                latency_ms: 0,
                chaos: chaos(),
            },
            fast(),
        ),
        "Compare Rust and Go",
        None,
    )
    .0;
    let b = run(
        &engine(
            SimConfig {
                latency_ms: 0,
                chaos: chaos(),
            },
            fast(),
        ),
        "Compare Rust and Go",
        None,
    )
    .0;
    assert_eq!(a.answer, b.answer);
    assert_eq!(a.metrics.repairs, b.metrics.repairs);
}

// ------------------------------------------------ mock local Ollama-compatible server

fn mock_ollama() -> u16 {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = server.server_addr().to_ip().unwrap().port();
    std::thread::spawn(move || {
        for mut req in server.incoming_requests() {
            let url = req.url().to_string();
            let (code, body) = if url.ends_with("/models") {
                (
                    200,
                    json!({"data": [{"id": "text-embedding-9"}, {"id": "big-model-xl"}, {"id": "tiny-mini-7"}]}),
                )
            } else {
                let mut b = String::new();
                std::io::Read::read_to_string(req.as_reader(), &mut b).unwrap();
                let v: serde_json::Value = serde_json::from_str(&b).unwrap();
                (
                    200,
                    json!({"choices": [{"message": {"content": format!("model={}", v["model"].as_str().unwrap())}}]}),
                )
            };
            let _ = req
                .respond(tiny_http::Response::from_string(body.to_string()).with_status_code(code));
        }
    });
    port
}

#[test]
fn http_provider_discovers_the_model_instead_of_hardcoding_one() {
    let port = mock_ollama();
    let p = HttpProvider::openai_compatible("mock", format!("http://127.0.0.1:{port}/v1"), None);
    p.warm_up().unwrap();
    assert_eq!(p.model(), "tiny-mini-7"); // skipped the embedding model, preferred the small chat model
    let req = raptor::provider::LlmRequest {
        role: raptor::provider::Role::Executor,
        system: "s".into(),
        user: "u".into(),
        ctx: Default::default(),
    };
    assert_eq!(p.complete(&req).unwrap(), "model=tiny-mini-7");
}

#[test]
fn http_provider_honors_model_override_and_rejects_remote_endpoints() {
    let port = mock_ollama();
    let pinned = HttpProvider::openai_compatible(
        "mock",
        format!("http://127.0.0.1:{port}/v1"),
        Some("my-model".into()),
    );
    assert_eq!(pinned.model(), "my-model");

    let remote = HttpProvider::openai_compatible("remote", "https://example.com/v1".into(), None);
    let error = remote.list_models().unwrap_err();
    assert!(!error.transient);
    assert!(error.msg.contains("offline mode"));
}

#[test]
fn full_run_through_local_model_provider_uses_the_real_code_path() {
    // The mock returns non-plan text, so the planner must reject twice and use the generic fallback,
    // proving the local-model path (discovery -> chat -> validation -> fallback) works end to end.
    let port = mock_ollama();
    let p: Arc<dyn Provider> = Arc::new(HttpProvider::openai_compatible(
        "mock",
        format!("http://127.0.0.1:{port}/v1"),
        None,
    ));
    let e = Engine {
        provider: p,
        tools: Arc::new(Tools::builtin()),
        memory: None,
        cfg: fast(),
    };
    let (r, log) = run(&e, "Explain recursion", None);
    assert_eq!(count(&log, EventKind::PlanFallback), 1);
    assert!(r.plan.is_some() && !r.simulated && r.model == "tiny-mini-7");
}

#[test]
fn ollama_refused_connections_explain_what_to_do() {
    let dead = raptor::provider::build_named(
        "ollama",
        &|k| {
            if k == "OLLAMA_HOST" {
                Some("127.0.0.1:9".into())
            } else {
                None
            }
        },
        SimConfig::default(),
    )
    .unwrap();
    let err = dead.warm_up().unwrap_err();
    assert!(
        err.contains("cannot reach") && err.contains("ollama serve"),
        "{err}"
    );
}
