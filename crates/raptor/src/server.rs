//! Minimal HTTP server (tiny_http) for the optional JSON API.
//! Binds to localhost by default; there is no auth, so do not expose it publicly.

use crate::config::RunConfig;
use crate::engine::{Engine, RunResult};
use crate::events::EventLog;
use crate::memory::MemoryStore;
use crate::provider::{Chaos, Provider, SimConfig, SimProvider};
use crate::tools::Tools;
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::Read;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tiny_http::{Header, Method, Request, Response, Server};

const MAX_BODY: u64 = 8 * 1024;
const MAX_REQUEST_CHARS: usize = 2000;
const MAX_ACTIVE: usize = 4;
const KEEP_RUNS: usize = 50;

pub struct RunEntry {
    pub id: String,
    pub request: String,
    pub log: Arc<EventLog>,
    pub result: Mutex<Option<RunResult>>,
    pub done: AtomicBool,
}

pub struct AppState {
    pub provider: Arc<dyn Provider>,
    pub cfg: RunConfig,
    pub tools: Arc<Tools>,
    pub memory: Option<Arc<MemoryStore>>,
    pub data_dir: std::path::PathBuf,
    pub default_latency_ms: u64,
    runs: Mutex<Vec<Arc<RunEntry>>>,
    counter: AtomicUsize,
}

impl AppState {
    pub fn new(
        provider: Arc<dyn Provider>,
        cfg: RunConfig,
        memory: Option<Arc<MemoryStore>>,
        data_dir: std::path::PathBuf,
        default_latency_ms: u64,
    ) -> AppState {
        AppState {
            provider,
            cfg,
            tools: Arc::new(Tools::builtin()),
            memory,
            data_dir,
            default_latency_ms,
            runs: Mutex::new(vec![]),
            counter: AtomicUsize::new(1),
        }
    }
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct StartBody {
    request: String,
    /// "auto" (configured provider) or "sim".
    mode: String,
    latency_ms: Option<u64>,
    chaos: Option<Chaos>,
}

fn json_resp(code: u16, v: Value) -> Response<std::io::Cursor<Vec<u8>>> {
    let mut r = Response::from_string(v.to_string()).with_status_code(code);
    for (k, val) in [
        ("Content-Type", "application/json"),
        ("Cache-Control", "no-store"),
        ("X-Content-Type-Options", "nosniff"),
    ] {
        r.add_header(Header::from_bytes(k, val).unwrap());
    }
    r
}

fn query_param(q: &str, key: &str) -> Option<String> {
    q.split('&')
        .find_map(|kv| kv.strip_prefix(&format!("{key}=")).map(|v| v.to_string()))
}

fn find_run(st: &AppState, id: &str) -> Option<Arc<RunEntry>> {
    st.runs.lock().unwrap().iter().find(|r| r.id == id).cloned()
}

fn start_run(st: &Arc<AppState>, body: StartBody) -> Result<String, (u16, String)> {
    let request = body.request.trim().to_string();
    if request.is_empty() {
        return Err((400, "request must not be empty".into()));
    }
    if request.chars().count() > MAX_REQUEST_CHARS {
        return Err((
            400,
            format!("request longer than {MAX_REQUEST_CHARS} characters"),
        ));
    }
    {
        let runs = st.runs.lock().unwrap();
        if runs
            .iter()
            .filter(|r| !r.done.load(Ordering::SeqCst))
            .count()
            >= MAX_ACTIVE
        {
            return Err((429, format!("too many active runs (max {MAX_ACTIVE})")));
        }
    }
    let n = st.counter.fetch_add(1, Ordering::SeqCst);
    let id = format!("r{n}");
    let use_sim = body.mode == "sim" || st.provider.simulated();
    let provider: Arc<dyn Provider> = if use_sim {
        Arc::new(SimProvider::new(SimConfig {
            latency_ms: body.latency_ms.unwrap_or(st.default_latency_ms).min(5000),
            chaos: body.chaos.unwrap_or_default(),
        }))
    } else {
        st.provider.clone()
    };
    let log = Arc::new(
        EventLog::new().with_trace_file(&st.data_dir.join("traces").join(format!("{id}.jsonl"))),
    );
    let entry = Arc::new(RunEntry {
        id: id.clone(),
        request: request.clone(),
        log: log.clone(),
        result: Mutex::new(None),
        done: AtomicBool::new(false),
    });
    {
        let mut runs = st.runs.lock().unwrap();
        runs.push(entry.clone());
        while runs.len() > KEEP_RUNS {
            runs.remove(0);
        }
    }
    let engine = Engine {
        provider,
        tools: st.tools.clone(),
        memory: st.memory.clone(),
        cfg: st.cfg.clone(),
    };
    std::thread::spawn(move || {
        let res = engine.run(&request, &log, None);
        *entry.result.lock().unwrap() = Some(res);
        entry.done.store(true, Ordering::SeqCst);
    });
    Ok(id)
}

fn handle(st: &Arc<AppState>, mut req: Request) {
    let url = req.url().to_string();
    let (path, query) = url.split_once('?').unwrap_or((url.as_str(), ""));
    let method = req.method().clone();
    let segs: Vec<&str> = path.trim_matches('/').split('/').collect();
    let resp = match (&method, segs.as_slice()) {
        (Method::Get, [""]) => Response::from_string(
            "rapTOR API is running. Use the terminal CLI; health endpoint: /api/health\n",
        )
        .with_header(Header::from_bytes("Content-Type", "text/plain; charset=utf-8").unwrap()),
        (Method::Get, ["api", "health"]) => json_resp(
            200,
            json!({"ok": true, "version": crate::VERSION, "provider": st.provider.name(), "model": st.provider.model(), "simulated": st.provider.simulated(),
                   "tools": st.tools.names(), "memory_records": st.memory.as_ref().map(|m| m.len()).unwrap_or(0),
                   "limits": {"max_workers": st.cfg.max_workers, "max_tasks": st.cfg.max_tasks, "max_llm_calls": st.cfg.max_llm_calls, "retries": st.cfg.retries, "max_repairs": st.cfg.max_repairs}}),
        ),
        (Method::Post, ["api", "runs"]) => {
            let mut buf = String::new();
            let read = req.as_reader().take(MAX_BODY + 1).read_to_string(&mut buf);
            if read.is_err() || buf.len() as u64 > MAX_BODY {
                json_resp(413, json!({"error": "request body too large (max 8 KiB)"}))
            } else {
                match serde_json::from_str::<StartBody>(&buf) {
                    Err(e) => json_resp(400, json!({"error": format!("invalid JSON body: {e}")})),
                    Ok(b) => match start_run(st, b) {
                        Ok(id) => json_resp(202, json!({"id": id})),
                        Err((c, m)) => json_resp(c, json!({"error": m})),
                    },
                }
            }
        }
        (Method::Get, ["api", "runs"]) => {
            let runs = st.runs.lock().unwrap();
            json_resp(
                200,
                json!({"runs": runs.iter().rev().map(|r| json!({"id": r.id, "request": r.request, "done": r.done.load(Ordering::SeqCst)})).collect::<Vec<_>>()}),
            )
        }
        (Method::Get, ["api", "runs", id]) => match find_run(st, id) {
            None => json_resp(404, json!({"error": "run not found"})),
            Some(r) => json_resp(
                200,
                json!({"id": r.id, "request": r.request, "done": r.done.load(Ordering::SeqCst), "result": *r.result.lock().unwrap()}),
            ),
        },
        (Method::Get, ["api", "runs", id, "events"]) => match find_run(st, id) {
            None => json_resp(404, json!({"error": "run not found"})),
            Some(r) => {
                let after = query_param(query, "after")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(0);
                let done = r.done.load(Ordering::SeqCst);
                json_resp(200, json!({"events": r.log.since(after), "done": done}))
            }
        },
        _ => json_resp(404, json!({"error": "not found"})),
    };
    let _ = req.respond(resp);
}

pub fn serve(state: Arc<AppState>, host: &str, port: u16) -> Result<(), String> {
    let server = Server::http(format!("{host}:{port}"))
        .map_err(|e| format!("cannot bind {host}:{port}: {e}"))?;
    for req in server.incoming_requests() {
        let st = state.clone();
        std::thread::spawn(move || handle(&st, req));
    }
    Ok(())
}

/// Bind on an ephemeral port and serve in the background (used by integration tests).
pub fn serve_background(state: Arc<AppState>) -> u16 {
    let server = Server::http("127.0.0.1:0").expect("bind");
    let port = server.server_addr().to_ip().map(|a| a.port()).unwrap_or(0);
    std::thread::spawn(move || {
        for req in server.incoming_requests() {
            let st = state.clone();
            std::thread::spawn(move || handle(&st, req));
        }
    });
    port
}
