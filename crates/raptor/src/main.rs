use raptor::config::{data_dir_from_env, load_dotenv, save_dotenv_values, RunConfig};
use raptor::engine::{Engine, RunResult};
use raptor::events::{Event, EventLog};
use raptor::memory::MemoryStore;
use raptor::model::{Plan, RunStatus};
use raptor::provider::{self, Chaos, SimConfig};
use raptor::server::{self, AppState};
use raptor::tools::Tools;
use raptor::util::truncate;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::{io, io::Write};

const HELP: &str = "raptor - Rust-core multi-agent orchestrator

USAGE:
  raptor serve [--host H] [--port N] [--latency-ms N] [--data-dir D] (optional local JSON API)
  raptor run \"<request>\" [options]
  raptor chat [--sim]   start a terminal chat; /clear resets the current session
  raptor setup
  raptor providers
  raptor doctor            diagnose provider / local-model connectivity

RUN OPTIONS:
  --json                   print one JSON document (result + events) instead of a live timeline
  --sim                    force the deterministic offline simulator
  --latency-ms N           simulated model latency per call (default 250 for run, 0 with --json)
  --chaos-transient R      probability (0-1) of injected transient failures (simulator)
  --chaos-degrade R        probability (0-1) of deliberately poor first drafts (simulator)
  --chaos-fail a,b         task ids that always fail (simulator)
  --seed N                 seed for fault injection (default 1)
  --plan-file F            skip the planner and run this plan JSON (still validated)
  --max-llm-calls N        hard budget for provider calls
  --workers N              parallel workers (default 4)
  --data-dir D             where memory and traces are stored (default ./data)
  --no-memory              disable memory recall/save

Offline providers: local Ollama (auto-detected or selected in setup) and the built-in deterministic simulator.
Run `raptor setup` to choose one and select an installed model or explicitly download one.
";

struct Opts {
    positional: Vec<String>,
    flags: std::collections::HashMap<String, String>,
}

fn parse(args: &[String]) -> Opts {
    let (mut positional, mut flags) = (vec![], std::collections::HashMap::new());
    let bool_flags = ["json", "sim", "no-memory", "help", "version"];
    let mut i = 0;
    while i < args.len() {
        if let Some(name) = args[i].strip_prefix("--") {
            if bool_flags.contains(&name) {
                flags.insert(name.to_string(), "true".into());
            } else if let Some(v) = args.get(i + 1) {
                flags.insert(name.to_string(), v.clone());
                i += 1;
            }
        } else {
            positional.push(args[i].clone());
        }
        i += 1;
    }
    Opts { positional, flags }
}

fn num<T: std::str::FromStr>(o: &Opts, k: &str) -> Option<T> {
    o.flags.get(k).and_then(|v| v.parse().ok())
}

fn fail(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1);
}

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok()
}

fn prompt_choice(max: usize) -> Result<usize, String> {
    loop {
        print!("Select [1-{max}]: ");
        io::stdout()
            .flush()
            .map_err(|e| format!("cannot write to terminal: {e}"))?;
        let mut input = String::new();
        let count = io::stdin()
            .read_line(&mut input)
            .map_err(|e| format!("cannot read from terminal: {e}"))?;
        if count == 0 {
            return Err("end of input while waiting for selection".into());
        }
        match input.trim().parse::<usize>() {
            Ok(choice) if (1..=max).contains(&choice) => return Ok(choice - 1),
            _ => println!("Enter a number from 1 to {max}."),
        }
    }
}

fn prompt_text(prompt: &str) -> Result<String, String> {
    loop {
        print!("{prompt}");
        io::stdout()
            .flush()
            .map_err(|e| format!("cannot write to terminal: {e}"))?;
        let mut input = String::new();
        let count = io::stdin()
            .read_line(&mut input)
            .map_err(|e| format!("cannot read from terminal: {e}"))?;
        if count == 0 {
            return Err("end of input while waiting for model name".into());
        }
        let model = input.trim();
        if !model.is_empty() {
            return Ok(model.to_string());
        }
        println!("Enter an Ollama model name, for example llama3.2:3b.");
    }
}

fn download_ollama_model(model: &str) -> Result<(), String> {
    println!("Downloading '{model}' with Ollama. This explicit action may use the internet.");
    let status = ollama_pull_command(model).status().map_err(|e| {
        format!("cannot run `ollama pull`: {e}. Install Ollama and ensure it is on PATH.")
    })?;
    if !status.success() {
        return Err(format!(
            "`ollama pull {model}` failed with exit status {status}"
        ));
    }
    Ok(())
}

fn ollama_pull_command(model: &str) -> Command {
    let mut command = Command::new("ollama");
    command.arg("pull").arg(model);
    command
}

fn select_ollama_model(
    models: &[String],
    selected_index: usize,
    prompt_model: impl FnOnce() -> Result<String, String>,
    install_model: impl FnOnce(&str) -> Result<(), String>,
) -> Result<String, String> {
    if let Some(model) = models.get(selected_index) {
        return Ok(model.clone());
    }
    if selected_index != models.len() {
        return Err("selected model option is out of range".into());
    }

    let model = prompt_model()?;
    if model.trim().is_empty() {
        return Err("model name cannot be empty".into());
    }
    install_model(&model)?;
    Ok(model)
}

fn cmd_setup() -> i32 {
    let options = [
        (
            "sim",
            "Deterministic simulator (no network, placeholder responses)",
        ),
        ("ollama", "Ollama (local model, runs on this computer)"),
    ];
    println!("rapTOR offline setup\n\nChoose a local provider:");
    println!("\nAvailable providers:");
    for (index, (_, label)) in options.iter().enumerate() {
        println!("  {}. {label}", index + 1);
    }
    let provider_index = prompt_choice(options.len()).unwrap_or_else(|e| fail(&e));
    let (name, _) = options[provider_index];

    if name == "sim" {
        save_dotenv_values(
            std::path::Path::new(".env"),
            &[("RAPTOR_PROVIDER", name), ("RAPTOR_MODEL", "")],
        )
        .unwrap_or_else(|e| fail(&e));
        println!("\nSaved simulator mode to .env. Its responses are deterministic placeholders.");
        println!("Run `raptor run \"your request\"` to start.");
        return 0;
    }

    std::env::set_var("RAPTOR_PROVIDER", name);
    std::env::set_var("RAPTOR_MODEL", "");
    let model_provider =
        provider::build_named(name, &env, SimConfig::default()).unwrap_or_else(|e| fail(&e));
    let models = model_provider.available_models().unwrap_or_else(|e| {
        fail(&format!(
            "cannot list local Ollama models: {e}. Start Ollama with `ollama serve` and try again."
        ))
    });

    println!("\nInstalled local models:");
    for (index, model) in models.iter().enumerate() {
        println!("  {}. {model}", index + 1);
    }
    if models.is_empty() {
        println!("  (none found)");
    }
    println!(
        "  {}. Download another model (explicit internet download)",
        models.len() + 1
    );
    let model_index = prompt_choice(models.len() + 1).unwrap_or_else(|e| fail(&e));
    let model = select_ollama_model(
        &models,
        model_index,
        || prompt_text("Ollama model name/tag (for example llama3.2:3b): "),
        download_ollama_model,
    )
    .unwrap_or_else(|e| fail(&e));

    std::env::set_var("RAPTOR_MODEL", &model);
    let selected =
        provider::build_named(name, &env, SimConfig::default()).unwrap_or_else(|e| fail(&e));
    selected
        .warm_up()
        .unwrap_or_else(|e| fail(&format!("selected provider could not be initialized: {e}")));

    let values = [("RAPTOR_PROVIDER", name), ("RAPTOR_MODEL", model.as_str())];
    save_dotenv_values(std::path::Path::new(".env"), &values).unwrap_or_else(|e| fail(&e));

    println!("\nSaved provider '{name}' and model '{model}' to .env.");
    println!("Run `raptor run \"your request\"` to start.");
    0
}

fn build_cfg(o: &Opts) -> RunConfig {
    let mut c = RunConfig::default();
    if let Some(v) = num::<usize>(o, "max-llm-calls") {
        c.max_llm_calls = v;
    }
    if let Some(v) = num::<usize>(o, "workers") {
        c.max_workers = v.clamp(1, 16);
    }
    if o.flags.contains_key("json") {
        c.backoff_base_ms = 5;
    }
    c
}

fn sim_cfg(o: &Opts, default_latency: u64) -> SimConfig {
    SimConfig {
        latency_ms: num(o, "latency-ms").unwrap_or(default_latency),
        chaos: Chaos {
            transient_rate: num(o, "chaos-transient").unwrap_or(0.0),
            degrade_rate: num(o, "chaos-degrade").unwrap_or(0.0),
            seed: num(o, "seed").unwrap_or(1),
            fail_tasks: o
                .flags
                .get("chaos-fail")
                .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
                .unwrap_or_default(),
        },
    }
}

fn choose_provider(o: &Opts, sim: SimConfig) -> Arc<dyn provider::Provider> {
    let p = if o.flags.contains_key("sim") {
        provider::build_named("sim", &env, sim)
    } else {
        provider::detect(&env, sim, true)
    }
    .unwrap_or_else(|e| fail(&e));
    if let Err(e) = p.warm_up() {
        fail(&format!("{e}\nhint: start Ollama with `ollama serve` and install a model, or select the simulator with `raptor setup`."));
    }
    p
}

fn banner(p: &Arc<dyn provider::Provider>) -> String {
    if p.simulated() {
        "SIMULATED MODE: deterministic placeholder output (orchestration is real).".into()
    } else {
        format!("LOCAL MODEL: {} / {}", p.name(), p.model())
    }
}

fn print_event(e: &Event) {
    use std::io::Write;
    // Ignore write errors (e.g. closed pipe) instead of panicking inside the event log.
    let _ = writeln!(
        std::io::stdout(),
        "{:>7.2}s  {:<12} {}",
        e.t_ms as f64 / 1000.0,
        e.agent,
        e.msg
    );
}

const CHAT_HISTORY_TURNS: usize = 4;
const CHAT_USER_CONTEXT_CHARS: usize = 2000;
const CHAT_ASSISTANT_CONTEXT_CHARS: usize = 3000;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ChatTurn {
    user: String,
    assistant: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChatCommand {
    Continue,
    Clear,
    Exit,
}

fn chat_command(input: &str) -> ChatCommand {
    match input.trim().to_ascii_lowercase().as_str() {
        "exit" | "quit" => ChatCommand::Exit,
        "/clear" | "clear" | "/new" | "new" => ChatCommand::Clear,
        _ => ChatCommand::Continue,
    }
}

fn format_chat_request(history: &VecDeque<ChatTurn>, latest: &str) -> String {
    if history.is_empty() {
        return latest.to_string();
    }

    let turns: Vec<_> = history.iter().rev().take(CHAT_HISTORY_TURNS).collect();
    let mut prompt = String::from(
        "Continue this terminal chat using only the recent turns below. Treat the latest user message as the current request.\n\n",
    );
    for turn in turns.into_iter().rev() {
        prompt.push_str("User:\n");
        prompt.push_str(&truncate(&turn.user, CHAT_USER_CONTEXT_CHARS));
        prompt.push_str("\nAssistant:\n");
        prompt.push_str(&truncate(&turn.assistant, CHAT_ASSISTANT_CONTEXT_CHARS));
        prompt.push_str("\n\n");
    }
    prompt.push_str("Latest user message:\n");
    prompt.push_str(latest);
    prompt
}

fn cmd_chat(o: &Opts) -> i32 {
    let provider = choose_provider(o, sim_cfg(o, 250));
    let engine = Engine {
        provider: provider.clone(),
        tools: Arc::new(Tools::builtin()),
        memory: None,
        cfg: build_cfg(o),
    };
    let mut history = VecDeque::new();

    println!(
        "{}\n\nChat started. Use /clear to reset this session; type exit or quit to leave.",
        banner(&provider)
    );
    loop {
        print!("\nYou: ");
        io::stdout()
            .flush()
            .unwrap_or_else(|e| fail(&format!("cannot write to terminal: {e}")));
        let mut message = String::new();
        let count = io::stdin()
            .read_line(&mut message)
            .unwrap_or_else(|e| fail(&format!("cannot read from terminal: {e}")));
        if count == 0 {
            break;
        }
        let message = message.trim();
        if message.is_empty() {
            continue;
        }
        match chat_command(message) {
            ChatCommand::Exit => break,
            ChatCommand::Clear => {
                history.clear();
                println!("Current chat context cleared.");
                continue;
            }
            ChatCommand::Continue => {}
        }

        let request = format_chat_request(&history, message);
        let log = Arc::new(EventLog::new());
        log.set_listener(print_event);
        let result = engine.run(&request, &log, None);
        println!(
            "\n--- answer ({:?}) ---\n{}\n",
            result.status, result.answer
        );
        if let Some(error) = &result.error {
            println!("note: {error}");
        }

        history.push_back(ChatTurn {
            user: message.to_string(),
            assistant: result.answer,
        });
        while history.len() > CHAT_HISTORY_TURNS {
            history.pop_front();
        }
    }
    0
}

fn cmd_run(o: &Opts) -> i32 {
    let request = o
        .positional
        .get(1)
        .cloned()
        .unwrap_or_else(|| fail("usage: raptor run \"<request>\""));
    let json_mode = o.flags.contains_key("json");
    let data_dir = o
        .flags
        .get("data-dir")
        .map(PathBuf::from)
        .unwrap_or_else(data_dir_from_env);
    let provider = choose_provider(o, sim_cfg(o, if json_mode { 0 } else { 250 }));
    let plan: Option<Plan> = o.flags.get("plan-file").map(|f| {
        let text =
            std::fs::read_to_string(f).unwrap_or_else(|e| fail(&format!("cannot read {f}: {e}")));
        serde_json::from_str(&text)
            .unwrap_or_else(|e| fail(&format!("invalid plan JSON in {f}: {e}")))
    });
    let memory = if o.flags.contains_key("no-memory") {
        None
    } else {
        Some(Arc::new(MemoryStore::open(&data_dir)))
    };
    let engine = Engine {
        provider: provider.clone(),
        tools: Arc::new(Tools::builtin()),
        memory,
        cfg: build_cfg(o),
    };
    let log = Arc::new(EventLog::new());
    if !json_mode {
        println!("{}\n", banner(&provider));
        log.set_listener(print_event);
    }
    let t0 = std::time::Instant::now();
    let res: RunResult = engine.run(&request, &log, plan);
    if json_mode {
        println!(
            "{}",
            serde_json::json!({"result": res, "events": log.all(), "elapsed_ms": t0.elapsed().as_millis() as u64})
        );
    } else {
        println!("\n--- answer ({:?}) ---\n{}\n", res.status, res.answer);
        if let Some(e) = &res.error {
            println!("note: {e}");
        }
        let m = &res.metrics;
        println!(
            "tasks {}/{} ok | llm calls {} | retries {} | repairs {} | reviews {} | wall {} ms | parallelism x{}",
            m.tasks_completed, m.tasks_total, m.llm_calls, m.retries, m.repairs, m.critic_reviews, m.wall_ms, m.parallelism
        );
    }
    match res.status {
        RunStatus::Succeeded => 0,
        RunStatus::Partial => 3,
        RunStatus::Failed => 2,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let o = parse(&args);
    if o.flags.contains_key("help") || args.is_empty() {
        print!("{HELP}");
        return;
    }
    if o.flags.contains_key("version") {
        println!("raptor {}", raptor::VERSION);
        return;
    }
    if let Err(e) = load_dotenv(std::path::Path::new(".env")) {
        fail(&e);
    }
    let code = match o.positional.first().map(|s| s.as_str()) {
        Some("run") => cmd_run(&o),
        Some("chat") => cmd_chat(&o),
        Some("setup") => cmd_setup(),
        Some("doctor") => {
            println!("rapTOR offline doctor\n\nLocal Ollama server (2 s probe):");
            let cands = provider::local_candidates(&env);
            for (name, base) in cands {
                match provider::http::probe(&base) {
                    Ok(m) if m.is_empty() => {
                        println!("  {name:<9} {base}  reachable, but NO models installed/loaded")
                    }
                    Ok(m) => println!(
                        "  {name:<9} {base}  OK, {} model(s): {}",
                        m.len(),
                        m.iter().take(4).cloned().collect::<Vec<_>>().join(", ")
                    ),
                    Err(e) => println!("  {name:<9} {base}  not reachable: {e}"),
                }
            }
            let p = provider::detect(&env, SimConfig::default(), true).unwrap_or_else(|e| fail(&e));
            println!(
                "\nWould use: {} ({})",
                p.name(),
                if p.simulated() {
                    "simulator"
                } else {
                    "local model"
                }
            );
            if p.simulated() {
                println!("\nTo use a local model: install Ollama, run `ollama serve`, then `ollama pull llama3.2`.");
            }
            0
        }
        Some("providers") => {
            let p = provider::detect(&env, SimConfig::default(), true).unwrap_or_else(|e| fail(&e));
            println!(
                "selected: {} ({})",
                p.name(),
                if p.simulated() {
                    "simulator"
                } else {
                    "local model"
                }
            );
            println!("offline providers: local Ollama > simulator");
            0
        }
        Some("serve") => {
            let data_dir = o
                .flags
                .get("data-dir")
                .map(PathBuf::from)
                .unwrap_or_else(data_dir_from_env);
            let provider = choose_provider(&o, sim_cfg(&o, 250));
            let (host, port) = (
                o.flags
                    .get("host")
                    .cloned()
                    .unwrap_or_else(|| "127.0.0.1".into()),
                num::<u16>(&o, "port").unwrap_or(8787),
            );
            println!(
                "rapTOR {} on http://{host}:{port}\n{}",
                raptor::VERSION,
                banner(&provider)
            );
            let state = Arc::new(AppState::new(
                provider,
                build_cfg(&o),
                Some(Arc::new(MemoryStore::open(&data_dir))),
                data_dir,
                num(&o, "latency-ms").unwrap_or(250),
            ));
            if let Err(e) = server::serve(state, &host, port) {
                fail(&e);
            }
            0
        }
        _ => {
            print!("{HELP}");
            1
        }
    };
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn chat_context_includes_both_roles_and_is_limited_to_recent_turns() {
        let history: VecDeque<_> = (0..6)
            .map(|index| ChatTurn {
                user: format!("user-{index}"),
                assistant: format!("assistant-{index}"),
            })
            .collect();

        let prompt = format_chat_request(&history, "latest");

        assert!(!prompt.contains("user-1"));
        assert!(prompt.contains("user-2"));
        assert!(prompt.contains("assistant-5"));
        assert!(prompt.contains("Latest user message:\nlatest"));
    }

    #[test]
    fn chat_commands_reset_or_exit_without_becoming_prompts() {
        assert_eq!(chat_command("/clear"), ChatCommand::Clear);
        assert_eq!(chat_command(" NEW "), ChatCommand::Clear);
        assert_eq!(chat_command("QUIT"), ChatCommand::Exit);
        assert_eq!(chat_command("continue"), ChatCommand::Continue);
    }

    #[test]
    fn installing_an_ollama_model_is_explicit_and_propagates_failures() {
        let installed = vec!["qwen2.5:3b".to_string()];
        let selected = select_ollama_model(
            &installed,
            0,
            || panic!("installed model selection must not prompt"),
            |_| panic!("installed model selection must not download"),
        )
        .unwrap();
        assert_eq!(selected, "qwen2.5:3b");

        let selected = select_ollama_model(
            &installed,
            installed.len(),
            || Ok("llama3.2:3b".into()),
            |model| {
                assert_eq!(model, "llama3.2:3b");
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(selected, "llama3.2:3b");

        let error = select_ollama_model(
            &installed,
            installed.len(),
            || Ok("llama3.2:3b".into()),
            |_| Err("Ollama pull failed".into()),
        )
        .unwrap_err();
        assert_eq!(error, "Ollama pull failed");
    }

    #[test]
    fn ollama_download_passes_the_model_as_one_argument_without_a_shell() {
        let command = ollama_pull_command("model; do-not-run");
        let args: Vec<_> = command.get_args().collect();

        assert_eq!(command.get_program(), OsStr::new("ollama"));
        assert_eq!(args, [OsStr::new("pull"), OsStr::new("model; do-not-run")]);
    }
}
