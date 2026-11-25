# Changelog

## Unreleased — rapTOR local terminal workflow
- Moved the session chat loop into the Rust CLI; bounded user/assistant context stays in-process and `/clear` resets it.
- Added a repeatable, environment-stamped CLI benchmark and deterministic terminal demo.
- CI now runs formatting, strict Clippy, Rust/Python tests, and evals on Windows and Ubuntu.
- Renamed the Rust crate/binary and Python SDK/eval package to rapTOR/raptor; configuration variables now use the `RAPTOR_` prefix.
- Terminal chats keep follow-up context only for the current launch, disable persistent recall, and support `/clear`.
- Setup can optionally run `ollama pull` to download a model; this explicit action requires internet, while inference remains local.
- Removed online/cloud provider selection and API-key configuration.
- Restricted model requests to a local Ollama server on `127.0.0.1`; remote endpoints are rejected.
- Setup offers local Ollama or the deterministic offline simulator.
- This offline-only policy supersedes the earlier OpenAI-compatible endpoint support described under 2.0.0.

## 2.0.0 — Rust core rewrite
**Architecture:** engine, server and providers rewritten in Rust; Python reduced to SDK + eval harness; new event-sourced black-and-white UI.

**Fixed (all verified by a named regression eval or test):**
- No API key no longer crashes; deterministic simulator is the labelled last resort. Model ids are discovered, not hard-coded.
- Dependency outputs now reach dependent tasks (v1 never populated `input_data`).
- Critic reviews every LLM task, including the last, and gates release of dependents (v1 skipped the final wave).
- Permanently failed tasks skip their dependents explicitly and yield a `partial` run with an honest answer.
- Invalid plans are rejected with a reason and retried; fallback plan is topic-agnostic (v1 fell back to "Python vs Rust").
- Cycles, unknown dependencies, unknown tools and oversized plans are rejected before execution (v1 silently dropped bad deps and never validated cycles).
- Calculator is a bounded parser (v1 `9**9**9` could pin a CPU).
- Memory is actually recalled before planning and saved after success (v1 was never wired and its calls did not match its API).
- Hard per-run LLM-call budget and deadline.
- Event log is poison-tolerant; a listener panic cannot wedge a run.

- Earlier 2.0.0 releases supported Ollama, LM Studio, llama.cpp and custom OpenAI-compatible endpoints; these remote/custom endpoint options were removed in the offline-only update above.

**Removed:** dead modules (`model_router`, `router`, `plan_schema`, unused memory manager), unused dependencies, and README claims for features that were not implemented (ADK, FastAPI, OpenTelemetry).

## 1.x — Python prototype
Superseded.
