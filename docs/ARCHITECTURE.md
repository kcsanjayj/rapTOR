# Architecture

## Principles

1. **Rust owns correctness and concurrency; Python owns iteration speed.** The engine (graph, scheduler, retries, budgets, server) is Rust. Evals and the SDK are Python because they change often and need no performance.
2. **One event stream.** Every state change is a typed `Event`. The CLI timeline, JSONL traces and optional HTTP API are projections of it.
3. **Fail loudly, never silently.** Unknown dependency → error. Cycle → error. Critic unavailable → recorded. Partial result → labelled. Simulator → labelled everywhere.
4. **Small trusted surface.** Four direct Rust dependencies. Parsers and validators are hand-written and bounded.

## Components

```text
            ┌──────────────────────────── Engine::run ────────────────────────────┐
 request ─▶ │ memory.recall ─▶ planner ─▶ dag::validate ─▶ scheduler loop ─▶ synth │ ─▶ RunResult
            └───────────────────────────────┬──────────────────────┬───────────────┘
                                            │ spawn                │ events
                                     ┌──────▼───────┐        ┌─────▼──────┐
                                     │ worker thread│        │  EventLog  │──▶ JSONL, CLI, /api
                                     │ exec→critic→ │        └────────────┘
                                     │ repair loop  │
                                     └──────┬───────┘
                                       llm::Llm (budget, retry/backoff)
                                            │
                                  provider::{Sim, local Ollama HTTP}
```

### Scheduler (`engine.rs`)
A single coordinating thread owns all graph state (`remaining` dependency counts, `ready` queue, free worker ids). Workers never touch shared graph state; they send one `Outcome` over an `mpsc` channel. This removes locking from the hot path and makes ordering easy to reason about. Worker panics are caught (`catch_unwind`) and become task failures instead of hanging the run.

### Critic-gated release
A worker runs `execute → review → (repair → review)*` **before** reporting completion. The scheduler therefore never releases a dependent on unreviewed output. If the critic itself is unavailable the task is accepted and a `critic_unavailable` event is recorded.

### Provider layer
`Provider::complete(&LlmRequest)`. Real providers render `system`/`user` prompts; the simulator reads the structured `Ctx` so it never parses prompts. Detection order and model discovery are in `provider/mod.rs`.

### Failure semantics
| Situation | Result |
|---|---|
| Transient provider error | retry with backoff (≤ `retries`), then task fails |
| Local server rejects a request | fail fast with a clear error |
| Task fails | dependents `skipped`; run `partial` if anything else completed |
| Nothing completes | run `failed`, with the first error |
| Budget exhausted | `budget_exhausted` event; unfinished tasks fail/skip |
| Deadline | no new tasks start; pending tasks `skipped` |
| Plan invalid twice | neutral 3-step fallback plan, `plan_fallback` event |

## Event schema (abridged)
`run_started · memory_recalled · plan_requested · plan_rejected · plan_fallback · plan_accepted · node_started · tool_called · attempt_failed · review_started · critic_review · critic_unavailable · repair_requested · node_completed · node_failed · node_skipped · budget_exhausted · synthesis_started · synthesis_fallback · run_finished`
Each has `seq` (dense), `t_ms`, `agent`, optional `node`, `msg`, and a structured `data` object.

## What changed from v1 (Python prototype)
See [`../CHANGELOG.md`](../CHANGELOG.md).
