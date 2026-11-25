# Eval report

**26/26 passed** · generated 2026-11-25 10:28 UTC

| Result | Case | Suite | Checks | Time |
|---|---|---|---|---|
| PASS | `dag-parallel-roots` — Comparison request becomes a DAG with parallel research roots | core | 4/4 | 1.1s |
| PASS | `data-flow` — Every dependent task receives its upstream outputs (v1 bug) | core | 3/3 | 0.58s |
| PASS | `critic-reviews-every-llm-node` — Critic reviews 100% of LLM nodes, including the last (v1 bug) | core | 2/2 | 0.61s |
| PASS | `critic-gates-release` — No task starts before its dependencies are completed and reviewed | core | 1/1 | 0.8s |
| PASS | `no-demo-topic-leak` — Unrelated request never mentions the old hard-coded Python/Rust demo (v1 bug) | core | 2/2 | 0.6s |
| PASS | `vs-syntax` — 'A vs B' phrasing is understood | core | 2/2 | 0.65s |
| PASS | `calculator-tool` — Arithmetic is delegated to the deterministic tool and surfaced | core | 3/3 | 0.59s |
| PASS | `calculator-dos-contained` — 9**9**9 fails fast with a clear error instead of hanging (v1 bug) | core | 5/5 | 0.61s |
| PASS | `transient-retry` — Injected 503s are retried with backoff and the run still succeeds | core | 3/3 | 0.7s |
| PASS | `critic-repair` — Deliberately poor drafts are caught by the critic and repaired | core | 4/4 | 0.65s |
| PASS | `permanent-failure-partial` — A permanently failing task yields an honest partial result and skips dependents | core | 6/6 | 0.61s |
| PASS | `reject-cycle` — A cyclic plan is rejected before any work starts (v1 bug) | core | 3/3 | 0.59s |
| PASS | `reject-dangling-dep` — Unknown dependencies are an error, not silently dropped | core | 2/2 | 0.63s |
| PASS | `reject-invented-tool` — A plan that invents a tool is rejected | core | 2/2 | 0.63s |
| PASS | `reject-oversized-plan` — A 13-task plan exceeds the limit and is rejected | core | 2/2 | 0.63s |
| PASS | `budget-hard-cap` — --max-llm-calls is a hard cap and exhaustion is reported | core | 3/3 | 0.65s |
| PASS | `real-parallelism` — Independent tasks overlap in time (parallelism > 1.2x) | core | 1/1 | 3.06s |
| PASS | `deterministic` — Same seed => identical plan, answer and metrics | core | 1/1 | 2.46s |
| PASS | `memory-recall` — A similar earlier request is recalled and fed to the planner | core | 1/1 | 2.05s |
| PASS | `api-health` — Server reports provider, tools and limits | api | 1/1 | 0.0s |
| PASS | `api-empty-400` — Empty request is rejected with 400 | api | 1/1 | 0.0s |
| PASS | `api-badjson-400` — Malformed JSON is rejected with 400 | api | 1/1 | 0.0s |
| PASS | `api-oversize-413` — Oversized body is rejected with 413 | api | 1/1 | 0.0s |
| PASS | `api-404` — Unknown run id returns 404 | api | 1/1 | 0.0s |
| PASS | `api-full-run` — Full run over HTTP: ordered, dense events ending in run_finished | api | 1/1 | 0.07s |
| PASS | `api-root` — Root endpoint points users to the terminal CLI | api | 1/1 | 0.01s |

These evals assert *orchestration behaviour* against the deterministic simulator. They do not measure real-model quality (see `docs/EVALS.md`).
