# Evals

## Purpose
Prove the **orchestration** behaves correctly and keeps behaving correctly. Each v1 defect has a named regression case.

## Layers
| Layer | Count | Runs against | Catches |
|---|---|---|---|
| Rust unit tests | 48 | pure functions and CLI helpers | parser, DAG, retry, budget, chat context, model setup |
| Rust integration tests | 15 | engine + mock local Ollama server | data flow, ordering, recovery, model discovery, offline endpoint guard |
| Python grader tests | 5 | synthetic runs | broken graders (an eval that cannot fail is worthless) |
| Python benchmark tests | 1 | benchmark statistics | nearest-rank percentiles and summary values |
| Core evals (black-box CLI) | 19 | `raptor run --json --sim` | end-to-end behaviour |
| API evals | 7 | `raptor serve` over HTTP | validation, limits, event ordering, CLI guidance |
| Local-model evals (opt-in) | 2 | local Ollama, if running | basic sanity only |

## Running
```bash
cd python
python -m raptor.evals            # core + api, writes evals/reports/latest.{md,json}
python -m raptor.evals --live     # also run local-model cases (needs Ollama + an installed model)
python -m raptor.evals --filter calculator
```
Exit code is non-zero on any failure, so it can gate CI.

From the project root, run `python scripts/benchmark.py --runs 30` for a separate, environment-stamped simulator timing report; timing is not used as a correctness gate.

## Case format (`evals/cases.json`)
```json
{"id":"critic-repair","request":"Compare Rust and Go for a CLI tool",
 "args":["--chaos-degrade","1.0","--seed","3"],
 "expect":[{"type":"status","is":"succeeded"},{"type":"metric","name":"repairs","gte":4}]}
```
Graders live in `python/raptor/evals/graders.py` (status, data_flow, critic_coverage, no_unreviewed_dependents, metric, node_status, error_contains, deterministic, …).

## Honest scope
- Core/API evals use the deterministic simulator. They validate **machinery**, not intelligence.
- The simulator's critic is a heuristic; passing the repair eval shows the *loop* works (detect → feedback → re-run → accept), not that real critics are accurate.
- Local-model evals require a running local Ollama server and installed model. No cloud API keys or remote model endpoints are supported.
- Possible next step: add a model-graded quality suite with a fixed rubric and report variance across seeds.
