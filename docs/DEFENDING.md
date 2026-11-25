# Defending this project (interview guide)

## 30-second pitch
"rapTOR turns a request into a dependency graph, runs independent tasks in parallel, has a critic review every result, and repairs weak ones. The core is Rust; evals and the SDK are Python. Inference uses local Ollama or a labelled deterministic simulator; 26 end-to-end evals exercise orchestration behavior without a model dependency."

## Questions you should expect

**Why Rust for the core?** The scheduler is one thread owning graph state with workers reporting over channels, so graph coordination has a single owner. Python stays where iteration speed matters (evals). The repository benchmark reports both end-to-end CLI time and the engine's coarse millisecond timer; neither should be quoted without its machine and measurement conditions.

**Why not tokio/async?** The work is dominated by blocking HTTP waits across ≤ 16 workers. Threads + channels are simpler, have fewer dependencies, and are easier to reason about. I'd reconsider at hundreds of concurrent runs.

**Isn't a simulator cheating?** It's labelled in the CLI banner and API (`simulated: true`). It exists so that retries, repair, partial failure and parallelism are *deterministically testable* — you can't unit-test "the provider returned 503 twice" against a live API. It never claims to be intelligence.

**Did you test with a real model?** The inference path is local-only. The Ollama-compatible HTTP path is tested against a mock server; a live installation should be tested separately before claiming real-model validation. Optional `--live` evals use Ollama already running on the same computer.

**How do you know the critic works?** In the simulator it's a heuristic, so the eval proves the *loop*. For real models a critic is a heuristic too; that's why weak results are flagged `low_confidence` and surfaced instead of hidden, and why the critic being unavailable is an explicit event.

**What was wrong with v1?** I reviewed my own prototype and found, with a mock-LLM harness, that (1) upstream outputs never reached dependents, (2) the critic skipped the last wave, (3) failed tasks never reached repair, (4) bad plans fell back to a hard-coded Python-vs-Rust plan, (5) cycles weren't validated, (6) the cycle budget conflated depth with repairs, (7) memory was never wired and would have failed, (8) `9**9**9` hung the calculator. v2 fixes each, and each has a regression eval.

**What would you do next?** Run local-model evals and tune prompts; add model-graded quality evals with variance reporting; persist runs; per-task timeouts with cancellation; embeddings-based memory.

**What's the weakest part?** The local Ollama adapter has only been tested against a mock server, and model-picking is a heuristic.

## Know the numbers
63 Rust tests · 6 Python tests (5 grader + 1 benchmark) · 26 deterministic evals · 4 direct Rust dependencies · 0 Python runtime dependencies. The benchmark script reports environment-specific p50/p95 values; generate a fresh report before quoting performance numbers.
