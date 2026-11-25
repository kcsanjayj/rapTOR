# Resume-ready project bullets

Pick 3–4. Each is backed by something in the repo.

- Built a multi-agent orchestrator in **Rust** (4 direct dependencies) that runs validated task DAGs on a parallel worker pool, **by** implementing a single-owner scheduler with channel-based workers and critic-gated task release.
- Raised reliability from "crashes without credentials" to **runs with zero configuration**, as verified by an end-to-end keyless run and a pinned unit test, **by** supporting only local Ollama and a clearly labelled offline simulator, with model discovery from Ollama.
- Eliminated **8 defects** found in a self-review of my prototype (lost data flow, unreviewed final tasks, unvalidated cycles, unbounded calculator, topic-leaking fallback…), as measured by **26 end-to-end evals** and **63 Rust tests**, **by** rewriting the core and adding seeded fault injection (503s, weak drafts, permanent failures).
- Shipped a terminal-first experience with interactive local-provider setup, session-scoped user/assistant context, and a live per-agent timeline, **by** deriving CLI output and API/JSONL traces from one typed event stream.
- Added repeatable performance reporting with warmups, sample distributions, environment details, and explicit measurement boundaries instead of quoting machine-specific latency as a universal result.
- Designed an **eval harness in Python** (stdlib-only) with declarative graders, adversarial plan fixtures, determinism checks and CI-gating exit codes, **by** testing the graders themselves so evals cannot silently pass.

Honesty note for interviews: the local Ollama-compatible adapter was validated against a mock server; a live Ollama installation has not yet been tested.
