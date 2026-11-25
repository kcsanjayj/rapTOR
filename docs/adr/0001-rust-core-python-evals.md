# ADR 0001 — Rust core, Python for evals/SDK
**Status:** accepted
**Context:** v1 was pure Python. Its concurrency and failure handling were hard to reason about and produced silent bugs.
**Decision:** Rust owns the engine, server and providers. Python owns the SDK and eval harness (stdlib only).
**Consequences:** + predictable overhead, single static binary, no data races by construction. − two languages; evals talk to the binary over CLI/HTTP (which is also a feature: black-box tests).
