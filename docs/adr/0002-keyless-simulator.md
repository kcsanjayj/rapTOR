# ADR 0002 — Offline simulator and local Ollama providers
**Status:** accepted
**Context:** v1 crashed without credentials and shipped stale model ids. Reliability behaviour (retries, repair) cannot be unit-tested against live models.
**Decision:** rapTOR runs fully offline. It supports the built-in deterministic simulator and Ollama on loopback only; cloud credentials and remote endpoints are ignored or rejected. The simulator is seeded and **always labelled** (`simulated:true`) and supports fault injection.
**Consequences:** + no cloud credentials or remote model calls, reproducible CI, deterministic evals. − real-model use requires a locally installed Ollama model; simulator output is a placeholder, not AI.
