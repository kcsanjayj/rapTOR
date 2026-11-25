# ADR 0005 — std threads and channels, no async runtime
**Status:** accepted
**Context:** workload is ≤ 16 blocking HTTP calls per run.
**Decision:** a single scheduler thread plus worker threads over `mpsc`; `ureq` and `tiny_http` (blocking).
**Consequences:** + 4 direct dependencies, simple mental model. − does not scale to thousands of concurrent runs per process; revisit if needed.
