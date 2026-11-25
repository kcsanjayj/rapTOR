# ADR 0004 — UI is a pure function of the event stream; polling transport
**Status:** accepted
**Decision:** all orchestration state is expressed as typed events. The UI reduces events; it contains no orchestration logic. Transport is HTTP polling (`events?after=N`, 160 ms).
**Consequences:** + one source of truth (CLI, traces, API, UI), trivially testable and replayable, works behind any proxy. − ≤ 160 ms latency and some idle requests; SSE is a straightforward future swap.
