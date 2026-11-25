# ADR 0003 — The critic gates release of dependents
**Status:** accepted
**Context:** v1 ran the critic after a whole wave and skipped the last wave; dependents could consume unreviewed (or later-repaired) output.
**Decision:** review and repair happen inside the worker, before the task is reported complete. Weak-after-repair results are accepted but flagged `low_confidence`; critic outage is recorded and the task accepted.
**Consequences:** + no stale downstream state, 100% review coverage (eval-enforced). − extra LLM calls per task (bounded by the call budget).
