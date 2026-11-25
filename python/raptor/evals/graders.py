"""Declarative graders. Each takes (run, spec) and returns (passed, detail).

`run` is the dict produced by `run_cli`: {"result", "events", "elapsed_ms", "exit_code"}.
They assert *orchestration behaviour* (DAG shape, data flow, recovery, limits), not model quality.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

Result = Tuple[bool, str]


def _nodes(run): return run["result"]["nodes"]
def _events(run, kind): return [e for e in run["events"] if e["kind"] == kind]
def _plan(run): return (run["result"].get("plan") or {}).get("tasks", [])


def status(run, s) -> Result:
    got = run["result"]["status"]
    want = s["is"] if isinstance(s["is"], list) else [s["is"]]
    return got in want, f"status={got}, wanted {want}"


def min_tasks(run, s) -> Result:
    n = len(_plan(run)); return n >= s["n"], f"{n} tasks (min {s['n']})"


def parallel_roots(run, s) -> Result:
    n = sum(1 for t in _plan(run) if not t["dependencies"]); return n >= s["min"], f"{n} root tasks (min {s['min']})"


def data_flow(run, s) -> Result:
    """Every started node received exactly its declared dependencies' outputs, with non-zero size."""
    plan = {t["id"]: t["dependencies"] for t in _plan(run)}
    bad = []
    for e in _events(run, "node_started"):
        want, got = plan[e["node"]], e["data"]["inputs_from"]
        if want != got or (want and e["data"]["input_chars"] == 0):
            bad.append(f"{e['node']}: wanted {want}, got {got}")
    return not bad, "all dependents received upstream outputs" if not bad else "; ".join(bad)


def critic_coverage(run, s) -> Result:
    llm = [n for n in _nodes(run) if n["status"] == "completed" and not any(t["id"] == n["id"] and t["kind"] == "tool" for t in _plan(run))]
    if not llm: return True, "no LLM nodes"
    cov = sum(1 for n in llm if n["reviewed"]) / len(llm)
    return cov >= s["min"], f"{cov:.0%} of completed LLM nodes reviewed (min {s['min']:.0%})"


def answer_contains(run, s) -> Result:
    a = run["result"]["answer"].lower(); missing = [x for x in s["all"] if x.lower() not in a]
    return not missing, "all present" if not missing else f"missing {missing}"


def answer_not_contains(run, s) -> Result:
    a = (run["result"]["answer"] + str(_plan(run))).lower(); hit = [x for x in s["any"] if x.lower() in a]
    return not hit, "none present" if not hit else f"found {hit}"


def metric(run, s) -> Result:
    v = run["result"]["metrics"][s["name"]]
    ok = (("gte" not in s or v >= s["gte"]) and ("lte" not in s or v <= s["lte"]) and ("eq" not in s or v == s["eq"]))
    return ok, f"{s['name']}={v} ({ {k: s[k] for k in ('gte','lte','eq') if k in s} })"


def node_status(run, s) -> Result:
    n = next((n for n in _nodes(run) if n["id"] == s["id"]), None)
    return (n is not None and n["status"] == s["is"]), f"{s['id']}={n['status'] if n else 'missing'}, wanted {s['is']}"


def node_output(run, s) -> Result:
    n = next((n for n in _nodes(run) if n["id"] == s["id"]), None)
    return (n is not None and s["contains"] in (n.get("output") or "")), f"{s['id']} output={(n or {}).get('output')!r}"


def error_contains(run, s) -> Result:
    e = (run["result"].get("error") or "") + " " + " ".join((n.get("error") or "") for n in _nodes(run))
    return s["text"].lower() in e.lower(), f"error text: {e.strip()[:120]!r}"


def task_ids_contain(run, s) -> Result:
    ids = " ".join(t["id"] for t in _plan(run)); missing = [x for x in s["all"] if x not in ids]
    return not missing, f"ids={ids}"


def event_count(run, s) -> Result:
    n = len(_events(run, s["kind"])); return n >= s["gte"], f"{n} '{s['kind']}' events (min {s['gte']})"


def memory_recalled(run, s) -> Result:
    ev = _events(run, "memory_recalled"); n = ev[0]["data"]["count"] if ev else 0
    return n >= 1, f"recalled {n} past run(s)"


def elapsed_lte(run, s) -> Result:
    ms = run["elapsed_ms"]; return ms <= s["ms"], f"{ms} ms (max {s['ms']})"


def no_unreviewed_dependents(run, s) -> Result:
    """No node may start before every dependency has completed AND been reviewed (critic gates release)."""
    seq = {}
    for e in run["events"]:
        if e["kind"] == "node_completed": seq[e["node"]] = e["seq"]
    bad = [f"{e['node']}" for e in _events(run, "node_started") for d in e["data"]["inputs_from"] if seq.get(d, 1 << 30) > e["seq"]]
    return not bad, "ordering respected" if not bad else f"started early: {bad}"


GRADERS: Dict[str, Callable[[dict, dict], Result]] = {
    "status": status, "min_tasks": min_tasks, "parallel_roots": parallel_roots, "data_flow": data_flow,
    "critic_coverage": critic_coverage, "answer_contains": answer_contains, "answer_not_contains": answer_not_contains,
    "metric": metric, "node_status": node_status, "node_output": node_output, "error_contains": error_contains,
    "task_ids_contain": task_ids_contain, "event_count": event_count, "memory_recalled": memory_recalled,
    "elapsed_lte": elapsed_lte, "no_unreviewed_dependents": no_unreviewed_dependents,
}
