"""Eval runner. `python -m raptor.evals [--live] [--filter text]`.

Suites: core (CLI black-box, simulator, deterministic), api (HTTP server + SDK), live (real model, opt-in).
Exits non-zero if any non-skipped case fails, so it can gate CI.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..client import Raptor, RunError, find_binary, run_cli, ROOT
from .graders import GRADERS

EVALS = ROOT / "evals"


def run_case(case: dict, live: bool) -> dict:
    t0 = time.time()
    data = tempfile.mkdtemp(prefix="raptor-eval-")
    args = list(case.get("args", []))
    use_memory = "--memory" in args
    args = [a for a in args if a != "--memory"] + ["--data-dir", data]
    if not use_memory:
        args.append("--no-memory")
    if case.get("plan_file"):
        args += ["--plan-file", str(EVALS / "plans" / case["plan_file"])]
    if "--latency-ms" not in args and not live:
        args += ["--latency-ms", "0"]
    try:
        reqs = case.get("sequence") or [case["request"]]
        runs = []
        for r in reqs * (case.get("repeat", 1) if not case.get("sequence") else 1):
            runs.append(run_cli(r, args, timeout=90) if not live else run_cli_live(r, args))
        run = runs[-1]
        checks = []
        for spec in case["expect"]:
            if spec["type"] == "deterministic":
                a, b = runs[0]["result"], runs[1]["result"]
                ok = a["answer"] == b["answer"] and a["plan"] == b["plan"] and a["metrics"]["repairs"] == b["metrics"]["repairs"] and a["metrics"]["llm_calls"] == b["metrics"]["llm_calls"]
                checks.append({"type": "deterministic", "passed": ok, "detail": "runs identical" if ok else "runs differ"})
            else:
                ok, detail = GRADERS[spec["type"]](run, spec)
                checks.append({"type": spec["type"], "passed": ok, "detail": detail})
        passed = all(c["passed"] for c in checks)
    except Exception as e:  # noqa: BLE001
        checks, passed = [{"type": "execution", "passed": False, "detail": repr(e)[:200]}], False
    return {"id": case["id"], "title": case["title"], "suite": case.get("suite", "core"), "passed": passed, "checks": checks, "seconds": round(time.time() - t0, 2)}


def run_cli_live(request: str, args: list) -> dict:
    cmd = [str(find_binary()), "run", request, "--json", *args]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    out = json.loads(p.stdout)
    out["exit_code"] = p.returncode
    return out


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def api_suite() -> list:
    import urllib.request, urllib.error
    port = free_port()
    proc = subprocess.Popen([str(find_binary()), "serve", "--port", str(port), "--sim", "--latency-ms", "0", "--data-dir", tempfile.mkdtemp()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    out = []
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(base + "/api/health", timeout=0.5); break
            except Exception:  # noqa: BLE001
                time.sleep(0.1)
        d = Raptor(base)

        def post(body: bytes):
            req = urllib.request.Request(base + "/api/runs", data=body, method="POST", headers={"Content-Type": "application/json"})
            try:
                return urllib.request.urlopen(req, timeout=5).status
            except urllib.error.HTTPError as e:
                return e.code

        def case(id_, title, fn):
            t0 = time.time()
            try:
                ok, detail = fn()
            except Exception as e:  # noqa: BLE001
                ok, detail = False, repr(e)[:200]
            out.append({"id": id_, "title": title, "suite": "api", "passed": ok, "checks": [{"type": "api", "passed": ok, "detail": detail}], "seconds": round(time.time() - t0, 2)})

        h = d.health()
        case("api-health", "Server reports provider, tools and limits", lambda: (h.get("ok") and h.get("simulated") and "calculator" in h["tools"], str(h)[:100]))
        case("api-empty-400", "Empty request is rejected with 400", lambda: ((c := post(b'{"request":"  "}')) == 400, f"HTTP {c}"))
        case("api-badjson-400", "Malformed JSON is rejected with 400", lambda: ((c := post(b"{nope")) == 400, f"HTTP {c}"))
        case("api-oversize-413", "Oversized body is rejected with 413", lambda: ((c := post(json.dumps({"request": "x" * 20000}).encode())) == 413, f"HTTP {c}"))
        case("api-404", "Unknown run id returns 404", lambda: (d._req("GET", "/api/runs/nope")[0] == 404, "404"))

        def full():
            r = d.run("Compare Rust and Go for a CLI tool", timeout=30)
            seqs = [e["seq"] for e in r["events"]]
            ok = r["result"]["status"] == "succeeded" and seqs == list(range(len(seqs))) and r["events"][-1]["kind"] == "run_finished"
            return ok, f"{len(seqs)} events, dense seq, ends with run_finished"
        case("api-full-run", "Full run over HTTP: ordered, dense events ending in run_finished", full)
        case("api-root", "Root endpoint points users to the terminal CLI", lambda: ((r := urllib.request.urlopen(base + "/", timeout=3)).status == 200 and "terminal CLI" in r.read().decode("utf-8"), "GET / returns terminal guidance"))
    finally:
        proc.terminate()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="raptor-evals")
    ap.add_argument("--live", action="store_true", help="also run local-model cases (needs Ollama and an installed model)")
    ap.add_argument("--filter", default="")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args(argv)
    try:
        find_binary()
    except RunError as e:
        print(e, file=sys.stderr); return 2
    cases = json.loads((EVALS / "cases.json").read_text())
    results = []
    for c in cases:
        if a.filter and a.filter not in c["id"]:
            continue
        if c.get("suite") == "live" and not a.live:
            continue
        r = run_case(c, live=c.get("suite") == "live")
        results.append(r)
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['id']:<34} {r['seconds']:>5.2f}s  {c['title']}")
        for ch in r["checks"]:
            if not ch["passed"]:
                print(f"        x {ch['type']}: {ch['detail']}")
    if not a.filter:
        for r in api_suite():
            results.append(r)
            print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['id']:<34} {r['seconds']:>5.2f}s  {r['title']}")
            for ch in r["checks"]:
                if not ch["passed"]:
                    print(f"        x {ch['detail']}")
    passed = sum(r["passed"] for r in results)
    print(f"\n{passed}/{len(results)} passed")
    if not a.no_report and not a.filter:
        write_report(results)
    return 0 if passed == len(results) else 1


def write_report(results: list) -> None:
    rep = EVALS / "reports"; rep.mkdir(exist_ok=True)
    (rep / "latest.json").write_text(json.dumps(results, indent=2))
    n = sum(r["passed"] for r in results)
    lines = ["# Eval report", "", f"**{n}/{len(results)} passed** · generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", "",
             "| Result | Case | Suite | Checks | Time |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {'PASS' if r['passed'] else '**FAIL**'} | `{r['id']}` — {r['title']} | {r['suite']} | {sum(c['passed'] for c in r['checks'])}/{len(r['checks'])} | {r['seconds']}s |")
    lines += ["", "These evals assert *orchestration behaviour* against the deterministic simulator. They do not measure real-model quality (see `docs/EVALS.md`)."]
    (rep / "latest.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
