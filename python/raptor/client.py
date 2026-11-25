"""Thin clients for the Rust core: HTTP (server) and CLI (subprocess, JSON mode)."""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]


class RunError(RuntimeError):
    pass


def find_binary() -> Path:
    """RAPTOR_BIN, else target/release, else target/debug."""
    env = os.environ.get("RAPTOR_BIN")
    candidates = [Path(env)] if env else []
    binary = "raptor.exe" if os.name == "nt" else "raptor"
    candidates += [ROOT / "target/release" / binary, ROOT / "target/debug" / binary]
    for c in candidates:
        if c.is_file():
            return c
    raise RunError("raptor binary not found; run `cargo build --release` (or set RAPTOR_BIN)")


class Raptor:
    """HTTP client for `raptor serve`."""

    def __init__(self, base_url: str = "http://127.0.0.1:8787", timeout: float = 10.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _req(self, method: str, path: str, body: Optional[dict] = None) -> tuple[int, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def health(self) -> Dict[str, Any]:
        return self._req("GET", "/api/health")[1]

    def start(self, request: str, **opts: Any) -> str:
        code, j = self._req("POST", "/api/runs", {"request": request, **opts})
        if code != 202:
            raise RunError(f"HTTP {code}: {j.get('error')}")
        return j["id"]

    def events(self, run_id: str, after: int = 0) -> Dict[str, Any]:
        return self._req("GET", f"/api/runs/{run_id}/events?after={after}")[1]

    def result(self, run_id: str) -> Dict[str, Any]:
        return self._req("GET", f"/api/runs/{run_id}")[1]

    def run(self, request: str, timeout: float = 60.0, **opts: Any) -> Dict[str, Any]:
        """Start a run, wait for completion, return {'result': ..., 'events': [...]}."""
        rid = self.start(request, **opts)
        events: List[dict] = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            j = self.events(rid, len(events))
            events += j["events"]
            if j["done"]:
                j2 = self.result(rid)
                if j2.get("result"):
                    return {"result": j2["result"], "events": events, "id": rid}
            time.sleep(0.05)
        raise RunError(f"run {rid} did not finish within {timeout}s")


def run_cli(request: str, args: Optional[List[str]] = None, env: Optional[dict] = None, timeout: float = 60.0) -> Dict[str, Any]:
    """Run `raptor run --json` and return {'result','events','elapsed_ms','exit_code'}."""
    cmd = [str(find_binary()), "run", request, "--json", "--sim", *(args or [])]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env={**os.environ, **(env or {})})
    if not p.stdout.strip():
        raise RunError(f"no JSON output (exit {p.returncode}): {p.stderr.strip()[:300]}")
    out = json.loads(p.stdout)
    out["exit_code"] = p.returncode
    return out
