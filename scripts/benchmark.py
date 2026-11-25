"""Repeatable zero-latency CLI timing benchmark for rapTOR's simulator path."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from raptor.client import find_binary  # noqa: E402


REQUEST = "Compare Rust and Go for a CLI tool"


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[index]


def run_once(binary: Path) -> tuple[float, float]:
    command = [
        str(binary),
        "run",
        REQUEST,
        "--json",
        "--sim",
        "--latency-ms",
        "0",
        "--no-memory",
        "--seed",
        "1",
    ]
    started = time.perf_counter()
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    wall_ms = (time.perf_counter() - started) * 1000
    if process.returncode != 0:
        raise RuntimeError(
            f"raptor exited {process.returncode}: {process.stderr.strip()[:500]}"
        )
    try:
        document = json.loads(process.stdout)
        result = document["result"]
        engine_ms = float(result["metrics"]["wall_ms"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"raptor returned an invalid benchmark result: {error}") from error
    if result.get("status") != "succeeded":
        raise RuntimeError(f"benchmark run did not succeed: {result.get('status')}")
    return wall_ms, engine_ms


def summarize(samples: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(statistics.median(samples), 3),
        "p95_ms": round(percentile(samples, 0.95), 3),
        "mean_ms": round(statistics.fmean(samples), 3),
        "stdev_ms": round(statistics.stdev(samples), 3) if len(samples) > 1 else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.runs < 1:
        parser.error("--warmups must be >= 0 and --runs must be >= 1")

    binary = find_binary()
    wall_samples: list[float] = []
    engine_samples: list[float] = []
    for index in range(args.warmups + args.runs):
        wall_ms, engine_ms = run_once(binary)
        if index >= args.warmups:
            wall_samples.append(wall_ms)
            engine_samples.append(engine_ms)

    report = {
        "benchmark": "four-task simulator CLI run",
        "command": "raptor run --json --sim --latency-ms 0 --no-memory --seed 1",
        "warmups": args.warmups,
        "runs": args.runs,
        "timing_notes": {
            "wall_clock": "Includes process startup, execution, and JSON output.",
            "engine": "Uses the engine's integer-millisecond wall clock; sub-millisecond values round down.",
            "scope": "Zero-latency deterministic simulator; not a model-quality or live-Ollama benchmark.",
        },
        "environment": {
            "os": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
            "python": platform.python_version(),
            "rustc": subprocess.run(
                ["rustc", "--version"], capture_output=True, text=True, check=False
            ).stdout.strip(),
            "binary": str(binary),
        },
        "wall_clock": summarize(wall_samples),
        "engine": summarize(engine_samples),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
