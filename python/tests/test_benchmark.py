import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "raptor_benchmark", ROOT / "scripts" / "benchmark.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def test_nearest_rank_percentiles_and_summary_are_deterministic(self):
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(benchmark.percentile(samples, 0.95), 5.0)
        self.assertEqual(
            benchmark.summarize(samples),
            {"p50_ms": 3.0, "p95_ms": 5.0, "mean_ms": 3.0, "stdev_ms": 1.581},
        )


if __name__ == "__main__":
    unittest.main()
