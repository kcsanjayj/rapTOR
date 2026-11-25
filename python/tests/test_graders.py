import unittest
from raptor.evals.graders import GRADERS


def run(nodes=(), events=(), plan=(), **result):
    base = {"status": "succeeded", "answer": "", "error": None, "nodes": list(nodes), "plan": {"tasks": list(plan)}, "metrics": {"repairs": 2}}
    base.update(result)
    return {"result": base, "events": list(events), "elapsed_ms": 10}


class GraderTests(unittest.TestCase):
    def test_status(self):
        self.assertTrue(GRADERS["status"](run(), {"is": "succeeded"})[0])
        self.assertFalse(GRADERS["status"](run(), {"is": "failed"})[0])
        self.assertTrue(GRADERS["status"](run(), {"is": ["partial", "succeeded"]})[0])

    def test_data_flow_detects_missing_inputs(self):
        plan = [{"id": "a", "dependencies": [], "kind": "llm"}, {"id": "b", "dependencies": ["a"], "kind": "llm"}]
        good = [{"kind": "node_started", "node": "a", "seq": 0, "data": {"inputs_from": [], "input_chars": 0}},
                {"kind": "node_started", "node": "b", "seq": 1, "data": {"inputs_from": ["a"], "input_chars": 40}}]
        bad = [dict(good[0]), {"kind": "node_started", "node": "b", "seq": 1, "data": {"inputs_from": ["a"], "input_chars": 0}}]
        self.assertTrue(GRADERS["data_flow"](run(plan=plan, events=good), {})[0])
        self.assertFalse(GRADERS["data_flow"](run(plan=plan, events=bad), {})[0])

    def test_critic_coverage_ignores_tool_nodes(self):
        plan = [{"id": "t", "dependencies": [], "kind": "tool"}, {"id": "l", "dependencies": [], "kind": "llm"}]
        nodes = [{"id": "t", "status": "completed", "reviewed": False}, {"id": "l", "status": "completed", "reviewed": True}]
        self.assertTrue(GRADERS["critic_coverage"](run(nodes=nodes, plan=plan), {"min": 1.0})[0])
        nodes[1]["reviewed"] = False
        self.assertFalse(GRADERS["critic_coverage"](run(nodes=nodes, plan=plan), {"min": 1.0})[0])

    def test_ordering_grader_catches_early_start(self):
        ev = [{"kind": "node_started", "node": "b", "seq": 1, "data": {"inputs_from": ["a"]}}, {"kind": "node_completed", "node": "a", "seq": 2, "data": {}}]
        self.assertFalse(GRADERS["no_unreviewed_dependents"](run(events=ev), {})[0])

    def test_metric_bounds_and_answer_checks(self):
        self.assertTrue(GRADERS["metric"](run(), {"name": "repairs", "gte": 2, "lte": 2})[0])
        self.assertFalse(GRADERS["metric"](run(), {"name": "repairs", "gte": 3})[0])
        self.assertTrue(GRADERS["answer_contains"](run(answer="The result is 84"), {"all": ["84"]})[0])
        self.assertFalse(GRADERS["answer_not_contains"](run(answer="uses Python"), {"any": ["python"]})[0])


if __name__ == "__main__":
    unittest.main()
