import json
import re
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class InferenceScriptTests(unittest.TestCase):
    def test_inference_stdout_matches_required_structure(self) -> None:
        proc = subprocess.run(
            [sys.executable, "inference.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
        stderr_text = proc.stderr

        self.assertEqual(len([line for line in stdout_lines if line.startswith("[START]")]), 3)
        self.assertEqual(len([line for line in stdout_lines if line.startswith("[END]")]), 3)
        self.assertTrue(all(self._is_structured_log(line) for line in stdout_lines))
        self.assertIn("Overall average score: 0.8333", stderr_text)

        results = json.loads((ROOT / "baseline_results.json").read_text(encoding="utf-8"))
        self.assertEqual(results["policy"], "heuristic")
        self.assertEqual(results["model"], "offline-heuristic")
        self.assertEqual(len(results["results"]), 3)
        self.assertGreater(results["overall_avg"], 0.0)
        self.assertLess(results["overall_avg"], 1.0)

        task1, task2, task3 = results["results"]
        self.assertEqual(task1["task_id"], "task1_easy")
        self.assertEqual(task2["task_id"], "task2_medium")
        self.assertEqual(task3["task_id"], "task3_hard")

        self.assertEqual(task1["final_score"], 0.9999)
        self.assertEqual(task2["final_score"], 0.9999)
        self.assertEqual(task3["final_score"], 0.5)
        self.assertLess(task3["final_score"], task2["final_score"])
        self.assertEqual(task3["breakdown"]["currency_match_score"], 0.0)
        self.assertEqual(task3["breakdown"]["medication_match_score"], 0.0)

    @staticmethod
    def _is_structured_log(line: str) -> bool:
        patterns = [
            r"^\[START\] task=[A-Za-z0-9_]+$",
            r"^\[STEP\] step=\d+ reward=-?\d+\.\d{4}$",
            r"^\[END\] task=[A-Za-z0-9_]+ score=\d+\.\d{4} steps=\d+$",
        ]
        return any(re.match(pattern, line) for pattern in patterns)


if __name__ == "__main__":
    unittest.main()
