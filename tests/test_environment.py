import unittest
from unittest.mock import patch

import pandas as pd

from environment import TASK_SCORE_EPS, DataQualityEnv
from inference import heuristic_action
from models import Action
from tasks import TASKS, make_row_id


def build_target_df(task_id: str) -> pd.DataFrame:
    task = TASKS[task_id]
    rows = []
    for source_row, row in zip(task["target_source_rows"], task["target_data"]):
        payload = dict(row)
        payload["__row_id__"] = make_row_id(task_id, source_row)
        rows.append(payload)
    return pd.DataFrame(rows)


class EnvironmentBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = DataQualityEnv()

    def test_reset_hides_hidden_row_id(self) -> None:
        obs = self.env.reset("task1_easy")
        state = self.env.state()

        self.assertTrue(TASK_SCORE_EPS <= obs.quality_score <= 1.0 - TASK_SCORE_EPS)
        self.assertNotIn("__row_id__", obs.column_schema)
        self.assertTrue(all("__row_id__" not in row for row in obs.table))
        self.assertEqual(state["table_shape"], [15, 5])

    def test_exact_target_match_exposes_open_interval_max(self) -> None:
        self.env.reset("task3_hard")
        self.env._df = build_target_df("task3_hard")

        self.assertEqual(self.env._compute_score(), 0.9999)
        self.assertEqual(self.env._get_score_breakdown()["total"], 0.9999)
        self.assertEqual(self.env.state()["quality_score"], 0.9999)

    def test_zero_total_is_exposed_as_open_interval_min(self) -> None:
        with patch("environment.grade", return_value={"total": 0.0, "row_fidelity_score": 0.0}):
            obs = self.env.reset("task1_easy")
            result = self.env.step(Action(operation="done"))
            state = self.env.state()

        self.assertEqual(obs.quality_score, 0.0001)
        self.assertEqual(result.observation.quality_score, 0.0001)
        self.assertEqual(result.reward.score_breakdown["total"], 0.0001)
        self.assertEqual(result.info["quality_score"], 0.0001)
        self.assertEqual(state["quality_score"], 0.0001)

    def test_task2_deletion_exploit_no_longer_solves_cleanup(self) -> None:
        self.env.reset("task2_medium")
        result = self.env.step(Action(operation="remove_negative", column="amount"))

        self.assertLess(result.observation.quality_score, 0.5)
        self.assertEqual(result.reward.score_breakdown["date_match_score"], 0.0)
        self.assertEqual(result.reward.score_breakdown["phone_match_score"], 0.0)
        self.assertLess(result.reward.score_breakdown["row_fidelity_score"], 1.0)

    def test_task3_wrong_order_cannot_reach_full_score(self) -> None:
        actions = [
            Action(operation="remove_duplicates", params={"subset": ["patient_name", "dob", "visit_date"]}),
            Action(operation="standardize_date", column="dob"),
            Action(operation="standardize_date", column="visit_date"),
            Action(
                operation="fill_missing",
                column="currency",
                params={
                    "strategy": "mapping",
                    "source_column": "country",
                    "mapping": {
                        "India": "INR",
                        "United States": "USD",
                        "United Kingdom": "GBP",
                    },
                },
            ),
            Action(
                operation="fill_missing",
                column="medication",
                params={
                    "strategy": "mapping",
                    "source_column": "diagnosis",
                    "mapping": {
                        "Diabetes": "Metformin",
                        "Hypertension": "Amlodipine",
                        "Hypothyroidism": "Levothyroxine",
                    },
                },
            ),
            Action(operation="standardize_phone", column="emergency_contact"),
            Action(operation="clip_outliers", column="bp_systolic", params={"lower": 60, "upper": 200}),
            Action(operation="clip_outliers", column="bp_diastolic", params={"lower": 40, "upper": 130}),
            Action(operation="clip_outliers", column="glucose", params={"lower": 50, "upper": 500}),
            Action(operation="done"),
        ]

        final_score = self._run_sequence("task3_hard", actions)
        self.assertLess(final_score, 0.9999)

    def test_task3_mapping_strategy_beats_constant_fill(self) -> None:
        correct_actions = [
            Action(operation="standardize_date", column="dob"),
            Action(operation="standardize_date", column="visit_date"),
            Action(operation="remove_duplicates", params={"subset": ["patient_name", "dob", "visit_date"]}),
            Action(
                operation="fill_missing",
                column="currency",
                params={
                    "strategy": "mapping",
                    "source_column": "country",
                    "mapping": {
                        "India": "INR",
                        "United States": "USD",
                        "United Kingdom": "GBP",
                    },
                },
            ),
            Action(
                operation="fill_missing",
                column="medication",
                params={
                    "strategy": "mapping",
                    "source_column": "diagnosis",
                    "mapping": {
                        "Diabetes": "Metformin",
                        "Hypertension": "Amlodipine",
                        "Hypothyroidism": "Levothyroxine",
                    },
                },
            ),
            Action(operation="standardize_phone", column="emergency_contact"),
            Action(operation="clip_outliers", column="bp_systolic", params={"lower": 60, "upper": 200}),
            Action(operation="clip_outliers", column="bp_diastolic", params={"lower": 40, "upper": 130}),
            Action(operation="clip_outliers", column="glucose", params={"lower": 50, "upper": 500}),
            Action(operation="done"),
        ]
        constant_actions = [
            Action(operation="standardize_date", column="dob"),
            Action(operation="standardize_date", column="visit_date"),
            Action(operation="remove_duplicates", params={"subset": ["patient_name", "dob", "visit_date"]}),
            Action(operation="fill_missing", column="currency", params={"strategy": "constant", "value": "USD"}),
            Action(operation="fill_missing", column="medication", params={"strategy": "constant", "value": "Metformin"}),
            Action(operation="standardize_phone", column="emergency_contact"),
            Action(operation="clip_outliers", column="bp_systolic", params={"lower": 60, "upper": 200}),
            Action(operation="clip_outliers", column="bp_diastolic", params={"lower": 40, "upper": 130}),
            Action(operation="clip_outliers", column="glucose", params={"lower": 50, "upper": 500}),
            Action(operation="done"),
        ]

        mapped_score = self._run_sequence("task3_hard", correct_actions)
        constant_score = self._run_sequence("task3_hard", constant_actions)

        self.assertEqual(mapped_score, 0.9999)
        self.assertLess(constant_score, mapped_score)

    def test_hallucinated_mapping_keys_are_ignored_without_crashing(self) -> None:
        self.env.reset("task3_hard")
        self.env.step(Action(operation="standardize_date", column="dob"))
        self.env.step(Action(operation="standardize_date", column="visit_date"))
        self.env.step(Action(operation="remove_duplicates", params={"subset": ["patient_name", "dob", "visit_date"]}))

        result = self.env.step(
            Action(
                operation="fill_missing",
                column="currency",
                params={
                    "strategy": "mapping",
                    "source_column": "country",
                    "mapping": {
                        "India": "INR",
                        "United States": "USD",
                        "United Kingdom": "GBP",
                        "FakeCountry": "CHF",
                    },
                },
            )
        )

        self.assertIn("strategy='mapping'", result.reward.message)
        self.assertGreater(result.observation.quality_score, 0.0)

    def test_heuristic_baseline_is_honest_on_task3(self) -> None:
        task1 = self._run_episode("task1_easy")
        task2 = self._run_episode("task2_medium")
        task3 = self._run_episode("task3_hard")

        self.assertEqual(task1["final_score"], 0.9999)
        self.assertEqual(task2["final_score"], 0.9999)
        self.assertLess(task3["final_score"], 0.8)
        self.assertEqual(task3["breakdown"]["currency_match_score"], 0.0)
        self.assertEqual(task3["breakdown"]["medication_match_score"], 0.0)

    def _run_sequence(self, task_id: str, actions: list[Action]) -> float:
        self.env.reset(task_id)
        result = None
        for action in actions:
            result = self.env.step(action)
        self.assertIsNotNone(result)
        return round(result.observation.quality_score, 4)

    def _run_episode(self, task_id: str) -> dict:
        obs = self.env.reset(task_id)
        result = None

        while True:
            action = heuristic_action(obs)
            result = self.env.step(action)
            obs = result.observation
            if result.done:
                break

        self.assertIsNotNone(result)
        return {
            "final_score": round(result.observation.quality_score, 4),
            "breakdown": result.reward.score_breakdown,
        }


if __name__ == "__main__":
    unittest.main()
