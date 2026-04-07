import json
import unittest

from fastapi.testclient import TestClient

from app import app


class ApiSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_schema_and_reset_do_not_leak_hidden_row_id(self) -> None:
        schema_response = self.client.get("/schema")
        reset_response = self.client.post("/reset", json={"task_id": "task3_hard"})
        tasks_response = self.client.get("/tasks")

        self.assertEqual(schema_response.status_code, 200)
        self.assertEqual(reset_response.status_code, 200)
        self.assertEqual(tasks_response.status_code, 200)

        schema_payload = schema_response.json()
        reset_payload = reset_response.json()
        tasks_payload = tasks_response.json()

        self.assertNotIn("__row_id__", json.dumps(schema_payload))
        self.assertNotIn("__row_id__", json.dumps(tasks_payload))
        self.assertNotIn("__row_id__", reset_payload["column_schema"])
        self.assertTrue(all("__row_id__" not in row for row in reset_payload["table"]))
        self.assertGreater(reset_payload["quality_score"], 0.0)
        self.assertLess(reset_payload["quality_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
