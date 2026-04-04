"""
Baseline inference script for DataQualityEnv.

This script supports two modes:
1. Heuristic mode, which runs fully offline and requires no API key.
2. LLM mode, which uses an OpenAI-compatible endpoint when credentials exist.
"""

import json
import os
from typing import Callable, Optional

from openai import OpenAI

from environment import DataQualityEnv
from models import Action, Observation

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASKS_TO_RUN = ["task1_easy", "task2_medium", "task3_hard"]

SYSTEM_PROMPT = """You are an expert data quality engineer.
Return only a JSON object with this shape:
{"operation": "<op>", "column": "<col or null>", "params": {}}
"""


def build_user_message(obs: Observation) -> str:
    issues_str = "\n".join(f"- {issue}" for issue in obs.quality_issues)
    table_preview = json.dumps(obs.table[:8], indent=2, default=str)
    return (
        f"TASK: {obs.task_name}\n"
        f"DESCRIPTION: {obs.task_description}\n\n"
        f"CURRENT QUALITY SCORE: {obs.quality_score:.4f}\n"
        f"STEP: {obs.step_count}/{obs.max_steps}\n\n"
        f"QUALITY ISSUES:\n{issues_str}\n\n"
        f"COLUMN SCHEMA:\n{json.dumps(obs.column_schema, indent=2)}\n\n"
        f"CURRENT DATA (first 8 rows):\n{table_preview}\n"
    )


def parse_action(content: str) -> Action:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines() if not line.startswith("```")
        ).strip()
    payload = json.loads(cleaned)
    return Action(
        operation=payload.get("operation", "done"),
        column=payload.get("column"),
        params=payload.get("params", {}),
    )


def heuristic_action(obs: Observation) -> Action:
    task_id = obs.task_id
    issues_text = " ".join(obs.quality_issues).lower()

    if task_id == "task1_easy":
        if "duplicate" in issues_text:
            return Action(operation="remove_duplicates")
        if "column 'age'" in issues_text:
            return Action(
                operation="fill_missing",
                column="age",
                params={"strategy": "median"},
            )
        if "column 'email'" in issues_text:
            return Action(
                operation="fill_missing",
                column="email",
                params={"strategy": "constant", "value": "unknown@example.com"},
            )
        return Action(operation="done")

    if task_id == "task2_medium":
        if "date(s) not in yyyy-mm-dd format" in issues_text:
            return Action(operation="standardize_date", column="date")
        if "phone number(s)" in issues_text:
            return Action(operation="standardize_phone", column="phone")
        if "negative value(s)" in issues_text:
            return Action(operation="remove_negative", column="amount")
        if "column 'region'" in issues_text:
            return Action(
                operation="fill_missing",
                column="region",
                params={"strategy": "mode"},
            )
        return Action(operation="done")

    if task_id == "task3_hard":
        if "duplicate" in issues_text:
            return Action(
                operation="remove_duplicates",
                params={"subset": ["patient_id"]},
            )
        if "column 'diagnosis'" in issues_text:
            return Action(
                operation="fill_missing",
                column="diagnosis",
                params={"strategy": "mode"},
            )
        if "column 'medication'" in issues_text:
            return Action(
                operation="fill_missing",
                column="medication",
                params={"strategy": "mode"},
            )
        if "column 'dob'" in issues_text:
            return Action(operation="standardize_date", column="dob")
        if "column 'last_visit'" in issues_text:
            return Action(operation="standardize_date", column="last_visit")
        if "negative value(s)" in issues_text:
            return Action(operation="remove_negative", column="glucose")
        if "bp_systolic" in issues_text:
            return Action(operation="clip_outliers", column="bp_systolic")
        if "bp_diastolic" in issues_text:
            return Action(operation="clip_outliers", column="bp_diastolic")
        if "glucose" in issues_text:
            return Action(operation="clip_outliers", column="glucose")
        return Action(operation="done")

    return Action(operation="done")


def make_llm_policy() -> Optional[Callable[[Observation], Action]]:
    if not API_KEY:
        return None

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    def llm_policy(obs: Observation) -> Action:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(obs)},
            ],
            max_tokens=120,
            temperature=0.1,
        )
        raw = response.choices[0].message.content or ""
        return parse_action(raw)

    return llm_policy


def run_task(env: DataQualityEnv, task_id: str, policy: Callable[[Observation], Action]) -> dict:
    print(f"\n{'=' * 60}")
    print(f"TASK: {task_id}")
    print(f"{'=' * 60}")

    obs = env.reset(task_id)
    print(f"Initial quality score : {obs.quality_score:.4f}")
    print(f"Initial issues        : {len(obs.quality_issues)}")

    step_count = 0
    done = False
    last_breakdown = {}

    while not done:
        step_count += 1
        try:
            action = policy(obs)
        except Exception as exc:
            print(f"  [POLICY ERROR] {exc}")
            action = Action(operation="done")

        result = env.step(action)
        obs = result.observation
        done = result.done
        last_breakdown = result.reward.score_breakdown

        print(
            f"  Step {step_count:02d} | op={action.operation:<18} "
            f"col={str(action.column):<14} reward={result.reward.value:+.4f} "
            f"score={obs.quality_score:.4f}"
        )

    print(f"\n  Final quality score : {obs.quality_score:.4f}")
    print(f"  Score breakdown     : {last_breakdown}")
    return {
        "task_id": task_id,
        "steps": step_count,
        "final_score": obs.quality_score,
        "breakdown": last_breakdown,
    }


def main() -> None:
    policy = make_llm_policy()
    policy_name = "llm" if policy is not None else "heuristic"
    if policy is None:
        policy = heuristic_action
    model_label = MODEL_NAME if policy_name == "llm" else "offline-heuristic"

    print("DataQualityEnv baseline inference")
    print(f"Policy     : {policy_name}")
    print(f"Model      : {model_label}")
    print(f"API base   : {API_BASE_URL}")

    env = DataQualityEnv()
    results = [run_task(env, task_id, policy) for task_id in TASKS_TO_RUN]

    overall = sum(result["final_score"] for result in results) / len(results)
    print(f"\nOverall average score : {overall:.4f}")

    with open("baseline_results.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"policy": policy_name, "model": model_label, "results": results, "overall_avg": overall},
            handle,
            indent=2,
        )
    print("Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
