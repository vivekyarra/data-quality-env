#!/usr/bin/env python3
"""
Baseline inference script for DataQualityEnv.

Structured stdout output (required by the validator):
[START] task=<task_id>
[STEP] step=<n> reward=<r>
[END] task=<task_id> score=<s> steps=<n>
"""

from __future__ import annotations

import json
import os
import sys
from typing import Callable, Optional

from openai import OpenAI

from environment import DataQualityEnv
from models import Action, Observation

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = HF_TOKEN or OPENAI_API_KEY

TASKS_TO_RUN = ["task1_easy", "task2_medium", "task3_hard"]

SYSTEM_PROMPT = """\
You are an expert data quality engineer. Clean the dataset and return only JSON.

Return ONLY a JSON object with this shape:
{"operation": "<op>", "column": "<col or null>", "params": {}}

Available operations:
- remove_duplicates: params {"subset": ["col1"]} or {}
- fill_missing: REQUIRES column. params:
  - {"strategy": "mean"}
  - {"strategy": "median"}
  - {"strategy": "mode"}
  - {"strategy": "constant", "value": <any>}
  - {"strategy": "mapping", "source_column": "<col>", "mapping": {"A": "B"}}
- standardize_date: REQUIRES column. params {}
- standardize_phone: REQUIRES column. params {}
- remove_negative: REQUIRES column. params {}
- clip_outliers: REQUIRES column. params {"lower": <number>, "upper": <number>}
- done: use when no more meaningful progress is possible.

Fix issues in order: dates, deduplication, targeted imputations, phone cleanup,
vitals cleanup, then done.
"""


def build_user_message(obs: Observation) -> str:
    issues_text = "\n".join(f"  - {issue}" for issue in obs.quality_issues)
    table_preview = json.dumps(obs.table[:8], indent=2, default=str)
    return (
        f"TASK: {obs.task_name}\n"
        f"DESCRIPTION: {obs.task_description}\n\n"
        f"CURRENT QUALITY SCORE: {obs.quality_score:.4f}\n"
        f"STEP: {obs.step_count}/{obs.max_steps}\n\n"
        f"QUALITY ISSUES:\n{issues_text}\n\n"
        f"COLUMN SCHEMA:\n{json.dumps(obs.column_schema, indent=2)}\n\n"
        f"CURRENT DATA (first 8 rows):\n{table_preview}\n\n"
        "Reply with a single JSON object only."
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
    """
    Deterministic baseline.

    The heuristic intentionally does not solve Task 3's contextual mapping
    requirements. That gap is part of the benchmark story.
    """

    task_id = obs.task_id
    issues_text = " ".join(obs.quality_issues).lower()

    if "no quality issues detected" in issues_text:
        return Action(operation="done")

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
        if "date(s) not in yyyy-mm-dd" in issues_text:
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
        if "column 'dob'" in issues_text:
            return Action(operation="standardize_date", column="dob")
        if "column 'visit_date'" in issues_text:
            return Action(operation="standardize_date", column="visit_date")
        if "duplicate rows detected" in issues_text:
            return Action(
                operation="remove_duplicates",
                params={"subset": ["patient_name", "dob", "visit_date"]},
            )
        if "column 'emergency_contact'" in issues_text:
            return Action(operation="standardize_phone", column="emergency_contact")
        if "column 'bp_systolic'" in issues_text:
            return Action(
                operation="clip_outliers",
                column="bp_systolic",
                params={"lower": 60, "upper": 200},
            )
        if "column 'bp_diastolic'" in issues_text:
            return Action(
                operation="clip_outliers",
                column="bp_diastolic",
                params={"lower": 40, "upper": 130},
            )
        if "column 'glucose'" in issues_text and (
            "negative value(s)" in issues_text or "outside valid range" in issues_text
        ):
            return Action(
                operation="clip_outliers",
                column="glucose",
                params={"lower": 50, "upper": 500},
            )
        return Action(operation="done")

    return Action(operation="done")


def make_llm_policy() -> Optional[Callable[[Observation], Action]]:
    if not API_KEY:
        return None

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    def llm_policy(obs: Observation) -> Action:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(obs)},
                ],
                max_tokens=180,
                temperature=0.1,
            )
            raw = response.choices[0].message.content or ""
            return parse_action(raw)
        except Exception as exc:
            print(f"[LLM error] {exc} - falling back to heuristic", file=sys.stderr, flush=True)
            return heuristic_action(obs)

    return llm_policy


def run_task(
    env: DataQualityEnv,
    task_id: str,
    policy: Callable[[Observation], Action],
) -> dict:
    print(f"[START] task={task_id}", flush=True)

    step_count = 0
    final_score = 0.0001
    last_breakdown: dict = {}

    try:
        obs = env.reset(task_id)
        final_score = obs.quality_score
        done = False

        while not done:
            step_count += 1

            try:
                action = policy(obs)
            except Exception as exc:
                print(f"[policy error step {step_count}] {exc}", file=sys.stderr, flush=True)
                action = Action(operation="done")

            try:
                result = env.step(action)
                obs = result.observation
                done = result.done
                final_score = obs.quality_score
                last_breakdown = result.reward.score_breakdown
                reward_value = result.reward.value
            except Exception as exc:
                print(f"[step error step {step_count}] {exc}", file=sys.stderr, flush=True)
                reward_value = -0.01
                done = True

            print(f"[STEP] step={step_count} reward={reward_value:.4f}", flush=True)

    except Exception as exc:
        print(f"[run_task error] {exc}", file=sys.stderr, flush=True)

    print(f"[END] task={task_id} score={final_score:.4f} steps={step_count}", flush=True)

    return {
        "task_id": task_id,
        "steps": step_count,
        "final_score": round(final_score, 4),
        "breakdown": last_breakdown,
    }


def main() -> None:
    llm_policy = make_llm_policy()
    policy_name = "llm" if llm_policy is not None else "heuristic"
    policy = llm_policy if llm_policy is not None else heuristic_action
    model_label = MODEL_NAME if policy_name == "llm" else "offline-heuristic"

    print(
        f"DataQualityEnv inference | policy={policy_name} | model={model_label}",
        file=sys.stderr,
        flush=True,
    )

    env = DataQualityEnv()
    results = [run_task(env, task_id, policy) for task_id in TASKS_TO_RUN]
    overall = round(sum(item["final_score"] for item in results) / len(results), 4)

    print(f"\nOverall average score: {overall:.4f}", file=sys.stderr, flush=True)

    payload = {
        "policy": policy_name,
        "model": model_label,
        "results": results,
        "overall_avg": overall,
    }
    with open("baseline_results.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("Results saved -> baseline_results.json", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
