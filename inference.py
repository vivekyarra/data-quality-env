"""
inference.py — Baseline inference script for DataQualityEnv.

MANDATORY environment variables
--------------------------------
API_BASE_URL   The OpenAI-compatible API endpoint
MODEL_NAME     The model identifier for inference
HF_TOKEN       Your Hugging Face API token

Structured stdout output (required by Phase 2 validator)
---------------------------------------------------------
[START] task=<task_id>
[STEP] step=<n> reward=<r>
[END] task=<task_id> score=<s> steps=<n>
"""

import json
import os
import sys
from typing import Callable, Optional

from openai import OpenAI

from environment import DataQualityEnv
from models import Action, Observation

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASKS_TO_RUN = ["task1_easy", "task2_medium", "task3_hard"]

SYSTEM_PROMPT = """You are an expert data quality engineer. Your job is to clean a messy dataset.

Return ONLY a JSON object with this exact shape — no markdown, no explanation:
{"operation": "<op>", "column": "<col or null>", "params": {}}

Available operations:
- remove_duplicates  : params: {"subset": ["col1"]} or {}
- fill_missing       : REQUIRES column. params: {"strategy": "mean"|"median"|"mode"|"constant", "value": <any>}
- standardize_date   : REQUIRES column. params: {}
- standardize_phone  : REQUIRES column. params: {}
- remove_negative    : REQUIRES column. params: {}
- clip_outliers      : REQUIRES column. params: {"lower": <number>, "upper": <number>}
- done               : use when quality_score > 0.90 or no issues remain.

Strategy: fix issues in this order: duplicates → missing values → date formats → phone formats → negative values → outliers → done.
"""


def build_user_message(obs: Observation) -> str:
    issues_str   = "\n".join(f"- {i}" for i in obs.quality_issues)
    table_preview = json.dumps(obs.table[:8], indent=2, default=str)
    return (
        f"TASK: {obs.task_name}\n"
        f"DESCRIPTION: {obs.task_description}\n\n"
        f"CURRENT QUALITY SCORE: {obs.quality_score:.4f}\n"
        f"STEP: {obs.step_count}/{obs.max_steps}\n\n"
        f"QUALITY ISSUES:\n{issues_str}\n\n"
        f"COLUMN SCHEMA:\n{json.dumps(obs.column_schema, indent=2)}\n\n"
        f"CURRENT DATA (first 8 rows):\n{table_preview}\n\n"
        f"Reply with JSON only."
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


# ── Heuristic fallback policy (no LLM required) ───────────────────────────────

def heuristic_action(obs: Observation) -> Action:
    task_id     = obs.task_id
    issues_text = " ".join(obs.quality_issues).lower()

    if task_id == "task1_easy":
        if "duplicate" in issues_text:
            return Action(operation="remove_duplicates")
        if "column 'age'" in issues_text:
            return Action(operation="fill_missing", column="age",
                          params={"strategy": "median"})
        if "column 'email'" in issues_text:
            return Action(operation="fill_missing", column="email",
                          params={"strategy": "constant", "value": "unknown@example.com"})
        return Action(operation="done")

    if task_id == "task2_medium":
        if "date(s) not in yyyy-mm-dd" in issues_text:
            return Action(operation="standardize_date", column="date")
        if "phone number(s)" in issues_text:
            return Action(operation="standardize_phone", column="phone")
        if "negative value(s)" in issues_text:
            return Action(operation="remove_negative", column="amount")
        if "column 'region'" in issues_text:
            return Action(operation="fill_missing", column="region",
                          params={"strategy": "mode"})
        return Action(operation="done")

    if task_id == "task3_hard":
        if "duplicate" in issues_text:
            return Action(operation="remove_duplicates",
                          params={"subset": ["patient_id"]})
        if "column 'diagnosis'" in issues_text:
            return Action(operation="fill_missing", column="diagnosis",
                          params={"strategy": "constant", "value": "Unknown"})
        if "column 'medication'" in issues_text:
            return Action(operation="fill_missing", column="medication",
                          params={"strategy": "constant", "value": "None prescribed"})
        if "column 'dob'" in issues_text:
            return Action(operation="standardize_date", column="dob")
        if "column 'last_visit'" in issues_text:
            return Action(operation="standardize_date", column="last_visit")
        if "column 'emergency_contact'" in issues_text:
            return Action(operation="standardize_phone", column="emergency_contact")
        if "negative value(s)" in issues_text:
            return Action(operation="remove_negative", column="glucose")
        if "bp_systolic" in issues_text:
            return Action(operation="clip_outliers", column="bp_systolic",
                          params={"lower": 60, "upper": 200})
        if "bp_diastolic" in issues_text:
            return Action(operation="clip_outliers", column="bp_diastolic",
                          params={"lower": 40, "upper": 130})
        if "glucose" in issues_text and "outside valid range" in issues_text:
            return Action(operation="clip_outliers", column="glucose",
                          params={"lower": 50, "upper": 500})
        if "column 'glucose'" in issues_text and "missing" in issues_text:
            return Action(operation="fill_missing", column="glucose",
                          params={"strategy": "median"})
        return Action(operation="done")

    return Action(operation="done")


# ── LLM policy ────────────────────────────────────────────────────────────────

def make_llm_policy() -> Optional[Callable[[Observation], Action]]:
    if not API_KEY:
        return None
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    def llm_policy(obs: Observation) -> Action:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_message(obs)},
            ],
            max_tokens=120,
            temperature=0.1,
        )
        raw = response.choices[0].message.content or ""
        try:
            return parse_action(raw)
        except Exception:
            return Action(operation="done")

    return llm_policy


# ── Task runner — emits required [START]/[STEP]/[END] tokens ─────────────────

def run_task(
    env: DataQualityEnv,
    task_id: str,
    policy: Callable[[Observation], Action],
) -> dict:
    obs = env.reset(task_id)

    # ── REQUIRED: [START] block ───────────────────────────────────────────────
    print(f"[START] task={task_id}", flush=True)

    step_count     = 0
    done           = False
    last_breakdown = {}

    while not done:
        step_count += 1

        try:
            action = policy(obs)
        except Exception as exc:
            print(f"[STEP] step={step_count} reward=-0.01", flush=True)
            action = Action(operation="done")

        result = env.step(action)
        obs    = result.observation
        done   = result.done
        last_breakdown = result.reward.score_breakdown

        # ── REQUIRED: [STEP] block ────────────────────────────────────────────
        print(
            f"[STEP] step={step_count} reward={result.reward.value:.4f}",
            flush=True,
        )

    # ── REQUIRED: [END] block ─────────────────────────────────────────────────
    print(
        f"[END] task={task_id} score={obs.quality_score:.4f} steps={step_count}",
        flush=True,
    )

    return {
        "task_id":     task_id,
        "steps":       step_count,
        "final_score": obs.quality_score,
        "breakdown":   last_breakdown,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    policy      = make_llm_policy()
    policy_name = "llm" if policy is not None else "heuristic"
    if policy is None:
        policy = heuristic_action

    model_label = MODEL_NAME if policy_name == "llm" else "offline-heuristic"

    # Diagnostic header to stderr so it doesn't pollute stdout parsing
    print(f"DataQualityEnv inference | policy={policy_name} | model={model_label}",
          file=sys.stderr, flush=True)

    env     = DataQualityEnv()
    results = [run_task(env, task_id, policy) for task_id in TASKS_TO_RUN]
    overall = sum(r["final_score"] for r in results) / len(results)

    # Summary to stderr
    print(f"\nOverall average score: {overall:.4f}", file=sys.stderr, flush=True)

    with open("baseline_results.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "policy":      policy_name,
                "model":       model_label,
                "results":     results,
                "overall_avg": round(overall, 4),
            },
            fh,
            indent=2,
        )
    print("Results saved to baseline_results.json", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
