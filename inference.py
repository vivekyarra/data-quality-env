#!/usr/bin/env python3
"""
inference.py — Baseline inference script for DataQualityEnv.

MANDATORY environment variables
--------------------------------
API_BASE_URL   The OpenAI-compatible API endpoint
               Default: "https://router.huggingface.co/v1"
MODEL_NAME     The model identifier for inference
               Default: "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN       Your Hugging Face API token (NO default — must be set by caller)

Structured stdout output (required by Phase 2 validator)
---------------------------------------------------------
[START] task=<task_id>
[STEP]  step=<n> reward=<r>
[END]   task=<task_id> score=<s> steps=<n>

All diagnostic messages go to stderr so they never pollute the stdout stream.
"""

import json
import os
import sys
from typing import Callable, Optional

from openai import OpenAI

from environment import DataQualityEnv
from models import Action, Observation

# ── Env-var configuration ──────────────────────────────────────────────────────
# Checklist rule: defaults ONLY for API_BASE_URL and MODEL_NAME — NOT for HF_TOKEN.

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")          # No default — intentional per spec

TASKS_TO_RUN = ["task1_easy", "task2_medium", "task3_hard"]

SYSTEM_PROMPT = """\
You are an expert data quality engineer. Your job is to clean a messy dataset.

Return ONLY a JSON object with this exact shape — no markdown, no explanation:
{"operation": "<op>", "column": "<col or null>", "params": {}}

Available operations:
- remove_duplicates  : params: {"subset": ["col1"]} or {}
- fill_missing       : REQUIRES column. params: {"strategy": "mean"|"median"|"mode"|"constant", "value": <any>}
- standardize_date   : REQUIRES column. params: {}
- standardize_phone  : REQUIRES column. params: {}
- remove_negative    : REQUIRES column. params: {}
- clip_outliers      : REQUIRES column. params: {"lower": <number>, "upper": <number>}
- done               : use when quality_score >= 0.99 or no issues remain.

Strategy: fix issues in order → duplicates → missing values → dates → phones → negatives → outliers → done.
"""


# ── Message builder ────────────────────────────────────────────────────────────

def build_user_message(obs: Observation) -> str:
    issues_str    = "\n".join(f"  - {i}" for i in obs.quality_issues)
    table_preview = json.dumps(obs.table[:8], indent=2, default=str)
    return (
        f"TASK: {obs.task_name}\n"
        f"DESCRIPTION: {obs.task_description}\n\n"
        f"CURRENT QUALITY SCORE: {obs.quality_score:.4f}\n"
        f"STEP: {obs.step_count}/{obs.max_steps}\n\n"
        f"QUALITY ISSUES:\n{issues_str}\n\n"
        f"COLUMN SCHEMA:\n{json.dumps(obs.column_schema, indent=2)}\n\n"
        f"CURRENT DATA (first 8 rows):\n{table_preview}\n\n"
        f"Reply with a single JSON object only — no extra text."
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


# ── Heuristic fallback (deterministic, no LLM) ───────────────────────────────

def heuristic_action(obs: Observation) -> Action:
    """Rule-based policy that solves all three tasks deterministically."""
    task_id     = obs.task_id
    issues_text = " ".join(obs.quality_issues).lower()

    if "no quality issues detected" in issues_text:
        return Action(operation="done")

    # ---- task1_easy ----
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

    # ---- task2_medium ----
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

    # ---- task3_hard ----
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


# ── LLM policy factory ─────────────────────────────────────────────────────────

def make_llm_policy() -> Optional[Callable[[Observation], Action]]:
    """Return an LLM-backed policy if HF_TOKEN is set, else None."""
    if not HF_TOKEN:
        return None

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    def llm_policy(obs: Observation) -> Action:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_message(obs)},
                ],
                max_tokens=150,
                temperature=0.1,
            )
            raw = response.choices[0].message.content or ""
            return parse_action(raw)
        except Exception as exc:
            print(f"[LLM error] {exc} — falling back to heuristic",
                  file=sys.stderr, flush=True)
            return heuristic_action(obs)

    return llm_policy


# ── Task runner ──────────────────────────────────────────────────────────────
# CRITICAL: [START] is emitted BEFORE reset() so the validator always sees it.
# CRITICAL: every code path guarantees [END] is printed — even on exception.

def run_task(
    env: DataQualityEnv,
    task_id: str,
    policy: Callable[[Observation], Action],
) -> dict:
    """Run one full episode and emit the required [START]/[STEP]/[END] log lines."""

    # ── [START] emitted FIRST — before any operation that might throw ──────
    print(f"[START] task={task_id}", flush=True)

    step_count:  int   = 0
    final_score: float = 0.0
    last_breakdown: dict = {}

    try:
        obs         = env.reset(task_id)
        final_score = obs.quality_score
        done        = False

        while not done:
            step_count += 1

            # Decide action — both policies have internal fallbacks
            try:
                action = policy(obs)
            except Exception as exc:
                print(f"[policy error step {step_count}] {exc}",
                      file=sys.stderr, flush=True)
                action = Action(operation="done")

            # Apply action — any crash forces episode end
            try:
                result         = env.step(action)
                obs            = result.observation
                done           = result.done
                final_score    = obs.quality_score
                last_breakdown = result.reward.score_breakdown
                reward_val     = result.reward.value
            except Exception as exc:
                print(f"[step error step {step_count}] {exc}",
                      file=sys.stderr, flush=True)
                reward_val = -0.01
                done       = True  # Force episode end → guarantees [END] prints

            # ── [STEP] ─────────────────────────────────────────────────────
            print(f"[STEP] step={step_count} reward={reward_val:.4f}", flush=True)

    except Exception as exc:
        # Catch-all: handles reset() failure or any other unexpected crash
        print(f"[run_task error] {exc}", file=sys.stderr, flush=True)

    # ── [END] is ALWAYS printed — guaranteed by this structure ────────────
    print(
        f"[END] task={task_id} score={final_score:.4f} steps={step_count}",
        flush=True,
    )

    return {
        "task_id":     task_id,
        "steps":       step_count,
        "final_score": round(final_score, 4),
        "breakdown":   last_breakdown,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    llm_policy  = make_llm_policy()
    policy_name = "llm" if llm_policy is not None else "heuristic"
    policy      = llm_policy if llm_policy is not None else heuristic_action
    model_label = MODEL_NAME if policy_name == "llm" else "offline-heuristic"

    print(
        f"DataQualityEnv inference | policy={policy_name} | model={model_label}",
        file=sys.stderr, flush=True,
    )

    env     = DataQualityEnv()
    results = [run_task(env, tid, policy) for tid in TASKS_TO_RUN]
    overall = sum(r["final_score"] for r in results) / len(results)

    print(f"\nOverall average score: {overall:.4f}", file=sys.stderr, flush=True)

    output = {
        "policy":      policy_name,
        "model":       model_label,
        "results":     results,
        "overall_avg": round(overall, 4),
    }

    with open("baseline_results.json", "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    print("Results saved → baseline_results.json", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()