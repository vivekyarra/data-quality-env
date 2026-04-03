"""
inference.py — Baseline inference script for DataQualityEnv
=============================================================

MANDATORY environment variables
--------------------------------
API_BASE_URL   The OpenAI-compatible API endpoint  (e.g. https://router.huggingface.co/v1)
MODEL_NAME     The model identifier for inference   (e.g. Qwen/Qwen2.5-72B-Instruct)
HF_TOKEN       Your Hugging Face API token

Usage
-----
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
    export HF_TOKEN="hf_..."
    python inference.py
"""

import json
import os
import sys

from openai import OpenAI

from environment import DataQualityEnv
from models import Action

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

if not API_KEY:
    print("[ERROR] HF_TOKEN or API_KEY environment variable not set.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

TASKS_TO_RUN = ["task1_easy", "task2_medium", "task3_hard"]

# ── LLM Prompt Templates ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert data quality engineer. Your job is to clean a messy dataset by applying one transformation operation at a time.

AVAILABLE OPERATIONS:
- remove_duplicates : params: {"subset": ["col1","col2"]} or empty {}
- fill_missing      : REQUIRES column. params: {"strategy": "mean"|"median"|"mode"|"constant", "value": <any>}
- standardize_date  : REQUIRES column. params: {}
- standardize_phone : REQUIRES column. params: {}
- remove_negative   : REQUIRES column. params: {}
- clip_outliers     : REQUIRES column. params: {"lower": <number>, "upper": <number>}
- done              : Signal episode end. Use when score > 0.9 or no issues remain.

RULES:
1. Respond with ONLY a valid JSON object — no markdown, no preamble, no explanation.
2. Format: {"operation": "<op>", "column": "<col or null>", "params": {}}
3. Fix the most impactful issue first. Prioritise duplicates → missing values → format issues → outliers.
4. Call "done" when quality_score > 0.90 or the issues list is empty.
5. Never repeat a no-op action (one that does not change the score).

EXAMPLES:
{"operation": "remove_duplicates", "column": null, "params": {}}
{"operation": "fill_missing", "column": "age", "params": {"strategy": "median"}}
{"operation": "standardize_date", "column": "date", "params": {}}
{"operation": "standardize_phone", "column": "phone", "params": {}}
{"operation": "remove_negative", "column": "amount", "params": {}}
{"operation": "clip_outliers", "column": "bp_systolic", "params": {"lower": 60, "upper": 200}}
{"operation": "done", "column": null, "params": {}}
"""


def build_user_message(obs) -> str:
    issues_str = "\n".join(f"  - {i}" for i in obs.quality_issues)
    table_preview = json.dumps(obs.table[:8], indent=2, default=str)
    return f"""TASK: {obs.task_name}
DESCRIPTION: {obs.task_description}

CURRENT QUALITY SCORE: {obs.quality_score:.4f}  (step {obs.step_count}/{obs.max_steps})

QUALITY ISSUES DETECTED:
{issues_str}

COLUMN SCHEMA:
{json.dumps(obs.column_schema, indent=2)}

CURRENT DATA (first 8 rows):
{table_preview}

What single operation should you apply next? Respond with JSON only."""


# ── Agent Loop ────────────────────────────────────────────────────────────────

def parse_action(content: str) -> Action:
    """Extract a valid Action from LLM response text."""
    # Strip markdown fences if present
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            l for l in lines if not l.startswith("```")
        ).strip()
    data = json.loads(cleaned)
    return Action(
        operation=data.get("operation", "done"),
        column=data.get("column"),
        params=data.get("params", {}),
    )


def run_task(env: DataQualityEnv, task_id: str) -> dict:
    print(f"\n{'='*60}")
    print(f"TASK: {task_id}")
    print(f"{'='*60}")

    obs = env.reset(task_id)
    print(f"Initial quality score : {obs.quality_score:.4f}")
    print(f"Initial issues        : {len(obs.quality_issues)}")

    step = 0
    done = False
    final_score = obs.quality_score

    while not done:
        step += 1
        user_msg = build_user_message(obs)

        # Call the LLM
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=120,
                temperature=0.1,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [LLM ERROR] {exc}")
            break

        # Parse action
        try:
            action = parse_action(raw)
        except Exception as exc:
            print(f"  [PARSE ERROR] Could not parse '{raw[:80]}': {exc}")
            action = Action(operation="done")

        # Apply to environment
        result = env.step(action)
        final_score = result.observation.quality_score

        print(
            f"  Step {step:02d} | op={action.operation:<20} col={str(action.column):<18} "
            f"reward={result.reward.value:+.4f}  score={final_score:.4f}"
        )

        done = result.done
        obs  = result.observation

    print(f"\n  Final quality score : {final_score:.4f}")
    print(f"  Score breakdown     : {result.reward.score_breakdown}")  # type: ignore[possibly-undefined]
    return {
        "task_id":     task_id,
        "steps":       step,
        "final_score": final_score,
        "breakdown":   result.reward.score_breakdown,  # type: ignore[possibly-undefined]
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"DataQualityEnv — Baseline Inference")
    print(f"Model      : {MODEL_NAME}")
    print(f"API base   : {API_BASE_URL}")

    env = DataQualityEnv()
    results = []

    for task_id in TASKS_TO_RUN:
        res = run_task(env, task_id)
        results.append(res)

    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['task_id']:<18}  score={r['final_score']:.4f}  steps={r['steps']}")

    overall = sum(r["final_score"] for r in results) / len(results)
    print(f"\n  Overall average score : {overall:.4f}")

    # Write results to JSON for reproducibility
    with open("baseline_results.json", "w") as f:
        json.dump(
            {"model": MODEL_NAME, "results": results, "overall_avg": overall},
            f, indent=2
        )
    print("\n  Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
