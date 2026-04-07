# DataQualityEnv

An OpenEnv-compatible benchmark for agentic data cleaning on realistic tabular workflows.

`DataQualityEnv` asks an agent to repair messy tables through a standard `reset()` / `step()` / `state()` loop. The benchmark covers CRM cleanup, sales-data normalization, and a chained healthcare-billing task that requires ordering, context, and precision rather than brute-force row deletion.

## Why This Benchmark Matters

Data cleaning is one of the highest-cost parts of modern analytics and ML work. Teams routinely lose days to duplicate records, broken dates, malformed phones, missing labels, and invalid numeric values before any model training even starts.

This environment is built to evaluate whether an agent can:
- preserve valid data while fixing bad data
- choose the right operation for the right column
- plan multi-step cleanup sequences across a trajectory
- avoid gaming the score by deleting rows

That makes it useful for agent evaluation, RL training, and tool-use benchmarking in a domain people actually care about.

## What Makes It Hard To Game

The grader does not reward "fewer visible issues" by itself.

It uses three guardrails:
- Hidden target outputs: each task has an internal target dataframe that is never exposed to the agent.
- Stable hidden row ids: every source row gets an internal `__row_id__` at `reset()` time, so grading matches rows by identity instead of dataframe index.
- Row-fidelity cap: if precision or recall drops below the threshold, the total score is capped by row fidelity. Deleting lots of rows tanks the score.

The result is simple: a lazy strategy like `remove_negative` or `clip_outliers` on the wrong column cannot win by shrinking the table.

## Environment Mechanics

The agent sees:
- the current table
- the schema
- a human-readable list of remaining quality issues
- a task-level `quality_score`
- the available operations

The environment supports these operations:
- `remove_duplicates`
- `fill_missing`
- `standardize_date`
- `standardize_phone`
- `remove_negative`
- `clip_outliers`
- `done`

Reward is dense:

```text
reward(t) = exposed_quality_score(t) - exposed_quality_score(t-1) - 0.01
```

Raw grader totals are computed in `[0.0, 1.0]`, but every exposed task total is forced into the strict open interval `(0.0001, 0.9999)`. That rule is applied consistently in observations, state, step info, reward breakdown totals, and the saved baseline results.

## Tasks

| Task | Domain | Difficulty | Input Rows | Core Challenge |
| --- | --- | --- | ---: | --- |
| `task1_easy` | CRM records | Easy | 15 | Deduplicate customer rows, then fill missing age and email values without losing valid rows |
| `task2_medium` | Sales operations | Medium | 15 | Standardize dates and phones, remove only invalid negative transactions, and fill missing regions |
| `task3_hard` | Healthcare billing | Hard | 16 | Chained reasoning across date normalization, deduplication, contextual mapping, phone cleanup, and vital-sign repair |

### Task 1: Customer Records Deduplication and Imputation

The table contains duplicate customers plus missing `age` and `email` values. A strong agent should remove duplicates first, then restore the missing fields.

### Task 2: Sales Data Standardization and Cleansing

The table mixes ISO and non-ISO dates, malformed Indian phone numbers, negative transaction rows that should not survive, and missing `region` values. The benchmark punishes accidental data loss, so the agent has to fix the formatting issues while preserving legitimate rows.

### Task 3: Healthcare Billing Reconciliation and Quality Repair

This task is intentionally harder than the first two.

The public schema is:
- `patient_name`
- `dob`
- `visit_date`
- `country`
- `currency`
- `diagnosis`
- `medication`
- `emergency_contact`
- `glucose`
- `bp_systolic`
- `bp_diastolic`

Full-credit behavior requires effective sequencing:
1. standardize `dob`
2. standardize `visit_date`
3. remove duplicates using `['patient_name', 'dob', 'visit_date']`
4. fill missing `currency` from `country`
5. fill missing `medication` from `diagnosis`
6. standardize `emergency_contact`
7. clip invalid vitals

The duplicates are encoded so they are only fully visible after date normalization. The contextual fills also matter: the correct `currency` depends on `country`, and the correct `medication` depends on `diagnosis`.

That is why the deterministic heuristic baseline succeeds on Tasks 1 and 2 but clearly underperforms on Task 3.

## Action Space

Actions use this shape:

```json
{
  "operation": "<op>",
  "column": "<col or null>",
  "params": {}
}
```

| Operation | Column | Params | Example |
| --- | --- | --- | --- |
| `remove_duplicates` | Optional | `subset` list | `{"operation": "remove_duplicates", "params": {"subset": ["patient_name", "dob", "visit_date"]}}` |
| `fill_missing` | Required | `strategy` plus optional strategy-specific params | `{"operation": "fill_missing", "column": "currency", "params": {"strategy": "mapping", "source_column": "country", "mapping": {"India": "INR"}}}` |
| `standardize_date` | Required | none | `{"operation": "standardize_date", "column": "dob"}` |
| `standardize_phone` | Required | none | `{"operation": "standardize_phone", "column": "emergency_contact"}` |
| `remove_negative` | Required | none | `{"operation": "remove_negative", "column": "amount"}` |
| `clip_outliers` | Required | `lower`, `upper` | `{"operation": "clip_outliers", "column": "glucose", "params": {"lower": 50, "upper": 500}}` |
| `done` | No | none | `{"operation": "done"}` |

`fill_missing` strategies:
- `mean`
- `median`
- `mode`
- `constant`
- `mapping`

The `mapping` strategy fills only rows where the target column is missing. It reads a value from `source_column`, applies mappings that actually match present source values, and ignores hallucinated keys without crashing.

## Grading

Each task uses hidden target-based grading, not issue counting.

### Task 1 weights

| Component | Weight |
| --- | ---: |
| `row_fidelity_score` | 0.40 |
| `age_match_score` | 0.30 |
| `email_match_score` | 0.30 |

### Task 2 weights

| Component | Weight |
| --- | ---: |
| `row_fidelity_score` | 0.35 |
| `date_match_score` | 0.20 |
| `phone_match_score` | 0.20 |
| `amount_match_score` | 0.15 |
| `region_match_score` | 0.10 |

### Task 3 weights

| Component | Weight |
| --- | ---: |
| `row_fidelity_score` | 0.35 |
| `date_match_score` | 0.15 |
| `phone_match_score` | 0.10 |
| `currency_match_score` | 0.15 |
| `medication_match_score` | 0.10 |
| `vitals_match_score` | 0.15 |

`row_fidelity_score` is computed from precision/recall F1 on stable hidden row ids. If row precision or recall falls below the threshold, the total score is capped by row fidelity.

## Baseline Reproduction

### Local setup

```bash
pip install -r requirements.txt
```

### Run the API locally

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run the baseline script

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"
# optional alternative:
# export OPENAI_API_KEY="hf_your_token_here"

python inference.py
```

If no API credentials are set, `inference.py` automatically uses the offline heuristic policy.

## Reproducible Baseline Results

These numbers come from the current checked-in `inference.py` run and are also saved in [baseline_results.json](baseline_results.json).

| Task | Policy | Final Score | Notes |
| --- | --- | ---: | --- |
| `task1_easy` | heuristic | `0.9999` | Solves the easy CRM workflow cleanly |
| `task2_medium` | heuristic | `0.9999` | Solves the medium sales workflow cleanly |
| `task3_hard` | heuristic | `0.5000` | Fails contextual currency and medication fills |
| `overall_avg` | heuristic | `0.8333` | Honest baseline across all three tasks |

That gap is intentional and important. The heuristic baseline struggles on Task 3's chained reasoning, which is exactly the evidence that stronger model-based agents still have room to improve.

## Validation

Run the local checks before submission:

```bash
openenv validate
python -m unittest discover -s tests -v
python inference.py
python scripts/preflight.py
```

What those checks cover:
- `openenv validate`: spec compliance
- `unittest`: regression coverage for grading, API behavior, and inference logs
- `inference.py`: reproducible structured baseline output
- `scripts/preflight.py`: validate, tests, inference, a FastAPI smoke test, and a Docker build when Docker is available

## API Surface

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Metadata |
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | Task metadata |
| `POST` | `/reset` | Start a task |
| `POST` | `/step` | Apply one action |
| `GET` | `/state` | Lightweight episode state |
| `GET` | `/schema` | JSON schemas for action, observation, and state |
| `GET` | `/docs` | Swagger UI |

## Project Structure

```text
data-quality-env/
|- app.py
|- environment.py
|- graders.py
|- inference.py
|- models.py
|- tasks.py
|- openenv.yaml
|- baseline_results.json
|- scripts/preflight.py
|- tests/
|- Dockerfile
|- README.md
```

## Design Notes

- Real-world utility: the tasks reflect actual cleanup work in CRM, sales, and healthcare operations.
- Dense rewards: agents get step-by-step learning signal instead of a single episode-end label.
- Deterministic grading: identical state always yields identical scores.
- Hidden targets: the benchmark is auditable but not trivially exploitable.
- Honest baseline: the shipped heuristic is strong on easy and medium, and visibly weaker on hard.

## License

MIT. See [LICENSE](LICENSE).
