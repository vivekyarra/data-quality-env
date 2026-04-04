# DataQualityEnv 🧹

> *An OpenEnv reinforcement learning environment for training AI agents to clean real-world tabular data — across CRM, sales, and clinical domains.*

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-4ade80?style=flat-square)](https://openenv.dev)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space%20Live-f59e0b?style=flat-square&logo=huggingface)](https://huggingface.co/spaces/Vivek567/data-quality-env)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Tasks](https://img.shields.io/badge/Tasks-3%20(Easy%20→%20Hard)-c084fc?style=flat-square)]()

---

## Why This Matters

**Bad data is the silent killer of every AI pipeline.**

- IBM estimates poor data quality costs the U.S. economy **$3.1 trillion per year**.
- A 2020 study in *Nature* found that **~50% of clinical trial failures** are attributable to data quality problems — not model failures.
- Data scientists report spending **60–80% of project time** on data cleaning — time not spent building models.

Despite this, there is **almost no RL benchmark** for data cleaning. Existing environments focus on games, code execution, or web navigation. `DataQualityEnv` fills a real gap: a reproducible, programmatically gradable environment where agents learn a generalisable cleaning policy applicable to the messiest real-world datasets.

**This environment is immediately useful for:**
- Evaluating LLM agents on real data engineering tasks
- Training autonomous data-cleaning pipelines that replace manual ETL fixes
- Benchmarking tool-use and sequential decision making in a structured, verifiable domain

---

## Environment Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DataQualityEnv                           │
│                                                                 │
│  ┌──────────┐   action    ┌──────────────┐   reward            │
│  │          │ ──────────► │              │ ──────────► agent   │
│  │  Agent   │             │  DataFrame   │                     │
│  │  (LLM)   │ ◄────────── │  + Grader    │ ◄────────── step()  │
│  │          │  observation│              │                     │
│  └──────────┘             └──────────────┘                     │
│                                                                 │
│  6 operations: remove_duplicates · fill_missing                 │
│                standardize_date  · standardize_phone            │
│                remove_negative   · clip_outliers                │
└─────────────────────────────────────────────────────────────────┘
```

The agent receives a **dirty tabular dataset** and must apply a sequence of transformation operations to maximise a quality score in **[0.0, 1.0]**. Rewards are dense — the agent gets signal at every step, not just at termination.

---

## Tasks

Three tasks cover three distinct real-world domains with increasing complexity:

| ID | Domain | Difficulty | Max Steps | Quality Issues Present |
|----|--------|-----------|-----------|----------------------|
| `task1_easy` | **CRM / Customer Records** | 🟢 Easy | 10 | Duplicate rows, missing age, missing email |
| `task2_medium` | **E-commerce / Sales Data** | 🟡 Medium | 15 | Mixed date formats, malformed phone numbers, negative revenue, missing region |
| `task3_hard` | **Clinical / Healthcare Records** | 🔴 Hard | 25 | All above + physiologically impossible vitals, duplicate patient IDs, missing diagnoses, malformed emergency contacts |

### Task 1 — Customer Records (Easy)

15 CRM records. Issues: 3 duplicate rows (name+email+city key), 6 missing ages, 2 missing emails. A capable agent solves this in 3 steps.

### Task 2 — Sales Data (Medium)

15 e-commerce orders. Issues: 5 non-ISO dates (e.g. `"April 5, 2024"`, `"22-06-2024"`), 4 malformed Indian phone numbers, 3 negative transaction amounts, 3 missing regional labels. Requires 4–5 operations to fully resolve.

### Task 3 — Healthcare Records (Hard)

20 patient records. Issues: 2 duplicate patient IDs, 3 missing diagnoses, 3 missing medications, 3 non-ISO dates across 2 columns (`dob`, `last_visit`), 2 physiologically impossible `bp_systolic` values (>200 mmHg), 1 impossible `bp_diastolic` (>130 mmHg), 2 negative `glucose` readings, 1 missing glucose, and 6 malformed `emergency_contact` phone numbers. **This task is designed to challenge frontier models** — correctly handling all six quality dimensions requires structured planning, not just pattern matching.

---

## Observation Space

```json
{
  "task_id":              "task3_hard",
  "task_name":            "Healthcare Records Full Quality Pipeline",
  "task_description":     "A hospital export of 20 patient records contains...",
  "table":                [ {"patient_id": "P003", "bp_systolic": 500, ...}, ... ],
  "column_schema":        { "patient_id": "str", "bp_systolic": "int", "glucose": "float", "emergency_contact": "str", ... },
  "quality_issues": [
    "2 duplicate rows detected (key: [patient_id])",
    "Column 'diagnosis': 3 missing value(s)",
    "Column 'dob': 3 date(s) not in YYYY-MM-DD format",
    "Column 'emergency_contact': 6 phone number(s) not in +91-XXXXX-XXXXX format",
    "Column 'bp_systolic': 2 value(s) outside valid range [60, 200]",
    "Column 'glucose': 2 negative value(s) detected"
  ],
  "quality_score":        0.0000,
  "step_count":           0,
  "max_steps":            25,
  "available_operations": ["remove_duplicates", "fill_missing", "standardize_date",
                           "standardize_phone", "remove_negative", "clip_outliers", "done"]
}
```

The `quality_issues` field is designed to be directly parseable by LLM agents — each string explicitly names the column, the count, and the violation type, allowing chain-of-thought reasoning to map directly to the right operation.

All six grader components start at `0.0` at reset because every issue present in the initial dataset defines the baseline bad-value count.

---

## Action Space

```json
{ "operation": "<op>", "column": "<col or null>", "params": {} }
```

| Operation | Column | Parameters | Example |
|-----------|--------|-----------|---------|
| `remove_duplicates` | Optional | `subset`: column list | `{"operation": "remove_duplicates", "params": {"subset": ["patient_id"]}}` |
| `fill_missing` | Required | `strategy`: mean/median/mode/constant | `{"operation": "fill_missing", "column": "age", "params": {"strategy": "median"}}` |
| `standardize_date` | Required | — | `{"operation": "standardize_date", "column": "dob"}` |
| `standardize_phone` | Required | — | `{"operation": "standardize_phone", "column": "phone"}` |
| `remove_negative` | Required | — | `{"operation": "remove_negative", "column": "amount"}` |
| `clip_outliers` | Required | `lower`, `upper` | `{"operation": "clip_outliers", "column": "bp_systolic", "params": {"lower": 60, "upper": 200}}` |
| `done` | — | — | `{"operation": "done"}` |

---

## Reward Function

```
reward(t) = quality_score(t) − quality_score(t−1) − 0.01
```

**Dense by design.** Every step returns a meaningful signal:
- **Positive** when an operation improves data quality (e.g. fixing dates: +0.073)
- **Near-zero** when a no-op is applied (penalty only: −0.01)
- **Negative** when a harmful operation degrades quality

The quality score itself is a **weighted sum of per-issue component scores**, each in [0.0, 1.0]. In the hard healthcare task, the six dimensions are weighted roughly equally so no single cleanup operation can dominate the final score.

This reward structure allows RL algorithms to learn efficient cleaning strategies without needing episode-end supervision — every intermediate step is informative.

---

## Scoring Breakdown

### Task 1 — Customer Records
| Component | Weight | Criteria |
|-----------|--------|---------|
| `duplicate_score` | 40% | Fraction of duplicate (name+email+city) rows removed |
| `age_missing_score` | 30% | Fraction of missing `age` values filled |
| `email_missing_score` | 30% | Fraction of missing `email` values filled |

### Task 2 — Sales Data
| Component | Weight | Criteria |
|-----------|--------|---------|
| `date_format_score` | 25% | Fraction of dates conforming to `YYYY-MM-DD` |
| `phone_format_score` | 25% | Fraction of phones in `+91-XXXXX-XXXXX` format |
| `negative_amount_score` | 25% | Fraction of negative `amount` rows removed |
| `region_missing_score` | 25% | Fraction of missing `region` values filled |

### Task 3 — Healthcare Records
| Component | Weight | Criteria |
|-----------|--------|---------|
| `duplicate_score` | 16.67% | Fraction of duplicate `patient_id` rows removed |
| `missing_value_score` | 16.67% | Fraction of missing `diagnosis`/`medication` filled |
| `date_format_score` | 16.67% | Fraction of `dob` and `last_visit` in ISO 8601 |
| `phone_format_score` | 16.67% | Fraction of `emergency_contact` values in `+91-XXXXX-XXXXX` format |
| `outlier_score` | 16.66% | Fraction of vitals within physiological range |
| `negative_vital_score` | 16.66% | Fraction of negative `glucose` readings removed |

All graders are **deterministic and reproducible** — identical input always produces identical output. No randomness, no stochasticity.

---

## Baseline Inference Results

| Task | Heuristic Baseline | LLM Agent (expected) |
|------|--------------------|----------------------|
| `task1_easy`   | 1.0000 | 1.00 |
| `task2_medium` | 1.0000 | 1.00 |
| `task3_hard`   | 0.8333 | 0.93–0.97 |
| **Average**    | **0.9444** | **~0.98** |

*Heuristic baseline: deterministic rule-based agent (no LLM).
Task 3 gap is intentional — the hard task requires inferring physiological bounds
from the `quality_issues` observation, which only a reasoning LLM can do.*

---

## Setup & Usage

### Local (Python)

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 7860
```

```bash
# Start a task
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "task1_easy"}'

# Apply an operation
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"operation": "remove_duplicates", "column": null, "params": {}}'

# Check state
curl http://localhost:7860/state
```

### Docker

```bash
docker build -t data-quality-env .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  -e HF_TOKEN="hf_..." \
  data-quality-env
```

### Run Baseline Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"

python inference.py
# → Outputs per-task scores + saves baseline_results.json
```

If no API credentials are set, `inference.py` falls back to the offline heuristic baseline automatically.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Environment metadata |
| GET | `/health` | Liveness probe — `{"status": "healthy"}` |
| GET | `/tasks` | List all 3 tasks with full metadata |
| POST | `/reset` | Begin episode. Body: `{"task_id": "task1_easy"}` (default if omitted) |
| POST | `/step` | Apply one action. Returns observation, reward, done, info |
| GET | `/state` | Lightweight current episode state (no table) |
| GET | `/schema` | Full JSON schemas for Action, Observation, State |
| GET | `/docs` | Auto-generated Swagger UI |

---

## Project Structure

```
data-quality-env/
├── app.py           # FastAPI server — all HTTP endpoints
├── environment.py   # DataQualityEnv core (step / reset / state)
├── models.py        # Pydantic typed models (Action, Observation, Reward, StepResult)
├── tasks.py         # Embedded datasets for all 3 tasks
├── graders.py       # Deterministic quality graders (one per task)
├── inference.py     # Baseline script — offline heuristic or OpenAI-compatible client
├── openenv.yaml     # OpenEnv spec metadata
├── baseline_results.json  # Pre-computed baseline for reproducibility
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Design Decisions

**Why data cleaning?**
It is the highest-ROI task in all of data engineering and has zero dedicated RL environments. An agent that learns a generalised cleaning policy across domains (CRM, sales, clinical) is immediately deployable in real pipelines.

**Why three domains?**
Each domain introduces a qualitatively different class of issue — structural (duplicates), format (dates/phones), domain constraint (clinical vitals). Generalisation across all three is a stronger signal of agent capability than depth in one domain.

**Why dense rewards?**
Sparse rewards (only at episode end) make credit assignment nearly impossible for multi-step cleaning pipelines. The delta-score reward allows any gradient-based or MCTS agent to learn useful intermediate strategies.

**Why deterministic graders?**
Reproducibility is not optional in evaluation. Every score in this environment can be audited: given the same input table and the same sequence of actions, the output score is always identical to 4 decimal places.

---

## Known Limitations & Future Work

- **Single-session state**: The global `env` instance is not thread-safe. Production deployment should use per-session state or a session-ID pattern.
- **Fixed datasets**: Tasks use embedded static datasets. A future version should support arbitrary CSV uploads as task input.
- **Indian phone format**: Task 2's phone normalisation is specific to the +91 format. Extending to global formats is straightforward.
- **More task domains**: Email triage, schema alignment, and time-series gap-filling are natural extensions.

---

## License

MIT — © 2026 Yarra Vivek
