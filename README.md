# DataQualityEnv 🧹

**An OpenEnv reinforcement learning environment for real-world data cleaning tasks.**

An AI agent interacts with messy tabular datasets and learns to apply data-transformation operations — deduplication, imputation, format normalisation, outlier clipping — in order to maximise a quality score. Three tasks span easy → medium → hard, each with a programmatic grader that returns a deterministic score in [0.0, 1.0].

---

## Motivation

Data scientists spend ~80% of their time cleaning data. Yet almost no RL benchmark exists for this task. `DataQualityEnv` fills that gap: it provides a reproducible, programmatically gradable environment where agents can learn a generalisable data-cleaning policy applicable to CRM records, sales exports, and clinical datasets.

---

## Tasks

| ID | Name | Difficulty | Max Steps | Issues |
|----|------|-----------|-----------|--------|
| `task1_easy` | Customer Records Deduplication & Imputation | Easy | 10 | Duplicate rows, missing ages, missing emails |
| `task2_medium` | Sales Data Standardisation & Cleansing | Medium | 15 | Mixed date formats, malformed phones, negative amounts, missing regions |
| `task3_hard` | Healthcare Records Full Quality Pipeline | Hard | 25 | All of the above + physiologically impossible vitals |

---

## Observation Space

```json
{
  "task_id":              "task2_medium",
  "task_name":            "Sales Data Standardisation & Cleansing",
  "task_description":     "...",
  "table":                [ {"order_id": "ORD001", "date": "15/02/2024", ...}, ... ],
  "column_schema":        { "order_id": "str", "date": "str", "amount": "float", ... },
  "quality_issues":       [ "Column 'date': 4 date(s) not in YYYY-MM-DD format", ... ],
  "quality_score":        0.3125,
  "step_count":           0,
  "max_steps":            15,
  "available_operations": ["remove_duplicates", "fill_missing", "standardize_date", ...]
}
```

## Action Space

```json
{ "operation": "<op>", "column": "<col or null>", "params": {} }
```

| Operation | Column Required | Key Params |
|-----------|----------------|-----------|
| `remove_duplicates` | No | `subset` (list of col names, optional) |
| `fill_missing` | Yes | `strategy`: `"mean"` \| `"median"` \| `"mode"` \| `"constant"` |
| `standardize_date` | Yes | — |
| `standardize_phone` | Yes | — |
| `remove_negative` | Yes | — |
| `clip_outliers` | Yes | `lower`, `upper` (floats) |
| `done` | No | — |

---

## Reward Function

```
reward = (quality_score_new − quality_score_old) − 0.01
```

- **Dense**: the agent receives a signal at every step, not just at termination.
- **Positive** when the operation improves data quality.
- **Negative** for no-op or harmful actions (the −0.01 step penalty discourages unnecessary steps).
- Quality score is a weighted combination of per-issue component scores, each in [0.0, 1.0].

---

## Setup & Usage

### Local (Python)

```bash
pip install -r requirements.txt

# Run the FastAPI server
uvicorn app:app --reload --port 7860

# In another terminal — interact via HTTP
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" \
     -d '{"task_id": "task1_easy"}'

curl -X POST http://localhost:7860/step -H "Content-Type: application/json" \
     -d '{"operation": "remove_duplicates", "column": null, "params": {}}'

curl http://localhost:7860/state
```

### Docker

```bash
docker build -t data-quality-env .
docker run -p 7860:7860 data-quality-env
```

### Run Baseline Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"

python inference.py
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Environment info |
| GET | `/health` | Liveness probe |
| GET | `/tasks` | List all tasks |
| POST | `/reset` | Start a new episode. Body: `{"task_id": "..."}` |
| POST | `/step` | Apply one action. Body: Action JSON |
| GET | `/state` | Current episode state (lightweight) |

---

## Scoring Breakdown

### Task 1 — Easy
| Component | Weight | Passes when |
|-----------|--------|-------------|
| `duplicate_score` | 40% | No duplicate (name+email+city) rows remain |
| `age_missing_score` | 30% | No missing values in `age` column |
| `email_missing_score` | 30% | No missing values in `email` column |

### Task 2 — Medium
| Component | Weight | Passes when |
|-----------|--------|-------------|
| `date_format_score` | 25% | All dates in `YYYY-MM-DD` |
| `phone_format_score` | 25% | All phones in `+91-XXXXX-XXXXX` |
| `negative_amount_score` | 25% | No negative `amount` values |
| `region_missing_score` | 25% | No missing `region` values |

### Task 3 — Hard
| Component | Weight | Passes when |
|-----------|--------|-------------|
| `duplicate_score` | 20% | No duplicate `patient_id` rows |
| `missing_value_score` | 20% | No missing `diagnosis` or `medication` |
| `date_format_score` | 20% | All `dob` and `last_visit` in `YYYY-MM-DD` |
| `outlier_score` | 20% | All vitals within physiological ranges |
| `negative_vital_score` | 20% | No negative `glucose` values |

---

## Baseline Scores

Scores produced by `Qwen/Qwen2.5-72B-Instruct` via HF Inference Router:

| Task | Score |
|------|-------|
| `task1_easy` | 0.97 |
| `task2_medium` | 0.88 |
| `task3_hard` | 0.74 |
| **Average** | **0.86** |

*(Run `python inference.py` to reproduce)*

---

## Project Structure

```
data-quality-env/
├── app.py          # FastAPI server (OpenEnv HTTP interface)
├── environment.py  # Core DataQualityEnv class
├── models.py       # Typed Pydantic models (Observation, Action, Reward)
├── tasks.py        # Task definitions and embedded datasets
├── graders.py      # Deterministic quality graders for all 3 tasks
├── inference.py    # Baseline inference script (OpenAI client)
├── openenv.yaml    # OpenEnv metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## License

MIT — © 2026 Yarra Vivek
