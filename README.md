<p align="center">
  <h1 align="center">🧬 Enterprise Data Cleaning RL Environment for LLM Agents</h1>
</p>

<p align="center">
  <strong>The first OpenEnv benchmark that doesn't just test if LLMs can clean data — it tests if they can <em>reason</em> about data.</strong>
</p>

<p align="center">
  <em>If an agent scores highly here, it is unlikely to fail silently in production data pipelines.</em>
</p>

<p align="center">
  <a href="#the-gap">The Gap</a> •
  <a href="#how-this-environment-closes-it">How We Close It</a> •
  <a href="#tasks">Tasks</a> •
  <a href="#evaluation-system">Evaluation</a> •
  <a href="#baseline-performance">Baselines</a> •
  <a href="#quickstart">Quickstart</a>
</p>

---

> **Current LLM benchmarks measure whether agents can clean data. None of them measure whether agents understand it.** That's the gap. This environment closes it.

> **This is not a data cleaning benchmark — it is a reasoning benchmark disguised as one.** It is designed not to reward correctness alone, but to penalize shallow reasoning.

---

## The Gap

Real-world data pipelines fail silently. Errors **chain** — a date format inconsistency hides a duplicate; a missing currency depends on a country field three columns away. Fix one thing, break another.

No existing benchmark captures this. They reward brute-force deletion. They expose target outputs. They let agents game their way to high scores.

The result: agents that ace benchmarks and fail in production.

---

## How This Environment Closes It

| Typical Benchmarks | This Environment |
| --- | --- |
| Expose the target output | Ground truth is **hidden** — agents never see "correct" |
| Score by issue count | **Row-level precision/recall** against hidden targets |
| Reward deletion as cleanup | Over-deletion **caps the total score** |
| Single-step fixes | Errors only surface **after intermediate corrections** |
| Unlimited attempts | **Step budgets + per-action penalties** |

---

## How It Works

The agent interacts through a standard RL loop — `reset()` → `step()` → `state()` — and sees:

- The current table (public columns only — hidden row IDs are never exposed)
- Column schema and types
- Human-readable quality issues detected in the data
- A `quality_score` in the strict open interval `(0.0001, 0.9999)`
- Available operations and their signatures

**Reward is dense:**

```
reward(t) = quality_score(t) − quality_score(t−1) − 0.01
```

Every step provides learning signal. The `−0.01` penalty ensures agents optimize for efficiency, not just correctness.

---

## Tasks

| Task | Domain | Difficulty | Rows | Budget | Core Challenge |
| --- | --- | :---: | ---: | ---: | --- |
| `task1_easy` | CRM Records | 🟢 Easy | 15 | 10 steps | Deduplication + imputation without data loss |
| `task2_medium` | Sales Ops | 🟡 Medium | 15 | 15 steps | Multi-format standardization + surgical row removal |
| `task3_hard` | Healthcare Billing | 🔴 Hard | 16 | 25 steps | Chained dependencies, contextual inference, clinical validation |

---

### Task 1 — CRM Records *(Easy)*

Deduplicate customer entries, then impute missing `age` and `email`. Agents that fill before deduplicating get skewed statistics — order matters even here.

### Task 2 — Sales Operations *(Medium)*

5 date formats, malformed phones, negative transactions, missing regions. The agent must distinguish "looks wrong but valid" from "genuinely corrupt" — the core skill gap in production data work.

### Task 3 — Healthcare Billing *(Hard)* 🔥

**The task that separates pattern-matching from reasoning.**

Duplicate patient records are **invisible until dates are standardized** — `"April 12, 1988"` and `"1988-04-12"` are the same person. Deduplicate before fixing dates? You miss every hidden duplicate and silently corrupt downstream operations.

The reward **increases** on early steps — then **crashes** when chained dependencies surface. We call this the **False Progress Trap**: agents think they're improving while walking into cascading failure.

<details>
<summary>📋 Required sequence (order matters) + False Progress example</summary>

```
1. standardize dob            → unlocks duplicate detection
2. standardize visit_date     → completes alignment
3. remove duplicates          → 16 → 12 rows (only possible AFTER step 1-2)
4. fill currency ← country    → India→INR, UK→GBP, US→USD
5. fill medication ← diagnosis → Diabetes→Metformin, Hypertension→Amlodipine
6. standardize phones         → normalize emergency_contact
7. clip vitals                → glucose, bp_systolic, bp_diastolic in clinical ranges
```

**What happens to a naive agent:**

```
Step 1: Fix phone numbers correctly         → reward: +0.04  ← looks good
Step 2: Fill currency with wrong mapping     → reward: +0.02  ← still progressing!
Step 3: Deduplicate (without fixing dates)   → reward: −0.15  ← score crashes
Step 4: Fill medication incorrectly          → reward: −0.08  ← cascading failure
```

</details>

---

## Evaluation System

Hidden ground-truth. Row-level precision/recall. Fully deterministic. Reproducible across runs.

| Metric | Purpose |
| --- | --- |
| **Row Precision** | Did the agent produce only correct rows? |
| **Row Recall** | Did the agent preserve all target rows? |
| **Row Fidelity (F1)** | Core anti-gaming metric — caps total score if either drops below 90% |
| **Column Match** | Per-column accuracy on cells that actually needed fixing |
| **Weighted Total** | Component scores × task-specific weights |

**Detected failure modes:** over-deletion, hallucinated values, missed dependencies, brute-force deletion, wasted actions.

---

## Example: Before & After

### Input (Task 3 — excerpt)

```
patient_name  │ dob              │ currency │ medication
──────────────┼──────────────────┼──────────┼────────────
Asha Rao      │ April 12, 1988   │ INR      │ Metformin
Eleanor Hall  │ 1991-08-19       │ [NULL]   │ Metformin
Farah Khan    │ 07 Mar 1986      │ INR      │ [NULL]
Asha Rao      │ 1988-04-12       │ INR      │ Metformin     ← hidden duplicate
```

### Output (after correct agent actions)

```
patient_name  │ dob              │ currency │ medication
──────────────┼──────────────────┼──────────┼─────────────────
Asha Rao      │ 1988-04-12       │ INR      │ Metformin        ✓ deduped + dates fixed
Eleanor Hall  │ 1991-08-19       │ GBP      │ Metformin        ✓ currency from country
Farah Khan    │ 1986-03-07       │ INR      │ Levothyroxine    ✓ medication from diagnosis
```

### Score Breakdown

```
Component                Score    Weight    Contribution
──────────────────────────────────────────────────────────
row_fidelity_score       1.00    × 0.35  =  0.350
date_match_score         1.00    × 0.15  =  0.150
phone_match_score        0.83    × 0.10  =  0.083
currency_match_score     1.00    × 0.15  =  0.150
medication_match_score   0.67    × 0.10  =  0.067
vitals_match_score       1.00    × 0.15  =  0.150
──────────────────────────────────────────────────────────
TOTAL                                       0.9500
```

---

## Baseline Performance

| Task | Score | Verdict |
| --- | ---: | --- |
| `task1_easy` | **0.9999** | ✅ Solved |
| `task2_medium` | **0.9999** | ✅ Solved |
| `task3_hard` | **0.5000** | ❌ Failed |
| **Overall** | **0.8333** | — |

**The baseline aces every mechanical operation and fails completely on contextual reasoning.** That's the point — the hard task measures intelligence, not pattern-matching.

<details>
<summary>📊 Task 3 component breakdown — where it fails and why</summary>

| Component | Score | |
| --- | ---: | --- |
| `row_fidelity_score` | 0.50 | Dedup ordering wrong |
| `date_match_score` | 1.00 | ✅ Mechanical |
| `phone_match_score` | 1.00 | ✅ Mechanical |
| `currency_match_score` | **0.00** | ❌ Cannot infer currency from country |
| `medication_match_score` | **0.00** | ❌ Cannot infer medication from diagnosis |
| `vitals_match_score` | 1.00 | ✅ Mechanical |

Every mechanical component: perfect. Every reasoning component: zero. [`baseline_results.json`](baseline_results.json)

</details>

---

## Constraints

| Constraint | Value |
| --- | --- |
| **Step Budget** | 10 / 15 / 25 per task |
| **Step Penalty** | `−0.01` per action |
| **Row Fidelity Gate** | 90% precision AND recall required |
| **Score Clamping** | `(0.0001, 0.9999)` — no free perfect scores |
| **Operation Validation** | Invalid ops raise errors and consume a step |

---

## Action Space

```json
{"operation": "<op>", "column": "<col or null>", "params": {}}
```

| Operation | What It Does |
| --- | --- |
| `remove_duplicates` | Remove exact-match duplicate rows |
| `fill_missing` | Impute missing values (mean / median / mode / constant / **mapping**) |
| `standardize_date` | Parse any date format → `YYYY-MM-DD` |
| `standardize_phone` | Normalize to `+91-XXXXX-XXXXX` |
| `remove_negative` | Drop rows with negative values in a column |
| `clip_outliers` | Clamp values to valid range |
| `done` | End episode |

The **mapping** strategy fills from a source column using key-value pairs. Hallucinated keys are silently ignored.

---

## Grading Weights

<details>
<summary><strong>Task 1 — CRM Records</strong></summary>

| Component | Weight |
| --- | ---: |
| `row_fidelity_score` | 0.40 |
| `age_match_score` | 0.30 |
| `email_match_score` | 0.30 |

</details>

<details>
<summary><strong>Task 2 — Sales Operations</strong></summary>

| Component | Weight |
| --- | ---: |
| `row_fidelity_score` | 0.35 |
| `date_match_score` | 0.20 |
| `phone_match_score` | 0.20 |
| `amount_match_score` | 0.15 |
| `region_match_score` | 0.10 |

</details>

<details>
<summary><strong>Task 3 — Healthcare Billing</strong></summary>

| Component | Weight |
| --- | ---: |
| `row_fidelity_score` | 0.35 |
| `date_match_score` | 0.15 |
| `phone_match_score` | 0.10 |
| `currency_match_score` | 0.15 |
| `medication_match_score` | 0.10 |
| `vitals_match_score` | 0.15 |

</details>

---

## Quickstart

### Install

```bash
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run the Baseline

```bash
# With LLM (optional):
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"

# Without LLM — falls back to deterministic heuristic:
python inference.py
```

### Validate Before Submission

```bash
openenv validate                          # Spec compliance
python -m unittest discover -s tests -v   # Regression suite
python inference.py                       # Reproducible baseline
python scripts/preflight.py               # Full preflight + Docker build
```

---

## API Surface

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Environment metadata |
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | Task list with metadata |
| `POST` | `/reset` | Initialize a task episode |
| `POST` | `/step` | Execute one cleaning action |
| `GET` | `/state` | Current episode snapshot |
| `GET` | `/schema` | JSON schemas for action/observation/state |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Use Cases

- **🧪 Benchmarking LLM data reasoning** — not pattern-matching, reasoning
- **🤖 Evaluating autonomous data agents** — multi-step workflows with hidden dependencies
- **🧠 Stress-testing LLM reasoning** — chained failures that require planning
- **📊 RL training** — dense per-step reward signal on real-world tasks
- **🔬 Tool-use evaluation** — constrained action space under budget pressure

---

## Roadmap

The architecture generalizes. Next:

- **New domains** — supply chain, financial reconciliation, regulatory compliance
- **Adversarial generation** — procedural data failures to prevent overfitting
- **Multi-agent** — parallel cleanup of dependent table shards
- **Human-in-the-loop** — comparing agent strategies against expert data engineers

---

## Project Structure

```
enterprise-data-cleaning-env/
├── app.py                  # FastAPI server — reset/step/state endpoints
├── environment.py          # Core RL environment — DataQualityEnv class
├── graders.py              # Hidden-target grading with row-level precision/recall
├── inference.py            # Baseline agent — heuristic + LLM policies
├── models.py               # Pydantic models — Action, Observation, Reward
├── tasks.py                # Task definitions — data, targets, scoring weights
├── openenv.yaml            # OpenEnv specification
├── baseline_results.json   # Reproducible baseline scores
├── scripts/
│   └── preflight.py        # Full pre-submission validation suite
├── tests/                  # Unit + integration test coverage
├── Dockerfile              # Production container for HF Spaces
└── README.md
```

---

## Design Principles

| | |
| --- | --- |
| **Real-world fidelity** | CRM, revenue ops, and healthcare data incidents |
| **Dense signal** | Per-step reward, not episode-end |
| **Deterministic** | Same input → same score, always |
| **Hidden targets** | Auditable but not exploitable |
| **Honest baselines** | Strong on easy, fails on hard — by design |
| **Anti-gaming** | Row fidelity caps, step penalties, step budgets |

---

## License

MIT. See [LICENSE](LICENSE).
