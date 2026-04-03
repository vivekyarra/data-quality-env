"""
models.py — Typed Pydantic models for DataQualityEnv (OpenEnv spec)
"""

from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class Action(BaseModel):
    """
    Agent action: a single data-transformation operation.

    operation  : one of the ALLOWED_OPERATIONS below
    column     : target column name (required for column-level ops)
    params     : extra kwargs for the operation
    """
    operation: str
    column: Optional[str] = None
    params: Dict[str, Any] = {}


ALLOWED_OPERATIONS = [
    "remove_duplicates",   # Drop duplicate rows
    "fill_missing",        # Impute NaN values
    "standardize_date",    # Normalise date strings → YYYY-MM-DD
    "standardize_phone",   # Normalise phone strings → +91-XXXXX-XXXXX
    "remove_negative",     # Drop rows with negative values in a column
    "clip_outliers",       # Clip column to a physiological / business range
    "done",                # Agent signals episode end
]


class Observation(BaseModel):
    """
    Environment observation returned to the agent after reset() or step().
    """
    task_id: str
    task_name: str
    task_description: str
    table: List[Dict[str, Any]]           # Current dataset as list of row-dicts
    column_schema: Dict[str, str]         # col → expected dtype string
    quality_issues: List[str]             # Human-readable list of detected issues
    quality_score: float                  # Current 0.0–1.0 quality score
    step_count: int
    max_steps: int
    available_operations: List[str]


class Reward(BaseModel):
    """
    Reward signal emitted on every step.
    value          : delta in quality score minus small step penalty
    score_breakdown: per-component quality scores for interpretability
    """
    value: float
    message: str
    score_breakdown: Dict[str, float]


class StepResult(BaseModel):
    """Full return value of step()."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


# ── HTTP request schemas ──────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task1_easy"


class TaskInfo(BaseModel):
    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int
    columns: List[str]
