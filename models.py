"""
Typed Pydantic models for DataQualityEnv.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


ALLOWED_OPERATIONS = [
    "remove_duplicates",
    "fill_missing",
    "standardize_date",
    "standardize_phone",
    "remove_negative",
    "clip_outliers",
    "done",
]


class Action(BaseModel):
    """
    Agent action for a single data-cleaning step.

    ``fill_missing`` supports ``mean``, ``median``, ``mode``, ``constant``,
    and ``mapping`` strategies.
    """

    operation: str
    column: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """
    Environment observation returned after ``reset()`` or ``step()``.
    """

    task_id: str
    task_name: str
    task_description: str
    table: List[Dict[str, Any]]
    column_schema: Dict[str, str]
    quality_issues: List[str]
    quality_score: float = Field(gt=0.0, lt=1.0)
    step_count: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    available_operations: List[str]


class Reward(BaseModel):
    """
    Dense reward signal emitted on every step.
    """

    value: float = Field(ge=-1.0, le=1.0)
    message: str
    score_breakdown: Dict[str, float]


class StepResult(BaseModel):
    """
    Full return value of ``step()``.
    """

    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


class ResetRequest(BaseModel):
    task_id: str = "task1_easy"


class StateSnapshot(BaseModel):
    task_id: Optional[str] = None
    step_count: int = Field(ge=0)
    max_steps: Optional[int] = Field(default=None, ge=1)
    done: bool
    episode_reward: float
    quality_score: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    table_shape: Optional[List[int]] = None


class TaskInfo(BaseModel):
    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int = Field(ge=1)
    columns: List[str]
