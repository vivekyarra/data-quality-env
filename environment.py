"""
environment.py — DataQualityEnv core class (importable standalone or via FastAPI).

Public API
----------
env = DataQualityEnv()
obs          = env.reset(task_id)          → Observation
result       = env.step(action)            → StepResult
state_dict   = env.state()                 → dict
task_list    = env.list_tasks()            → list[TaskInfo]
"""

import re
from typing import Any, Dict, List, Optional

import pandas as pd
from dateutil import parser as _dateparser

from models import (
    Action, Observation, Reward, StepResult, TaskInfo,
    ALLOWED_OPERATIONS,
)
from tasks import TASKS
from graders import grade

# ── Regex helpers ─────────────────────────────────────────────────────────────

_ISO_DATE  = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_PHONE_FMT = re.compile(r'^\+91-\d{5}-\d{5}$')


def _is_iso_date(s: str) -> bool:
    return bool(_ISO_DATE.match(str(s).strip()))


def _is_valid_phone(s: str) -> bool:
    return bool(_PHONE_FMT.match(str(s).strip()))


def _normalize_phone(s: str) -> str:
    """Convert any 10-digit Indian number to +91-XXXXX-XXXXX."""
    digits = re.sub(r'\D', '', str(s))
    if digits.startswith('91') and len(digits) == 12:
        digits = digits[2:]
    if digits.startswith('0') and len(digits) == 11:
        digits = digits[1:]
    if len(digits) == 10:
        return f"+91-{digits[:5]}-{digits[5:]}"
    return s  # Return unchanged if can't normalise


def _safe_parse_date(v: Any) -> str:
    """Parse any date string and return YYYY-MM-DD, or return original on failure."""
    try:
        return _dateparser.parse(str(v)).strftime("%Y-%m-%d")
    except Exception:
        return str(v)


# ── Environment ───────────────────────────────────────────────────────────────

class DataQualityEnv:
    """
    OpenEnv-compliant environment for data quality / cleaning tasks.

    The agent receives a dirty tabular dataset and must apply a sequence of
    transformation operations to maximise a quality score.  Rewards are dense:
    each step returns (new_score − old_score − step_penalty) so the agent
    gets signal throughout the episode, not just at the end.
    """

    STEP_PENALTY = 0.01   # Small cost per step to encourage efficiency

    def __init__(self) -> None:
        self._task_id: Optional[str] = None
        self._df: Optional[pd.DataFrame] = None
        self._original_df: Optional[pd.DataFrame] = None
        self._step_count: int = 0
        self._done: bool = False
        self._episode_reward: float = 0.0
        self._prev_score: float = 0.0

    # ── OpenEnv interface ────────────────────────────────────────────────────

    def reset(self, task_id: str = "task1_easy") -> Observation:
        """Initialise a new episode for *task_id* and return the first observation."""
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task '{task_id}'. Available: {list(TASKS.keys())}"
            )
        task = TASKS[task_id]
        self._task_id = task_id
        self._df = pd.DataFrame(task["data"]).copy()
        self._original_df = pd.DataFrame(task["data"]).copy()
        self._step_count = 0
        self._done = False
        self._episode_reward = 0.0
        self._prev_score = self._compute_score()
        return self._get_observation()

    def step(self, action: Action) -> StepResult:
        """Apply *action*, compute reward, and return StepResult."""
        if self._df is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._step_count += 1
        info: Dict[str, Any] = {"step": self._step_count}

        # Apply the operation
        try:
            msg = self._apply_operation(action)
            info["applied"] = msg
        except Exception as exc:
            msg = f"Operation failed: {exc}"
            info["error"] = msg

        # Compute new quality score
        new_score = self._compute_score()
        breakdown = self._get_score_breakdown()
        delta = new_score - self._prev_score

        # Dense reward: improvement delta minus step cost
        reward_value = round(delta - self.STEP_PENALTY, 4)
        self._prev_score = new_score
        self._episode_reward = round(self._episode_reward + reward_value, 4)

        # Termination
        max_steps = TASKS[self._task_id]["max_steps"]
        if action.operation == "done" or self._step_count >= max_steps:
            self._done = True

        reward = Reward(
            value=reward_value,
            message=msg,
            score_breakdown=breakdown,
        )
        obs = self._get_observation()

        info["episode_reward"] = self._episode_reward
        info["quality_score"]  = new_score

        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info=info,
        )

    def state(self) -> Dict[str, Any]:
        """Return lightweight current state (does not include full table)."""
        return {
            "task_id":        self._task_id,
            "step_count":     self._step_count,
            "max_steps":      TASKS[self._task_id]["max_steps"] if self._task_id else None,
            "done":           self._done,
            "episode_reward": self._episode_reward,
            "quality_score":  self._compute_score() if self._df is not None else None,
            "table_shape":    list(self._df.shape) if self._df is not None else None,
        }

    def list_tasks(self) -> List[TaskInfo]:
        return [
            TaskInfo(
                task_id=tid,
                name=cfg["name"],
                description=cfg["description"],
                difficulty=cfg["difficulty"],
                max_steps=cfg["max_steps"],
                columns=list(cfg["schema"].keys()),
            )
            for tid, cfg in TASKS.items()
        ]

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _compute_score(self) -> float:
        if self._df is None or self._original_df is None:
            return 0.0
        breakdown = grade(self._task_id, self._df, self._original_df)
        return breakdown["total"]

    def _get_score_breakdown(self) -> Dict[str, float]:
        if self._df is None:
            return {}
        return grade(self._task_id, self._df, self._original_df)

    def _get_observation(self) -> Observation:
        task = TASKS[self._task_id]
        return Observation(
            task_id=self._task_id,
            task_name=task["name"],
            task_description=task["description"],
            table=self._df.where(pd.notnull(self._df), None).to_dict(orient="records"),
            column_schema=task["schema"],
            quality_issues=self._detect_issues(),
            quality_score=self._compute_score(),
            step_count=self._step_count,
            max_steps=task["max_steps"],
            available_operations=ALLOWED_OPERATIONS,
        )

    def _detect_issues(self) -> List[str]:
        """Scan the current DataFrame and return human-readable issue descriptions."""
        df = self._df
        task = TASKS[self._task_id]
        issues: List[str] = []

        # Duplicates
        subset = task.get("dup_subset")
        dups = df.duplicated(subset=subset, keep="first").sum()
        if dups > 0:
            key_desc = f"[{', '.join(subset)}]" if subset else "all columns"
            issues.append(f"{int(dups)} duplicate rows detected (key: {key_desc})")

        # Missing values per column
        for col in df.columns:
            missing = int(df[col].isna().sum())
            if missing > 0:
                issues.append(f"Column '{col}': {missing} missing value(s)")

        # Date format issues
        for col in task.get("date_columns", []):
            if col in df.columns:
                bad = sum(
                    1 for v in df[col].astype(str)
                    if str(v).strip() not in ("nan", "None", "") and not _is_iso_date(v)
                )
                if bad > 0:
                    issues.append(
                        f"Column '{col}': {bad} date(s) not in YYYY-MM-DD format"
                    )

        # Phone format issues
        for col in task.get("phone_columns", []):
            if col in df.columns:
                bad = sum(
                    1 for v in df[col].astype(str)
                    if str(v).strip() not in ("nan", "None", "") and not _is_valid_phone(v)
                )
                if bad > 0:
                    issues.append(
                        f"Column '{col}': {bad} phone number(s) not in +91-XXXXX-XXXXX format"
                    )

        # Negative values
        for col in task.get("positive_columns", []):
            if col in df.columns:
                neg = int((pd.to_numeric(df[col], errors="coerce") < 0).sum())
                if neg > 0:
                    issues.append(f"Column '{col}': {neg} negative value(s) detected")

        # Outliers
        for col, (lo, hi) in task.get("outlier_ranges", {}).items():
            if col in df.columns:
                outliers = int((~df[col].between(lo, hi)).sum())
                if outliers > 0:
                    issues.append(
                        f"Column '{col}': {outliers} value(s) outside valid range [{lo}, {hi}]"
                    )

        return issues if issues else ["✓ No quality issues detected"]

    # ── Operation dispatcher ─────────────────────────────────────────────────

    def _apply_operation(self, action: Action) -> str:
        op     = action.operation.strip()
        col    = action.column
        params = action.params or {}

        if op == "remove_duplicates":
            subset = params.get("subset", TASKS[self._task_id].get("dup_subset"))
            before = len(self._df)
            self._df = (
                self._df.drop_duplicates(subset=subset, keep="first")
                        .reset_index(drop=True)
            )
            removed = before - len(self._df)
            return f"Removed {removed} duplicate row(s) (subset={subset})"

        elif op == "fill_missing":
            self._require_column(col)
            strategy = params.get("strategy", "mode")
            before   = int(self._df[col].isna().sum())
            if strategy == "mean":
                self._df[col] = self._df[col].fillna(self._df[col].mean())
            elif strategy == "median":
                self._df[col] = self._df[col].fillna(self._df[col].median())
            elif strategy == "mode":
                modes = self._df[col].mode()
                if len(modes):
                    self._df[col] = self._df[col].fillna(modes[0])
            elif strategy == "constant":
                fill_val = params.get("value", "N/A")
                self._df[col] = self._df[col].fillna(fill_val)
            else:
                raise ValueError(f"Unknown fill strategy '{strategy}'")
            after = int(self._df[col].isna().sum())
            return f"Filled {before - after} missing value(s) in '{col}' using strategy='{strategy}'"

        elif op == "standardize_date":
            self._require_column(col)
            before_bad = sum(
                1 for v in self._df[col].astype(str)
                if str(v) not in ("nan", "None") and not _is_iso_date(v)
            )
            self._df[col] = self._df[col].apply(_safe_parse_date)
            after_bad = sum(
                1 for v in self._df[col].astype(str)
                if str(v) not in ("nan", "None") and not _is_iso_date(v)
            )
            fixed = before_bad - after_bad
            return f"Standardised {fixed} date(s) in '{col}' → YYYY-MM-DD"

        elif op == "standardize_phone":
            self._require_column(col)
            before_bad = sum(
                1 for v in self._df[col].astype(str)
                if str(v) not in ("nan", "None") and not _is_valid_phone(v)
            )
            self._df[col] = self._df[col].apply(_normalize_phone)
            after_bad = sum(
                1 for v in self._df[col].astype(str)
                if str(v) not in ("nan", "None") and not _is_valid_phone(v)
            )
            fixed = before_bad - after_bad
            return f"Normalised {fixed} phone number(s) in '{col}' → +91-XXXXX-XXXXX"

        elif op == "remove_negative":
            self._require_column(col)
            before = len(self._df)
            self._df = (
                self._df[pd.to_numeric(self._df[col], errors="coerce") >= 0]
                        .reset_index(drop=True)
            )
            removed = before - len(self._df)
            return f"Removed {removed} row(s) with negative values in '{col}'"

        elif op == "clip_outliers":
            self._require_column(col)
            lower = params.get("lower")
            upper = params.get("upper")
            if lower is None and upper is None:
                # Use task-defined ranges if available
                ranges = TASKS[self._task_id].get("outlier_ranges", {})
                if col in ranges:
                    lower, upper = ranges[col]
                else:
                    raise ValueError(f"clip_outliers requires 'lower' and/or 'upper' params for '{col}'")
            before_count = 0
            if lower is not None:
                before_count += int((self._df[col] < lower).sum())
            if upper is not None:
                before_count += int((self._df[col] > upper).sum())
            self._df[col] = self._df[col].clip(lower=lower, upper=upper)
            return f"Clipped {before_count} outlier(s) in '{col}' to [{lower}, {upper}]"

        elif op == "done":
            return "Agent signalled end of episode"

        else:
            raise ValueError(
                f"Unknown operation '{op}'. Allowed: {ALLOWED_OPERATIONS}"
            )

    def _require_column(self, col: Optional[str]) -> None:
        if not col:
            raise ValueError("'column' field is required for this operation")
        if col not in self._df.columns:
            raise ValueError(
                f"Column '{col}' not found. Available: {list(self._df.columns)}"
            )
