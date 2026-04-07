"""
Core DataQualityEnv environment.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd
from dateutil import parser as date_parser

from graders import grade
from models import ALLOWED_OPERATIONS, Action, Observation, Reward, StepResult, TaskInfo
from tasks import TASKS, make_row_id

TASK_SCORE_EPS = 0.0001

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PHONE_FMT = re.compile(r"^\+91-\d{5}-\d{5}$")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _expose_task_total(raw_total: float) -> float:
    return round(_clamp(raw_total, TASK_SCORE_EPS, 1.0 - TASK_SCORE_EPS), 4)


def _is_iso_date(value: Any) -> bool:
    return bool(ISO_DATE.match(str(value).strip()))


def _is_valid_phone(value: Any) -> bool:
    return bool(PHONE_FMT.match(str(value).strip()))


def _normalize_phone(value: Any) -> Any:
    if pd.isna(value):
        return value
    digits = re.sub(r"\D", "", str(value))
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    if digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]
    if len(digits) == 10:
        return f"+91-{digits[:5]}-{digits[5:]}"
    return value


def _safe_parse_date(value: Any) -> Any:
    if pd.isna(value):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return value
    try:
        return date_parser.parse(text).strftime("%Y-%m-%d")
    except Exception:
        return value


def _inject_hidden_row_ids(task_id: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        payload = dict(row)
        payload["__row_id__"] = make_row_id(task_id, index)
        enriched.append(payload)
    return enriched


class DataQualityEnv:
    STEP_PENALTY = 0.01

    def __init__(self) -> None:
        self._task_id: Optional[str] = None
        self._df: Optional[pd.DataFrame] = None
        self._original_df: Optional[pd.DataFrame] = None
        self._step_count = 0
        self._done = False
        self._episode_reward = 0.0
        self._prev_score = TASK_SCORE_EPS

    def reset(self, task_id: str = "task1_easy") -> Observation:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task '{task_id}'. Available: {list(TASKS.keys())}")

        task = TASKS[task_id]
        self._task_id = task_id
        self._df = pd.DataFrame(_inject_hidden_row_ids(task_id, task["data"]))
        self._original_df = pd.DataFrame(_inject_hidden_row_ids(task_id, task["data"]))
        self._step_count = 0
        self._done = False
        self._episode_reward = 0.0
        self._prev_score = self._compute_score()
        return self._get_observation()

    def step(self, action: Action) -> StepResult:
        if self._df is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._step_count += 1
        info: Dict[str, Any] = {"step": self._step_count}

        try:
            message = self._apply_operation(action)
            info["applied"] = message
        except Exception as exc:
            message = f"Operation failed: {exc}"
            info["error"] = message

        new_score = self._compute_score()
        breakdown = self._get_score_breakdown()
        delta = new_score - self._prev_score

        reward_value = round(_clamp(delta - self.STEP_PENALTY, -1.0, 1.0), 4)
        self._prev_score = new_score
        self._episode_reward = round(self._episode_reward + reward_value, 4)

        max_steps = TASKS[self._task_id]["max_steps"]
        if action.operation == "done" or self._step_count >= max_steps:
            self._done = True

        reward = Reward(value=reward_value, message=message, score_breakdown=breakdown)
        observation = self._get_observation()

        info["episode_reward"] = self._episode_reward
        info["quality_score"] = new_score

        return StepResult(
            observation=observation,
            reward=reward,
            done=self._done,
            info=info,
        )

    def state(self) -> Dict[str, Any]:
        public_columns = len(TASKS[self._task_id]["schema"]) if self._task_id else None
        return {
            "task_id": self._task_id,
            "step_count": self._step_count,
            "max_steps": TASKS[self._task_id]["max_steps"] if self._task_id else None,
            "done": self._done,
            "episode_reward": self._episode_reward,
            "quality_score": self._compute_score() if self._df is not None else None,
            "table_shape": [len(self._df), public_columns] if self._df is not None else None,
        }

    def list_tasks(self) -> List[TaskInfo]:
        return [
            TaskInfo(
                task_id=task_id,
                name=task["name"],
                description=task["description"],
                difficulty=task["difficulty"],
                max_steps=task["max_steps"],
                columns=list(task["schema"].keys()),
            )
            for task_id, task in TASKS.items()
        ]

    def _public_dataframe(self) -> pd.DataFrame:
        hidden_cols = {"__row_id__"}
        return self._df[[column for column in self._df.columns if column not in hidden_cols]].copy()

    def _raw_score_breakdown(self) -> Dict[str, float]:
        if self._df is None or self._original_df is None:
            return {"total": 0.0}
        return grade(self._task_id, self._df, self._original_df)

    def _compute_score(self) -> float:
        raw_total = float(self._raw_score_breakdown().get("total", 0.0))
        return _expose_task_total(raw_total)

    def _get_score_breakdown(self) -> Dict[str, float]:
        breakdown = self._raw_score_breakdown()
        sanitized: Dict[str, float] = {}
        for key, value in breakdown.items():
            if not isinstance(value, (int, float)):
                sanitized[key] = value
            elif key == "total":
                sanitized[key] = _expose_task_total(float(value))
            else:
                sanitized[key] = round(_clamp(float(value), 0.0, 1.0), 4)
        return sanitized

    def _get_observation(self) -> Observation:
        task = TASKS[self._task_id]
        public_df = self._public_dataframe().astype(object).where(pd.notnull(self._public_dataframe()), None)
        return Observation(
            task_id=self._task_id,
            task_name=task["name"],
            task_description=task["description"],
            table=public_df.to_dict(orient="records"),
            column_schema=task["schema"],
            quality_issues=self._detect_issues(),
            quality_score=self._compute_score(),
            step_count=self._step_count,
            max_steps=task["max_steps"],
            available_operations=ALLOWED_OPERATIONS,
        )

    def _detect_issues(self) -> List[str]:
        df = self._public_dataframe()
        task = TASKS[self._task_id]
        issues: List[str] = []

        subset = task.get("dup_subset")
        if subset:
            duplicate_count = int(df.duplicated(subset=subset, keep="first").sum())
            if duplicate_count > 0:
                issues.append(f"{duplicate_count} duplicate rows detected (key: [{', '.join(subset)}])")

        for column in df.columns:
            missing = int(df[column].isna().sum())
            if missing > 0:
                issues.append(f"Column '{column}': {missing} missing value(s)")

        for column in task.get("date_columns", []):
            bad_count = sum(
                1
                for value in df[column]
                if not pd.isna(value) and str(value).strip() not in {"", "None", "nan"} and not _is_iso_date(value)
            )
            if bad_count > 0:
                issues.append(f"Column '{column}': {bad_count} date(s) not in YYYY-MM-DD format")

        for column in task.get("phone_columns", []):
            bad_count = sum(
                1
                for value in df[column]
                if not pd.isna(value) and str(value).strip() not in {"", "None", "nan"} and not _is_valid_phone(value)
            )
            if bad_count > 0:
                issues.append(f"Column '{column}': {bad_count} phone number(s) not in +91-XXXXX-XXXXX format")

        for column in task.get("positive_columns", []):
            numeric = pd.to_numeric(df[column], errors="coerce")
            negative_count = int((numeric < 0).sum())
            if negative_count > 0:
                issues.append(f"Column '{column}': {negative_count} negative value(s) detected")

        for column, (lower, upper) in task.get("outlier_ranges", {}).items():
            numeric = pd.to_numeric(df[column], errors="coerce")
            outlier_count = int((~numeric.between(lower, upper)).sum())
            if outlier_count > 0:
                issues.append(f"Column '{column}': {outlier_count} value(s) outside valid range [{lower}, {upper}]")

        return issues if issues else ["No quality issues detected"]

    def _apply_operation(self, action: Action) -> str:
        task = TASKS[self._task_id]
        operation = action.operation.strip()
        column = action.column
        params = action.params or {}

        if operation == "remove_duplicates":
            subset = params.get("subset", task.get("dup_subset"))
            before = len(self._df)
            self._df = self._df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
            removed = before - len(self._df)
            return f"Removed {removed} duplicate row(s) (subset={subset})"

        if operation == "fill_missing":
            self._require_public_column(column)
            strategy = params.get("strategy", "mode")
            before_missing = int(self._df[column].isna().sum())

            if strategy == "mean":
                self._df[column] = self._df[column].fillna(self._df[column].mean())
            elif strategy == "median":
                self._df[column] = self._df[column].fillna(self._df[column].median())
            elif strategy == "mode":
                modes = self._df[column].mode(dropna=True)
                if len(modes) > 0:
                    self._df[column] = self._df[column].fillna(modes.iloc[0])
            elif strategy == "constant":
                self._df[column] = self._df[column].fillna(params.get("value", "N/A"))
            elif strategy == "mapping":
                source_column = params.get("source_column")
                mapping = params.get("mapping", {})
                self._require_public_column(source_column)
                if not isinstance(mapping, dict):
                    raise ValueError("mapping strategy requires a 'mapping' object")
                fill_count = 0
                missing_mask = self._df[column].isna()
                for row_index in self._df.index[missing_mask]:
                    source_value = self._df.at[row_index, source_column]
                    if source_value in mapping:
                        self._df.at[row_index, column] = mapping[source_value]
                        fill_count += 1
                after_missing = int(self._df[column].isna().sum())
                return (
                    f"Filled {fill_count} missing value(s) in '{column}' using strategy='mapping' "
                    f"from '{source_column}'"
                )
            else:
                raise ValueError(f"Unknown fill strategy '{strategy}'")

            after_missing = int(self._df[column].isna().sum())
            return f"Filled {before_missing - after_missing} missing value(s) in '{column}' using strategy='{strategy}'"

        if operation == "standardize_date":
            self._require_public_column(column)
            before_bad = sum(
                1
                for value in self._df[column]
                if not pd.isna(value) and str(value).strip() not in {"", "None", "nan"} and not _is_iso_date(value)
            )
            self._df[column] = self._df[column].apply(_safe_parse_date)
            after_bad = sum(
                1
                for value in self._df[column]
                if not pd.isna(value) and str(value).strip() not in {"", "None", "nan"} and not _is_iso_date(value)
            )
            return f"Standardized {before_bad - after_bad} date(s) in '{column}' to YYYY-MM-DD"

        if operation == "standardize_phone":
            self._require_public_column(column)
            before_bad = sum(
                1
                for value in self._df[column]
                if not pd.isna(value) and str(value).strip() not in {"", "None", "nan"} and not _is_valid_phone(value)
            )
            self._df[column] = self._df[column].apply(_normalize_phone)
            after_bad = sum(
                1
                for value in self._df[column]
                if not pd.isna(value) and str(value).strip() not in {"", "None", "nan"} and not _is_valid_phone(value)
            )
            return f"Normalized {before_bad - after_bad} phone number(s) in '{column}'"

        if operation == "remove_negative":
            self._require_public_column(column)
            before = len(self._df)
            numeric = pd.to_numeric(self._df[column], errors="coerce")
            self._df = self._df[numeric.isna() | (numeric >= 0)].reset_index(drop=True)
            return f"Removed {before - len(self._df)} row(s) with negative values in '{column}'"

        if operation == "clip_outliers":
            self._require_public_column(column)
            lower = params.get("lower")
            upper = params.get("upper")
            if lower is None and upper is None:
                if column not in task.get("outlier_ranges", {}):
                    raise ValueError(
                        f"clip_outliers requires 'lower' and/or 'upper' params for '{column}'"
                    )
                lower, upper = task["outlier_ranges"][column]

            before = 0
            numeric = pd.to_numeric(self._df[column], errors="coerce")
            if lower is not None:
                before += int((numeric < lower).sum())
            if upper is not None:
                before += int((numeric > upper).sum())
            self._df[column] = numeric.clip(lower=lower, upper=upper)
            return f"Clipped {before} outlier(s) in '{column}' to [{lower}, {upper}]"

        if operation == "done":
            return "Agent signaled end of episode"

        raise ValueError(f"Unknown operation '{operation}'. Allowed: {ALLOWED_OPERATIONS}")

    def _require_public_column(self, column: Optional[str]) -> None:
        if not column:
            raise ValueError("'column' field is required for this operation")
        if column not in TASKS[self._task_id]["schema"]:
            raise ValueError(
                f"Column '{column}' not found. Available: {list(TASKS[self._task_id]['schema'].keys())}"
            )
