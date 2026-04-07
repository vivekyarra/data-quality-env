"""
Hidden-target graders for DataQualityEnv.

Each grader compares the agent-visible dataframe against a hidden target
dataframe using stable row ids. This makes row deletion and accidental data loss
visible to the scorer instead of letting agents game issue counts.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import pandas as pd

from tasks import TASKS, make_row_id

ROW_FIDELITY_THRESHOLD = 0.90


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _is_missing(value: Any) -> bool:
    return pd.isna(value)


def _values_equal(left: Any, right: Any) -> bool:
    if _is_missing(left) and _is_missing(right):
        return True
    return left == right


def _with_hidden_row_ids(task_id: str, df: pd.DataFrame) -> pd.DataFrame:
    if "__row_id__" in df.columns:
        return df.copy()
    enriched = df.copy()
    enriched["__row_id__"] = [make_row_id(task_id, idx) for idx in range(len(enriched))]
    return enriched


def _target_dataframe(task_id: str) -> pd.DataFrame:
    task = TASKS[task_id]
    rows = []
    for source_row, row in zip(task["target_source_rows"], task["target_data"]):
        enriched = dict(row)
        enriched["__row_id__"] = make_row_id(task_id, source_row)
        rows.append(enriched)
    return pd.DataFrame(rows)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _row_fidelity_score(current_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple[float, float, float]:
    produced_ids = set(current_df["__row_id__"].tolist())
    target_ids = set(target_df["__row_id__"].tolist())
    shared_ids = produced_ids & target_ids

    current_indexed = current_df.set_index("__row_id__")
    target_indexed = target_df.set_index("__row_id__")
    public_columns = [column for column in target_df.columns if column != "__row_id__"]

    matched_correct = 0
    for row_id in shared_ids:
        if all(
            _values_equal(current_indexed.at[row_id, column], target_indexed.at[row_id, column])
            for column in public_columns
        ):
            matched_correct += 1

    precision = matched_correct / len(produced_ids) if produced_ids else 0.0
    recall = matched_correct / len(target_ids) if target_ids else 1.0
    return _f1(precision, recall), precision, recall


def _column_match_score(
    current_df: pd.DataFrame,
    original_df: pd.DataFrame,
    target_df: pd.DataFrame,
    columns: Iterable[str],
) -> float:
    target_indexed = target_df.set_index("__row_id__")
    target_ids = target_indexed.index
    original_indexed = original_df.set_index("__row_id__").reindex(target_ids)
    current_indexed = current_df.set_index("__row_id__").reindex(target_ids)

    dirty_cells = 0
    matched_cells = 0

    for row_id in target_ids:
        for column in columns:
            original_value = original_indexed.at[row_id, column]
            target_value = target_indexed.at[row_id, column]
            if not _values_equal(original_value, target_value):
                dirty_cells += 1
                current_value = current_indexed.at[row_id, column]
                if _values_equal(current_value, target_value):
                    matched_cells += 1

    if dirty_cells == 0:
        return 1.0
    return matched_cells / dirty_cells


def _removed_row_score(current_df: pd.DataFrame, task_id: str, source_rows: Iterable[int]) -> float:
    source_row_ids = [make_row_id(task_id, row) for row in source_rows]
    if not source_row_ids:
        return 1.0
    current_ids = set(current_df["__row_id__"].tolist())
    removed = sum(1 for row_id in source_row_ids if row_id not in current_ids)
    return removed / len(source_row_ids)


def _sanitize_breakdown(breakdown: Dict[str, float]) -> Dict[str, float]:
    return {
        key: round(_clamp(float(value), 0.0, 1.0), 4) if isinstance(value, (int, float)) else value
        for key, value in breakdown.items()
    }


def _grade(task_id: str, df: pd.DataFrame, original_df: pd.DataFrame) -> Dict[str, float]:
    task = TASKS[task_id]
    current_df = _with_hidden_row_ids(task_id, df)
    original_with_ids = _with_hidden_row_ids(task_id, original_df)
    target_df = _target_dataframe(task_id)

    row_fidelity, row_precision, row_recall = _row_fidelity_score(current_df, target_df)

    breakdown: Dict[str, float] = {
        "row_fidelity_score": row_fidelity,
    }

    for score_name, spec in task["score_components"].items():
        if spec["type"] == "columns":
            breakdown[score_name] = _column_match_score(
                current_df=current_df,
                original_df=original_with_ids,
                target_df=target_df,
                columns=spec["columns"],
            )
        elif spec["type"] == "removed_rows":
            breakdown[score_name] = _removed_row_score(
                current_df=current_df,
                task_id=task_id,
                source_rows=spec["source_rows"],
            )
        else:
            raise ValueError(f"Unknown score component type: {spec['type']}")

    raw_total = 0.0
    for score_name, weight in task["score_weights"].items():
        raw_total += weight * breakdown[score_name]

    if row_precision < ROW_FIDELITY_THRESHOLD or row_recall < ROW_FIDELITY_THRESHOLD:
        raw_total = min(raw_total, row_fidelity)

    breakdown["total"] = raw_total
    return _sanitize_breakdown(breakdown)


def grade(task_id: str, df: pd.DataFrame, original_df: pd.DataFrame) -> Dict[str, float]:
    if task_id not in TASKS:
        raise ValueError(f"No grader for task_id='{task_id}'")
    return _grade(task_id, df, original_df)
