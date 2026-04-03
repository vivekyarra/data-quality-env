"""
graders.py — Deterministic quality graders for all three tasks.

Each grader receives the *current* DataFrame and the *original* DataFrame
(as loaded at reset time) and returns a dict:
  { "total": float, "<component>_score": float, ... }

All scores are in [0.0, 1.0]. Graders are purely functional — no side effects.
"""

import re
from typing import Dict, Tuple

import pandas as pd

# ── Regex validators ──────────────────────────────────────────────────────────

_ISO_DATE   = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_PHONE_FMT  = re.compile(r'^\+91-\d{5}-\d{5}$')


def _is_iso_date(s: str) -> bool:
    return bool(_ISO_DATE.match(str(s).strip()))


def _is_valid_phone(s: str) -> bool:
    return bool(_PHONE_FMT.match(str(s).strip()))


def _pct_fixed(original_bad: int, current_bad: int) -> float:
    """Fraction of originally-bad values that are now fixed (0.0–1.0)."""
    if original_bad == 0:
        return 1.0
    return max(0.0, 1.0 - current_bad / original_bad)


# ── Task 1 — Customer Records ─────────────────────────────────────────────────

def grade_task1(df: pd.DataFrame, original_df: pd.DataFrame) -> Dict[str, float]:
    """
    Weights:
      duplicate_score      40%
      age_missing_score    30%
      email_missing_score  30%
    """
    dup_key = ["name", "email", "city"]

    orig_dups  = int(original_df.duplicated(subset=dup_key, keep="first").sum())
    curr_dups  = int(df.duplicated(subset=dup_key, keep="first").sum())
    dup_score  = _pct_fixed(orig_dups, curr_dups)

    orig_miss_age   = int(original_df["age"].isna().sum())
    curr_miss_age   = int(df["age"].isna().sum())
    age_score       = _pct_fixed(orig_miss_age, curr_miss_age)

    orig_miss_email = int(original_df["email"].isna().sum())
    curr_miss_email = int(df["email"].isna().sum())
    email_score     = _pct_fixed(orig_miss_email, curr_miss_email)

    total = 0.40 * dup_score + 0.30 * age_score + 0.30 * email_score

    return {
        "total":               round(total,       4),
        "duplicate_score":     round(dup_score,   4),
        "age_missing_score":   round(age_score,   4),
        "email_missing_score": round(email_score, 4),
    }


# ── Task 2 — Sales Data ───────────────────────────────────────────────────────

def grade_task2(df: pd.DataFrame, original_df: pd.DataFrame) -> Dict[str, float]:
    """
    Weights:
      date_format_score      25%
      phone_format_score     25%
      negative_amount_score  25%
      region_missing_score   25%
    """
    n = len(df)

    # Date format
    valid_dates = sum(1 for v in df["date"].astype(str) if _is_iso_date(v))
    date_score  = valid_dates / max(n, 1)

    # Phone format
    valid_phones = sum(1 for v in df["phone"].astype(str) if _is_valid_phone(v))
    phone_score  = valid_phones / max(n, 1)

    # Negative amounts
    orig_neg = int((original_df["amount"] < 0).sum())
    curr_neg = int((df["amount"] < 0).sum())
    neg_score = _pct_fixed(orig_neg, curr_neg)

    # Missing regions
    orig_miss_reg = int(original_df["region"].isna().sum())
    curr_miss_reg = int(df["region"].isna().sum())
    region_score  = _pct_fixed(orig_miss_reg, curr_miss_reg)

    total = 0.25 * (date_score + phone_score + neg_score + region_score)

    return {
        "total":                round(total,        4),
        "date_format_score":    round(date_score,   4),
        "phone_format_score":   round(phone_score,  4),
        "negative_amount_score":round(neg_score,    4),
        "region_missing_score": round(region_score, 4),
    }


# ── Task 3 — Healthcare Records ───────────────────────────────────────────────

def grade_task3(df: pd.DataFrame, original_df: pd.DataFrame) -> Dict[str, float]:
    """
    Weights (each 20%):
      duplicate_score      — duplicate patient_ids
      missing_value_score  — missing diagnosis / medication
      date_format_score    — dob + last_visit in ISO format
      outlier_score        — vitals inside physiological range
      negative_vital_score — no negative glucose values
    """
    # --- Duplicates ---
    orig_dups = int(original_df.duplicated(subset=["patient_id"], keep="first").sum())
    curr_dups = int(df.duplicated(subset=["patient_id"], keep="first").sum())
    dup_score = _pct_fixed(orig_dups, curr_dups)

    # --- Missing diagnosis / medication ---
    key_cols = ["diagnosis", "medication"]
    orig_miss = sum(int(original_df[c].isna().sum()) for c in key_cols)
    curr_miss = sum(int(df[c].isna().sum()) for c in key_cols)
    miss_score = _pct_fixed(orig_miss, curr_miss)

    # --- Date format ---
    date_cols = ["dob", "last_visit"]
    total_cells = len(df) * len(date_cols)
    valid_dates = sum(
        sum(1 for v in df[c].astype(str) if _is_iso_date(v))
        for c in date_cols
    )
    date_score = valid_dates / max(total_cells, 1)

    # --- Physiological outliers ---
    ranges: Dict[str, Tuple[int, int]] = {
        "bp_systolic":  (60, 200),
        "bp_diastolic": (40, 130),
        "glucose":      (50, 500),
    }
    orig_outliers = sum(
        int((~original_df[col].between(lo, hi)).sum())
        for col, (lo, hi) in ranges.items()
    )
    curr_outliers = sum(
        int((~df[col].between(lo, hi)).sum())
        for col, (lo, hi) in ranges.items()
    )
    outlier_score = _pct_fixed(orig_outliers, curr_outliers)

    # --- Negative vitals (glucose) ---
    orig_neg_vital = int((original_df["glucose"] < 0).sum())
    curr_neg_vital = int((df["glucose"] < 0).sum())
    neg_vital_score = _pct_fixed(orig_neg_vital, curr_neg_vital)

    total = 0.20 * (dup_score + miss_score + date_score + outlier_score + neg_vital_score)

    return {
        "total":               round(total,           4),
        "duplicate_score":     round(dup_score,       4),
        "missing_value_score": round(miss_score,      4),
        "date_format_score":   round(date_score,      4),
        "outlier_score":       round(outlier_score,   4),
        "negative_vital_score":round(neg_vital_score, 4),
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────

_GRADERS = {
    "task1_easy":   grade_task1,
    "task2_medium": grade_task2,
    "task3_hard":   grade_task3,
}


def grade(task_id: str, df: pd.DataFrame, original_df: pd.DataFrame) -> Dict[str, float]:
    """Call the correct grader for *task_id* and return its score dict."""
    if task_id not in _GRADERS:
        raise ValueError(f"No grader for task_id='{task_id}'")
    return _GRADERS[task_id](df, original_df)
