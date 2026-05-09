"""Prepare the overlap between risk-predicted rows and gold power/respect labels.

This module builds the input dataset for the graph-guided translation pipeline.
It combines three sources:

1. Risk classifier outputs for each movie:
   data/data_prelabel_predictions/<movie_slug>/dialogue_metadata_risk_predictions.csv
2. Gold human-labeled power/respect annotations:
   data/labeled/power_respect_labels.csv
3. Generative relationship extraction outputs for each movie:
   data/processed/<movie_slug>/re_clean.csv

The output keeps only rows that are both risk-relevant and present in the gold
agreement dataset, then attaches relationship type from GE results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_movies
from src.utils.io import ensure_parent_dir, load_csv, save_csv
from src.utils.paths import project_path, slugify_movie_name

TARGET_RISK_COLUMNS = [
    "_risk_social",
    "_risk_pragmatic",
    "_risk_register",
    "_risk_ambiguity",
]

JOIN_KEYS = ["movie_name", "conversation_id", "utterance_id"]

DEFAULT_RISK_FILENAME = "dialogue_metadata_risk_predictions.csv"
DEFAULT_GOLD_LABELS_PATH = project_path(Path("data") / "labeled" / "power_respect_labels.csv")
DEFAULT_OUTPUT_PATH = project_path(Path("data") / "interim" / "risk_gold_overlap.csv")
DEFAULT_SUMMARY_PATH = project_path(Path("outputs") / "tables" / "overlap_summary.csv")


@dataclass(frozen=True)
class RiskGoldOverlapResult:
    """Paths and summary statistics for the risk-gold overlap step."""

    output_path: Path
    summary_path: Path
    rows: int
    gold_rows: int
    risk_rows: int
    movies_processed: int
    movies_with_risk_file: int
    movies_with_re_file: int


def normalize_id_column(series: pd.Series) -> pd.Series:
    """Normalize ID columns for robust joins across CSV sources."""
    return series.astype(str).str.strip()


def normalize_movie_name(series: pd.Series) -> pd.Series:
    """Normalize movie names for robust joins."""
    return series.astype(str).str.strip().str.lower()


def normalize_label_value(series: pd.Series) -> pd.Series:
    """Normalize label columns for agreement checks while preserving missing values."""
    return series.astype("string").str.strip().str.lower()


# Helper to check for clear, present labels
def is_clear_label(series: pd.Series) -> pd.Series:
    """Return True for labels that are present and not unclear."""
    normalized = normalize_label_value(series)
    return normalized.notna() & normalized.ne("") & normalized.ne("nan") & normalized.ne("unclear")


def filter_gold_agreement_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only judge-agreement rows with clear final power/respect labels.

    The source power_respect_labels.csv may include disagreement rows and unclear
    rows. For the graph-guided translation experiment, we only use gold rows
    where judge1 and judge2 agree for both power and respect, and where the final
    labels are not unclear.
    """
    required_columns = [
        "judge1_power_dynamic",
        "judge2_power_dynamic",
        "judge1_respect_level",
        "judge2_respect_level",
        "final_power_dynamic",
        "final_respect_level",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Gold agreement filtering requires these columns in power_respect_labels.csv: "
            f"{missing}"
        )

    updated = df.copy()
    for column in required_columns:
        updated[column] = normalize_label_value(updated[column])

    power_agreement = updated["judge1_power_dynamic"] == updated["judge2_power_dynamic"]
    respect_agreement = updated["judge1_respect_level"] == updated["judge2_respect_level"]
    clear_power = is_clear_label(updated["final_power_dynamic"])
    clear_respect = is_clear_label(updated["final_respect_level"])

    return updated.loc[power_agreement & respect_agreement & clear_power & clear_respect].copy()


def ensure_join_keys(df: pd.DataFrame, source_name: str) -> None:
    """Validate that a dataframe contains the required join keys."""
    missing = [column for column in JOIN_KEYS if column not in df.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required join columns: {missing}")


def normalize_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a dataframe with normalized join keys."""
    normalized = df.copy()
    normalized["movie_name"] = normalize_movie_name(normalized["movie_name"])
    normalized["conversation_id"] = normalize_id_column(normalized["conversation_id"])
    normalized["utterance_id"] = normalize_id_column(normalized["utterance_id"])
    return normalized


def to_bool_risk(series: pd.Series) -> pd.Series:
    """Convert a risk prediction column to boolean values."""
    if series.dtype == bool:
        return series.fillna(False)

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "risk", "risky"})


def add_missing_risk_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all target risk columns exist."""
    updated = df.copy()
    for column in TARGET_RISK_COLUMNS:
        if column not in updated.columns:
            updated[column] = False
    return updated


def filter_target_risk_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with at least one target risk type."""
    updated = add_missing_risk_columns(df)
    for column in TARGET_RISK_COLUMNS:
        updated[column] = to_bool_risk(updated[column])

    risk_mask = updated[TARGET_RISK_COLUMNS].any(axis=1)
    filtered = updated.loc[risk_mask].copy()
    filtered["target_risk_count"] = filtered[TARGET_RISK_COLUMNS].sum(axis=1).astype(int)
    filtered["target_risk_types"] = filtered[TARGET_RISK_COLUMNS].apply(
        lambda row: ";".join(column for column, value in row.items() if bool(value)),
        axis=1,
    )
    return filtered


def get_risk_prediction_path(movie_name: str, risk_filename: str = DEFAULT_RISK_FILENAME) -> Path:
    """Return the risk prediction file path for one movie."""
    movie_slug = slugify_movie_name(movie_name)
    return project_path(Path("data") / "data_prelabel_predictions" / movie_slug / risk_filename)


def get_relationship_extraction_path(movie_name: str) -> Path:
    """Return the GE relationship extraction file path for one movie."""
    movie_slug = slugify_movie_name(movie_name)
    return project_path(Path("data") / "processed" / movie_slug / "re_clean.csv")


def load_movie_risk_predictions(movie_name: str, risk_filename: str = DEFAULT_RISK_FILENAME) -> pd.DataFrame:
    """Load and filter target-risk rows for one movie."""
    path = get_risk_prediction_path(movie_name, risk_filename=risk_filename)
    if not path.exists():
        return pd.DataFrame()

    df = load_csv(path)
    if "movie_name" not in df.columns:
        df["movie_name"] = movie_name
    ensure_join_keys(df, str(path))
    df = normalize_join_keys(df)
    df = filter_target_risk_rows(df)
    df["risk_source_path"] = str(path)
    return df


def load_all_risk_predictions(movie_names: list[str], risk_filename: str = DEFAULT_RISK_FILENAME) -> tuple[pd.DataFrame, int]:
    """Load filtered risk predictions for all configured movies."""
    frames: list[pd.DataFrame] = []
    files_found = 0

    for movie_name in movie_names:
        movie_df = load_movie_risk_predictions(movie_name, risk_filename=risk_filename)
        if not movie_df.empty:
            files_found += 1
            frames.append(movie_df)

    if not frames:
        return pd.DataFrame(), files_found

    return pd.concat(frames, ignore_index=True), files_found


def load_gold_labels(gold_labels_path: Path = DEFAULT_GOLD_LABELS_PATH) -> pd.DataFrame:
    """Load the gold power/respect label dataset."""
    if not gold_labels_path.exists():
        raise FileNotFoundError(f"Gold power/respect label file not found: {gold_labels_path}")

    df = load_csv(gold_labels_path)
    ensure_join_keys(df, str(gold_labels_path))
    df = normalize_join_keys(df)

    required_label_columns = ["final_power_dynamic", "final_respect_level"]
    missing = [column for column in required_label_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Gold label file is missing required label columns: {missing}")

    return filter_gold_agreement_rows(df)


def select_gold_columns(gold_df: pd.DataFrame) -> pd.DataFrame:
    """Keep gold columns useful for graph-guided translation."""
    preferred_columns = [
        *JOIN_KEYS,
        "speaker_id",
        "speaker_name",
        "listener_id",
        "listener_name",
        "current_text",
        "context_text",
        "context_6_turns",
        "final_power_dynamic",
        "final_respect_level",
        "judge1_power_dynamic",
        "judge2_power_dynamic",
        "judge1_respect_level",
        "judge2_respect_level",
    ]
    existing_columns = [column for column in preferred_columns if column in gold_df.columns]
    return gold_df[existing_columns].copy()


def load_movie_relationship_extraction(movie_name: str) -> pd.DataFrame:
    """Load GE relationship extraction results for one movie."""
    path = get_relationship_extraction_path(movie_name)
    if not path.exists():
        return pd.DataFrame()

    df = load_csv(path)
    if "movie_name" not in df.columns:
        df["movie_name"] = movie_name
    ensure_join_keys(df, str(path))
    df = normalize_join_keys(df)

    preferred_columns = [
        *JOIN_KEYS,
        "relationship_type",
        "global_category",
        "confidence",
        "evidence",
        "status",
        "parse_success",
    ]
    existing_columns = [column for column in preferred_columns if column in df.columns]
    df = df[existing_columns].copy()
    df["re_source_path"] = str(path)
    return df


def load_all_relationship_extractions(movie_names: list[str]) -> tuple[pd.DataFrame, int]:
    """Load GE relationship extraction results for all configured movies."""
    frames: list[pd.DataFrame] = []
    files_found = 0

    for movie_name in movie_names:
        movie_df = load_movie_relationship_extraction(movie_name)
        if not movie_df.empty:
            files_found += 1
            frames.append(movie_df)

    if not frames:
        return pd.DataFrame(), files_found

    return pd.concat(frames, ignore_index=True), files_found


def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean common duplicate columns created by merges."""
    updated = df.copy()

    for base_column in [
        "speaker_id",
        "speaker_name",
        "listener_id",
        "listener_name",
        "current_text",
        "context_text",
        "context_6_turns",
    ]:
        risk_column = f"{base_column}_risk"
        gold_column = f"{base_column}_gold"
        if risk_column in updated.columns and gold_column in updated.columns:
            updated[base_column] = updated[gold_column].combine_first(updated[risk_column])
            updated = updated.drop(columns=[risk_column, gold_column])
        elif gold_column in updated.columns:
            updated = updated.rename(columns={gold_column: base_column})
        elif risk_column in updated.columns:
            updated = updated.rename(columns={risk_column: base_column})

    return updated


# Remove rows without usable power, respect, or relationship labels
def filter_valid_social_overlap_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows without usable power, respect, or relationship labels."""
    required_columns = ["final_power_dynamic", "final_respect_level", "relationship_type"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Risk-gold overlap output is missing required social columns: {missing}")

    valid_power = is_clear_label(df["final_power_dynamic"])
    valid_respect = is_clear_label(df["final_respect_level"])
    valid_relationship = is_clear_label(df["relationship_type"])
    return df.loc[valid_power & valid_respect & valid_relationship].copy()


def build_risk_gold_overlap(
    *,
    movie_names: list[str] | None = None,
    risk_filename: str = DEFAULT_RISK_FILENAME,
    gold_labels_path: Path = DEFAULT_GOLD_LABELS_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
) -> RiskGoldOverlapResult:
    """Create the risk-gold overlap dataset for graph-guided translation."""
    if movie_names is None:
        movie_names = load_movies()

    risk_df, risk_files_found = load_all_risk_predictions(movie_names, risk_filename=risk_filename)
    if risk_df.empty:
        raise FileNotFoundError(
            "No target-risk prediction rows found. Expected files like: "
            "data/data_prelabel_predictions/<movie_slug>/dialogue_metadata_risk_predictions.csv"
        )

    gold_df = select_gold_columns(load_gold_labels(gold_labels_path))
    gold_rows = len(gold_df)
    risk_rows = len(risk_df)
    re_df, re_files_found = load_all_relationship_extractions(movie_names)

    overlap_df = risk_df.merge(
        gold_df,
        on=JOIN_KEYS,
        how="inner",
        suffixes=("_risk", "_gold"),
    )
    overlap_df = coalesce_duplicate_columns(overlap_df)

    if not re_df.empty:
        overlap_df = overlap_df.merge(re_df, on=JOIN_KEYS, how="left")
    else:
        overlap_df["relationship_type"] = "unclear"
        overlap_df["global_category"] = "unclear"

    if "relationship_type" in overlap_df.columns:
        overlap_df["relationship_type"] = overlap_df["relationship_type"].fillna("unclear")
    if "global_category" in overlap_df.columns:
        overlap_df["global_category"] = overlap_df["global_category"].fillna("unclear")

    overlap_df = filter_valid_social_overlap_rows(overlap_df)

    sort_columns = [column for column in ["movie_name", "conversation_id", "turn_index", "utterance_id"] if column in overlap_df.columns]
    if sort_columns:
        overlap_df = overlap_df.sort_values(sort_columns).reset_index(drop=True)

    ensure_parent_dir(output_path)
    save_csv(overlap_df, output_path)

    summary_df = make_overlap_summary(
        overlap_df=overlap_df,
        gold_df=gold_df,
        risk_df=risk_df,
        movie_names=movie_names,
        risk_files_found=risk_files_found,
        re_files_found=re_files_found,
    )
    ensure_parent_dir(summary_path)
    save_csv(summary_df, summary_path)

    return RiskGoldOverlapResult(
        output_path=output_path,
        summary_path=summary_path,
        rows=len(overlap_df),
        gold_rows=gold_rows,
        risk_rows=risk_rows,
        movies_processed=len(movie_names),
        movies_with_risk_file=risk_files_found,
        movies_with_re_file=re_files_found,
    )


def make_overlap_summary(
    *,
    overlap_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    movie_names: list[str],
    risk_files_found: int,
    re_files_found: int,
) -> pd.DataFrame:
    """Build a compact summary table for the overlap step."""
    rows: list[dict[str, Any]] = []

    for movie_name in movie_names:
        normalized_movie_name = movie_name.strip().lower()
        movie_overlap_df = overlap_df[overlap_df["movie_name"] == normalized_movie_name]
        movie_gold_df = gold_df[gold_df["movie_name"] == normalized_movie_name]
        movie_risk_df = risk_df[risk_df["movie_name"] == normalized_movie_name]
        row: dict[str, Any] = {
            "movie_name": normalized_movie_name,
            "gold_rows": len(movie_gold_df),
            "target_risk_rows": len(movie_risk_df),
            "overlap_rows": len(movie_overlap_df),
            "overlap_rate_vs_gold": len(movie_overlap_df) / len(movie_gold_df) if len(movie_gold_df) else 0.0,
            "overlap_rate_vs_target_risk": len(movie_overlap_df) / len(movie_risk_df) if len(movie_risk_df) else 0.0,
        }
        for risk_column in TARGET_RISK_COLUMNS:
            row[f"{risk_column}_overlap_rows"] = int(movie_overlap_df[risk_column].sum()) if risk_column in movie_overlap_df.columns else 0
            row[f"{risk_column}_target_risk_rows"] = int(movie_risk_df[risk_column].sum()) if risk_column in movie_risk_df.columns else 0
        rows.append(row)

    total_row: dict[str, Any] = {
        "movie_name": "__total__",
        "gold_rows": len(gold_df),
        "target_risk_rows": len(risk_df),
        "overlap_rows": len(overlap_df),
        "overlap_rate_vs_gold": len(overlap_df) / len(gold_df) if len(gold_df) else 0.0,
        "overlap_rate_vs_target_risk": len(overlap_df) / len(risk_df) if len(risk_df) else 0.0,
        "movies_configured": len(movie_names),
        "movies_with_risk_file": risk_files_found,
        "movies_with_re_file": re_files_found,
    }
    for risk_column in TARGET_RISK_COLUMNS:
        total_row[f"{risk_column}_overlap_rows"] = int(overlap_df[risk_column].sum()) if risk_column in overlap_df.columns else 0
        total_row[f"{risk_column}_target_risk_rows"] = int(risk_df[risk_column].sum()) if risk_column in risk_df.columns else 0
    rows.append(total_row)

    return pd.DataFrame(rows)


def print_overlap_summary(result: RiskGoldOverlapResult) -> None:
    """Print a compact summary of the overlap output."""
    print("Risk-gold overlap preparation complete")
    print(f"  rows                 : {result.rows}")
    print(f"  gold agreement rows  : {result.gold_rows}")
    print(f"  target risk rows     : {result.risk_rows}")
    print(f"  movies processed     : {result.movies_processed}")
    print(f"  movies with risk file: {result.movies_with_risk_file}")
    print(f"  movies with RE file  : {result.movies_with_re_file}")
    print(f"  output               : {result.output_path}")
    print(f"  summary              : {result.summary_path}")


__all__ = [
    "DEFAULT_GOLD_LABELS_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_RISK_FILENAME",
    "DEFAULT_SUMMARY_PATH",
    "JOIN_KEYS",
    "TARGET_RISK_COLUMNS",
    "RiskGoldOverlapResult",
    "build_risk_gold_overlap",
    "filter_target_risk_rows",
    "filter_gold_agreement_rows",
    "filter_valid_social_overlap_rows",
    "get_relationship_extraction_path",
    "get_risk_prediction_path",
    "load_all_relationship_extractions",
    "load_all_risk_predictions",
    "load_gold_labels",
    "make_overlap_summary",
    "print_overlap_summary",
]