from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_MOVIE_NAME, ensure_project_dirs
from src.re.schema import (
    DEFAULT_RELATIONSHIP_LABEL,
    get_global_category,
    normalize_relationship_label,
)
from src.utils.io import load_csv, print_file_summary, save_csv
from src.utils.paths import get_movie_re_clean_output_path


STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
RAW_RESPONSE_COLUMN = "raw_response"

BASE_COLUMN_ORDER = [
    "movie_name",
    "movie_idx",
    "conversation_id",
    "utterance_id",
    "timestamp",
    "turn_index",
    "utterance_order",
    "speaker_id",
    "speaker_name",
    "reply_to",
    "listener_id",
    "listener_name",
    "current_turn",
    "current_text",
    "context_text",
    "relationship_type",
    "global_category",
    "confidence",
    "evidence",
    "status",
    "parse_success",
    "error",
    RAW_RESPONSE_COLUMN,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize CSV-only relationship extraction outputs."
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=DEFAULT_MOVIE_NAME,
        help=f'Movie name used to locate movie-specific RE files. Default: "{DEFAULT_MOVIE_NAME}".',
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Optional explicit input CSV path. Defaults to data/processed/<movie>/re_clean.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional explicit output CSV path. Defaults to data/processed/<movie>/re_clean.csv.",
    )
    parser.add_argument(
        "--review-sheet-path",
        type=str,
        default=None,
        help="Optional path for a reviewer-friendly CSV with editable review columns.",
    )
    return parser.parse_args()


def normalize_prompt_value(value: object, fallback: str = "") -> str:
    """Normalize metadata values from CSV records."""
    if value is None or pd.isna(value):
        return fallback
    text = " ".join(str(value).strip().split())
    return text if text else fallback


def normalize_optional_value(value: object) -> Any:
    """Convert pandas missing values to None while preserving real values."""
    if value is None or pd.isna(value):
        return None
    return value


def get_record_movie_name(record: dict[str, Any], default_movie: str) -> str:
    """Return normalized movie name for movie-specific schema lookup."""
    return normalize_prompt_value(record.get("movie_name"), fallback=default_movie).strip().lower()


def normalize_status(value: object, relationship_type: str, parse_success: bool) -> str:
    """Normalize status values for resumable CSV output."""
    status = normalize_prompt_value(value, fallback="").strip().lower()
    if status in {STATUS_SUCCESS, STATUS_FAILED, STATUS_SKIPPED}:
        return status
    if relationship_type != DEFAULT_RELATIONSHIP_LABEL or parse_success:
        return STATUS_SUCCESS
    return STATUS_FAILED


def normalize_parse_success(value: object, relationship_type: str) -> bool:
    """Normalize parse_success to bool."""
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return relationship_type != DEFAULT_RELATIONSHIP_LABEL
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return relationship_type != DEFAULT_RELATIONSHIP_LABEL


def normalize_confidence(value: object) -> float:
    """Normalize confidence to a bounded float in [0, 1]."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(confidence):
        return 0.0
    return max(0.0, min(1.0, confidence))


def normalize_re_record(
    record: dict[str, Any],
    default_movie: str = DEFAULT_MOVIE_NAME,
) -> dict[str, Any]:
    """Normalize one CSV relationship extraction record into a stable schema."""
    movie_name = get_record_movie_name(record, default_movie=default_movie)

    relationship_type = normalize_relationship_label(
        record.get("relationship_type"),
        movie_name=movie_name,
    )
    global_category = get_global_category(
        relationship_type,
        movie_name=movie_name,
    )
    parse_success = normalize_parse_success(
        record.get("parse_success"),
        relationship_type=relationship_type,
    )
    status = normalize_status(
        record.get("status"),
        relationship_type=relationship_type,
        parse_success=parse_success,
    )

    error = normalize_optional_value(record.get("error"))
    if status == STATUS_SUCCESS:
        error = None

    normalized_record = {
        "movie_name": movie_name,
        "movie_idx": normalize_optional_value(record.get("movie_idx")),
        "conversation_id": normalize_prompt_value(record.get("conversation_id")),
        "utterance_id": normalize_prompt_value(record.get("utterance_id")),
        "timestamp": normalize_optional_value(record.get("timestamp")),
        "turn_index": normalize_optional_value(record.get("turn_index")),
        "utterance_order": normalize_optional_value(record.get("utterance_order")),
        "speaker_id": normalize_prompt_value(record.get("speaker_id")),
        "speaker_name": normalize_optional_value(record.get("speaker_name")),
        "reply_to": normalize_optional_value(record.get("reply_to")),
        "listener_id": normalize_optional_value(record.get("listener_id")),
        "listener_name": normalize_optional_value(record.get("listener_name")),
        "current_turn": normalize_optional_value(record.get("current_turn")) or "",
        "current_text": normalize_optional_value(record.get("current_text")) or "",
        "context_text": normalize_optional_value(record.get("context_text")) or "",
        "relationship_type": relationship_type,
        "global_category": global_category,
        "confidence": normalize_confidence(record.get("confidence")),
        "evidence": normalize_optional_value(record.get("evidence")) or "",
        "status": status,
        "parse_success": parse_success,
        "error": error,
        RAW_RESPONSE_COLUMN: normalize_optional_value(record.get(RAW_RESPONSE_COLUMN)),
    }

    return normalized_record


def normalize_re_records(
    records: list[dict[str, Any]],
    default_movie: str = DEFAULT_MOVIE_NAME,
) -> list[dict[str, Any]]:
    """Normalize a batch of CSV RE records."""
    return [normalize_re_record(record, default_movie=default_movie) for record in records]


def records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert normalized records into a DataFrame with stable column order."""
    df = pd.DataFrame(records)
    for column in BASE_COLUMN_ORDER:
        if column not in df.columns:
            df[column] = None
    return df[BASE_COLUMN_ORDER]


def build_review_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Create a reviewer-friendly sheet with extra columns for manual correction."""
    review_df = df.copy()
    review_df["reviewed_relationship_type"] = review_df["relationship_type"]
    review_df["reviewed_global_category"] = review_df["global_category"]
    review_df["reviewer_notes"] = ""
    return review_df


def load_and_normalize_re_csv(
    path: str | Path,
    default_movie: str = DEFAULT_MOVIE_NAME,
) -> pd.DataFrame:
    """Load CSV RE output and return a normalized DataFrame."""
    df = load_csv(path)
    records = df.to_dict(orient="records")
    normalized_records = normalize_re_records(records, default_movie=default_movie)
    return records_to_dataframe(normalized_records)


def sort_re_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sort normalized RE output in dialogue order when ordering columns exist."""
    output_df = df.copy()
    sort_cols = [
        column
        for column in ["movie_name", "conversation_id", "turn_index", "utterance_order", "utterance_id"]
        if column in output_df.columns
    ]
    if sort_cols:
        output_df = output_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return output_df


def save_re_outputs(
    df: pd.DataFrame,
    clean_output_path: str | Path,
    review_sheet_path: str | Path | None = None,
) -> None:
    """Save normalized RE outputs and optionally a review sheet."""
    output_df = sort_re_dataframe(df)
    save_csv(output_df, clean_output_path, index=False)

    if review_sheet_path is not None:
        review_df = build_review_sheet(output_df)
        save_csv(review_df, review_sheet_path, index=False)


def normalize_missing_relationships(
    df: pd.DataFrame,
    movie_name: str = DEFAULT_MOVIE_NAME,
) -> pd.DataFrame:
    """Ensure missing/invalid relationship values are replaced with the default label."""
    normalized_df = df.copy()
    normalized_df["relationship_type"] = normalized_df["relationship_type"].apply(
        lambda label: normalize_relationship_label(label, movie_name=movie_name)
    )
    normalized_df["global_category"] = normalized_df["relationship_type"].apply(
        lambda label: get_global_category(label, movie_name=movie_name)
    )
    return normalized_df


def print_postprocess_summary(df: pd.DataFrame) -> None:
    """Print compact postprocessing summary statistics."""
    print(f"Rows normalized: {len(df)}")
    if "status" in df.columns:
        print("\nStatus counts:")
        for label, count in df["status"].value_counts(dropna=False).items():
            print(f"- {label}: {count}")
    if "relationship_type" in df.columns:
        print("\nRelationship label counts:")
        for label, count in df["relationship_type"].value_counts(dropna=False).items():
            print(f"- {label}: {count}")
    if "global_category" in df.columns:
        print("\nGlobal category counts:")
        for label, count in df["global_category"].value_counts(dropna=False).items():
            print(f"- {label}: {count}")
    if "parse_success" in df.columns:
        print(f"\nParse success rows: {int(df['parse_success'].sum())}/{len(df)}")


def postprocess_re_outputs(
    movie_name: str = DEFAULT_MOVIE_NAME,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    review_sheet_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load, normalize, and save CSV relationship extraction outputs for one movie."""
    movie = movie_name.strip().lower()
    ensure_project_dirs(movie)

    csv_input_path = Path(input_path) if input_path is not None else get_movie_re_clean_output_path(movie)
    clean_output_path = Path(output_path) if output_path is not None else get_movie_re_clean_output_path(movie)
    review_output_path = Path(review_sheet_path) if review_sheet_path is not None else None

    print(f"Loading CSV RE outputs from: {csv_input_path}")
    df = load_and_normalize_re_csv(csv_input_path, default_movie=movie)
    save_re_outputs(
        df,
        clean_output_path=clean_output_path,
        review_sheet_path=review_output_path,
    )

    print("\nRelationship extraction postprocessing complete.")
    print_file_summary(clean_output_path, label="Saved clean CSV RE outputs")
    if review_output_path is not None:
        print_file_summary(review_output_path, label="Saved RE review sheet")
    print_postprocess_summary(df)
    return df


def main() -> None:
    args = parse_args()
    postprocess_re_outputs(
        movie_name=args.movie,
        input_path=args.input_path,
        output_path=args.output_path,
        review_sheet_path=args.review_sheet_path,
    )


if __name__ == "__main__":
    main()