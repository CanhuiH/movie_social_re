from __future__ import annotations

import argparse
import re
from typing import Any

import pandas as pd

from src.config import DEFAULT_MOVIE_NAME, ensure_project_dirs
from src.utils.io import load_csv, print_file_summary, require_columns, save_csv
from src.utils.paths import (
    get_movie_clean_dialogue_path,
    get_movie_dialogue_metadata_path,
)


# Expected input format from src/data/extract_dialogue.py and user-provided raw CSVs.
INPUT_COLUMNS = [
    "movie_name",
    "movie_idx",
    "conversation_id",
    "utterance_id",
    "timestamp",
    "speaker_id",
    "speaker_name",
    "reply_to",
    "listener_id",
    "listener_name",
    "text",
]

REQUIRED_INPUT_COLUMNS = set(INPUT_COLUMNS)

# Clean output format expected by src/data/build_turn_windows.py.
# timestamp is preserved for compatibility but is not used for ordering.
OUTPUT_COLUMNS = [
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
    "text",
]


MISSING_TEXT_VALUES = {
    "",
    "none",
    "null",
    "nan",
    "unknown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess extracted movie dialogue metadata into a clean dialogue table."
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=DEFAULT_MOVIE_NAME,
        help=f'Movie title used to locate movie-specific interim files. Default: "{DEFAULT_MOVIE_NAME}".',
    )
    parser.add_argument(
        "--drop-missing-listener",
        action="store_true",
        help="If set, drop rows where listener_id is missing after cleaning.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    """Normalize dialogue text while preserving meaningful punctuation."""
    if pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_optional_string(value: object) -> str | None:
    """Normalize optional string values and convert empty placeholders to None."""
    if pd.isna(value):
        return None

    text = normalize_text(value)
    if text.lower() in MISSING_TEXT_VALUES:
        return None
    return text or None


def normalize_required_string(value: object, fallback: str = "UNKNOWN") -> str:
    """Normalize required string values and use a fallback if missing."""
    text = normalize_optional_string(value)
    return text if text is not None else fallback


def add_turn_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add a stable zero-based within-conversation turn_index column."""
    indexed = df.copy()
    indexed["turn_index"] = indexed.groupby("conversation_id").cumcount()
    return indexed


def add_utterance_order(df: pd.DataFrame) -> pd.DataFrame:
    """Add a stable zero-based within-conversation utterance_order column."""
    ordered = df.copy()
    ordered["utterance_order"] = ordered.groupby("conversation_id").cumcount()
    return ordered


def standardize_dialogue_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw dialogue columns into consistent types and values."""
    require_columns(df, REQUIRED_INPUT_COLUMNS, label="Extracted dialogue metadata")

    cleaned = df.copy()

    cleaned["movie_name"] = cleaned["movie_name"].apply(
        lambda value: normalize_required_string(value)
    )
    cleaned["movie_idx"] = cleaned["movie_idx"].apply(normalize_optional_string)
    cleaned["conversation_id"] = cleaned["conversation_id"].apply(
        lambda value: normalize_required_string(value)
    )
    cleaned["utterance_id"] = cleaned["utterance_id"].apply(
        lambda value: normalize_required_string(value)
    )
    cleaned["speaker_id"] = cleaned["speaker_id"].apply(
        lambda value: normalize_required_string(value)
    )
    cleaned["speaker_name"] = cleaned["speaker_name"].apply(
        lambda value: normalize_required_string(value)
    )

    cleaned["reply_to"] = cleaned["reply_to"].apply(normalize_optional_string)
    cleaned["listener_id"] = cleaned["listener_id"].apply(normalize_optional_string)
    cleaned["listener_name"] = cleaned["listener_name"].apply(normalize_optional_string)
    cleaned["text"] = cleaned["text"].apply(normalize_text)
    # Keep timestamp as an optional raw placeholder. It is not used for ordering.
    cleaned["timestamp"] = cleaned["timestamp"].apply(normalize_optional_string)

    return cleaned


def sort_dialogue_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve original row order within each conversation.

    The raw/interim dialogue file keeps `timestamp` only as a compatibility
    placeholder, so preprocessing should not use it for ordering. Instead, we
    preserve the existing row order from dialogue_metadata.csv and then create
    turn_index / utterance_order after sorting by conversation.
    """
    sorted_df = df.copy()
    sorted_df["_row_order"] = range(len(sorted_df))

    sorted_df = sorted_df.sort_values(
        ["conversation_id", "_row_order"],
        kind="mergesort",
    ).drop(columns=["_row_order"])

    return sorted_df.reset_index(drop=True)


def clean_dialogue_dataframe(
    df: pd.DataFrame,
    drop_missing_listener: bool = False,
) -> pd.DataFrame:
    """Clean extracted dialogue metadata for downstream turn-window construction."""
    cleaned = standardize_dialogue_columns(df)

    cleaned = cleaned[
        (cleaned["conversation_id"] != "UNKNOWN")
        & (cleaned["utterance_id"] != "UNKNOWN")
        & (cleaned["speaker_id"] != "UNKNOWN")
    ].copy()
    cleaned = cleaned[cleaned["text"] != ""].copy()

    if drop_missing_listener:
        cleaned = cleaned[cleaned["listener_id"].notna()].copy()

    # Preserve dialogue_metadata.csv row order within each conversation.
    # Do not rely on timestamp because it may be intentionally empty.
    cleaned = sort_dialogue_dataframe(cleaned)
    cleaned = add_turn_index(cleaned)
    cleaned = add_utterance_order(cleaned)

    for column in OUTPUT_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = None

    cleaned = cleaned[OUTPUT_COLUMNS].copy()
    return cleaned.reset_index(drop=True)


def summarize_cleaned_dialogue(df: pd.DataFrame) -> dict[str, Any]:
    """Return simple summary statistics for the cleaned dialogue table."""
    return {
        "rows": len(df),
        "conversations": int(df["conversation_id"].nunique()) if "conversation_id" in df else 0,
        "speakers": int(df["speaker_id"].nunique()) if "speaker_id" in df else 0,
        "listener_present": int(df["listener_id"].notna().sum()) if "listener_id" in df else 0,
        "listener_missing": int(df["listener_id"].isna().sum()) if "listener_id" in df else 0,
    }


def print_cleaned_dialogue_summary(df: pd.DataFrame) -> None:
    """Print a compact preprocessing summary."""
    summary = summarize_cleaned_dialogue(df)
    print(f"Rows after cleaning: {summary['rows']}")
    print(f"Unique conversations: {summary['conversations']}")
    print(f"Unique speakers: {summary['speakers']}")
    print(f"Rows with listener_id present: {summary['listener_present']}")
    print(f"Rows with listener_id missing: {summary['listener_missing']}")


def preprocess_and_save_dialogue(
    movie_name: str = DEFAULT_MOVIE_NAME,
    drop_missing_listener: bool = False,
) -> pd.DataFrame:
    """Load extracted dialogue metadata, clean it, and save the result."""
    ensure_project_dirs(movie_name)

    input_path = get_movie_dialogue_metadata_path(movie_name)
    output_path = get_movie_clean_dialogue_path(movie_name)

    print(f"Loading extracted dialogue metadata from: {input_path}")
    df = load_csv(input_path)
    print(f"Loaded {len(df)} rows.")

    cleaned_df = clean_dialogue_dataframe(
        df,
        drop_missing_listener=drop_missing_listener,
    )

    save_csv(cleaned_df, output_path, index=False)

    print("\nPreprocessing complete.")
    print_file_summary(output_path, label="Saved cleaned dialogue")
    print_cleaned_dialogue_summary(cleaned_df)
    return cleaned_df


def main() -> None:
    args = parse_args()
    preprocess_and_save_dialogue(
        movie_name=args.movie,
        drop_missing_listener=args.drop_missing_listener,
    )


if __name__ == "__main__":
    main()