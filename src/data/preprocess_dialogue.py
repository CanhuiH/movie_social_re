

from __future__ import annotations

import argparse
import re
from typing import Optional

import pandas as pd

from src.config import DEFAULT_MOVIE_NAME, ensure_project_dirs
from src.utils.io import load_csv, print_file_summary, save_csv
from src.utils.paths import (
    get_movie_clean_dialogue_path,
    get_movie_dialogue_metadata_path,
    get_movie_listener_only_dialogue_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess extracted movie dialogue metadata into a clean, "
            "movie-agnostic dialogue table."
        )
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=DEFAULT_MOVIE_NAME,
        help=(
            f'Movie name used to locate movie-specific interim files. '
            f'Default: "{DEFAULT_MOVIE_NAME}".'
        ),
    )
    parser.add_argument(
        "--listener-only",
        action="store_true",
        help=(
            "If set, load the listener-only extracted file instead of the full "
            "dialogue metadata file."
        ),
    )
    parser.add_argument(
        "--drop-missing-listener",
        action="store_true",
        help="If set, drop rows where listener_id is missing after cleaning.",
    )
    return parser.parse_args()


def clean_text(text: object) -> str:
    """Normalize dialogue text for downstream RE and translation steps."""
    if pd.isna(text):
        return ""

    cleaned = str(text)
    cleaned = cleaned.replace("\n", " ").replace("\r", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_name(name: object) -> Optional[str]:
    """Normalize character names while preserving missing values as None."""
    if pd.isna(name):
        return None

    normalized = str(name).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized if normalized else None


def add_utterance_order(df: pd.DataFrame) -> pd.DataFrame:
    """Add a stable within-conversation utterance order column."""
    ordered = df.copy()
    ordered["utterance_order"] = ordered.groupby("conversation_id").cumcount()
    return ordered


def clean_dialogue_dataframe(
    df: pd.DataFrame,
    drop_missing_listener: bool = False,
) -> pd.DataFrame:
    """Clean extracted dialogue metadata for downstream turn-window building."""
    cleaned = df.copy()

    expected_columns = {
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
    }
    missing_columns = expected_columns - set(cleaned.columns)
    if missing_columns:
        raise ValueError(
            "Input dialogue metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    cleaned["speaker_name"] = cleaned["speaker_name"].apply(normalize_name)
    cleaned["listener_name"] = cleaned["listener_name"].apply(normalize_name)
    cleaned["text"] = cleaned["text"].apply(clean_text)

    cleaned = cleaned.dropna(subset=["conversation_id", "utterance_id", "speaker_id"])
    cleaned = cleaned[cleaned["text"] != ""].copy()

    if drop_missing_listener:
        cleaned = cleaned.dropna(subset=["listener_id"]).copy()

    sort_cols = [
        column
        for column in ["conversation_id", "timestamp", "utterance_id"]
        if column in cleaned.columns
    ]
    cleaned = cleaned.sort_values(sort_cols).reset_index(drop=True)
    cleaned = add_utterance_order(cleaned)

    return cleaned


def main() -> None:
    args = parse_args()
    ensure_project_dirs(args.movie)

    input_path = (
        get_movie_listener_only_dialogue_path(args.movie)
        if args.listener_only
        else get_movie_dialogue_metadata_path(args.movie)
    )
    output_path = get_movie_clean_dialogue_path(args.movie)

    print(f"Loading extracted dialogue metadata from: {input_path}")
    df = load_csv(input_path)
    print(f"Loaded {len(df)} rows.")

    cleaned_df = clean_dialogue_dataframe(
        df,
        drop_missing_listener=args.drop_missing_listener,
    )

    save_csv(cleaned_df, output_path, index=False)

    print("\nPreprocessing complete.")
    print_file_summary(output_path, label="Saved cleaned dialogue")
    print(f"Rows after cleaning: {len(cleaned_df)}")
    print(f"Rows with listener_id present: {cleaned_df['listener_id'].notna().sum()}")
    print(f"Unique conversations: {cleaned_df['conversation_id'].nunique()}")


if __name__ == "__main__":
    main()