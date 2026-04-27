

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from src.config import DEFAULT_CONTEXT_WINDOW, DEFAULT_MOVIE_NAME, ensure_project_dirs
from src.utils.io import load_csv, print_file_summary, save_csv
from src.utils.paths import (
    get_movie_clean_dialogue_path,
    get_movie_turn_windows_path,
)


REQUIRED_COLUMNS = {
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
    "utterance_order",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build turn-level dialogue windows for relationship extraction and "
            "downstream translation."
        )
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=DEFAULT_MOVIE_NAME,
        help=f'Movie name used to locate movie-specific files. Default: "{DEFAULT_MOVIE_NAME}".',
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        help=(
            "Number of previous turns to include as local context for each target turn. "
            f"Default: {DEFAULT_CONTEXT_WINDOW}."
        ),
    )
    parser.add_argument(
        "--drop-missing-listener",
        action="store_true",
        help="If set, only keep target turns with a known listener.",
    )
    return parser.parse_args()


def validate_input_dataframe(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Clean dialogue data is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def format_turn_text(speaker_name: Any, text: Any) -> str:
    speaker = speaker_name if pd.notna(speaker_name) and str(speaker_name).strip() else "UNKNOWN"
    utterance = str(text).strip() if pd.notna(text) else ""
    return f"{speaker}: {utterance}" if utterance else f"{speaker}:"


def build_prev_turn_columns(prev_turns: list[str], context_window: int) -> dict[str, str]:
    columns: dict[str, str] = {}
    padded_prev_turns = [""] * (context_window - len(prev_turns)) + prev_turns

    for index, turn_text in enumerate(padded_prev_turns, start=1):
        columns[f"prev_turn_{index}"] = turn_text

    return columns


def build_context_text(prev_turns: list[str]) -> str:
    return "\n".join(prev_turns)


def build_turn_windows_dataframe(
    df: pd.DataFrame,
    context_window: int,
    drop_missing_listener: bool = False,
) -> pd.DataFrame:
    validate_input_dataframe(df)

    if context_window < 0:
        raise ValueError("context_window must be non-negative.")

    rows: list[dict[str, Any]] = []

    grouped = df.groupby("conversation_id", sort=False)
    for conversation_id, conversation_df in grouped:
        ordered_conversation = conversation_df.sort_values("utterance_order").reset_index(drop=True)

        for idx, row in ordered_conversation.iterrows():
            if drop_missing_listener and pd.isna(row["listener_id"]):
                continue

            start_idx = max(0, idx - context_window)
            prev_df = ordered_conversation.iloc[start_idx:idx]
            prev_turns = [
                format_turn_text(prev_row["speaker_name"], prev_row["text"])
                for _, prev_row in prev_df.iterrows()
            ]

            current_turn_formatted = format_turn_text(row["speaker_name"], row["text"])
            prev_turn_columns = build_prev_turn_columns(prev_turns, context_window)

            rows.append(
                {
                    "movie_name": row["movie_name"],
                    "movie_idx": row["movie_idx"],
                    "conversation_id": conversation_id,
                    "utterance_id": row["utterance_id"],
                    "utterance_order": row["utterance_order"],
                    "timestamp": row["timestamp"],
                    "speaker_id": row["speaker_id"],
                    "speaker_name": row["speaker_name"],
                    "listener_id": row["listener_id"],
                    "listener_name": row["listener_name"],
                    "reply_to": row["reply_to"],
                    "current_turn": current_turn_formatted,
                    "current_text": row["text"],
                    "context_text": build_context_text(prev_turns),
                    "num_prev_turns": len(prev_turns),
                    **prev_turn_columns,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    ensure_project_dirs(args.movie)

    input_path = get_movie_clean_dialogue_path(args.movie)
    output_path = get_movie_turn_windows_path(args.movie)

    print(f"Loading cleaned dialogue data from: {input_path}")
    df = load_csv(input_path)
    print(f"Loaded {len(df)} rows.")

    turn_windows_df = build_turn_windows_dataframe(
        df,
        context_window=args.context_window,
        drop_missing_listener=args.drop_missing_listener,
    )

    save_csv(turn_windows_df, output_path, index=False)

    print("\nTurn-window construction complete.")
    print_file_summary(output_path, label="Saved turn windows")
    print(f"Number of turn-window examples: {len(turn_windows_df)}")
    print(f"Unique conversations: {turn_windows_df['conversation_id'].nunique()}")
    print(f"Context window size used: {args.context_window}")


if __name__ == "__main__":
    main()