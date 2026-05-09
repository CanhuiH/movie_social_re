from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from src.config import DEFAULT_CONTEXT_WINDOW, DEFAULT_MOVIE_NAME, ensure_project_dirs
from src.utils.io import load_csv, print_file_summary, require_columns, save_csv
from src.utils.paths import (
    get_movie_clean_dialogue_path,
    get_movie_turn_windows_path,
)


REQUIRED_COLUMNS = {
    "movie_name",
    "conversation_id",
    "utterance_id",
    "turn_index",
    "utterance_order",
    "speaker_id",
    "speaker_name",
    "listener_id",
    "listener_name",
    "text",
}

BASE_OUTPUT_COLUMNS = [
    "movie_name",
    "conversation_id",
    "utterance_id",
    "turn_index",
    "utterance_order",
    "speaker_id",
    "speaker_name",
    "listener_id",
    "listener_name",
    "current_turn",
    "current_text",
    "context_text",
    "num_prev_turns",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build turn-level local dialogue windows for relationship extraction."
        )
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=DEFAULT_MOVIE_NAME,
        help=f'Movie title used to locate movie-specific files. Default: "{DEFAULT_MOVIE_NAME}".',
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


def validate_context_window(context_window: int) -> None:
    if context_window < 0:
        raise ValueError("context_window must be non-negative.")


def validate_input_dataframe(df: pd.DataFrame) -> None:
    require_columns(df, REQUIRED_COLUMNS, label="Clean dialogue data")


def normalize_display_value(value: object, fallback: str = "UNKNOWN") -> str:
    """Normalize a value for prompt display."""
    if pd.isna(value):
        return fallback
    text = " ".join(str(value).strip().split())
    return text if text else fallback


def normalize_text(value: object) -> str:
    """Normalize dialogue text for prompt display."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def format_turn_text(speaker_name: object, text: object) -> str:
    """Format one dialogue turn as `Speaker: text`."""
    speaker = normalize_display_value(speaker_name)
    utterance = normalize_text(text)
    return f"{speaker}: {utterance}" if utterance else f"{speaker}:"


def build_prev_turn_columns(prev_turns: list[str], context_window: int) -> dict[str, str]:
    """Build fixed-width previous-turn columns for easier inspection."""
    columns: dict[str, str] = {}

    if context_window == 0:
        return columns

    padded_prev_turns = [""] * (context_window - len(prev_turns)) + prev_turns
    for index, turn_text in enumerate(padded_prev_turns, start=1):
        columns[f"prev_turn_{index}"] = turn_text

    return columns


def build_context_text(prev_turns: list[str]) -> str:
    """Join previous turns into a single local context string."""
    return "\n".join(prev_turns)


def sort_clean_dialogue(df: pd.DataFrame) -> pd.DataFrame:
    """Return dialogue sorted by conversation and within-conversation order."""
    sorted_df = df.copy()
    sorted_df["_utterance_order_missing"] = sorted_df["utterance_order"].isna()
    sorted_df["_utterance_order_sort"] = (
        sorted_df["utterance_order"].fillna(10**12).astype(int)
    )

    sorted_df = sorted_df.sort_values(
        [
            "conversation_id",
            "_utterance_order_missing",
            "_utterance_order_sort",
            "utterance_id",
        ],
        kind="mergesort",
    ).drop(columns=["_utterance_order_missing", "_utterance_order_sort"])

    return sorted_df.reset_index(drop=True)


def build_turn_window_row(
    row: pd.Series,
    conversation_id: object,
    prev_turns: list[str],
    context_window: int,
) -> dict[str, Any]:
    """Build one turn-window output row."""
    current_turn_formatted = format_turn_text(row["speaker_name"], row["text"])
    prev_turn_columns = build_prev_turn_columns(prev_turns, context_window)

    return {
        "movie_name": row["movie_name"],
        "conversation_id": conversation_id,
        "utterance_id": row["utterance_id"],
        "turn_index": row["turn_index"],
        "utterance_order": row["utterance_order"],
        "speaker_id": row["speaker_id"],
        "speaker_name": row["speaker_name"],
        "listener_id": row["listener_id"],
        "listener_name": row["listener_name"],
        "current_turn": current_turn_formatted,
        "current_text": row["text"],
        "context_text": build_context_text(prev_turns),
        "num_prev_turns": len(prev_turns),
        **prev_turn_columns,
    }


def build_turn_windows_dataframe(
    df: pd.DataFrame,
    context_window: int,
    drop_missing_listener: bool = False,
) -> pd.DataFrame:
    """Build one relationship-extraction example per target dialogue turn.

    Each output row represents a target turn. The local context consists of up
    to `context_window` previous turns from the same conversation.
    """
    validate_input_dataframe(df)
    validate_context_window(context_window)

    rows: list[dict[str, Any]] = []
    sorted_df = sort_clean_dialogue(df)

    grouped = sorted_df.groupby("conversation_id", sort=False)
    for conversation_id, conversation_df in grouped:
        ordered_conversation = conversation_df.sort_values(
            ["utterance_order", "utterance_id"],
            kind="mergesort",
        ).reset_index(drop=True)

        formatted_turns = [
            format_turn_text(row["speaker_name"], row["text"])
            for _, row in ordered_conversation.iterrows()
        ]

        for idx, row in ordered_conversation.iterrows():
            if drop_missing_listener and pd.isna(row["listener_id"]):
                continue

            start_idx = max(0, idx - context_window)
            prev_turns = formatted_turns[start_idx:idx]

            rows.append(
                build_turn_window_row(
                    row=row,
                    conversation_id=conversation_id,
                    prev_turns=prev_turns,
                    context_window=context_window,
                )
            )

    output_df = pd.DataFrame(rows)

    prev_turn_columns = [
        f"prev_turn_{index}"
        for index in range(1, context_window + 1)
    ]
    column_order = BASE_OUTPUT_COLUMNS + prev_turn_columns
    for column in column_order:
        if column not in output_df.columns:
            output_df[column] = ""

    return output_df[column_order].reset_index(drop=True)


def summarize_turn_windows(df: pd.DataFrame) -> dict[str, int]:
    """Return simple summary statistics for turn-window outputs."""
    return {
        "rows": len(df),
        "conversations": int(df["conversation_id"].nunique()) if "conversation_id" in df else 0,
        "speakers": int(df["speaker_id"].nunique()) if "speaker_id" in df else 0,
        "listener_present": int(df["listener_id"].notna().sum()) if "listener_id" in df else 0,
        "listener_missing": int(df["listener_id"].isna().sum()) if "listener_id" in df else 0,
    }


def print_turn_window_summary(df: pd.DataFrame, context_window: int) -> None:
    """Print a compact summary of turn-window construction."""
    summary = summarize_turn_windows(df)
    print(f"Number of turn-window examples: {summary['rows']}")
    print(f"Unique conversations: {summary['conversations']}")
    print(f"Unique speakers: {summary['speakers']}")
    print(f"Rows with listener_id present: {summary['listener_present']}")
    print(f"Rows with listener_id missing: {summary['listener_missing']}")
    print(f"Context window size used: {context_window}")


def build_and_save_turn_windows(
    movie_name: str = DEFAULT_MOVIE_NAME,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    drop_missing_listener: bool = False,
) -> pd.DataFrame:
    """Load cleaned dialogue, build turn windows, and save the result."""
    ensure_project_dirs(movie_name)

    input_path = get_movie_clean_dialogue_path(movie_name)
    output_path = get_movie_turn_windows_path(movie_name)

    print(f"Loading cleaned dialogue data from: {input_path}")
    df = load_csv(input_path)
    print(f"Loaded {len(df)} rows.")

    turn_windows_df = build_turn_windows_dataframe(
        df,
        context_window=context_window,
        drop_missing_listener=drop_missing_listener,
    )

    save_csv(turn_windows_df, output_path, index=False)

    print("\nTurn-window construction complete.")
    print_file_summary(output_path, label="Saved turn windows")
    print_turn_window_summary(turn_windows_df, context_window)
    return turn_windows_df


def main() -> None:
    args = parse_args()
    build_and_save_turn_windows(
        movie_name=args.movie,
        context_window=args.context_window,
        drop_missing_listener=args.drop_missing_listener,
    )


if __name__ == "__main__":
    main()