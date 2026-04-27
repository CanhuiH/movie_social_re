from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from src.config import DEFAULT_MOVIE_NAME, ensure_project_dirs
from src.re.schema import normalize_relationship_label
from src.utils.io import load_csv, print_file_summary, save_jsonl
from src.utils.paths import (
    get_movie_social_summaries_path,
    get_movie_translation_inputs_path,
    get_movie_turn_windows_path,
)


TRANSLATION_INPUT_MODES = {"context_only", "with_graph"}
DEFAULT_TRANSLATION_INPUT_MODE = "with_graph"


REQUIRED_TURN_WINDOW_COLUMNS = {
    "movie_name",
    "conversation_id",
    "utterance_id",
    "speaker_id",
    "speaker_name",
    "listener_id",
    "listener_name",
    "current_turn",
    "current_text",
    "context_text",
}

REQUIRED_SUMMARY_COLUMNS = {
    "source_id",
    "source_name",
    "target_id",
    "target_name",
    "relationship_type",
    "social_summary",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build prompt-ready translation inputs by combining turn windows "
            "with graph-derived social summaries."
        )
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=DEFAULT_MOVIE_NAME,
        help=f'Movie name used to locate movie-specific files. Default: "{DEFAULT_MOVIE_NAME}".',
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=sorted(TRANSLATION_INPUT_MODES),
        default=DEFAULT_TRANSLATION_INPUT_MODE,
        help=(
            "Which translation input variant to build. "
            "Use 'context_only' for a baseline without graph-derived social summaries, "
            "or 'with_graph' to include graph-derived relationship information. "
            f"Default: '{DEFAULT_TRANSLATION_INPUT_MODE}'."
        ),
    )
    return parser.parse_args()


def get_mode_specific_output_path(movie_name: str, mode: str) -> str:
    """Return a mode-specific JSONL output path for translation inputs."""
    base_path = get_movie_translation_inputs_path(movie_name)
    return str(base_path.with_name(f"{base_path.stem}_{mode}{base_path.suffix}"))


def validate_turn_windows_dataframe(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_TURN_WINDOW_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Turn-window data is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def validate_summaries_dataframe(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_SUMMARY_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Social summary data is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def safe_optional_str(value: Any) -> str | None:
    if pd.isna(value) or value is None:
        return None
    text = str(value).strip()
    return text if text else None


def safe_text(value: Any) -> str:
    if pd.isna(value) or value is None:
        return ""
    return str(value).strip()


def build_summary_lookup(summaries_df: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}

    for _, row in summaries_df.iterrows():
        source_id = safe_optional_str(row["source_id"])
        target_id = safe_optional_str(row["target_id"])
        if source_id is None or target_id is None:
            continue

        lookup[(source_id, target_id)] = {
            "relationship_type": normalize_relationship_label(row.get("relationship_type")),
            "social_summary": safe_text(row.get("social_summary")),
        }

    return lookup


def build_translation_input_record(
    turn_row: pd.Series,
    summary_lookup: dict[tuple[str, str], dict[str, Any]],
    mode: str = DEFAULT_TRANSLATION_INPUT_MODE,
) -> dict[str, Any]:
    speaker_id = safe_optional_str(turn_row.get("speaker_id"))
    listener_id = safe_optional_str(turn_row.get("listener_id"))

    summary_info = None
    if mode == "with_graph" and speaker_id is not None and listener_id is not None:
        summary_info = summary_lookup.get((speaker_id, listener_id))

    if mode == "with_graph" and summary_info is not None:
        relationship_type = summary_info["relationship_type"]
        social_summary = summary_info["social_summary"]
    else:
        relationship_type = "unclear"
        social_summary = ""

    return {
        "movie_name": safe_text(turn_row.get("movie_name")),
        "conversation_id": safe_text(turn_row.get("conversation_id")),
        "utterance_id": safe_text(turn_row.get("utterance_id")),
        "speaker_id": speaker_id,
        "speaker_name": safe_optional_str(turn_row.get("speaker_name")),
        "listener_id": listener_id,
        "listener_name": safe_optional_str(turn_row.get("listener_name")),
        "current_turn": safe_text(turn_row.get("current_turn")),
        "current_text": safe_text(turn_row.get("current_text")),
        "context_text": safe_text(turn_row.get("context_text")),
        "relationship_type": relationship_type,
        "social_summary": social_summary,
        "translation_input_mode": mode,
    }


def build_translation_inputs_dataframe(
    turn_windows_df: pd.DataFrame,
    summaries_df: pd.DataFrame | None = None,
    mode: str = DEFAULT_TRANSLATION_INPUT_MODE,
) -> pd.DataFrame:
    validate_turn_windows_dataframe(turn_windows_df)

    if mode not in TRANSLATION_INPUT_MODES:
        raise ValueError(
            f"Unsupported translation input mode: {mode}. "
            f"Expected one of {sorted(TRANSLATION_INPUT_MODES)}."
        )

    if mode == "with_graph":
        if summaries_df is None:
            raise ValueError("summaries_df is required when mode='with_graph'.")
        validate_summaries_dataframe(summaries_df)
        summary_lookup = build_summary_lookup(summaries_df)
    else:
        summary_lookup = {}

    rows = [
        build_translation_input_record(turn_row, summary_lookup, mode=mode)
        for _, turn_row in turn_windows_df.iterrows()
    ]

    translation_inputs_df = pd.DataFrame(rows)
    sort_cols = [
        column
        for column in ["movie_name", "conversation_id", "utterance_id"]
        if column in translation_inputs_df.columns
    ]
    if sort_cols:
        translation_inputs_df = translation_inputs_df.sort_values(sort_cols).reset_index(drop=True)

    return translation_inputs_df


def main() -> None:
    args = parse_args()
    ensure_project_dirs(args.movie)

    turn_windows_path = get_movie_turn_windows_path(args.movie)
    output_path = get_mode_specific_output_path(args.movie, args.mode)

    print(f"Loading turn windows from: {turn_windows_path}")
    turn_windows_df = load_csv(turn_windows_path)
    print(f"Loaded {len(turn_windows_df)} turn-window rows.")

    summaries_df = None
    if args.mode == "with_graph":
        summaries_path = get_movie_social_summaries_path(args.movie)
        print(f"Loading social summaries from: {summaries_path}")
        summaries_df = load_csv(summaries_path)
        print(f"Loaded {len(summaries_df)} social summaries.")

    translation_inputs_df = build_translation_inputs_dataframe(
        turn_windows_df=turn_windows_df,
        summaries_df=summaries_df,
        mode=args.mode,
    )

    records = translation_inputs_df.to_dict(orient="records")
    save_jsonl(records, output_path)

    print("\nTranslation input construction complete.")
    print_file_summary(output_path, label="Saved translation inputs")
    print(f"Translation input mode: {args.mode}")
    print(f"Number of translation inputs: {len(translation_inputs_df)}")


if __name__ == "__main__":
    main()
