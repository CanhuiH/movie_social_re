

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_MOVIE_NAME, ensure_project_dirs
from src.utils.io import load_csv, load_jsonl, print_file_summary, save_csv
from src.utils.paths import (
    get_movie_social_summaries_path,
    get_movie_turn_windows_path,
)


TRANSLATION_MODE_CONTEXT_ONLY = "context_only"
TRANSLATION_MODE_WITH_GRAPH = "with_graph"


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

REQUIRED_TRANSLATION_COLUMNS = {
    "movie_name",
    "conversation_id",
    "utterance_id",
    "translation_text",
    "translation_model",
    "translation_input_mode",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a comparison CSV that merges dialogue context, graph-derived "
            "relationship information, and translations from the context-only and "
            "with-graph conditions."
        )
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=DEFAULT_MOVIE_NAME,
        help=f'Movie name used to locate movie-specific files. Default: "{DEFAULT_MOVIE_NAME}".',
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help=(
            "Optional custom output CSV path. If omitted, the file is saved to "
            "data/processed/<Movie>/translation_comparison.csv."
        ),
    )
    return parser.parse_args()


def get_translation_output_path(movie_name: str, mode: str) -> Path:
    from src.utils.paths import get_movie_processed_dir

    processed_dir = get_movie_processed_dir(movie_name)
    return processed_dir / f"translations_{mode}.jsonl"


def get_default_output_path(movie_name: str) -> Path:
    from src.utils.paths import get_movie_processed_dir

    processed_dir = get_movie_processed_dir(movie_name)
    return processed_dir / "translation_comparison.csv"


def validate_columns(df: pd.DataFrame, required_columns: set[str], label: str) -> None:
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{label} is missing required columns: {sorted(missing_columns)}"
        )


def load_translation_dataframe(path: Path) -> pd.DataFrame:
    records = load_jsonl(path)
    df = pd.DataFrame(records)
    validate_columns(df, REQUIRED_TRANSLATION_COLUMNS, f"Translation file {path}")
    return df


def prepare_turn_windows_dataframe(turn_windows_df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(turn_windows_df, REQUIRED_TURN_WINDOW_COLUMNS, "Turn-window data")

    selected_columns = [
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
    ]
    return turn_windows_df[selected_columns].copy()


def prepare_summaries_dataframe(summaries_df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(summaries_df, REQUIRED_SUMMARY_COLUMNS, "Social summaries")

    prepared = summaries_df[
        [
            "source_id",
            "target_id",
            "relationship_type",
            "social_summary",
        ]
    ].copy()
    prepared = prepared.rename(
        columns={
            "source_id": "speaker_id",
            "target_id": "listener_id",
            "social_summary": "graph_summary",
        }
    )
    prepared = prepared.drop_duplicates(subset=["speaker_id", "listener_id"])
    return prepared


def prepare_translation_dataframe(translations_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    validate_columns(translations_df, REQUIRED_TRANSLATION_COLUMNS, f"Translations ({mode})")

    prepared = translations_df[
        [
            "movie_name",
            "conversation_id",
            "utterance_id",
            "translation_text",
            "translation_model",
        ]
    ].copy()

    if mode == TRANSLATION_MODE_CONTEXT_ONLY:
        prepared = prepared.rename(
            columns={
                "translation_text": "translation_context_only",
                "translation_model": "translation_model_context_only",
            }
        )
    elif mode == TRANSLATION_MODE_WITH_GRAPH:
        prepared = prepared.rename(
            columns={
                "translation_text": "translation_with_graph",
                "translation_model": "translation_model_with_graph",
            }
        )
    else:
        raise ValueError(f"Unsupported translation mode: {mode}")

    return prepared


def build_translation_comparison_dataframe(
    turn_windows_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
    context_only_df: pd.DataFrame,
    with_graph_df: pd.DataFrame,
) -> pd.DataFrame:
    base_df = prepare_turn_windows_dataframe(turn_windows_df)
    summary_df = prepare_summaries_dataframe(summaries_df)
    context_df = prepare_translation_dataframe(
        context_only_df,
        TRANSLATION_MODE_CONTEXT_ONLY,
    )
    graph_df = prepare_translation_dataframe(
        with_graph_df,
        TRANSLATION_MODE_WITH_GRAPH,
    )

    merged_df = base_df.merge(
        summary_df,
        on=["speaker_id", "listener_id"],
        how="left",
    )
    merged_df = merged_df.merge(
        context_df,
        on=["movie_name", "conversation_id", "utterance_id"],
        how="left",
    )
    merged_df = merged_df.merge(
        graph_df,
        on=["movie_name", "conversation_id", "utterance_id"],
        how="left",
    )

    if "relationship_type" not in merged_df.columns:
        merged_df["relationship_type"] = "unclear"
    else:
        merged_df["relationship_type"] = merged_df["relationship_type"].fillna("unclear")

    if "graph_summary" not in merged_df.columns:
        merged_df["graph_summary"] = ""
    else:
        merged_df["graph_summary"] = merged_df["graph_summary"].fillna("")

    column_order = [
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
        "relationship_type",
        "graph_summary",
        "translation_context_only",
        "translation_model_context_only",
        "translation_with_graph",
        "translation_model_with_graph",
    ]

    for column in column_order:
        if column not in merged_df.columns:
            merged_df[column] = None

    merged_df = merged_df[column_order]
    merged_df = merged_df.sort_values(
        ["movie_name", "conversation_id", "utterance_id"]
    ).reset_index(drop=True)
    return merged_df


def main() -> None:
    args = parse_args()
    ensure_project_dirs(args.movie)

    turn_windows_path = get_movie_turn_windows_path(args.movie)
    summaries_path = get_movie_social_summaries_path(args.movie)
    context_only_path = get_translation_output_path(
        args.movie,
        TRANSLATION_MODE_CONTEXT_ONLY,
    )
    with_graph_path = get_translation_output_path(
        args.movie,
        TRANSLATION_MODE_WITH_GRAPH,
    )
    output_path = Path(args.output_file) if args.output_file else get_default_output_path(args.movie)

    print(f"Loading turn windows from: {turn_windows_path}")
    turn_windows_df = load_csv(turn_windows_path)
    print(f"Loaded {len(turn_windows_df)} turn-window rows.")

    print(f"Loading social summaries from: {summaries_path}")
    summaries_df = load_csv(summaries_path)
    print(f"Loaded {len(summaries_df)} social summaries.")

    print(f"Loading context-only translations from: {context_only_path}")
    context_only_df = load_translation_dataframe(context_only_path)
    print(f"Loaded {len(context_only_df)} context-only translations.")

    print(f"Loading with-graph translations from: {with_graph_path}")
    with_graph_df = load_translation_dataframe(with_graph_path)
    print(f"Loaded {len(with_graph_df)} with-graph translations.")

    comparison_df = build_translation_comparison_dataframe(
        turn_windows_df=turn_windows_df,
        summaries_df=summaries_df,
        context_only_df=context_only_df,
        with_graph_df=with_graph_df,
    )

    save_csv(comparison_df, output_path, index=False)

    print("\nTranslation comparison export complete.")
    print_file_summary(output_path, label="Saved translation comparison CSV")
    print(f"Number of comparison rows: {len(comparison_df)}")


if __name__ == "__main__":
    main()