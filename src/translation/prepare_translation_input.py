"""Prepare translation input for context-only and graph-guided translation.

This module merges the risk/gold/current-row social annotation table with the
aggregate and recent speaker-listener graph states. The output file is used by
both translation conditions:

- context-only translation uses dialogue context and current text only.
- graph-guided translation additionally uses structured social signals and graph states.

This module does not generate an LLM or rule-based graph summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import ensure_parent_dir, load_csv, save_csv
from src.utils.paths import project_path

DEFAULT_GRAPH_PATH = project_path(Path("data") / "graph" / "social_graph_edges.csv")
DEFAULT_OVERLAP_PATH = project_path(Path("data") / "interim" / "risk_gold_overlap.csv")
DEFAULT_TRANSLATION_INPUT_PATH = project_path(Path("data") / "interim" / "translation_input.csv")
DEFAULT_SUMMARY_STATS_PATH = project_path(Path("outputs") / "tables" / "translation_input_summary.csv")

JOIN_KEYS = ["movie_name", "conversation_id", "utterance_id"]
EDGE_KEY_COLUMNS = ["movie_name", "speaker_id", "listener_id"]

GRAPH_STATE_COLUMNS = [
    *JOIN_KEYS,
    "aggregate_power_dynamic",
    "aggregate_respect_level",
    "aggregate_relationship_type",
    "recent_power_dynamic",
    "recent_respect_level",
    "recent_relationship_type",
    "edge_observation_count",
    "recent_observation_count",
]

OPTIONAL_GRAPH_COLUMNS = [
    "aggregate_global_category",
    "recent_global_category",
    "current_power_dynamic",
    "current_respect_level",
    "current_relationship_type",
    "current_global_category",
    "supporting_utterance_ids",
    "recent_utterance_ids",
    "recent_evidence_lines",
    "active_risk_types",
    "target_risk_count",
]

CURRENT_ROW_SOCIAL_COLUMNS = [
    "final_power_dynamic",
    "final_respect_level",
    "relationship_type",
    "global_category",
    "evidence",
    "confidence",
    "status",
    "parse_success",
]


@dataclass(frozen=True)
class TranslationInputResult:
    """Paths and summary statistics for translation input preparation."""

    translation_input_path: Path
    summary_stats_path: Path
    rows: int
    movies: int
    rows_with_graph_state: int


def normalize_text(value: Any, default: str = "") -> str:
    """Normalize a scalar value to a clean string."""
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def validate_required_columns(df: pd.DataFrame, required_columns: list[str], file_label: str) -> None:
    """Raise a clear error if a dataframe is missing required columns."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{file_label} is missing required columns: {missing}")


def select_graph_columns(edges_df: pd.DataFrame) -> pd.DataFrame:
    """Select graph-state columns needed by graph-guided translation."""
    validate_required_columns(edges_df, GRAPH_STATE_COLUMNS, "Social graph edge file")

    selected_columns = [column for column in GRAPH_STATE_COLUMNS + OPTIONAL_GRAPH_COLUMNS if column in edges_df.columns]
    graph_df = edges_df[selected_columns].copy()

    # One graph row should correspond to one translated dialogue row. If duplicated
    # rows exist, keep the latest occurrence after the graph builder's sort order.
    graph_df = graph_df.drop_duplicates(subset=JOIN_KEYS, keep="last")
    return graph_df


def merge_translation_input(overlap_df: pd.DataFrame, graph_df: pd.DataFrame) -> pd.DataFrame:
    """Merge risk/gold/current-row social annotations with graph-state columns."""
    validate_required_columns(overlap_df, JOIN_KEYS, "Risk-gold overlap file")
    validate_required_columns(graph_df, JOIN_KEYS, "Selected graph-state dataframe")

    translation_df = overlap_df.merge(
        graph_df,
        on=JOIN_KEYS,
        how="left",
        suffixes=("", "_graph"),
    )

    graph_indicator_columns = [
        "aggregate_power_dynamic",
        "aggregate_respect_level",
        "aggregate_relationship_type",
        "recent_power_dynamic",
        "recent_respect_level",
        "recent_relationship_type",
    ]
    for column in graph_indicator_columns:
        if column in translation_df.columns:
            translation_df[column] = translation_df[column].fillna("unclear")

    for column in ["edge_observation_count", "recent_observation_count"]:
        if column in translation_df.columns:
            translation_df[column] = translation_df[column].fillna(0).astype(int)

    return translation_df


def make_translation_input_summary(translation_df: pd.DataFrame) -> pd.DataFrame:
    """Create compact summary statistics for the prepared translation input."""
    rows: list[dict[str, Any]] = []
    grouped = translation_df.groupby("movie_name", dropna=False) if "movie_name" in translation_df.columns else []

    for movie_name, movie_df in grouped:
        rows.append(
            {
                "movie_name": movie_name,
                "rows": len(movie_df),
                "rows_with_graph_state": int(
                    movie_df.get("edge_observation_count", pd.Series(dtype=float)).fillna(0).gt(0).sum()
                ),
                "unique_relationship_types": movie_df["relationship_type"].nunique(dropna=True)
                if "relationship_type" in movie_df.columns
                else 0,
                "unique_recent_relationship_types": movie_df["recent_relationship_type"].nunique(dropna=True)
                if "recent_relationship_type" in movie_df.columns
                else 0,
                "unique_aggregate_relationship_types": movie_df["aggregate_relationship_type"].nunique(dropna=True)
                if "aggregate_relationship_type" in movie_df.columns
                else 0,
            }
        )

    rows.append(
        {
            "movie_name": "__total__",
            "rows": len(translation_df),
            "rows_with_graph_state": int(
                translation_df.get("edge_observation_count", pd.Series(dtype=float)).fillna(0).gt(0).sum()
            ),
            "unique_relationship_types": translation_df["relationship_type"].nunique(dropna=True)
            if "relationship_type" in translation_df.columns
            else 0,
            "unique_recent_relationship_types": translation_df["recent_relationship_type"].nunique(dropna=True)
            if "recent_relationship_type" in translation_df.columns
            else 0,
            "unique_aggregate_relationship_types": translation_df["aggregate_relationship_type"].nunique(dropna=True)
            if "aggregate_relationship_type" in translation_df.columns
            else 0,
        }
    )
    return pd.DataFrame(rows)


def prepare_translation_input(
    *,
    graph_path: Path = DEFAULT_GRAPH_PATH,
    overlap_path: Path = DEFAULT_OVERLAP_PATH,
    translation_input_path: Path = DEFAULT_TRANSLATION_INPUT_PATH,
    summary_stats_path: Path = DEFAULT_SUMMARY_STATS_PATH,
) -> TranslationInputResult:
    """Prepare translation_input.csv by merging overlap rows with graph states."""
    if not graph_path.exists():
        raise FileNotFoundError(f"Social graph edge file not found: {graph_path}")
    if not overlap_path.exists():
        raise FileNotFoundError(f"Risk-gold overlap file not found: {overlap_path}")

    edges_df = load_csv(graph_path)
    overlap_df = load_csv(overlap_path)

    graph_df = select_graph_columns(edges_df)
    translation_df = merge_translation_input(overlap_df, graph_df)

    ensure_parent_dir(translation_input_path)
    save_csv(translation_df, translation_input_path)

    stats_df = make_translation_input_summary(translation_df)
    ensure_parent_dir(summary_stats_path)
    save_csv(stats_df, summary_stats_path)

    rows_with_graph_state = int(
        translation_df.get("edge_observation_count", pd.Series(dtype=float)).fillna(0).gt(0).sum()
    )

    return TranslationInputResult(
        translation_input_path=translation_input_path,
        summary_stats_path=summary_stats_path,
        rows=len(translation_df),
        movies=translation_df["movie_name"].nunique() if "movie_name" in translation_df.columns else 0,
        rows_with_graph_state=rows_with_graph_state,
    )


def print_translation_input_result(result: TranslationInputResult) -> None:
    """Print a compact summary for translation-input preparation."""
    print("Translation input preparation complete")
    print(f"  rows                 : {result.rows}")
    print(f"  movies               : {result.movies}")
    print(f"  rows with graph state: {result.rows_with_graph_state}")
    print(f"  translation input    : {result.translation_input_path}")
    print(f"  summary stats        : {result.summary_stats_path}")


# Backward-compatible aliases for older script names/imports.
generate_graph_summaries = prepare_translation_input
print_graph_summary_result = print_translation_input_result
GraphSummaryResult = TranslationInputResult
DEFAULT_OUTPUT_PATH = DEFAULT_TRANSLATION_INPUT_PATH


__all__ = [
    "CURRENT_ROW_SOCIAL_COLUMNS",
    "DEFAULT_GRAPH_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_OVERLAP_PATH",
    "DEFAULT_SUMMARY_STATS_PATH",
    "DEFAULT_TRANSLATION_INPUT_PATH",
    "GRAPH_STATE_COLUMNS",
    "JOIN_KEYS",
    "OPTIONAL_GRAPH_COLUMNS",
    "TranslationInputResult",
    "GraphSummaryResult",
    "generate_graph_summaries",
    "make_translation_input_summary",
    "merge_translation_input",
    "prepare_translation_input",
    "print_graph_summary_result",
    "print_translation_input_result",
    "select_graph_columns",
]