"""Merge context-only and graph-guided translation outputs.

This module combines the two translation conditions into one comparison table:

- context-only baseline translations
- graph-guided translations
- LLM-produced social guidance summaries for the graph-guided condition

The merged output is useful for manual inspection, qualitative comparison, and
later evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import ensure_parent_dir, load_csv, save_csv
from src.utils.paths import project_path

DEFAULT_CONTEXT_ONLY_PATH = project_path(Path("data") / "translation_eval" / "translation_context_only.csv")
DEFAULT_GRAPH_GUIDED_PATH = project_path(Path("data") / "translation_eval" / "translation_graph_guided.csv")
DEFAULT_OUTPUT_PATH = project_path(Path("data") / "translation_eval" / "translation_comparison.csv")
DEFAULT_SUMMARY_PATH = project_path(Path("outputs") / "tables" / "translation_comparison_summary.csv")

JOIN_KEYS = ["movie_name", "conversation_id", "utterance_id"]

CORE_COLUMNS = [
    "movie_name",
    "conversation_id",
    "utterance_id",
    "speaker_name",
    "listener_name",
    "current_text",
    "context_6_turns",
]

SOCIAL_SIGNAL_COLUMNS = [
    "final_power_dynamic",
    "final_respect_level",
    "relationship_type",
    "evidence",
    "recent_power_dynamic",
    "recent_respect_level",
    "recent_relationship_type",
    "aggregate_power_dynamic",
    "aggregate_respect_level",
    "aggregate_relationship_type",
]

CONTEXT_ONLY_COLUMNS = [
    "translation_context_only",
    "translation_model_context_only",
]

GRAPH_GUIDED_COLUMNS = [
    "llm_social_guidance_summary",
    "translation_graph_guided",
    "translation_model_graph_guided",
]

OPTIONAL_DEBUG_COLUMNS = [
    "translation_prompt_context_only",
    "translation_prompt_graph_guided",
    "translation_raw_response_graph_guided",
]


@dataclass(frozen=True)
class MergeTranslationOutputsResult:
    """Paths and summary statistics for merged translation outputs."""

    output_path: Path
    summary_path: Path
    rows: int
    context_only_rows: int
    graph_guided_rows: int
    paired_rows: int


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


def deduplicate_by_join_keys(df: pd.DataFrame, file_label: str) -> pd.DataFrame:
    """Keep the last row for duplicated join keys."""
    validate_required_columns(df, JOIN_KEYS, file_label)
    return df.drop_duplicates(subset=JOIN_KEYS, keep="last").copy()


def select_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return the subset of columns that exists in a dataframe."""
    return [column for column in columns if column in df.columns]


def get_graph_guided_subset(graph_df: pd.DataFrame) -> pd.DataFrame:
    """Select graph-guided-only columns for merging."""
    selected_columns = [
        *JOIN_KEYS,
        *select_existing_columns(graph_df, GRAPH_GUIDED_COLUMNS),
    ]
    return graph_df[selected_columns].copy()


def add_comparison_columns(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Add helper columns for paired comparison."""
    output_df = merged_df.copy()

    if "translation_context_only" in output_df.columns:
        output_df["has_context_only_translation"] = output_df["translation_context_only"].fillna("").astype(str).str.strip() != ""
    else:
        output_df["has_context_only_translation"] = False

    if "translation_graph_guided" in output_df.columns:
        output_df["has_graph_guided_translation"] = output_df["translation_graph_guided"].fillna("").astype(str).str.strip() != ""
    else:
        output_df["has_graph_guided_translation"] = False

    output_df["has_paired_translations"] = (
        output_df["has_context_only_translation"] & output_df["has_graph_guided_translation"]
    )

    if "translation_context_only" in output_df.columns and "translation_graph_guided" in output_df.columns:
        output_df["translations_identical"] = (
            output_df["translation_context_only"].fillna("").astype(str).str.strip()
            == output_df["translation_graph_guided"].fillna("").astype(str).str.strip()
        )
    else:
        output_df["translations_identical"] = False

    return output_df


def filter_valid_social_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with usable power, respect, and relationship labels."""
    required_columns = ["final_power_dynamic", "final_respect_level", "relationship_type"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Merged translation output is missing required social columns: {missing}")

    output_df = df.copy()
    power = output_df["final_power_dynamic"].fillna("").astype(str).str.strip().str.lower()
    respect = output_df["final_respect_level"].fillna("").astype(str).str.strip().str.lower()
    relationship = output_df["relationship_type"].fillna("").astype(str).str.strip().str.lower()

    valid_mask = (
        power.ne("")
        & power.ne("nan")
        & power.ne("unclear")
        & respect.ne("")
        & respect.ne("nan")
        & respect.ne("unclear")
        & relationship.ne("")
        & relationship.ne("nan")
        & relationship.ne("unclear")
    )
    return output_df[valid_mask].copy()


def order_output_columns(df: pd.DataFrame, include_debug_columns: bool = False) -> pd.DataFrame:
    """Keep only the necessary columns for the final comparison output."""
    preferred_columns = [
        *CORE_COLUMNS,
        *SOCIAL_SIGNAL_COLUMNS,
        "llm_social_guidance_summary",
        "translation_context_only",
        "translation_graph_guided",
        "translation_model_context_only",
        "translation_model_graph_guided",
    ]
    if include_debug_columns:
        preferred_columns.extend(OPTIONAL_DEBUG_COLUMNS)

    ordered_columns: list[str] = []
    for column in preferred_columns:
        if column in df.columns and column not in ordered_columns:
            ordered_columns.append(column)

    return df[ordered_columns]


def make_merge_summary(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Create compact summary statistics for the merged comparison output."""
    rows: list[dict[str, Any]] = []
    grouped = merged_df.groupby("movie_name", dropna=False) if "movie_name" in merged_df.columns else []

    for movie_name, movie_df in grouped:
        rows.append(
            {
                "movie_name": movie_name,
                "rows": len(movie_df),
                "context_only_translated": int(movie_df["has_context_only_translation"].sum()),
                "graph_guided_translated": int(movie_df["has_graph_guided_translation"].sum()),
                "paired_translations": int(movie_df["has_paired_translations"].sum()),
                "identical_translations": int(movie_df["translations_identical"].sum()),
            }
        )

    rows.append(
        {
            "movie_name": "__total__",
            "rows": len(merged_df),
            "context_only_translated": int(merged_df["has_context_only_translation"].sum()),
            "graph_guided_translated": int(merged_df["has_graph_guided_translation"].sum()),
            "paired_translations": int(merged_df["has_paired_translations"].sum()),
            "identical_translations": int(merged_df["translations_identical"].sum()),
        }
    )
    return pd.DataFrame(rows)


def merge_translation_outputs(
    *,
    context_only_path: Path = DEFAULT_CONTEXT_ONLY_PATH,
    graph_guided_path: Path = DEFAULT_GRAPH_GUIDED_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    include_debug_columns: bool = False,
) -> MergeTranslationOutputsResult:
    """Merge context-only and graph-guided translation outputs."""
    if not context_only_path.exists():
        raise FileNotFoundError(f"Context-only translation file not found: {context_only_path}")
    if not graph_guided_path.exists():
        raise FileNotFoundError(f"Graph-guided translation file not found: {graph_guided_path}")

    context_df = deduplicate_by_join_keys(load_csv(context_only_path), "Context-only translation file")
    graph_df = deduplicate_by_join_keys(load_csv(graph_guided_path), "Graph-guided translation file")

    validate_required_columns(context_df, [*JOIN_KEYS, "translation_context_only"], "Context-only translation file")
    validate_required_columns(graph_df, [*JOIN_KEYS, "translation_graph_guided"], "Graph-guided translation file")

    graph_subset_df = get_graph_guided_subset(graph_df)
    merged_df = context_df.merge(graph_subset_df, on=JOIN_KEYS, how="outer", suffixes=("", "_graph"))
    merged_df = add_comparison_columns(merged_df)
    merged_df = filter_valid_social_rows(merged_df)

    summary_df = make_merge_summary(merged_df)

    output_df = order_output_columns(merged_df, include_debug_columns=include_debug_columns)

    ensure_parent_dir(output_path)
    save_csv(output_df, output_path)

    ensure_parent_dir(summary_path)
    save_csv(summary_df, summary_path)


    return MergeTranslationOutputsResult(
        output_path=output_path,
        summary_path=summary_path,
        rows=len(merged_df),
        context_only_rows=int(merged_df["has_context_only_translation"].sum()),
        graph_guided_rows=int(merged_df["has_graph_guided_translation"].sum()),
        paired_rows=int(merged_df["has_paired_translations"].sum()),
    )


def print_merge_translation_outputs_result(result: MergeTranslationOutputsResult) -> None:
    """Print a compact summary for merged translation outputs."""
    print("Translation outputs merged")
    print(f"  rows                   : {result.rows}")
    print(f"  context-only translated: {result.context_only_rows}")
    print(f"  graph-guided translated: {result.graph_guided_rows}")
    print(f"  paired translations    : {result.paired_rows}")
    print(f"  output                 : {result.output_path}")
    print(f"  summary                : {result.summary_path}")


__all__ = [
    "CONTEXT_ONLY_COLUMNS",
    "CORE_COLUMNS",
    "DEFAULT_CONTEXT_ONLY_PATH",
    "DEFAULT_GRAPH_GUIDED_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_SUMMARY_PATH",
    "GRAPH_GUIDED_COLUMNS",
    "JOIN_KEYS",
    "MergeTranslationOutputsResult",
    "OPTIONAL_DEBUG_COLUMNS",
    "SOCIAL_SIGNAL_COLUMNS",
    "add_comparison_columns",
    "deduplicate_by_join_keys",
    "filter_valid_social_rows",
    "get_graph_guided_subset",
    "make_merge_summary",
    "merge_translation_outputs",
    "order_output_columns",
    "print_merge_translation_outputs_result",
]