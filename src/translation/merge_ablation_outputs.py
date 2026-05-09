

"""Merge translation outputs for ablation comparison.

This module merges five translation conditions:

1. context-only baseline
2. power/respect-only ablation
3. relationship-only ablation
4. social-label-guided ablation
5. full graph-guided translation

The output is designed for ablation evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_parent_dir, load_csv, save_csv
from src.utils.paths import project_path


DEFAULT_CONTEXT_ONLY_PATH = project_path(Path("data") / "translation_eval" / "translation_context_only.csv")
DEFAULT_POWER_RESPECT_ONLY_PATH = project_path(
    Path("data") / "translation_eval" / "translation_power_respect_only.csv"
)
DEFAULT_RELATIONSHIP_ONLY_PATH = project_path(
    Path("data") / "translation_eval" / "translation_relationship_only.csv"
)
DEFAULT_SOCIAL_LABELS_ONLY_PATH = project_path(
    Path("data") / "translation_eval" / "translation_social_labels_only.csv"
)
DEFAULT_GRAPH_GUIDED_PATH = project_path(Path("data") / "translation_eval" / "translation_graph_guided.csv")
DEFAULT_OUTPUT_PATH = project_path(Path("data") / "translation_eval" / "translation_ablation_comparison.csv")
DEFAULT_SUMMARY_PATH = project_path(Path("outputs") / "tables" / "translation_ablation_summary.csv")

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

CONDITION_COLUMNS = [
    "translation_context_only",
    "llm_power_respect_only_guidance_summary",
    "translation_power_respect_only",
    "llm_relationship_only_guidance_summary",
    "translation_relationship_only",
    "llm_social_labels_only_guidance_summary",
    "translation_social_labels_only",
    "llm_social_guidance_summary",
    "translation_graph_guided",
    "translation_model_context_only",
    "translation_model_power_respect_only",
    "translation_model_relationship_only",
    "translation_model_social_labels_only",
    "translation_model_graph_guided",
]

COMPARISON_COLUMNS = [
    "has_context_only_translation",
    "has_power_respect_only_translation",
    "has_relationship_only_translation",
    "has_social_labels_only_translation",
    "has_graph_guided_translation",
    "has_all_ablation_translations",
    "context_equals_power_respect_only",
    "context_equals_relationship_only",
    "context_equals_social_labels_only",
    "context_equals_graph_guided",
    "power_respect_only_equals_relationship_only",
    "power_respect_only_equals_social_labels_only",
    "relationship_only_equals_social_labels_only",
    "social_labels_only_equals_graph_guided",
]

OPTIONAL_DEBUG_COLUMNS = [
    "translation_prompt_context_only",
    "translation_prompt_power_respect_only",
    "translation_prompt_relationship_only",
    "translation_prompt_social_labels_only",
    "translation_prompt_graph_guided",
    "translation_raw_response_power_respect_only",
    "translation_raw_response_relationship_only",
    "translation_raw_response_social_labels_only",
    "translation_raw_response_graph_guided",
    "translation_status_context_only",
    "translation_status_power_respect_only",
    "translation_status_relationship_only",
    "translation_status_social_labels_only",
    "translation_status_graph_guided",
    "translation_error_context_only",
    "translation_error_power_respect_only",
    "translation_error_relationship_only",
    "translation_error_social_labels_only",
    "translation_error_graph_guided",
]


@dataclass(frozen=True)
class AblationMergeResult:
    """Summary information for an ablation merge run."""

    context_only_path: Path
    power_respect_only_path: Path
    relationship_only_path: Path
    social_labels_only_path: Path
    graph_guided_path: Path
    output_path: Path
    summary_path: Path
    context_only_rows: int
    power_respect_only_rows: int
    relationship_only_rows: int
    social_labels_only_rows: int
    graph_guided_rows: int
    output_rows: int
def get_power_respect_only_subset(power_respect_df: pd.DataFrame) -> pd.DataFrame:
    """Select power/respect-only columns for merging."""
    columns = [
        *JOIN_KEYS,
        *select_existing_columns(
            power_respect_df,
            [
                "llm_power_respect_only_guidance_summary",
                "translation_power_respect_only",
                "translation_model_power_respect_only",
                "translation_provider_power_respect_only",
                "translation_status_power_respect_only",
                "translation_error_power_respect_only",
                "translation_prompt_power_respect_only",
                "translation_raw_response_power_respect_only",
            ],
        ),
    ]
    return power_respect_df[columns].copy()


def get_relationship_only_subset(relationship_df: pd.DataFrame) -> pd.DataFrame:
    """Select relationship-only columns for merging."""
    columns = [
        *JOIN_KEYS,
        *select_existing_columns(
            relationship_df,
            [
                "llm_relationship_only_guidance_summary",
                "translation_relationship_only",
                "translation_model_relationship_only",
                "translation_provider_relationship_only",
                "translation_status_relationship_only",
                "translation_error_relationship_only",
                "translation_prompt_relationship_only",
                "translation_raw_response_relationship_only",
            ],
        ),
    ]
    return relationship_df[columns].copy()


def select_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return columns that exist in a dataframe."""
    return [column for column in columns if column in df.columns]


def deduplicate_by_join_keys(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Drop duplicate rows by join keys, keeping the latest row."""
    missing = [column for column in JOIN_KEYS if column not in df.columns]
    if missing:
        raise ValueError(f"{source_name} is missing join key columns: {missing}")
    return df.drop_duplicates(subset=JOIN_KEYS, keep="last").copy()


def get_social_labels_only_subset(social_df: pd.DataFrame) -> pd.DataFrame:
    """Select social-label-guided-only columns for merging."""
    columns = [
        *JOIN_KEYS,
        *select_existing_columns(
            social_df,
            [
                "llm_social_labels_only_guidance_summary",
                "translation_social_labels_only",
                "translation_model_social_labels_only",
                "translation_provider_social_labels_only",
                "translation_status_social_labels_only",
                "translation_error_social_labels_only",
                "translation_prompt_social_labels_only",
                "translation_raw_response_social_labels_only",
            ],
        ),
    ]
    return social_df[columns].copy()


def get_graph_guided_subset(graph_df: pd.DataFrame) -> pd.DataFrame:
    """Select graph-guided-only columns for merging."""
    columns = [
        *JOIN_KEYS,
        *select_existing_columns(
            graph_df,
            [
                "llm_social_guidance_summary",
                "translation_graph_guided",
                "translation_model_graph_guided",
                "translation_provider_graph_guided",
                "translation_status_graph_guided",
                "translation_error_graph_guided",
                "translation_prompt_graph_guided",
                "translation_raw_response_graph_guided",
            ],
        ),
    ]
    return graph_df[columns].copy()


def nonempty_text(series: pd.Series) -> pd.Series:
    """Return whether each value is non-empty text."""
    return series.notna() & series.astype(str).str.strip().ne("")


def equal_text(left: pd.Series, right: pd.Series) -> pd.Series:
    """Compare two text columns after filling missing values and trimming whitespace."""
    left_text = left.fillna("").astype(str).str.strip()
    right_text = right.fillna("").astype(str).str.strip()
    return left_text.eq(right_text) & left_text.ne("") & right_text.ne("")


def add_ablation_comparison_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add helper columns for ablation comparison."""
    output_df = df.copy()

    if "translation_context_only" in output_df.columns:
        output_df["has_context_only_translation"] = nonempty_text(output_df["translation_context_only"])
    else:
        output_df["has_context_only_translation"] = False

    if "translation_power_respect_only" in output_df.columns:
        output_df["has_power_respect_only_translation"] = nonempty_text(
            output_df["translation_power_respect_only"]
        )
    else:
        output_df["has_power_respect_only_translation"] = False

    if "translation_relationship_only" in output_df.columns:
        output_df["has_relationship_only_translation"] = nonempty_text(
            output_df["translation_relationship_only"]
        )
    else:
        output_df["has_relationship_only_translation"] = False

    if "translation_social_labels_only" in output_df.columns:
        output_df["has_social_labels_only_translation"] = nonempty_text(
            output_df["translation_social_labels_only"]
        )
    else:
        output_df["has_social_labels_only_translation"] = False

    if "translation_graph_guided" in output_df.columns:
        output_df["has_graph_guided_translation"] = nonempty_text(output_df["translation_graph_guided"])
    else:
        output_df["has_graph_guided_translation"] = False

    output_df["has_all_ablation_translations"] = (
        output_df["has_context_only_translation"]
        & output_df["has_power_respect_only_translation"]
        & output_df["has_relationship_only_translation"]
        & output_df["has_social_labels_only_translation"]
        & output_df["has_graph_guided_translation"]
    )

    if {"translation_context_only", "translation_power_respect_only"}.issubset(output_df.columns):
        output_df["context_equals_power_respect_only"] = equal_text(
            output_df["translation_context_only"],
            output_df["translation_power_respect_only"],
        )
    else:
        output_df["context_equals_power_respect_only"] = False

    if {"translation_context_only", "translation_relationship_only"}.issubset(output_df.columns):
        output_df["context_equals_relationship_only"] = equal_text(
            output_df["translation_context_only"],
            output_df["translation_relationship_only"],
        )
    else:
        output_df["context_equals_relationship_only"] = False

    if {"translation_context_only", "translation_social_labels_only"}.issubset(output_df.columns):
        output_df["context_equals_social_labels_only"] = equal_text(
            output_df["translation_context_only"],
            output_df["translation_social_labels_only"],
        )
    else:
        output_df["context_equals_social_labels_only"] = False

    if {"translation_context_only", "translation_graph_guided"}.issubset(output_df.columns):
        output_df["context_equals_graph_guided"] = equal_text(
            output_df["translation_context_only"],
            output_df["translation_graph_guided"],
        )
    else:
        output_df["context_equals_graph_guided"] = False

    if {"translation_power_respect_only", "translation_relationship_only"}.issubset(output_df.columns):
        output_df["power_respect_only_equals_relationship_only"] = equal_text(
            output_df["translation_power_respect_only"],
            output_df["translation_relationship_only"],
        )
    else:
        output_df["power_respect_only_equals_relationship_only"] = False

    if {"translation_power_respect_only", "translation_social_labels_only"}.issubset(output_df.columns):
        output_df["power_respect_only_equals_social_labels_only"] = equal_text(
            output_df["translation_power_respect_only"],
            output_df["translation_social_labels_only"],
        )
    else:
        output_df["power_respect_only_equals_social_labels_only"] = False

    if {"translation_relationship_only", "translation_social_labels_only"}.issubset(output_df.columns):
        output_df["relationship_only_equals_social_labels_only"] = equal_text(
            output_df["translation_relationship_only"],
            output_df["translation_social_labels_only"],
        )
    else:
        output_df["relationship_only_equals_social_labels_only"] = False

    if {"translation_social_labels_only", "translation_graph_guided"}.issubset(output_df.columns):
        output_df["social_labels_only_equals_graph_guided"] = equal_text(
            output_df["translation_social_labels_only"],
            output_df["translation_graph_guided"],
        )
    else:
        output_df["social_labels_only_equals_graph_guided"] = False

    return output_df


def filter_valid_social_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with usable power, respect, and relationship labels."""
    required_columns = ["final_power_dynamic", "final_respect_level", "relationship_type"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Ablation output is missing required social columns: {missing}")

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
    """Order and optionally limit columns for the ablation comparison output."""
    preferred_columns = [
        *CORE_COLUMNS,
        *SOCIAL_SIGNAL_COLUMNS,
        *CONDITION_COLUMNS,
        *COMPARISON_COLUMNS,
    ]
    if include_debug_columns:
        preferred_columns.extend(OPTIONAL_DEBUG_COLUMNS)

    ordered_columns: list[str] = []
    for column in preferred_columns:
        if column in df.columns and column not in ordered_columns:
            ordered_columns.append(column)

    if include_debug_columns:
        remaining_columns = [column for column in df.columns if column not in ordered_columns]
        return df[ordered_columns + remaining_columns]
    return df[ordered_columns]


def make_ablation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary table for the ablation comparison output."""
    rows: list[dict[str, object]] = []
    grouped = df.groupby("movie_name", dropna=False) if "movie_name" in df.columns else [("__all__", df)]

    for movie_name, movie_df in grouped:
        rows.append(
            {
                "movie_name": movie_name,
                "rows": len(movie_df),
                "context_only_translated": int(movie_df["has_context_only_translation"].sum()),
                "power_respect_only_translated": int(movie_df["has_power_respect_only_translation"].sum()),
                "relationship_only_translated": int(movie_df["has_relationship_only_translation"].sum()),
                "social_labels_only_translated": int(movie_df["has_social_labels_only_translation"].sum()),
                "graph_guided_translated": int(movie_df["has_graph_guided_translation"].sum()),
                "all_ablation_translated": int(movie_df["has_all_ablation_translations"].sum()),
                "context_equals_power_respect_only": int(movie_df["context_equals_power_respect_only"].sum()),
                "context_equals_relationship_only": int(movie_df["context_equals_relationship_only"].sum()),
                "context_equals_social_labels_only": int(movie_df["context_equals_social_labels_only"].sum()),
                "context_equals_graph_guided": int(movie_df["context_equals_graph_guided"].sum()),
                "power_respect_only_equals_relationship_only": int(
                    movie_df["power_respect_only_equals_relationship_only"].sum()
                ),
                "power_respect_only_equals_social_labels_only": int(
                    movie_df["power_respect_only_equals_social_labels_only"].sum()
                ),
                "relationship_only_equals_social_labels_only": int(
                    movie_df["relationship_only_equals_social_labels_only"].sum()
                ),
                "social_labels_only_equals_graph_guided": int(
                    movie_df["social_labels_only_equals_graph_guided"].sum()
                ),
            }
        )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        total_row = {
            "movie_name": "__total__",
            "rows": int(summary_df["rows"].sum()),
            "context_only_translated": int(summary_df["context_only_translated"].sum()),
            "power_respect_only_translated": int(summary_df["power_respect_only_translated"].sum()),
            "relationship_only_translated": int(summary_df["relationship_only_translated"].sum()),
            "social_labels_only_translated": int(summary_df["social_labels_only_translated"].sum()),
            "graph_guided_translated": int(summary_df["graph_guided_translated"].sum()),
            "all_ablation_translated": int(summary_df["all_ablation_translated"].sum()),
            "context_equals_power_respect_only": int(
                summary_df["context_equals_power_respect_only"].sum()
            ),
            "context_equals_relationship_only": int(
                summary_df["context_equals_relationship_only"].sum()
            ),
            "context_equals_social_labels_only": int(
                summary_df["context_equals_social_labels_only"].sum()
            ),
            "context_equals_graph_guided": int(summary_df["context_equals_graph_guided"].sum()),
            "power_respect_only_equals_relationship_only": int(
                summary_df["power_respect_only_equals_relationship_only"].sum()
            ),
            "power_respect_only_equals_social_labels_only": int(
                summary_df["power_respect_only_equals_social_labels_only"].sum()
            ),
            "relationship_only_equals_social_labels_only": int(
                summary_df["relationship_only_equals_social_labels_only"].sum()
            ),
            "social_labels_only_equals_graph_guided": int(
                summary_df["social_labels_only_equals_graph_guided"].sum()
            ),
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)
    return summary_df


def merge_ablation_outputs(
    context_only_path: Path = DEFAULT_CONTEXT_ONLY_PATH,
    power_respect_only_path: Path = DEFAULT_POWER_RESPECT_ONLY_PATH,
    relationship_only_path: Path = DEFAULT_RELATIONSHIP_ONLY_PATH,
    social_labels_only_path: Path = DEFAULT_SOCIAL_LABELS_ONLY_PATH,
    graph_guided_path: Path = DEFAULT_GRAPH_GUIDED_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    include_debug_columns: bool = False,
) -> AblationMergeResult:
    """Merge context-only, power/respect-only, relationship-only, social-label-guided, and graph-guided outputs."""
    context_df = deduplicate_by_join_keys(load_csv(context_only_path), "context-only output")
    power_respect_df = deduplicate_by_join_keys(
        load_csv(power_respect_only_path),
        "power/respect-only output",
    )
    relationship_df = deduplicate_by_join_keys(
        load_csv(relationship_only_path),
        "relationship-only output",
    )
    social_df = deduplicate_by_join_keys(load_csv(social_labels_only_path), "social-label-guided output")
    graph_df = deduplicate_by_join_keys(load_csv(graph_guided_path), "graph-guided output")

    social_subset_df = get_social_labels_only_subset(social_df)
    graph_subset_df = get_graph_guided_subset(graph_df)
    power_respect_subset_df = get_power_respect_only_subset(power_respect_df)
    relationship_subset_df = get_relationship_only_subset(relationship_df)

    merged_df = context_df.merge(
        power_respect_subset_df,
        on=JOIN_KEYS,
        how="outer",
        suffixes=("", "_power_respect"),
    )
    merged_df = merged_df.merge(
        relationship_subset_df,
        on=JOIN_KEYS,
        how="outer",
        suffixes=("", "_relationship"),
    )
    merged_df = merged_df.merge(social_subset_df, on=JOIN_KEYS, how="outer", suffixes=("", "_social"))
    merged_df = merged_df.merge(graph_subset_df, on=JOIN_KEYS, how="outer", suffixes=("", "_graph"))
    merged_df = add_ablation_comparison_columns(merged_df)
    merged_df = filter_valid_social_rows(merged_df)

    summary_df = make_ablation_summary(merged_df)
    output_df = order_output_columns(merged_df, include_debug_columns=include_debug_columns)

    ensure_parent_dir(output_path)
    save_csv(output_df, output_path)

    ensure_parent_dir(summary_path)
    save_csv(summary_df, summary_path)

    return AblationMergeResult(
        context_only_path=context_only_path,
        power_respect_only_path=power_respect_only_path,
        relationship_only_path=relationship_only_path,
        social_labels_only_path=social_labels_only_path,
        graph_guided_path=graph_guided_path,
        output_path=output_path,
        summary_path=summary_path,
        context_only_rows=len(context_df),
        power_respect_only_rows=len(power_respect_df),
        relationship_only_rows=len(relationship_df),
        social_labels_only_rows=len(social_df),
        graph_guided_rows=len(graph_df),
        output_rows=len(output_df),
    )


def print_ablation_merge_summary(result: AblationMergeResult) -> None:
    """Print a concise ablation merge summary."""
    print("Ablation translation merge complete")
    print(f"  context-only path        : {result.context_only_path}")
    print(f"  power/respect-only path : {result.power_respect_only_path}")
    print(f"  relationship-only path  : {result.relationship_only_path}")
    print(f"  social-label-guided path : {result.social_labels_only_path}")
    print(f"  graph-guided path        : {result.graph_guided_path}")
    print(f"  output path              : {result.output_path}")
    print(f"  summary path             : {result.summary_path}")
    print(f"  context-only rows        : {result.context_only_rows}")
    print(f"  power/respect rows      : {result.power_respect_only_rows}")
    print(f"  relationship rows       : {result.relationship_only_rows}")
    print(f"  social-label rows        : {result.social_labels_only_rows}")
    print(f"  graph-guided rows        : {result.graph_guided_rows}")
    print(f"  output rows              : {result.output_rows}")


__all__ = [
    "AblationMergeResult",
    "COMPARISON_COLUMNS",
    "CONDITION_COLUMNS",
    "CORE_COLUMNS",
    "DEFAULT_CONTEXT_ONLY_PATH",
    "DEFAULT_POWER_RESPECT_ONLY_PATH",
    "DEFAULT_RELATIONSHIP_ONLY_PATH",
    "DEFAULT_GRAPH_GUIDED_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_SOCIAL_LABELS_ONLY_PATH",
    "DEFAULT_SUMMARY_PATH",
    "JOIN_KEYS",
    "SOCIAL_SIGNAL_COLUMNS",
    "add_ablation_comparison_columns",
    "deduplicate_by_join_keys",
    "filter_valid_social_rows",
    "get_power_respect_only_subset",
    "get_relationship_only_subset",
    "get_graph_guided_subset",
    "get_social_labels_only_subset",
    "make_ablation_summary",
    "merge_ablation_outputs",
    "order_output_columns",
    "print_ablation_merge_summary",
]