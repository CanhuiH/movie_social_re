

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from src.config import DEFAULT_MOVIE_NAME, ensure_project_dirs
from src.re.schema import relationship_guidance
from src.utils.io import load_csv, print_file_summary, save_csv
from src.utils.paths import (
    get_movie_graph_edges_path,
    get_movie_social_summaries_path,
)


REQUIRED_EDGE_COLUMNS = {
    "source_id",
    "source_name",
    "target_id",
    "target_name",
    "relationship_type",
    "confidence",
    "conversation_id",
    "utterance_id_last_updated",
    "evidence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build short social summaries from graph edges for downstream translation prompts."
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=DEFAULT_MOVIE_NAME,
        help=f'Movie name used to locate movie-specific files. Default: "{DEFAULT_MOVIE_NAME}".',
    )
    return parser.parse_args()


def validate_edge_dataframe(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_EDGE_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Graph edge data is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def safe_str(value: Any, default: str = "UNKNOWN") -> str:
    if pd.isna(value) or value is None:
        return default
    text = str(value).strip()
    return text if text else default


def build_social_summary_text(row: pd.Series) -> str:
    speaker_name = safe_str(row.get("source_name"))
    listener_name = safe_str(row.get("target_name"))
    relationship_type = safe_str(row.get("relationship_type"), default="unclear")
    guidance = relationship_guidance(relationship_type)

    lines = [
        f"Speaker: {speaker_name}",
        f"Listener: {listener_name}",
        f"Relationship: {relationship_type}",
        f"Translation guidance: {guidance}",
    ]
    return "\n".join(lines)


def build_social_summaries_dataframe(edges_df: pd.DataFrame) -> pd.DataFrame:
    validate_edge_dataframe(edges_df)

    summaries_df = edges_df.copy()
    summaries_df["social_summary"] = summaries_df.apply(build_social_summary_text, axis=1)

    column_order = [
        "source_id",
        "source_name",
        "target_id",
        "target_name",
        "relationship_type",
        "confidence",
        "conversation_id",
        "utterance_id_last_updated",
        "evidence",
        "social_summary",
    ]
    return summaries_df[column_order].sort_values(
        ["source_name", "target_name", "source_id", "target_id"],
        na_position="last",
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    ensure_project_dirs(args.movie)

    input_path = get_movie_graph_edges_path(args.movie)
    output_path = get_movie_social_summaries_path(args.movie)

    print(f"Loading graph edges from: {input_path}")
    edges_df = load_csv(input_path)
    print(f"Loaded {len(edges_df)} graph edges.")

    summaries_df = build_social_summaries_dataframe(edges_df)
    save_csv(summaries_df, output_path, index=False)

    print("\nSocial summary generation complete.")
    print_file_summary(output_path, label="Saved social summaries")
    print(f"Number of summaries: {len(summaries_df)}")


if __name__ == "__main__":
    main()