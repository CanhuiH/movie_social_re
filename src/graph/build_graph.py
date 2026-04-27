

from __future__ import annotations

import argparse
from typing import Any

import networkx as nx
import pandas as pd

from src.config import DEFAULT_MOVIE_NAME, ensure_project_dirs
from src.re.schema import GraphEdge, normalize_relationship_label
from src.utils.io import load_csv, print_file_summary, save_csv
from src.utils.paths import (
    get_movie_graph_edges_path,
    get_movie_graph_nodes_path,
    get_movie_re_clean_output_path,
)


REQUIRED_RE_COLUMNS = {
    "movie_name",
    "conversation_id",
    "utterance_id",
    "speaker_id",
    "speaker_name",
    "listener_id",
    "listener_name",
    "relationship_type",
    "confidence",
    "evidence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a lightweight directed relationship graph from clean RE outputs."
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=DEFAULT_MOVIE_NAME,
        help=f'Movie name used to locate movie-specific files. Default: "{DEFAULT_MOVIE_NAME}".',
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Optional minimum confidence required to keep an edge update. Default: 0.0.",
    )
    parser.add_argument(
        "--drop-unclear",
        action="store_true",
        help="If set, skip rows whose relationship_type is 'unclear'.",
    )
    return parser.parse_args()


def validate_re_dataframe(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_RE_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Clean RE data is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def normalize_re_dataframe(
    df: pd.DataFrame,
    confidence_threshold: float = 0.0,
    drop_unclear: bool = False,
) -> pd.DataFrame:
    normalized = df.copy()

    normalized["relationship_type"] = normalized["relationship_type"].apply(
        normalize_relationship_label
    )
    normalized["confidence"] = pd.to_numeric(normalized["confidence"], errors="coerce").fillna(0.0)

    if confidence_threshold > 0.0:
        normalized = normalized[normalized["confidence"] >= confidence_threshold].copy()

    if drop_unclear:
        normalized = normalized[normalized["relationship_type"] != "unclear"].copy()

    sort_cols = [
        column
        for column in ["movie_name", "conversation_id", "utterance_id"]
        if column in normalized.columns
    ]
    if sort_cols:
        normalized = normalized.sort_values(sort_cols).reset_index(drop=True)

    return normalized


def initialize_graph() -> nx.DiGraph:
    return nx.DiGraph()


def ensure_node(graph: nx.DiGraph, node_id: str, node_name: str | None) -> None:
    if not graph.has_node(node_id):
        graph.add_node(node_id, character_id=node_id, character_name=node_name)
    else:
        existing_name = graph.nodes[node_id].get("character_name")
        if not existing_name and node_name:
            graph.nodes[node_id]["character_name"] = node_name


def build_edge_from_row(row: pd.Series) -> GraphEdge:
    return GraphEdge(
        source_id=str(row["speaker_id"]),
        source_name=None if pd.isna(row["speaker_name"]) else str(row["speaker_name"]),
        target_id=str(row["listener_id"]),
        target_name=None if pd.isna(row["listener_name"]) else str(row["listener_name"]),
        relationship_type=normalize_relationship_label(row["relationship_type"]),
        confidence=float(row["confidence"]),
        conversation_id=None if pd.isna(row["conversation_id"]) else str(row["conversation_id"]),
        utterance_id_last_updated=None if pd.isna(row["utterance_id"]) else str(row["utterance_id"]),
        evidence="" if pd.isna(row["evidence"]) else str(row["evidence"]),
    )


def update_graph_from_row(graph: nx.DiGraph, row: pd.Series) -> None:
    speaker_id = row["speaker_id"]
    listener_id = row["listener_id"]

    if pd.isna(speaker_id) or pd.isna(listener_id):
        return

    speaker_id = str(speaker_id)
    listener_id = str(listener_id)
    speaker_name = None if pd.isna(row["speaker_name"]) else str(row["speaker_name"])
    listener_name = None if pd.isna(row["listener_name"]) else str(row["listener_name"])

    ensure_node(graph, speaker_id, speaker_name)
    ensure_node(graph, listener_id, listener_name)

    edge = build_edge_from_row(row)
    graph.add_edge(
        speaker_id,
        listener_id,
        **edge.to_dict(),
    )


def build_graph_from_dataframe(df: pd.DataFrame) -> nx.DiGraph:
    graph = initialize_graph()
    for _, row in df.iterrows():
        update_graph_from_row(graph, row)
    return graph


def graph_nodes_to_dataframe(graph: nx.DiGraph) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for node_id, attrs in graph.nodes(data=True):
        rows.append(
            {
                "character_id": node_id,
                "character_name": attrs.get("character_name"),
            }
        )
    return pd.DataFrame(rows).sort_values(["character_name", "character_id"], na_position="last").reset_index(drop=True)


def graph_edges_to_dataframe(graph: nx.DiGraph) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_id, target_id, attrs in graph.edges(data=True):
        rows.append(
            {
                "source_id": source_id,
                "source_name": attrs.get("source_name"),
                "target_id": target_id,
                "target_name": attrs.get("target_name"),
                "relationship_type": attrs.get("relationship_type"),
                "confidence": attrs.get("confidence"),
                "conversation_id": attrs.get("conversation_id"),
                "utterance_id_last_updated": attrs.get("utterance_id_last_updated"),
                "evidence": attrs.get("evidence"),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "source_id",
                "source_name",
                "target_id",
                "target_name",
                "relationship_type",
                "confidence",
                "conversation_id",
                "utterance_id_last_updated",
                "evidence",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["source_name", "target_name", "source_id", "target_id"],
        na_position="last",
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    ensure_project_dirs(args.movie)

    input_path = get_movie_re_clean_output_path(args.movie)
    nodes_output_path = get_movie_graph_nodes_path(args.movie)
    edges_output_path = get_movie_graph_edges_path(args.movie)

    print(f"Loading clean RE outputs from: {input_path}")
    df = load_csv(input_path)
    print(f"Loaded {len(df)} RE rows.")

    validate_re_dataframe(df)
    normalized_df = normalize_re_dataframe(
        df,
        confidence_threshold=args.confidence_threshold,
        drop_unclear=args.drop_unclear,
    )
    print(f"Rows retained for graph construction: {len(normalized_df)}")

    graph = build_graph_from_dataframe(normalized_df)
    nodes_df = graph_nodes_to_dataframe(graph)
    edges_df = graph_edges_to_dataframe(graph)

    save_csv(nodes_df, nodes_output_path, index=False)
    save_csv(edges_df, edges_output_path, index=False)

    print("\nGraph construction complete.")
    print_file_summary(nodes_output_path, label="Saved graph nodes")
    print_file_summary(edges_output_path, label="Saved graph edges")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")


if __name__ == "__main__":
    main()