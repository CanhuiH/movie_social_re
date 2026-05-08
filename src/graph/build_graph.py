

"""Build dynamic speaker-listener social graph edges from risk-gold overlap rows.

The graph is represented as a flat edge table. Each row corresponds to the graph
state for one directed speaker-listener edge after observing one dialogue turn.
This keeps the graph dynamic: later steps can use the edge state available at
that point in the movie rather than a single final relationship status.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.data.prepare_risk_gold_overlap import TARGET_RISK_COLUMNS
from src.utils.io import ensure_parent_dir, load_csv, save_csv
from src.utils.paths import project_path

DEFAULT_INPUT_PATH = project_path(Path("data") / "interim" / "risk_gold_overlap.csv")
DEFAULT_OUTPUT_PATH = project_path(Path("data") / "graph" / "social_graph_edges.csv")
DEFAULT_SUMMARY_PATH = project_path(Path("outputs") / "tables" / "graph_summary_stats.csv")

DEFAULT_RECENT_WINDOW = 6
EDGE_KEY_COLUMNS = ["movie_name", "speaker_id", "listener_id"]


@dataclass(frozen=True)
class SocialGraphBuildResult:
    """Paths and summary statistics for social graph construction."""

    output_path: Path
    summary_path: Path
    input_rows: int
    edge_state_rows: int
    unique_edges: int
    movies: int
    recent_window: int


def normalize_text_value(value: Any, default: str = "") -> str:
    """Normalize a scalar value to a clean string."""
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def normalize_label(value: Any, default: str = "unclear") -> str:
    """Normalize label values used in graph state."""
    text = normalize_text_value(value, default=default).lower()
    return text if text and text != "nan" else default


def choose_text_column(df: pd.DataFrame) -> str:
    """Choose the best available current-utterance text column."""
    for column in ["current_text", "text", "current_turn"]:
        if column in df.columns:
            return column
    raise ValueError("Input overlap file must contain one of: current_text, text, current_turn")


def choose_order_columns(df: pd.DataFrame) -> list[str]:
    """Choose stable sort columns for chronological graph construction."""
    candidates = [
        "movie_name",
        "conversation_id",
        "turn_index",
        "utterance_order",
        "utterance_id",
    ]
    return [column for column in candidates if column in df.columns]


def mode_label(values: Iterable[str], default: str = "unclear") -> str:
    """Return the most common non-empty label, with deterministic tie-breaking."""
    cleaned = [normalize_label(value, default=default) for value in values]
    cleaned = [value for value in cleaned if value and value != "unclear"]
    if not cleaned:
        return default

    counts = Counter(cleaned)
    max_count = max(counts.values())
    tied = sorted(label for label, count in counts.items() if count == max_count)
    return tied[0]


def join_recent_values(values: Iterable[Any], max_items: int | None = None) -> str:
    """Join recent values into a compact semicolon-separated string."""
    cleaned = [normalize_text_value(value) for value in values]
    cleaned = [value for value in cleaned if value]
    if max_items is not None:
        cleaned = cleaned[-max_items:]
    return " || ".join(cleaned)


def collect_active_risks(row: pd.Series) -> str:
    """Return active target risk names for one row."""
    active: list[str] = []
    for column in TARGET_RISK_COLUMNS:
        if column in row.index and bool(row[column]):
            active.append(column)
    return ";".join(active)


def make_edge_key(row: pd.Series) -> tuple[str, str, str]:
    """Create a directed movie-specific speaker-listener edge key."""
    movie_name = normalize_text_value(row.get("movie_name"), default="unknown_movie").lower()
    speaker_id = normalize_text_value(row.get("speaker_id"), default=normalize_text_value(row.get("speaker_name"), default="unknown_speaker"))
    listener_id = normalize_text_value(row.get("listener_id"), default=normalize_text_value(row.get("listener_name"), default="unknown_listener"))
    return movie_name, speaker_id, listener_id


def build_edge_state_row(
    *,
    row: pd.Series,
    edge_key: tuple[str, str, str],
    edge_history: list[dict[str, Any]],
    recent_history: deque[dict[str, Any]],
) -> dict[str, Any]:
    """Build one output row representing the current dynamic edge state."""
    movie_name, speaker_id, listener_id = edge_key

    aggregate_power = mode_label(item["power_dynamic"] for item in edge_history)
    aggregate_respect = mode_label(item["respect_level"] for item in edge_history)
    aggregate_relationship = mode_label(item["relationship_type"] for item in edge_history)

    recent_power = mode_label(item["power_dynamic"] for item in recent_history)
    recent_respect = mode_label(item["respect_level"] for item in recent_history)
    recent_relationship = mode_label(item["relationship_type"] for item in recent_history)

    active_risk_types = collect_active_risks(row)

    return {
        "movie_name": movie_name,
        "conversation_id": normalize_text_value(row.get("conversation_id")),
        "utterance_id": normalize_text_value(row.get("utterance_id")),
        "turn_index": row.get("turn_index", ""),
        "utterance_order": row.get("utterance_order", ""),
        "speaker_id": speaker_id,
        "speaker_name": normalize_text_value(row.get("speaker_name"), default=speaker_id),
        "listener_id": listener_id,
        "listener_name": normalize_text_value(row.get("listener_name"), default=listener_id),
        "current_power_dynamic": normalize_label(row.get("final_power_dynamic")),
        "current_respect_level": normalize_label(row.get("final_respect_level")),
        "current_relationship_type": normalize_label(row.get("relationship_type")),
        "current_global_category": normalize_label(row.get("global_category")),
        "active_risk_types": active_risk_types,
        "target_risk_count": int(row.get("target_risk_count", 0)) if str(row.get("target_risk_count", "")).strip() else 0,
        "aggregate_power_dynamic": aggregate_power,
        "aggregate_respect_level": aggregate_respect,
        "aggregate_relationship_type": aggregate_relationship,
        "recent_power_dynamic": recent_power,
        "recent_respect_level": recent_respect,
        "recent_relationship_type": recent_relationship,
        "edge_observation_count": len(edge_history),
        "recent_observation_count": len(recent_history),
        "supporting_utterance_ids": join_recent_values(item["utterance_id"] for item in edge_history),
        "recent_utterance_ids": join_recent_values(item["utterance_id"] for item in recent_history),
        "recent_evidence_lines": join_recent_values(item["text"] for item in recent_history),
    }


def build_social_graph_edges(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    recent_window: int = DEFAULT_RECENT_WINDOW,
) -> SocialGraphBuildResult:
    """Build dynamic social graph edge states from risk-gold overlap rows."""
    if recent_window <= 0:
        raise ValueError("recent_window must be positive.")
    if not input_path.exists():
        raise FileNotFoundError(f"Risk-gold overlap file not found: {input_path}")

    df = load_csv(input_path)
    if df.empty:
        raise ValueError(f"Risk-gold overlap file is empty: {input_path}")

    required_columns = [
        "movie_name",
        "conversation_id",
        "utterance_id",
        "final_power_dynamic",
        "final_respect_level",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Input overlap file is missing required columns: {missing}")

    text_column = choose_text_column(df)
    sort_columns = choose_order_columns(df)
    if sort_columns:
        df = df.sort_values(sort_columns).reset_index(drop=True)

    edge_histories: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    recent_histories: dict[tuple[str, str, str], deque[dict[str, Any]]] = defaultdict(
        lambda: deque(maxlen=recent_window)
    )

    output_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        edge_key = make_edge_key(row)
        observation = {
            "movie_name": normalize_text_value(row.get("movie_name"), default="unknown_movie").lower(),
            "conversation_id": normalize_text_value(row.get("conversation_id")),
            "utterance_id": normalize_text_value(row.get("utterance_id")),
            "speaker_id": edge_key[1],
            "listener_id": edge_key[2],
            "power_dynamic": normalize_label(row.get("final_power_dynamic")),
            "respect_level": normalize_label(row.get("final_respect_level")),
            "relationship_type": normalize_label(row.get("relationship_type")),
            "global_category": normalize_label(row.get("global_category")),
            "text": normalize_text_value(row.get(text_column)),
            "risk_types": collect_active_risks(row),
        }

        edge_histories[edge_key].append(observation)
        recent_histories[edge_key].append(observation)

        output_rows.append(
            build_edge_state_row(
                row=row,
                edge_key=edge_key,
                edge_history=edge_histories[edge_key],
                recent_history=recent_histories[edge_key],
            )
        )

    edges_df = pd.DataFrame(output_rows)
    ensure_parent_dir(output_path)
    save_csv(edges_df, output_path)

    summary_df = make_graph_summary_stats(edges_df)
    ensure_parent_dir(summary_path)
    save_csv(summary_df, summary_path)

    return SocialGraphBuildResult(
        output_path=output_path,
        summary_path=summary_path,
        input_rows=len(df),
        edge_state_rows=len(edges_df),
        unique_edges=edges_df[EDGE_KEY_COLUMNS].drop_duplicates().shape[0],
        movies=edges_df["movie_name"].nunique(),
        recent_window=recent_window,
    )


def make_graph_summary_stats(edges_df: pd.DataFrame) -> pd.DataFrame:
    """Create per-movie graph summary statistics."""
    rows: list[dict[str, Any]] = []
    for movie_name, movie_df in edges_df.groupby("movie_name", dropna=False):
        rows.append(
            {
                "movie_name": movie_name,
                "edge_state_rows": len(movie_df),
                "unique_edges": movie_df[EDGE_KEY_COLUMNS].drop_duplicates().shape[0],
                "unique_speakers": movie_df[["speaker_id"]].drop_duplicates().shape[0],
                "unique_listeners": movie_df[["listener_id"]].drop_duplicates().shape[0],
                "avg_edge_observation_count": movie_df["edge_observation_count"].mean(),
                "max_edge_observation_count": movie_df["edge_observation_count"].max(),
            }
        )

    rows.append(
        {
            "movie_name": "__total__",
            "edge_state_rows": len(edges_df),
            "unique_edges": edges_df[EDGE_KEY_COLUMNS].drop_duplicates().shape[0],
            "unique_speakers": edges_df[["movie_name", "speaker_id"]].drop_duplicates().shape[0],
            "unique_listeners": edges_df[["movie_name", "listener_id"]].drop_duplicates().shape[0],
            "avg_edge_observation_count": edges_df["edge_observation_count"].mean(),
            "max_edge_observation_count": edges_df["edge_observation_count"].max(),
        }
    )
    return pd.DataFrame(rows)


def print_graph_build_summary(result: SocialGraphBuildResult) -> None:
    """Print a compact summary for graph construction."""
    print("Social graph construction complete")
    print(f"  input rows       : {result.input_rows}")
    print(f"  edge state rows  : {result.edge_state_rows}")
    print(f"  unique edges     : {result.unique_edges}")
    print(f"  movies           : {result.movies}")
    print(f"  recent window    : {result.recent_window}")
    print(f"  output           : {result.output_path}")
    print(f"  summary          : {result.summary_path}")


__all__ = [
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_RECENT_WINDOW",
    "DEFAULT_SUMMARY_PATH",
    "EDGE_KEY_COLUMNS",
    "SocialGraphBuildResult",
    "build_social_graph_edges",
    "make_graph_summary_stats",
    "mode_label",
    "print_graph_build_summary",
]