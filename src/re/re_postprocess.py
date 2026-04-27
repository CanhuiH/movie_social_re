from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DEFAULT_RELATIONSHIP_LABEL
from src.re.schema import coerce_confidence, normalize_relationship_label
from src.utils.io import load_jsonl, save_csv


RAW_RESPONSE_COLUMN = "raw_response"


def safe_parse_json_response(raw_response: str | None) -> dict[str, Any]:
    """Parse a raw JSON string from the LLM into a dictionary.

    Returns an empty dictionary if parsing fails or if the value is missing.
    """
    if raw_response is None:
        return {}

    text = str(raw_response).strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def normalize_re_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize one raw relationship extraction record into a clean schema.

    Expected inputs may come from either:
    1. already-parsed JSONL dictionaries, or
    2. dictionaries that store a raw LLM response string in `raw_response`.
    """
    raw_response = record.get(RAW_RESPONSE_COLUMN)
    parsed_response = safe_parse_json_response(raw_response)
    parse_success = bool(parsed_response)

    relationship_type = record.get("relationship_type")
    if relationship_type is None:
        relationship_type = parsed_response.get("relationship_type")

    confidence = record.get("confidence")
    if confidence is None:
        confidence = parsed_response.get("confidence")

    evidence = record.get("evidence")
    if evidence is None:
        evidence = parsed_response.get("evidence", "")

    normalized_record = {
        "movie_name": record.get("movie_name", ""),
        "conversation_id": record.get("conversation_id", ""),
        "utterance_id": record.get("utterance_id", ""),
        "speaker_id": record.get("speaker_id", ""),
        "speaker_name": record.get("speaker_name"),
        "listener_id": record.get("listener_id"),
        "listener_name": record.get("listener_name"),
        "current_turn": record.get("current_turn", ""),
        "current_text": record.get("current_text", ""),
        "context_text": record.get("context_text", ""),
        "relationship_type": normalize_relationship_label(relationship_type),
        "confidence": coerce_confidence(confidence, default=0.0),
        "evidence": "" if evidence is None else str(evidence).strip(),
        "parse_success": parse_success,
        RAW_RESPONSE_COLUMN: raw_response,
    }

    return normalized_record


def normalize_re_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize a batch of raw RE records."""
    return [normalize_re_record(record) for record in records]


def records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert normalized records into a DataFrame with stable column order."""
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
        "confidence",
        "evidence",
        "parse_success",
        RAW_RESPONSE_COLUMN,
    ]

    df = pd.DataFrame(records)
    for column in column_order:
        if column not in df.columns:
            df[column] = None

    return df[column_order]


def build_review_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Create a reviewer-friendly sheet with extra columns for manual correction."""
    review_df = df.copy()
    review_df["reviewed_relationship_type"] = review_df["relationship_type"]
    review_df["reviewer_notes"] = ""
    return review_df


def load_and_normalize_re_jsonl(path: str | Path) -> pd.DataFrame:
    """Load raw RE JSONL output and return a normalized DataFrame."""
    records = load_jsonl(path)
    normalized_records = normalize_re_records(records)
    return records_to_dataframe(normalized_records)


def save_re_outputs(
    df: pd.DataFrame,
    clean_output_path: str | Path,
    review_sheet_path: str | Path | None = None,
) -> None:
    """Save normalized RE outputs and optionally a review sheet."""
    output_df = df.copy()

    sort_cols = [
        column
        for column in ["movie_name", "conversation_id", "utterance_id"]
        if column in output_df.columns
    ]
    if sort_cols:
        output_df = output_df.sort_values(sort_cols).reset_index(drop=True)

    save_csv(output_df, clean_output_path, index=False)

    if review_sheet_path is not None:
        review_df = build_review_sheet(output_df)
        save_csv(review_df, review_sheet_path, index=False)


def normalize_missing_relationships(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure missing relationship values are replaced with the default label."""
    normalized_df = df.copy()
    normalized_df["relationship_type"] = normalized_df["relationship_type"].apply(
        lambda label: normalize_relationship_label(label)
        if label is not None
        else DEFAULT_RELATIONSHIP_LABEL
    )
    return normalized_df