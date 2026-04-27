

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_JSONL_ENCODING = "utf-8"
DEFAULT_CSV_ENCODING = "utf-8"


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for a file path if needed."""
    ensure_dir(path.parent)


def load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path, encoding=DEFAULT_CSV_ENCODING, **kwargs)


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False, **kwargs: Any) -> None:
    """Save a pandas DataFrame to CSV."""
    csv_path = Path(path)
    ensure_parent_dir(csv_path)
    df.to_csv(csv_path, index=index, encoding=DEFAULT_CSV_ENCODING, **kwargs)


def load_json(path: str | Path) -> dict[str, Any] | list[Any]:
    """Load a JSON file."""
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with json_path.open("r", encoding=DEFAULT_JSONL_ENCODING) as file:
        return json.load(file)


def save_json(data: dict[str, Any] | list[Any], path: str | Path, indent: int = 2) -> None:
    """Save a Python object to a JSON file."""
    json_path = Path(path)
    ensure_parent_dir(json_path)
    with json_path.open("w", encoding=DEFAULT_JSONL_ENCODING) as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding=DEFAULT_JSONL_ENCODING) as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {jsonl_path}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {jsonl_path}, "
                    f"but got {type(record).__name__}."
                )
            records.append(record)
    return records


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Save a list of dictionaries to a JSONL file."""
    jsonl_path = Path(path)
    ensure_parent_dir(jsonl_path)

    with jsonl_path.open("w", encoding=DEFAULT_JSONL_ENCODING) as file:
        for record in records:
            if not isinstance(record, dict):
                raise TypeError(
                    "save_jsonl expects a list of dictionaries; "
                    f"got item of type {type(record).__name__}."
                )
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into a list of dictionaries."""
    return df.to_dict(orient="records")


def records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of dictionaries into a DataFrame."""
    return pd.DataFrame(records)


def print_file_summary(path: str | Path, label: str | None = None) -> None:
    """Print a simple summary message for a saved file."""
    file_path = Path(path)
    prefix = f"{label}: " if label else ""
    print(f"{prefix}{file_path}")