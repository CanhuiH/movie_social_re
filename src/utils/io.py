from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TEXT_ENCODING = "utf-8"
DEFAULT_JSON_ENCODING = "utf-8"
DEFAULT_JSONL_ENCODING = "utf-8"
DEFAULT_CSV_ENCODING = "utf-8"


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a file path if needed and return it."""
    file_path = Path(path)
    return ensure_dir(file_path.parent)


def read_text(path: str | Path, encoding: str = DEFAULT_TEXT_ENCODING) -> str:
    """Read a text file."""
    text_path = Path(path)
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")
    return text_path.read_text(encoding=encoding)


def write_text(path: str | Path, text: str, encoding: str = DEFAULT_TEXT_ENCODING) -> None:
    """Write a text file, creating the parent directory if needed."""
    text_path = Path(path)
    ensure_parent_dir(text_path)
    text_path.write_text(text, encoding=encoding)


def load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path, encoding=DEFAULT_CSV_ENCODING, **kwargs)


def save_csv(
    df: pd.DataFrame,
    path: str | Path,
    index: bool = False,
    **kwargs: Any,
) -> None:
    """Save a pandas DataFrame to CSV, creating the parent directory if needed."""
    csv_path = Path(path)
    ensure_parent_dir(csv_path)
    df.to_csv(csv_path, index=index, encoding=DEFAULT_CSV_ENCODING, **kwargs)


def load_json(path: str | Path) -> dict[str, Any] | list[Any]:
    """Load a JSON file."""
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with json_path.open("r", encoding=DEFAULT_JSON_ENCODING) as file:
        return json.load(file)


def save_json(
    data: dict[str, Any] | list[Any],
    path: str | Path,
    indent: int = 2,
) -> None:
    """Save a Python object to a JSON file, creating the parent directory if needed."""
    json_path = Path(path)
    ensure_parent_dir(json_path)
    with json_path.open("w", encoding=DEFAULT_JSON_ENCODING) as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)
        file.write("\n")


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
                    f"Invalid JSON on line {line_number} of {jsonl_path}: {exc}"
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
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise TypeError(
                    "save_jsonl expects a list of dictionaries; "
                    f"item {index} has type {type(record).__name__}."
                )
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    """Append one dictionary record to a JSONL file."""
    if not isinstance(record, dict):
        raise TypeError(
            f"append_jsonl expects a dictionary; got {type(record).__name__}."
        )

    jsonl_path = Path(path)
    ensure_parent_dir(jsonl_path)
    with jsonl_path.open("a", encoding=DEFAULT_JSONL_ENCODING) as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into a list of dictionaries."""
    return df.to_dict(orient="records")


def records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of dictionaries into a DataFrame."""
    return pd.DataFrame(records)


def require_columns(df: pd.DataFrame, required_columns: set[str], label: str = "DataFrame") -> None:
    """Raise a clear error if a DataFrame is missing required columns."""
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{label} is missing required columns: {sorted(missing_columns)}"
        )


def print_file_summary(path: str | Path, label: str | None = None) -> None:
    """Print a simple summary message for a saved file."""
    file_path = Path(path)
    prefix = f"{label}: " if label else ""
    if file_path.exists():
        size_kb = file_path.stat().st_size / 1024
        print(f"{prefix}{file_path} ({size_kb:.1f} KB)")
    else:
        print(f"{prefix}{file_path}")