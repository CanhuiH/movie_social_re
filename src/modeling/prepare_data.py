"""Prepare labeled power/respect data for BERT + Logistic Regression modeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import load_csv, load_json, require_columns, save_csv
from src.utils.paths import (
    ensure_modeling_dirs,
    get_labeled_power_respect_path,
    get_power_respect_modeling_data_path,
    get_power_respect_schema_path,
    get_power_respect_test_path,
    get_power_respect_train_path,
    get_power_respect_val_path,
    load_modeling_settings,
    project_path,
)


@dataclass(frozen=True)
class ModelingDataPaths:
    """Paths produced by the modeling data preparation step."""

    full_data_path: Path
    train_path: Path
    val_path: Path
    test_path: Path
    translation_eval_path: Path | None = None


@dataclass(frozen=True)
class ModelingDataSummary:
    """Summary statistics for the prepared modeling data."""

    input_rows: int
    output_rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    translation_eval_rows: int
    dropped_rows: int
    power_dynamic_counts: dict[str, int]
    respect_level_counts: dict[str, int]
    paths: ModelingDataPaths


def normalize_label(value: Any, missing_label: str = "unclear") -> str:
    """Normalize a label value to lowercase snake_case."""
    if pd.isna(value):
        return missing_label

    label = str(value).strip().lower()
    if not label:
        return missing_label

    label = label.replace("-", " ").replace("/", " ")
    label = "_".join(label.split())
    return label


def combine_text_fields(
    row: pd.Series,
    context_column: str,
    current_text_column: str,
) -> str:
    """Combine dialogue context and current turn into one model input string."""
    context_text = "" if pd.isna(row.get(context_column)) else str(row.get(context_column)).strip()
    current_text = "" if pd.isna(row.get(current_text_column)) else str(row.get(current_text_column)).strip()

    if context_text and current_text:
        return f"Context:\n{context_text}\n\nCurrent turn:\n{current_text}"
    if current_text:
        return f"Current turn:\n{current_text}"
    if context_text:
        return f"Context:\n{context_text}"
    return ""


def load_power_respect_schema() -> dict[str, Any]:
    """Load the power/respect label schema."""
    schema_path = get_power_respect_schema_path()
    if not schema_path.exists():
        raise FileNotFoundError(f"Power/respect schema file not found: {schema_path}")

    schema = load_json(schema_path)
    if not isinstance(schema, dict):
        raise ValueError(f"Power/respect schema must be a JSON object: {schema_path}")
    return schema


def get_allowed_labels(schema: dict[str, Any], section_name: str) -> set[str]:
    """Return allowed labels from a schema section."""
    section = schema.get(section_name, {})
    if not isinstance(section, dict):
        raise ValueError(f"Schema section '{section_name}' must be an object.")

    labels = section.get("labels", [])
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"Schema section '{section_name}' must define a non-empty labels list.")

    return {normalize_label(label) for label in labels}


def validate_target_labels(
    df: pd.DataFrame,
    power_target: str,
    respect_target: str,
    schema: dict[str, Any],
) -> None:
    """Validate that target labels are included in the configured schema."""
    allowed_power_labels = get_allowed_labels(schema, "power_dynamic")
    allowed_respect_labels = get_allowed_labels(schema, "respect_level")

    observed_power_labels = set(df[power_target].dropna().unique())
    observed_respect_labels = set(df[respect_target].dropna().unique())

    unknown_power_labels = sorted(observed_power_labels - allowed_power_labels)
    unknown_respect_labels = sorted(observed_respect_labels - allowed_respect_labels)

    if unknown_power_labels:
        raise ValueError(
            f"Unknown power dynamic labels found: {unknown_power_labels}. "
            f"Allowed labels: {sorted(allowed_power_labels)}"
        )

    if unknown_respect_labels:
        raise ValueError(
            f"Unknown respect level labels found: {unknown_respect_labels}. "
            f"Allowed labels: {sorted(allowed_respect_labels)}"
        )


def clean_labeled_dataframe(
    df: pd.DataFrame,
    settings: dict[str, Any],
    schema: dict[str, Any],
) -> pd.DataFrame:
    """Clean, validate, and prepare the labeled dataframe for modeling."""
    required_columns = settings.get("required_columns", [])
    if not isinstance(required_columns, list) or not required_columns:
        raise ValueError("modeling_config.json must define a non-empty required_columns list.")
    require_columns(df, set(required_columns))

    text_columns = settings.get("text_columns", [])
    if not isinstance(text_columns, list) or len(text_columns) != 2:
        raise ValueError("modeling_config.json text_columns must contain exactly two columns.")

    targets = settings.get("targets", [])
    if not isinstance(targets, list) or len(targets) != 2:
        raise ValueError("modeling_config.json targets must contain exactly two target columns.")

    context_column, current_text_column = text_columns
    power_target, respect_target = targets
    combined_text_column = str(settings.get("combined_text_column", "model_text"))

    label_policy = settings.get("label_policy", {})
    if not isinstance(label_policy, dict):
        raise ValueError("modeling_config.json key 'label_policy' must be an object.")

    missing_label = str(schema.get("label_normalization", {}).get("missing_label", "unclear"))

    cleaned = df.copy()
    cleaned[power_target] = cleaned[power_target].apply(lambda value: normalize_label(value, missing_label))
    cleaned[respect_target] = cleaned[respect_target].apply(lambda value: normalize_label(value, missing_label))

    validate_target_labels(cleaned, power_target, respect_target, schema)

    cleaned[context_column] = cleaned[context_column].fillna("").astype(str)
    cleaned[current_text_column] = cleaned[current_text_column].fillna("").astype(str)
    cleaned[combined_text_column] = cleaned.apply(
        combine_text_fields,
        axis=1,
        context_column=context_column,
        current_text_column=current_text_column,
    )

    cleaned = cleaned[cleaned[combined_text_column].str.strip().astype(bool)].copy()

    if bool(label_policy.get("drop_unclear_power_dynamic", False)):
        cleaned = cleaned[cleaned[power_target] != "unclear"].copy()

    if bool(label_policy.get("drop_unclear_respect_level", False)):
        cleaned = cleaned[cleaned[respect_target] != "unclear"].copy()

    cleaned = cleaned.reset_index(drop=True)
    cleaned["modeling_row_id"] = range(len(cleaned))

    output_columns = []
    for column in ["modeling_row_id"] + required_columns + [combined_text_column]:
        if column in cleaned.columns and column not in output_columns:
            output_columns.append(column)

    optional_columns = [
        "speaker_id",
        "speaker_name",
        "listener_id",
        "listener_name",
        "judge1_power_dynamic",
        "judge2_power_dynamic",
        "judge1_respect_level",
        "judge2_respect_level",
    ]
    for column in optional_columns:
        if column in cleaned.columns and column not in output_columns:
            output_columns.append(column)

    return cleaned[output_columns].copy()


def safe_stratify_labels(labels: pd.Series) -> pd.Series | None:
    """Return labels for stratification only when every class has at least two rows."""
    counts = labels.value_counts(dropna=False)
    if len(counts) < 2:
        return None
    if counts.min() < 2:
        return None
    return labels


def split_fixed_size(
    df: pd.DataFrame,
    train_size: int,
    val_size: int,
    test_size: int,
    translation_eval_size: int,
    random_state: int,
    stratify_target: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into fixed-size train/val/test/translation-eval subsets."""
    total_requested = train_size + val_size + test_size + translation_eval_size
    if total_requested > len(df):
        raise ValueError(
            f"Requested {total_requested} rows, but only {len(df)} prepared rows are available."
        )
    if min(train_size, val_size, test_size, translation_eval_size) < 0:
        raise ValueError("Fixed split sizes must be non-negative.")
    if train_size <= 0:
        raise ValueError("train_size must be positive for fixed-size splitting.")

    working_df = df.copy()

    stratify_for_train = safe_stratify_labels(working_df[stratify_target])
    train_df, remaining_df = train_test_split(
        working_df,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_for_train,
    )

    if val_size == 0:
        val_df = remaining_df.iloc[0:0].copy()
    elif val_size == len(remaining_df):
        val_df = remaining_df.copy()
        remaining_df = remaining_df.iloc[0:0].copy()
    else:
        stratify_for_val = safe_stratify_labels(remaining_df[stratify_target])
        val_df, remaining_df = train_test_split(
            remaining_df,
            train_size=val_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify_for_val,
        )

    if test_size == 0:
        test_df = remaining_df.iloc[0:0].copy()
    elif test_size == len(remaining_df):
        test_df = remaining_df.copy()
        remaining_df = remaining_df.iloc[0:0].copy()
    else:
        stratify_for_test = safe_stratify_labels(remaining_df[stratify_target])
        test_df, remaining_df = train_test_split(
            remaining_df,
            train_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify_for_test,
        )

    if translation_eval_size == 0:
        translation_eval_df = remaining_df.iloc[0:0].copy()
    elif translation_eval_size == len(remaining_df):
        translation_eval_df = remaining_df.copy()
        remaining_df = remaining_df.iloc[0:0].copy()
    else:
        stratify_for_translation = safe_stratify_labels(remaining_df[stratify_target])
        translation_eval_df, remaining_df = train_test_split(
            remaining_df,
            train_size=translation_eval_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify_for_translation,
        )

    return (
        train_df.sort_values("modeling_row_id").reset_index(drop=True),
        val_df.sort_values("modeling_row_id").reset_index(drop=True),
        test_df.sort_values("modeling_row_id").reset_index(drop=True),
        translation_eval_df.sort_values("modeling_row_id").reset_index(drop=True),
    )


def get_optional_translation_eval_path(settings: dict[str, Any]) -> Path | None:
    """Return the optional translation-evaluation split path if configured."""
    split_settings = settings.get("split", {})
    if not isinstance(split_settings, dict):
        return None

    translation_eval_file = split_settings.get("translation_eval_file")
    if not isinstance(translation_eval_file, str) or not translation_eval_file.strip():
        return None

    return project_path(translation_eval_file)


def split_modeling_dataframe(
    df: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split prepared data into train, validation, test, and optional translation-eval sets."""
    split_settings = settings.get("split", {})
    if not isinstance(split_settings, dict):
        raise ValueError("modeling_config.json key 'split' must be an object.")

    random_state = int(split_settings.get("random_state", 42))
    stratify_target = str(split_settings.get("stratify_target", "final_power_dynamic"))
    split_strategy = str(split_settings.get("split_strategy", "ratio")).lower()

    if stratify_target not in df.columns:
        raise ValueError(f"stratify_target column not found: {stratify_target}")

    if split_strategy == "fixed_size":
        train_size = int(split_settings.get("train_size", 800))
        val_size = int(split_settings.get("val_size", 200))
        test_size = int(split_settings.get("test_size", 300))
        translation_eval_size = int(split_settings.get("translation_eval_size", 0))
        return split_fixed_size(
            df=df,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            translation_eval_size=translation_eval_size,
            random_state=random_state,
            stratify_target=stratify_target,
        )

    test_size = float(split_settings.get("test_size", 0.2))
    val_size = float(split_settings.get("val_size", 0.1))

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1 for ratio splitting.")
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be between 0 and 1 for ratio splitting.")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1 for ratio splitting.")

    stratify_for_test = safe_stratify_labels(df[stratify_target])
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_for_test,
    )

    if val_size == 0:
        train_df = train_val_df.copy()
        val_df = train_val_df.iloc[0:0].copy()
    else:
        relative_val_size = val_size / (1 - test_size)
        stratify_for_val = safe_stratify_labels(train_val_df[stratify_target])
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify_for_val,
        )

    translation_eval_df = df.iloc[0:0].copy()
    return (
        train_df.sort_values("modeling_row_id").reset_index(drop=True),
        val_df.sort_values("modeling_row_id").reset_index(drop=True),
        test_df.sort_values("modeling_row_id").reset_index(drop=True),
        translation_eval_df.reset_index(drop=True),
    )


def prepare_power_respect_modeling_data(
    input_path: Path | None = None,
    overwrite: bool = False,
) -> ModelingDataSummary:
    """Prepare the labeled power/respect dataset and save train/val/test splits."""
    ensure_modeling_dirs()

    settings = load_modeling_settings()
    schema = load_power_respect_schema()

    source_path = input_path or get_labeled_power_respect_path()
    if not source_path.exists():
        raise FileNotFoundError(
            f"Labeled power/respect data not found: {source_path}. "
            "Put the file at data/labeled/power_respect_labels.csv."
        )

    output_path = get_power_respect_modeling_data_path()
    train_path = get_power_respect_train_path()
    val_path = get_power_respect_val_path()
    test_path = get_power_respect_test_path()
    translation_eval_path = get_optional_translation_eval_path(settings)

    output_files = [output_path, train_path, val_path, test_path]
    if translation_eval_path is not None:
        output_files.append(translation_eval_path)
    existing_outputs = [path for path in output_files if path.exists()]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            "Prepared modeling outputs already exist: "
            + ", ".join(str(path) for path in existing_outputs)
            + ". Use overwrite=True or pass --overwrite to regenerate them."
        )

    raw_df = load_csv(source_path)
    prepared_df = clean_labeled_dataframe(raw_df, settings=settings, schema=schema)
    train_df, val_df, test_df, translation_eval_df = split_modeling_dataframe(
        prepared_df,
        settings=settings,
    )

    save_csv(prepared_df, output_path)
    save_csv(train_df, train_path)
    save_csv(val_df, val_path)
    save_csv(test_df, test_path)
    if translation_eval_path is not None:
        save_csv(translation_eval_df, translation_eval_path)

    targets = settings.get("targets", ["final_power_dynamic", "final_respect_level"])
    power_target, respect_target = targets

    return ModelingDataSummary(
        input_rows=len(raw_df),
        output_rows=len(prepared_df),
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df),
        translation_eval_rows=len(translation_eval_df),
        dropped_rows=len(raw_df) - len(prepared_df),
        power_dynamic_counts=prepared_df[power_target].value_counts().to_dict(),
        respect_level_counts=prepared_df[respect_target].value_counts().to_dict(),
        paths=ModelingDataPaths(
            full_data_path=output_path,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            translation_eval_path=translation_eval_path,
        ),
    )


def print_modeling_data_summary(summary: ModelingDataSummary) -> None:
    """Print a compact summary of prepared modeling data."""
    print("Prepared power/respect modeling data")
    print(f"  input rows  : {summary.input_rows}")
    print(f"  output rows : {summary.output_rows}")
    print(f"  dropped rows: {summary.dropped_rows}")
    print(f"  train rows  : {summary.train_rows}")
    print(f"  val rows    : {summary.val_rows}")
    print(f"  test rows   : {summary.test_rows}")
    print(f"  translation : {summary.translation_eval_rows}")
    print("  power dynamic counts:")
    for label, count in sorted(summary.power_dynamic_counts.items()):
        print(f"    {label}: {count}")
    print("  respect level counts:")
    for label, count in sorted(summary.respect_level_counts.items()):
        print(f"    {label}: {count}")
    print("  saved files:")
    print(f"    full : {summary.paths.full_data_path}")
    print(f"    train: {summary.paths.train_path}")
    print(f"    val  : {summary.paths.val_path}")
    print(f"    test : {summary.paths.test_path}")
    if summary.paths.translation_eval_path is not None:
        print(f"    translation_eval: {summary.paths.translation_eval_path}")


__all__ = [
    "ModelingDataPaths",
    "ModelingDataSummary",
    "clean_labeled_dataframe",
    "combine_text_fields",
    "load_power_respect_schema",
    "normalize_label",
    "prepare_power_respect_modeling_data",
    "print_modeling_data_summary",
    "get_optional_translation_eval_path",
    "split_fixed_size",
    "split_modeling_dataframe",
]