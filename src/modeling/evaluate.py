

"""Evaluation utilities for power/respect Logistic Regression models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder

from src.modeling.embeddings import get_or_compute_embeddings
from src.utils.io import load_csv, save_csv, write_text
from src.utils.paths import (
    ensure_modeling_dirs,
    get_label_encoders_path,
    get_modeling_evaluation_path,
    get_power_dynamic_model_path,
    get_power_respect_test_path,
    get_respect_level_model_path,
    load_modeling_settings,
)


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation result for one target model."""

    target_name: str
    split_name: str
    metrics: dict[str, float]
    metrics_path: Path
    predictions_path: Path
    report_path: Path
    confusion_matrix_path: Path


@dataclass(frozen=True)
class PowerRespectEvaluationResult:
    """Evaluation results for power dynamic and respect level models."""

    power_dynamic: EvaluationResult
    respect_level: EvaluationResult


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute compact classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def build_metrics_dataframe(
    target_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
) -> pd.DataFrame:
    """Build an overall plus per-class metrics dataframe."""
    summary_metrics = compute_metrics(y_true, y_pred)
    rows: list[dict[str, Any]] = [
        {
            "target": target_name,
            "split": split_name,
            "label": "__overall__",
            "accuracy": summary_metrics["accuracy"],
            "precision": np.nan,
            "recall": np.nan,
            "f1": summary_metrics["macro_f1"],
            "macro_f1": summary_metrics["macro_f1"],
            "weighted_f1": summary_metrics["weighted_f1"],
            "support": int(len(y_true)),
        }
    ]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        zero_division=0,
    )

    for index, label in enumerate(label_encoder.classes_):
        rows.append(
            {
                "target": target_name,
                "split": split_name,
                "label": label,
                "accuracy": np.nan,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "macro_f1": np.nan,
                "weighted_f1": np.nan,
                "support": int(support[index]),
            }
        )

    return pd.DataFrame(rows)


def build_predictions_dataframe(
    df: pd.DataFrame,
    target_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
    split_name: str,
) -> pd.DataFrame:
    """Build a prediction dataframe containing original rows and predicted labels."""
    output_df = df.copy()
    true_column = f"true_{target_name}"
    pred_column = f"pred_{target_name}"
    correct_column = f"correct_{target_name}"

    output_df["split"] = split_name
    output_df[true_column] = label_encoder.inverse_transform(y_true)
    output_df[pred_column] = label_encoder.inverse_transform(y_pred)
    output_df[correct_column] = output_df[true_column] == output_df[pred_column]
    return output_df


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
    output_path: Path,
) -> None:
    """Save a confusion matrix as a CSV file."""
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
    )
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in label_encoder.classes_],
        columns=[f"pred_{label}" for label in label_encoder.classes_],
    )
    save_csv(matrix_df, output_path)


def load_saved_models_and_encoders() -> tuple[dict[str, LogisticRegression], dict[str, LabelEncoder]]:
    """Load trained models and label encoders from disk."""
    settings = load_modeling_settings()
    targets = settings.get("targets", ["final_power_dynamic", "final_respect_level"])
    if not isinstance(targets, list) or len(targets) != 2:
        raise ValueError("modeling_config.json targets must contain exactly two target columns.")

    power_target, respect_target = str(targets[0]), str(targets[1])
    model_paths = {
        power_target: get_power_dynamic_model_path(),
        respect_target: get_respect_level_model_path(),
    }

    missing_models = [path for path in model_paths.values() if not path.exists()]
    if missing_models:
        raise FileNotFoundError(
            "Saved model files are missing: "
            + ", ".join(str(path) for path in missing_models)
            + ". Run python scripts/04_train_power_respect.py first."
        )

    label_encoders_path = get_label_encoders_path()
    if not label_encoders_path.exists():
        raise FileNotFoundError(
            f"Label encoders file not found: {label_encoders_path}. "
            "Run python scripts/04_train_power_respect.py first."
        )

    models = {target: joblib.load(path) for target, path in model_paths.items()}
    for target, model in models.items():
        if not isinstance(model, LogisticRegression):
            raise ValueError(
                f"Saved model for {target} must be a LogisticRegression instance. "
                f"Found: {type(model).__name__}. Re-run python scripts/04_train_power_respect.py."
            )
    label_encoders = joblib.load(label_encoders_path)
    if not isinstance(label_encoders, dict):
        raise ValueError(f"Label encoders file must contain a dictionary: {label_encoders_path}")

    return models, label_encoders


def evaluate_target_model(
    target_name: str,
    model: LogisticRegression,
    label_encoder: LabelEncoder,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    split_name: str,
    metrics_path: Path,
    predictions_path: Path,
    report_path: Path,
    confusion_matrix_path: Path,
) -> EvaluationResult:
    """Evaluate one saved target model on a dataframe split."""
    if target_name not in df.columns:
        raise ValueError(f"Target column not found in evaluation dataframe: {target_name}")
    if len(df) != len(embeddings):
        raise ValueError(
            f"Row count mismatch for {target_name}: {len(df)} dataframe rows vs "
            f"{len(embeddings)} embedding rows."
        )

    labels = df[target_name].astype(str)
    unseen_labels = sorted(set(labels) - set(label_encoder.classes_))
    if unseen_labels:
        raise ValueError(
            f"Evaluation split contains labels unseen during training for {target_name}: {unseen_labels}"
        )

    y_true = label_encoder.transform(labels)
    y_pred = model.predict(embeddings)

    metrics_df = build_metrics_dataframe(
        target_name=target_name,
        split_name=split_name,
        y_true=y_true,
        y_pred=y_pred,
        label_encoder=label_encoder,
    )
    save_csv(metrics_df, metrics_path)

    predictions_df = build_predictions_dataframe(
        df=df,
        target_name=target_name,
        y_true=y_true,
        y_pred=y_pred,
        label_encoder=label_encoder,
        split_name=split_name,
    )
    save_csv(predictions_df, predictions_path)

    report = classification_report(
        y_true,
        y_pred,
        target_names=list(label_encoder.classes_),
        zero_division=0,
    )
    write_text(report_path, report)
    save_confusion_matrix(y_true, y_pred, label_encoder, confusion_matrix_path)

    return EvaluationResult(
        target_name=target_name,
        split_name=split_name,
        metrics=compute_metrics(y_true, y_pred),
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        report_path=report_path,
        confusion_matrix_path=confusion_matrix_path,
    )


def evaluate_power_respect_models(
    split_path: Path | None = None,
    split_name: str = "test",
    overwrite_embeddings: bool = False,
    show_progress: bool = True,
) -> PowerRespectEvaluationResult:
    """Evaluate saved power/respect models on a prepared split.

    By default, this evaluates on data/modeling/test.csv. A custom split can be
    passed with split_path, for example data/modeling/translation_eval.csv.
    """
    ensure_modeling_dirs()
    settings = load_modeling_settings()
    targets = settings.get("targets", ["final_power_dynamic", "final_respect_level"])
    if not isinstance(targets, list) or len(targets) != 2:
        raise ValueError("modeling_config.json targets must contain exactly two target columns.")

    power_target, respect_target = str(targets[0]), str(targets[1])
    evaluation_path = split_path or get_power_respect_test_path()
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Evaluation split file not found: {evaluation_path}")

    df = load_csv(evaluation_path)
    embeddings = get_or_compute_embeddings(
        df,
        split_name=split_name,
        settings=settings,
        overwrite=overwrite_embeddings,
        show_progress=show_progress,
    )

    models, label_encoders = load_saved_models_and_encoders()

    power_result = evaluate_target_model(
        target_name=power_target,
        model=models[power_target],
        label_encoder=label_encoders[power_target],
        df=df,
        embeddings=embeddings,
        split_name=split_name,
        metrics_path=get_modeling_evaluation_path("power_dynamic_metrics_file"),
        predictions_path=get_modeling_evaluation_path("power_dynamic_predictions_file"),
        report_path=get_modeling_evaluation_path("power_dynamic_report_file"),
        confusion_matrix_path=get_modeling_evaluation_path("power_dynamic_confusion_matrix_file"),
    )

    respect_result = evaluate_target_model(
        target_name=respect_target,
        model=models[respect_target],
        label_encoder=label_encoders[respect_target],
        df=df,
        embeddings=embeddings,
        split_name=split_name,
        metrics_path=get_modeling_evaluation_path("respect_level_metrics_file"),
        predictions_path=get_modeling_evaluation_path("respect_level_predictions_file"),
        report_path=get_modeling_evaluation_path("respect_level_report_file"),
        confusion_matrix_path=get_modeling_evaluation_path("respect_level_confusion_matrix_file"),
    )

    return PowerRespectEvaluationResult(
        power_dynamic=power_result,
        respect_level=respect_result,
    )


def print_evaluation_summary(result: PowerRespectEvaluationResult) -> None:
    """Print compact evaluation results."""
    print("Power/respect evaluation complete")
    for title, target_result in [
        ("Power dynamic", result.power_dynamic),
        ("Respect level", result.respect_level),
    ]:
        print(f"\n{title}: {target_result.target_name}")
        print(f"  split      : {target_result.split_name}")
        print(f"  accuracy   : {target_result.metrics['accuracy']:.4f}")
        print(f"  macro-F1   : {target_result.metrics['macro_f1']:.4f}")
        print(f"  weighted-F1: {target_result.metrics['weighted_f1']:.4f}")
        print(f"  metrics    : {target_result.metrics_path}")
        print(f"  predictions: {target_result.predictions_path}")
        print(f"  report     : {target_result.report_path}")
        print(f"  confusion  : {target_result.confusion_matrix_path}")


__all__ = [
    "EvaluationResult",
    "PowerRespectEvaluationResult",
    "build_metrics_dataframe",
    "build_predictions_dataframe",
    "compute_metrics",
    "evaluate_power_respect_models",
    "evaluate_target_model",
    "load_saved_models_and_encoders",
    "print_evaluation_summary",
    "save_confusion_matrix",
]