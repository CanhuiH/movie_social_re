"""Train frozen-embedding Logistic Regression models for power dynamic and respect level."""

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

from src.modeling.embeddings import compute_and_save_split_embeddings
from src.utils.io import load_csv, save_csv, write_text
from src.utils.paths import (
    ensure_modeling_dirs,
    get_label_encoders_path,
    get_modeling_evaluation_path,
    get_power_dynamic_model_path,
    get_power_respect_test_path,
    get_power_respect_train_path,
    get_power_respect_val_path,
    get_respect_level_model_path,
    load_modeling_settings,
)



@dataclass(frozen=True)
class TargetTrainingResult:
    """Training and evaluation results for one target label."""

    target_name: str
    model: LogisticRegression
    label_encoder: LabelEncoder
    metrics: dict[str, float]
    metrics_path: Path
    predictions_path: Path
    report_path: Path
    confusion_matrix_path: Path
    model_path: Path


@dataclass(frozen=True)
class PowerRespectTrainingResult:
    """Training results for both power dynamic and respect level classifiers."""

    power_dynamic: TargetTrainingResult
    respect_level: TargetTrainingResult
    label_encoders_path: Path


def load_split_dataframes() -> dict[str, pd.DataFrame]:
    """Load prepared train/validation/test split CSV files."""
    split_paths = {
        "train": get_power_respect_train_path(),
        "val": get_power_respect_val_path(),
        "test": get_power_respect_test_path(),
    }

    missing_paths = [path for path in split_paths.values() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Prepared modeling split files are missing: "
            + ", ".join(str(path) for path in missing_paths)
            + ". Run python scripts/03_prepare_modeling_data.py first."
        )

    return {split_name: load_csv(path) for split_name, path in split_paths.items()}


def get_classifier_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Return classifier settings from modeling_config.json."""
    classifier_settings = settings.get("classifier", {})
    if not isinstance(classifier_settings, dict):
        raise ValueError("modeling_config.json key 'classifier' must be an object.")
    return classifier_settings


def get_class_weight(classifier_settings: dict[str, Any]) -> str | dict[int, float] | None:
    """Return normalized class weight configuration."""
    class_weight = classifier_settings.get("class_weight", "balanced")
    if class_weight == "none":
        return None
    return class_weight


def build_logistic_regression(settings: dict[str, Any]) -> LogisticRegression:
    """Build a Logistic Regression classifier from config settings."""
    classifier_settings = get_classifier_settings(settings)
    model_settings = classifier_settings.get("logistic_regression", {})
    if not isinstance(model_settings, dict):
        raise ValueError("modeling_config.json classifier.logistic_regression must be an object.")

    return LogisticRegression(
        max_iter=int(model_settings.get("max_iter", 2000)),
        class_weight=get_class_weight(classifier_settings),
        solver=str(model_settings.get("solver", "lbfgs")),
        C=float(model_settings.get("C", 1.0)),
        random_state=int(classifier_settings.get("random_state", 42)),
    )








def encode_target_labels(
    train_labels: pd.Series,
    val_labels: pd.Series,
    test_labels: pd.Series,
) -> tuple[LabelEncoder, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a label encoder on train labels and transform train/val/test labels."""
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels.astype(str))

    known_labels = set(label_encoder.classes_)
    for split_name, labels in [("validation", val_labels), ("test", test_labels)]:
        unseen_labels = sorted(set(labels.astype(str)) - known_labels)
        if unseen_labels:
            raise ValueError(
                f"The {split_name} split contains labels not present in training data: {unseen_labels}. "
                "Use a stratified split or increase the training size."
            )

    y_val = label_encoder.transform(val_labels.astype(str))
    y_test = label_encoder.transform(test_labels.astype(str))
    return label_encoder, y_train, y_val, y_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute compact evaluation metrics."""
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
    """Build a one-row summary plus per-class metrics dataframe."""
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
    split_name: str = "test",
) -> pd.DataFrame:
    """Build a prediction dataframe with original IDs and predicted labels."""
    output_df = df.copy()
    output_df["split"] = split_name
    output_df[f"true_{target_name}"] = label_encoder.inverse_transform(y_true)
    output_df[f"pred_{target_name}"] = label_encoder.inverse_transform(y_pred)
    output_df[f"correct_{target_name}"] = output_df[f"true_{target_name}"] == output_df[f"pred_{target_name}"]
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


def train_and_evaluate_target(
    target_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_embeddings: np.ndarray,
    val_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    settings: dict[str, Any],
    model_path: Path,
    metrics_path: Path,
    predictions_path: Path,
    report_path: Path,
    confusion_matrix_path: Path,
) -> TargetTrainingResult:
    """Train and evaluate one frozen-embedding Logistic Regression model."""
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if target_name not in df.columns:
            raise ValueError(f"Target column '{target_name}' not found in {split_name} dataframe.")

    label_encoder, y_train, y_val, y_test = encode_target_labels(
        train_df[target_name],
        val_df[target_name],
        test_df[target_name],
    )

    model = build_logistic_regression(settings)
    model.fit(train_embeddings, y_train)

    val_pred = model.predict(val_embeddings)
    test_pred = model.predict(test_embeddings)

    val_metrics_df = build_metrics_dataframe(
        target_name=target_name,
        split_name="val",
        y_true=y_val,
        y_pred=val_pred,
        label_encoder=label_encoder,
    )
    test_metrics_df = build_metrics_dataframe(
        target_name=target_name,
        split_name="test",
        y_true=y_test,
        y_pred=test_pred,
        label_encoder=label_encoder,
    )
    metrics_df = pd.concat([val_metrics_df, test_metrics_df], ignore_index=True)
    save_csv(metrics_df, metrics_path)

    predictions_df = build_predictions_dataframe(
        df=test_df,
        target_name=target_name,
        y_true=y_test,
        y_pred=test_pred,
        label_encoder=label_encoder,
        split_name="test",
    )
    save_csv(predictions_df, predictions_path)

    report = classification_report(
        y_test,
        test_pred,
        target_names=list(label_encoder.classes_),
        zero_division=0,
    )
    write_text(report_path, report)
    save_confusion_matrix(y_test, test_pred, label_encoder, confusion_matrix_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    return TargetTrainingResult(
        target_name=target_name,
        model=model,
        label_encoder=label_encoder,
        metrics=compute_metrics(y_test, test_pred),
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        report_path=report_path,
        confusion_matrix_path=confusion_matrix_path,
        model_path=model_path,
    )


def train_power_respect_models(
    overwrite_embeddings: bool = False,
    show_progress: bool = True,
) -> PowerRespectTrainingResult:
    """Train power dynamic and respect level frozen-embedding Logistic Regression models."""
    ensure_modeling_dirs()
    settings = load_modeling_settings()

    targets = settings.get("targets", ["final_power_dynamic", "final_respect_level"])
    if not isinstance(targets, list) or len(targets) != 2:
        raise ValueError("modeling_config.json targets must contain exactly two target columns.")
    power_target, respect_target = str(targets[0]), str(targets[1])

    split_dataframes = load_split_dataframes()
    split_embeddings = compute_and_save_split_embeddings(
        split_dataframes,
        settings=settings,
        overwrite=overwrite_embeddings,
        show_progress=show_progress,
    )

    power_result = train_and_evaluate_target(
        target_name=power_target,
        train_df=split_dataframes["train"],
        val_df=split_dataframes["val"],
        test_df=split_dataframes["test"],
        train_embeddings=split_embeddings["train"],
        val_embeddings=split_embeddings["val"],
        test_embeddings=split_embeddings["test"],
        settings=settings,
        model_path=get_power_dynamic_model_path(),
        metrics_path=get_modeling_evaluation_path("power_dynamic_metrics_file"),
        predictions_path=get_modeling_evaluation_path("power_dynamic_predictions_file"),
        report_path=get_modeling_evaluation_path("power_dynamic_report_file"),
        confusion_matrix_path=get_modeling_evaluation_path("power_dynamic_confusion_matrix_file"),
    )

    respect_result = train_and_evaluate_target(
        target_name=respect_target,
        train_df=split_dataframes["train"],
        val_df=split_dataframes["val"],
        test_df=split_dataframes["test"],
        train_embeddings=split_embeddings["train"],
        val_embeddings=split_embeddings["val"],
        test_embeddings=split_embeddings["test"],
        settings=settings,
        model_path=get_respect_level_model_path(),
        metrics_path=get_modeling_evaluation_path("respect_level_metrics_file"),
        predictions_path=get_modeling_evaluation_path("respect_level_predictions_file"),
        report_path=get_modeling_evaluation_path("respect_level_report_file"),
        confusion_matrix_path=get_modeling_evaluation_path("respect_level_confusion_matrix_file"),
    )

    label_encoders_path = get_label_encoders_path()
    label_encoders_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            power_target: power_result.label_encoder,
            respect_target: respect_result.label_encoder,
        },
        label_encoders_path,
    )

    return PowerRespectTrainingResult(
        power_dynamic=power_result,
        respect_level=respect_result,
        label_encoders_path=label_encoders_path,
    )


def print_training_summary(result: PowerRespectTrainingResult) -> None:
    """Print a compact training summary."""
    print("Power/respect model training complete")
    classifier_name = result.power_dynamic.model.__class__.__name__
    print(f"Classifier: {classifier_name}")
    for title, target_result in [
        ("Power dynamic", result.power_dynamic),
        ("Respect level", result.respect_level),
    ]:
        print(f"\n{title}: {target_result.target_name}")
        print(f"  accuracy   : {target_result.metrics['accuracy']:.4f}")
        print(f"  macro-F1   : {target_result.metrics['macro_f1']:.4f}")
        print(f"  weighted-F1: {target_result.metrics['weighted_f1']:.4f}")
        print(f"  model      : {target_result.model_path}")
        print(f"  metrics    : {target_result.metrics_path}")
        print(f"  predictions: {target_result.predictions_path}")
        print(f"  report     : {target_result.report_path}")
        print(f"  confusion  : {target_result.confusion_matrix_path}")
    print(f"\nLabel encoders: {result.label_encoders_path}")


__all__ = [
    "PowerRespectTrainingResult",
    "TargetTrainingResult",
    "build_logistic_regression",
    "build_metrics_dataframe",
    "build_predictions_dataframe",
    "compute_metrics",
    "encode_target_labels",
    "get_classifier_settings",
    "get_class_weight",
    "load_split_dataframes",
    "print_training_summary",
    "save_confusion_matrix",
    "train_and_evaluate_target",
    "train_power_respect_models",
]