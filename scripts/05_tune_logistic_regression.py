"""Tune frozen-embedding Logistic Regression models using validation macro-F1.

This script tunes Logistic Regression hyperparameters for the power dynamic and
respect level classifiers. It uses the validation set for model selection and
keeps the test set for final evaluation after selecting the best configuration.
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.embeddings import get_or_compute_embeddings  # noqa: E402
from src.modeling.train import (  # noqa: E402
    build_logistic_regression,
    compute_metrics,
    encode_target_labels,
    load_split_dataframes,
    print_training_summary,
    train_power_respect_models,
)
from src.utils.io import ensure_parent_dir, load_json, save_csv, save_json  # noqa: E402
from src.utils.paths import get_modeling_config_path, load_modeling_settings  # noqa: E402

POWER_TARGET = "final_power_dynamic"
RESPECT_TARGET = "final_respect_level"
TUNING_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "modeling" / "logistic_regression_tuning.csv"

DEFAULT_C_VALUES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
DEFAULT_CLASS_WEIGHTS = ["balanced", "none"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tune frozen-embedding Logistic Regression using validation macro-F1."
    )
    parser.add_argument(
        "--c-values",
        nargs="+",
        type=float,
        default=DEFAULT_C_VALUES,
        help="Candidate Logistic Regression C values to try.",
    )
    parser.add_argument(
        "--class-weights",
        nargs="+",
        type=str,
        default=DEFAULT_CLASS_WEIGHTS,
        choices=["balanced", "none"],
        help="Candidate class_weight settings to try.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="average_val_macro_f1",
        choices=["average_val_macro_f1", "power_val_macro_f1", "respect_val_macro_f1"],
        help="Validation metric used to select the final Logistic Regression setting.",
    )
    parser.add_argument(
        "--overwrite-embeddings",
        action="store_true",
        help="Recompute cached train/val/test embeddings before tuning.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable embedding progress bars.",
    )
    parser.add_argument(
        "--skip-final-train",
        action="store_true",
        help="Only write the tuning table; do not retrain/save the selected final model.",
    )
    return parser.parse_args()


def settings_with_logistic_params(
    settings: dict[str, Any],
    *,
    c_value: float,
    class_weight: str,
) -> dict[str, Any]:
    """Return a copied settings dict with Logistic Regression params overwritten."""
    updated_settings = deepcopy(settings)
    classifier_settings = updated_settings.get("classifier", {})
    if not isinstance(classifier_settings, dict):
        raise ValueError("modeling_config.json key 'classifier' must be an object.")

    logistic_settings = classifier_settings.get("logistic_regression", {})
    if not isinstance(logistic_settings, dict):
        raise ValueError("modeling_config.json classifier.logistic_regression must be an object.")

    classifier_settings["model_type"] = "logistic_regression"
    classifier_settings["class_weight"] = class_weight
    logistic_settings["C"] = float(c_value)
    classifier_settings["logistic_regression"] = logistic_settings
    updated_settings["classifier"] = classifier_settings
    return updated_settings


def evaluate_logistic_for_target(
    *,
    target_name: str,
    settings: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_embeddings: Any,
    val_embeddings: Any,
    test_embeddings: Any,
) -> dict[str, float]:
    """Train Logistic Regression for one target and return validation/test metrics."""
    label_encoder, y_train, y_val, y_test = encode_target_labels(
        train_df[target_name],
        val_df[target_name],
        test_df[target_name],
    )

    model = build_logistic_regression(settings)
    model.fit(train_embeddings, y_train)

    val_pred = model.predict(val_embeddings)
    test_pred = model.predict(test_embeddings)

    val_metrics = compute_metrics(y_val, val_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    return {
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_weighted_f1": val_metrics["weighted_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
    }


def tune_logistic_regression(
    *,
    c_values: list[float],
    class_weights: list[str],
    overwrite_embeddings: bool,
    show_progress: bool,
) -> pd.DataFrame:
    """Tune Logistic Regression settings on the validation set."""
    settings = load_modeling_settings()
    split_dataframes = load_split_dataframes()

    train_df = split_dataframes["train"]

    val_df = split_dataframes["val"]

    test_df = split_dataframes["test"]

    train_embeddings = get_or_compute_embeddings(
        train_df, "train", settings=settings, overwrite=overwrite_embeddings, show_progress=show_progress
    )
    val_embeddings = get_or_compute_embeddings(
        val_df, "val", settings=settings, overwrite=overwrite_embeddings, show_progress=show_progress
    )
    test_embeddings = get_or_compute_embeddings(
        test_df, "test", settings=settings, overwrite=overwrite_embeddings, show_progress=show_progress
    )

    rows: list[dict[str, Any]] = []
    for c_value in c_values:
        for class_weight in class_weights:
            print(f"Tuning Logistic Regression: C={c_value}, class_weight={class_weight}")
            candidate_settings = settings_with_logistic_params(
                settings,
                c_value=c_value,
                class_weight=class_weight,
            )

            power_metrics = evaluate_logistic_for_target(
                target_name=POWER_TARGET,
                settings=candidate_settings,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                train_embeddings=train_embeddings,
                val_embeddings=val_embeddings,
                test_embeddings=test_embeddings,
            )
            respect_metrics = evaluate_logistic_for_target(
                target_name=RESPECT_TARGET,
                settings=candidate_settings,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                train_embeddings=train_embeddings,
                val_embeddings=val_embeddings,
                test_embeddings=test_embeddings,
            )

            row = {
                "model_type": "logistic_regression",
                "C": float(c_value),
                "class_weight": class_weight,
                "power_val_accuracy": power_metrics["val_accuracy"],
                "power_val_macro_f1": power_metrics["val_macro_f1"],
                "power_val_weighted_f1": power_metrics["val_weighted_f1"],
                "power_test_accuracy": power_metrics["test_accuracy"],
                "power_test_macro_f1": power_metrics["test_macro_f1"],
                "power_test_weighted_f1": power_metrics["test_weighted_f1"],
                "respect_val_accuracy": respect_metrics["val_accuracy"],
                "respect_val_macro_f1": respect_metrics["val_macro_f1"],
                "respect_val_weighted_f1": respect_metrics["val_weighted_f1"],
                "respect_test_accuracy": respect_metrics["test_accuracy"],
                "respect_test_macro_f1": respect_metrics["test_macro_f1"],
                "respect_test_weighted_f1": respect_metrics["test_weighted_f1"],
            }
            row["average_val_macro_f1"] = (
                row["power_val_macro_f1"] + row["respect_val_macro_f1"]
            ) / 2
            row["average_test_macro_f1"] = (
                row["power_test_macro_f1"] + row["respect_test_macro_f1"]
            ) / 2
            rows.append(row)

    return pd.DataFrame(rows)


def select_best_setting(tuning_df: pd.DataFrame, selection_metric: str) -> dict[str, Any]:
    """Select the best Logistic Regression setting using a validation metric."""
    if selection_metric not in tuning_df.columns:
        raise ValueError(f"Selection metric not found in tuning table: {selection_metric}")

    best_index = tuning_df[selection_metric].idxmax()
    tuning_df["selected"] = False
    tuning_df.loc[best_index, "selected"] = True

    best_row = tuning_df.loc[best_index]
    return {
        "C": float(best_row["C"]),
        "class_weight": str(best_row["class_weight"]),
    }


def update_config_with_best_setting(best_setting: dict[str, Any]) -> None:
    """Persist the selected Logistic Regression setting in modeling_config.json."""
    config_path = get_modeling_config_path()
    config = load_json(config_path)
    classifier_config = config.get("classifier", {})
    if not isinstance(classifier_config, dict):
        raise ValueError("modeling_config.json key 'classifier' must be an object.")

    logistic_config = classifier_config.get("logistic_regression", {})
    if not isinstance(logistic_config, dict):
        raise ValueError("modeling_config.json classifier.logistic_regression must be an object.")

    classifier_config["model_type"] = "logistic_regression"
    classifier_config["class_weight"] = best_setting["class_weight"]
    logistic_config["C"] = float(best_setting["C"])
    classifier_config["logistic_regression"] = logistic_config
    config["classifier"] = classifier_config
    save_json(config, config_path)


def print_tuning_summary(tuning_df: pd.DataFrame, selection_metric: str) -> None:
    """Print a compact tuning summary."""
    display_columns = [
        "C",
        "class_weight",
        "power_val_macro_f1",
        "respect_val_macro_f1",
        "average_val_macro_f1",
        "power_test_macro_f1",
        "respect_test_macro_f1",
        "average_test_macro_f1",
        "selected",
    ]
    print("\nLogistic Regression tuning results:")
    print(tuning_df[display_columns].to_string(index=False))
    print(f"\nSelection metric: {selection_metric}")


def main() -> None:
    """Run Logistic Regression tuning and optionally save the selected final model."""
    args = parse_args()
    show_progress = not args.no_progress

    tuning_df = tune_logistic_regression(
        c_values=args.c_values,
        class_weights=args.class_weights,
        overwrite_embeddings=args.overwrite_embeddings,
        show_progress=show_progress,
    )
    best_setting = select_best_setting(tuning_df, args.selection_metric)

    ensure_parent_dir(TUNING_OUTPUT_PATH)
    save_csv(tuning_df, TUNING_OUTPUT_PATH)
    print_tuning_summary(tuning_df, args.selection_metric)
    print(
        "\nBest Logistic Regression setting by "
        f"{args.selection_metric}: C={best_setting['C']}, "
        f"class_weight={best_setting['class_weight']}"
    )
    print(f"Tuning table saved to: {TUNING_OUTPUT_PATH}")

    update_config_with_best_setting(best_setting)
    print("Updated configs/modeling_config.json with the selected Logistic Regression setting.")

    if args.skip_final_train:
        return

    print("\nTraining and saving final selected Logistic Regression models...")
    final_result = train_power_respect_models(
        overwrite_embeddings=False,
        show_progress=show_progress,
    )
    print_training_summary(final_result)


if __name__ == "__main__":
    main()