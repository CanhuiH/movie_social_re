"""Command-line entry point for training power/respect classifiers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.train import (  # noqa: E402
    print_training_summary,
    train_power_respect_models,
)
from src.utils.paths import (  # noqa: E402
    ensure_modeling_dirs,
    get_modeling_path_summary,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train frozen-embedding Logistic Regression models for power/respect labels."
    )
    parser.add_argument(
        "--overwrite-embeddings",
        action="store_true",
        help="Recompute cached train/val/test embeddings even if .npy files already exist.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars while computing embeddings.",
    )
    parser.add_argument(
        "--print-paths",
        action="store_true",
        help="Print modeling input/output paths before training.",
    )
    return parser.parse_args()


def print_paths() -> None:
    """Print key modeling paths."""
    print("Modeling paths:")
    for name, path in get_modeling_path_summary().items():
        print(f"  {name}: {path}")




def main() -> None:
    """Train power dynamic and respect level classifiers."""
    args = parse_args()
    ensure_modeling_dirs()

    if args.print_paths:
        print_paths()


    result = train_power_respect_models(
        overwrite_embeddings=args.overwrite_embeddings,
        show_progress=not args.no_progress,
    )
    print_training_summary(result)


if __name__ == "__main__":
    main()