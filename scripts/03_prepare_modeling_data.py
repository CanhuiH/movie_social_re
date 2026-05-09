

"""Command-line entry point for preparing power/respect modeling data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.prepare_data import (  # noqa: E402
    prepare_power_respect_modeling_data,
    print_modeling_data_summary,
)
from src.utils.paths import (  # noqa: E402
    ensure_modeling_dirs,
    get_labeled_power_respect_path,
    get_modeling_path_summary,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare labeled power/respect data for BERT + Logistic Regression modeling."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help=(
            "Optional path to a labeled power/respect CSV file. "
            "Defaults to data/labeled/power_respect_labels.csv from configs/modeling_config.json."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prepared modeling CSV files.",
    )
    parser.add_argument(
        "--print-paths",
        action="store_true",
        help="Print modeling input/output paths before preparing data.",
    )
    return parser.parse_args()


def print_paths() -> None:
    """Print the main modeling paths."""
    print("Modeling paths:")
    for name, path in get_modeling_path_summary().items():
        print(f"  {name}: {path}")


def main() -> None:
    """Prepare modeling data and save train/val/test splits."""
    args = parse_args()
    ensure_modeling_dirs()

    if args.print_paths:
        print_paths()

    input_path = args.input_file
    if input_path is not None and not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    default_input_path = get_labeled_power_respect_path()
    source_path = input_path or default_input_path
    if not source_path.exists():
        raise FileNotFoundError(
            f"Labeled data file not found: {source_path}\n"
            "Move your file to data/labeled/power_respect_labels.csv, or pass --input-file PATH."
        )

    summary = prepare_power_respect_modeling_data(
        input_path=input_path,
        overwrite=args.overwrite,
    )
    print_modeling_data_summary(summary)


if __name__ == "__main__":
    main()