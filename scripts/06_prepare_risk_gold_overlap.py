

"""Command-line entry point for preparing the risk-gold overlap dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prepare_risk_gold_overlap import (  # noqa: E402
    DEFAULT_GOLD_LABELS_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_RISK_FILENAME,
    DEFAULT_SUMMARY_PATH,
    build_risk_gold_overlap,
    get_relationship_extraction_path,
    get_risk_prediction_path,
    print_overlap_summary,
)
from src.config import load_movies  # noqa: E402
from src.utils.paths import project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare overlap between risk-predicted dialogue rows, gold power/respect labels, "
            "and generative relationship extraction outputs."
        )
    )
    parser.add_argument(
        "--risk-filename",
        type=str,
        default=DEFAULT_RISK_FILENAME,
        help=(
            "Risk prediction CSV filename inside each movie folder under "
            "data/data_prelabel_predictions/<movie_slug>/."
        ),
    )
    parser.add_argument(
        "--gold-labels-path",
        type=Path,
        default=DEFAULT_GOLD_LABELS_PATH,
        help="Path to the gold power/respect label CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for the risk-gold overlap CSV.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Output path for the overlap summary CSV.",
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=None,
        help="Optional single movie name to process. If omitted, uses configs/movies.json.",
    )
    parser.add_argument(
        "--print-paths",
        action="store_true",
        help="Print expected input/output paths before running.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """Resolve a path relative to the project root if needed."""
    if path.is_absolute():
        return path
    return project_path(path)


def get_movie_names(movie: str | None) -> list[str]:
    """Return the movie names to process."""
    if movie is not None:
        return [movie]
    return load_movies()


def print_expected_paths(movie_names: list[str], risk_filename: str, gold_labels_path: Path, output_path: Path, summary_path: Path) -> None:
    """Print expected input/output paths for this step."""
    print("Risk-gold overlap expected paths:")
    print(f"  gold labels: {gold_labels_path}")
    print(f"  output     : {output_path}")
    print(f"  summary    : {summary_path}")
    print("  movie inputs:")
    for movie_name in movie_names:
        print(f"    {movie_name}")
        print(f"      risk: {get_risk_prediction_path(movie_name, risk_filename=risk_filename)}")
        print(f"      re  : {get_relationship_extraction_path(movie_name)}")


def main() -> None:
    """Prepare the risk-gold overlap dataset."""
    args = parse_args()

    movie_names = get_movie_names(args.movie)
    gold_labels_path = resolve_project_path(args.gold_labels_path)
    output_path = resolve_project_path(args.output_path)
    summary_path = resolve_project_path(args.summary_path)

    if args.print_paths:
        print_expected_paths(
            movie_names=movie_names,
            risk_filename=args.risk_filename,
            gold_labels_path=gold_labels_path,
            output_path=output_path,
            summary_path=summary_path,
        )

    result = build_risk_gold_overlap(
        movie_names=movie_names,
        risk_filename=args.risk_filename,
        gold_labels_path=gold_labels_path,
        output_path=output_path,
        summary_path=summary_path,
    )
    print_overlap_summary(result)


if __name__ == "__main__":
    main()