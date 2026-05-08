"""Command-line entry point for preparing translation input."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.translation.prepare_translation_input import (  # noqa: E402
    DEFAULT_GRAPH_PATH,
    DEFAULT_OVERLAP_PATH,
    DEFAULT_SUMMARY_STATS_PATH,
    DEFAULT_TRANSLATION_INPUT_PATH,
    prepare_translation_input,
    print_translation_input_result,
)
from src.utils.paths import project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare translation_input.csv by merging risk/gold social annotations "
            "with aggregate and recent speaker-listener graph states."
        )
    )
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=DEFAULT_GRAPH_PATH,
        help="Input social graph edge-state CSV path.",
    )
    parser.add_argument(
        "--overlap-path",
        type=Path,
        default=DEFAULT_OVERLAP_PATH,
        help="Input risk-gold overlap CSV path.",
    )
    parser.add_argument(
        "--translation-input-path",
        type=Path,
        default=DEFAULT_TRANSLATION_INPUT_PATH,
        help="Output translation input CSV path.",
    )
    parser.add_argument(
        "--summary-stats-path",
        type=Path,
        default=DEFAULT_SUMMARY_STATS_PATH,
        help="Output translation input summary CSV path.",
    )
    parser.add_argument(
        "--print-paths",
        action="store_true",
        help="Print input/output paths before running.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """Resolve a path relative to the project root if needed."""
    if path.is_absolute():
        return path
    return project_path(path)


def print_expected_paths(
    *,
    graph_path: Path,
    overlap_path: Path,
    translation_input_path: Path,
    summary_stats_path: Path,
) -> None:
    """Print expected input/output paths for this step."""
    print("Translation input preparation expected paths:")
    print(f"  graph edges       : {graph_path}")
    print(f"  risk-gold overlap : {overlap_path}")
    print(f"  translation input : {translation_input_path}")
    print(f"  summary stats     : {summary_stats_path}")


def main() -> None:
    """Prepare translation input for context-only and graph-guided translation."""
    args = parse_args()

    graph_path = resolve_project_path(args.graph_path)
    overlap_path = resolve_project_path(args.overlap_path)
    translation_input_path = resolve_project_path(args.translation_input_path)
    summary_stats_path = resolve_project_path(args.summary_stats_path)

    if args.print_paths:
        print_expected_paths(
            graph_path=graph_path,
            overlap_path=overlap_path,
            translation_input_path=translation_input_path,
            summary_stats_path=summary_stats_path,
        )

    result = prepare_translation_input(
        graph_path=graph_path,
        overlap_path=overlap_path,
        translation_input_path=translation_input_path,
        summary_stats_path=summary_stats_path,
    )
    print_translation_input_result(result)


if __name__ == "__main__":
    main()