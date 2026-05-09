

"""Command-line entry point for building dynamic social graph edges."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.build_graph import (  # noqa: E402
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_RECENT_WINDOW,
    DEFAULT_SUMMARY_PATH,
    build_social_graph_edges,
    print_graph_build_summary,
)
from src.utils.paths import project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build dynamic speaker-listener social graph edges from risk-gold overlap rows."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input risk-gold overlap CSV path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output social graph edge-state CSV path.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Output graph summary statistics CSV path.",
    )
    parser.add_argument(
        "--recent-window",
        type=int,
        default=DEFAULT_RECENT_WINDOW,
        help="Number of most recent speaker-listener observations used for recent graph state.",
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
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    recent_window: int,
) -> None:
    """Print expected input/output paths for this step."""
    print("Social graph build expected paths:")
    print(f"  input        : {input_path}")
    print(f"  output       : {output_path}")
    print(f"  summary      : {summary_path}")
    print(f"  recent window: {recent_window}")


def main() -> None:
    """Build dynamic speaker-listener social graph edges."""
    args = parse_args()

    input_path = resolve_project_path(args.input_path)
    output_path = resolve_project_path(args.output_path)
    summary_path = resolve_project_path(args.summary_path)

    if args.print_paths:
        print_expected_paths(
            input_path=input_path,
            output_path=output_path,
            summary_path=summary_path,
            recent_window=args.recent_window,
        )

    result = build_social_graph_edges(
        input_path=input_path,
        output_path=output_path,
        summary_path=summary_path,
        recent_window=args.recent_window,
    )
    print_graph_build_summary(result)


if __name__ == "__main__":
    main()