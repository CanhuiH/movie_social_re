"""Command-line entry point for merging translation outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.translation.merge_translation_outputs import (  # noqa: E402
    DEFAULT_CONTEXT_ONLY_PATH,
    DEFAULT_GRAPH_GUIDED_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SUMMARY_PATH,
    merge_translation_outputs,
    print_merge_translation_outputs_result,
)
from src.utils.paths import project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge context-only and graph-guided translation outputs into one comparison CSV."
    )
    parser.add_argument(
        "--context-only-path",
        type=Path,
        default=DEFAULT_CONTEXT_ONLY_PATH,
        help="Input context-only translation CSV path.",
    )
    parser.add_argument(
        "--graph-guided-path",
        type=Path,
        default=DEFAULT_GRAPH_GUIDED_PATH,
        help="Input graph-guided translation CSV path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output merged translation comparison CSV path.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Output merged translation summary CSV path.",
    )
    parser.add_argument(
        "--include-debug-columns",
        action="store_true",
        help="Keep prompt/raw-response debug columns near the front of the output.",
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
    context_only_path: Path,
    graph_guided_path: Path,
    output_path: Path,
    summary_path: Path,
) -> None:
    """Print expected input/output paths for this step."""
    print("Translation output merge expected paths:")
    print(f"  context-only : {context_only_path}")
    print(f"  graph-guided : {graph_guided_path}")
    print(f"  output       : {output_path}")
    print(f"  summary      : {summary_path}")


def main() -> None:
    """Merge context-only and graph-guided translation outputs."""
    args = parse_args()

    context_only_path = resolve_project_path(args.context_only_path)
    graph_guided_path = resolve_project_path(args.graph_guided_path)
    output_path = resolve_project_path(args.output_path)
    summary_path = resolve_project_path(args.summary_path)

    if args.print_paths:
        print_expected_paths(
            context_only_path=context_only_path,
            graph_guided_path=graph_guided_path,
            output_path=output_path,
            summary_path=summary_path,
        )

    result = merge_translation_outputs(
        context_only_path=context_only_path,
        graph_guided_path=graph_guided_path,
        output_path=output_path,
        summary_path=summary_path,
        include_debug_columns=args.include_debug_columns,
    )
    print_merge_translation_outputs_result(result)


if __name__ == "__main__":
    main()
