

"""CLI for merging ablation translation outputs.

This script merges five translation conditions:

1. context-only baseline
2. power/respect-only ablation
3. relationship-only ablation
4. social-label-guided ablation
5. full graph-guided translation
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.translation.merge_ablation_outputs import (
    DEFAULT_CONTEXT_ONLY_PATH,
    DEFAULT_GRAPH_GUIDED_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_POWER_RESPECT_ONLY_PATH,
    DEFAULT_RELATIONSHIP_ONLY_PATH,
    DEFAULT_SOCIAL_LABELS_ONLY_PATH,
    DEFAULT_SUMMARY_PATH,
    merge_ablation_outputs,
    print_ablation_merge_summary,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge context-only, power/respect-only, relationship-only, social-label-guided, and graph-guided translation outputs."
    )
    parser.add_argument(
        "--context-only-path",
        type=Path,
        default=DEFAULT_CONTEXT_ONLY_PATH,
        help="Context-only translation CSV path.",
    )
    parser.add_argument(
        "--power-respect-only-path",
        type=Path,
        default=DEFAULT_POWER_RESPECT_ONLY_PATH,
        help="Power/respect-only translation CSV path.",
    )
    parser.add_argument(
        "--relationship-only-path",
        type=Path,
        default=DEFAULT_RELATIONSHIP_ONLY_PATH,
        help="Relationship-only translation CSV path.",
    )
    parser.add_argument(
        "--social-labels-only-path",
        type=Path,
        default=DEFAULT_SOCIAL_LABELS_ONLY_PATH,
        help="Social-label-guided translation CSV path.",
    )
    parser.add_argument(
        "--graph-guided-path",
        type=Path,
        default=DEFAULT_GRAPH_GUIDED_PATH,
        help="Graph-guided translation CSV path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output ablation comparison CSV path.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Output ablation summary CSV path.",
    )
    parser.add_argument(
        "--include-debug-columns",
        action="store_true",
        help="Include prompt, raw response, status, and error columns in the merged output.",
    )
    parser.add_argument(
        "--print-paths",
        action="store_true",
        help="Print resolved paths without merging files.",
    )
    return parser.parse_args()


def main() -> None:
    """Run ablation merge."""
    args = parse_args()

    if args.print_paths:
        print("Ablation merge paths")
        print(f"  context-only path        : {args.context_only_path}")
        print(f"  power/respect-only path  : {args.power_respect_only_path}")
        print(f"  relationship-only path   : {args.relationship_only_path}")
        print(f"  social-label-guided path : {args.social_labels_only_path}")
        print(f"  graph-guided path        : {args.graph_guided_path}")
        print(f"  output path              : {args.output_path}")
        print(f"  summary path             : {args.summary_path}")
        return

    result = merge_ablation_outputs(
        context_only_path=args.context_only_path,
        power_respect_only_path=args.power_respect_only_path,
        relationship_only_path=args.relationship_only_path,
        social_labels_only_path=args.social_labels_only_path,
        graph_guided_path=args.graph_guided_path,
        output_path=args.output_path,
        summary_path=args.summary_path,
        include_debug_columns=args.include_debug_columns,
    )
    print_ablation_merge_summary(result)


if __name__ == "__main__":
    main()