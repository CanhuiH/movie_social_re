

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_MOVIE_NAME
from src.graph.build_graph import main as build_graph_main
from src.graph.summarize_relations import main as summarize_relations_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run graph construction and social summary generation for a movie."
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=DEFAULT_MOVIE_NAME,
        help=f'Movie name used to locate movie-specific files. Default: "{DEFAULT_MOVIE_NAME}".',
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Optional minimum confidence required to keep an edge update. Default: 0.0.",
    )
    parser.add_argument(
        "--drop-unclear",
        action="store_true",
        help="If set, skip rows whose relationship_type is 'unclear' during graph construction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    original_argv = sys.argv.copy()
    try:
        sys.argv = [
            "build_graph",
            "--movie",
            args.movie,
            "--confidence-threshold",
            str(args.confidence_threshold),
        ]
        if args.drop_unclear:
            sys.argv.append("--drop-unclear")
        build_graph_main()

        sys.argv = [
            "summarize_relations",
            "--movie",
            args.movie,
        ]
        summarize_relations_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()