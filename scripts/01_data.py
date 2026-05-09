from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_CONTEXT_WINDOW, DEFAULT_MOVIE_NAME, load_movies
from src.data.build_turn_windows import build_and_save_turn_windows
from src.data.extract_dialogue import extract_and_save_movie_dialogue
from src.data.preprocess_dialogue import preprocess_and_save_dialogue
from src.utils.paths import get_movie_path_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the data preparation stage for relationship extraction: "
            "extract dialogue, preprocess dialogue, and build turn windows."
        )
    )
    movie_group = parser.add_mutually_exclusive_group()
    movie_group.add_argument(
        "--movie",
        type=str,
        default=None,
        help=f'Movie title to process. Default: "{DEFAULT_MOVIE_NAME}" unless --all is set.',
    )
    movie_group.add_argument(
        "--all",
        action="store_true",
        help="Process all movies listed in configs/movies.json.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        help=(
            "Number of previous turns to include as local context. "
            f"Default: {DEFAULT_CONTEXT_WINDOW}."
        ),
    )
    parser.add_argument(
        "--drop-missing-listener",
        action="store_true",
        help=(
            "If set, drop rows with missing listener_id during preprocessing and "
            "only keep turn-window targets with a known listener."
        ),
    )
    parser.add_argument(
        "--use-raw-file",
        action="store_true",
        help=(
            "Load data/raw/<movie_slug>/dialogue_metadata.csv instead of "
            "extracting dialogue from ConvoKit. This option only affects the "
            "extract stage."
        ),
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip dialogue extraction and reuse existing dialogue_metadata.csv.",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing and reuse existing dialogue_metadata_clean.csv.",
    )
    parser.add_argument(
        "--skip-windows",
        action="store_true",
        help="Skip turn-window construction.",
    )
    parser.add_argument(
        "--print-paths",
        action="store_true",
        help="Print key output paths after each movie is processed.",
    )
    return parser.parse_args()


def get_target_movies(args: argparse.Namespace) -> list[str]:
    if args.all:
        return load_movies()
    return [args.movie or DEFAULT_MOVIE_NAME]


def validate_args(args: argparse.Namespace) -> None:
    if args.context_window < 0:
        raise ValueError("--context-window must be non-negative.")

    if args.skip_extract and args.skip_preprocess and args.skip_windows:
        raise ValueError(
            "All stages were skipped. Remove at least one of --skip-extract, "
            "--skip-preprocess, or --skip-windows."
        )

    if args.use_raw_file and args.skip_extract:
        raise ValueError(
            "--use-raw-file cannot be combined with --skip-extract because "
            "the raw file is loaded during the extract stage."
        )


def print_stage_header(movie_name: str, index: int, total: int) -> None:
    print("\n" + "=" * 80)
    print(f"Data pipeline for movie {index}/{total}: {movie_name}")
    print("=" * 80)


def print_paths(movie_name: str) -> None:
    print("\nKey paths:")
    for key, path in get_movie_path_summary(movie_name).items():
        print(f"- {key}: {path}")


def run_data_pipeline_for_movie(
    movie_name: str,
    context_window: int,
    drop_missing_listener: bool = False,
    use_raw_file: bool = False,
    skip_extract: bool = False,
    skip_preprocess: bool = False,
    skip_windows: bool = False,
    print_output_paths: bool = False,
) -> None:
    if not skip_extract:
        if use_raw_file:
            print("\n[1/3] Loading user-provided raw dialogue file")
        else:
            print("\n[1/3] Extracting dialogue from ConvoKit")
        extract_and_save_movie_dialogue(
            movie_name=movie_name,
            use_raw_file=use_raw_file,
        )
    else:
        print("\n[1/3] Skipping dialogue extraction")

    if not skip_preprocess:
        print("\n[2/3] Preprocessing dialogue")
        preprocess_and_save_dialogue(
            movie_name=movie_name,
            drop_missing_listener=drop_missing_listener,
        )
    else:
        print("\n[2/3] Skipping dialogue preprocessing")

    if not skip_windows:
        print("\n[3/3] Building turn windows")
        build_and_save_turn_windows(
            movie_name=movie_name,
            context_window=context_window,
            drop_missing_listener=drop_missing_listener,
        )
    else:
        print("\n[3/3] Skipping turn-window construction")

    if print_output_paths:
        print_paths(movie_name)


def main() -> None:
    args = parse_args()
    validate_args(args)

    movies = get_target_movies(args)
    print(f"Movies to process: {movies}")
    print(f"Context window: {args.context_window}")
    print(f"Use raw file: {args.use_raw_file}")

    for index, movie_name in enumerate(movies, start=1):
        print_stage_header(movie_name, index, len(movies))
        run_data_pipeline_for_movie(
            movie_name=movie_name,
            context_window=args.context_window,
            drop_missing_listener=args.drop_missing_listener,
            use_raw_file=args.use_raw_file,
            skip_extract=args.skip_extract,
            skip_preprocess=args.skip_preprocess,
            skip_windows=args.skip_windows,
            print_output_paths=args.print_paths,
        )

    print("\nData preparation pipeline complete.")


if __name__ == "__main__":
    main()