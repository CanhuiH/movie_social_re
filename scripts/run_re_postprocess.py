

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_MOVIE_NAME, ensure_project_dirs
from src.re.re_postprocess import load_and_normalize_re_jsonl, save_re_outputs
from src.utils.io import print_file_summary
from src.utils.paths import (
    get_movie_re_clean_output_path,
    get_movie_re_raw_output_path,
    get_movie_re_review_sheet_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize raw relationship extraction outputs and create a manual review sheet."
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=DEFAULT_MOVIE_NAME,
        help=f'Movie name used to locate movie-specific files. Default: "{DEFAULT_MOVIE_NAME}".',
    )
    parser.add_argument(
        "--skip-review-sheet",
        action="store_true",
        help="If set, only save the cleaned RE CSV and skip the manual review sheet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs(args.movie)

    raw_input_path = get_movie_re_raw_output_path(args.movie)
    clean_output_path = get_movie_re_clean_output_path(args.movie)
    review_sheet_path = None if args.skip_review_sheet else get_movie_re_review_sheet_path(args.movie)

    print(f"Loading raw RE outputs from: {raw_input_path}")
    normalized_df = load_and_normalize_re_jsonl(raw_input_path)
    print(f"Loaded and normalized {len(normalized_df)} RE rows.")

    save_re_outputs(
        df=normalized_df,
        clean_output_path=clean_output_path,
        review_sheet_path=review_sheet_path,
    )

    print("\nRE postprocessing complete.")
    print_file_summary(clean_output_path, label="Saved clean RE outputs")
    if review_sheet_path is not None:
        print_file_summary(review_sheet_path, label="Saved RE review sheet")


if __name__ == "__main__":
    main()