from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_MOVIE_NAME, DEFAULT_RE_MODEL, SETTINGS, load_movies
from src.re.generative_re import (
    DEFAULT_RE_PROVIDER,
    SUPPORTED_PROVIDERS,
    load_existing_success_records,
    load_llm_client,
    load_prompt_template,
    run_generative_re,
)
from src.re.postprocess import postprocess_re_outputs
from src.utils.io import load_csv, print_file_summary
from src.utils.paths import (
    get_movie_path_summary,
    get_movie_re_output_path,
    get_movie_turn_windows_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CSV-only relationship extraction for one movie or all configured movies. "
            "The script writes data/processed/<movie_slug>/re_clean.csv directly."
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
        "--provider",
        type=str,
        default=DEFAULT_RE_PROVIDER,
        choices=sorted(SUPPORTED_PROVIDERS),
        help=f'LLM provider to use. Default: "{DEFAULT_RE_PROVIDER}".',
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_RE_MODEL,
        help=f'LLM model name for generative RE. Default: "{DEFAULT_RE_MODEL}".',
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on the number of turn-window rows to process per movie.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional prompt template file path. Defaults to prompts/relationship_extraction.txt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing re_clean.csv output and rerun from scratch.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing re_clean.csv and skip rows with status == success.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save progress every N newly attempted rows. Default: 1.",
    )
    parser.add_argument(
        "--skip-re",
        action="store_true",
        help="Skip LLM relationship extraction and only normalize/postprocess existing re_clean.csv.",
    )
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Skip the final normalization/postprocess step after RE output is created.",
    )
    parser.add_argument(
        "--print-paths",
        action="store_true",
        help="Print key paths after each movie is processed.",
    )
    return parser.parse_args()


def get_target_movies(args: argparse.Namespace) -> list[str]:
    if args.all:
        return load_movies()
    return [args.movie or DEFAULT_MOVIE_NAME]


def validate_args(args: argparse.Namespace) -> None:
    if args.max_rows is not None and args.max_rows < 0:
        raise ValueError("--max-rows must be non-negative.")

    if args.save_every <= 0:
        raise ValueError("--save-every must be positive.")

    if args.overwrite and args.resume:
        raise ValueError("Use either --overwrite or --resume, not both.")

    if args.skip_re and args.skip_postprocess:
        raise ValueError("Both --skip-re and --skip-postprocess were set, so there is nothing to run.")


def normalize_movie_name(movie_name: str) -> str:
    return movie_name.strip().lower()


def print_stage_header(movie_name: str, index: int, total: int) -> None:
    print("\n" + "=" * 80)
    print(f"Relationship extraction pipeline for movie {index}/{total}: {movie_name}")
    print("=" * 80)


def print_paths(movie_name: str) -> None:
    print("\nKey paths:")
    for key, path in get_movie_path_summary(movie_name).items():
        print(f"- {key}: {path}")


def maybe_limit_rows(df, max_rows: int | None):
    if max_rows is None:
        return df
    return df.head(max_rows).copy()


def run_re_for_movie(
    *,
    movie_name: str,
    provider: str,
    model: str,
    prompt_template: str,
    overwrite: bool,
    resume: bool,
    max_rows: int | None,
    save_every: int,
    client,
) -> None:
    input_path = get_movie_turn_windows_path(movie_name)
    output_path = get_movie_re_output_path(movie_name)

    if output_path.exists() and not overwrite and not resume:
        raise FileExistsError(
            f"CSV RE output already exists: {output_path}. "
            "Use --resume to skip successful rows or --overwrite to rerun from scratch."
        )

    print(f"Loading turn-window data from: {input_path}")
    df = load_csv(input_path)
    print(f"Loaded {len(df)} turn-window rows.")

    df = maybe_limit_rows(df, max_rows)
    if max_rows is not None:
        print(f"Processing only the first {len(df)} rows due to --max-rows.")

    existing_success_records = {}
    if resume:
        existing_success_records = load_existing_success_records(
            output_path=output_path,
            default_movie=movie_name,
        )
        print(f"Resume mode: found {len(existing_success_records)} existing successful rows to skip.")

    outputs = run_generative_re(
        df=df,
        client=client,
        provider=provider,
        model=model,
        prompt_template=prompt_template,
        default_movie=movie_name,
        existing_success_records=existing_success_records,
        output_path=output_path,
        save_every=save_every,
    )

    print_file_summary(output_path, label="Saved CSV RE output")
    print(f"Rows in output: {len(outputs)}")


def run_pipeline_for_movie(
    *,
    movie_name: str,
    provider: str,
    model: str,
    prompt_template: str | None,
    overwrite: bool,
    resume: bool,
    max_rows: int | None,
    save_every: int,
    skip_re: bool,
    skip_postprocess: bool,
    client,
    print_output_paths: bool,
) -> None:
    movie = normalize_movie_name(movie_name)

    if not skip_re:
        print("\n[1/2] Running generative relationship extraction")
        if prompt_template is None:
            raise ValueError("prompt_template must be loaded when --skip-re is not set.")
        if client is None:
            raise ValueError("LLM client must be loaded when --skip-re is not set.")
        run_re_for_movie(
            movie_name=movie,
            provider=provider,
            model=model,
            prompt_template=prompt_template,
            overwrite=overwrite,
            resume=resume,
            max_rows=max_rows,
            save_every=save_every,
            client=client,
        )
    else:
        print("\n[1/2] Skipping generative relationship extraction")

    if not skip_postprocess:
        print("\n[2/2] Normalizing CSV relationship extraction outputs")
        postprocess_re_outputs(movie_name=movie)
    else:
        print("\n[2/2] Skipping postprocessing")

    if print_output_paths:
        print_paths(movie)


def main() -> None:
    args = parse_args()
    validate_args(args)

    movies = [normalize_movie_name(movie) for movie in get_target_movies(args)]
    print(f"Movies to process: {movies}")
    provider = args.provider.strip().lower()
    print(f"Provider: {provider}")
    print(f"Model: {args.model}")
    print(f"Max rows per movie: {args.max_rows}")
    print(f"Overwrite CSV RE output: {args.overwrite}")
    print(f"Resume CSV RE output: {args.resume}")
    print(f"Save every: {args.save_every}")
    print(f"Skip RE: {args.skip_re}")
    print(f"Skip postprocess: {args.skip_postprocess}")

    client = None
    prompt_template = None
    if not args.skip_re:
        client = load_llm_client(provider)
        prompt_path = Path(args.prompt_file) if args.prompt_file else PROJECT_ROOT / "prompts" / "relationship_extraction.txt"
        prompt_template = load_prompt_template(prompt_path)

    for index, movie_name in enumerate(movies, start=1):
        print_stage_header(movie_name, index, len(movies))
        run_pipeline_for_movie(
            movie_name=movie_name,
            provider=provider,
            model=args.model,
            prompt_template=prompt_template,
            overwrite=args.overwrite,
            resume=args.resume,
            max_rows=args.max_rows,
            save_every=args.save_every,
            skip_re=args.skip_re,
            skip_postprocess=args.skip_postprocess,
            client=client,
            print_output_paths=args.print_paths,
        )

    print("\nRelationship extraction pipeline complete.")


if __name__ == "__main__":
    main()