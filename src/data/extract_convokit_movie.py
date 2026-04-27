"""
Extract dialogue metadata for a target movie from ConvoKit's movie-corpus.

This script is movie-agnostic: it accepts a movie name at runtime, extracts
all utterances for that movie, infers a listener from `reply_to` when
available, and saves movie-specific outputs under the interim data directory.

Outputs:
- data/interim/<Movie>/dialogue_metadata.csv
- data/interim/<Movie>/dialogue_metadata_with_listener.csv

Examples:
    python -m src.data.extract_convokit_movie
    python -m src.data.extract_convokit_movie --movie "Titanic"
    python -m src.data.extract_convokit_movie --movie "The Devil Wears Prada"
    python -m src.data.extract_convokit_movie --list-matches
"""

from __future__ import annotations

import argparse
from typing import Optional

import pandas as pd

from src.config import DEFAULT_MOVIE_NAME, ensure_project_dirs
from src.utils.io import print_file_summary, save_csv
from src.utils.paths import get_movie_dialogue_metadata_path, get_movie_interim_dir

try:
    from convokit import Corpus, download
except ImportError as exc:
    raise SystemExit(
        "convokit is not installed. Install it with:\n\n"
        "    pip install convokit pandas\n"
    ) from exc


LISTENER_ONLY_FILENAME = "dialogue_metadata_with_listener.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract dialogue metadata for a target movie from ConvoKit movie-corpus."
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=DEFAULT_MOVIE_NAME,
        help=(
            "Target movie name. Exact match is preferred, but if there is exactly "
            "one partial match it will be used automatically. "
            f'Default: "{DEFAULT_MOVIE_NAME}".'
        ),
    )
    parser.add_argument(
        "--list-matches",
        action="store_true",
        help="Only print movie names containing the keyword and exit.",
    )
    return parser.parse_args()


def safe_get_character_name(speaker: object) -> Optional[str]:
    """Return the best available character name from a ConvoKit speaker."""
    if speaker is None:
        return None

    return (
        speaker.meta.get("character_name")
        or speaker.meta.get("character")
        or speaker.meta.get("name")
    )


def load_movie_corpus() -> Corpus:
    """Download and load the ConvoKit movie corpus."""
    print("Downloading / loading ConvoKit movie-corpus...")
    corpus = Corpus(filename=download("movie-corpus"))
    print("Corpus loaded.")
    return corpus


def find_matching_movie_names(corpus: Corpus, keyword: str) -> list[str]:
    """Return all movie names containing the provided keyword."""
    movie_names: set[str] = set()
    keyword_lower = keyword.lower()

    for utt in corpus.iter_utterances():
        movie_name = utt.speaker.meta.get("movie_name")
        if isinstance(movie_name, str) and keyword_lower in movie_name.lower():
            movie_names.add(movie_name)

    return sorted(movie_names)


def resolve_target_movie_name(corpus: Corpus, requested_movie: str) -> str:
    """Resolve the requested movie name against corpus metadata."""
    matching_names = find_matching_movie_names(corpus, requested_movie)

    if matching_names:
        print("\nMovie names containing your keyword:")
        for name in matching_names:
            print(f"  - {name}")
    else:
        raise SystemExit(f'\nNo movie names found containing "{requested_movie}".')

    if requested_movie in matching_names:
        return requested_movie

    if len(matching_names) == 1:
        resolved_name = matching_names[0]
        print(
            f'\nExact match not found. Using the only partial match: "{resolved_name}"'
        )
        return resolved_name

    raise SystemExit(
        f'\nExact movie name "{requested_movie}" was not found.\n'
        "Please re-run with one of the printed movie names."
    )


def build_dataframe(corpus: Corpus, target_movie_name: str) -> pd.DataFrame:
    """Extract a movie-specific dialogue DataFrame from the corpus."""
    utterance_lookup = {utt.id: utt for utt in corpus.iter_utterances()}
    rows: list[dict[str, object]] = []

    for utt in corpus.iter_utterances():
        speaker = utt.speaker
        movie_name = speaker.meta.get("movie_name")
        movie_idx = speaker.meta.get("movie_idx")

        if movie_name != target_movie_name:
            continue

        speaker_id = speaker.id
        speaker_name = safe_get_character_name(speaker)
        reply_to = utt.reply_to

        listener_id = None
        listener_name = None
        if reply_to is not None and reply_to in utterance_lookup:
            parent_utt = utterance_lookup[reply_to]
            parent_speaker = parent_utt.speaker
            listener_id = parent_speaker.id
            listener_name = safe_get_character_name(parent_speaker)

        rows.append(
            {
                "movie_name": movie_name,
                "movie_idx": movie_idx,
                "conversation_id": utt.conversation_id,
                "utterance_id": utt.id,
                "timestamp": utt.timestamp,
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "reply_to": reply_to,
                "listener_id": listener_id,
                "listener_name": listener_name,
                "text": utt.text,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    sort_cols = [
        column
        for column in ["conversation_id", "timestamp", "utterance_id"]
        if column in df.columns
    ]
    return df.sort_values(sort_cols).reset_index(drop=True)


def save_movie_outputs(df: pd.DataFrame, movie_name: str) -> None:
    """Save extracted dialogue metadata to movie-specific interim files."""
    ensure_project_dirs(movie_name)
    interim_dir = get_movie_interim_dir(movie_name)

    full_output_path = get_movie_dialogue_metadata_path(movie_name)
    listener_only_output_path = interim_dir / LISTENER_ONLY_FILENAME

    save_csv(df, full_output_path, index=False)
    save_csv(df[df["listener_id"].notna()].copy(), listener_only_output_path, index=False)

    print("\nExtraction complete.")
    print_file_summary(full_output_path, label="Saved full dialogue metadata")
    print_file_summary(listener_only_output_path, label="Saved listener-only dialogue metadata")
    print(f"Rows in full CSV: {len(df)}")
    print(f"Rows with inferred listener: {df['listener_id'].notna().sum()}")


def main() -> None:
    args = parse_args()
    corpus = load_movie_corpus()

    matching_names = find_matching_movie_names(corpus, args.movie)
    if matching_names:
        print("\nMovie names containing your keyword:")
        for name in matching_names:
            print(f"  - {name}")
    else:
        print(f'\nNo movie names found containing "{args.movie}".')

    if args.list_matches:
        return

    target_movie_name = resolve_target_movie_name(corpus, args.movie)
    print(f'\nExtracting utterances for "{target_movie_name}"...')

    df = build_dataframe(corpus, target_movie_name)
    if df.empty:
        raise SystemExit(f'No utterances found for movie "{target_movie_name}".')

    save_movie_outputs(df, target_movie_name)


if __name__ == "__main__":
    main()
