
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
from convokit import Corpus, download

from src.config import DEFAULT_MOVIE_NAME, ensure_project_dirs
from src.utils.io import load_csv, print_file_summary, require_columns, save_csv
from src.utils.paths import (
    get_movie_dialogue_metadata_path,
    get_movie_raw_dialogue_metadata_path,
)


MOVIE_CORPUS_NAME = "movie-corpus"
UNKNOWN_SPEAKER = "UNKNOWN"

# Match the manually prepared dialogue_metadata.csv format.
# The timestamp column is kept but intentionally left empty.
# movie_name,movie_idx,conversation_id,utterance_id,timestamp,
# speaker_id,speaker_name,reply_to,listener_id,listener_name,text
RAW_DIALOGUE_COLUMNS = [
    "movie_name",
    "movie_idx",
    "conversation_id",
    "utterance_id",
    "timestamp",
    "speaker_id",
    "speaker_name",
    "reply_to",
    "listener_id",
    "listener_name",
    "text",
]

REQUIRED_RAW_DIALOGUE_COLUMNS = set(RAW_DIALOGUE_COLUMNS)


@dataclass(frozen=True)
class DialogueRow:
    movie_name: str
    movie_idx: str | None
    conversation_id: str
    utterance_id: str
    timestamp: str | None
    speaker_id: str
    speaker_name: str
    reply_to: str | None
    listener_id: str | None
    listener_name: str | None
    text: str


def normalize_text(value: object) -> str:
    """Normalize plain text values from ConvoKit metadata."""
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def normalize_optional_text(value: object) -> str | None:
    """Normalize optional text values and preserve missing values as None."""
    text = normalize_text(value)
    if not text or text.lower() in {"none", "null", "nan", "unknown"}:
        return None
    return text


def normalize_required_text(value: object, fallback: str = "UNKNOWN") -> str:
    """Normalize required text values and use a fallback if missing."""
    text = normalize_optional_text(value)
    return text if text is not None else fallback


def normalize_movie_title(value: object) -> str:
    """Normalize a movie title for matching."""
    return normalize_text(value).lower()


def safe_meta(meta: object) -> dict[str, Any]:
    """Return metadata as a dictionary."""
    return meta if isinstance(meta, dict) else {}


def get_meta_value(meta: dict[str, Any], keys: Iterable[str]) -> Any | None:
    """Return the first non-empty metadata value for the given possible keys."""
    for key in keys:
        value = meta.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def get_conversation_movie_name(conversation: Any) -> str:
    """Extract a movie title from a ConvoKit conversation."""
    meta = safe_meta(getattr(conversation, "meta", None))
    value = get_meta_value(
        meta,
        [
            "movie_name",
            "movie_title",
            "movie",
            "title",
            "film_name",
            "film_title",
        ],
    )
    return normalize_text(value)


def get_conversation_movie_idx(conversation: Any) -> str | None:
    """Extract a movie index from conversation metadata if available."""
    meta = safe_meta(getattr(conversation, "meta", None))
    value = get_meta_value(
        meta,
        [
            "movie_idx",
            "movie_id",
            "film_idx",
            "film_id",
        ],
    )
    return normalize_optional_text(value)


def get_utterance_movie_name(utterance: Any) -> str:
    """Extract a movie title from a ConvoKit utterance if present."""
    meta = safe_meta(getattr(utterance, "meta", None))
    value = get_meta_value(
        meta,
        [
            "movie_name",
            "movie_title",
            "movie",
            "title",
            "film_name",
            "film_title",
        ],
    )
    return normalize_text(value)


def get_utterance_movie_idx(utterance: Any) -> str | None:
    """Extract movie index from utterance metadata if available."""
    meta = safe_meta(getattr(utterance, "meta", None))
    value = get_meta_value(
        meta,
        [
            "movie_idx",
            "movie_id",
            "film_idx",
            "film_id",
        ],
    )
    return normalize_optional_text(value)


def get_speaker_id(utterance: Any) -> str:
    speaker = getattr(utterance, "speaker", None)
    speaker_id = getattr(speaker, "id", None)
    if speaker_id is None:
        speaker_id = getattr(utterance, "speaker_id", None)
    return normalize_text(speaker_id) or UNKNOWN_SPEAKER


def get_speaker_name(corpus: Corpus, speaker_id: str | None) -> str | None:
    """Return speaker name if available; otherwise use speaker id."""
    if not speaker_id:
        return None

    try:
        speaker = corpus.get_speaker(speaker_id)
    except Exception:
        return speaker_id or UNKNOWN_SPEAKER

    speaker_meta = safe_meta(getattr(speaker, "meta", None))
    name = get_meta_value(
        speaker_meta,
        [
            "character_name",
            "name",
            "speaker_name",
            "movie_character_name",
        ],
    )
    normalized_name = normalize_text(name)
    return normalized_name or speaker_id or UNKNOWN_SPEAKER


def infer_listener_id_from_reply_to(corpus: Corpus, utterance: Any) -> str | None:
    """Infer listener as the speaker of the utterance being replied to."""
    reply_to = getattr(utterance, "reply_to", None)
    if reply_to is None:
        return None

    try:
        parent_utterance = corpus.get_utterance(reply_to)
    except Exception:
        return None

    listener_id = get_speaker_id(parent_utterance)
    return listener_id if listener_id and listener_id != UNKNOWN_SPEAKER else None


def infer_listener_from_previous_turn(
    utterances: list[Any],
    current_index: int,
) -> str | None:
    """Infer listener as the previous different speaker in the same conversation."""
    current_speaker_id = get_speaker_id(utterances[current_index])

    for previous_index in range(current_index - 1, -1, -1):
        previous_speaker_id = get_speaker_id(utterances[previous_index])
        if (
            previous_speaker_id
            and previous_speaker_id != UNKNOWN_SPEAKER
            and previous_speaker_id != current_speaker_id
        ):
            return previous_speaker_id
    return None


def infer_listener_id(
    corpus: Corpus,
    utterances: list[Any],
    current_index: int,
) -> str | None:
    """Infer listener using reply_to first and previous-speaker fallback second."""
    utterance = utterances[current_index]
    reply_to_listener = infer_listener_id_from_reply_to(corpus, utterance)
    if reply_to_listener is not None:
        return reply_to_listener
    return infer_listener_from_previous_turn(utterances, current_index)


def get_conversation_utterances(corpus: Corpus, conversation: Any) -> list[Any]:
    """Return utterances for one conversation sorted in dialogue order."""
    try:
        utterance_ids = list(conversation.get_utterance_ids())
        utterances = [corpus.get_utterance(utterance_id) for utterance_id in utterance_ids]
    except Exception:
        utterances = [
            utterance
            for utterance in corpus.iter_utterances()
            if utterance.conversation_id == conversation.id
        ]

    return sorted(
        utterances,
        key=lambda utt: (
            getattr(utt, "timestamp", None) is None,
            getattr(utt, "timestamp", 0) or 0,
            str(getattr(utt, "id", "")),
        ),
    )


def iter_matching_conversations(corpus: Corpus, movie_name: str) -> list[Any]:
    """Return conversations whose movie title matches the requested movie."""
    target_title = normalize_movie_title(movie_name)
    matches = []

    for conversation in corpus.iter_conversations():
        conversation_movie = get_conversation_movie_name(conversation)
        if normalize_movie_title(conversation_movie) == target_title:
            matches.append(conversation)

    if matches:
        return matches

    # Fallback for corpora where movie title lives only on utterances.
    matching_conversation_ids: set[str] = set()
    for utterance in corpus.iter_utterances():
        utterance_movie = get_utterance_movie_name(utterance)
        if normalize_movie_title(utterance_movie) == target_title:
            conversation_id = normalize_text(getattr(utterance, "conversation_id", ""))
            if conversation_id:
                matching_conversation_ids.add(conversation_id)

    return [
        corpus.get_conversation(conversation_id)
        for conversation_id in sorted(matching_conversation_ids)
    ]


def list_available_movie_titles(corpus: Corpus) -> list[str]:
    """List available movie titles from conversation metadata with utterance fallback."""
    titles: set[str] = set()
    for conversation in corpus.iter_conversations():
        title = get_conversation_movie_name(conversation)
        if title:
            titles.add(title)

    if titles:
        return sorted(titles)

    for utterance in corpus.iter_utterances():
        title = get_utterance_movie_name(utterance)
        if title:
            titles.add(title)
    return sorted(titles)


def build_dialogue_rows_for_conversation(
    corpus: Corpus,
    conversation: Any,
    movie_name: str,
) -> list[DialogueRow]:
    """Build dialogue rows for one matched conversation."""
    conversation_id = normalize_text(getattr(conversation, "id", ""))
    conversation_movie_idx = get_conversation_movie_idx(conversation)
    utterances = get_conversation_utterances(corpus, conversation)
    rows: list[DialogueRow] = []

    for utterance in utterances:
        text = normalize_text(getattr(utterance, "text", ""))
        if not text:
            continue

        speaker_id = get_speaker_id(utterance)
        speaker_name = get_speaker_name(corpus, speaker_id) or UNKNOWN_SPEAKER
        reply_to = normalize_optional_text(getattr(utterance, "reply_to", None))
        listener_id = infer_listener_id_from_reply_to(corpus, utterance)
        listener_name = get_speaker_name(corpus, listener_id) if listener_id else None
        movie_idx = conversation_movie_idx or get_utterance_movie_idx(utterance)

        rows.append(
            DialogueRow(
                movie_name=movie_name,
                movie_idx=movie_idx,
                conversation_id=conversation_id,
                utterance_id=normalize_text(getattr(utterance, "id", "")),
                timestamp=None,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                reply_to=reply_to,
                listener_id=listener_id,
                listener_name=listener_name,
                text=text,
            )
        )

    return rows


def extract_movie_dialogue_dataframe(
    movie_name: str = DEFAULT_MOVIE_NAME,
    corpus: Corpus | None = None,
) -> pd.DataFrame:
    """Extract dialogue metadata for one movie from the ConvoKit movie corpus."""
    corpus = corpus or Corpus(filename=download(MOVIE_CORPUS_NAME))
    conversations = iter_matching_conversations(corpus, movie_name)

    if not conversations:
        available_titles = list_available_movie_titles(corpus)
        sample_titles = available_titles[:30]
        raise ValueError(
            f"No conversations found for movie '{movie_name}'. "
            f"Found {len(available_titles)} available movie titles. "
            f"Sample titles: {sample_titles}"
        )

    rows: list[DialogueRow] = []
    for conversation in conversations:
        rows.extend(build_dialogue_rows_for_conversation(corpus, conversation, movie_name))

    if not rows:
        raise ValueError(f"Matched movie '{movie_name}' but extracted 0 non-empty dialogue rows.")

    df = pd.DataFrame([row.__dict__ for row in rows])
    df["_row_order"] = range(len(df))
    df = df.sort_values(
        ["conversation_id", "_row_order", "utterance_id"],
        kind="mergesort",
    ).drop(columns=["_row_order"]).reset_index(drop=True)
    df["timestamp"] = None
    return df[RAW_DIALOGUE_COLUMNS].copy()


def standardize_raw_dialogue_dataframe(
    df: pd.DataFrame,
    movie_name: str = DEFAULT_MOVIE_NAME,
) -> pd.DataFrame:
    """Standardize a raw dialogue dataframe into the expected 11-column format."""
    require_columns(
        df,
        REQUIRED_RAW_DIALOGUE_COLUMNS,
        label="Raw dialogue metadata",
    )

    prepared = df.copy()
    prepared["movie_name"] = prepared["movie_name"].apply(
        lambda value: normalize_required_text(value, fallback=movie_name)
    )
    prepared["movie_idx"] = prepared["movie_idx"].apply(normalize_optional_text)
    prepared["conversation_id"] = prepared["conversation_id"].apply(
        lambda value: normalize_required_text(value)
    )
    prepared["utterance_id"] = prepared["utterance_id"].apply(
        lambda value: normalize_required_text(value)
    )
    prepared["speaker_id"] = prepared["speaker_id"].apply(
        lambda value: normalize_required_text(value)
    )
    prepared["speaker_name"] = prepared["speaker_name"].apply(
        lambda value: normalize_required_text(value)
    )
    prepared["reply_to"] = prepared["reply_to"].apply(normalize_optional_text)
    prepared["listener_id"] = prepared["listener_id"].apply(normalize_optional_text)
    prepared["listener_name"] = prepared["listener_name"].apply(normalize_optional_text)
    prepared["text"] = prepared["text"].apply(normalize_text)

    # Keep timestamp intentionally empty in the raw/interim dialogue metadata.
    # Dialogue order is preserved by the original row order and later converted
    # into turn_index / utterance_order during preprocessing.
    prepared["_row_order"] = range(len(prepared))
    prepared["timestamp"] = None

    prepared = prepared[
        (prepared["conversation_id"] != "UNKNOWN")
        & (prepared["utterance_id"] != "UNKNOWN")
        & (prepared["speaker_id"] != "UNKNOWN")
        & (prepared["text"] != "")
    ].copy()

    prepared = prepared.sort_values(
        ["conversation_id", "_row_order", "utterance_id"],
        kind="mergesort",
    ).drop(columns=["_row_order"]).reset_index(drop=True)

    return prepared[RAW_DIALOGUE_COLUMNS].copy()


def load_user_raw_dialogue_dataframe(movie_name: str = DEFAULT_MOVIE_NAME) -> pd.DataFrame:
    """Load a user-provided raw dialogue CSV from data/raw/<movie>/.

    This is useful when a manually prepared dialogue file already exists and
    should be used instead of extracting from ConvoKit.
    """
    raw_path = get_movie_raw_dialogue_metadata_path(movie_name)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dialogue metadata file not found: {raw_path}. "
            "Place the file at data/raw/<movie_slug>/dialogue_metadata.csv "
            "or run without --use-raw-file to extract from ConvoKit."
        )

    print(f"Loading user-provided raw dialogue from: {raw_path}")
    df = load_csv(raw_path)
    return standardize_raw_dialogue_dataframe(df, movie_name=movie_name)


def save_dialogue_to_raw_and_interim(
    df: pd.DataFrame,
    movie_name: str = DEFAULT_MOVIE_NAME,
) -> tuple[pd.DataFrame, Any, Any]:
    """Save standardized dialogue metadata to both raw and interim folders."""
    standardized_df = standardize_raw_dialogue_dataframe(df, movie_name=movie_name)

    raw_output_path = get_movie_raw_dialogue_metadata_path(movie_name)
    interim_output_path = get_movie_dialogue_metadata_path(movie_name)

    save_csv(standardized_df, raw_output_path, index=False)
    save_csv(standardized_df, interim_output_path, index=False)

    return standardized_df, raw_output_path, interim_output_path


def summarize_dialogue_dataframe(df: pd.DataFrame) -> dict[str, int]:
    """Return simple extraction summary statistics."""
    return {
        "rows": len(df),
        "conversations": int(df["conversation_id"].nunique()) if "conversation_id" in df else 0,
        "speakers": int(df["speaker_id"].nunique()) if "speaker_id" in df else 0,
        "missing_listener_rows": int(df["listener_id"].isna().sum()) if "listener_id" in df else 0,
    }


def print_dialogue_summary(df: pd.DataFrame, movie_name: str) -> None:
    """Print a compact extraction summary."""
    summary = summarize_dialogue_dataframe(df)
    print(f"\nDialogue metadata for movie: {movie_name}")
    print(f"Rows: {summary['rows']}")
    print(f"Conversations: {summary['conversations']}")
    print(f"Speakers: {summary['speakers']}")
    print(f"Rows with missing listener: {summary['missing_listener_rows']}")

    if "speaker_name" in df.columns:
        top_speakers = Counter(df["speaker_name"].dropna()).most_common(10)
        if top_speakers:
            print("Top speakers:")
            for speaker, count in top_speakers:
                print(f"- {speaker}: {count}")


def extract_and_save_movie_dialogue(
    movie_name: str = DEFAULT_MOVIE_NAME,
    use_raw_file: bool = False,
) -> pd.DataFrame:
    """Extract or load dialogue metadata and save it to raw + interim folders.

    If use_raw_file is True, this loads data/raw/<movie_slug>/dialogue_metadata.csv,
    standardizes it, and refreshes the interim working copy.

    If use_raw_file is False, this extracts from ConvoKit, then saves the same
    standardized dialogue metadata to both:
    - data/raw/<movie_slug>/dialogue_metadata.csv
    - data/interim/<movie_slug>/dialogue_metadata.csv
    """
    ensure_project_dirs(movie_name)

    if use_raw_file:
        df = load_user_raw_dialogue_dataframe(movie_name=movie_name)
        source_label = "user-provided raw dialogue"
    else:
        print(f"Loading ConvoKit corpus: {MOVIE_CORPUS_NAME}")
        df = extract_movie_dialogue_dataframe(movie_name=movie_name)
        source_label = "ConvoKit extraction"

    df, raw_output_path, interim_output_path = save_dialogue_to_raw_and_interim(
        df,
        movie_name=movie_name,
    )

    print_dialogue_summary(df, movie_name)
    print_file_summary(raw_output_path, label=f"Saved raw dialogue metadata from {source_label}")
    print_file_summary(interim_output_path, label="Saved interim dialogue metadata")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract movie dialogue metadata from the ConvoKit movie corpus."
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=DEFAULT_MOVIE_NAME,
        help=f'Movie title to extract. Default: "{DEFAULT_MOVIE_NAME}".',
    )
    parser.add_argument(
        "--use-raw-file",
        action="store_true",
        help=(
            "Load data/raw/<movie_slug>/dialogue_metadata.csv instead of extracting "
            "dialogue from ConvoKit."
        ),
    )
    parser.add_argument(
        "--list-movies",
        action="store_true",
        help="List available movie titles and exit.",
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=50,
        help="Number of movie titles to print with --list-movies. Default: 50.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_movies:
        corpus = Corpus(filename=download(MOVIE_CORPUS_NAME))
        titles = list_available_movie_titles(corpus)
        print(f"Found {len(titles)} movie titles.")
        for title in titles[: args.list_limit]:
            print(f"- {title}")
        return

    extract_and_save_movie_dialogue(
        movie_name=args.movie,
        use_raw_file=args.use_raw_file,
    )


if __name__ == "__main__":
    main()