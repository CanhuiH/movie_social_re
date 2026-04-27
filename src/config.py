

from __future__ import annotations

import re
from pathlib import Path
from typing import Final


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
SRC_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = SRC_DIR.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
INTERIM_DATA_DIR: Final[Path] = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
OUTPUTS_DIR: Final[Path] = PROJECT_ROOT / "outputs"
PROMPTS_DIR: Final[Path] = PROJECT_ROOT / "prompts"
NOTEBOOKS_DIR: Final[Path] = PROJECT_ROOT / "notebooks"


# -----------------------------------------------------------------------------
# Movie / experiment defaults
# -----------------------------------------------------------------------------
DEFAULT_MOVIE_NAME: Final[str] = "Titanic"
DEFAULT_CONTEXT_WINDOW: Final[int] = 6
MAX_CONTEXT_WINDOW: Final[int] = 6


# -----------------------------------------------------------------------------
# Relationship extraction schema
# -----------------------------------------------------------------------------
RELATIONSHIP_LABELS: Final[list[str]] = [
    "familial",
    "romantic_intent",
    "formal_engagement",
    "friend",
    "acquaintance",
    "authority",
    "class_service",
    "adversarial",
    "ally",
    "unclear",
]

DEFAULT_RELATIONSHIP_LABEL: Final[str] = "unclear"


# -----------------------------------------------------------------------------
# LLM / prompt settings
# -----------------------------------------------------------------------------
DEFAULT_RE_MODEL: Final[str] = "gpt-5.5"
DEFAULT_TRANSLATION_MODEL: Final[str] = "gpt-5.4-mini"
LLM_TEMPERATURE: Final[float] = 0.0

RE_PROMPT_FILE: Final[Path] = PROMPTS_DIR / "relationship_extraction.txt"
TRANSLATION_PROMPT_FILE: Final[Path] = PROMPTS_DIR / "translation.txt"


# -----------------------------------------------------------------------------
# File naming conventions
# -----------------------------------------------------------------------------
CLEAN_DIALOGUE_FILENAME: Final[str] = "dialogue_metadata_clean.csv"
RAW_DIALOGUE_FILENAME: Final[str] = "dialogue_metadata.csv"
LISTENER_ONLY_DIALOGUE_FILENAME: Final[str] = "dialogue_metadata_with_listener.csv"
CLEAN_DIALOGUE_FILENAME: Final[str] = "dialogue_metadata_clean.csv"
TURN_WINDOWS_FILENAME: Final[str] = "turn_windows.csv"
RE_RAW_OUTPUT_FILENAME: Final[str] = "re_llm_outputs.jsonl"
RE_CLEAN_OUTPUT_FILENAME: Final[str] = "re_clean.csv"
RE_REVIEW_SHEET_FILENAME: Final[str] = "re_review_sheet.csv"
RE_REVIEWED_FILENAME: Final[str] = "re_reviewed.csv"
GRAPH_NODES_FILENAME: Final[str] = "graph_nodes.csv"
GRAPH_EDGES_FILENAME: Final[str] = "graph_edges.csv"
SOCIAL_SUMMARIES_FILENAME: Final[str] = "social_summaries.csv"
TRANSLATION_INPUTS_FILENAME: Final[str] = "translation_inputs.jsonl"


# -----------------------------------------------------------------------------
# Environment variables
# -----------------------------------------------------------------------------
OPENAI_API_KEY_ENV: Final[str] = "OPENAI_API_KEY"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def slugify_movie_name(movie_name: str) -> str:
    """Convert a movie name into a filesystem-friendly folder name."""
    slug = movie_name.strip()
    slug = re.sub(r"\s+", "_", slug)
    slug = re.sub(r"[^A-Za-z0-9_-]", "", slug)
    return slug


def get_movie_interim_dir(movie_name: str = DEFAULT_MOVIE_NAME) -> Path:
    """Return the interim data directory for a specific movie."""
    return INTERIM_DATA_DIR / slugify_movie_name(movie_name)


def get_movie_processed_dir(movie_name: str = DEFAULT_MOVIE_NAME) -> Path:
    """Return the processed data directory for a specific movie."""
    return PROCESSED_DATA_DIR / slugify_movie_name(movie_name)


def ensure_project_dirs(movie_name: str = DEFAULT_MOVIE_NAME) -> None:
    """Create the main project directories needed for a movie run."""
    directories = [
        RAW_DATA_DIR,
        get_movie_interim_dir(movie_name),
        get_movie_processed_dir(movie_name),
        OUTPUTS_DIR,
        PROMPTS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def validate_relationship_label(label: str) -> str:
    """Return a valid relationship label, defaulting to 'unclear'."""
    normalized = label.strip().lower()
    return normalized if normalized in RELATIONSHIP_LABELS else DEFAULT_RELATIONSHIP_LABEL