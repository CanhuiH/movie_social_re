

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import load_json
from src.utils.paths import PROJECT_ROOT, project_path, slugify_movie_name


CONFIGS_DIR = PROJECT_ROOT / "configs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DATA_DIR = PROJECT_ROOT / "data"

SETTINGS_PATH = CONFIGS_DIR / "settings.json"
MOVIES_PATH = CONFIGS_DIR / "movies.json"
RELATIONSHIP_SCHEMA_PATH = CONFIGS_DIR / "relationship_schema.json"


DEFAULT_SETTINGS: dict[str, Any] = {
    "default_movie": "titanic",
    "default_context_window": 6,
    "default_re_model": "gpt-5.5",
    "openai_api_key_env": "OPENAI_API_KEY",
    "prompt_file": "prompts/relationship_extraction.txt",
    "movies_file": "configs/movies.json",
    "relationship_schema_file": "configs/relationship_schema.json",
    "data_dir": "data",
    "raw_dir": "data/raw",
    "interim_dir": "data/interim",
    "processed_dir": "data/processed",
    "llm": {
        "max_retries": 3,
        "retry_sleep_seconds": 2,
        "request_timeout_seconds": 120,
    },
    "outputs": {
        "dialogue_metadata": "dialogue_metadata.csv",
        "dialogue_metadata_clean": "dialogue_metadata_clean.csv",
        "turn_windows": "turn_windows.csv",
        "re_llm_outputs": "re_llm_outputs.jsonl",
        "re_clean": "re_clean.csv",
    },
}


def merge_dicts(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override values into default settings."""
    merged = defaults.copy()
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings() -> dict[str, Any]:
    """Load project settings from configs/settings.json with fallback defaults."""
    if SETTINGS_PATH.exists():
        settings = load_json(SETTINGS_PATH)
        if not isinstance(settings, dict):
            raise ValueError(f"Settings file must contain a JSON object: {SETTINGS_PATH}")
        return merge_dicts(DEFAULT_SETTINGS, settings)
    return DEFAULT_SETTINGS.copy()


SETTINGS = load_settings()

DEFAULT_MOVIE_NAME: str = str(SETTINGS["default_movie"])
DEFAULT_CONTEXT_WINDOW: int = int(SETTINGS["default_context_window"])
DEFAULT_RE_MODEL: str = str(SETTINGS["default_re_model"])
OPENAI_API_KEY_ENV: str = str(SETTINGS["openai_api_key_env"])

RAW_DATA_DIR: Path = project_path(SETTINGS["raw_dir"])
INTERIM_DATA_DIR: Path = project_path(SETTINGS["interim_dir"])
PROCESSED_DATA_DIR: Path = project_path(SETTINGS["processed_dir"])

RE_PROMPT_FILE: Path = project_path(SETTINGS["prompt_file"])
MOVIES_FILE: Path = project_path(SETTINGS["movies_file"])
RELATIONSHIP_SCHEMA_FILE: Path = project_path(SETTINGS["relationship_schema_file"])

LLM_MAX_RETRIES: int = int(SETTINGS.get("llm", {}).get("max_retries", 3))
LLM_RETRY_SLEEP_SECONDS: int = int(SETTINGS.get("llm", {}).get("retry_sleep_seconds", 2))
LLM_REQUEST_TIMEOUT_SECONDS: int = int(SETTINGS.get("llm", {}).get("request_timeout_seconds", 120))

OUTPUT_FILENAMES: dict[str, str] = {
    key: str(value)
    for key, value in SETTINGS.get("outputs", {}).items()
}


def get_output_filename(key: str) -> str:
    """Return a configured output filename by key."""
    if key not in OUTPUT_FILENAMES:
        raise KeyError(
            f"Unknown output filename key: {key}. "
            f"Available keys: {sorted(OUTPUT_FILENAMES)}"
        )
    return OUTPUT_FILENAMES[key]


def load_movies() -> list[str]:
    """Load the configured movie list from configs/movies.json."""
    movies_data = load_json(MOVIES_FILE)
    if not isinstance(movies_data, dict):
        raise ValueError(f"Movies file must contain a JSON object: {MOVIES_FILE}")

    movies = movies_data.get("movies")
    if not isinstance(movies, list):
        raise ValueError(f"Movies file must contain a list under key 'movies': {MOVIES_FILE}")

    cleaned_movies = []
    for movie in movies:
        if not isinstance(movie, str) or not movie.strip():
            raise ValueError(f"Invalid movie entry in {MOVIES_FILE}: {movie!r}")
        cleaned_movies.append(movie.strip())

    return cleaned_movies


def load_relationship_schema_config() -> dict[str, Any]:
    """Load the full relationship schema configuration."""
    schema_data = load_json(RELATIONSHIP_SCHEMA_FILE)
    if not isinstance(schema_data, dict):
        raise ValueError(
            f"Relationship schema file must contain a JSON object: {RELATIONSHIP_SCHEMA_FILE}"
        )
    return schema_data


def normalize_movie_key(movie_name: str) -> str:
    """Normalize a movie title for config lookup."""
    return movie_name.strip().lower()


def get_movie_schema(movie_name: str) -> dict[str, Any]:
    """Return the movie-specific relationship schema for a movie."""
    schema_config = load_relationship_schema_config()
    movie_schemas = schema_config.get("movie_schemas")
    if not isinstance(movie_schemas, dict):
        raise ValueError("relationship_schema.json must contain a 'movie_schemas' object.")

    movie_key = normalize_movie_key(movie_name)
    if movie_key not in movie_schemas:
        available = sorted(movie_schemas.keys())
        raise KeyError(
            f"No relationship schema found for movie '{movie_name}'. "
            f"Available movie schemas: {available}"
        )

    movie_schema = movie_schemas[movie_key]
    if not isinstance(movie_schema, dict):
        raise ValueError(f"Schema for movie '{movie_name}' must be a JSON object.")
    return movie_schema


def get_movie_relationship_labels(movie_name: str) -> list[str]:
    """Return the allowed relationship labels for a movie."""
    movie_schema = get_movie_schema(movie_name)
    labels = movie_schema.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"Movie schema for '{movie_name}' must contain a non-empty 'labels' list.")

    cleaned_labels = []
    for label in labels:
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Invalid relationship label for '{movie_name}': {label!r}")
        cleaned_labels.append(label.strip())
    return cleaned_labels


def get_movie_schema_focus(movie_name: str) -> str:
    """Return a short description of the relationship focus for a movie."""
    movie_schema = get_movie_schema(movie_name)
    focus = movie_schema.get("schema_focus", "")
    return str(focus).strip()


def get_movie_relationship_definitions(movie_name: str) -> dict[str, str]:
    """Return relationship label definitions for a movie."""
    movie_schema = get_movie_schema(movie_name)
    definitions = movie_schema.get("definitions")
    if not isinstance(definitions, dict):
        raise ValueError(f"Movie schema for '{movie_name}' must contain a 'definitions' object.")

    return {
        str(label): str(definition)
        for label, definition in definitions.items()
    }


def get_movie_priority_rules(movie_name: str) -> list[str]:
    """Return movie-specific priority rules for resolving ambiguous labels."""
    movie_schema = get_movie_schema(movie_name)
    priority_rules = movie_schema.get("priority_rules", [])
    if not isinstance(priority_rules, list):
        raise ValueError(f"Movie schema for '{movie_name}' has invalid 'priority_rules'.")
    return [str(rule) for rule in priority_rules]


def get_movie_global_category_mapping(movie_name: str) -> dict[str, str]:
    """Return mapping from movie-specific labels to broad global categories."""
    movie_schema = get_movie_schema(movie_name)
    mapping = movie_schema.get("global_category_mapping", {})
    if not isinstance(mapping, dict):
        raise ValueError(
            f"Movie schema for '{movie_name}' has invalid 'global_category_mapping'."
        )
    return {
        str(label): str(category)
        for label, category in mapping.items()
    }


def format_relationship_labels(movie_name: str) -> str:
    """Format allowed labels for prompt insertion."""
    labels = get_movie_relationship_labels(movie_name)
    return "\n".join(f"- {label}" for label in labels)


def format_relationship_definitions(movie_name: str) -> str:
    """Format movie-specific relationship definitions for prompt insertion."""
    labels = get_movie_relationship_labels(movie_name)
    definitions = get_movie_relationship_definitions(movie_name)

    lines = []
    for label in labels:
        definition = definitions.get(label, "No definition provided.")
        lines.append(f"- {label}: {definition}")
    return "\n".join(lines)


def format_priority_rules(movie_name: str) -> str:
    """Format movie-specific priority rules for prompt insertion."""
    rules = get_movie_priority_rules(movie_name)
    if not rules:
        return "- No additional movie-specific priority rules."
    return "\n".join(f"- {rule}" for rule in rules)


def ensure_project_dirs(movie_name: str | None = None) -> None:
    """Create core project directories, optionally including movie-specific folders."""
    dirs = [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR]

    if movie_name is not None:
        movie_slug = slugify_movie_name(movie_name)
        dirs.extend(
            [
                RAW_DATA_DIR / movie_slug,
                INTERIM_DATA_DIR / movie_slug,
                PROCESSED_DATA_DIR / movie_slug,
            ]
        )

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)