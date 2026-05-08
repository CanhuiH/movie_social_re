from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.utils.io import load_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
SETTINGS_PATH = CONFIGS_DIR / "settings.json"


DEFAULT_SETTINGS: dict[str, Any] = {
    "default_movie": "titanic",
    "data_dir": "data",
    "raw_dir": "data/raw",
    "interim_dir": "data/interim",
    "processed_dir": "data/processed",
    "labeled_dir": "data/labeled",
    "modeling_dir": "data/modeling",
    "models_dir": "models",
    "embeddings_dir": "models/embeddings",
    "outputs_dir": "outputs",
    "modeling_outputs_dir": "outputs/modeling",
    "outputs": {
        "dialogue_metadata": "dialogue_metadata.csv",
        "dialogue_metadata_clean": "dialogue_metadata_clean.csv",
        "turn_windows": "turn_windows.csv",
        "re_output": "re_clean.csv",
        "re_clean": "re_clean.csv",
    },
}


def load_settings() -> dict[str, Any]:
    """Load project settings with safe defaults.

    The project is configured through `configs/settings.json`, but keeping
    defaults here makes path utilities robust during early setup or testing.
    """
    if SETTINGS_PATH.exists():
        settings = load_json(SETTINGS_PATH)
        return merge_settings(DEFAULT_SETTINGS, settings)
    return DEFAULT_SETTINGS.copy()


def merge_settings(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge user settings into defaults."""
    merged = defaults.copy()
    for key, value in overrides.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = merge_settings(merged[key], value)
        else:
            merged[key] = value
    return merged


def project_path(relative_path: str | Path) -> Path:
    """Return an absolute path under the project root."""
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def slugify_movie_name(movie_name: str) -> str:
    """Convert a movie title into a safe directory name.

    Examples:
        "The Godfather" -> "the_godfather"
        "an officer and a gentleman" -> "an_officer_and_a_gentleman"
    """
    cleaned = movie_name.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown_movie"


def get_default_movie_name() -> str:
    settings = load_settings()
    return str(settings["default_movie"])


def get_data_dir() -> Path:
    settings = load_settings()
    return project_path(settings["data_dir"])


def get_raw_data_dir() -> Path:
    settings = load_settings()
    return project_path(settings["raw_dir"])


def get_interim_data_dir() -> Path:
    settings = load_settings()
    return project_path(settings["interim_dir"])


def get_processed_data_dir() -> Path:
    settings = load_settings()
    return project_path(settings["processed_dir"])


def get_labeled_data_dir() -> Path:
    settings = load_settings()
    return project_path(settings.get("labeled_dir", "data/labeled"))


def get_modeling_data_dir() -> Path:
    settings = load_settings()
    return project_path(settings.get("modeling_dir", "data/modeling"))


def get_models_dir() -> Path:
    settings = load_settings()
    return project_path(settings.get("models_dir", "models"))


def get_embeddings_dir() -> Path:
    settings = load_settings()
    return project_path(settings.get("embeddings_dir", "models/embeddings"))


def get_outputs_dir() -> Path:
    settings = load_settings()
    return project_path(settings.get("outputs_dir", "outputs"))


def get_modeling_outputs_dir() -> Path:
    settings = load_settings()
    return project_path(settings.get("modeling_outputs_dir", "outputs/modeling"))


def get_movie_raw_dir(movie_name: str | None = None) -> Path:
    movie = movie_name or get_default_movie_name()
    return get_raw_data_dir() / slugify_movie_name(movie)


def get_movie_interim_dir(movie_name: str | None = None) -> Path:
    movie = movie_name or get_default_movie_name()
    return get_interim_data_dir() / slugify_movie_name(movie)


def get_movie_processed_dir(movie_name: str | None = None) -> Path:
    movie = movie_name or get_default_movie_name()
    return get_processed_data_dir() / slugify_movie_name(movie)


def ensure_project_dirs(movie_name: str | None = None) -> None:
    """Create core project directories for a movie if they do not exist."""
    dirs = [
        get_data_dir(),
        get_raw_data_dir(),
        get_interim_data_dir(),
        get_processed_data_dir(),
        get_labeled_data_dir(),
        get_modeling_data_dir(),
        get_models_dir(),
        get_embeddings_dir(),
        get_outputs_dir(),
        get_modeling_outputs_dir(),
    ]

    if movie_name is not None:
        dirs.extend(
            [
                get_movie_raw_dir(movie_name),
                get_movie_interim_dir(movie_name),
                get_movie_processed_dir(movie_name),
            ]
        )

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def get_output_filename(key: str) -> str:
    settings = load_settings()
    outputs = settings.get("outputs", {})
    if key not in outputs:
        raise KeyError(
            f"Unknown output filename key: {key}. "
            f"Available keys: {sorted(outputs.keys())}"
        )
    return str(outputs[key])


def get_movie_raw_dialogue_metadata_path(movie_name: str | None = None) -> Path:
    """Return the raw path for an optional user-provided dialogue metadata CSV."""
    return get_movie_raw_dir(movie_name) / get_output_filename("dialogue_metadata")


def get_movie_dialogue_metadata_path(movie_name: str | None = None) -> Path:
    """Return the interim path for extracted dialogue metadata."""
    return get_movie_interim_dir(movie_name) / get_output_filename("dialogue_metadata")


def get_movie_clean_dialogue_path(movie_name: str | None = None) -> Path:
    """Return the interim path for cleaned dialogue metadata."""
    return get_movie_interim_dir(movie_name) / get_output_filename("dialogue_metadata_clean")


def get_movie_turn_windows_path(movie_name: str | None = None) -> Path:
    """Return the interim path for turn-window examples."""
    return get_movie_interim_dir(movie_name) / get_output_filename("turn_windows")


def get_movie_re_output_path(movie_name: str | None = None) -> Path:
    """Return the processed path for the CSV relationship extraction output."""
    return get_movie_processed_dir(movie_name) / get_output_filename("re_output")


def get_movie_re_raw_output_path(movie_name: str | None = None) -> Path:
    """Return the processed path for relationship extraction output.

    Kept as a backward-compatible alias during the CSV-only migration.
    """
    return get_movie_re_output_path(movie_name)


def get_movie_re_clean_output_path(movie_name: str | None = None) -> Path:
    """Return the processed path for the clean CSV relationship extraction output."""
    return get_movie_re_output_path(movie_name)


# --------- Modeling path helpers ---------

def get_modeling_config_path() -> Path:
    """Return the path to the modeling configuration file."""
    return CONFIGS_DIR / "modeling_config.json"


def get_power_respect_schema_path() -> Path:
    """Return the path to the power/respect label schema file."""
    return CONFIGS_DIR / "power_respect_schema.json"


def load_modeling_settings() -> dict[str, Any]:
    """Load modeling-specific settings from configs/modeling_config.json."""
    modeling_config_path = get_modeling_config_path()
    if not modeling_config_path.exists():
        raise FileNotFoundError(f"Modeling config file not found: {modeling_config_path}")
    modeling_settings = load_json(modeling_config_path)
    if not isinstance(modeling_settings, dict):
        raise ValueError(f"Modeling config must contain a JSON object: {modeling_config_path}")
    return modeling_settings


def get_modeling_path(key: str) -> Path:
    """Return an absolute path from configs/modeling_config.json by key."""
    settings = load_modeling_settings()
    if key not in settings:
        raise KeyError(
            f"Unknown modeling path key: {key}. "
            f"Available top-level keys: {sorted(settings.keys())}"
        )
    value = settings[key]
    if not isinstance(value, str):
        raise TypeError(f"Modeling config key '{key}' must be a string path.")
    return project_path(value)


def get_labeled_power_respect_path() -> Path:
    """Return the path to the labeled power/respect dataset."""
    return get_modeling_path("input_file")


def get_power_respect_modeling_data_path() -> Path:
    """Return the prepared modeling data path."""
    return get_modeling_path("prepared_data_file")


def get_power_respect_train_path() -> Path:
    """Return the modeling train split path."""
    return get_modeling_path("train_file")


def get_power_respect_val_path() -> Path:
    """Return the modeling validation split path."""
    return get_modeling_path("val_file")


def get_power_respect_test_path() -> Path:
    """Return the modeling test split path."""
    return get_modeling_path("test_file")


def get_embedding_file_path(split_name: str) -> Path:
    """Return the cached BERT embedding path for train/val/test."""
    settings = load_modeling_settings()
    embedding_settings = settings.get("embedding", {})
    if not isinstance(embedding_settings, dict):
        raise ValueError("modeling_config.json key 'embedding' must be an object.")

    key = f"{split_name}_embeddings_file"
    if key not in embedding_settings:
        raise KeyError(
            f"Unknown embedding split '{split_name}'. Expected one of: train, val, test."
        )
    return project_path(str(embedding_settings[key]))


def get_power_dynamic_model_path() -> Path:
    """Return the saved power dynamic classifier path."""
    settings = load_modeling_settings()
    return project_path(str(settings["saved_models"]["power_dynamic_model"]))


def get_respect_level_model_path() -> Path:
    """Return the saved respect level classifier path."""
    settings = load_modeling_settings()
    return project_path(str(settings["saved_models"]["respect_level_model"]))


def get_label_encoders_path() -> Path:
    """Return the saved label encoders path."""
    settings = load_modeling_settings()
    return project_path(str(settings["saved_models"]["label_encoders"]))


def get_modeling_evaluation_path(key: str) -> Path:
    """Return an evaluation output path from modeling_config.json."""
    settings = load_modeling_settings()
    evaluation_settings = settings.get("evaluation", {})
    if not isinstance(evaluation_settings, dict):
        raise ValueError("modeling_config.json key 'evaluation' must be an object.")
    if key not in evaluation_settings:
        raise KeyError(
            f"Unknown evaluation path key: {key}. "
            f"Available keys: {sorted(evaluation_settings.keys())}"
        )
    return project_path(str(evaluation_settings[key]))


def ensure_modeling_dirs() -> None:
    """Create directories needed by the power/respect modeling pipeline."""
    dirs = [
        get_labeled_data_dir(),
        get_modeling_data_dir(),
        get_models_dir(),
        get_embeddings_dir(),
        get_outputs_dir(),
        get_modeling_outputs_dir(),
    ]

    modeling_settings = load_modeling_settings()
    path_keys = [
        "input_file",
        "prepared_data_file",
        "train_file",
        "val_file",
        "test_file",
    ]
    for key in path_keys:
        value = modeling_settings.get(key)
        if isinstance(value, str):
            dirs.append(project_path(value).parent)

    for section_name in ["embedding", "saved_models", "evaluation"]:
        section = modeling_settings.get(section_name, {})
        if isinstance(section, dict):
            for value in section.values():
                if isinstance(value, str) and ("/" in value or "." in Path(value).name):
                    dirs.append(project_path(value).parent)

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def get_modeling_path_summary() -> dict[str, Path]:
    """Return key paths for the power/respect modeling pipeline."""
    return {
        "labeled_power_respect": get_labeled_power_respect_path(),
        "modeling_data": get_power_respect_modeling_data_path(),
        "train": get_power_respect_train_path(),
        "val": get_power_respect_val_path(),
        "test": get_power_respect_test_path(),
        "train_embeddings": get_embedding_file_path("train"),
        "val_embeddings": get_embedding_file_path("val"),
        "test_embeddings": get_embedding_file_path("test"),
        "power_dynamic_model": get_power_dynamic_model_path(),
        "respect_level_model": get_respect_level_model_path(),
        "label_encoders": get_label_encoders_path(),
    }


def get_movie_path_summary(movie_name: str | None = None) -> dict[str, Path]:
    """Return the key paths for one movie, useful for debugging."""
    movie = movie_name or get_default_movie_name()
    return {
        "movie_raw_dir": get_movie_raw_dir(movie),
        "movie_interim_dir": get_movie_interim_dir(movie),
        "movie_processed_dir": get_movie_processed_dir(movie),
        "raw_dialogue_metadata": get_movie_raw_dialogue_metadata_path(movie),
        "dialogue_metadata": get_movie_dialogue_metadata_path(movie),
        "dialogue_metadata_clean": get_movie_clean_dialogue_path(movie),
        "turn_windows": get_movie_turn_windows_path(movie),
        "re_output": get_movie_re_output_path(movie),
        "re_clean": get_movie_re_clean_output_path(movie),
    }