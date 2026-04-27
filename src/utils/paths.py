from __future__ import annotations

from pathlib import Path

from src.config import (
    CLEAN_DIALOGUE_FILENAME,
    DEFAULT_MOVIE_NAME,
    GRAPH_EDGES_FILENAME,
    GRAPH_NODES_FILENAME,
    INTERIM_DATA_DIR,
    LISTENER_ONLY_DIALOGUE_FILENAME,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RAW_DIALOGUE_FILENAME,
    RE_CLEAN_OUTPUT_FILENAME,
    RE_RAW_OUTPUT_FILENAME,
    RE_REVIEW_SHEET_FILENAME,
    RE_REVIEWED_FILENAME,
    SOCIAL_SUMMARIES_FILENAME,
    TRANSLATION_INPUTS_FILENAME,
    TURN_WINDOWS_FILENAME,
    get_movie_interim_dir,
    get_movie_processed_dir,
    slugify_movie_name,
)


def get_raw_data_dir() -> Path:
    """Return the root raw data directory."""
    return RAW_DATA_DIR


def get_interim_data_dir() -> Path:
    """Return the root interim data directory."""
    return INTERIM_DATA_DIR


def get_processed_data_dir() -> Path:
    """Return the root processed data directory."""
    return PROCESSED_DATA_DIR


def get_movie_raw_dir(movie_name: str = DEFAULT_MOVIE_NAME) -> Path:
    """Return the raw directory for a given movie.

    For now, this mirrors the global raw directory because movie-specific raw
    extraction may come from shared corpus sources.
    """
    return RAW_DATA_DIR / slugify_movie_name(movie_name)


def get_movie_dialogue_metadata_path(movie_name: str = DEFAULT_MOVIE_NAME) -> Path:
    """Return the interim path for extracted dialogue metadata."""
    return get_movie_interim_dir(movie_name) / RAW_DIALOGUE_FILENAME


def get_movie_listener_only_dialogue_path(movie_name: str = DEFAULT_MOVIE_NAME) -> Path:
    """Return the interim path for extracted dialogue metadata with known listeners only."""
    return get_movie_interim_dir(movie_name) / LISTENER_ONLY_DIALOGUE_FILENAME


def get_movie_clean_dialogue_path(movie_name: str = DEFAULT_MOVIE_NAME) -> Path:
    """Return the interim path for cleaned dialogue metadata."""
    return get_movie_interim_dir(movie_name) / CLEAN_DIALOGUE_FILENAME


def get_movie_turn_windows_path(movie_name: str = DEFAULT_MOVIE_NAME) -> Path:
    """Return the interim path for turn-window examples."""
    return get_movie_interim_dir(movie_name) / TURN_WINDOWS_FILENAME


def get_movie_re_raw_output_path(movie_name: str = DEFAULT_MOVIE_NAME) -> Path:
    """Return the processed path for raw LLM RE outputs."""
    return get_movie_processed_dir(movie_name) / RE_RAW_OUTPUT_FILENAME


def get_movie_re_clean_output_path(movie_name: str = DEFAULT_MOVIE_NAME) -> Path:
    """Return the processed path for normalized RE outputs."""
    return get_movie_processed_dir(movie_name) / RE_CLEAN_OUTPUT_FILENAME


def get_movie_re_review_sheet_path(movie_name: str = DEFAULT_MOVIE_NAME) -> Path:
    """Return the processed path for the manual review sheet."""
    return get_movie_processed_dir(movie_name) / RE_REVIEW_SHEET_FILENAME


def get_movie_re_reviewed_path(movie_name: str = DEFAULT_MOVIE_NAME) -> Path:
    """Return the processed path for reviewed RE labels."""
    return get_movie_processed_dir(movie_name) / RE_REVIEWED_FILENAME


def get_movie_graph_nodes_path(movie_name: str = DEFAULT_MOVIE_NAME) -> Path:
    """Return the processed path for graph nodes."""
    return get_movie_processed_dir(movie_name) / GRAPH_NODES_FILENAME


def get_movie_graph_edges_path(movie_name: str = DEFAULT_MOVIE_NAME) -> Path:
    """Return the processed path for graph edges."""
    return get_movie_processed_dir(movie_name) / GRAPH_EDGES_FILENAME


def get_movie_social_summaries_path(movie_name: str = DEFAULT_MOVIE_NAME) -> Path:
    """Return the processed path for social summaries."""
    return get_movie_processed_dir(movie_name) / SOCIAL_SUMMARIES_FILENAME


def get_movie_translation_inputs_path(movie_name: str = DEFAULT_MOVIE_NAME) -> Path:
    """Return the processed path for translation-ready inputs."""
    return get_movie_processed_dir(movie_name) / TRANSLATION_INPUTS_FILENAME