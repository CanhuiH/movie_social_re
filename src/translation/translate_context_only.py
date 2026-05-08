"""Context-only Mandarin translation utilities.

This module implements the baseline translation condition. It uses only dialogue
context, speaker/listener metadata, and the current English line. It intentionally
excludes graph_summary and gold social labels so it can be compared against the
graph-guided translation condition.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from src.utils.io import ensure_parent_dir, load_csv, save_csv
from src.utils.paths import project_path

DEFAULT_INPUT_PATH = project_path(Path("data") / "interim" / "translation_input.csv")
DEFAULT_OUTPUT_PATH = project_path(Path("data") / "translation_eval" / "translation_context_only.csv")
DEFAULT_PROMPT_PATH = project_path(Path("prompts") / "translate_context_only.txt")
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_PROVIDER = "gemini"
DEFAULT_API_KEY_ENV = "GEMINI_API_KEY"
JOIN_KEYS = ["movie_name", "conversation_id", "utterance_id"]


@dataclass(frozen=True)
class ContextOnlyTranslationResult:
    """Paths and summary statistics for context-only translation."""

    output_path: Path
    rows: int
    success_rows: int
    failed_rows: int
    provider: str
    model: str


def normalize_text(value: Any, default: str = "") -> str:
    """Normalize a scalar value to clean text."""
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def choose_context(row: pd.Series) -> str:
    """Choose the best available dialogue context column."""
    for column in ["context_6_turns", "context_text", "previous_context"]:
        if column in row.index:
            value = normalize_text(row.get(column))
            if value:
                return value
    return ""


def choose_current_text(row: pd.Series) -> str:
    """Choose the best available current-text column."""
    for column in ["current_text", "text", "current_turn"]:
        if column in row.index:
            value = normalize_text(row.get(column))
            if value:
                return value
    raise ValueError("Input row is missing current text. Expected one of: current_text, text, current_turn")


def make_row_key(row: pd.Series) -> tuple[str, str, str]:
    """Create a stable row key for resume logic."""
    return tuple(normalize_text(row.get(column)) for column in JOIN_KEYS)


def load_prompt_template(prompt_path: Path = DEFAULT_PROMPT_PATH) -> str:
    """Load the context-only translation prompt template."""
    if not prompt_path.exists():
        raise FileNotFoundError(f"Context-only prompt template not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def build_context_only_prompt(row: pd.Series, prompt_template: str | None = None) -> str:
    """Build a context-only translation prompt.

    This prompt intentionally excludes graph_summary and any gold social labels.
    """
    movie_name = normalize_text(row.get("movie_name"), default="unknown movie")
    speaker_name = normalize_text(row.get("speaker_name"), default="Unknown speaker")
    listener_name = normalize_text(row.get("listener_name"), default="Unknown listener")
    context = choose_context(row)
    current_text = choose_current_text(row)

    if prompt_template is None:
        prompt_template = load_prompt_template()

    return prompt_template.format(
        movie_name=movie_name,
        speaker_name=speaker_name,
        listener_name=listener_name,
        context_text=context if context else "[No previous context provided.]",
        current_text=current_text,
    )


def call_gemini(prompt: str, model: str, api_key_env: str) -> str:
    """Call Gemini and return translated text."""
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise EnvironmentError(f"Missing API key environment variable: {api_key_env}")

    try:
        from google import genai
    except ImportError as exc:
        raise ImportError("google-genai is required. Install it with: pip install google-genai") from exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    text = getattr(response, "text", None)
    return normalize_text(text)


def call_openai(prompt: str, model: str, api_key_env: str) -> str:
    """Call OpenAI and return translated text."""
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise EnvironmentError(f"Missing API key environment variable: {api_key_env}")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai is required. Install it with: pip install openai") from exc

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a careful Mandarin subtitle translator. Output only the translation.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return normalize_text(response.choices[0].message.content)


def translate_prompt(prompt: str, provider: str, model: str, api_key_env: str, dry_run: bool = False) -> str:
    """Translate one prompt using the selected provider."""
    if dry_run:
        return "[DRY RUN TRANSLATION]"
    if provider == "gemini":
        return call_gemini(prompt, model=model, api_key_env=api_key_env)
    if provider == "openai":
        return call_openai(prompt, model=model, api_key_env=api_key_env)
    raise ValueError(f"Unsupported provider: {provider}")


def load_existing_output(output_path: Path, overwrite: bool) -> pd.DataFrame:
    """Load an existing output file for resume mode."""
    if overwrite or not output_path.exists():
        return pd.DataFrame()
    return load_csv(output_path)


def get_completed_keys(existing_df: pd.DataFrame) -> set[tuple[str, str, str]]:
    """Return keys that already have non-empty context-only translations."""
    if existing_df.empty or "translation_context_only" not in existing_df.columns:
        return set()
    missing_keys = [column for column in JOIN_KEYS if column not in existing_df.columns]
    if missing_keys:
        return set()

    completed_df = existing_df[existing_df["translation_context_only"].notna()].copy()
    completed_df = completed_df[completed_df["translation_context_only"].astype(str).str.strip() != ""]
    return {make_row_key(row) for _, row in completed_df.iterrows()}


def build_output_row(row: pd.Series, prompt: str, translation: str, provider: str, model: str) -> dict[str, Any]:
    """Build one context-only translation output row."""
    output = {column: row.get(column, "") for column in row.index}
    output["translation_context_only"] = translation
    output["translation_model_context_only"] = model
    output["translation_provider_context_only"] = provider
    output["translation_prompt_context_only"] = prompt
    return output


def translate_context_only_dataset(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    max_rows: int | None = None,
    overwrite: bool = False,
    save_every: int = 10,
    sleep_seconds: float = 0.0,
    dry_run: bool = False,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
) -> pd.DataFrame:
    """Generate context-only translations and save progress."""
    if not input_path.exists():
        raise FileNotFoundError(f"Translation input file not found: {input_path}")

    input_df = load_csv(input_path)
    if max_rows is not None:
        input_df = input_df.head(max_rows).copy()

    prompt_template = load_prompt_template(prompt_path)

    existing_df = load_existing_output(output_path, overwrite=overwrite)
    completed_keys = get_completed_keys(existing_df)
    output_rows: list[dict[str, Any]] = []
    if not existing_df.empty and not overwrite:
        output_rows.extend(existing_df.to_dict(orient="records"))

    total_rows = len(input_df)
    translated_count = 0

    rows_to_process = [row for _, row in input_df.iterrows() if make_row_key(row) not in completed_keys]

    progress_bar = tqdm(
        rows_to_process,
        total=len(rows_to_process),
        desc="Translating context-only",
        unit="row",
        dynamic_ncols=True,
    )

    for row in progress_bar:
        prompt = build_context_only_prompt(row, prompt_template=prompt_template)
        try:
            translation = translate_prompt(
                prompt=prompt,
                provider=provider,
                model=model,
                api_key_env=api_key_env,
                dry_run=dry_run,
            )
            status = "success"
            error = ""
        except Exception as exc:  # keep batch running and save error row
            translation = ""
            status = "failed"
            error = str(exc)

        output_row = build_output_row(row, prompt, translation, provider, model)
        output_row["translation_status_context_only"] = status
        output_row["translation_error_context_only"] = error
        output_rows.append(output_row)
        translated_count += 1
        progress_bar.set_postfix(
            {
                "success": sum(
                    1 for item in output_rows if item.get("translation_status_context_only") == "success"
                ),
                "failed": sum(
                    1 for item in output_rows if item.get("translation_status_context_only") == "failed"
                ),
            }
        )

        if translated_count % max(save_every, 1) == 0:
            ensure_parent_dir(output_path)
            save_csv(pd.DataFrame(output_rows), output_path)
            print(f"Saved progress: {translated_count} new translations processed / {total_rows} input rows")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    output_df = pd.DataFrame(output_rows)
    ensure_parent_dir(output_path)
    save_csv(output_df, output_path)
    return output_df


def summarize_context_only_output(output_df: pd.DataFrame, output_path: Path, provider: str, model: str) -> ContextOnlyTranslationResult:
    """Create a compact result object from a context-only translation output dataframe."""
    if "translation_status_context_only" in output_df.columns:
        success_rows = int((output_df["translation_status_context_only"] == "success").sum())
        failed_rows = int((output_df["translation_status_context_only"] == "failed").sum())
    else:
        success_rows = 0
        failed_rows = 0

    return ContextOnlyTranslationResult(
        output_path=output_path,
        rows=len(output_df),
        success_rows=success_rows,
        failed_rows=failed_rows,
        provider=provider,
        model=model,
    )


def print_translation_summary(result: ContextOnlyTranslationResult) -> None:
    """Print a compact summary for context-only translation."""
    print("Context-only translation complete")
    print(f"  rows     : {result.rows}")
    print(f"  success  : {result.success_rows}")
    print(f"  failed   : {result.failed_rows}")
    print(f"  provider : {result.provider}")
    print(f"  model    : {result.model}")
    print(f"  output   : {result.output_path}")


__all__ = [
    "DEFAULT_API_KEY_ENV",
    "DEFAULT_INPUT_PATH",
    "DEFAULT_MODEL",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_PROMPT_PATH",
    "DEFAULT_PROVIDER",
    "ContextOnlyTranslationResult",
    "build_context_only_prompt",
    "call_gemini",
    "call_openai",
    "choose_context",
    "choose_current_text",
    "load_prompt_template",
    "print_translation_summary",
    "summarize_context_only_output",
    "translate_context_only_dataset",
    "translate_prompt",
]