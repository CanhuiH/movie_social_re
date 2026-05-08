"""Graph-guided Mandarin translation utilities.

This module implements the graph-guided translation condition. It uses the same
basic dialogue context as the context-only baseline, plus structured social
signals from the current row and aggregate/recent speaker-listener graph states.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from src.utils.io import ensure_parent_dir, load_csv, load_json, save_csv
from src.utils.paths import project_path

DEFAULT_INPUT_PATH = project_path(Path("data") / "interim" / "translation_input.csv")
DEFAULT_OUTPUT_PATH = project_path(Path("data") / "translation_eval" / "translation_graph_guided.csv")
DEFAULT_PROMPT_PATH = project_path(Path("prompts") / "translate_graph_guided.txt")
DEFAULT_RELATIONSHIP_SCHEMA_PATH = project_path(Path("configs") / "relationship_schema.json")
DEFAULT_POWER_RESPECT_SCHEMA_PATH = project_path(Path("configs") / "power_respect_schema.json")
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_PROVIDER = "gemini"
DEFAULT_API_KEY_ENV = "GEMINI_API_KEY"
JOIN_KEYS = ["movie_name", "conversation_id", "utterance_id"]

POWER_LABEL_DEFINITIONS = {
    "speaker_higher_power": "The speaker has higher social, institutional, role-based, or situational power than the listener.",
    "listener_higher_power": "The listener has higher social, institutional, role-based, or situational power than the speaker.",
    "roughly_equal_power": "The speaker and listener have roughly equal power in this local interaction.",
    "unclear": "The power dynamic is unclear from the available context.",
}

RESPECT_LABEL_DEFINITIONS = {
    "high_respect": "The speaker shows deference, politeness, admiration, caution, or formal respect toward the listener.",
    "neutral_respect": "The speaker uses a neutral or ordinary level of respect without strong deference or disrespect.",
    "low_respect": "The speaker shows disrespect, hostility, resistance, contempt, sarcasm, or strong informality toward the listener.",
    "unclear": "The respect level is unclear from the available context.",
}


@dataclass(frozen=True)
class GraphGuidedTranslationResult:
    """Paths and summary statistics for graph-guided translation."""

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


def get_optional_field(row: pd.Series, column: str, default: str = "unclear") -> str:
    """Return a normalized optional field from a row."""
    if column not in row.index:
        return default
    return normalize_text(row.get(column), default=default)


def choose_relationship_evidence(row: pd.Series) -> str:
    """Return relationship-extraction evidence for the current row."""
    for column in ["evidence", "relationship_evidence", "re_evidence"]:
        if column in row.index:
            value = normalize_text(row.get(column))
            if value:
                return value
    return "[No relationship evidence available.]"


def make_row_key(row: pd.Series) -> tuple[str, str, str]:
    """Create a stable row key for resume logic."""
    return tuple(normalize_text(row.get(column)) for column in JOIN_KEYS)



def load_prompt_template(prompt_path: Path = DEFAULT_PROMPT_PATH) -> str:
    """Load the graph-guided translation prompt template."""
    if not prompt_path.exists():
        raise FileNotFoundError(f"Graph-guided prompt template not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def load_schema(path: Path) -> dict[str, Any]:
    """Load a JSON schema file if it exists."""
    if not path.exists():
        return {}
    schema = load_json(path)
    return schema if isinstance(schema, dict) else {}


def normalize_schema_key(value: Any) -> str:
    """Normalize schema keys for movie names and labels."""
    return normalize_text(value).lower()


def load_relationship_schema(path: Path = DEFAULT_RELATIONSHIP_SCHEMA_PATH) -> dict[str, Any]:
    """Load the movie-specific relationship schema."""
    return load_schema(path)


def load_power_respect_schema(path: Path = DEFAULT_POWER_RESPECT_SCHEMA_PATH) -> dict[str, Any]:
    """Load the fixed power/respect schema."""
    return load_schema(path)


def get_movie_relationship_schema(relationship_schema: dict[str, Any], movie_name: str) -> dict[str, Any]:
    """Return the movie-specific relationship schema for a movie."""
    movie_schemas = relationship_schema.get("movie_schemas", {})
    if not isinstance(movie_schemas, dict):
        return {}
    return movie_schemas.get(normalize_schema_key(movie_name), {})


def get_relationship_type_definition(
    relationship_schema: dict[str, Any],
    movie_name: str,
    relationship_type: str,
) -> str:
    """Return the movie-specific definition for a relationship type."""
    movie_schema = get_movie_relationship_schema(relationship_schema, movie_name)
    definitions = movie_schema.get("definitions", {}) if isinstance(movie_schema, dict) else {}
    relationship_type = normalize_schema_key(relationship_type)
    if isinstance(definitions, dict) and relationship_type in definitions:
        return normalize_text(definitions[relationship_type])
    if relationship_type == "unclear":
        return "Not enough evidence is available to infer a reliable relationship type."
    return f"The relationship type is labeled as {relationship_type.replace('_', ' ')}."


def get_movie_schema_focus(relationship_schema: dict[str, Any], movie_name: str) -> str:
    """Return movie-specific schema focus text."""
    movie_schema = get_movie_relationship_schema(relationship_schema, movie_name)
    if isinstance(movie_schema, dict):
        return normalize_text(movie_schema.get("schema_focus"))
    return ""


def get_movie_priority_rules(relationship_schema: dict[str, Any], movie_name: str) -> str:
    """Return movie-specific relationship priority rules as compact text."""
    movie_schema = get_movie_relationship_schema(relationship_schema, movie_name)
    priority_rules = movie_schema.get("priority_rules", []) if isinstance(movie_schema, dict) else []
    if isinstance(priority_rules, list):
        return " ".join(str(rule).strip() for rule in priority_rules if str(rule).strip())
    return ""


def get_power_label_definition(label: str, power_respect_schema: dict[str, Any] | None = None) -> str:
    """Return a definition for a power label."""
    label = normalize_schema_key(label) or "unclear"
    return POWER_LABEL_DEFINITIONS.get(label, f"The power dynamic is labeled as {label.replace('_', ' ')}.")


def get_respect_label_definition(label: str, power_respect_schema: dict[str, Any] | None = None) -> str:
    """Return a definition for a respect label."""
    label = normalize_schema_key(label) or "unclear"
    return RESPECT_LABEL_DEFINITIONS.get(label, f"The respect level is labeled as {label.replace('_', ' ')}.")



def build_graph_guided_prompt(
    row: pd.Series,
    prompt_template: str | None = None,
    relationship_schema: dict[str, Any] | None = None,
    power_respect_schema: dict[str, Any] | None = None,
) -> str:
    """Build a graph-guided translation prompt.

    The prompt includes dialogue context, current-row social signals, schema
    definitions, and aggregate/recent graph-state background. It asks the model
    to silently use the guidance and output a JSON summary plus translation.
    """
    movie_name = normalize_text(row.get("movie_name"), default="unknown movie")
    speaker_name = normalize_text(row.get("speaker_name"), default="Unknown speaker")
    listener_name = normalize_text(row.get("listener_name"), default="Unknown listener")
    context = choose_context(row)
    current_text = choose_current_text(row)

    final_power_dynamic = get_optional_field(row, "final_power_dynamic")
    final_respect_level = get_optional_field(row, "final_respect_level")
    relationship_type = get_optional_field(row, "relationship_type")
    relationship_evidence = choose_relationship_evidence(row)

    recent_power_dynamic = get_optional_field(row, "recent_power_dynamic")
    recent_respect_level = get_optional_field(row, "recent_respect_level")
    recent_relationship_type = get_optional_field(row, "recent_relationship_type")

    aggregate_power_dynamic = get_optional_field(row, "aggregate_power_dynamic")
    aggregate_respect_level = get_optional_field(row, "aggregate_respect_level")
    aggregate_relationship_type = get_optional_field(row, "aggregate_relationship_type")

    if prompt_template is None:
        prompt_template = load_prompt_template()
    if relationship_schema is None:
        relationship_schema = load_relationship_schema()
    if power_respect_schema is None:
        power_respect_schema = load_power_respect_schema()

    return prompt_template.format(
        movie_name=movie_name,
        speaker_name=speaker_name,
        listener_name=listener_name,
        context_text=context if context else "[No previous context provided.]",
        current_text=current_text,
        final_power_dynamic=final_power_dynamic,
        final_power_dynamic_definition=get_power_label_definition(final_power_dynamic, power_respect_schema),
        final_respect_level=final_respect_level,
        final_respect_level_definition=get_respect_label_definition(final_respect_level, power_respect_schema),
        relationship_type=relationship_type,
        relationship_type_definition=get_relationship_type_definition(
            relationship_schema, movie_name, relationship_type
        ),
        relationship_evidence=relationship_evidence,
        movie_schema_focus=get_movie_schema_focus(relationship_schema, movie_name),
        movie_relationship_priority_rules=get_movie_priority_rules(relationship_schema, movie_name),
        recent_power_dynamic=recent_power_dynamic,
        recent_power_dynamic_definition=get_power_label_definition(recent_power_dynamic, power_respect_schema),
        recent_respect_level=recent_respect_level,
        recent_respect_level_definition=get_respect_label_definition(recent_respect_level, power_respect_schema),
        recent_relationship_type=recent_relationship_type,
        recent_relationship_type_definition=get_relationship_type_definition(
            relationship_schema, movie_name, recent_relationship_type
        ),
        aggregate_power_dynamic=aggregate_power_dynamic,
        aggregate_power_dynamic_definition=get_power_label_definition(aggregate_power_dynamic, power_respect_schema),
        aggregate_respect_level=aggregate_respect_level,
        aggregate_respect_level_definition=get_respect_label_definition(aggregate_respect_level, power_respect_schema),
        aggregate_relationship_type=aggregate_relationship_type,
        aggregate_relationship_type_definition=get_relationship_type_definition(
            relationship_schema, movie_name, aggregate_relationship_type
        ),
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


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from an LLM response.

    The prompt asks for strict JSON, but this helper is tolerant of fenced JSON
    blocks or extra surrounding text.
    """
    cleaned_text = normalize_text(text)
    if not cleaned_text:
        return {}

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_text, flags=re.DOTALL)
    if fenced_match:
        cleaned_text = fenced_match.group(1).strip()

    try:
        parsed = json.loads(cleaned_text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    start = cleaned_text.find("{")
    end = cleaned_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned_text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    return {}


def parse_graph_guided_response(response_text: str) -> tuple[str, str, str]:
    """Parse graph-guided LLM output into summary, translation, and raw response.

    Returns:
        A tuple of (llm_social_guidance_summary, translation_graph_guided, raw_response).
    """
    raw_response = normalize_text(response_text)
    parsed = extract_json_object(raw_response)
    if parsed:
        summary = normalize_text(parsed.get("social_guidance_summary"))
        translation = normalize_text(parsed.get("translation_graph_guided"))
        return summary, translation, raw_response

    # Fallback: if the model ignores JSON, keep the raw response as translation.
    return "", raw_response, raw_response


def translate_prompt(prompt: str, provider: str, model: str, api_key_env: str, dry_run: bool = False) -> str:
    """Call the selected provider and return the raw LLM response text."""
    if dry_run:
        return json.dumps(
            {
                "social_guidance_summary": "[DRY RUN SOCIAL GUIDANCE SUMMARY]",
                "translation_graph_guided": "[DRY RUN GRAPH-GUIDED TRANSLATION]",
            },
            ensure_ascii=False,
        )
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
    """Return keys that already have non-empty graph-guided outputs."""
    required_output_columns = ["translation_graph_guided", "llm_social_guidance_summary"]
    if existing_df.empty or any(column not in existing_df.columns for column in required_output_columns):
        return set()
    missing_keys = [column for column in JOIN_KEYS if column not in existing_df.columns]
    if missing_keys:
        return set()

    completed_df = existing_df.copy()
    for column in required_output_columns:
        completed_df = completed_df[completed_df[column].notna()]
        completed_df = completed_df[completed_df[column].astype(str).str.strip() != ""]
    return {make_row_key(row) for _, row in completed_df.iterrows()}


def build_output_row(
    row: pd.Series,
    prompt: str,
    social_guidance_summary: str,
    translation: str,
    raw_response: str,
    provider: str,
    model: str,
) -> dict[str, Any]:
    """Build one graph-guided translation output row."""
    output = {column: row.get(column, "") for column in row.index}
    output["llm_social_guidance_summary"] = social_guidance_summary
    output["translation_graph_guided"] = translation
    output["translation_model_graph_guided"] = model
    output["translation_provider_graph_guided"] = provider
    output["translation_prompt_graph_guided"] = prompt
    output["translation_raw_response_graph_guided"] = raw_response
    return output


def translate_graph_guided_dataset(
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
    """Generate graph-guided translations and save progress."""
    if not input_path.exists():
        raise FileNotFoundError(f"Translation input file not found: {input_path}")

    input_df = load_csv(input_path)
    if max_rows is not None:
        input_df = input_df.head(max_rows).copy()

    prompt_template = load_prompt_template(prompt_path)
    relationship_schema = load_relationship_schema()
    power_respect_schema = load_power_respect_schema()

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
        desc="Translating graph-guided",
        unit="row",
        dynamic_ncols=True,
    )

    for row in progress_bar:
        prompt = build_graph_guided_prompt(
            row,
            prompt_template=prompt_template,
            relationship_schema=relationship_schema,
            power_respect_schema=power_respect_schema,
        )
        try:
            raw_response = translate_prompt(
                prompt=prompt,
                provider=provider,
                model=model,
                api_key_env=api_key_env,
                dry_run=dry_run,
            )
            social_guidance_summary, translation, raw_response = parse_graph_guided_response(raw_response)
            if not translation:
                status = "failed"
                error = "Parsed response did not contain translation_graph_guided."
            else:
                status = "success"
                error = ""
        except Exception as exc:  # keep batch running and save error row
            social_guidance_summary = ""
            translation = ""
            raw_response = ""
            status = "failed"
            error = str(exc)

        output_row = build_output_row(
            row=row,
            prompt=prompt,
            social_guidance_summary=social_guidance_summary,
            translation=translation,
            raw_response=raw_response,
            provider=provider,
            model=model,
        )
        output_row["translation_status_graph_guided"] = status
        output_row["translation_error_graph_guided"] = error
        output_rows.append(output_row)
        translated_count += 1
        progress_bar.set_postfix(
            {
                "success": sum(
                    1 for item in output_rows if item.get("translation_status_graph_guided") == "success"
                ),
                "failed": sum(
                    1 for item in output_rows if item.get("translation_status_graph_guided") == "failed"
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


def summarize_graph_guided_output(
    output_df: pd.DataFrame,
    output_path: Path,
    provider: str,
    model: str,
) -> GraphGuidedTranslationResult:
    """Create a compact result object from a graph-guided translation output dataframe."""
    if "translation_status_graph_guided" in output_df.columns:
        success_rows = int((output_df["translation_status_graph_guided"] == "success").sum())
        failed_rows = int((output_df["translation_status_graph_guided"] == "failed").sum())
    else:
        success_rows = 0
        failed_rows = 0

    return GraphGuidedTranslationResult(
        output_path=output_path,
        rows=len(output_df),
        success_rows=success_rows,
        failed_rows=failed_rows,
        provider=provider,
        model=model,
    )


def print_translation_summary(result: GraphGuidedTranslationResult) -> None:
    """Print a compact summary for graph-guided translation."""
    print("Graph-guided translation complete")
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
    "DEFAULT_POWER_RESPECT_SCHEMA_PATH",
    "DEFAULT_PROVIDER",
    "DEFAULT_RELATIONSHIP_SCHEMA_PATH",
    "GraphGuidedTranslationResult",
    "extract_json_object",
    "build_graph_guided_prompt",
    "call_gemini",
    "call_openai",
    "choose_context",
    "choose_current_text",
    "choose_relationship_evidence",
    "get_movie_priority_rules",
    "get_movie_schema_focus",
    "get_optional_field",
    "get_power_label_definition",
    "get_relationship_type_definition",
    "get_respect_label_definition",
    "load_power_respect_schema",
    "load_prompt_template",
    "load_relationship_schema",
    "parse_graph_guided_response",
    "print_translation_summary",
    "summarize_graph_guided_output",
    "translate_graph_guided_dataset",
    "translate_prompt",
]