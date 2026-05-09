

"""Generate power/respect-only guided translations.

This ablation condition uses the current dialogue context plus only the
current-row power and respect labels:

- final_power_dynamic
- final_respect_level

It intentionally does not use relationship type, relationship evidence,
recent graph state, or aggregate graph state.
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

from src.utils.io import ensure_parent_dir, load_csv, load_json, save_csv
from src.utils.paths import project_path


DEFAULT_INPUT_PATH = project_path(Path("data") / "interim" / "translation_input.csv")
DEFAULT_OUTPUT_PATH = project_path(Path("data") / "translation_eval" / "translation_power_respect_only.csv")
DEFAULT_PROMPT_PATH = project_path(Path("prompts") / "translate_power_respect_only.txt")
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
class PowerRespectOnlyResult:
    """Summary information for a power/respect-only translation run."""

    input_path: Path
    output_path: Path
    rows_loaded: int
    rows_to_process: int
    rows_written: int
    success_count: int
    failure_count: int


def choose_context(row: pd.Series) -> str:
    """Choose the best available previous dialogue context."""
    for column in ["context_6_turns", "context_text", "previous_context"]:
        if column in row.index and pd.notna(row[column]) and str(row[column]).strip():
            return str(row[column]).strip()
    return "[No previous dialogue context available]"


def choose_current_text(row: pd.Series) -> str:
    """Choose the current source line."""
    for column in ["current_text", "text", "en"]:
        if column in row.index and pd.notna(row[column]) and str(row[column]).strip():
            return str(row[column]).strip()
    return ""


def get_optional_field(row: pd.Series, column: str, default: str = "unclear") -> str:
    """Return a cleaned row field with a default fallback."""
    if column not in row.index or pd.isna(row[column]) or not str(row[column]).strip():
        return default
    return str(row[column]).strip()


def load_prompt_template(prompt_path: Path = DEFAULT_PROMPT_PATH) -> str:
    """Load the power/respect-only prompt template."""
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}. "
            "Create prompts/translate_power_respect_only.txt before running this script."
        )
    return prompt_path.read_text(encoding="utf-8")


def load_schema(path: Path) -> dict[str, Any]:
    """Load a JSON schema file if it exists."""
    if not path.exists():
        return {}
    data = load_json(path)
    return data if isinstance(data, dict) else {}


def normalize_schema_key(value: Any) -> str:
    """Normalize schema keys for lookup."""
    return str(value).strip().lower().replace(" ", "_")


def load_power_respect_schema(path: Path = DEFAULT_POWER_RESPECT_SCHEMA_PATH) -> dict[str, Any]:
    """Load shared power/respect schema definitions."""
    return load_schema(path)


def get_power_label_definition(label: str, power_respect_schema: dict[str, Any] | None = None) -> str:
    """Return a definition for a power label."""
    normalized = normalize_schema_key(label)
    if power_respect_schema:
        power_schema = power_respect_schema.get("power_dynamic") or power_respect_schema.get("power")
        if isinstance(power_schema, dict):
            value = power_schema.get(normalized) or power_schema.get(label)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                for key in ["definition", "description", "meaning"]:
                    definition = value.get(key)
                    if isinstance(definition, str) and definition.strip():
                        return definition.strip()
    return POWER_LABEL_DEFINITIONS.get(normalized, POWER_LABEL_DEFINITIONS["unclear"])


def get_respect_label_definition(label: str, power_respect_schema: dict[str, Any] | None = None) -> str:
    """Return a definition for a respect label."""
    normalized = normalize_schema_key(label)
    if power_respect_schema:
        respect_schema = power_respect_schema.get("respect_level") or power_respect_schema.get("respect")
        if isinstance(respect_schema, dict):
            value = respect_schema.get(normalized) or respect_schema.get(label)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                for key in ["definition", "description", "meaning"]:
                    definition = value.get(key)
                    if isinstance(definition, str) and definition.strip():
                        return definition.strip()
    return RESPECT_LABEL_DEFINITIONS.get(normalized, RESPECT_LABEL_DEFINITIONS["unclear"])


def build_power_respect_only_prompt(
    row: pd.Series,
    prompt_template: str | None = None,
    power_respect_schema: dict[str, Any] | None = None,
) -> str:
    """Build the power/respect-only translation prompt for one row."""
    if prompt_template is None:
        prompt_template = load_prompt_template()
    if power_respect_schema is None:
        power_respect_schema = load_power_respect_schema()

    power_label = get_optional_field(row, "final_power_dynamic")
    respect_label = get_optional_field(row, "final_respect_level")

    return prompt_template.format(
        movie_name=get_optional_field(row, "movie_name", "unknown"),
        speaker_name=get_optional_field(row, "speaker_name", "unknown speaker"),
        listener_name=get_optional_field(row, "listener_name", "unknown listener"),
        context_text=choose_context(row),
        current_text=choose_current_text(row),
        final_power_dynamic=power_label,
        final_power_dynamic_definition=get_power_label_definition(power_label, power_respect_schema),
        final_respect_level=respect_label,
        final_respect_level_definition=get_respect_label_definition(respect_label, power_respect_schema),
    )


def call_gemini(prompt: str, model: str, api_key: str) -> str:
    """Call Gemini and return text."""
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text or ""


def call_openai(prompt: str, model: str, api_key: str) -> str:
    """Call OpenAI and return text."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from model output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON is not an object.")
    return parsed


def parse_power_respect_only_response(response_text: str) -> tuple[str, str, str]:
    """Parse model output into guidance summary and translation."""
    parsed = extract_json_object(response_text)
    guidance = str(parsed.get("power_respect_guidance_summary", "")).strip()
    translation = str(parsed.get("translation_power_respect_only", "")).strip()
    if not translation:
        raise ValueError("Model response is missing translation_power_respect_only.")
    return guidance, translation, response_text


def translate_prompt(
    prompt: str,
    provider: str,
    model: str,
    api_key: str,
    dry_run: bool = False,
) -> str:
    """Translate one prompt using the selected provider."""
    if dry_run:
        return json.dumps(
            {
                "power_respect_guidance_summary": "[DRY RUN POWER/RESPECT GUIDANCE SUMMARY]",
                "translation_power_respect_only": "[DRY RUN POWER/RESPECT-GUIDED TRANSLATION]",
            },
            ensure_ascii=False,
        )

    provider = provider.lower().strip()
    if provider == "gemini":
        return call_gemini(prompt, model=model, api_key=api_key)
    if provider == "openai":
        return call_openai(prompt, model=model, api_key=api_key)
    raise ValueError(f"Unsupported provider: {provider}")


def make_row_key(row: pd.Series) -> tuple[str, str, str]:
    """Build a stable key for resume behavior."""
    return tuple(str(row[column]) for column in JOIN_KEYS)


def load_existing_output(output_path: Path) -> pd.DataFrame:
    """Load existing output if present."""
    if output_path.exists():
        return load_csv(output_path)
    return pd.DataFrame()


def get_completed_keys(existing_df: pd.DataFrame) -> set[tuple[str, str, str]]:
    """Return keys for rows that already have a successful translation."""
    if existing_df.empty:
        return set()
    required_columns = set(JOIN_KEYS + ["translation_power_respect_only"])
    if not required_columns.issubset(existing_df.columns):
        return set()

    completed_df = existing_df[
        existing_df["translation_power_respect_only"].notna()
        & existing_df["translation_power_respect_only"].astype(str).str.strip().ne("")
    ]
    return {make_row_key(row) for _, row in completed_df.iterrows()}


def translate_power_respect_only_dataset(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    max_rows: int | None = None,
    sleep_seconds: float = 0.0,
    overwrite: bool = False,
    dry_run: bool = False,
) -> PowerRespectOnlyResult:
    """Generate power/respect-only translations for the input dataset."""
    load_dotenv()

    input_df = load_csv(input_path)
    if max_rows is not None:
        input_df = input_df.head(max_rows).copy()

    prompt_template = load_prompt_template(prompt_path)
    power_respect_schema = load_power_respect_schema()

    existing_df = pd.DataFrame() if overwrite else load_existing_output(output_path)
    completed_keys = set() if overwrite else get_completed_keys(existing_df)

    rows_to_process: list[pd.Series] = []
    for _, row in input_df.iterrows():
        if not overwrite and make_row_key(row) in completed_keys:
            continue
        rows_to_process.append(row)

    api_key = os.getenv(api_key_env, "")
    if not dry_run and not api_key:
        raise ValueError(f"Missing API key. Set {api_key_env} in your environment or .env file.")

    output_rows: list[dict[str, Any]] = []
    success_count = 0
    failure_count = 0

    progress_bar = tqdm(
        rows_to_process,
        total=len(rows_to_process),
        desc="Translating power/respect-only",
        unit="row",
        dynamic_ncols=True,
    )

    for row in progress_bar:
        row_output = row.to_dict()
        prompt = build_power_respect_only_prompt(
            row,
            prompt_template=prompt_template,
            power_respect_schema=power_respect_schema,
        )
        row_output["translation_prompt_power_respect_only"] = prompt
        row_output["translation_model_power_respect_only"] = model
        row_output["translation_provider_power_respect_only"] = provider

        try:
            response_text = translate_prompt(
                prompt=prompt,
                provider=provider,
                model=model,
                api_key=api_key,
                dry_run=dry_run,
            )
            guidance, translation, raw_response = parse_power_respect_only_response(response_text)
            row_output["llm_power_respect_only_guidance_summary"] = guidance
            row_output["translation_power_respect_only"] = translation
            row_output["translation_raw_response_power_respect_only"] = raw_response
            row_output["translation_status_power_respect_only"] = "success"
            row_output["translation_error_power_respect_only"] = ""
            success_count += 1
        except Exception as error:  # noqa: BLE001 - keep row-level failures in output CSV
            row_output["llm_power_respect_only_guidance_summary"] = ""
            row_output["translation_power_respect_only"] = ""
            row_output["translation_raw_response_power_respect_only"] = ""
            row_output["translation_status_power_respect_only"] = "failed"
            row_output["translation_error_power_respect_only"] = str(error)
            failure_count += 1

        output_rows.append(row_output)
        progress_bar.set_postfix(success=success_count, failed=failure_count)

        if sleep_seconds > 0 and not dry_run:
            time.sleep(sleep_seconds)

    new_output_df = pd.DataFrame(output_rows)
    if not overwrite and not existing_df.empty:
        output_df = pd.concat([existing_df, new_output_df], ignore_index=True)
        output_df = output_df.drop_duplicates(subset=JOIN_KEYS, keep="last")
    else:
        output_df = new_output_df

    ensure_parent_dir(output_path)
    save_csv(output_df, output_path)

    return PowerRespectOnlyResult(
        input_path=input_path,
        output_path=output_path,
        rows_loaded=len(input_df),
        rows_to_process=len(rows_to_process),
        rows_written=len(output_df),
        success_count=success_count,
        failure_count=failure_count,
    )


def print_translation_summary(result: PowerRespectOnlyResult) -> None:
    """Print a concise run summary."""
    print("Power/respect-only translation complete")
    print(f"  input path       : {result.input_path}")
    print(f"  output path      : {result.output_path}")
    print(f"  rows loaded      : {result.rows_loaded}")
    print(f"  rows processed   : {result.rows_to_process}")
    print(f"  rows written     : {result.rows_written}")
    print(f"  success count    : {result.success_count}")
    print(f"  failure count    : {result.failure_count}")


__all__ = [
    "DEFAULT_API_KEY_ENV",
    "DEFAULT_INPUT_PATH",
    "DEFAULT_MODEL",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_PROMPT_PATH",
    "DEFAULT_PROVIDER",
    "JOIN_KEYS",
    "PowerRespectOnlyResult",
    "build_power_respect_only_prompt",
    "parse_power_respect_only_response",
    "print_translation_summary",
    "translate_power_respect_only_dataset",
]