from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

try:
    from google import genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None

from src.config import (
    DEFAULT_MOVIE_NAME,
    DEFAULT_RE_MODEL,
    LLM_MAX_RETRIES,
    LLM_REQUEST_TIMEOUT_SECONDS,
    LLM_RETRY_SLEEP_SECONDS,
    OPENAI_API_KEY_ENV,
    RE_PROMPT_FILE,
    SETTINGS,
    ensure_project_dirs,
    format_priority_rules,
    format_relationship_definitions,
    format_relationship_labels,
    get_movie_schema_focus,
)
from src.re.schema import build_relation_prediction, coerce_confidence, normalize_relationship_label
from src.utils.io import (
    dataframe_to_records,
    load_csv,
    print_file_summary,
    read_text,
    require_columns,
    save_csv,
)
from src.utils.paths import (
    get_movie_re_clean_output_path,
    get_movie_re_output_path,
    get_movie_turn_windows_path,
)


DEFAULT_MAX_ROWS: int | None = None
DEFAULT_RE_PROVIDER = str(SETTINGS.get("default_re_provider", "openai")).strip().lower()
GEMINI_API_KEY_ENV = str(SETTINGS.get("gemini_api_key_env", "GEMINI_API_KEY"))
SUPPORTED_PROVIDERS = {"openai", "gemini"}
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

ROW_KEY_COLUMNS = ["movie_name", "conversation_id", "utterance_id"]

REQUIRED_TURN_WINDOW_COLUMNS = {
    "movie_name",
    "conversation_id",
    "utterance_id",
    "speaker_id",
    "speaker_name",
    "listener_id",
    "listener_name",
    "current_turn",
    "current_text",
    "context_text",
}

RE_OUTPUT_COLUMNS = [
    "movie_name",
    "movie_idx",
    "conversation_id",
    "utterance_id",
    "timestamp",
    "turn_index",
    "utterance_order",
    "speaker_id",
    "speaker_name",
    "reply_to",
    "listener_id",
    "listener_name",
    "current_turn",
    "current_text",
    "context_text",
    "relationship_type",
    "global_category",
    "confidence",
    "evidence",
    "status",
    "parse_success",
    "error",
    "raw_response",
]

SYSTEM_PROMPT = """You are a careful information extraction assistant.
Use only the provided dialogue context.
Do not use outside movie knowledge.
Return valid JSON only.
"""

FALLBACK_PROMPT_TEMPLATE = """You are a careful information extraction assistant.

Your task is to identify the relationship type between the current speaker and listener for the current turn in a movie dialogue.

Use only the provided local dialogue context and the current turn.
Do not use outside movie knowledge.
Do not invent facts not supported by the provided text.
If the evidence is insufficient, return "unclear".

Movie:
{movie_name}

Schema focus for this movie:
{schema_focus}

Allowed relationship labels:
{relationship_labels}

Relationship label definitions:
{relationship_definitions}

Priority rules:
{priority_rules}

Decision rules:
- Choose exactly one relationship_type from the allowed labels.
- The relationship is directional: infer the relationship from Speaker to Listener for the current turn.
- Use the current turn as the target and the previous turns only as supporting context.
- If multiple labels seem possible, choose the one most directly supported by the provided text.
- Prefer the movie-specific label definition and priority rules over generic intuition.
- Do not infer long-term plot facts unless they are clearly expressed in the provided text.
- If there is not enough local evidence, return "unclear".
- Confidence must be a number between 0 and 1.
- Evidence must be brief and grounded in the provided text.
- Return valid JSON only.
- Do not include markdown.
- Do not include any explanation outside the JSON object.

Required JSON format:
{{
  "relationship_type": "...",
  "confidence": 0.0,
  "evidence": "..."
}}

Now perform the task on the following target turn.

Conversation ID: {conversation_id}
Utterance ID: {utterance_id}
Speaker: {speaker_name}
Listener: {listener_name}

Previous turns:
{context_text}

Current turn:
{current_turn}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generative relationship extraction over turn-window dialogue examples and save CSV output."
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=DEFAULT_MOVIE_NAME,
        help=f'Movie name used to locate movie-specific files. Default: "{DEFAULT_MOVIE_NAME}".',
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
        default=DEFAULT_MAX_ROWS,
        help="Optional limit on the number of turn-window rows to process.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=str(RE_PROMPT_FILE),
        help="Prompt template file path. Falls back to an internal template if missing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite an existing CSV RE output file and rerun from scratch.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, reuse an existing CSV output and skip rows with status == success.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save progress every N newly attempted rows. Default: 1.",
    )
    return parser.parse_args()


def load_openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv(OPENAI_API_KEY_ENV)
    if not api_key:
        raise EnvironmentError(
            f"Missing required environment variable: {OPENAI_API_KEY_ENV}. "
            "Set it in your shell or .env file before running generative RE."
        )
    return OpenAI(api_key=api_key, timeout=LLM_REQUEST_TIMEOUT_SECONDS)


def load_gemini_client():
    """Load a Google Gemini client using the configured Gemini API key env var."""
    load_dotenv()
    if genai is None:
        raise ImportError(
            "google-genai is not installed. Install it with: pip install google-genai"
        )

    api_key = os.getenv(GEMINI_API_KEY_ENV)
    if not api_key:
        raise EnvironmentError(
            f"Missing required environment variable: {GEMINI_API_KEY_ENV}. "
            "Set it in your shell or .env file before running Gemini RE."
        )
    return genai.Client(api_key=api_key)


def load_llm_client(provider: str):
    """Load the client for the selected LLM provider."""
    normalized_provider = provider.strip().lower()
    if normalized_provider == "openai":
        return load_openai_client()
    if normalized_provider == "gemini":
        return load_gemini_client()
    raise ValueError(f"Unsupported provider: {provider}. Supported providers: {sorted(SUPPORTED_PROVIDERS)}")


def load_prompt_template(prompt_file: str | Path) -> str:
    prompt_path = Path(prompt_file)
    if prompt_path.exists():
        return read_text(prompt_path)
    print(f"Prompt file not found: {prompt_path}. Using fallback prompt template.")
    return FALLBACK_PROMPT_TEMPLATE


def normalize_prompt_value(value: object, fallback: str = "UNKNOWN") -> str:
    if value is None or pd.isna(value):
        return fallback
    text = " ".join(str(value).strip().split())
    return text if text else fallback


def get_record_movie_name(record: dict[str, Any], default_movie: str) -> str:
    movie_name = normalize_prompt_value(record.get("movie_name"), fallback=default_movie)
    return movie_name.strip().lower()


def build_row_key(record: dict[str, Any], default_movie: str) -> tuple[str, str, str]:
    return (
        get_record_movie_name(record, default_movie=default_movie),
        normalize_prompt_value(record.get("conversation_id"), fallback=""),
        normalize_prompt_value(record.get("utterance_id"), fallback=""),
    )


def build_row_key_from_series(row: pd.Series, default_movie: str) -> tuple[str, str, str]:
    return (
        normalize_prompt_value(row.get("movie_name"), fallback=default_movie).strip().lower(),
        normalize_prompt_value(row.get("conversation_id"), fallback=""),
        normalize_prompt_value(row.get("utterance_id"), fallback=""),
    )


def build_user_prompt(
    record: dict[str, Any],
    prompt_template: str,
    default_movie: str,
) -> str:
    movie_name = get_record_movie_name(record, default_movie=default_movie)
    context_text = record.get("context_text", "")
    context_text = context_text if isinstance(context_text, str) and context_text.strip() else "[NO PREVIOUS CONTEXT]"

    speaker_name = normalize_prompt_value(record.get("speaker_name"))
    listener_name = normalize_prompt_value(record.get("listener_name"))

    return prompt_template.format(
        movie_name=movie_name,
        schema_focus=get_movie_schema_focus(movie_name),
        relationship_labels=format_relationship_labels(movie_name),
        relationship_definitions=format_relationship_definitions(movie_name),
        priority_rules=format_priority_rules(movie_name),
        conversation_id=normalize_prompt_value(record.get("conversation_id"), fallback=""),
        utterance_id=normalize_prompt_value(record.get("utterance_id"), fallback=""),
        speaker_name=speaker_name,
        listener_name=listener_name,
        context_text=context_text,
        current_turn=record.get("current_turn", ""),
    )


def should_omit_temperature(model: str, provider: str = "openai") -> bool:
    """Return True for models where this pipeline should not send temperature."""
    provider_name = provider.strip().lower()
    model_name = model.strip().lower()
    return provider_name == "openai" and model_name.startswith("gpt-5")


def parse_json_response(raw_text: str) -> dict[str, Any]:
    """Parse a JSON object from model text output."""
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    if text.startswith("```"):
        text = text.removeprefix("```").strip()
    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def call_openai_json_once(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> tuple[dict[str, Any], str]:
    request: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    response = client.responses.create(**request)
    raw_text = response.output_text.strip()
    return parse_json_response(raw_text), raw_text


def call_gemini_json_once(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> tuple[dict[str, Any], str]:
    """Call Gemini once and parse a JSON object from the response text."""
    full_prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
    )
    raw_text = (getattr(response, "text", None) or "").strip()
    return parse_json_response(raw_text), raw_text


def call_llm_json(
    client: Any,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> tuple[dict[str, Any], str]:
    """Call the selected LLM provider with retry handling."""
    normalized_provider = provider.strip().lower()
    last_error: Exception | None = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            if normalized_provider == "openai":
                return call_openai_json_once(
                    client=client,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            if normalized_provider == "gemini":
                return call_gemini_json_once(
                    client=client,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            raise ValueError(
                f"Unsupported provider: {provider}. Supported providers: {sorted(SUPPORTED_PROVIDERS)}"
            )
        except Exception as exc:
            last_error = exc
            if attempt >= LLM_MAX_RETRIES:
                break
            sleep_seconds = LLM_RETRY_SLEEP_SECONDS * attempt
            print(
                f"{normalized_provider} request failed on attempt {attempt}/{LLM_MAX_RETRIES}: {exc}. "
                f"Retrying in {sleep_seconds}s."
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(
        f"{normalized_provider} request failed after {LLM_MAX_RETRIES} attempts: {last_error}"
    )


def normalize_model_output(
    parsed_output: dict[str, Any],
    raw_text: str,
    movie_name: str,
) -> dict[str, Any]:
    relationship_type = normalize_relationship_label(
        parsed_output.get("relationship_type"),
        movie_name=movie_name,
    )
    confidence = coerce_confidence(parsed_output.get("confidence"), default=0.0)
    evidence = parsed_output.get("evidence", "")
    evidence_text = "" if evidence is None else str(evidence).strip()
    parse_success = bool(parsed_output)

    return {
        "relationship_type": relationship_type,
        "confidence": confidence,
        "evidence": evidence_text,
        "parse_success": parse_success,
        "raw_response": raw_text,
    }


def build_output_record(
    input_record: dict[str, Any],
    normalized_output: dict[str, Any],
    default_movie: str,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    movie_name = get_record_movie_name(input_record, default_movie=default_movie)

    prediction = build_relation_prediction(
        movie_name=movie_name,
        conversation_id=normalize_prompt_value(input_record.get("conversation_id"), fallback=""),
        utterance_id=normalize_prompt_value(input_record.get("utterance_id"), fallback=""),
        speaker_id=normalize_prompt_value(input_record.get("speaker_id"), fallback=""),
        speaker_name=input_record.get("speaker_name"),
        listener_id=input_record.get("listener_id"),
        listener_name=input_record.get("listener_name"),
        relationship_type=normalized_output.get("relationship_type"),
        confidence=normalized_output.get("confidence"),
        evidence=normalized_output.get("evidence"),
        raw_response=normalized_output.get("raw_response"),
    )

    output = {
        "movie_name": movie_name,
        "movie_idx": input_record.get("movie_idx"),
        "conversation_id": prediction.conversation_id,
        "utterance_id": prediction.utterance_id,
        "timestamp": input_record.get("timestamp"),
        "turn_index": input_record.get("turn_index"),
        "utterance_order": input_record.get("utterance_order"),
        "speaker_id": prediction.speaker_id,
        "speaker_name": prediction.speaker_name,
        "reply_to": input_record.get("reply_to"),
        "listener_id": prediction.listener_id,
        "listener_name": prediction.listener_name,
        "current_turn": input_record.get("current_turn", ""),
        "current_text": input_record.get("current_text", ""),
        "context_text": input_record.get("context_text", ""),
        "relationship_type": prediction.relationship_type,
        "global_category": prediction.global_category,
        "confidence": prediction.confidence,
        "evidence": prediction.evidence,
        "status": status,
        "parse_success": bool(normalized_output.get("parse_success", False)),
        "error": error,
        "raw_response": prediction.raw_response,
    }
    return output


def build_failed_output_record(
    input_record: dict[str, Any],
    default_movie: str,
    error: Exception,
) -> dict[str, Any]:
    return build_output_record(
        input_record=input_record,
        normalized_output={
            "relationship_type": "unclear",
            "confidence": 0.0,
            "evidence": "",
            "parse_success": False,
            "raw_response": "",
        },
        default_movie=default_movie,
        status=STATUS_FAILED,
        error=str(error),
    )


def validate_turn_window_dataframe(df: pd.DataFrame) -> None:
    require_columns(df, REQUIRED_TURN_WINDOW_COLUMNS, label="Turn-window data")


def standardize_output_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()
    for column in RE_OUTPUT_COLUMNS:
        if column not in output_df.columns:
            output_df[column] = None
    return output_df[RE_OUTPUT_COLUMNS]


def sort_output_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    output_df = standardize_output_dataframe(df)
    sort_cols = [
        column
        for column in ["movie_name", "conversation_id", "turn_index", "utterance_order", "utterance_id"]
        if column in output_df.columns
    ]
    if sort_cols:
        output_df = output_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return output_df


def load_existing_success_records(
    output_path: str | Path,
    default_movie: str,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    path = Path(output_path)
    if not path.exists():
        return {}

    existing_df = load_csv(path)
    existing_success: dict[tuple[str, str, str], dict[str, Any]] = {}
    for _, row in existing_df.iterrows():
        status = normalize_prompt_value(row.get("status"), fallback="").lower()
        if status != STATUS_SUCCESS:
            continue
        row_dict = row.to_dict()
        key = build_row_key_from_series(row, default_movie=default_movie)
        existing_success[key] = row_dict
    return existing_success


def save_progress(records: list[dict[str, Any]], output_path: str | Path) -> None:
    output_df = sort_output_dataframe(pd.DataFrame(records))
    save_csv(output_df, output_path, index=False)


def run_generative_re(
    df: pd.DataFrame,
    client: Any,
    provider: str,
    model: str,
    prompt_template: str,
    default_movie: str,
    existing_success_records: dict[tuple[str, str, str], dict[str, Any]] | None = None,
    output_path: str | Path | None = None,
    save_every: int = 1,
) -> list[dict[str, Any]]:
    validate_turn_window_dataframe(df)

    if save_every <= 0:
        raise ValueError("save_every must be positive.")

    records = dataframe_to_records(df)
    existing_success_records = existing_success_records or {}
    outputs: list[dict[str, Any]] = []
    attempted_since_save = 0

    for record in tqdm(records, desc="Running generative RE"):
        key = build_row_key(record, default_movie=default_movie)
        if key in existing_success_records:
            previous_record = dict(existing_success_records[key])
            previous_record["status"] = STATUS_SUCCESS
            outputs.append(previous_record)
            continue

        try:
            movie_name = get_record_movie_name(record, default_movie=default_movie)
            user_prompt = build_user_prompt(
                record=record,
                prompt_template=prompt_template,
                default_movie=default_movie,
            )
            parsed_output, raw_text = call_llm_json(
                client=client,
                provider=provider,
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            normalized_output = normalize_model_output(
                parsed_output=parsed_output,
                raw_text=raw_text,
                movie_name=movie_name,
            )
            output_record = build_output_record(
                input_record=record,
                normalized_output=normalized_output,
                default_movie=default_movie,
                status=STATUS_SUCCESS,
                error=None,
            )
        except Exception as exc:
            output_record = build_failed_output_record(
                input_record=record,
                default_movie=default_movie,
                error=exc,
            )

        outputs.append(output_record)
        attempted_since_save += 1

        if output_path is not None and attempted_since_save >= save_every:
            save_progress(outputs, output_path)
            attempted_since_save = 0

    if output_path is not None:
        save_progress(outputs, output_path)

    return outputs


def maybe_limit_rows(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None:
        return df
    if max_rows < 0:
        raise ValueError("--max-rows must be non-negative.")
    return df.head(max_rows).copy()


def main() -> None:
    args = parse_args()
    movie_name = args.movie.strip().lower()
    ensure_project_dirs(movie_name)

    if args.overwrite and args.resume:
        raise ValueError("Use either --overwrite or --resume, not both.")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive.")

    input_path = get_movie_turn_windows_path(movie_name)
    output_path = get_movie_re_output_path(movie_name)

    if output_path.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Use --resume to skip successful rows or --overwrite to replace it."
        )

    print(f"Loading turn-window data from: {input_path}")
    df = load_csv(input_path)
    print(f"Loaded {len(df)} turn-window rows.")

    provider = args.provider.strip().lower()
    if should_omit_temperature(args.model, provider=provider):
        print(f"Model {args.model} will be called without temperature.")

    df = maybe_limit_rows(df, args.max_rows)
    if args.max_rows is not None:
        print(f"Processing only the first {len(df)} rows due to --max-rows.")

    existing_success_records = {}
    if args.resume:
        existing_success_records = load_existing_success_records(
            output_path=output_path,
            default_movie=movie_name,
        )
        print(f"Resume mode: found {len(existing_success_records)} existing successful rows to skip.")

    client = load_llm_client(provider)
    prompt_template = load_prompt_template(args.prompt_file)

    outputs = run_generative_re(
        df=df,
        client=client,
        provider=provider,
        model=args.model,
        prompt_template=prompt_template,
        default_movie=movie_name,
        existing_success_records=existing_success_records,
        output_path=output_path,
        save_every=args.save_every,
    )

    print("\nGenerative relationship extraction complete.")
    print_file_summary(output_path, label="Saved CSV RE output")
    print(f"Rows in output: {len(outputs)}")
    print(f"Model used: {args.model}")
    print(f"Provider used: {provider}")


if __name__ == "__main__":
    main()