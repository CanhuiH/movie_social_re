

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from src.config import (
    DEFAULT_MOVIE_NAME,
    DEFAULT_RE_MODEL,
    OPENAI_API_KEY_ENV,
    RELATIONSHIP_LABELS,
    RE_PROMPT_FILE,
    ensure_project_dirs,
)
from src.re.schema import coerce_confidence, normalize_relationship_label
from src.utils.io import (
    dataframe_to_records,
    load_csv,
    print_file_summary,
    save_jsonl,
)
from src.utils.paths import (
    get_movie_re_raw_output_path,
    get_movie_turn_windows_path,
)


DEFAULT_MAX_ROWS: int | None = None


SYSTEM_PROMPT = """You are a careful information extraction assistant.
Your task is to infer the relationship type between the current speaker and listener in a movie dialogue turn.
Use only the provided local dialogue context and the current turn.
Do not use outside movie knowledge.
Return JSON only.
"""


PROMPT_TEMPLATE = """You will extract the relationship type between the speaker and listener for the current turn.

Allowed relationship labels:
{relationship_labels}

Instructions:
- Choose exactly one relationship_type from the allowed labels.
- Use only the provided local dialogue context and current turn.
- If there is not enough evidence, choose \"unclear\".
- Confidence must be a number between 0 and 1.
- Evidence should be brief and grounded in the provided text.
- Return valid JSON only with the following keys:
  relationship_type, confidence, evidence

Movie: {movie_name}
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
        description="Run generative relationship extraction over turn-window dialogue examples."
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=DEFAULT_MOVIE_NAME,
        help=f'Movie name used to locate movie-specific files. Default: "{DEFAULT_MOVIE_NAME}".',
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_RE_MODEL,
        help=f'OpenAI model name for generative RE. Default: "{DEFAULT_RE_MODEL}".',
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on the number of turn-window rows to process.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=str(RE_PROMPT_FILE),
        help="Optional prompt template file path. If missing, the built-in template is used.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite an existing raw RE output file.",
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
    return OpenAI(api_key=api_key)


def load_prompt_template(prompt_file: str | Path) -> str:
    prompt_path = Path(prompt_file)
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return PROMPT_TEMPLATE


def build_user_prompt(record: dict[str, Any], prompt_template: str) -> str:
    relationship_labels_str = ", ".join(RELATIONSHIP_LABELS)
    context_text = record.get("context_text", "") or "[NO PREVIOUS CONTEXT]"
    speaker_name = record.get("speaker_name") or "UNKNOWN"
    listener_name = record.get("listener_name") or "UNKNOWN"

    return prompt_template.format(
        relationship_labels=relationship_labels_str,
        movie_name=record.get("movie_name", ""),
        conversation_id=record.get("conversation_id", ""),
        utterance_id=record.get("utterance_id", ""),
        speaker_name=speaker_name,
        listener_name=listener_name,
        context_text=context_text,
        current_turn=record.get("current_turn", ""),
    )


def supports_temperature(model: str) -> bool:
    """Return whether a model is expected to support temperature in this pipeline.

    GPT-5.5 in the Responses API currently rejects the temperature parameter, so
    this pipeline omits temperature entirely. This helper is kept for future
    extension if model-specific request shaping is needed.
    """
    model_name = model.strip().lower()
    return not model_name.startswith("gpt-5.5")


def call_openai_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> tuple[dict[str, Any], str]:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_text = response.output_text.strip()
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = {}

    return parsed if isinstance(parsed, dict) else {}, raw_text


def normalize_model_output(parsed_output: dict[str, Any], raw_text: str) -> dict[str, Any]:
    relationship_type = normalize_relationship_label(parsed_output.get("relationship_type"))
    confidence = coerce_confidence(parsed_output.get("confidence"), default=0.0)
    evidence = parsed_output.get("evidence", "")
    evidence_text = "" if evidence is None else str(evidence).strip()

    return {
        "relationship_type": relationship_type,
        "confidence": confidence,
        "evidence": evidence_text,
        "raw_response": raw_text,
    }


def build_output_record(
    input_record: dict[str, Any],
    normalized_output: dict[str, Any],
) -> dict[str, Any]:
    return {
        "movie_name": input_record.get("movie_name", ""),
        "conversation_id": input_record.get("conversation_id", ""),
        "utterance_id": input_record.get("utterance_id", ""),
        "speaker_id": input_record.get("speaker_id", ""),
        "speaker_name": input_record.get("speaker_name"),
        "listener_id": input_record.get("listener_id"),
        "listener_name": input_record.get("listener_name"),
        **normalized_output,
    }


def run_generative_re(
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
    prompt_template: str,
) -> list[dict[str, Any]]:
    records = dataframe_to_records(df)
    outputs: list[dict[str, Any]] = []

    for record in tqdm(records, desc="Running generative RE"):
        user_prompt = build_user_prompt(record, prompt_template)
        parsed_output, raw_text = call_openai_json(
            client=client,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        normalized_output = normalize_model_output(parsed_output, raw_text)
        output_record = build_output_record(record, normalized_output)
        outputs.append(output_record)

    return outputs


def maybe_limit_rows(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None:
        return df
    return df.head(max_rows).copy()


def main() -> None:
    args = parse_args()
    ensure_project_dirs(args.movie)

    input_path = get_movie_turn_windows_path(args.movie)
    output_path = get_movie_re_raw_output_path(args.movie)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Use --overwrite to replace it."
        )

    print(f"Loading turn-window data from: {input_path}")
    df = load_csv(input_path)
    print(f"Loaded {len(df)} turn-window rows.")

    if not supports_temperature(args.model):
        print(f"Model {args.model} will be called without temperature.")

    df = maybe_limit_rows(df, args.max_rows)
    if args.max_rows is not None:
        print(f"Processing only the first {len(df)} rows due to --max-rows.")

    client = load_openai_client()
    prompt_template = load_prompt_template(args.prompt_file)

    outputs = run_generative_re(
        df=df,
        client=client,
        model=args.model,
        prompt_template=prompt_template,
    )

    save_jsonl(outputs, output_path)

    print("\nGenerative relationship extraction complete.")
    print_file_summary(output_path, label="Saved raw RE outputs")
    print(f"Rows processed: {len(outputs)}")
    print(f"Model used: {args.model}")


if __name__ == "__main__":
    main()