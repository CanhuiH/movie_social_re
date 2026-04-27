

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from src.config import (
    DEFAULT_MOVIE_NAME,
    DEFAULT_TRANSLATION_MODEL,
    OPENAI_API_KEY_ENV,
    PROMPTS_DIR,
    ensure_project_dirs,
)
from src.utils.io import load_jsonl, print_file_summary, save_jsonl
from src.utils.paths import get_movie_processed_dir


TRANSLATION_MODES = {"context_only", "with_graph"}
DEFAULT_TRANSLATION_MODE = "with_graph"

DEFAULT_CONTEXT_ONLY_PROMPT = PROMPTS_DIR / "translation_context_only.txt"
DEFAULT_WITH_GRAPH_PROMPT = PROMPTS_DIR / "translation_with_graph.txt"

SYSTEM_PROMPT = """You are a careful movie dialogue translator.
Return only the final Mandarin translation of the current turn.
Do not add commentary, notes, or explanations.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run translation generation over prompt-ready translation input files."
        )
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=DEFAULT_MOVIE_NAME,
        help=f'Movie name used to locate movie-specific files. Default: "{DEFAULT_MOVIE_NAME}".',
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=sorted(TRANSLATION_MODES),
        default=DEFAULT_TRANSLATION_MODE,
        help=(
            "Which translation input variant to translate. "
            "Use 'context_only' or 'with_graph'. "
            f"Default: '{DEFAULT_TRANSLATION_MODE}'."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_TRANSLATION_MODEL,
        help=f'OpenAI model name for translation. Default: "{DEFAULT_TRANSLATION_MODEL}".',
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on the number of translation inputs to process.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional custom prompt template path. If omitted, the mode-specific default prompt is used.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite an existing translation output file.",
    )
    return parser.parse_args()


def get_translation_input_path(movie_name: str, mode: str) -> Path:
    processed_dir = get_movie_processed_dir(movie_name)
    return processed_dir / f"translation_inputs_{mode}.jsonl"


def get_translation_output_path(movie_name: str, mode: str) -> Path:
    processed_dir = get_movie_processed_dir(movie_name)
    return processed_dir / f"translations_{mode}.jsonl"


def get_default_prompt_path(mode: str) -> Path:
    if mode == "context_only":
        return DEFAULT_CONTEXT_ONLY_PROMPT
    return DEFAULT_WITH_GRAPH_PROMPT


def load_openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv(OPENAI_API_KEY_ENV)
    if not api_key:
        raise EnvironmentError(
            f"Missing required environment variable: {OPENAI_API_KEY_ENV}. "
            "Set it in your shell or .env file before running translation."
        )
    return OpenAI(api_key=api_key)


def load_prompt_template(mode: str, prompt_file: str | None = None) -> str:
    prompt_path = Path(prompt_file) if prompt_file else get_default_prompt_path(mode)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def maybe_limit_records(records: list[dict[str, Any]], max_rows: int | None) -> list[dict[str, Any]]:
    if max_rows is None:
        return records
    return records[:max_rows]


def build_user_prompt(record: dict[str, Any], prompt_template: str, mode: str) -> str:
    base_kwargs = {
        "movie_name": record.get("movie_name", ""),
        "conversation_id": record.get("conversation_id", ""),
        "utterance_id": record.get("utterance_id", ""),
        "speaker_name": record.get("speaker_name") or "UNKNOWN",
        "listener_name": record.get("listener_name") or "UNKNOWN",
        "context_text": record.get("context_text", "") or "[NO PREVIOUS CONTEXT]",
        "current_turn": record.get("current_turn", ""),
    }

    if mode == "with_graph":
        base_kwargs["social_summary"] = record.get("social_summary", "") or "[NO SOCIAL SUMMARY]"

    return prompt_template.format(**base_kwargs)


def call_translation_model(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.output_text.strip()


def build_output_record(input_record: dict[str, Any], translation_text: str, model: str, mode: str) -> dict[str, Any]:
    return {
        "movie_name": input_record.get("movie_name", ""),
        "conversation_id": input_record.get("conversation_id", ""),
        "utterance_id": input_record.get("utterance_id", ""),
        "speaker_id": input_record.get("speaker_id"),
        "speaker_name": input_record.get("speaker_name"),
        "listener_id": input_record.get("listener_id"),
        "listener_name": input_record.get("listener_name"),
        "current_turn": input_record.get("current_turn", ""),
        "current_text": input_record.get("current_text", ""),
        "context_text": input_record.get("context_text", ""),
        "relationship_type": input_record.get("relationship_type", "unclear"),
        "social_summary": input_record.get("social_summary", ""),
        "translation_input_mode": mode,
        "translation_model": model,
        "translation_text": translation_text,
    }


def run_translation(
    records: list[dict[str, Any]],
    client: OpenAI,
    model: str,
    prompt_template: str,
    mode: str,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []

    for record in tqdm(records, desc=f"Running translation ({mode})"):
        user_prompt = build_user_prompt(record, prompt_template, mode)
        translation_text = call_translation_model(
            client=client,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        outputs.append(build_output_record(record, translation_text, model, mode))

    return outputs


def main() -> None:
    args = parse_args()
    ensure_project_dirs(args.movie)

    input_path = get_translation_input_path(args.movie, args.mode)
    output_path = get_translation_output_path(args.movie, args.mode)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Use --overwrite to replace it."
        )

    print(f"Loading translation inputs from: {input_path}")
    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} translation input rows.")

    records = maybe_limit_records(records, args.max_rows)
    if args.max_rows is not None:
        print(f"Processing only the first {len(records)} rows due to --max-rows.")

    client = load_openai_client()
    prompt_template = load_prompt_template(args.mode, args.prompt_file)

    outputs = run_translation(
        records=records,
        client=client,
        model=args.model,
        prompt_template=prompt_template,
        mode=args.mode,
    )

    save_jsonl(outputs, output_path)

    print("\nTranslation generation complete.")
    print_file_summary(output_path, label="Saved translations")
    print(f"Rows translated: {len(outputs)}")
    print(f"Translation mode: {args.mode}")
    print(f"Model used: {args.model}")


if __name__ == "__main__":
    main()