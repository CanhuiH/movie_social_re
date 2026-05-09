

"""CLI for generating relationship-only translations.

This script runs the ablation condition that uses the current dialogue context
plus current-row relationship type and relationship evidence only. It does not
use power/respect labels, recent graph state, or aggregate graph state.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.translation.translate_relationship_only import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_INPUT_PATH,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_PROMPT_PATH,
    DEFAULT_PROVIDER,
    print_translation_summary,
    translate_relationship_only_dataset,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate relationship-only translations for ablation study."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input CSV path. Defaults to data/interim/translation_input.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path. Defaults to data/translation_eval/translation_relationship_only.csv.",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="Prompt template path. Defaults to prompts/translate_relationship_only.txt.",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["gemini", "openai"],
        help="LLM provider to use.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name to use for translation.",
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help="Environment variable containing the API key.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of rows to process.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between API calls.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate translations even if output rows already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the API; write placeholder translations instead.",
    )
    parser.add_argument(
        "--print-paths",
        action="store_true",
        help="Print resolved input, output, and prompt paths without running translation.",
    )
    return parser.parse_args()


def main() -> None:
    """Run relationship-only translation."""
    args = parse_args()

    if args.print_paths:
        print("Relationship-only translation paths")
        print(f"  input path  : {args.input_path}")
        print(f"  output path : {args.output_path}")
        print(f"  prompt path : {args.prompt_path}")
        return

    result = translate_relationship_only_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        prompt_path=args.prompt_path,
        provider=args.provider,
        model=args.model,
        api_key_env=args.api_key_env,
        max_rows=args.max_rows,
        sleep_seconds=args.sleep_seconds,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print_translation_summary(result)


if __name__ == "__main__":
    main()