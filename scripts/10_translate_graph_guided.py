

"""Command-line entry point for graph-guided Mandarin translation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.translation.translate_graph_guided import (  # noqa: E402
    DEFAULT_API_KEY_ENV,
    DEFAULT_INPUT_PATH,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_PROMPT_PATH,
    DEFAULT_PROVIDER,
    print_translation_summary,
    summarize_graph_guided_output,
    translate_graph_guided_dataset,
)
from src.utils.paths import project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate graph-guided Mandarin translations using social graph summaries."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input translation_input.csv path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output graph-guided translation CSV path.",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="Graph-guided prompt template path.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=DEFAULT_PROVIDER,
        choices=["gemini", "openai"],
        help="LLM provider used for translation.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name used for translation.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=DEFAULT_API_KEY_ENV,
        help="Environment variable that stores the API key. This can come from .env.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of rows to translate for testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing translations instead of resuming from the output file.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save progress every N translated rows.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional sleep between API calls.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the LLM. Save prompts with placeholder translations for debugging.",
    )
    parser.add_argument(
        "--print-paths",
        action="store_true",
        help="Print input/output paths before running.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """Resolve a path relative to the project root if needed."""
    if path.is_absolute():
        return path
    return project_path(path)


def main() -> None:
    """Run graph-guided translation."""
    args = parse_args()

    input_path = resolve_project_path(args.input_path)
    output_path = resolve_project_path(args.output_path)
    prompt_path = resolve_project_path(args.prompt_path)

    if args.print_paths:
        print("Graph-guided translation paths:")
        print(f"  input : {input_path}")
        print(f"  prompt: {prompt_path}")
        print(f"  output: {output_path}")

    output_df = translate_graph_guided_dataset(
        input_path=input_path,
        output_path=output_path,
        provider=args.provider,
        model=args.model,
        api_key_env=args.api_key_env,
        max_rows=args.max_rows,
        overwrite=args.overwrite,
        save_every=args.save_every,
        sleep_seconds=args.sleep_seconds,
        dry_run=args.dry_run,
        prompt_path=prompt_path,
    )
    result = summarize_graph_guided_output(
        output_df=output_df,
        output_path=output_path,
        provider=args.provider,
        model=args.model,
    )
    print_translation_summary(result)


if __name__ == "__main__":
    main()