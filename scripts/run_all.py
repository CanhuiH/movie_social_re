

"""Run the full movie social relationship translation pipeline.

This script runs the numbered pipeline scripts in order.

By default, it runs the main pipeline plus the optional ablation study:

01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09 -> 10 -> 11 -> 12 -> 14 -> 15 -> 13

Use `--no-ablation` to run only the main pipeline through Step 11:

01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09 -> 10 -> 11

The ablation merge step is run last because it needs all ablation translation
outputs to exist first.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

MAIN_PIPELINE = [
    "01_data.py",
    "02_run_all_movies_re.py",
    "03_prepare_modeling_data.py",
    "04_train_power_respect.py",
    "05_tune_logistic_regression.py",
    "06_prepare_risk_gold_overlap.py",
    "07_build_social_graph.py",
    "08_prepare_translation_input.py",
    "09_translate_context_only.py",
    "10_translate_graph_guided.py",
    "11_merge_translation_outputs.py",
]

ABLATION_PIPELINE = [
    "12_translate_social_labels_only.py",
    "14_translate_power_respect_only.py",
    "15_translate_relationship_only.py",
    "13_merge_ablation_outputs.py",
]

DEFAULT_STEP_ARGS = {
    "07_build_social_graph.py": ["--recent-window", "6"],
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the full project pipeline.")
    parser.add_argument(
        "--no-ablation",
        action="store_true",
        help="Run only the main pipeline and skip the optional ablation study scripts.",
    )
    parser.add_argument(
        "--skip-modeling",
        action="store_true",
        help="Skip Steps 03-05 for power/respect classifier modeling.",
    )
    parser.add_argument(
        "--start-at",
        default=None,
        help="Start from a specific script name, for example 06_prepare_risk_gold_overlap.py.",
    )
    parser.add_argument(
        "--end-at",
        default=None,
        help="End at a specific script name, for example 11_merge_translation_outputs.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--overwrite-step3",
        action="store_true",
        help="Pass --overwrite to Step 03 when regenerating modeling data.",
    )
    parser.add_argument(
        "--overwrite-embeddings",
        action="store_true",
        help="Pass --overwrite-embeddings to Step 04 and Step 05 when supported.",
    )
    parser.add_argument(
        "--translation-dry-run",
        action="store_true",
        help="Pass --dry-run to translation scripts 09, 10, 12, 14, and 15.",
    )
    parser.add_argument(
        "--translation-max-rows",
        type=int,
        default=None,
        help="Pass --max-rows to translation scripts 09, 10, 12, 14, and 15.",
    )
    parser.add_argument(
        "--translation-overwrite",
        action="store_true",
        help="Pass --overwrite to translation scripts 09, 10, 12, 14, and 15.",
    )
    return parser.parse_args()


def build_pipeline(no_ablation: bool, skip_modeling: bool) -> list[str]:
    """Build the script list to run."""
    pipeline = MAIN_PIPELINE.copy()
    if skip_modeling:
        pipeline = [script for script in pipeline if not script.startswith(("03_", "04_", "05_"))]
    if not no_ablation:
        pipeline.extend(ABLATION_PIPELINE)
    return pipeline


def slice_pipeline(pipeline: list[str], start_at: str | None, end_at: str | None) -> list[str]:
    """Slice pipeline by optional start/end script names."""
    if start_at is not None:
        if start_at not in pipeline:
            raise ValueError(f"start-at script is not in selected pipeline: {start_at}")
        pipeline = pipeline[pipeline.index(start_at) :]

    if end_at is not None:
        if end_at not in pipeline:
            raise ValueError(f"end-at script is not in selected pipeline: {end_at}")
        pipeline = pipeline[: pipeline.index(end_at) + 1]

    return pipeline


def build_command(script_name: str, args: argparse.Namespace) -> list[str]:
    """Build the command for a script."""
    command = [sys.executable, str(SCRIPTS_DIR / script_name)]
    command.extend(DEFAULT_STEP_ARGS.get(script_name, []))

    if script_name == "03_prepare_modeling_data.py" and args.overwrite_step3:
        command.append("--overwrite")

    if script_name in {"04_train_power_respect.py", "05_tune_logistic_regression.py"}:
        if args.overwrite_embeddings:
            command.append("--overwrite-embeddings")

    if script_name in {
        "09_translate_context_only.py",
        "10_translate_graph_guided.py",
        "12_translate_social_labels_only.py",
        "14_translate_power_respect_only.py",
        "15_translate_relationship_only.py",
    }:
        if args.translation_dry_run:
            command.append("--dry-run")
        if args.translation_overwrite:
            command.append("--overwrite")
        if args.translation_max_rows is not None:
            command.extend(["--max-rows", str(args.translation_max_rows)])

    return command


def run_command(command: list[str], dry_run: bool, env: dict[str, str]) -> None:
    """Run one command or print it in dry-run mode."""
    printable = " ".join(command)
    print(f"\n$ {printable}")
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def main() -> None:
    """Run the selected pipeline."""
    args = parse_args()

    pipeline = build_pipeline(
        no_ablation=args.no_ablation,
        skip_modeling=args.skip_modeling,
    )
    pipeline = slice_pipeline(pipeline, start_at=args.start_at, end_at=args.end_at)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not existing_pythonpath else f"{PROJECT_ROOT}:{existing_pythonpath}"

    print("Selected pipeline:")
    for script in pipeline:
        print(f"  - {script}")

    for script in pipeline:
        script_path = SCRIPTS_DIR / script
        if not script_path.exists():
            raise FileNotFoundError(f"Pipeline script not found: {script_path}")
        command = build_command(script, args)
        run_command(command, dry_run=args.dry_run, env=env)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()