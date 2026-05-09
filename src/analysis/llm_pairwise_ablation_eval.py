from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import OPENAI_API_KEY_ENV, PROMPTS_DIR
from src.utils.io import load_csv, print_file_summary, save_csv, save_json, save_jsonl
from src.utils.paths import project_path

DEFAULT_INPUT_PATH = project_path(Path("data") / "translation_eval" / "translation_ablation_comparison.csv")
DEFAULT_OUTPUT_ROOT = project_path(Path("outputs") / "translation_eval")
DEFAULT_PROMPT_PATH = PROMPTS_DIR / "translation_ablation_pairwise_judge.txt"
DEFAULT_EVAL_MODEL = "gpt-4.1-mini"
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_RANDOM_SEED = 7
DEFAULT_MAX_RETRIES = 3
DEFAULT_GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

SYSTEM_PROMPT = """You are a careful bilingual evaluator for movie dialogue translation.
You compare two candidate Chinese translations against the English source and return only strict JSON.
"""

WINNER_VALUES = {"A", "B", "tie"}
CONFIDENCE_VALUES = {"low", "medium", "high"}
CRITERIA = [
    "meaning_accuracy",
    "social_relationship",
    "register_tone",
    "fluency",
]


@dataclass(frozen=True)
class EvalPaths:
    output_dir: Path
    judgments_jsonl: Path
    judgments_csv: Path
    summary_json: Path
    summary_csv: Path
    relationship_csv: Path
    power_csv: Path
    respect_csv: Path
    criteria_csv: Path
    prompt_snapshot: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pairwise LLM-as-a-judge evaluation on any two translation columns in the ablation CSV."
    )
    parser.add_argument("--input-file", type=str, default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_EVAL_MODEL)
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "gemini"])
    parser.add_argument("--api-key-env", type=str, default=None)
    parser.add_argument("--candidate-1-column", type=str, required=True)
    parser.add_argument("--candidate-2-column", type=str, required=True)
    parser.add_argument("--candidate-1-label", type=str, required=True)
    parser.add_argument("--candidate-2-label", type=str, required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--only-different",
        action="store_true",
        help="Only evaluate rows where the two candidate translations are not identical after stripping whitespace.",
    )
    return parser.parse_args()


def build_default_output_dir(candidate_1_label: str, candidate_2_label: str) -> Path:
    safe_1 = candidate_1_label.strip().lower().replace(" ", "_")
    safe_2 = candidate_2_label.strip().lower().replace(" ", "_")
    return DEFAULT_OUTPUT_ROOT / f"{safe_1}_vs_{safe_2}"


def build_eval_paths(output_dir: Path) -> EvalPaths:
    return EvalPaths(
        output_dir=output_dir,
        judgments_jsonl=output_dir / "judgments.jsonl",
        judgments_csv=output_dir / "judgments.csv",
        summary_json=output_dir / "summary.json",
        summary_csv=output_dir / "summary_overall.csv",
        relationship_csv=output_dir / "summary_by_relationship.csv",
        power_csv=output_dir / "summary_by_power.csv",
        respect_csv=output_dir / "summary_by_respect.csv",
        criteria_csv=output_dir / "summary_by_criterion.csv",
        prompt_snapshot=output_dir / "judge_prompt_snapshot.txt",
    )


def make_row_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        normalize_text(record.get("movie_name", "")).lower(),
        normalize_text(record.get("conversation_id", "")),
        normalize_text(record.get("utterance_id", "")),
    )


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def validate_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing_columns)}")


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if overwrite and output_dir.exists() and any(output_dir.iterdir()):
        return
    output_dir.mkdir(parents=True, exist_ok=True)


def load_llm_client(provider: str, api_key_env: str | None = None) -> Any:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv()

    provider_name = provider.strip().lower()
    resolved_api_key_env = (
        api_key_env or (OPENAI_API_KEY_ENV if provider_name == "openai" else DEFAULT_GEMINI_API_KEY_ENV)
    )
    api_key = os.getenv(resolved_api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Missing required environment variable: {resolved_api_key_env}. "
            "Set it in your shell or .env file before running evaluation."
        )

    if provider_name == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The `openai` package is required to run OpenAI evaluation. Install project dependencies first."
            ) from exc
        return OpenAI(api_key=api_key)

    if provider_name == "gemini":
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "The `google-genai` package is required to run Gemini evaluation. Install project dependencies first."
            ) from exc
        return genai.Client(api_key=api_key)

    raise ValueError(f"Unsupported provider: {provider}")


def load_prompt_template(prompt_file: str | None) -> str:
    prompt_path = Path(prompt_file) if prompt_file else DEFAULT_PROMPT_PATH
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def load_existing_judgments(judgments_csv_path: Path, overwrite: bool) -> pd.DataFrame:
    if overwrite or not judgments_csv_path.exists():
        return pd.DataFrame()
    return load_csv(judgments_csv_path)


def get_completed_row_keys(existing_df: pd.DataFrame) -> set[tuple[str, str, str]]:
    if existing_df.empty:
        return set()
    required = {"movie_name", "conversation_id", "utterance_id", "winner_label"}
    if not required.issubset(existing_df.columns):
        return set()
    completed = existing_df.loc[existing_df["winner_label"].astype(str).str.strip() != ""].copy()
    keys: set[tuple[str, str, str]] = set()
    for _, row in completed.iterrows():
        keys.add(
            (
                normalize_text(row.get("movie_name", "")).lower(),
                normalize_text(row.get("conversation_id", "")),
                normalize_text(row.get("utterance_id", "")),
            )
        )
    return keys


def load_eval_dataframe(
    path: Path,
    *,
    candidate_1_column: str,
    candidate_2_column: str,
    max_rows: int | None,
    only_different: bool,
    completed_row_keys: set[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    df = load_csv(path)
    required_columns = {
        "movie_name",
        "conversation_id",
        "utterance_id",
        "speaker_name",
        "listener_name",
        "current_text",
        "context_6_turns",
        "final_power_dynamic",
        "final_respect_level",
        "relationship_type",
        "evidence",
        "recent_power_dynamic",
        "recent_respect_level",
        "recent_relationship_type",
        "aggregate_power_dynamic",
        "aggregate_respect_level",
        "aggregate_relationship_type",
        candidate_1_column,
        candidate_2_column,
    }
    validate_columns(df, required_columns)

    for column in required_columns:
        df[column] = df[column].apply(normalize_text)

    filtered_df = df.loc[
        df["current_text"].ne("") & df[candidate_1_column].ne("") & df[candidate_2_column].ne("")
    ].copy()

    filtered_df["translations_identical"] = (
        filtered_df[candidate_1_column].str.strip() == filtered_df[candidate_2_column].str.strip()
    )

    if only_different:
        filtered_df = filtered_df.loc[~filtered_df["translations_identical"]].copy()

    if completed_row_keys:
        filtered_df = filtered_df.loc[
            ~filtered_df.apply(lambda row: make_row_key(row.to_dict()) in completed_row_keys, axis=1)
        ].copy()

    if max_rows is not None:
        filtered_df = filtered_df.head(max_rows).copy()

    return filtered_df.reset_index(drop=True)


def build_user_prompt(
    record: dict[str, Any],
    prompt_template: str,
    *,
    candidate_a_label: str,
    candidate_b_label: str,
    translation_a: str,
    translation_b: str,
) -> str:
    return prompt_template.format(
        movie_name=record.get("movie_name", ""),
        conversation_id=record.get("conversation_id", ""),
        utterance_id=record.get("utterance_id", ""),
        speaker_name=record.get("speaker_name", "") or "UNKNOWN",
        listener_name=record.get("listener_name", "") or "UNKNOWN",
        context_text=record.get("context_6_turns", "") or "[NO PREVIOUS CONTEXT]",
        current_text=record.get("current_text", ""),
        final_power_dynamic=record.get("final_power_dynamic", "") or "unclear",
        final_respect_level=record.get("final_respect_level", "") or "unclear",
        relationship_type=record.get("relationship_type", "") or "unclear",
        evidence=record.get("evidence", "") or "[NO EVIDENCE]",
        recent_power_dynamic=record.get("recent_power_dynamic", "") or "unclear",
        recent_respect_level=record.get("recent_respect_level", "") or "unclear",
        recent_relationship_type=record.get("recent_relationship_type", "") or "unclear",
        aggregate_power_dynamic=record.get("aggregate_power_dynamic", "") or "unclear",
        aggregate_respect_level=record.get("aggregate_respect_level", "") or "unclear",
        aggregate_relationship_type=record.get("aggregate_relationship_type", "") or "unclear",
        candidate_a_label=candidate_a_label,
        candidate_b_label=candidate_b_label,
        translation_a=translation_a,
        translation_b=translation_b,
    )


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start_index = stripped.find("{")
    end_index = stripped.rfind("}")
    if start_index == -1 or end_index == -1 or end_index <= start_index:
        raise ValueError("Judge response did not contain a JSON object.")

    candidate = stripped[start_index : end_index + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Judge JSON response was not an object.")
    return parsed


def normalize_winner(value: Any) -> str:
    normalized = normalize_text(value).lower()
    if normalized == "a":
        normalized = "A"
    elif normalized == "b":
        normalized = "B"
    if normalized not in WINNER_VALUES:
        raise ValueError(f"Unexpected winner value: {value}")
    return normalized


def normalize_confidence(value: Any) -> str:
    normalized = normalize_text(value).lower()
    if normalized not in CONFIDENCE_VALUES:
        raise ValueError(f"Unexpected confidence value: {value}")
    return normalized


def normalize_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = normalize_text(value).lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0", ""}:
        return False
    return bool(value)


def parse_judge_response(response_text: str) -> dict[str, Any]:
    payload = extract_json_object(response_text)
    winner = normalize_winner(payload.get("winner"))
    confidence = normalize_confidence(payload.get("confidence"))
    reason = normalize_text(payload.get("reason"))

    raw_criteria = payload.get("criterion_winners")
    if not isinstance(raw_criteria, dict):
        raise ValueError("Judge response is missing criterion_winners.")

    criterion_winners: dict[str, str] = {}
    for criterion in CRITERIA:
        criterion_winners[criterion] = normalize_winner(raw_criteria.get(criterion))

    issue_focus_observed = normalize_boolean(payload.get("issue_focus_observed"))

    return {
        "winner": winner,
        "confidence": confidence,
        "reason": reason,
        "criterion_winners": criterion_winners,
        "issue_focus_observed": issue_focus_observed,
        "raw_response": response_text,
    }


def call_judge_model(
    client: Any,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    total_attempts = max(1, max_retries)

    for attempt in range(1, total_attempts + 1):
        try:
            provider_name = provider.strip().lower()
            if provider_name == "openai":
                response = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                raw_text = response.output_text
            elif provider_name == "gemini":
                response = client.models.generate_content(
                    model=model,
                    contents=f"{system_prompt.strip()}\n\n{user_prompt.strip()}",
                )
                raw_text = getattr(response, "text", None) or ""
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            return parse_judge_response(raw_text)
        except Exception as exc:
            last_error = exc
            if attempt == total_attempts:
                break

    assert last_error is not None
    raise RuntimeError(f"Judge call failed after {total_attempts} attempts.") from last_error


def map_choice_to_label(choice: str, candidate_a_label: str, candidate_b_label: str) -> str:
    if choice == "A":
        return candidate_a_label
    if choice == "B":
        return candidate_b_label
    return "tie"


def build_judgment_record(
    record: dict[str, Any],
    *,
    row_index: int,
    candidate_a_label: str,
    candidate_b_label: str,
    candidate_a_column: str,
    candidate_b_column: str,
    judge_result: dict[str, Any],
) -> dict[str, Any]:
    winner_label = map_choice_to_label(
        judge_result["winner"],
        candidate_a_label=candidate_a_label,
        candidate_b_label=candidate_b_label,
    )

    result: dict[str, Any] = {
        "row_index": row_index,
        "movie_name": record.get("movie_name", ""),
        "conversation_id": record.get("conversation_id", ""),
        "utterance_id": record.get("utterance_id", ""),
        "speaker_name": record.get("speaker_name", ""),
        "listener_name": record.get("listener_name", ""),
        "current_text": record.get("current_text", ""),
        "context_6_turns": record.get("context_6_turns", ""),
        "final_power_dynamic": record.get("final_power_dynamic", ""),
        "final_respect_level": record.get("final_respect_level", ""),
        "relationship_type": record.get("relationship_type", ""),
        "evidence": record.get("evidence", ""),
        "recent_power_dynamic": record.get("recent_power_dynamic", ""),
        "recent_respect_level": record.get("recent_respect_level", ""),
        "recent_relationship_type": record.get("recent_relationship_type", ""),
        "aggregate_power_dynamic": record.get("aggregate_power_dynamic", ""),
        "aggregate_respect_level": record.get("aggregate_respect_level", ""),
        "aggregate_relationship_type": record.get("aggregate_relationship_type", ""),
        "candidate_a_label": candidate_a_label,
        "candidate_b_label": candidate_b_label,
        "candidate_a_column": candidate_a_column,
        "candidate_b_column": candidate_b_column,
        "candidate_a_text": record.get(candidate_a_column, ""),
        "candidate_b_text": record.get(candidate_b_column, ""),
        "winner_letter": judge_result["winner"],
        "winner_label": winner_label,
        "confidence": judge_result["confidence"],
        "reason": judge_result["reason"],
        "issue_focus_observed": judge_result["issue_focus_observed"],
        "translations_identical": bool(record.get("translations_identical", False)),
        "raw_judge_response": judge_result["raw_response"],
    }

    for criterion, choice in judge_result["criterion_winners"].items():
        result[f"{criterion}_winner_letter"] = choice
        result[f"{criterion}_winner_label"] = map_choice_to_label(
            choice,
            candidate_a_label=candidate_a_label,
            candidate_b_label=candidate_b_label,
        )

    return result


def run_pairwise_evaluation(
    df: pd.DataFrame,
    *,
    client: Any,
    provider: str,
    model: str,
    prompt_template: str,
    candidate_1_column: str,
    candidate_2_column: str,
    candidate_1_label: str,
    candidate_2_label: str,
    seed: int,
    max_retries: int,
    sleep_seconds: float,
    save_every: int,
    eval_paths: EvalPaths,
    existing_judgments_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []
    if existing_judgments_df is not None and not existing_judgments_df.empty:
        results.extend(existing_judgments_df.to_dict(orient="records"))

    new_results = 0
    for row_index, row in tqdm(
        list(df.iterrows()),
        total=len(df),
        desc="Running LLM pairwise evaluation",
    ):
        record = row.to_dict()

        if rng.random() < 0.5:
            candidate_a_column = candidate_1_column
            candidate_b_column = candidate_2_column
            candidate_a_label = candidate_1_label
            candidate_b_label = candidate_2_label
        else:
            candidate_a_column = candidate_2_column
            candidate_b_column = candidate_1_column
            candidate_a_label = candidate_2_label
            candidate_b_label = candidate_1_label

        translation_a = record[candidate_a_column]
        translation_b = record[candidate_b_column]

        user_prompt = build_user_prompt(
            record=record,
            prompt_template=prompt_template,
            candidate_a_label=candidate_a_label,
            candidate_b_label=candidate_b_label,
            translation_a=translation_a,
            translation_b=translation_b,
        )
        judge_result = call_judge_model(
            client=client,
            provider=provider,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_retries=max_retries,
        )
        results.append(
            build_judgment_record(
                record=record,
                row_index=row_index,
                candidate_a_label=candidate_a_label,
                candidate_b_label=candidate_b_label,
                candidate_a_column=candidate_a_column,
                candidate_b_column=candidate_b_column,
                judge_result=judge_result,
            )
        )
        new_results += 1

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        if save_every > 0 and new_results % save_every == 0:
            checkpoint_df = pd.DataFrame(results)
            save_jsonl(results, eval_paths.judgments_jsonl)
            save_csv(checkpoint_df, eval_paths.judgments_csv)

    return results


def summarize_group(
    df: pd.DataFrame,
    *,
    candidate_1_label: str,
    candidate_2_label: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    n = int(len(df))
    wins_1 = int((df["winner_label"] == candidate_1_label).sum())
    wins_2 = int((df["winner_label"] == candidate_2_label).sum())
    ties = int((df["winner_label"] == "tie").sum())
    decided = wins_1 + wins_2

    summary: dict[str, Any] = {
        "n": n,
        f"{candidate_1_label}_wins": wins_1,
        f"{candidate_2_label}_wins": wins_2,
        "ties": ties,
        "decided": decided,
        f"{candidate_1_label}_win_rate_all": wins_1 / n if n else None,
        f"{candidate_2_label}_win_rate_all": wins_2 / n if n else None,
        f"{candidate_2_label}_win_rate_decided": wins_2 / decided if decided else None,
        f"net_preference_{candidate_2_label}_minus_{candidate_1_label}": (wins_2 - wins_1) / n if n else None,
        "issue_focus_observed_rate": float(df["issue_focus_observed"].mean()) if n else None,
    }

    if not n:
        summary.update(
            {
                f"{candidate_2_label}_win_rate_all_ci_low": None,
                f"{candidate_2_label}_win_rate_all_ci_high": None,
                f"net_preference_{candidate_2_label}_minus_{candidate_1_label}_ci_low": None,
                f"net_preference_{candidate_2_label}_minus_{candidate_1_label}_ci_high": None,
            }
        )
        return summary

    scores = np.where(
        df["winner_label"].to_numpy() == candidate_2_label,
        1.0,
        np.where(df["winner_label"].to_numpy() == candidate_1_label, -1.0, 0.0),
    )
    wins_2_binary = (df["winner_label"] == candidate_2_label).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, n, size=(bootstrap_samples, n))
    sampled_scores = scores[sampled_indices]
    sampled_wins = wins_2_binary[sampled_indices]

    win_rate_samples = sampled_wins.mean(axis=1)
    net_preference_samples = sampled_scores.mean(axis=1)

    summary[f"{candidate_2_label}_win_rate_all_ci_low"] = float(np.quantile(win_rate_samples, 0.025))
    summary[f"{candidate_2_label}_win_rate_all_ci_high"] = float(np.quantile(win_rate_samples, 0.975))
    summary[f"net_preference_{candidate_2_label}_minus_{candidate_1_label}_ci_low"] = float(
        np.quantile(net_preference_samples, 0.025)
    )
    summary[f"net_preference_{candidate_2_label}_minus_{candidate_1_label}_ci_high"] = float(
        np.quantile(net_preference_samples, 0.975)
    )
    return summary


def build_group_summary(
    df: pd.DataFrame,
    group_column: str,
    *,
    candidate_1_label: str,
    candidate_2_label: str,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(group_column, dropna=False)
    for offset, (group_name, group_df) in enumerate(grouped):
        row = {group_column: group_name}
        row.update(
            summarize_group(
                group_df,
                candidate_1_label=candidate_1_label,
                candidate_2_label=candidate_2_label,
                bootstrap_samples=bootstrap_samples,
                seed=seed + offset,
            )
        )
        rows.append(row)
    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df
    return result_df.sort_values(["n", group_column], ascending=[False, True]).reset_index(drop=True)


def build_criterion_summary(df: pd.DataFrame, *, candidate_1_label: str, candidate_2_label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = len(df)
    for criterion in CRITERIA:
        column = f"{criterion}_winner_label"
        wins_1 = int((df[column] == candidate_1_label).sum())
        wins_2 = int((df[column] == candidate_2_label).sum())
        ties = int((df[column] == "tie").sum())
        decided = wins_1 + wins_2
        rows.append(
            {
                "criterion": criterion,
                "n": n,
                f"{candidate_1_label}_wins": wins_1,
                f"{candidate_2_label}_wins": wins_2,
                "ties": ties,
                "decided": decided,
                f"{candidate_1_label}_win_rate_all": wins_1 / n if n else None,
                f"{candidate_2_label}_win_rate_all": wins_2 / n if n else None,
                f"{candidate_2_label}_win_rate_decided": wins_2 / decided if decided else None,
                f"net_preference_{candidate_2_label}_minus_{candidate_1_label}": (wins_2 - wins_1) / n if n else None,
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_json_ready_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", force_ascii=False))


def build_summary_payload(
    *,
    input_path: Path,
    output_dir: Path,
    model: str,
    provider: str,
    seed: int,
    bootstrap_samples: int,
    evaluated_rows: int,
    skipped_rows: int,
    candidate_1_label: str,
    candidate_2_label: str,
    overall_summary_df: pd.DataFrame,
    relationship_summary_df: pd.DataFrame,
    power_summary_df: pd.DataFrame,
    respect_summary_df: pd.DataFrame,
    criterion_summary_df: pd.DataFrame,
) -> dict[str, Any]:
    overall = overall_summary_df.iloc[0].to_dict()
    return {
        "configuration": {
            "input_file": str(input_path),
            "output_dir": str(output_dir),
            "judge_model": model,
            "judge_provider": provider,
            "seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "evaluated_rows": evaluated_rows,
            "skipped_rows_due_to_missing_fields": skipped_rows,
            "candidate_1_label": candidate_1_label,
            "candidate_2_label": candidate_2_label,
        },
        "headline": overall,
        "overall_summary": dataframe_to_json_ready_records(overall_summary_df),
        "relationship_summary": dataframe_to_json_ready_records(relationship_summary_df),
        "power_summary": dataframe_to_json_ready_records(power_summary_df),
        "respect_summary": dataframe_to_json_ready_records(respect_summary_df),
        "criterion_summary": dataframe_to_json_ready_records(criterion_summary_df),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir) if args.output_dir else build_default_output_dir(
        args.candidate_1_label,
        args.candidate_2_label,
    )
    eval_paths = build_eval_paths(output_dir)

    ensure_output_dir(output_dir, overwrite=args.overwrite)

    print(f"Loading translation ablation CSV from: {input_path}")
    full_df = load_csv(input_path)
    input_rows = len(full_df)
    existing_judgments_df = load_existing_judgments(eval_paths.judgments_csv, overwrite=args.overwrite)
    completed_row_keys = get_completed_row_keys(existing_judgments_df)

    eval_df = load_eval_dataframe(
        input_path,
        candidate_1_column=args.candidate_1_column,
        candidate_2_column=args.candidate_2_column,
        max_rows=args.max_rows,
        only_different=args.only_different,
        completed_row_keys=completed_row_keys,
    )
    comparable_rows = len(eval_df)
    skipped_rows = input_rows - comparable_rows

    if comparable_rows == 0:
        if completed_row_keys:
            print("No new rows to evaluate after filtering and resume checks.")
            return
        raise ValueError("No rows were eligible for comparison after filtering.")

    print(f"Comparable rows to evaluate: {comparable_rows}")
    print(f"Skipped rows due to filtering: {skipped_rows}")
    if completed_row_keys:
        print(f"Resuming from existing judgments: {len(completed_row_keys)} completed rows found.")

    prompt_template = load_prompt_template(args.prompt_file)
    eval_paths.prompt_snapshot.write_text(prompt_template, encoding="utf-8")

    client = load_llm_client(args.provider, api_key_env=args.api_key_env)
    judgments = run_pairwise_evaluation(
        eval_df,
        client=client,
        provider=args.provider,
        model=args.model,
        prompt_template=prompt_template,
        candidate_1_column=args.candidate_1_column,
        candidate_2_column=args.candidate_2_column,
        candidate_1_label=args.candidate_1_label,
        candidate_2_label=args.candidate_2_label,
        seed=args.seed,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
        save_every=args.save_every,
        eval_paths=eval_paths,
        existing_judgments_df=existing_judgments_df,
    )

    judgments_df = pd.DataFrame(judgments)
    overall_summary_df = pd.DataFrame(
        [
            summarize_group(
                judgments_df,
                candidate_1_label=args.candidate_1_label,
                candidate_2_label=args.candidate_2_label,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
        ]
    )
    relationship_summary_df = build_group_summary(
        judgments_df,
        "relationship_type",
        candidate_1_label=args.candidate_1_label,
        candidate_2_label=args.candidate_2_label,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 100,
    )
    power_summary_df = build_group_summary(
        judgments_df,
        "final_power_dynamic",
        candidate_1_label=args.candidate_1_label,
        candidate_2_label=args.candidate_2_label,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 200,
    )
    respect_summary_df = build_group_summary(
        judgments_df,
        "final_respect_level",
        candidate_1_label=args.candidate_1_label,
        candidate_2_label=args.candidate_2_label,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 300,
    )
    criterion_summary_df = build_criterion_summary(
        judgments_df,
        candidate_1_label=args.candidate_1_label,
        candidate_2_label=args.candidate_2_label,
    )

    summary_payload = build_summary_payload(
        input_path=input_path,
        output_dir=output_dir,
        model=args.model,
        provider=args.provider,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        evaluated_rows=comparable_rows,
        skipped_rows=skipped_rows,
        candidate_1_label=args.candidate_1_label,
        candidate_2_label=args.candidate_2_label,
        overall_summary_df=overall_summary_df,
        relationship_summary_df=relationship_summary_df,
        power_summary_df=power_summary_df,
        respect_summary_df=respect_summary_df,
        criterion_summary_df=criterion_summary_df,
    )

    save_jsonl(judgments, eval_paths.judgments_jsonl)
    save_csv(judgments_df, eval_paths.judgments_csv)
    save_csv(overall_summary_df, eval_paths.summary_csv)
    save_csv(relationship_summary_df, eval_paths.relationship_csv)
    save_csv(power_summary_df, eval_paths.power_csv)
    save_csv(respect_summary_df, eval_paths.respect_csv)
    save_csv(criterion_summary_df, eval_paths.criteria_csv)
    save_json(summary_payload, eval_paths.summary_json, indent=2)

    print("\nEvaluation complete.")
    print_file_summary(eval_paths.judgments_jsonl, label="Saved judgment records")
    print_file_summary(eval_paths.summary_json, label="Saved summary JSON")
    print_file_summary(eval_paths.summary_csv, label="Saved overall summary CSV")
    print_file_summary(eval_paths.relationship_csv, label="Saved relationship summary CSV")
    print_file_summary(eval_paths.power_csv, label="Saved power summary CSV")
    print_file_summary(eval_paths.respect_csv, label="Saved respect summary CSV")
    print_file_summary(eval_paths.criteria_csv, label="Saved criterion summary CSV")

    headline = summary_payload["headline"]
    print(
        f"Overall: {args.candidate_1_label} wins={headline[f'{args.candidate_1_label}_wins']}, "
        f"{args.candidate_2_label} wins={headline[f'{args.candidate_2_label}_wins']}, "
        f"ties={headline['ties']}, n={headline['n']}"
    )


if __name__ == "__main__":
    main()
