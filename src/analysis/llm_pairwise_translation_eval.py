from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import DEFAULT_MOVIE_NAME, OPENAI_API_KEY_ENV, PROMPTS_DIR, ensure_project_dirs
from src.utils.io import load_csv, print_file_summary, save_csv, save_json, save_jsonl
from src.utils.paths import get_movie_processed_dir


DEFAULT_EVAL_MODEL = "gpt-5.5"
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_RANDOM_SEED = 7
DEFAULT_MAX_RETRIES = 3
DEFAULT_PROMPT_PATH = PROMPTS_DIR / "translation_pairwise_judge.txt"
DEFAULT_GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

SYSTEM_PROMPT = """You are a careful bilingual evaluator for movie dialogue translation.
You compare two candidate Chinese translations against the English source and return only strict JSON.
"""

REQUIRED_COLUMNS = {
    "movie_name",
    "conversation_id",
    "utterance_id",
    "speaker_name",
    "listener_name",
    "current_turn",
    "current_text",
    "context_text",
    "relationship_type",
    "graph_summary",
    "translation_context_only",
    "translation_with_graph",
}

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
    criteria_csv: Path
    prompt_snapshot: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate context-only versus graph-aware translations with an "
            "LLM-as-a-judge pairwise ranking workflow."
        )
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=DEFAULT_MOVIE_NAME,
        help=f'Movie name used to locate the default comparison CSV. Default: "{DEFAULT_MOVIE_NAME}".',
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help=(
            "Optional path to a translation comparison CSV. If omitted, the script "
            "uses data/processed/<movie>/translation_comparison.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Optional directory for evaluation artifacts. If omitted, outputs are "
            "saved under data/processed/<movie>/llm_judge_eval/."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_EVAL_MODEL,
        help=f'Judge model name. Default: "{DEFAULT_EVAL_MODEL}".',
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "gemini"],
        help='LLM provider used as the judge. Default: "openai".',
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=None,
        help="Optional environment variable name containing the selected provider API key.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on the number of comparable rows to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for candidate-order shuffling and bootstrap. Default: {DEFAULT_RANDOM_SEED}.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=(
            "Number of bootstrap resamples used for confidence intervals. "
            f"Default: {DEFAULT_BOOTSTRAP_SAMPLES}."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional custom pairwise judge prompt template path.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=(
            "Maximum retries per row when the judge response is malformed or the "
            f"API call fails. Default: {DEFAULT_MAX_RETRIES}."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite an existing evaluation output directory.",
    )
    return parser.parse_args()


def get_default_input_path(movie_name: str) -> Path:
    return get_movie_processed_dir(movie_name) / "translation_comparison.csv"


def get_default_output_dir(movie_name: str) -> Path:
    return get_movie_processed_dir(movie_name) / "llm_judge_eval"


def build_eval_paths(output_dir: Path) -> EvalPaths:
    return EvalPaths(
        output_dir=output_dir,
        judgments_jsonl=output_dir / "judgments.jsonl",
        judgments_csv=output_dir / "judgments.csv",
        summary_json=output_dir / "summary.json",
        summary_csv=output_dir / "summary_by_subset.csv",
        relationship_csv=output_dir / "summary_by_relationship.csv",
        criteria_csv=output_dir / "summary_by_criterion.csv",
        prompt_snapshot=output_dir / "judge_prompt_snapshot.txt",
    )


def validate_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing_columns)}")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def load_eval_dataframe(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    df = load_csv(path)
    validate_columns(df, REQUIRED_COLUMNS)

    for column in REQUIRED_COLUMNS:
        df[column] = df[column].apply(normalize_text)

    comparable_mask = (
        df["translation_context_only"].ne("")
        & df["translation_with_graph"].ne("")
        & df["current_text"].ne("")
    )
    filtered_df = df.loc[comparable_mask].copy()

    if max_rows is not None:
        filtered_df = filtered_df.head(max_rows).copy()

    filtered_df["has_graph_summary"] = filtered_df["graph_summary"].ne("")
    filtered_df["has_relationship_label"] = filtered_df["relationship_type"].str.lower().ne("unclear")
    filtered_df["evaluation_subset"] = np.where(
        filtered_df["has_graph_summary"],
        "graph_available",
        "graph_missing",
    )
    return filtered_df.reset_index(drop=True)


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


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already contains files: {output_dir}. "
            "Use --overwrite to replace the previous evaluation artifacts."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def build_user_prompt(
    record: dict[str, Any],
    prompt_template: str,
    translation_a: str,
    translation_b: str,
) -> str:
    return prompt_template.format(
        movie_name=record.get("movie_name", ""),
        conversation_id=record.get("conversation_id", ""),
        utterance_id=record.get("utterance_id", ""),
        speaker_name=record.get("speaker_name", "") or "UNKNOWN",
        listener_name=record.get("listener_name", "") or "UNKNOWN",
        relationship_type=record.get("relationship_type", "") or "unclear",
        graph_summary=record.get("graph_summary", "") or "[NO GRAPH SUMMARY]",
        context_text=record.get("context_text", "") or "[NO PREVIOUS CONTEXT]",
        current_turn=record.get("current_turn", ""),
        current_text=record.get("current_text", ""),
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


def map_choice_to_system(choice: str, candidate_a_system: str, candidate_b_system: str) -> str:
    if choice == "A":
        return candidate_a_system
    if choice == "B":
        return candidate_b_system
    return "tie"


def build_judgment_record(
    record: dict[str, Any],
    row_index: int,
    candidate_a_system: str,
    candidate_b_system: str,
    judge_result: dict[str, Any],
) -> dict[str, Any]:
    winner_system = map_choice_to_system(
        judge_result["winner"],
        candidate_a_system=candidate_a_system,
        candidate_b_system=candidate_b_system,
    )

    if winner_system == "with_graph":
        with_graph_outcome = "win"
        outcome_score = 1
    elif winner_system == "context_only":
        with_graph_outcome = "loss"
        outcome_score = -1
    else:
        with_graph_outcome = "tie"
        outcome_score = 0

    result: dict[str, Any] = {
        "row_index": row_index,
        "movie_name": record.get("movie_name", ""),
        "conversation_id": record.get("conversation_id", ""),
        "utterance_id": record.get("utterance_id", ""),
        "speaker_name": record.get("speaker_name", ""),
        "listener_name": record.get("listener_name", ""),
        "relationship_type": record.get("relationship_type", ""),
        "graph_summary": record.get("graph_summary", ""),
        "has_graph_summary": bool(record.get("has_graph_summary", False)),
        "has_relationship_label": bool(record.get("has_relationship_label", False)),
        "current_turn": record.get("current_turn", ""),
        "current_text": record.get("current_text", ""),
        "context_text": record.get("context_text", ""),
        "translation_context_only": record.get("translation_context_only", ""),
        "translation_with_graph": record.get("translation_with_graph", ""),
        "candidate_a_system": candidate_a_system,
        "candidate_b_system": candidate_b_system,
        "winner_label": judge_result["winner"],
        "winner_system": winner_system,
        "with_graph_outcome": with_graph_outcome,
        "outcome_score": outcome_score,
        "confidence": judge_result["confidence"],
        "reason": judge_result["reason"],
        "issue_focus_observed": judge_result["issue_focus_observed"],
        "raw_judge_response": judge_result["raw_response"],
    }

    for criterion, choice in judge_result["criterion_winners"].items():
        result[f"{criterion}_winner_label"] = choice
        result[f"{criterion}_winner_system"] = map_choice_to_system(
            choice,
            candidate_a_system=candidate_a_system,
            candidate_b_system=candidate_b_system,
        )

    return result


def run_pairwise_evaluation(
    df: pd.DataFrame,
    client: Any,
    provider: str,
    model: str,
    prompt_template: str,
    seed: int,
    max_retries: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []

    for row_index, row in tqdm(
        list(df.iterrows()),
        total=len(df),
        desc="Running LLM pairwise evaluation",
    ):
        record = row.to_dict()

        if rng.random() < 0.5:
            candidate_a_system = "context_only"
            candidate_b_system = "with_graph"
        else:
            candidate_a_system = "with_graph"
            candidate_b_system = "context_only"

        translation_a = record[f"translation_{candidate_a_system}"]
        translation_b = record[f"translation_{candidate_b_system}"]

        user_prompt = build_user_prompt(
            record=record,
            prompt_template=prompt_template,
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
                candidate_a_system=candidate_a_system,
                candidate_b_system=candidate_b_system,
                judge_result=judge_result,
            )
        )

    return results


def summarize_group(df: pd.DataFrame, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    n = int(len(df))
    wins = int((df["with_graph_outcome"] == "win").sum())
    losses = int((df["with_graph_outcome"] == "loss").sum())
    ties = int((df["with_graph_outcome"] == "tie").sum())
    decided = wins + losses

    summary: dict[str, Any] = {
        "n": n,
        "with_graph_wins": wins,
        "context_only_wins": losses,
        "ties": ties,
        "decided": decided,
        "with_graph_win_rate_all": wins / n if n else None,
        "with_graph_win_rate_decided": wins / decided if decided else None,
        "net_preference": (wins - losses) / n if n else None,
        "issue_focus_observed_rate": float(df["issue_focus_observed"].mean()) if n else None,
    }

    if not n:
        summary.update(
            {
                "win_rate_all_ci_low": None,
                "win_rate_all_ci_high": None,
                "net_preference_ci_low": None,
                "net_preference_ci_high": None,
            }
        )
        return summary

    scores = df["outcome_score"].to_numpy(dtype=float)
    wins_binary = (df["with_graph_outcome"] == "win").to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, n, size=(bootstrap_samples, n))
    sampled_scores = scores[sampled_indices]
    sampled_wins = wins_binary[sampled_indices]

    win_rate_samples = sampled_wins.mean(axis=1)
    net_preference_samples = sampled_scores.mean(axis=1)

    summary["win_rate_all_ci_low"] = float(np.quantile(win_rate_samples, 0.025))
    summary["win_rate_all_ci_high"] = float(np.quantile(win_rate_samples, 0.975))
    summary["net_preference_ci_low"] = float(np.quantile(net_preference_samples, 0.025))
    summary["net_preference_ci_high"] = float(np.quantile(net_preference_samples, 0.975))
    return summary


def build_subset_summary(df: pd.DataFrame, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    subsets = {
        "all_rows": df,
        "graph_available": df.loc[df["has_graph_summary"]].copy(),
        "relationship_labeled": df.loc[df["has_relationship_label"]].copy(),
        "graph_and_relationship": df.loc[df["has_graph_summary"] & df["has_relationship_label"]].copy(),
    }

    rows: list[dict[str, Any]] = []
    for offset, (subset_name, subset_df) in enumerate(subsets.items()):
        row = {"subset": subset_name}
        row.update(summarize_group(subset_df, bootstrap_samples=bootstrap_samples, seed=seed + offset))
        rows.append(row)
    return pd.DataFrame(rows)


def build_relationship_summary(df: pd.DataFrame, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = df.groupby("relationship_type", dropna=False)
    for offset, (relationship_type, group_df) in enumerate(grouped):
        row = {"relationship_type": relationship_type}
        row.update(summarize_group(group_df, bootstrap_samples=bootstrap_samples, seed=seed + offset))
        rows.append(row)
    relationship_df = pd.DataFrame(rows)
    if relationship_df.empty:
        return relationship_df
    return relationship_df.sort_values(["n", "relationship_type"], ascending=[False, True]).reset_index(drop=True)


def build_criterion_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = len(df)
    for criterion in CRITERIA:
        column = f"{criterion}_winner_system"
        wins = int((df[column] == "with_graph").sum())
        losses = int((df[column] == "context_only").sum())
        ties = int((df[column] == "tie").sum())
        decided = wins + losses
        rows.append(
            {
                "criterion": criterion,
                "n": n,
                "with_graph_wins": wins,
                "context_only_wins": losses,
                "ties": ties,
                "decided": decided,
                "with_graph_win_rate_all": wins / n if n else None,
                "with_graph_win_rate_decided": wins / decided if decided else None,
                "net_preference": (wins - losses) / n if n else None,
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
    seed: int,
    bootstrap_samples: int,
    evaluated_rows: int,
    skipped_rows: int,
    subset_summary_df: pd.DataFrame,
    relationship_summary_df: pd.DataFrame,
    criterion_summary_df: pd.DataFrame,
) -> dict[str, Any]:
    all_rows_summary = subset_summary_df.loc[subset_summary_df["subset"] == "all_rows"].iloc[0].to_dict()
    graph_rows_summary = subset_summary_df.loc[subset_summary_df["subset"] == "graph_available"].iloc[0].to_dict()

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
        },
        "headline": {
            "overall": all_rows_summary,
            "graph_available_subset": graph_rows_summary,
        },
        "subset_summary": dataframe_to_json_ready_records(subset_summary_df),
        "relationship_summary": dataframe_to_json_ready_records(relationship_summary_df),
        "criterion_summary": dataframe_to_json_ready_records(criterion_summary_df),
    }


def main() -> None:
    args = parse_args()
    ensure_project_dirs(args.movie)

    input_path = Path(args.input_file) if args.input_file else get_default_input_path(args.movie)
    output_dir = Path(args.output_dir) if args.output_dir else get_default_output_dir(args.movie)
    eval_paths = build_eval_paths(output_dir)

    ensure_output_dir(output_dir, overwrite=args.overwrite)

    print(f"Loading translation comparison CSV from: {input_path}")
    full_df = load_csv(input_path)
    input_rows = len(full_df)

    eval_df = load_eval_dataframe(input_path, max_rows=args.max_rows)
    comparable_rows = len(eval_df)
    skipped_rows = input_rows - comparable_rows

    if comparable_rows == 0:
        raise ValueError("No rows were eligible for comparison after filtering missing translations.")

    print(f"Comparable rows to evaluate: {comparable_rows}")
    print(f"Skipped rows due to missing fields: {skipped_rows}")

    prompt_template = load_prompt_template(args.prompt_file)
    eval_paths.prompt_snapshot.write_text(prompt_template, encoding="utf-8")

    client = load_llm_client(args.provider, api_key_env=args.api_key_env)
    judgments = run_pairwise_evaluation(
        df=eval_df,
        client=client,
        provider=args.provider,
        model=args.model,
        prompt_template=prompt_template,
        seed=args.seed,
        max_retries=args.max_retries,
    )

    judgments_df = pd.DataFrame(judgments)
    subset_summary_df = build_subset_summary(
        judgments_df,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    relationship_summary_df = build_relationship_summary(
        judgments_df,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 1000,
    )
    criterion_summary_df = build_criterion_summary(judgments_df)

    summary_payload = build_summary_payload(
        input_path=input_path,
        output_dir=output_dir,
        model=args.model,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        evaluated_rows=comparable_rows,
        skipped_rows=skipped_rows,
        subset_summary_df=subset_summary_df,
        relationship_summary_df=relationship_summary_df,
        criterion_summary_df=criterion_summary_df,
    )

    save_jsonl(judgments, eval_paths.judgments_jsonl)
    save_csv(judgments_df, eval_paths.judgments_csv)
    save_csv(subset_summary_df, eval_paths.summary_csv)
    save_csv(relationship_summary_df, eval_paths.relationship_csv)
    save_csv(criterion_summary_df, eval_paths.criteria_csv)
    save_json(summary_payload, eval_paths.summary_json, indent=2)

    print("\nEvaluation complete.")
    print_file_summary(eval_paths.judgments_jsonl, label="Saved judgment records")
    print_file_summary(eval_paths.summary_json, label="Saved summary JSON")
    print_file_summary(eval_paths.summary_csv, label="Saved subset summary CSV")
    print_file_summary(eval_paths.relationship_csv, label="Saved relationship summary CSV")
    print_file_summary(eval_paths.criteria_csv, label="Saved criterion summary CSV")

    headline = summary_payload["headline"]["overall"]
    print(
        "Overall with-graph wins: "
        f"{headline['with_graph_wins']} / {headline['n']} "
        f"(ties={headline['ties']}, "
        f"win_rate_all={headline['with_graph_win_rate_all']:.3f}, "
        f"net_preference={headline['net_preference']:.3f})"
    )


if __name__ == "__main__":
    main()
