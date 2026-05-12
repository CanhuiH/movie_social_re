#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-gemini}"
JUDGE_MODEL="${JUDGE_MODEL:-gemini-3.1-flash-lite}"
GRAPH_RECENT_WINDOW="${GRAPH_RECENT_WINDOW:-6}"
EVAL_SLEEP_SECONDS="${EVAL_SLEEP_SECONDS:-0}"
EVAL_SAVE_EVERY="${EVAL_SAVE_EVERY:-10}"
SUPP_MAX_ROWS="${SUPP_MAX_ROWS:-150}"

RUN_PRELABEL=1
RUN_PIPELINE=1
RUN_MAIN_EVAL=1
RUN_SUPP_EVAL=0
NO_ABLATION=0
SKIP_MODELING=0
TRANSLATION_DRY_RUN=0
TRANSLATION_OVERWRITE=0
TRANSLATION_MAX_ROWS=""

print_usage() {
  cat <<'EOF'
Usage:
  ./run_full_pipeline.sh [options]

Options:
  --track-b-only            Skip Track A prelabel and start from the integrated Track B pipeline.
  --skip-modeling           Pass --skip-modeling to scripts/run_all.py.
  --no-ablation             Skip ablation translations in scripts/run_all.py.
  --no-eval                 Skip all pairwise evaluation.
  --with-supp-eval          Also run standardized supplemental A0-vs-A1/A2/A3 evaluations.
  --translation-dry-run     Pass --translation-dry-run to scripts/run_all.py.
  --translation-overwrite   Pass --translation-overwrite to scripts/run_all.py.
  --translation-max-rows N  Pass --translation-max-rows N to scripts/run_all.py.
  --judge-provider NAME     Override evaluation provider. Default: gemini.
  --judge-model NAME        Override evaluation model. Default: gemini-3.1-flash-lite.
  --sleep-seconds N         Sleep between evaluation requests. Default: 0.
  --save-every N            Checkpoint every N rows during evaluation. Default: 10.
  --supp-max-rows N         Sample size for supplemental A0 ablations. Default: 150.
  -h, --help                Show this help message.

Environment overrides:
  PYTHON_BIN
  JUDGE_PROVIDER
  JUDGE_MODEL
  GRAPH_RECENT_WINDOW
  EVAL_SLEEP_SECONDS
  EVAL_SAVE_EVERY
  SUPP_MAX_ROWS
EOF
}

run_cmd() {
  echo
  echo "$ $*"
  "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --track-b-only)
      RUN_PRELABEL=0
      shift
      ;;
    --skip-modeling)
      SKIP_MODELING=1
      shift
      ;;
    --no-ablation)
      NO_ABLATION=1
      shift
      ;;
    --no-eval)
      RUN_MAIN_EVAL=0
      RUN_SUPP_EVAL=0
      shift
      ;;
    --with-supp-eval)
      RUN_SUPP_EVAL=1
      shift
      ;;
    --translation-dry-run)
      TRANSLATION_DRY_RUN=1
      shift
      ;;
    --translation-overwrite)
      TRANSLATION_OVERWRITE=1
      shift
      ;;
    --translation-max-rows)
      TRANSLATION_MAX_ROWS="${2:?missing value for --translation-max-rows}"
      shift 2
      ;;
    --judge-provider)
      JUDGE_PROVIDER="${2:?missing value for --judge-provider}"
      shift 2
      ;;
    --judge-model)
      JUDGE_MODEL="${2:?missing value for --judge-model}"
      shift 2
      ;;
    --sleep-seconds)
      EVAL_SLEEP_SECONDS="${2:?missing value for --sleep-seconds}"
      shift 2
      ;;
    --save-every)
      EVAL_SAVE_EVERY="${2:?missing value for --save-every}"
      shift 2
      ;;
    --supp-max-rows)
      SUPP_MAX_ROWS="${2:?missing value for --supp-max-rows}"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_usage >&2
      exit 1
      ;;
  esac
done

if [[ "$NO_ABLATION" -eq 1 ]]; then
  RUN_SUPP_EVAL=0
fi

if [[ "$RUN_PRELABEL" -eq 1 ]]; then
  run_cmd "$PYTHON_BIN" scripts/run_prelabel_risk_prediction.py
fi

if [[ "$RUN_PIPELINE" -eq 1 ]]; then
  PIPELINE_ARGS=()
  if [[ "$NO_ABLATION" -eq 1 ]]; then
    PIPELINE_ARGS+=(--no-ablation)
  fi
  if [[ "$SKIP_MODELING" -eq 1 ]]; then
    PIPELINE_ARGS+=(--skip-modeling)
  fi
  if [[ "$TRANSLATION_DRY_RUN" -eq 1 ]]; then
    PIPELINE_ARGS+=(--translation-dry-run)
  fi
  if [[ "$TRANSLATION_OVERWRITE" -eq 1 ]]; then
    PIPELINE_ARGS+=(--translation-overwrite)
  fi
  if [[ -n "$TRANSLATION_MAX_ROWS" ]]; then
    PIPELINE_ARGS+=(--translation-max-rows "$TRANSLATION_MAX_ROWS")
  fi

  run_cmd "$PYTHON_BIN" scripts/run_all.py "${PIPELINE_ARGS[@]}"
fi

if [[ "$RUN_MAIN_EVAL" -eq 1 ]]; then
  COMMON_EVAL_ARGS=(
    --provider "$JUDGE_PROVIDER"
    --model "$JUDGE_MODEL"
    --only-different
    --sleep-seconds "$EVAL_SLEEP_SECONDS"
    --save-every "$EVAL_SAVE_EVERY"
  )

  run_cmd "$PYTHON_BIN" scripts/16_eval_translation_ablation.py \
    "${COMMON_EVAL_ARGS[@]}" \
    --candidate-1-column translation_context_only \
    --candidate-1-label context_only \
    --candidate-2-column translation_graph_guided \
    --candidate-2-label graph_guided \
    --output-dir outputs/translation_eval/context_vs_graph_gemini31lite

  if [[ "$NO_ABLATION" -eq 0 ]]; then
    run_cmd "$PYTHON_BIN" scripts/16_eval_translation_ablation.py \
      "${COMMON_EVAL_ARGS[@]}" \
      --candidate-1-column translation_social_labels_only \
      --candidate-1-label social_labels_only \
      --candidate-2-column translation_graph_guided \
      --candidate-2-label graph_guided \
      --output-dir outputs/translation_eval/social_vs_graph_gemini31lite
  fi
fi

if [[ "$RUN_SUPP_EVAL" -eq 1 ]]; then
  COMMON_SUPP_ARGS=(
    --provider "$JUDGE_PROVIDER"
    --model "$JUDGE_MODEL"
    --only-different
    --max-rows "$SUPP_MAX_ROWS"
    --sleep-seconds "$EVAL_SLEEP_SECONDS"
    --save-every "$EVAL_SAVE_EVERY"
  )

  run_cmd "$PYTHON_BIN" scripts/16_eval_translation_ablation.py \
    "${COMMON_SUPP_ARGS[@]}" \
    --candidate-1-column translation_context_only \
    --candidate-1-label context_only \
    --candidate-2-column translation_power_respect_only \
    --candidate-2-label power_respect_only \
    --output-dir outputs/translation_eval/context_vs_powerrespect_gemini31lite_150

  run_cmd "$PYTHON_BIN" scripts/16_eval_translation_ablation.py \
    "${COMMON_SUPP_ARGS[@]}" \
    --candidate-1-column translation_context_only \
    --candidate-1-label context_only \
    --candidate-2-column translation_relationship_only \
    --candidate-2-label relationship_only \
    --output-dir outputs/translation_eval/context_vs_relationship_gemini31lite_150

  run_cmd "$PYTHON_BIN" scripts/16_eval_translation_ablation.py \
    "${COMMON_SUPP_ARGS[@]}" \
    --candidate-1-column translation_context_only \
    --candidate-1-label context_only \
    --candidate-2-column translation_social_labels_only \
    --candidate-2-label social_labels_only \
    --output-dir outputs/translation_eval/context_vs_sociallabels_gemini31lite_150
fi

echo
echo "Full pipeline finished."
