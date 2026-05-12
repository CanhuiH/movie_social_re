# Movie Social Relationship Translation Project

This repository studies whether explicit social relationship information improves English-to-Mandarin subtitle translation for movie dialogue.

For submission, our materials are split into two bundles:

- `TRACK_B_movie_social_re.zip`
  - the current repository, covering the Track B relationship-aware translation pipeline and the integrated project write-up
- `TRACK_A_qe_reproducibility_bundle.zip`
  - the companion reproducibility bundle for the Track A source-side pragmatic QE / prelabel pipeline

The project has two linked parts:

1. `Track A`: source-side pragmatic risk screening (`prelabel`)
2. `Track B`: relationship-aware translation and pairwise evaluation

The final Track B experiment compares:

- `A0`: context-only translation
- `A4`: full graph-guided translation

and uses additional ablations:

- `A1`: power/respect-only
- `A2`: relationship-only
- `A3`: local social labels

## Quickstart

Create `.env` in the project root:

```env
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you only reproduce the current evaluation setup, `GEMINI_API_KEY` is enough.

An environment template is included:

```bash
copy .env.example .env
```

For a one-command bash wrapper around the full pipeline, see:

```bash
./run_full_pipeline.sh --help
```

## Dataset Summary

Track B uses dialogue from 7 movies:

```text
amadeus
an_officer_and_a_gentleman
dead_poets_society
gladiator
mrs_brown
the_godfather
titanic
```

Key dataset sizes:

- human power/respect annotation set: `3,791` dialogue rows
- cleaned agreed non-unclear gold-label set: `1,917` rows
- final translation comparison set: `599` rows

The final `599` rows are the overlap of:

- prelabel risk rows
- valid relationship extraction
- agreed non-unclear gold power/respect labels

## Repository Structure

Top-level structure:

```text
configs/
data/
outputs/
prompts/
scripts/
src/
third_party/
```

What each folder contains:

- `configs/`
  - JSON config files for modeling, schemas, and translation settings
- `data/`
  - project data artifacts, including labeled data, intermediate overlap files, graph outputs, and final translation CSVs
- `outputs/`
  - paper-facing outputs such as summary tables, evaluation judgments, and report notes
- `prompts/`
  - translation and evaluation prompt templates
- `scripts/`
  - command-line entry points for each pipeline step
- `src/`
  - core implementation code used by the scripts
- `third_party/`
  - bundled minimal external dependency code needed for Track A prelabel reproduction

Important subfolders:

```text
data/data_prelabel_predictions/   precomputed Track A risk predictions
data/labeled/                     human gold power/respect annotations
data/interim/                     overlap and translation-input CSVs
data/graph/                       social graph edge-state outputs
data/translation_eval/            final translation and ablation comparison CSVs
outputs/tables/                   summary CSVs referenced in the paper
outputs/translation_eval/         pairwise LLM-as-a-judge outputs
outputs/reports/                  working notes and report planning files
```

Paper-facing artifacts mainly live in:

```text
outputs/tables/
outputs/translation_eval/
outputs/reports/
```

## Pipeline Overview

### Track A: Prelabel risk screening

Run:

```bash
python scripts/run_prelabel_risk_prediction.py
```

Important dependency note:

- this repo integrates Track A prelabel outputs into the Track B translation pipeline through the overlap step and downstream graph-guided translation workflow
- the minimal Track A feature code required for rerunning prelabel is bundled in:
  `third_party/subtitle_translation_english_to_mandarin_codes/`
- this bundled code comes from the related repository:
  `https://github.com/abss123/subtitle_translation_english_to_mandarin`
- the final submission bundle includes this minimal bundled copy so that the Track A prelabel stage can be rerun from the submitted materials
- `scripts/run_prelabel_risk_prediction.py` uses the bundled copy first
- a legacy fallback to a sibling checkout of the same Track A repository is kept only for local compatibility

Input:

```text
data/data_prelabel/*.zip
```

Main outputs:

```text
data/data_prelabel_predictions/all_prelabel_risk_predictions.csv
data/data_prelabel_predictions/all_prelabel_risk_summary.csv
```

Important output columns:

- `pred_translation_error_possible`
- `pred_risk_category_count`
- `pred_risk_categories`
- `pred_risk_confidence_summary`
- `pred_risk_cues`

### Track B: Translation input construction

Relationship extraction:

```bash
python scripts/02_run_all_movies_re.py
```

Main output:

```text
data/processed/<movie_name>/re_clean.csv
```

Human gold labels:

```text
data/labeled/power_respect_labels.csv
```

The final translation pipeline uses:

- `final_power_dynamic`
- `final_respect_level`

not classifier predictions.

Overlap filtering:

```bash
python scripts/06_prepare_risk_gold_overlap.py
```

Outputs:

```text
data/interim/risk_gold_overlap.csv
outputs/tables/overlap_summary.csv
```

Social graph:

```bash
python scripts/07_build_social_graph.py --recent-window 6
```

Outputs:

```text
data/graph/social_graph_edges.csv
outputs/tables/graph_summary_stats.csv
```

Important graph columns:

- `recent_power_dynamic`
- `recent_respect_level`
- `recent_relationship_type`
- `aggregate_power_dynamic`
- `aggregate_respect_level`
- `aggregate_relationship_type`

Final translation input:

```bash
python scripts/08_prepare_translation_input.py
```

Outputs:

```text
data/interim/translation_input.csv
outputs/tables/translation_input_summary.csv
```

## Translation Conditions

```text
A0: Context-only
A1: Power/respect-only
A2: Relationship-only
A3: Local social labels
A4: Full graph-guided
```

Interpretation:

- `A0`: no explicit social metadata
- `A1`: current-row power + respect only
- `A2`: current-row relationship + evidence only
- `A3`: all current-row social labels
- `A4`: A3 plus recent and aggregate graph history

## Translation Pipeline

If you prefer a single bash entrypoint instead of running each script manually, you can use:

```bash
./run_full_pipeline.sh
```

Useful variants:

```bash
./run_full_pipeline.sh --track-b-only
./run_full_pipeline.sh --track-b-only --skip-modeling
./run_full_pipeline.sh --with-supp-eval
./run_full_pipeline.sh --translation-max-rows 10 --translation-dry-run
```

Main comparison:

```bash
python scripts/09_translate_context_only.py
python scripts/10_translate_graph_guided.py
python scripts/11_merge_translation_outputs.py
```

Ablations:

```bash
python scripts/12_translate_social_labels_only.py
python scripts/14_translate_power_respect_only.py
python scripts/15_translate_relationship_only.py
python scripts/13_merge_ablation_outputs.py
```

Main outputs:

```text
data/translation_eval/translation_comparison.csv
outputs/tables/translation_comparison_summary.csv
```

Ablation outputs:

```text
data/translation_eval/translation_ablation_comparison.csv
outputs/tables/translation_ablation_summary.csv
```

The ablation comparison file contains:

- `translation_context_only`
- `translation_power_respect_only`
- `translation_relationship_only`
- `translation_social_labels_only`
- `translation_graph_guided`

## Evaluation Pipeline

Main comparison `A0 vs A4`:

```bash
python scripts/16_eval_translation_ablation.py --provider gemini --model gemini-3.1-flash-lite --candidate-1-column translation_context_only --candidate-1-label context_only --candidate-2-column translation_graph_guided --candidate-2-label graph_guided --only-different --output-dir outputs/translation_eval/context_vs_graph_gemini31lite
```

Key ablation `A3 vs A4`:

```bash
python scripts/16_eval_translation_ablation.py --provider gemini --model gemini-3.1-flash-lite --candidate-1-column translation_social_labels_only --candidate-1-label social_labels_only --candidate-2-column translation_graph_guided --candidate-2-label graph_guided --only-different --output-dir outputs/translation_eval/social_vs_graph_gemini31lite
```

Supplemental A0-based ablations used in the report:

- `A0 vs A1`
- `A0 vs A2`
- `A0 vs A3`

Relevant evaluation folders:

```text
outputs/translation_eval/context_vs_graph_gemini31lite/
outputs/translation_eval/social_vs_graph_gemini31lite/
outputs/translation_eval/context_vs_powerrespect_gemini31lite/
outputs/translation_eval/context_vs_relationship_gemini31lite_150/
outputs/translation_eval/context_vs_sociallabels_gemini31lite_150/
```

Each evaluation directory contains artifacts such as:

```text
judgments.csv
judgments.jsonl
summary.json
summary_overall.csv
summary_by_criterion.csv
summary_by_power.csv
summary_by_respect.csv
summary_by_relationship.csv
judge_prompt_snapshot.txt
```

Judge criteria:

1. `meaning_accuracy`
2. `social_relationship`
3. `register_tone`
4. `fluency`

## Reproducibility

### Exact settings used for the current paper write-up

Keep these settings fixed if you want to match the current reported results:

- translation/evaluation provider: `gemini`
- judge model: `gemini-3.1-flash-lite`
- graph recent window: `6`
- main evaluation flag: `--only-different`
- supplemental `A0` ablations reported at standardized `n=150`

### Reproduction scopes

There are two realistic reproduction scopes for this repo:

1. `Full Track A + Track B`
   - uses the bundled prelabel feature code in `third_party/subtitle_translation_english_to_mandarin_codes/`
   - reruns risk prediction, overlap construction, translation, and evaluation
2. `Track B only`
   - uses the included precomputed risk predictions in `data/data_prelabel_predictions/`
   - reruns overlap construction, graph building, translation, merge, and evaluation

If your goal is to reproduce the final paper results, Track B-only reproduction is sufficient.

### Reproduce the paper-facing datasets and summary tables

Run:

```bash
python scripts/run_prelabel_risk_prediction.py
python scripts/02_run_all_movies_re.py
python scripts/06_prepare_risk_gold_overlap.py
python scripts/07_build_social_graph.py --recent-window 6
python scripts/08_prepare_translation_input.py
python scripts/09_translate_context_only.py
python scripts/10_translate_graph_guided.py
python scripts/11_merge_translation_outputs.py
python scripts/12_translate_social_labels_only.py
python scripts/14_translate_power_respect_only.py
python scripts/15_translate_relationship_only.py
python scripts/13_merge_ablation_outputs.py
```

This reproduces:

```text
data/translation_eval/translation_comparison.csv
data/translation_eval/translation_ablation_comparison.csv
outputs/tables/overlap_summary.csv
outputs/tables/translation_input_summary.csv
outputs/tables/translation_comparison_summary.csv
outputs/tables/translation_ablation_summary.csv
```

### Reproduce the reported main Track B evaluation

Run the two main evaluation commands from the previous section.

These should reproduce the reported main counts:

- `A0 vs A4`: `n=467`, `107 / 351 / 9`
- `A3 vs A4`: `n=454`, `150 / 293 / 11`

Format:

```text
A wins / B wins / tie
```

### Reproduce the standardized supplemental A0 ablations

The report standardizes the auxiliary A0-based ablations to the first `150` judged differing-output pairs in each saved `judgments.csv`:

- `A0 vs A1`: `45 / 104 / 1`
- `A0 vs A2`: `44 / 105 / 1`
- `A0 vs A3`: `47 / 101 / 2`

### Resume behavior

- translation scripts skip completed rows unless `--overwrite` is used
- evaluation can resume from existing `judgments.csv` and `judgments.jsonl`
- if a run stops because of rate limits, rerun the same command without `--overwrite`

### Paper consistency notes

- do not mix outputs from different judge models in the same evaluation directory
- if prompts change, regenerate the affected translation outputs and rerun merge/evaluation
- if `power_respect_labels.csv` changes, rerun from Step 6
- if relationship extraction changes, rerun from Step 6

## Modeling Note

The power/respect classifier is included as a modeling experiment, not as a source of final translation labels.

Run:

```bash
python scripts/03_prepare_modeling_data.py
python scripts/04_train_power_respect.py
python scripts/05_tune_logistic_regression.py
```

Outputs:

```text
outputs/modeling/
```

The final translation pipeline still uses human gold labels from:

```text
data/labeled/power_respect_labels.csv
```

## Prompt Files

Prompt templates live in `prompts/`:

```text
translate_context_only.txt
translate_power_respect_only.txt
translate_relationship_only.txt
translate_social_labels_only.txt
translate_graph_guided.txt
translation_pairwise_judge.txt
translation_ablation_pairwise_judge.txt
```

The graph-guided prompt consumes:

- `final_power_dynamic`
- `final_respect_level`
- `relationship_type`
- `evidence`
- `recent_*`
- `aggregate_*`

## Key Paper Artifacts

Tables and summaries:

```text
outputs/tables/overlap_summary.csv
outputs/tables/translation_input_summary.csv
outputs/tables/translation_comparison_summary.csv
outputs/tables/translation_ablation_summary.csv
outputs/tables/graph_summary_stats.csv
outputs/tables/graph_summary_generation_stats.csv
```

Main evaluation folders:

```text
outputs/translation_eval/context_vs_graph_gemini31lite/
outputs/translation_eval/social_vs_graph_gemini31lite/
```

Supplemental evaluation folders:

```text
outputs/translation_eval/context_vs_powerrespect_gemini31lite/
outputs/translation_eval/context_vs_relationship_gemini31lite_150/
outputs/translation_eval/context_vs_sociallabels_gemini31lite_150/
```
