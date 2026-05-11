# Movie Social Relationship Translation Project

This project studies whether explicit social relationship information can improve Mandarin subtitle translation for English movie dialogue. The main comparison is between a context-only translation baseline and a graph-guided translation condition that uses power, respect, and speaker-listener relationship signals.

The project focuses on dialogue lines where social context may affect translation choices, such as politeness, formality, intimacy, resistance, sarcasm, deference, or relationship-specific wording.

## Project Goal

The goal is to evaluate whether adding structured social context helps an LLM produce more sociopragmatically appropriate Mandarin translations.

The main pipeline compares two translation settings:

1. **Context-only translation**
   - Uses the current English line, speaker/listener metadata, and previous dialogue context.
   - Does not use social labels or graph information.

2. **Graph-guided translation**
   - Uses the same context as the baseline.
   - Adds structured social guidance, including power dynamic, respect level, relationship type, evidence, recent graph state, and aggregate graph state.


The project also includes an optional ablation study with five translation conditions:

```text
A0: Context-only baseline
A1: Power/respect-only guidance
A2: Relationship-only guidance
A3: Local social-label guidance
A4: Full graph-guided translation
```


In the final translation experiment, the power dynamic and respect level are **gold human annotation labels**, not model predictions. The power/respect classifier is still included in the project as a modeling experiment, but its predicted labels are not used as the final social guidance for translation.

## Dataset Summary

The project uses dialogue data from **7 movies**:

```text
amadeus
an_officer_and_a_gentleman
dead_poets_society
gladiator
mrs_brown
the_godfather
titanic
```

For the human-labeled power/respect dataset, we annotated **3,791 dialogue rows** across these 7 movies. After filtering for target risk rows, agreed non-unclear human power/respect labels, and overlap with valid relationship types, the final translation comparison set contains **599 dialogue rows**.

```text
Human-labeled power/respect dataset: 3,791 dialogues from 7 movies
Final translation comparison dataset: 599 dialogues
```

## Project Structure

```text
movie_social_re/
├── README.md
├── requirements.txt
├── configs/
│   ├── modeling_config.json
│   ├── movies.json
│   ├── power_respect_schema.json
│   ├── relationship_schema.json
│   ├── settings.json
│   └── translation_config.json
├── data/
│   ├── data_prelabel/
│   │   ├── amadeus.zip
│   │   ├── an_officer_and_a_gentleman.zip
│   │   ├── dead_poets_society.zip
│   │   ├── gladiator.zip
│   │   ├── mrs_brown.zip
│   │   ├── the_godfather.zip
│   │   └── titanic.zip
│   ├── data_prelabel_predictions/
│   │   ├── all_prelabel_risk_predictions.csv
│   │   ├── all_prelabel_risk_summary.csv
│   │   └── <movie_name>/
│   ├── raw/
│   │   └── <movie_name>/
│   ├── processed/
│   │   └── <movie_name>/re_clean.csv
│   ├── interim/
│   │   ├── <movie_name>/
│   │   ├── risk_gold_overlap.csv
│   │   └── translation_input.csv
│   ├── labeled/
│   │   └── power_respect_labels.csv
│   ├── modeling/
│   │   ├── power_respect_modeling_data.csv
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── translation_eval.csv
│   ├── graph/
│   │   ├── graph_summaries.csv
│   │   └── social_graph_edges.csv
│   └── translation_eval/
│       ├── translation_context_only.csv
│       ├── translation_power_respect_only.csv
│       ├── translation_relationship_only.csv
│       ├── translation_social_labels_only.csv
│       ├── translation_graph_guided.csv
│       ├── translation_comparison.csv
│       └── translation_ablation_comparison.csv
├── models/
│   ├── embeddings/
│   │   ├── train_bert_embeddings.npy
│   │   ├── val_bert_embeddings.npy
│   │   └── test_bert_embeddings.npy
│   ├── label_encoders.joblib
│   ├── power_dynamic_logreg.joblib
│   └── respect_level_logreg.joblib
├── outputs/
│   ├── modeling/
│   │   ├── logistic_regression_tuning.csv
│   │   ├── power_dynamic_classification_report.txt
│   │   ├── power_dynamic_confusion_matrix.csv
│   │   ├── power_dynamic_metrics.csv
│   │   ├── power_dynamic_predictions.csv
│   │   ├── respect_level_classification_report.txt
│   │   ├── respect_level_confusion_matrix.csv
│   │   ├── respect_level_metrics.csv
│   │   └── respect_level_predictions.csv
│   ├── reports/
│   │   └── translation_ablation_evaluation_plan.md
│   ├── tables/
│   │   ├── graph_summary_generation_stats.csv
│   │   ├── graph_summary_stats.csv
│   │   ├── overlap_summary.csv
│   │   ├── translation_ablation_summary.csv
│   │   ├── translation_comparison_summary.csv
│   │   └── translation_input_summary.csv
│   └── translation_eval/
│       ├── context_vs_graph_gemini31lite/
│       ├── social_vs_graph_gemini31lite/
│       ├── smoke_context_vs_graph/
│       ├── smoke_gemini31lite/
│       └── smoke_social_vs_graph_gemini31lite_retry/
├── prompts/
│   ├── graph_summary_generation.txt
│   ├── relationship_extraction.txt
│   ├── translate_context_only.txt
│   ├── translate_power_respect_only.txt
│   ├── translate_relationship_only.txt
│   ├── translate_social_labels_only.txt
│   ├── translate_graph_guided.txt
│   ├── translation_pairwise_judge.txt
│   └── translation_ablation_pairwise_judge.txt
├── scripts/
│   ├── run_all.py
│   ├── run_prelabel_risk_prediction.py
│   ├── run_translation_eval.py
│   ├── 01_data.py
│   ├── 02_run_all_movies_re.py
│   ├── 03_prepare_modeling_data.py
│   ├── 04_train_power_respect.py
│   ├── 05_tune_logistic_regression.py
│   ├── 06_prepare_risk_gold_overlap.py
│   ├── 07_build_social_graph.py
│   ├── 08_prepare_translation_input.py
│   ├── 09_translate_context_only.py
│   ├── 10_translate_graph_guided.py
│   ├── 11_merge_translation_outputs.py
│   ├── 12_translate_social_labels_only.py
│   ├── 13_merge_ablation_outputs.py
│   ├── 14_translate_power_respect_only.py
│   ├── 15_translate_relationship_only.py
│   └── 16_eval_translation_ablation.py
└── src/
    ├── analysis/
    │   ├── llm_pairwise_ablation_eval.py
    │   └── llm_pairwise_translation_eval.py
    ├── config.py
    ├── data/
    │   ├── build_turn_windows.py
    │   ├── extract_dialogue.py
    │   ├── prepare_risk_gold_overlap.py
    │   └── preprocess_dialogue.py
    ├── graph/
    │   └── build_graph.py
    ├── modeling/
    │   ├── embeddings.py
    │   ├── evaluate.py
    │   ├── prepare_data.py
    │   └── train.py
    ├── re/
    │   ├── generative_re.py
    │   ├── postprocess.py
    │   └── schema.py
    ├── translation/
    │   ├── merge_ablation_outputs.py
    │   ├── merge_translation_outputs.py
    │   ├── prepare_translation_input.py
    │   ├── translate_context_only.py
    │   ├── translate_power_respect_only.py
    │   ├── translate_relationship_only.py
    │   ├── translate_social_labels_only.py
    │   └── translate_graph_guided.py
    └── utils/
        ├── io.py
        └── paths.py
```

## Full Pipeline Overview

The full project is organized as numbered scripts in `scripts/`.

```text
01     Prepare movie dialogue data and turn-level metadata
02     Run relationship extraction for all movies
03     Prepare modeling data for power/respect classification
04     Train power/respect classifier
05     Tune logistic regression model
06     Prepare overlap between risk predictions, human gold labels, and relationship extraction
07     Build speaker-listener social relationship graph
08     Prepare translation input
09     Generate context-only translations
10     Generate graph-guided translations
11     Merge main translation outputs for comparison
12     Generate local social-label-guided ablation translations
14     Generate power/respect-only ablation translations
15     Generate relationship-only ablation translations
13     Merge ablation translation outputs
16     Evaluate translation ablation outputs with pairwise LLM judging
```

Steps 3-5 are the modeling part of the project. They are useful for exploring whether power dynamic and respect level can be predicted automatically. However, the final translation comparison uses **human-labeled gold power/respect annotations** from `data/labeled/power_respect_labels.csv`, rather than classifier predictions.

## Running the Full Project from Step 1

Run commands from the project root.

```bash
python scripts/01_data.py
python scripts/02_run_all_movies_re.py
python scripts/03_prepare_modeling_data.py
python scripts/04_train_power_respect.py
python scripts/05_tune_logistic_regression.py
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
python scripts/16_eval_translation_ablation.py
```

## Recommended Translation Pipeline

If the upstream dialogue files, risk predictions, relationship extraction outputs, and human annotations already exist, the main translation comparison pipeline can start from Step 6:

```bash
python scripts/06_prepare_risk_gold_overlap.py
python scripts/07_build_social_graph.py --recent-window 6
python scripts/08_prepare_translation_input.py
python scripts/09_translate_context_only.py
python scripts/10_translate_graph_guided.py
python scripts/11_merge_translation_outputs.py
```

Steps 9 and 10 support resume behavior. If translation output files already exist, the scripts skip rows that have already been translated unless `--overwrite` is used.

## Optional Ablation Study Pipeline

The ablation study separates different sources of social guidance to test which information changes translation behavior.

```text
A0: Context-only baseline
A1: Power/respect-only guidance
A2: Relationship-only guidance
A3: Local social-label guidance
A4: Full graph-guided translation
```

Run the ablation translations after Step 8 has created `data/interim/translation_input.csv`:

```bash
python scripts/12_translate_social_labels_only.py
python scripts/14_translate_power_respect_only.py
python scripts/15_translate_relationship_only.py
python scripts/13_merge_ablation_outputs.py
```

The ablation output is saved to:

```text
data/translation_eval/translation_ablation_comparison.csv
outputs/tables/translation_ablation_summary.csv
```

The ablation comparison includes these translation columns:

```text
translation_context_only
translation_power_respect_only
translation_relationship_only
translation_social_labels_only
translation_graph_guided
```

## Translation Evaluation Pipeline

The evaluation component compares translation outputs using pairwise LLM-as-a-judge evaluation. It supports the main context-only vs. graph-guided comparison and the ablation comparison between local social-label guidance and full graph-guided translation.

Evaluation scripts and prompts:

```text
scripts/run_translation_eval.py
scripts/16_eval_translation_ablation.py
prompts/translation_pairwise_judge.txt
prompts/translation_ablation_pairwise_judge.txt
src/analysis/llm_pairwise_translation_eval.py
src/analysis/llm_pairwise_ablation_eval.py
```

Run the ablation evaluation after `data/translation_eval/translation_ablation_comparison.csv` has been created:

```bash
python scripts/16_eval_translation_ablation.py
```

The evaluation outputs are saved under:

```text
outputs/translation_eval/
```

Important evaluation output folders include:

```text
outputs/translation_eval/context_vs_graph_gemini31lite/
outputs/translation_eval/social_vs_graph_gemini31lite/
```

Each evaluation folder contains files such as:

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

The current main evaluated comparisons are:

```text
Context-only vs. graph-guided translation
Local social-label-guided vs. graph-guided translation
```

## Modeling Part: Power and Respect Classification

The modeling part uses labeled examples to train and tune classifiers for power dynamic and respect level.

```bash
python scripts/03_prepare_modeling_data.py
python scripts/04_train_power_respect.py
python scripts/05_tune_logistic_regression.py
```

The modeling pipeline is useful for:

```text
1. testing whether power and respect can be predicted from dialogue features,
2. producing classifier baselines,
3. analyzing label separability and feature usefulness,
4. supporting future work where human labels may not be available.
```

In the final translation experiment, these classifier predictions are not used. Instead, the final power dynamic and respect level come from human annotation.

## Human Gold Annotation

The gold power dynamic and respect level labels are stored in:

```text
data/labeled/power_respect_labels.csv
```

Important columns include:

```text
movie_name
conversation_id
utterance_id
judge1_power_dynamic
judge2_power_dynamic
judge1_respect_level
judge2_respect_level
final_power_dynamic
final_respect_level
```

The final translation pipeline uses:

```text
final_power_dynamic
final_respect_level
```

These are treated as human-labeled gold annotations. Rows are kept only when the labels are present, agreed, and not `unclear`.

The human annotation dataset contains **3,791 dialogue rows** from **7 movies**.

## Important Inputs

### Risk prediction files

Risk prediction files identify dialogue lines that may require social or pragmatic translation care.

Example path pattern:

```text
data/data_prelabel_predictions/<movie_name>/dialogue_metadata_risk_predictions.csv
```

Target risk columns used in Step 6 include:

```text
_risk_social
_risk_pragmatic
_risk_register
_risk_ambiguity
```

### Relationship extraction files

Relationship extraction files provide speaker-listener relationship labels and evidence.

Example path pattern:

```text
data/processed/<movie_name>/re_clean.csv
```

Important columns include:

```text
movie_name
conversation_id
utterance_id
relationship_type
global_category
evidence
```

Rows with missing, blank, or `unclear` relationship type are removed from the final translation comparison.

## Main Outputs

### Step 3: Modeling data

The modeling data is prepared from the human-labeled power/respect annotation file. It is used for classifier training and tuning.

### Step 4: Power/respect model training

This step trains classifiers for predicting power dynamic and respect level.

### Step 5: Logistic regression tuning

This step tunes logistic regression models for the power/respect classification task.

### Step 6: Risk-gold overlap

```text
data/interim/risk_gold_overlap.csv
outputs/tables/overlap_summary.csv
```

This file keeps rows that satisfy all of the following:

```text
1. The row is marked as a target risk row.
2. The row has agreed human-labeled final power and respect labels.
3. The final power and respect labels are not missing or unclear.
4. The relationship type is not missing or unclear.
```

### Step 7: Social graph

```text
data/graph/social_graph_edges.csv
outputs/tables/graph_summary_stats.csv
```

This step builds a directed speaker-listener graph. Each row represents the social state of a speaker-listener edge after processing a dialogue turn.

Important graph columns include:

```text
aggregate_power_dynamic
aggregate_respect_level
aggregate_relationship_type
recent_power_dynamic
recent_respect_level
recent_relationship_type
edge_observation_count
recent_observation_count
recent_evidence_lines
```

### Step 8: Translation input

```text
data/interim/translation_input.csv
outputs/tables/translation_input_summary.csv
```

This file merges the selected risky dialogue rows with graph state information.

### Step 9: Context-only translation

```text
data/translation_eval/translation_context_only.csv
```

This file contains the baseline translations.

### Step 10: Graph-guided translation

```text
data/translation_eval/translation_graph_guided.csv
```

This file contains graph-guided translations and the model-generated social guidance summary.

### Step 11: Final comparison file

```text
data/translation_eval/translation_comparison.csv
outputs/tables/translation_comparison_summary.csv
```

The current final comparison file contains **599 dialogue rows** after cleaning and overlap filtering.

This is the main final output dataset for the next stage of the project:

```text
data/translation_eval/translation_comparison.csv
```

The final comparison file keeps only the necessary columns:

```text
movie_name
conversation_id
utterance_id
speaker_name
listener_name
current_text
context_6_turns
final_power_dynamic
final_respect_level
relationship_type
evidence
recent_power_dynamic
recent_respect_level
recent_relationship_type
aggregate_power_dynamic
aggregate_respect_level
aggregate_relationship_type
llm_social_guidance_summary
translation_context_only
translation_graph_guided
translation_model_context_only
translation_model_graph_guided
```

### Step 16: Translation evaluation outputs

```text
outputs/translation_eval/context_vs_graph_gemini31lite/
outputs/translation_eval/social_vs_graph_gemini31lite/
```

These folders contain pairwise LLM judgment results and summary files for the main translation comparisons. The evaluation compares candidate translations on meaning accuracy, social relationship faithfulness, register and tone, and Mandarin fluency.

## Prompts

Translation prompts are stored in `prompts/`.

```text
prompts/translate_context_only.txt
prompts/translate_power_respect_only.txt
prompts/translate_relationship_only.txt
prompts/translate_social_labels_only.txt
prompts/translate_graph_guided.txt
prompts/translation_pairwise_judge.txt
prompts/translation_ablation_pairwise_judge.txt
```

The context-only prompt uses only dialogue context and speaker/listener metadata.

The graph-guided prompt additionally uses:

```text
final_power_dynamic
final_respect_level
relationship_type
evidence
recent graph state
aggregate graph state
schema definitions
```

The graph-guided prompt asks the model to silently interpret the social guidance and return strict JSON with:

```json
{
  "social_guidance_summary": "...",
  "translation_graph_guided": "..."
}
```

The ablation prompts use restricted subsets of the social information:

```text
translate_power_respect_only.txt   -> power and respect labels only
translate_relationship_only.txt    -> relationship type and evidence only
translate_social_labels_only.txt   -> power, respect, relationship type, and evidence
translate_graph_guided.txt         -> local social labels plus recent/aggregate graph state
```

## Configuration Files

```text
configs/modeling_config.json
configs/movies.json
configs/power_respect_schema.json
configs/relationship_schema.json
configs/settings.json
configs/translation_config.json
```

`relationship_schema.json` contains movie-specific relationship definitions.

`power_respect_schema.json` contains shared definitions for power and respect labels.

`movies.json`, `settings.json`, `modeling_config.json`, and `translation_config.json` store project-level settings for movie processing, modeling, and translation.

## Environment Setup

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_api_key_here
```

Install required packages:

```bash
pip install -r requirements.txt
```

## Useful Commands

Print Step 6 paths:

```bash
python scripts/06_prepare_risk_gold_overlap.py --print-paths
```

Build graph with a six-turn recent window:

```bash
python scripts/07_build_social_graph.py --recent-window 6
```

Dry run context-only translation:

```bash
python scripts/09_translate_context_only.py --max-rows 3 --dry-run --overwrite
```

Dry run graph-guided translation:

```bash
python scripts/10_translate_graph_guided.py --max-rows 3 --dry-run --overwrite
```

Merge final comparison output:

```bash
python scripts/11_merge_translation_outputs.py
```

Dry run social-label-guided ablation translation:

```bash
python scripts/12_translate_social_labels_only.py --max-rows 3 --dry-run --overwrite
```

Dry run power/respect-only ablation translation:

```bash
python scripts/14_translate_power_respect_only.py --max-rows 3 --dry-run --overwrite
```

Dry run relationship-only ablation translation:

```bash
python scripts/15_translate_relationship_only.py --max-rows 3 --dry-run --overwrite
```

Merge ablation comparison output:

```bash
python scripts/13_merge_ablation_outputs.py
```

Run pairwise ablation evaluation:

```bash
python scripts/16_eval_translation_ablation.py
```

## When to Restart the Pipeline

If you update `power_respect_labels.csv`, restart the translation pipeline from Step 6:

```bash
python scripts/06_prepare_risk_gold_overlap.py
python scripts/07_build_social_graph.py --recent-window 6
python scripts/08_prepare_translation_input.py
python scripts/09_translate_context_only.py
python scripts/10_translate_graph_guided.py
python scripts/11_merge_translation_outputs.py
```

If you want to update the power/respect modeling results, rerun Steps 3-5:

```bash
python scripts/03_prepare_modeling_data.py
python scripts/04_train_power_respect.py
python scripts/05_tune_logistic_regression.py
```

If you change risk prediction outputs or relationship extraction outputs, rerun from the step that produces the changed file, then continue through Step 11.

## Notes

- The final power dynamic and respect level are human-labeled gold annotations.
- The power/respect classifier is included as a modeling experiment, but its predictions are not used in the final translation comparison.
- Step 6 removes rows with missing, blank, or `unclear` power/respect labels.
- Step 6 also removes rows with missing, blank, or `unclear` relationship type.
- Steps 9 and 10 can resume from existing translation outputs.
- Use `--overwrite` only when you want to regenerate existing translations.
- The ablation study is optional, but it can help identify whether improvements come from power/respect labels, relationship labels, local social labels, or graph history.
- The evaluation outputs are stored under `outputs/translation_eval/`.
- The main evaluation comparisons are context-only vs. graph-guided and local social-label-guided vs. graph-guided.
- The final comparison file is designed for manual and automatic evaluation of whether graph guidance changes translation quality, tone, and sociopragmatic faithfulness.