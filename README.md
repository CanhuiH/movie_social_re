# Movie Social Relationship Translation Project

This project studies whether explicit social relationship information can improve Mandarin subtitle translation for English movie dialogue. The main comparison is between a context-only translation baseline and a graph-guided translation condition that uses power, respect, and speaker-listener relationship signals.

The project focuses on dialogue lines where social context may affect translation choices, such as politeness, formality, intimacy, resistance, sarcasm, deference, or relationship-specific wording.

## Project Goal

The goal is to evaluate whether adding structured social context helps an LLM produce more sociopragmatically appropriate Mandarin translations.

The pipeline compares two translation settings:

1. **Context-only translation**
   - Uses the current English line, speaker/listener metadata, and previous dialogue context.
   - Does not use social labels or graph information.

2. **Graph-guided translation**
   - Uses the same context as the baseline.
   - Adds structured social guidance, including power dynamic, respect level, relationship type, evidence, recent graph state, and aggregate graph state.


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
│   ├── data_prelabel_predictions/
│   │   └── <movie_name>/
│   ├── graph/
│   │   ├── graph_summaries.csv
│   │   └── social_graph_edges.csv
│   └── translation_eval/
│       ├── translation_context_only.csv
│       ├── translation_graph_guided.csv
│       └── translation_comparison.csv
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
│   └── tables/
│       ├── graph_summary_generation_stats.csv
│       ├── graph_summary_stats.csv
│       ├── overlap_summary.csv
│       ├── translation_comparison_summary.csv
│       └── translation_input_summary.csv
├── prompts/
│   ├── graph_summary_generation.txt
│   ├── relationship_extraction.txt
│   ├── translate_context_only.txt
│   └── translate_graph_guided.txt
├── scripts/
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
│   └── 11_merge_translation_outputs.py
└── src/
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
    │   ├── merge_translation_outputs.py
    │   ├── prepare_translation_input.py
    │   ├── translate_context_only.py
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
11     Merge translation outputs for comparison
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

## Prompts

Translation prompts are stored in `prompts/`.

```text
prompts/translate_context_only.txt
prompts/translate_graph_guided.txt
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

## TODO / Next Steps

The current pipeline produces the final translation comparison dataset:

```text
data/translation_eval/translation_comparison.csv
```

The next teammate can continue from this file and focus on the evaluation and report-writing stages.

Suggested next tasks:

```text
1. Design an evaluation rubric for comparing context-only and graph-guided translations.
2. Evaluate whether graph guidance improves tone, politeness, register, relationship faithfulness, and subtitle naturalness.
3. Add human or LLM-based judging for translation quality comparison.
4. Analyze cases where graph-guided translation improves, hurts, or does not change the output.
5. Summarize quantitative results from the 599-row comparison dataset.
6. Select representative qualitative examples for the final report.
7. Write the final project report, including motivation, data, methodology, experiments, results, limitations, and future work.
```

The modeling outputs in `outputs/modeling/` can be discussed as an exploratory component, but the final translation evaluation should use the human-labeled gold power/respect annotations included in `translation_comparison.csv`.

## Notes

- The final power dynamic and respect level are human-labeled gold annotations.
- The power/respect classifier is included as a modeling experiment, but its predictions are not used in the final translation comparison.
- Step 6 removes rows with missing, blank, or `unclear` power/respect labels.
- Step 6 also removes rows with missing, blank, or `unclear` relationship type.
- Steps 9 and 10 can resume from existing translation outputs.
- Use `--overwrite` only when you want to regenerate existing translations.
- The final comparison file is designed for manual and automatic evaluation of whether graph guidance changes translation quality, tone, and sociopragmatic faithfulness.