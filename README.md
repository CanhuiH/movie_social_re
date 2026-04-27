

# movie_social_re

A movie-dialogue NLP pipeline for **social-context-aware translation**.

This project builds a workflow for:

- extracting movie dialogue from ConvoKit
- preprocessing and building turn-level context windows
- using **Generative Relation Extraction (RE)** to infer speaker-listener relationships
- building a lightweight directed graph of relationships
- generating **two translation conditions**
  - **context only**
  - **context + graph information**
- exporting a comparison CSV for downstream analysis

The current working movie is **Titanic**, but the codebase is organized so it can support other movies later.

---

## Project goal

The goal is to test whether **relationship-aware social context** can help improve movie dialogue translation.

For each dialogue turn, the pipeline:

1. builds a short local dialogue window
2. predicts the relationship between speaker and listener
3. stores the result in a lightweight graph
4. converts that graph information into a short social summary
5. translates the same turn under two conditions:
   - using only local dialogue context
   - using local dialogue context plus graph-derived social context

This makes it possible to compare whether graph information helps translation quality.

---

## Current pipeline

```text
ConvoKit movie dialogue
    -> extraction
    -> preprocessing
    -> turn-window construction
    -> Generative RE
    -> RE postprocessing
    -> relationship graph
    -> social summaries
    -> translation inputs
    -> translation generation
    -> translation comparison CSV
```

---

## Project structure

```text
movie_social_re/
├── data/
│   ├── raw/
│   ├── interim/
│   │   └── Titanic/
│   └── processed/
│       └── Titanic/
├── prompts/
│   ├── relationship_extraction.txt
│   ├── translation_context_only.txt
│   └── translation_with_graph.txt
├── scripts/
│   ├── run_extract.py
│   ├── run_preprocess.py
│   ├── run_build_windows.py
│   ├── run_re.py
│   ├── run_re_postprocess.py
│   ├── run_graph.py
│   ├── run_translation_inputs.py
│   ├── run_translate.py
│   └── run_translation_comparison.py
├── src/
│   ├── analysis/
│   ├── data/
│   ├── graph/
│   ├── re/
│   ├── translation/
│   └── utils/
├── .env
├── requirements.txt
└── README.md
```

---

## Environment setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Add your OpenAI API key

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_key_here
```

---

## Recommended models

### Relation extraction
- **best quality:** `gpt-5.5`
- **cheaper / faster:** `gpt-5.4-mini`

### Translation
- **best quality:** `gpt-5.5`
- **cheaper / faster:** `gpt-5.4-mini`

For debugging and pilot runs, smaller models are usually enough. For final comparisons, use the **same model** across both translation conditions.

---

## Running order

Run these commands from the project root.

### 0. Activate the environment

```bash
source .venv/bin/activate
```

### 1. Extract movie dialogue

```bash
python scripts/run_extract.py --movie "Titanic"
```

Creates:
- `data/interim/Titanic/dialogue_metadata.csv`
- `data/interim/Titanic/dialogue_metadata_with_listener.csv`

### 2. Preprocess dialogue

```bash
python scripts/run_preprocess.py --movie "Titanic"
```

Creates:
- `data/interim/Titanic/dialogue_metadata_clean.csv`

### 3. Build turn windows

```bash
python scripts/run_build_windows.py --movie "Titanic"
```

Creates:
- `data/interim/Titanic/turn_windows.csv`

Default context window: **3 previous turns**.

To use a different context window:

```bash
python scripts/run_build_windows.py --movie "Titanic" --context-window 5
```

### 4. Run Generative RE

Small debug run:

```bash
python scripts/run_re.py --movie "Titanic" --max-rows 20 --overwrite
```

Full run:

```bash
python scripts/run_re.py --movie "Titanic" --overwrite
```

Or choose a model explicitly:

```bash
python scripts/run_re.py --movie "Titanic" --model "gpt-5.5" --overwrite
```

Creates:
- `data/processed/Titanic/re_llm_outputs.jsonl`

### 5. Postprocess RE outputs

```bash
python scripts/run_re_postprocess.py --movie "Titanic"
```

Creates:
- `data/processed/Titanic/re_clean.csv`
- `data/processed/Titanic/re_review_sheet.csv`

### 6. Build graph and social summaries

```bash
python scripts/run_graph.py --movie "Titanic"
```

Creates:
- `data/processed/Titanic/graph_nodes.csv`
- `data/processed/Titanic/graph_edges.csv`
- `data/processed/Titanic/social_summaries.csv`

### 7. Build translation inputs

#### Context-only baseline

```bash
python scripts/run_translation_inputs.py --movie "Titanic" --mode context_only
```

Creates:
- `data/processed/Titanic/translation_inputs_context_only.jsonl`

#### With graph information

```bash
python scripts/run_translation_inputs.py --movie "Titanic" --mode with_graph
```

Creates:
- `data/processed/Titanic/translation_inputs_with_graph.jsonl`

### 8. Run translation

#### Context-only baseline

```bash
python scripts/run_translate.py --movie "Titanic" --mode context_only --model "gpt-5.4-mini" --overwrite
```

#### With graph information

```bash
python scripts/run_translate.py --movie "Titanic" --mode with_graph --model "gpt-5.4-mini" --overwrite
```

Creates:
- `data/processed/Titanic/translations_context_only.jsonl`
- `data/processed/Titanic/translations_with_graph.jsonl`

### 9. Build comparison CSV

```bash
python scripts/run_translation_comparison.py --movie "Titanic"
```

Creates:
- `data/processed/Titanic/translation_comparison.csv`

---

## Key intermediate and final files

### Interim files
- `dialogue_metadata.csv`
- `dialogue_metadata_clean.csv`
- `turn_windows.csv`

### RE files
- `re_llm_outputs.jsonl`
- `re_clean.csv`
- `re_review_sheet.csv`

### Graph files
- `graph_nodes.csv`
- `graph_edges.csv`
- `social_summaries.csv`

### Translation files
- `translation_inputs_context_only.jsonl`
- `translation_inputs_with_graph.jsonl`
- `translations_context_only.jsonl`
- `translations_with_graph.jsonl`

### Analysis file
- `translation_comparison.csv`

---

## Translation comparison CSV

The comparison CSV merges:

- dialogue metadata
- local context
- relationship label
- graph summary
- translation from the context-only condition
- translation from the with-graph condition

Typical columns include:

- `movie_name`
- `conversation_id`
- `utterance_id`
- `speaker_name`
- `listener_name`
- `current_turn`
- `current_text`
- `context_text`
- `relationship_type`
- `graph_summary`
- `translation_context_only`
- `translation_with_graph`

This file is intended for qualitative comparison and downstream analysis.

---

## Prompt files

### Relationship extraction
- `prompts/relationship_extraction.txt`

### Translation
- `prompts/translation_context_only.txt`
- `prompts/translation_with_graph.txt`

Use separate translation prompts so the ablation remains clean:
- one prompt without graph information
- one prompt with graph information

---

## Notes

### Why use short local windows?
Relationship extraction is performed for a **specific target turn**, using a **small local dialogue chunk** as evidence. This is better than feeding the entire feature-length script for each decision.

### Why build a graph?
The graph acts as a lightweight social memory layer. It stores the most recent relation state between speaker and listener and helps inject structured social context into translation.

### Does the graph fully model time?
Not completely. The current graph stores the **latest edge state** with fields like:
- `conversation_id`
- `utterance_id_last_updated`

These provide partial temporal information, but not a full relationship history.

---

## Rerun guidance

### If you change the extraction logic
Rerun everything.

### If you change preprocessing
Rerun from preprocessing onward.

### If you change context window size
Rerun from turn-window construction onward.

### If you change the RE model or RE prompt
Rerun from Generative RE onward:
- `run_re.py`
- `run_re_postprocess.py`
- `run_graph.py`
- `run_translation_inputs.py --mode with_graph`
- `run_translate.py --mode with_graph`
- `run_translation_comparison.py`

### If you change translation prompts or translation model
Rerun translation and comparison:
- `run_translate.py`
- `run_translation_comparison.py`

---

## Current status

The current pipeline supports:
- end-to-end RE generation
- graph construction
- graph-aware translation input generation
- translation generation under two conditions
- comparison CSV export

This makes the project ready for qualitative comparison of:
- **translation with local context only**
- **translation with local context + graph-derived relationship information**