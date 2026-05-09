# Translation Ablation Evaluation Plan

## Scope

Dataset: `data/translation_eval/translation_ablation_comparison.csv`

The ablation comparison file contains 599 rows and 5 full translation variants:

- `translation_context_only`
- `translation_power_respect_only`
- `translation_relationship_only`
- `translation_social_labels_only`
- `translation_graph_guided`

All 599 rows contain all five translations.

## Recommended Evaluation Design

### Main comparison

Compare `context_only` vs `graph_guided`.

Reason:

- This is the cleanest test of the project's main research question: whether social graph guidance improves translation quality.
- It compares the baseline prompt against the full proposed method.
- In the current CSV, 467 / 599 rows (78.0%) differ between these two systems, so the comparison is informative.

### Key ablation comparison

Compare `social_labels_only` vs `graph_guided`.

Reason:

- This isolates the value of recent and aggregate graph history beyond local social labels.
- It directly tests whether graph structure/history contributes additional sociopragmatic information.
- In the current CSV, 454 / 599 rows (75.8%) differ between these two systems.

### Optional supplemental comparison

Compare `power_respect_only` vs `relationship_only`.

Reason:

- This helps explain whether improvements are driven more by politeness/power labels or by relationship labels and evidence.
- This is useful as a secondary interpretation analysis, but it should not replace the two comparisons above in the final report.

## Recommended Judging Method

Use pairwise `LLM-as-a-judge` with OpenAI as the evaluation model, plus manual audit.

Why this is the strongest write-up:

- It aligns with the project goal of evaluating subtle social and pragmatic translation effects.
- It avoids relying only on lexical-overlap metrics such as BLEU, which are weak for dialogue tone and politeness.
- It is consistent with the existing repository workflow and prompts.
- It supports a report claim such as: "Translations were generated with Gemini-family models, while evaluation used OpenAI as an external judge, followed by manual spot checking."

## Evaluation Criteria

Use four criteria in this order:

1. Meaning accuracy
2. Social relationship faithfulness
3. Register and tone
4. Mandarin fluency

This ordering should be kept explicit in the report so that the model is not rewarded for sounding polished while changing the source meaning.

## Human Review Recommendation

Add a manual audit on a subset of judged rows.

Recommended protocol:

- Randomly sample 30 to 50 rows from each main comparison.
- Ensure coverage across:
  - `high_respect`, `neutral_respect`, `low_respect`
  - `speaker_higher_power`, `listener_higher_power`, `roughly_equal_power`
  - frequent relationship types such as `friend`, `peer_friend`, `court_authority`, `romantic_intent`, and `adversarial`
- Mark whether the LLM judgment is clearly correct, debatable, or incorrect.

This supports a defensible statement such as:

"To improve reliability, we manually audited a stratified subset of LLM judgments and found that the judge was broadly aligned with human preferences on sociopragmatic distinctions."

## Suggested Report Framing

Use wording close to the following:

> We evaluated translation quality with a pairwise LLM-as-a-judge protocol using an OpenAI model as an external evaluator. The judge compared two Chinese translations of the same English dialogue line while considering dialogue context, gold social labels, and graph state metadata. Judgments prioritized meaning accuracy, sociopragmatic faithfulness, register/tone, and Mandarin fluency. To reduce position bias, candidate order was randomized. We additionally performed manual review on a stratified subset of examples to verify that the automatic judgments were reasonable.

If you use Gemini to generate the translations but OpenAI to evaluate them, you can write:

> All translation variants in the ablation file were generated with the same translation backbone family, while evaluation was conducted with an external OpenAI judge model to reduce same-model self-preference.

That sentence is accurate as long as the judge is actually run with OpenAI.

## Cheapest Practical API Plan

### Best low-cost report-friendly plan

Use OpenAI for evaluation only, not translation generation.

Recommended evaluation model:

- `gpt-4.1-mini`

Reason:

- cheap
- strong enough for structured pairwise judging
- easier to justify in the report as "OpenAI LLM-as-a-judge"

### Free or near-free fallback

Use Gemini free tier to generate translations or run internal pilot checks, but keep the final reported judge on OpenAI if possible.

Recommended fallback model for pilot runs:

- `gemini-2.5-flash` or `gemini-2.5-flash-lite` free tier, depending on availability in your account

If you must save OpenAI cost, run:

- full evaluation on changed rows only
- manual audit on a smaller subset

## Cost-Saving Advice

To minimize judge cost while keeping the experiment defensible:

1. Evaluate only rows where the two candidate translations differ.
2. Prioritize the two main comparisons instead of all five systems.
3. Run the optional third comparison only if time allows.
4. Keep the judge output in strict JSON and short reasons.

## Commands

Main experiment:

```bash
python scripts/16_eval_translation_ablation.py \
  --candidate-1-column translation_context_only \
  --candidate-1-label context_only \
  --candidate-2-column translation_graph_guided \
  --candidate-2-label graph_guided \
  --only-different \
  --model gpt-4.1-mini \
  --overwrite
```

Key ablation:

```bash
python scripts/16_eval_translation_ablation.py \
  --candidate-1-column translation_social_labels_only \
  --candidate-1-label social_labels_only \
  --candidate-2-column translation_graph_guided \
  --candidate-2-label graph_guided \
  --only-different \
  --model gpt-4.1-mini \
  --overwrite
```

Optional supplemental comparison:

```bash
python scripts/16_eval_translation_ablation.py \
  --candidate-1-column translation_power_respect_only \
  --candidate-1-label power_respect_only \
  --candidate-2-column translation_relationship_only \
  --candidate-2-label relationship_only \
  --only-different \
  --model gpt-4.1-mini \
  --overwrite
```

## What to Report

For each comparison, report:

- total judged rows
- wins / losses / ties
- win rate
- bootstrap confidence interval
- per-criterion wins for meaning, social relationship, register/tone, and fluency
- breakdown by `final_respect_level`, `final_power_dynamic`, and `relationship_type`

## Expected Interpretation Logic

If `graph_guided` beats `context_only`, the full social graph guidance helps overall dialogue translation.

If `graph_guided` also beats `social_labels_only`, then the improvement is not just from local labels; it suggests recent and aggregate graph context adds useful sociopragmatic signal.

If `power_respect_only` and `relationship_only` behave differently, that helps explain which component is doing more of the work.
