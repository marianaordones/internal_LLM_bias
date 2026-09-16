# OpinionQA demographic report: llama

- Result CSV: `C:\Users\mario\OneDrive - csr.ufmg.br\Documents\GitHub\internal_LLM_bias\results\demographic_opinionqa_llama.csv`
- Model ID: `meta-llama/Llama-2-13b-chat-hf`
- Comparison magnitude: `20`
- Matched model/human rows: `52380` / `71136`
- Rows without a mapped probe class: `17784`
- Rows without the matching human distribution: `108`
- Rows excluded because the question was not usefully ordinal: `864`
- Rows excluded because distributions could not be aligned: `0`
- Trailing non-ordinal model values removed: `1296`
- Trailing non-ordinal human values removed: `0`

Distances are ordinal Wasserstein distances after conditioning both model and human distributions on the options that have ordinal positions. Trailing refusal, don't-know, or other non-ordinal choices are excluded and the retained mass is renormalized.

## Unmapped model classes

- `age/adolescent`
- `age/child`
- `socioeco/mid`

## Tables

- [Per-question distances](distance_to_human.csv)
- [Regime means and 95% CIs](regime_summary_ci.csv)
- [Paired tests against neutral](paired_significance_tests.csv)
- [Best steering magnitudes](best_magnitudes.csv)
- [Gender separation by question](gender_separation_by_question.csv)
- [Gender distance by topic and regime](gender_topic_regime_summary.csv)
- [Model/human gap correlations](gap_correlations.csv)

## Figures

- [Distance to human groups with 95% CIs](distance_by_regime_ci.png)
- [Distance by steering magnitude](distance_by_magnitude.png)
- [Male–female separation by question type](gender_separation.png)
- [Gender distance by topic and regime](gender_topic_regimes.png)
- [Model subgroup gaps versus human subgroup gaps](gap_correlations.png)
- [Steered and declared distance panels](distance_panels.png)
