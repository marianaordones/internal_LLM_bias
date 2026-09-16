# Steering manipulation check: mistral

- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Existing OpinionQA results: `/mnt/users/mariana.soares/internal_LLM_bias/results/demographic_opinionqa_mistral.csv`
- Selected steering magnitude: `N=2`
- Identity test: three prompt variants per attribute and target profile.
- Steering layers and probe checkpoints are the same as in the demographic experiment.

## 1. Output coherence

A prespecified mechanical-degeneration alert was triggered: mean normalized entropy rose by at least 0.20.

| magnitude | n_questions | n_conditions | mean_normalized_entropy | mean_top_mass | mean_nonordinal_mass | mean_top_mass_ge_0_9 | mean_near_uniform | mean_low_entropy | delta_normalized_entropy_vs_n0 | delta_top_mass_vs_n0 | delta_nonordinal_mass_vs_n0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000 | 494.000 | 5928.000 | 0.141 | 0.921 | 0.008 | 0.747 | 0.006 | 0.662 | 0.000 | 0.000 | 0.000 |
| 2.000 | 494.000 | 5928.000 | 0.876 | 0.470 | 0.007 | 0.000 | 0.169 | 0.000 | 0.736 | -0.450 | -0.002 |

`normalized_entropy=0` indicates a point mass and `1` a uniform distribution. `top_mass` is the probability of the most likely answer. `nonordinal_mass` is mass assigned to trailing options without ordinal positions, such as don't-know. The original experiment removed an explicit final `Refused` option, so this CSV cannot estimate free-form refusal rates.

## 2. Behavioral self-identification

| attribute | strict_accuracy | target_probability | target_margin | unique_prediction_fraction | complete_collapse_rate | all_profiles_correct_rate | refusal_prediction_rate | pairwise_js |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age | 0.083 | 0.191 | 0.007 | 0.250 | 0.333 | 0.000 | 0.333 | 0.044 |
| education | 0.556 | 0.318 | 0.083 | 0.667 | 0.000 | 0.000 | 0.222 | 0.099 |
| gender | 0.000 | 0.279 | 0.028 | 0.000 | 1.000 | 0.000 | 1.000 | 0.002 |
| socioeco | 0.556 | 0.387 | 0.177 | 0.556 | 0.667 | 0.333 | 0.000 | 0.038 |

Accuracy requires each steered profile to select its matching identity. `unique_prediction_fraction=1` means the different targets produced all expected distinct demographic predictions. `complete_collapse_rate=1` means every target selected the same option within a prompt variant. The explicit no-identity option is counted as refusal and as an incorrect prediction.

### Confusion counts at the selected magnitude

| target_class | adult | college and more | high | high school | low | mid | refusal | some schooling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adolescent | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| adult | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| child | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| college and more | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 0 |
| female | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| high | 0 | 0 | 1 | 0 | 0 | 2 | 0 | 0 |
| high school | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 |
| low | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 |
| male | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| mid | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| older_adult | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| some schooling | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |

## 3. Short free-text coherence

| attribute | samples | empty_rate | repetition_alert_rate | mean_word_count |
| --- | --- | --- | --- | --- |
| age | 5.000 | 0.000 | 0.400 | 25.600 |
| education | 4.000 | 0.000 | 0.500 | 39.500 |
| gender | 3.000 | 0.000 | 0.667 | 52.333 |
| socioeco | 4.000 | 0.000 | 0.750 | 55.250 |

The repetition flags are mechanical screening rules, not a semantic evaluation. Inspect `free_text_samples.csv` before claiming that the outputs are coherent.

## Files

- `output_coherence_by_condition.csv`: one row per question, profile, and magnitude.
- `output_coherence_summary.csv`: summaries by attribute, profile, and magnitude.
- `output_coherence_by_magnitude.csv`: overall dose-response summary.
- `identity_predictions.csv`: direct identity distributions and predictions.
- `identity_summary.csv`: accuracy, distinctness, collapse, refusal, and JS divergence.
- `free_text_samples.csv`: generated samples and repetition diagnostics.
- `output_coherence.png` and `identity_distinctness.png`: diagnostic curves.

## Interpretation rule

The manipulation is supported only when the selected magnitude preserves ordinary output statistics and produces high matching-identity accuracy together with high distinctness across profiles. Low accuracy or a high complete-collapse rate means that a large steering effect should not be interpreted as installation of distinct demographic identities.
