# Steering manipulation check: qwen

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Existing OpinionQA results: `/mnt/users/mariana.soares/internal_LLM_bias/results/demographic_opinionqa_qwen.csv`
- Selected steering magnitude: `N=20`
- Identity test: three prompt variants per attribute and target profile.
- Steering layers and probe checkpoints are the same as in the demographic experiment.

## 1. Output coherence

No prespecified mean-level alert was triggered at the selected magnitude.

| magnitude | n_questions | n_conditions | mean_normalized_entropy | mean_top_mass | mean_nonordinal_mass | mean_top_mass_ge_0_9 | mean_near_uniform | mean_low_entropy | delta_normalized_entropy_vs_n0 | delta_top_mass_vs_n0 | delta_nonordinal_mass_vs_n0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000 | 494.000 | 5928.000 | 0.094 | 0.949 | 0.001 | 0.830 | 0.000 | 0.757 | 0.000 | 0.000 | 0.000 |
| 20.000 | 494.000 | 5928.000 | 0.209 | 0.886 | 0.003 | 0.665 | 0.004 | 0.502 | 0.116 | -0.063 | 0.002 |

`normalized_entropy=0` indicates a point mass and `1` a uniform distribution. `top_mass` is the probability of the most likely answer. `nonordinal_mass` is mass assigned to trailing options without ordinal positions, such as don't-know. The original experiment removed an explicit final `Refused` option, so this CSV cannot estimate free-form refusal rates.

## 2. Behavioral self-identification

| attribute | strict_accuracy | target_probability | target_margin | unique_prediction_fraction | complete_collapse_rate | all_profiles_correct_rate | refusal_prediction_rate | pairwise_js |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age | 0.250 | 0.173 | -0.102 | 0.250 | 1.000 | 0.000 | 0.000 | 0.064 |
| education | 0.333 | 0.309 | -0.033 | 0.333 | 1.000 | 0.000 | 0.000 | 0.048 |
| gender | 0.000 | 0.029 | 0.028 | 0.000 | 1.000 | 0.000 | 1.000 | 0.020 |
| socioeco | 0.333 | 0.341 | 0.122 | 0.333 | 0.333 | 0.000 | 0.222 | 0.311 |

Accuracy requires each steered profile to select its matching identity. `unique_prediction_fraction=1` means the different targets produced all expected distinct demographic predictions. `complete_collapse_rate=1` means every target selected the same option within a prompt variant. The explicit no-identity option is counted as refusal and as an incorrect prediction.

### Confusion counts at the selected magnitude

| target_class | adult | college and more | mid | refusal |
| --- | --- | --- | --- | --- |
| adolescent | 3 | 0 | 0 | 0 |
| adult | 3 | 0 | 0 | 0 |
| child | 3 | 0 | 0 | 0 |
| college and more | 0 | 3 | 0 | 0 |
| female | 0 | 0 | 0 | 3 |
| high | 0 | 0 | 1 | 2 |
| high school | 0 | 3 | 0 | 0 |
| low | 0 | 0 | 3 | 0 |
| male | 0 | 0 | 0 | 3 |
| mid | 0 | 0 | 3 | 0 |
| older_adult | 3 | 0 | 0 | 0 |
| some schooling | 0 | 3 | 0 | 0 |

## 3. Short free-text coherence

| attribute | samples | empty_rate | repetition_alert_rate | mean_word_count |
| --- | --- | --- | --- | --- |
| age | 5.000 | 0.000 | 0.400 | 50.400 |
| education | 4.000 | 0.000 | 0.250 | 40.250 |
| gender | 3.000 | 0.000 | 0.000 | 27.333 |
| socioeco | 4.000 | 0.000 | 0.000 | 42.750 |

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
