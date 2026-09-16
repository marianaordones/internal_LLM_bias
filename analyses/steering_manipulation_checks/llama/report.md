# Steering manipulation check: llama

- Model: `meta-llama/Llama-2-13b-chat-hf`
- Existing OpinionQA results: `/mnt/users/mariana.soares/internal_LLM_bias/results/demographic_opinionqa_llama.csv`
- Selected steering magnitude: `N=20`
- Identity test: three prompt variants per attribute and target profile.
- Steering layers and probe checkpoints are the same as in the demographic experiment.

## 1. Output coherence

No prespecified mean-level alert was triggered at the selected magnitude.

| magnitude | n_questions | n_conditions | mean_normalized_entropy | mean_top_mass | mean_nonordinal_mass | mean_top_mass_ge_0_9 | mean_near_uniform | mean_low_entropy | delta_normalized_entropy_vs_n0 | delta_top_mass_vs_n0 | delta_nonordinal_mass_vs_n0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000 | 494.000 | 5928.000 | 0.692 | 0.622 | 0.002 | 0.077 | 0.063 | 0.004 | 0.000 | 0.000 | 0.000 |
| 20.000 | 494.000 | 5928.000 | 0.807 | 0.538 | 0.005 | 0.030 | 0.146 | 0.002 | 0.115 | -0.085 | 0.003 |

`normalized_entropy=0` indicates a point mass and `1` a uniform distribution. `top_mass` is the probability of the most likely answer. `nonordinal_mass` is mass assigned to trailing options without ordinal positions, such as don't-know. The original experiment removed an explicit final `Refused` option, so this CSV cannot estimate free-form refusal rates.

## 2. Behavioral self-identification

| attribute | strict_accuracy | target_probability | target_margin | unique_prediction_fraction | complete_collapse_rate | all_profiles_correct_rate | refusal_prediction_rate | pairwise_js |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age | 0.167 | 0.132 | -0.103 | 0.667 | 0.000 | 0.000 | 0.167 | 0.234 |
| education | 0.333 | 0.256 | -0.054 | 0.667 | 0.000 | 0.000 | 0.000 | 0.166 |
| gender | 1.000 | 0.566 | 0.329 | 1.000 | 0.000 | 1.000 | 0.000 | 0.096 |
| socioeco | 0.444 | 0.443 | 0.208 | 0.556 | 0.333 | 0.000 | 0.000 | 0.131 |

Accuracy requires each steered profile to select its matching identity. `unique_prediction_fraction=1` means the different targets produced all expected distinct demographic predictions. `complete_collapse_rate=1` means every target selected the same option within a prompt variant. The explicit no-identity option is counted as refusal and as an incorrect prediction.

### Confusion counts at the selected magnitude

| target_class | adolescent | adult | college and more | female | high | high school | low | male | mid | older_adult | refusal | some schooling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adolescent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 0 |
| adult | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| child | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| college and more | 0 | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| female | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| high | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| high school | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| low | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| male | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mid | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 |
| older_adult | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| some schooling | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 2 |

## 3. Short free-text coherence

| attribute | samples | empty_rate | repetition_alert_rate | mean_word_count |
| --- | --- | --- | --- | --- |
| age | 5.000 | 0.000 | 0.200 | 33.400 |
| education | 4.000 | 0.250 | 0.500 | 23.250 |
| gender | 3.000 | 0.000 | 0.667 | 46.667 |
| socioeco | 4.000 | 0.000 | 0.250 | 34.750 |

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
