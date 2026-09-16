# Inferred income from occupations: mistral

- Input: `C:\Users\mario\OneDrive - csr.ufmg.br\Documents\GitHub\internal_LLM_bias\results\inferred_income_occupations_mistral_extreme5.csv`
- Baseline: `C:\Users\mario\OneDrive - csr.ufmg.br\Documents\GitHub\internal_LLM_bias\results\demographic_opinionqa_mistral.csv`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Questions: `494`; occupations: `10` (5 lower-wage, 5 higher-wage)
- Mean occupational wages: `$40,258` and `$68,644`
- Steering comparison magnitude: `2`

Answer distributions are averaged across occupations within wage group before comparison with humans. The labels are relative occupational-wage groups; they are not direct observations of the OpinionQA household-income endpoints.

## Headline

Across-layer majority accuracy is **70.0% after the occupation cue** and **25.7% at the final prompt token**. The weakest target-probe advantage is **-0.085** (socioeco vs gender, after_occupation, high).

At the answer level, occupation inference changes matching-group distance by **Delta W=-0.007**. Nearest-group lift is **-0.003** overall and **+0.004** in Q4. The continuous trend is **OR=0.976** per SD of human divergence.

## 1. Socioeconomic probe recovery

| Position | Wage group | Layer accuracy | Majority accuracy | Cases |
| --- | --- | ---: | ---: | ---: |
| after_occupation | low | 0.418 | 0.400 | 5 |
| final_prompt | low | 0.097 | 0.003 | 2470 |
| after_occupation | high | 0.697 | 1.000 | 5 |
| final_prompt | high | 0.452 | 0.512 | 2470 |

## 2. Socioeconomic probe versus placebo probes

Positive differences favor the socioeconomic probe. Chance-corrected selectivity is a relative diagnostic; independently trained probes may differ in calibration.

| Position | Wage group | Comparison | Mean difference | Socioeconomic stronger | p (one-sided) |
| --- | --- | --- | ---: | ---: | ---: |
| after_occupation | low | socioeco vs education | -0.033 | 0.200 | 0.938 |
| after_occupation | low | socioeco vs gender | -0.083 | 0.200 | 0.969 |
| after_occupation | low | socioeco vs age | +0.027 | 1.000 | 0.0312 |
| final_prompt | low | socioeco vs education | +0.008 | 0.568 | 2.39e-26 |
| final_prompt | low | socioeco vs gender | -0.018 | 0.321 | 1 |
| final_prompt | low | socioeco vs age | +0.058 | 0.963 | 0 |
| after_occupation | high | socioeco vs education | -0.004 | 0.400 | 0.594 |
| after_occupation | high | socioeco vs gender | -0.085 | 0.000 | 1 |
| after_occupation | high | socioeco vs age | +0.042 | 1.000 | 0.0312 |
| final_prompt | high | socioeco vs education | +0.011 | 0.595 | 4.32e-36 |
| final_prompt | high | socioeco vs gender | -0.010 | 0.392 | 1 |
| final_prompt | high | socioeco vs age | +0.060 | 0.953 | 0 |

## 3. Distance to matching human income group

Negative Delta W means closer than the neutral model.

| Regime | Mean W | Delta W | 95% CI of Delta | paired p |
| --- | ---: | ---: | ---: | ---: |
| Neutral (N=0) | 0.653 | +0.000 | ±0.000 | NA |
| Declared | 0.686 | +0.033 | ±0.016 | 0.299 |
| Steered N=2 | 0.341 | -0.312 | ±0.027 | 3.59e-61 |
| Inferred from occupation | 0.646 | -0.007 | ±0.014 | 0.0076 |

## 4. Cross-group specificity

Strict nearest-group accuracy is **0.496** and fractional-tie accuracy is **0.497**, against chance **0.500** (lift **-0.003**).

## 5. Human-gap trend

Q4 fractional accuracy is **0.504**, lift **+0.004**, bootstrap 95% CI **[-0.020, +0.029]**.

OR per gap SD is **0.976** (95% CI [0.900, 1.058], one-sided cluster-robust p=0.726).

## 6. Directional test in Q4

Negative own-minus-competitor values indicate the correct direction.

| Human target | n | Mean difference | Win rate | Holm p |
| --- | ---: | ---: | ---: | ---: |
| Less than $30,000 | 122 | +0.011 | 0.492 | 0.799 |
| $100,000 or more | 122 | -0.018 | 0.516 | 0.799 |

## Audit

- Cross-group cells: `1944`
- Incomplete choice sets excluded: `0`
- Model trailing values removed: `24`
- Human trailing values removed: `0`

## Output files

- [Probe cases](probe_case_summary.csv)
- [Probe accuracy by occupation](probe_accuracy_by_occupation.csv)
- [Probe results by layer](probe_by_layer.csv)
- [Probe summary](probe_summary.csv)
- [Probe placebo comparisons](probe_activation_comparisons.csv)
- [Inferred profiles](inferred_profiles.csv)
- [Matching-group distances](distance_to_matching_human.csv)
- [Distance summary](distance_vs_neutral_summary.csv)
- [Cross-group distances](cross_group_distances.csv)
- [Nearest-group predictions](nearest_group_predictions.csv)
- [Nearest-group accuracy](nearest_group_accuracy.csv)
- [Accuracy by human-gap quartile](accuracy_by_human_gap_quartile.csv)
- [Continuous human-gap trends](accuracy_gap_trends.csv)
- [Directional rows](directional_by_question.csv)
- [Directional Q4 tests](directional_tests_q4.csv)
- [Probe accuracy plot](socioeco_probe_accuracy_by_layer.png)
- [Probe placebo plot](probe_selectivity_by_attribute.png)
- [Distance plot](distance_delta_vs_neutral.png)
- [Nearest-group confusion](nearest_group_confusion.png)
- [Human-gap recovery](lift_by_human_gap_quartile.png)
- [Human-gap odds ratio](human_gap_odds_ratio.png)
- [Directional Q4](directional_specificity_q4.png)
