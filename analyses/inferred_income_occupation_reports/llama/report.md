# Inferred income from occupations: llama

- Input: `C:\Users\mario\OneDrive - csr.ufmg.br\Documents\GitHub\internal_LLM_bias\results\inferred_income_occupations_llama_extreme5.csv`
- Baseline: `C:\Users\mario\OneDrive - csr.ufmg.br\Documents\GitHub\internal_LLM_bias\results\demographic_opinionqa_llama.csv`
- Model: `meta-llama/Llama-2-13b-chat-hf`
- Questions: `494`; occupations: `10` (5 lower-wage, 5 higher-wage)
- Mean occupational wages: `$40,258` and `$68,644`
- Steering comparison magnitude: `20`

Answer distributions are averaged across occupations within wage group before comparison with humans. The labels are relative occupational-wage groups; they are not direct observations of the OpinionQA household-income endpoints.

## Headline

Across-layer majority accuracy is **50.0% after the occupation cue** and **17.4% at the final prompt token**. The weakest target-probe advantage is **-0.131** (socioeco vs gender, final_prompt, high).

At the answer level, occupation inference changes matching-group distance by **Delta W=+0.006**. Nearest-group lift is **+0.007** overall and **+0.025** in Q4. The continuous trend is **OR=1.045** per SD of human divergence.

## 1. Socioeconomic probe recovery

| Position | Wage group | Layer accuracy | Majority accuracy | Cases |
| --- | --- | ---: | ---: | ---: |
| after_occupation | low | 0.605 | 1.000 | 5 |
| final_prompt | low | 0.286 | 0.291 | 2470 |
| after_occupation | high | 0.176 | 0.000 | 5 |
| final_prompt | high | 0.299 | 0.057 | 2470 |

## 2. Socioeconomic probe versus placebo probes

Positive differences favor the socioeconomic probe. Chance-corrected selectivity is a relative diagnostic; independently trained probes may differ in calibration.

| Position | Wage group | Comparison | Mean difference | Socioeconomic stronger | p (one-sided) |
| --- | --- | --- | ---: | ---: | ---: |
| after_occupation | low | socioeco vs education | -0.011 | 0.200 | 0.781 |
| after_occupation | low | socioeco vs gender | -0.061 | 0.000 | 1 |
| after_occupation | low | socioeco vs age | +0.003 | 0.400 | 0.5 |
| final_prompt | low | socioeco vs education | +0.022 | 0.819 | 4.74e-267 |
| final_prompt | low | socioeco vs gender | -0.127 | 0.000 | 1 |
| final_prompt | low | socioeco vs age | -0.006 | 0.398 | 1 |
| after_occupation | high | socioeco vs education | -0.008 | 0.200 | 0.844 |
| after_occupation | high | socioeco vs gender | -0.080 | 0.000 | 1 |
| after_occupation | high | socioeco vs age | -0.036 | 0.000 | 1 |
| final_prompt | high | socioeco vs education | +0.008 | 0.662 | 4.99e-71 |
| final_prompt | high | socioeco vs gender | -0.131 | 0.000 | 1 |
| final_prompt | high | socioeco vs age | -0.019 | 0.191 | 1 |

## 3. Distance to matching human income group

Negative Delta W means closer than the neutral model.

| Regime | Mean W | Delta W | 95% CI of Delta | paired p |
| --- | ---: | ---: | ---: | ---: |
| Neutral (N=0) | 0.620 | +0.000 | ±0.000 | NA |
| Declared | 0.621 | +0.001 | ±0.029 | 0.834 |
| Steered N=20 | 0.564 | -0.056 | ±0.037 | 0.0989 |
| Inferred from occupation | 0.626 | +0.006 | ±0.024 | 0.127 |

## 4. Cross-group specificity

Strict nearest-group accuracy is **0.506** and fractional-tie accuracy is **0.507**, against chance **0.500** (lift **+0.007**).

## 5. Human-gap trend

Q4 fractional accuracy is **0.525**, lift **+0.025**, bootstrap 95% CI **[+0.000, +0.049]**.

OR per gap SD is **1.045** (95% CI [0.975, 1.120], one-sided cluster-robust p=0.108).

## 6. Directional test in Q4

Negative own-minus-competitor values indicate the correct direction.

| Human target | n | Mean difference | Win rate | Holm p |
| --- | ---: | ---: | ---: | ---: |
| Less than $30,000 | 122 | +0.029 | 0.385 | 0.994 |
| $100,000 or more | 122 | -0.032 | 0.623 | 0.002 |

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
