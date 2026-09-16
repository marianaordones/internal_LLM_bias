# Inferred gender from names: mistral

- Inferred-results input: `C:\Users\mario\OneDrive - csr.ufmg.br\Documents\GitHub\internal_LLM_bias\results\inferred_gender_names_mistral.csv`
- Baseline experiment: `C:\Users\mario\OneDrive - csr.ufmg.br\Documents\GitHub\internal_LLM_bias\results\demographic_opinionqa_mistral.csv`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Questions: `494`; names: `40` (20 male, 20 female)
- Minimum SSA sex-label dominance: `0.971`
- Steering reference magnitude: `2`

The response distribution is averaged across names within each sex label before human-alignment analyses. This keeps the unit comparable to the declared and steered profiles and prevents individual names from becoming pseudo-replicates.

## Headline

The across-layer majority vote recovers the name-associated gender with **97.5% accuracy immediately after the name**. Gender selectivity exceeds every unrelated probe on average, although its smallest advantage is only **+0.016** (gender vs socioeco, final_prompt).

At the response level, inferred identity changes matching-group distance by **Delta W=+0.012** relative to neutral. Nearest-group lift is **+0.001** overall and **+0.008** in Q4; the continuous trend is **OR=1.038** per SD of human divergence.

## 1. Gender inference by the probes

`Layer accuracy` averages correctness across layers. `Majority accuracy` asks whether the majority of layers predicts the SSA-associated gender.

| Position | Layer accuracy | Majority accuracy | Cases |
| --- | ---: | ---: | ---: |
| after_name | 0.610 | 0.975 | 40 |
| final_prompt | 0.527 | 0.594 | 19760 |

## 2. Gender versus unrelated probe selectivity

Selectivity is the normalized top-class confidence corrected for chance under each probe's number of classes. Positive differences favor gender. Because independently trained probes need not be calibrated identically, this is a relative diagnostic rather than a neuronal activation measure.

| Position | Comparison | Mean difference | Gender stronger | p (one-sided) |
| --- | --- | ---: | ---: | ---: |
| after_name | gender vs age | +0.195 | 1.000 | 9.09e-13 |
| after_name | gender vs education | +0.125 | 1.000 | 9.09e-13 |
| after_name | gender vs socioeco | +0.183 | 1.000 | 9.09e-13 |
| final_prompt | gender vs age | +0.078 | 0.993 | 0 |
| final_prompt | gender vs education | +0.026 | 0.714 | 0 |
| final_prompt | gender vs socioeco | +0.016 | 0.665 | 0 |

## 3. Distance to the matching human group

Negative delta values mean that the regime is closer to the matching human gender distribution than the neutral model.

| Regime | Mean W | Delta W vs neutral | 95% CI of delta | paired p |
| --- | ---: | ---: | ---: | ---: |
| Neutral (N=0) | 0.655 | +0.000 | ±0.000 | NA |
| Declared | 0.687 | +0.031 | ±0.017 | 0.0656 |
| Steered N=2 | 0.392 | -0.263 | ±0.027 | 1.32e-53 |
| Inferred from name | 0.667 | +0.012 | ±0.017 | 0.829 |

## 4. Cross-group specificity

Nearest-human-group strict accuracy is **0.501**; fractional-tie accuracy is **0.501** against chance **0.500** (lift **+0.001**).

## 5. Recovery as a function of human divergence

In Q4, fractional accuracy is **0.508**, with lift **+0.008** and bootstrap 95% CI [-0.008, +0.029].

The continuous logistic trend gives OR per human-gap SD = **1.038** (95% CI [0.962, 1.120]; one-sided cluster-robust p = 0.166).

## 6. Directional test in Q4

Negative own-minus-competitor values support gender-specific recovery; opposite signs across the two targets indicate collapse toward one pole.

| Human target | n | Mean own-minus-competitor | Win rate | Holm p |
| --- | ---: | ---: | ---: | ---: |
| Male | 121 | -0.015 | 0.521 | 0.393 |
| Female | 121 | -0.013 | 0.537 | 0.393 |

## Audit

- Cross-group cells: `1940`
- Incomplete nearest-neighbor sets excluded: `4`
- Model trailing non-ordinal values removed: `24`
- Human trailing non-ordinal values removed: `0`

## Output files

- [Probe case summary](probe_case_summary.csv)
- [Probe results by layer](probe_by_layer.csv)
- [Probe summary](probe_summary.csv)
- [Probe activation comparisons](probe_activation_comparisons.csv)
- [Inferred profiles](inferred_profiles.csv)
- [Matching-group paired distances](distance_to_matching_human.csv)
- [Distance summary](distance_vs_neutral_summary.csv)
- [Cross-group distances](cross_group_distances.csv)
- [Nearest-group predictions](nearest_group_predictions.csv)
- [Nearest-group accuracy](nearest_group_accuracy.csv)
- [Accuracy by human-gap quartile](accuracy_by_human_gap_quartile.csv)
- [Continuous human-gap trends](accuracy_gap_trends.csv)
- [Directional rows](directional_by_question.csv)
- [Directional Q4 tests](directional_tests_q4.csv)
- [Gender-probe accuracy plot](gender_probe_accuracy_by_layer.png)
- [Probe selectivity plot](probe_selectivity_by_attribute.png)
- [Distance delta plot](distance_delta_vs_neutral.png)
- [Nearest-group confusion plot](nearest_group_confusion.png)
- [Human-gap recovery plot](lift_by_human_gap_quartile.png)
- [Human-gap odds-ratio plot](human_gap_odds_ratio.png)
- [Directional Q4 plot](directional_specificity_q4.png)
