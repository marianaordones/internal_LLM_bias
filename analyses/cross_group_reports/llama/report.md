# Cross-group specificity report: llama

- Input: `C:\Users\mario\OneDrive - csr.ufmg.br\Documents\GitHub\internal_LLM_bias\results\demographic_opinionqa_llama.csv`
- Model ID: `meta-llama/Llama-2-13b-chat-hf`
- Comparison magnitude: `20`
- Cross-group distance cells: `122256`
- Incomplete model/question choice sets excluded from classification: `72`
- Unmapped model rows: `17784`

The nearest-neighbor result is correct only when the uniquely closest human group matches the configured group for that model profile. Fractional accuracy assigns 1/k credit when the correct group is among k tied nearest groups. The specificity margin is own-group distance minus the mean distance to other groups; negative values indicate profile-specific alignment.

## Headline nearest-group accuracy

| Attribute | Regime | n | Strict accuracy | Fractional-tie accuracy | Chance | Lift* | Tie rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gender | Declared | 968 | 0.520 | 0.520 | 0.500 | +0.020 | 0.000 |
| gender | Neutral (N=0) | 968 | 0.500 | 0.500 | 0.500 | +0.000 | 0.000 |
| gender | Steered N=20 | 968 | 0.546 | 0.546 | 0.500 | +0.046 | 0.000 |
| age | Declared | 964 | 0.495 | 0.495 | 0.500 | -0.005 | 0.000 |
| age | Neutral (N=0) | 964 | 0.500 | 0.500 | 0.500 | +0.000 | 0.000 |
| age | Steered N=20 | 964 | 0.506 | 0.506 | 0.500 | +0.006 | 0.000 |
| education | Declared | 1458 | 0.341 | 0.341 | 0.333 | +0.008 | 0.000 |
| education | Neutral (N=0) | 1458 | 0.333 | 0.333 | 0.333 | +0.000 | 0.000 |
| education | Steered N=20 | 1458 | 0.326 | 0.326 | 0.333 | -0.007 | 0.000 |
| socioeco | Declared | 972 | 0.503 | 0.504 | 0.500 | +0.004 | 0.001 |
| socioeco | Neutral (N=0) | 972 | 0.499 | 0.500 | 0.500 | +0.000 | 0.002 |
| socioeco | Steered N=20 | 972 | 0.460 | 0.460 | 0.500 | -0.040 | 0.000 |

\* Lift uses fractional credit for tied nearest groups.

## Directional tests at the comparison regimes

Negative differences support specificity. `p_holm` is the one-sided Wilcoxon p-value corrected within attribute and condition.

| Attribute | Regime | Target | Expected vs competitor | n | Mean Δ | Fractional win rate | p_holm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| gender | Declared | Male | male vs female | 484 | +0.036 | 0.465 | 0.998 |
| gender | Declared | Female | female vs male | 484 | -0.051 | 0.533 | 0.0029 |
| gender | Neutral (N=0) | Male | male vs female | 484 | +0.000 | 0.500 | nan |
| gender | Neutral (N=0) | Female | female vs male | 484 | +0.000 | 0.500 | nan |
| gender | Steered N=20 | Male | male vs female | 484 | +0.064 | 0.409 | 1 |
| gender | Steered N=20 | Female | female vs male | 484 | -0.074 | 0.574 | 3.95e-07 |
| age | Declared | 30-49 | adult vs older_adult | 482 | +0.023 | 0.459 | 0.989 |
| age | Declared | 65+ | older_adult vs adult | 482 | -0.013 | 0.533 | 0.09 |
| age | Neutral (N=0) | 30-49 | adult vs older_adult | 482 | +0.000 | 0.500 | nan |
| age | Neutral (N=0) | 65+ | older_adult vs adult | 482 | +0.000 | 0.500 | nan |
| age | Steered N=20 | 30-49 | adult vs older_adult | 482 | -0.090 | 0.604 | 5.76e-12 |
| age | Steered N=20 | 65+ | older_adult vs adult | 482 | +0.090 | 0.384 | 1 |
| education | Declared | Less than high school | some schooling vs high school | 486 | +0.241 | 0.292 | 1 |
| education | Declared | Less than high school | some schooling vs college and more | 486 | +0.250 | 0.282 | 1 |
| education | Declared | High school graduate | high school vs some schooling | 486 | -0.223 | 0.693 | 1.01e-23 |
| education | Declared | High school graduate | high school vs college and more | 486 | +0.001 | 0.444 | 1 |
| education | Declared | College graduate/some postgrad | college and more vs some schooling | 486 | -0.248 | 0.714 | 1.75e-21 |
| education | Declared | College graduate/some postgrad | college and more vs high school | 486 | -0.014 | 0.570 | 0.0188 |
| education | Neutral (N=0) | Less than high school | some schooling vs high school | 486 | +0.000 | 0.500 | nan |
| education | Neutral (N=0) | Less than high school | some schooling vs college and more | 486 | +0.000 | 0.500 | nan |
| education | Neutral (N=0) | High school graduate | high school vs some schooling | 486 | +0.000 | 0.500 | nan |
| education | Neutral (N=0) | High school graduate | high school vs college and more | 486 | +0.000 | 0.500 | nan |
| education | Neutral (N=0) | College graduate/some postgrad | college and more vs some schooling | 486 | +0.000 | 0.500 | nan |
| education | Neutral (N=0) | College graduate/some postgrad | college and more vs high school | 486 | +0.000 | 0.500 | nan |
| education | Steered N=20 | Less than high school | some schooling vs high school | 486 | -0.124 | 0.665 | 4.62e-15 |
| education | Steered N=20 | Less than high school | some schooling vs college and more | 486 | +0.051 | 0.428 | 1 |
| education | Steered N=20 | High school graduate | high school vs some schooling | 486 | +0.143 | 0.329 | 1 |
| education | Steered N=20 | High school graduate | high school vs college and more | 486 | +0.207 | 0.329 | 1 |
| education | Steered N=20 | College graduate/some postgrad | college and more vs some schooling | 486 | -0.062 | 0.584 | 1.18e-05 |
| education | Steered N=20 | College graduate/some postgrad | college and more vs high school | 486 | -0.171 | 0.628 | 5.86e-14 |
| socioeco | Declared | Less than $30,000 | low vs high | 486 | -0.009 | 0.492 | 0.53 |
| socioeco | Declared | $100,000 or more | high vs low | 486 | -0.010 | 0.519 | 0.53 |
| socioeco | Neutral (N=0) | Less than $30,000 | low vs high | 486 | +0.000 | 0.500 | nan |
| socioeco | Neutral (N=0) | $100,000 or more | high vs low | 486 | +0.000 | 0.500 | nan |
| socioeco | Steered N=20 | Less than $30,000 | low vs high | 486 | +0.294 | 0.270 | 1 |
| socioeco | Steered N=20 | $100,000 or more | high vs low | 486 | -0.282 | 0.726 | 2.62e-32 |

## Unmapped classes

- `age/adolescent`
- `age/child`
- `socioeco/mid`

## Output files

- [Cross-group distances](cross_group_distances.csv)
- [Nearest predictions](nearest_group_predictions.csv)
- [Nearest accuracy](nearest_group_accuracy.csv)
- [Confusion matrices](nearest_group_confusions.csv)
- [Specificity margins](specificity_margins.csv)
- [Specificity margin summary](specificity_margin_summary.csv)
- [Directional Wilcoxon tests](directional_tests.csv)
- [Accuracy plot](nearest_group_accuracy.png)
- [Margin plot](specificity_margin.png)
- [Confusion plot](nearest_group_confusions.png)
