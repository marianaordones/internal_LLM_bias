# Cross-group specificity report: mistral

- Input: `C:\Users\mario\OneDrive - csr.ufmg.br\Documents\GitHub\internal_LLM_bias\results\demographic_opinionqa_mistral.csv`
- Model ID: `mistralai/Mistral-7B-Instruct-v0.3`
- Comparison magnitude: `2`
- Cross-group distance cells: `122256`
- Incomplete model/question choice sets excluded from classification: `72`
- Unmapped model rows: `17784`

The nearest-neighbor result is correct only when the uniquely closest human group matches the configured group for that model profile. Fractional accuracy assigns 1/k credit when the correct group is among k tied nearest groups. The specificity margin is own-group distance minus the mean distance to other groups; negative values indicate profile-specific alignment.

## Headline nearest-group accuracy

| Attribute | Regime | n | Strict accuracy | Fractional-tie accuracy | Chance | Lift* | Tie rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gender | Declared | 968 | 0.511 | 0.511 | 0.500 | +0.011 | 0.000 |
| gender | Neutral (N=0) | 968 | 0.500 | 0.500 | 0.500 | +0.000 | 0.000 |
| gender | Steered N=2 | 968 | 0.456 | 0.456 | 0.500 | -0.044 | 0.000 |
| age | Declared | 964 | 0.503 | 0.503 | 0.500 | +0.003 | 0.000 |
| age | Neutral (N=0) | 964 | 0.500 | 0.500 | 0.500 | +0.000 | 0.000 |
| age | Steered N=2 | 964 | 0.513 | 0.513 | 0.500 | +0.013 | 0.000 |
| education | Declared | 1458 | 0.344 | 0.344 | 0.333 | +0.011 | 0.000 |
| education | Neutral (N=0) | 1458 | 0.333 | 0.333 | 0.333 | +0.000 | 0.000 |
| education | Steered N=2 | 1458 | 0.353 | 0.353 | 0.333 | +0.019 | 0.000 |
| socioeco | Declared | 972 | 0.550 | 0.550 | 0.500 | +0.050 | 0.000 |
| socioeco | Neutral (N=0) | 972 | 0.499 | 0.500 | 0.500 | +0.000 | 0.002 |
| socioeco | Steered N=2 | 972 | 0.502 | 0.502 | 0.500 | +0.002 | 0.000 |

\* Lift uses fractional credit for tied nearest groups.

## Directional tests at the comparison regimes

Negative differences support specificity. `p_holm` is the one-sided Wilcoxon p-value corrected within attribute and condition.

| Attribute | Regime | Target | Expected vs competitor | n | Mean Δ | Fractional win rate | p_holm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| gender | Declared | Male | male vs female | 484 | -0.029 | 0.554 | 8.09e-05 |
| gender | Declared | Female | female vs male | 484 | +0.006 | 0.483 | 0.985 |
| gender | Neutral (N=0) | Male | male vs female | 484 | +0.000 | 0.500 | nan |
| gender | Neutral (N=0) | Female | female vs male | 484 | +0.000 | 0.500 | nan |
| gender | Steered N=2 | Male | male vs female | 484 | -0.091 | 0.634 | 5.81e-09 |
| gender | Steered N=2 | Female | female vs male | 484 | +0.115 | 0.335 | 1 |
| age | Declared | 30-49 | adult vs older_adult | 482 | -0.003 | 0.391 | 0.982 |
| age | Declared | 65+ | older_adult vs adult | 482 | -0.025 | 0.613 | 0.0156 |
| age | Neutral (N=0) | 30-49 | adult vs older_adult | 482 | +0.000 | 0.500 | nan |
| age | Neutral (N=0) | 65+ | older_adult vs adult | 482 | +0.000 | 0.500 | nan |
| age | Steered N=2 | 30-49 | adult vs older_adult | 482 | +0.013 | 0.456 | 0.786 |
| age | Steered N=2 | 65+ | older_adult vs adult | 482 | -0.011 | 0.523 | 0.515 |
| education | Declared | Less than high school | some schooling vs high school | 486 | +0.042 | 0.603 | 0.242 |
| education | Declared | Less than high school | some schooling vs college and more | 486 | +0.050 | 0.603 | 0.35 |
| education | Declared | High school graduate | high school vs some schooling | 486 | -0.049 | 0.403 | 0.918 |
| education | Declared | High school graduate | high school vs college and more | 486 | +0.009 | 0.541 | 0.902 |
| education | Declared | College graduate/some postgrad | college and more vs some schooling | 486 | -0.084 | 0.444 | 0.65 |
| education | Declared | College graduate/some postgrad | college and more vs high school | 486 | -0.018 | 0.475 | 0.902 |
| education | Neutral (N=0) | Less than high school | some schooling vs high school | 486 | +0.000 | 0.500 | nan |
| education | Neutral (N=0) | Less than high school | some schooling vs college and more | 486 | +0.000 | 0.500 | nan |
| education | Neutral (N=0) | High school graduate | high school vs some schooling | 486 | +0.000 | 0.500 | nan |
| education | Neutral (N=0) | High school graduate | high school vs college and more | 486 | +0.000 | 0.500 | nan |
| education | Neutral (N=0) | College graduate/some postgrad | college and more vs some schooling | 486 | +0.000 | 0.500 | nan |
| education | Neutral (N=0) | College graduate/some postgrad | college and more vs high school | 486 | +0.000 | 0.500 | nan |
| education | Steered N=2 | Less than high school | some schooling vs high school | 486 | -0.182 | 0.640 | 2.66e-13 |
| education | Steered N=2 | Less than high school | some schooling vs college and more | 486 | +0.031 | 0.362 | 1 |
| education | Steered N=2 | High school graduate | high school vs some schooling | 486 | +0.129 | 0.397 | 1 |
| education | Steered N=2 | High school graduate | high school vs college and more | 486 | +0.172 | 0.284 | 1 |
| education | Steered N=2 | College graduate/some postgrad | college and more vs some schooling | 486 | -0.032 | 0.628 | 7.38e-08 |
| education | Steered N=2 | College graduate/some postgrad | college and more vs high school | 486 | -0.183 | 0.702 | 6.77e-16 |
| socioeco | Declared | Less than $30,000 | low vs high | 486 | +0.001 | 0.563 | 0.287 |
| socioeco | Declared | $100,000 or more | high vs low | 486 | -0.084 | 0.483 | 0.00494 |
| socioeco | Neutral (N=0) | Less than $30,000 | low vs high | 486 | +0.000 | 0.500 | nan |
| socioeco | Neutral (N=0) | $100,000 or more | high vs low | 486 | +0.000 | 0.500 | nan |
| socioeco | Steered N=2 | Less than $30,000 | low vs high | 486 | -0.007 | 0.527 | 0.345 |
| socioeco | Steered N=2 | $100,000 or more | high vs low | 486 | +0.006 | 0.451 | 0.855 |

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
