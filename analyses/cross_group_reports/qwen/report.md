# Cross-group specificity report: qwen

- Input: `C:\Users\mario\OneDrive - csr.ufmg.br\Documents\GitHub\internal_LLM_bias\results\demographic_opinionqa_qwen.csv`
- Model ID: `Qwen/Qwen2.5-7B-Instruct`
- Comparison magnitude: `20`
- Cross-group distance cells: `122256`
- Incomplete model/question choice sets excluded from classification: `72`
- Unmapped model rows: `17784`

The nearest-neighbor result is correct only when the uniquely closest human group matches the configured group for that model profile. Fractional accuracy assigns 1/k credit when the correct group is among k tied nearest groups. The specificity margin is own-group distance minus the mean distance to other groups; negative values indicate profile-specific alignment.

## Headline nearest-group accuracy

| Attribute | Regime | n | Strict accuracy | Fractional-tie accuracy | Chance | Lift* | Tie rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gender | Declared | 968 | 0.502 | 0.502 | 0.500 | +0.002 | 0.000 |
| gender | Neutral (N=0) | 968 | 0.500 | 0.500 | 0.500 | +0.000 | 0.000 |
| gender | Steered N=20 | 968 | 0.505 | 0.505 | 0.500 | +0.005 | 0.000 |
| age | Declared | 964 | 0.500 | 0.500 | 0.500 | +0.000 | 0.000 |
| age | Neutral (N=0) | 964 | 0.500 | 0.500 | 0.500 | +0.000 | 0.000 |
| age | Steered N=20 | 964 | 0.495 | 0.495 | 0.500 | -0.005 | 0.000 |
| education | Declared | 1458 | 0.339 | 0.339 | 0.333 | +0.005 | 0.000 |
| education | Neutral (N=0) | 1458 | 0.333 | 0.333 | 0.333 | +0.000 | 0.000 |
| education | Steered N=20 | 1458 | 0.346 | 0.346 | 0.333 | +0.012 | 0.000 |
| socioeco | Declared | 972 | 0.534 | 0.534 | 0.500 | +0.034 | 0.000 |
| socioeco | Neutral (N=0) | 972 | 0.500 | 0.500 | 0.500 | +0.000 | 0.000 |
| socioeco | Steered N=20 | 972 | 0.497 | 0.497 | 0.500 | -0.003 | 0.000 |

\* Lift uses fractional credit for tied nearest groups.

## Directional tests at the comparison regimes

Negative differences support specificity. `p_holm` is the one-sided Wilcoxon p-value corrected within attribute and condition.

| Attribute | Regime | Target | Expected vs competitor | n | Mean Δ | Fractional win rate | p_holm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| gender | Declared | Male | male vs female | 484 | +0.013 | 0.492 | 0.845 |
| gender | Declared | Female | female vs male | 484 | -0.017 | 0.548 | 0.175 |
| gender | Neutral (N=0) | Male | male vs female | 484 | +0.000 | 0.500 | nan |
| gender | Neutral (N=0) | Female | female vs male | 484 | +0.000 | 0.500 | nan |
| gender | Steered N=20 | Male | male vs female | 484 | -0.004 | 0.595 | 0.00657 |
| gender | Steered N=20 | Female | female vs male | 484 | +0.008 | 0.399 | 0.999 |
| age | Declared | 30-49 | adult vs older_adult | 482 | +0.002 | 0.432 | 0.962 |
| age | Declared | 65+ | older_adult vs adult | 482 | -0.025 | 0.606 | 0.0073 |
| age | Neutral (N=0) | 30-49 | adult vs older_adult | 482 | +0.000 | 0.500 | nan |
| age | Neutral (N=0) | 65+ | older_adult vs adult | 482 | +0.000 | 0.500 | nan |
| age | Steered N=20 | 30-49 | adult vs older_adult | 482 | -0.131 | 0.842 | 1.75e-38 |
| age | Steered N=20 | 65+ | older_adult vs adult | 482 | +0.125 | 0.176 | 1 |
| education | Declared | Less than high school | some schooling vs high school | 486 | +0.028 | 0.565 | 1 |
| education | Declared | Less than high school | some schooling vs college and more | 486 | +0.041 | 0.497 | 1 |
| education | Declared | High school graduate | high school vs some schooling | 486 | -0.031 | 0.452 | 1 |
| education | Declared | High school graduate | high school vs college and more | 486 | +0.024 | 0.394 | 1 |
| education | Declared | College graduate/some postgrad | college and more vs some schooling | 486 | -0.071 | 0.540 | 0.00485 |
| education | Declared | College graduate/some postgrad | college and more vs high school | 486 | -0.032 | 0.643 | 0.00041 |
| education | Neutral (N=0) | Less than high school | some schooling vs high school | 486 | +0.000 | 0.500 | nan |
| education | Neutral (N=0) | Less than high school | some schooling vs college and more | 486 | +0.000 | 0.500 | nan |
| education | Neutral (N=0) | High school graduate | high school vs some schooling | 486 | +0.000 | 0.500 | nan |
| education | Neutral (N=0) | High school graduate | high school vs college and more | 486 | +0.000 | 0.500 | nan |
| education | Neutral (N=0) | College graduate/some postgrad | college and more vs some schooling | 486 | +0.000 | 0.500 | nan |
| education | Neutral (N=0) | College graduate/some postgrad | college and more vs high school | 486 | +0.000 | 0.500 | nan |
| education | Steered N=20 | Less than high school | some schooling vs high school | 486 | -0.019 | 0.502 | 1 |
| education | Steered N=20 | Less than high school | some schooling vs college and more | 486 | -0.003 | 0.519 | 1 |
| education | Steered N=20 | High school graduate | high school vs some schooling | 486 | +0.011 | 0.512 | 1 |
| education | Steered N=20 | High school graduate | high school vs college and more | 486 | +0.010 | 0.547 | 1 |
| education | Steered N=20 | College graduate/some postgrad | college and more vs some schooling | 486 | -0.008 | 0.490 | 1 |
| education | Steered N=20 | College graduate/some postgrad | college and more vs high school | 486 | -0.009 | 0.440 | 1 |
| socioeco | Declared | Less than $30,000 | low vs high | 486 | +0.022 | 0.487 | 0.92 |
| socioeco | Declared | $100,000 or more | high vs low | 486 | -0.084 | 0.569 | 4.5e-05 |
| socioeco | Neutral (N=0) | Less than $30,000 | low vs high | 486 | +0.000 | 0.500 | nan |
| socioeco | Neutral (N=0) | $100,000 or more | high vs low | 486 | +0.000 | 0.500 | nan |
| socioeco | Steered N=20 | Less than $30,000 | low vs high | 486 | +0.017 | 0.416 | 0.999 |
| socioeco | Steered N=20 | $100,000 or more | high vs low | 486 | -0.021 | 0.597 | 0.00041 |

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
