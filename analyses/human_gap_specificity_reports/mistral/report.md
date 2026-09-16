# Specificity conditioned on human separability

- Nearest predictions: `C:\Users\mario\OneDrive - csr.ufmg.br\Documents\GitHub\internal_LLM_bias\analyses\cross_group_reports\mistral\nearest_group_predictions.csv`
- Steering comparison magnitude: `2`
- Bootstrap unit: `qkey`

Quartiles are computed within each attribute and exact human-group pair. For education, each of the three pairwise contrasts is therefore analyzed separately. Accuracy remains multiclass and chance is `1 / n_groups`.

Because a pair-specific education panel retains only the two endpoint model profiles from an original three-group classification, its empirical neutral accuracy need not equal theoretical chance. Read lift together with the neutral curve; the all-class cross-group report remains the calibrated global baseline.

## Q4 headline

| Attribute | Human pair | Regime | n questions | Q1 lift | Q4 lift | All lift | Q4 95% CI | p(Q4 lift <= 0) |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| gender | Male vs Female | Declared | 121 | -0.008 | +0.045 | +0.011 | [+0.012, +0.079] | 0.0045 |
| gender | Male vs Female | Neutral (N=0) | 121 | +0.000 | +0.000 | +0.000 | [+0.000, +0.000] | 1 |
| gender | Male vs Female | Steered N=2 | 121 | +0.017 | -0.103 | -0.044 | [-0.165, -0.041] | 1 |
| age | 30-49 vs 65+ | Declared | 121 | -0.017 | +0.017 | +0.003 | [-0.021, +0.054] | 0.21 |
| age | 30-49 vs 65+ | Neutral (N=0) | 121 | +0.000 | +0.000 | +0.000 | [+0.000, +0.000] | 1 |
| age | 30-49 vs 65+ | Steered N=2 | 121 | +0.017 | +0.012 | +0.013 | [-0.029, +0.054] | 0.313 |
| education | Less than high school vs High school graduate | Declared | 122 | -0.147 | -0.051 | -0.107 | [-0.100, -0.001] | 0.982 |
| education | Less than high school vs College graduate/some postgrad | Declared | 122 | +0.022 | +0.158 | +0.100 | [+0.117, +0.199] | 0.0005 |
| education | Less than high school vs High school graduate | Neutral (N=0) | 122 | -0.147 | -0.087 | -0.119 | [-0.133, -0.046] | 1 |
| education | Less than high school vs College graduate/some postgrad | Neutral (N=0) | 122 | +0.030 | +0.113 | +0.078 | [+0.085, +0.138] | 0.0005 |
| education | Less than high school vs High school graduate | Steered N=2 | 122 | +0.026 | +0.101 | +0.046 | [+0.036, +0.171] | 0.0025 |
| education | Less than high school vs College graduate/some postgrad | Steered N=2 | 122 | +0.010 | +0.036 | +0.034 | [-0.010, +0.085] | 0.077 |
| education | High school graduate vs College graduate/some postgrad | Declared | 122 | +0.014 | +0.072 | +0.040 | [+0.027, +0.113] | 0.001 |
| education | High school graduate vs College graduate/some postgrad | Neutral (N=0) | 122 | +0.022 | +0.064 | +0.041 | [+0.027, +0.097] | 0.0005 |
| education | High school graduate vs College graduate/some postgrad | Steered N=2 | 122 | -0.040 | +0.019 | -0.023 | [-0.038, +0.077] | 0.252 |
| socioeco | Less than $30,000 vs $100,000 or more | Declared | 122 | +0.025 | +0.139 | +0.050 | [+0.094, +0.189] | 0.0005 |
| socioeco | Less than $30,000 vs $100,000 or more | Neutral (N=0) | 122 | +0.000 | +0.000 | +0.000 | [+0.000, +0.000] | 1 |
| socioeco | Less than $30,000 vs $100,000 or more | Steered N=2 | 122 | +0.004 | -0.012 | +0.002 | [-0.053, +0.029] | 0.747 |

## Continuous-gap trend

Positive slopes mean strict nearest-group accuracy increases with human gap.

| Attribute | Human pair | Regime | n | OR per gap SD | 95% slope CI | p(slope <= 0) |
| --- | --- | --- | ---: | ---: | --- | ---: |
| gender | Male vs Female | Declared | 484 | 1.105 | [+0.019, +0.181] | 0.00791 |
| gender | Male vs Female | Neutral (N=0) | 484 | 1.000 | [+nan, +nan] | nan |
| gender | Male vs Female | Steered N=2 | 484 | 0.959 | [-0.130, +0.046] | 0.823 |
| age | 30-49 vs 65+ | Declared | 482 | 1.123 | [+0.029, +0.202] | 0.00441 |
| age | 30-49 vs 65+ | Neutral (N=0) | 482 | 1.000 | [+nan, +nan] | nan |
| age | 30-49 vs 65+ | Steered N=2 | 482 | 0.966 | [-0.111, +0.043] | 0.808 |
| education | Less than high school vs High school graduate | Declared | 486 | 1.218 | [+0.075, +0.320] | 0.000797 |
| education | Less than high school vs College graduate/some postgrad | Declared | 486 | 1.244 | [+0.134, +0.302] | 1.68e-07 |
| education | Less than high school vs High school graduate | Neutral (N=0) | 486 | 1.166 | [+0.032, +0.274] | 0.0065 |
| education | Less than high school vs College graduate/some postgrad | Neutral (N=0) | 486 | 1.126 | [+0.062, +0.176] | 2.01e-05 |
| education | Less than high school vs High school graduate | Steered N=2 | 486 | 1.162 | [+0.005, +0.294] | 0.0211 |
| education | Less than high school vs College graduate/some postgrad | Steered N=2 | 486 | 1.022 | [-0.090, +0.134] | 0.349 |
| education | High school graduate vs College graduate/some postgrad | Declared | 486 | 1.109 | [+0.018, +0.189] | 0.00873 |
| education | High school graduate vs College graduate/some postgrad | Neutral (N=0) | 486 | 1.067 | [-0.008, +0.137] | 0.0401 |
| education | High school graduate vs College graduate/some postgrad | Steered N=2 | 486 | 1.109 | [-0.020, +0.227] | 0.0503 |
| socioeco | Less than $30,000 vs $100,000 or more | Declared | 486 | 1.330 | [+0.186, +0.384] | 8.5e-09 |
| socioeco | Less than $30,000 vs $100,000 or more | Neutral (N=0) | 486 | 1.003 | [-0.003, +0.008] | 0.17 |
| socioeco | Less than $30,000 vs $100,000 or more | Steered N=2 | 486 | 1.049 | [-0.062, +0.158] | 0.195 |

## Q4 directional tests

Negative mean differences and win rates above 0.5 support the intended direction.

| Attribute | Pair | Regime | Target | n | Mean own-minus-competitor | Fractional win rate | p_holm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| gender | Male vs Female | Declared | Male | 121 | -0.051 | 0.649 | 0.00483 |
| gender | Male vs Female | Neutral (N=0) | Male | 121 | +0.000 | 0.500 | nan |
| gender | Male vs Female | Steered N=2 | Male | 121 | -0.091 | 0.628 | 0.00849 |
| gender | Male vs Female | Declared | Female | 121 | -0.039 | 0.459 | 0.454 |
| gender | Male vs Female | Neutral (N=0) | Female | 121 | +0.000 | 0.500 | nan |
| gender | Male vs Female | Steered N=2 | Female | 121 | +0.149 | 0.298 | 1 |
| age | 30-49 vs 65+ | Declared | 30-49 | 121 | +0.006 | 0.430 | 0.846 |
| age | 30-49 vs 65+ | Neutral (N=0) | 30-49 | 121 | +0.000 | 0.500 | nan |
| age | 30-49 vs 65+ | Steered N=2 | 30-49 | 121 | +0.023 | 0.471 | 1 |
| age | 30-49 vs 65+ | Declared | 65+ | 121 | -0.108 | 0.579 | 0.117 |
| age | 30-49 vs 65+ | Neutral (N=0) | 65+ | 121 | +0.000 | 0.500 | nan |
| age | 30-49 vs 65+ | Steered N=2 | 65+ | 121 | -0.003 | 0.471 | 1 |
| education | Less than high school vs High school graduate | Declared | Less than high school | 122 | +0.092 | 0.566 | 1 |
| education | Less than high school vs High school graduate | Neutral (N=0) | Less than high school | 122 | +0.000 | 0.500 | nan |
| education | Less than high school vs High school graduate | Steered N=2 | Less than high school | 122 | -0.235 | 0.672 | 0.000142 |
| education | Less than high school vs High school graduate | Declared | High school graduate | 122 | -0.106 | 0.418 | 1 |
| education | Less than high school vs High school graduate | Neutral (N=0) | High school graduate | 122 | +0.000 | 0.500 | nan |
| education | Less than high school vs High school graduate | Steered N=2 | High school graduate | 122 | +0.076 | 0.467 | 1 |
| education | Less than high school vs College graduate/some postgrad | Declared | Less than high school | 122 | +0.083 | 0.566 | 1 |
| education | Less than high school vs College graduate/some postgrad | Neutral (N=0) | Less than high school | 122 | +0.000 | 0.500 | nan |
| education | Less than high school vs College graduate/some postgrad | Steered N=2 | Less than high school | 122 | -0.043 | 0.418 | 1 |
| education | Less than high school vs College graduate/some postgrad | Declared | College graduate/some postgrad | 122 | -0.160 | 0.516 | 0.342 |
| education | Less than high school vs College graduate/some postgrad | Neutral (N=0) | College graduate/some postgrad | 122 | +0.000 | 0.500 | nan |
| education | Less than high school vs College graduate/some postgrad | Steered N=2 | College graduate/some postgrad | 122 | +0.049 | 0.557 | 1 |
| education | High school graduate vs College graduate/some postgrad | Declared | High school graduate | 122 | +0.031 | 0.508 | 1 |
| education | High school graduate vs College graduate/some postgrad | Neutral (N=0) | High school graduate | 122 | +0.000 | 0.500 | nan |
| education | High school graduate vs College graduate/some postgrad | Steered N=2 | High school graduate | 122 | +0.071 | 0.369 | 1 |
| education | High school graduate vs College graduate/some postgrad | Declared | College graduate/some postgrad | 122 | -0.052 | 0.516 | 0.94 |
| education | High school graduate vs College graduate/some postgrad | Neutral (N=0) | College graduate/some postgrad | 122 | +0.000 | 0.500 | nan |
| education | High school graduate vs College graduate/some postgrad | Steered N=2 | College graduate/some postgrad | 122 | -0.084 | 0.590 | 0.549 |
| socioeco | Less than $30,000 vs $100,000 or more | Declared | Less than $30,000 | 122 | -0.071 | 0.631 | 0.00557 |
| socioeco | Less than $30,000 vs $100,000 or more | Neutral (N=0) | Less than $30,000 | 122 | +0.000 | 0.500 | nan |
| socioeco | Less than $30,000 vs $100,000 or more | Steered N=2 | Less than $30,000 | 122 | -0.049 | 0.508 | 0.211 |
| socioeco | Less than $30,000 vs $100,000 or more | Declared | $100,000 or more | 122 | -0.230 | 0.549 | 0.00176 |
| socioeco | Less than $30,000 vs $100,000 or more | Neutral (N=0) | $100,000 or more | 122 | +0.000 | 0.500 | nan |
| socioeco | Less than $30,000 vs $100,000 or more | Steered N=2 | $100,000 or more | 122 | +0.032 | 0.410 | 0.751 |

## Files

- [Predictions with human-gap strata](predictions_by_human_gap.csv)
- [Quartile accuracy and bootstrap CIs](accuracy_by_human_gap_quartile.csv)
- [Continuous-gap logistic trends](accuracy_gap_trends.csv)
- [Directional comparisons by quartile](directional_tests_by_human_gap_quartile.csv)
- [Quartile lift plot](lift_by_human_gap_quartile.png)
