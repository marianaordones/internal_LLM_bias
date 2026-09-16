# Specificity conditioned on human separability

- Nearest predictions: `C:\Users\mario\OneDrive - csr.ufmg.br\Documents\GitHub\internal_LLM_bias\analyses\cross_group_reports\llama\nearest_group_predictions.csv`
- Steering comparison magnitude: `20`
- Bootstrap unit: `qkey`

Quartiles are computed within each attribute and exact human-group pair. For education, each of the three pairwise contrasts is therefore analyzed separately. Accuracy remains multiclass and chance is `1 / n_groups`.

Because a pair-specific education panel retains only the two endpoint model profiles from an original three-group classification, its empirical neutral accuracy need not equal theoretical chance. Read lift together with the neutral curve; the all-class cross-group report remains the calibrated global baseline.

## Q4 headline

| Attribute | Human pair | Regime | n questions | Q1 lift | Q4 lift | All lift | Q4 95% CI | p(Q4 lift <= 0) |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| gender | Male vs Female | Declared | 121 | -0.025 | +0.054 | +0.020 | [+0.008, +0.103] | 0.014 |
| gender | Male vs Female | Neutral (N=0) | 121 | +0.000 | +0.000 | +0.000 | [+0.000, +0.000] | 1 |
| gender | Male vs Female | Steered N=20 | 121 | -0.012 | +0.037 | +0.046 | [-0.012, +0.087] | 0.0795 |
| age | 30-49 vs 65+ | Declared | 121 | +0.012 | -0.008 | -0.005 | [-0.045, +0.029] | 0.724 |
| age | 30-49 vs 65+ | Neutral (N=0) | 121 | +0.000 | +0.000 | +0.000 | [+0.000, +0.000] | 1 |
| age | 30-49 vs 65+ | Steered N=20 | 121 | +0.033 | -0.004 | +0.006 | [-0.054, +0.041] | 0.6 |
| education | Less than high school vs High school graduate | Declared | 122 | +0.006 | +0.036 | -0.001 | [-0.014, +0.085] | 0.0955 |
| education | Less than high school vs College graduate/some postgrad | Declared | 122 | -0.036 | +0.101 | +0.032 | [+0.044, +0.163] | 0.0005 |
| education | Less than high school vs High school graduate | Neutral (N=0) | 122 | +0.001 | +0.023 | +0.014 | [-0.018, +0.064] | 0.133 |
| education | Less than high school vs College graduate/some postgrad | Neutral (N=0) | 122 | -0.069 | +0.072 | +0.023 | [+0.036, +0.105] | 0.0005 |
| education | Less than high school vs High school graduate | Steered N=20 | 122 | -0.007 | +0.003 | +0.019 | [-0.051, +0.056] | 0.48 |
| education | Less than high school vs College graduate/some postgrad | Steered N=20 | 122 | +0.034 | +0.072 | +0.052 | [+0.023, +0.122] | 0.0035 |
| education | High school graduate vs College graduate/some postgrad | Declared | 122 | -0.052 | +0.052 | -0.008 | [+0.007, +0.101] | 0.0185 |
| education | High school graduate vs College graduate/some postgrad | Neutral (N=0) | 122 | -0.090 | +0.007 | -0.037 | [-0.038, +0.048] | 0.383 |
| education | High school graduate vs College graduate/some postgrad | Steered N=20 | 122 | -0.102 | -0.063 | -0.092 | [-0.116, -0.010] | 0.989 |
| socioeco | Less than $30,000 vs $100,000 or more | Declared | 122 | -0.035 | +0.045 | +0.004 | [+0.012, +0.082] | 0.005 |
| socioeco | Less than $30,000 vs $100,000 or more | Neutral (N=0) | 122 | +0.000 | +0.000 | +0.000 | [+0.000, +0.000] | 1 |
| socioeco | Less than $30,000 vs $100,000 or more | Steered N=20 | 122 | -0.017 | -0.029 | -0.040 | [-0.090, +0.033] | 0.827 |

## Continuous-gap trend

Positive slopes mean strict nearest-group accuracy increases with human gap.

| Attribute | Human pair | Regime | n | OR per gap SD | 95% slope CI | p(slope <= 0) |
| --- | --- | --- | ---: | ---: | --- | ---: |
| gender | Male vs Female | Declared | 484 | 1.036 | [-0.036, +0.107] | 0.165 |
| gender | Male vs Female | Neutral (N=0) | 484 | 1.000 | [+nan, +nan] | nan |
| gender | Male vs Female | Steered N=20 | 484 | 0.995 | [-0.077, +0.068] | 0.55 |
| age | 30-49 vs 65+ | Declared | 482 | 1.016 | [-0.063, +0.095] | 0.345 |
| age | 30-49 vs 65+ | Neutral (N=0) | 482 | 1.000 | [+nan, +nan] | nan |
| age | 30-49 vs 65+ | Steered N=20 | 482 | 0.960 | [-0.111, +0.031] | 0.867 |
| education | Less than high school vs High school graduate | Declared | 486 | 1.021 | [-0.102, +0.143] | 0.37 |
| education | Less than high school vs College graduate/some postgrad | Declared | 486 | 1.239 | [+0.078, +0.350] | 0.000986 |
| education | Less than high school vs High school graduate | Neutral (N=0) | 486 | 0.985 | [-0.111, +0.081] | 0.62 |
| education | Less than high school vs College graduate/some postgrad | Neutral (N=0) | 486 | 1.219 | [+0.122, +0.274] | 1.76e-07 |
| education | Less than high school vs High school graduate | Steered N=20 | 486 | 0.932 | [-0.196, +0.055] | 0.865 |
| education | Less than high school vs College graduate/some postgrad | Steered N=20 | 486 | 1.031 | [-0.075, +0.135] | 0.286 |
| education | High school graduate vs College graduate/some postgrad | Declared | 486 | 1.155 | [+0.037, +0.251] | 0.00422 |
| education | High school graduate vs College graduate/some postgrad | Neutral (N=0) | 486 | 1.115 | [+0.003, +0.214] | 0.022 |
| education | High school graduate vs College graduate/some postgrad | Steered N=20 | 486 | 1.057 | [-0.089, +0.199] | 0.226 |
| socioeco | Less than $30,000 vs $100,000 or more | Declared | 486 | 1.109 | [+0.030, +0.177] | 0.00282 |
| socioeco | Less than $30,000 vs $100,000 or more | Neutral (N=0) | 486 | 1.003 | [-0.003, +0.008] | 0.17 |
| socioeco | Less than $30,000 vs $100,000 or more | Steered N=20 | 486 | 1.090 | [-0.048, +0.220] | 0.104 |

## Q4 directional tests

Negative mean differences and win rates above 0.5 support the intended direction.

| Attribute | Pair | Regime | Target | n | Mean own-minus-competitor | Fractional win rate | p_holm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| gender | Male vs Female | Declared | Male | 121 | +0.033 | 0.438 | 0.927 |
| gender | Male vs Female | Neutral (N=0) | Male | 121 | +0.000 | 0.500 | nan |
| gender | Male vs Female | Steered N=20 | Male | 121 | +0.050 | 0.388 | 0.992 |
| gender | Male vs Female | Declared | Female | 121 | -0.078 | 0.529 | 0.0766 |
| gender | Male vs Female | Neutral (N=0) | Female | 121 | +0.000 | 0.500 | nan |
| gender | Male vs Female | Steered N=20 | Female | 121 | -0.046 | 0.521 | 0.0857 |
| age | 30-49 vs 65+ | Declared | 30-49 | 121 | +0.008 | 0.430 | 0.859 |
| age | 30-49 vs 65+ | Neutral (N=0) | 30-49 | 121 | +0.000 | 0.500 | nan |
| age | 30-49 vs 65+ | Steered N=20 | 30-49 | 121 | -0.198 | 0.727 | 1.63e-09 |
| age | 30-49 vs 65+ | Declared | 65+ | 121 | +0.018 | 0.521 | 0.803 |
| age | 30-49 vs 65+ | Neutral (N=0) | 65+ | 121 | +0.000 | 0.500 | nan |
| age | 30-49 vs 65+ | Steered N=20 | 65+ | 121 | +0.206 | 0.281 | 1 |
| education | Less than high school vs High school graduate | Declared | Less than high school | 122 | +0.294 | 0.238 | 1 |
| education | Less than high school vs High school graduate | Neutral (N=0) | Less than high school | 122 | +0.000 | 0.500 | nan |
| education | Less than high school vs High school graduate | Steered N=20 | Less than high school | 122 | -0.152 | 0.713 | 0.000309 |
| education | Less than high school vs High school graduate | Declared | High school graduate | 122 | -0.244 | 0.697 | 5.12e-07 |
| education | Less than high school vs High school graduate | Neutral (N=0) | High school graduate | 122 | +0.000 | 0.500 | nan |
| education | Less than high school vs High school graduate | Steered N=20 | High school graduate | 122 | +0.254 | 0.230 | 1 |
| education | Less than high school vs College graduate/some postgrad | Declared | Less than high school | 122 | +0.186 | 0.311 | 1 |
| education | Less than high school vs College graduate/some postgrad | Neutral (N=0) | Less than high school | 122 | +0.000 | 0.500 | nan |
| education | Less than high school vs College graduate/some postgrad | Steered N=20 | Less than high school | 122 | +0.085 | 0.410 | 1 |
| education | Less than high school vs College graduate/some postgrad | Declared | College graduate/some postgrad | 122 | -0.148 | 0.656 | 0.0162 |
| education | Less than high school vs College graduate/some postgrad | Neutral (N=0) | College graduate/some postgrad | 122 | +0.000 | 0.500 | nan |
| education | Less than high school vs College graduate/some postgrad | Steered N=20 | College graduate/some postgrad | 122 | -0.108 | 0.648 | 0.000685 |
| education | High school graduate vs College graduate/some postgrad | Declared | High school graduate | 122 | -0.045 | 0.492 | 1 |
| education | High school graduate vs College graduate/some postgrad | Neutral (N=0) | High school graduate | 122 | +0.000 | 0.500 | nan |
| education | High school graduate vs College graduate/some postgrad | Steered N=20 | High school graduate | 122 | +0.305 | 0.205 | 1 |
| education | High school graduate vs College graduate/some postgrad | Declared | College graduate/some postgrad | 122 | +0.017 | 0.541 | 1 |
| education | High school graduate vs College graduate/some postgrad | Neutral (N=0) | College graduate/some postgrad | 122 | +0.000 | 0.500 | nan |
| education | High school graduate vs College graduate/some postgrad | Steered N=20 | College graduate/some postgrad | 122 | -0.230 | 0.680 | 1.07e-06 |
| socioeco | Less than $30,000 vs $100,000 or more | Declared | Less than $30,000 | 122 | +0.003 | 0.492 | 0.558 |
| socioeco | Less than $30,000 vs $100,000 or more | Neutral (N=0) | Less than $30,000 | 122 | +0.000 | 0.500 | nan |
| socioeco | Less than $30,000 vs $100,000 or more | Steered N=20 | Less than $30,000 | 122 | +0.349 | 0.279 | 1 |
| socioeco | Less than $30,000 vs $100,000 or more | Declared | $100,000 or more | 122 | -0.080 | 0.574 | 0.0203 |
| socioeco | Less than $30,000 vs $100,000 or more | Neutral (N=0) | $100,000 or more | 122 | +0.000 | 0.500 | nan |
| socioeco | Less than $30,000 vs $100,000 or more | Steered N=20 | $100,000 or more | 122 | -0.356 | 0.730 | 3.15e-09 |

## Files

- [Predictions with human-gap strata](predictions_by_human_gap.csv)
- [Quartile accuracy and bootstrap CIs](accuracy_by_human_gap_quartile.csv)
- [Continuous-gap logistic trends](accuracy_gap_trends.csv)
- [Directional comparisons by quartile](directional_tests_by_human_gap_quartile.csv)
- [Quartile lift plot](lift_by_human_gap_quartile.png)
