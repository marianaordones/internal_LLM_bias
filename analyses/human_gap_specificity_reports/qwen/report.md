# Specificity conditioned on human separability

- Nearest predictions: `C:\Users\mario\OneDrive - csr.ufmg.br\Documents\GitHub\internal_LLM_bias\analyses\cross_group_reports\qwen\nearest_group_predictions.csv`
- Steering comparison magnitude: `20`
- Bootstrap unit: `qkey`

Quartiles are computed within each attribute and exact human-group pair. For education, each of the three pairwise contrasts is therefore analyzed separately. Accuracy remains multiclass and chance is `1 / n_groups`.

Because a pair-specific education panel retains only the two endpoint model profiles from an original three-group classification, its empirical neutral accuracy need not equal theoretical chance. Read lift together with the neutral curve; the all-class cross-group report remains the calibrated global baseline.

## Q4 headline

| Attribute | Human pair | Regime | n questions | Q1 lift | Q4 lift | All lift | Q4 95% CI | p(Q4 lift <= 0) |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| gender | Male vs Female | Declared | 121 | -0.004 | +0.029 | +0.002 | [+0.008, +0.050] | 0.002 |
| gender | Male vs Female | Neutral (N=0) | 121 | +0.000 | +0.000 | +0.000 | [+0.000, +0.000] | 1 |
| gender | Male vs Female | Steered N=20 | 121 | +0.017 | +0.029 | +0.005 | [-0.008, +0.066] | 0.079 |
| age | 30-49 vs 65+ | Declared | 121 | -0.008 | +0.025 | +0.000 | [+0.000, +0.050] | 0.0295 |
| age | 30-49 vs 65+ | Neutral (N=0) | 121 | +0.000 | +0.000 | +0.000 | [+0.000, +0.000] | 1 |
| age | 30-49 vs 65+ | Steered N=20 | 121 | -0.046 | +0.012 | -0.005 | [-0.021, +0.045] | 0.281 |
| education | Less than high school vs High school graduate | Declared | 122 | -0.139 | -0.059 | -0.105 | [-0.104, -0.014] | 0.996 |
| education | Less than high school vs College graduate/some postgrad | Declared | 122 | +0.014 | +0.126 | +0.082 | [+0.081, +0.167] | 0.0005 |
| education | Less than high school vs High school graduate | Neutral (N=0) | 122 | -0.127 | -0.079 | -0.105 | [-0.120, -0.034] | 1 |
| education | Less than high school vs College graduate/some postgrad | Neutral (N=0) | 122 | +0.030 | +0.109 | +0.076 | [+0.077, +0.134] | 0.0005 |
| education | Less than high school vs High school graduate | Steered N=20 | 122 | -0.106 | -0.055 | -0.083 | [-0.108, -0.001] | 0.978 |
| education | Less than high school vs College graduate/some postgrad | Steered N=20 | 122 | +0.043 | +0.126 | +0.087 | [+0.089, +0.167] | 0.0005 |
| education | High school graduate vs College graduate/some postgrad | Declared | 122 | +0.022 | +0.089 | +0.039 | [+0.048, +0.130] | 0.0005 |
| education | High school graduate vs College graduate/some postgrad | Neutral (N=0) | 122 | +0.010 | +0.081 | +0.029 | [+0.044, +0.113] | 0.0005 |
| education | High school graduate vs College graduate/some postgrad | Steered N=20 | 122 | +0.030 | +0.068 | +0.033 | [+0.023, +0.117] | 0.0035 |
| socioeco | Less than $30,000 vs $100,000 or more | Declared | 122 | +0.004 | +0.102 | +0.034 | [+0.061, +0.148] | 0.0005 |
| socioeco | Less than $30,000 vs $100,000 or more | Neutral (N=0) | 122 | +0.000 | +0.000 | +0.000 | [+0.000, +0.000] | 1 |
| socioeco | Less than $30,000 vs $100,000 or more | Steered N=20 | 122 | -0.008 | +0.016 | -0.003 | [-0.012, +0.049] | 0.164 |

## Continuous-gap trend

Positive slopes mean strict nearest-group accuracy increases with human gap.

| Attribute | Human pair | Regime | n | OR per gap SD | 95% slope CI | p(slope <= 0) |
| --- | --- | --- | ---: | ---: | --- | ---: |
| gender | Male vs Female | Declared | 484 | 1.021 | [-0.003, +0.044] | 0.0413 |
| gender | Male vs Female | Neutral (N=0) | 484 | 1.000 | [+nan, +nan] | nan |
| gender | Male vs Female | Steered N=20 | 484 | 0.966 | [-0.112, +0.044] | 0.805 |
| age | 30-49 vs 65+ | Declared | 482 | 1.155 | [+0.068, +0.220] | 0.000104 |
| age | 30-49 vs 65+ | Neutral (N=0) | 482 | 1.000 | [+nan, +nan] | nan |
| age | 30-49 vs 65+ | Steered N=20 | 482 | 1.046 | [-0.027, +0.116] | 0.111 |
| education | Less than high school vs High school graduate | Declared | 486 | 1.143 | [+0.013, +0.255] | 0.0151 |
| education | Less than high school vs College graduate/some postgrad | Declared | 486 | 1.196 | [+0.096, +0.261] | 1.11e-05 |
| education | Less than high school vs High school graduate | Neutral (N=0) | 486 | 1.106 | [-0.018, +0.220] | 0.0486 |
| education | Less than high school vs College graduate/some postgrad | Neutral (N=0) | 486 | 1.105 | [+0.038, +0.162] | 0.000795 |
| education | Less than high school vs High school graduate | Steered N=20 | 486 | 1.106 | [-0.036, +0.237] | 0.0741 |
| education | Less than high school vs College graduate/some postgrad | Steered N=20 | 486 | 1.093 | [+0.010, +0.169] | 0.0139 |
| education | High school graduate vs College graduate/some postgrad | Declared | 486 | 1.156 | [+0.063, +0.227] | 0.00025 |
| education | High school graduate vs College graduate/some postgrad | Neutral (N=0) | 486 | 1.162 | [+0.082, +0.219] | 9.1e-06 |
| education | High school graduate vs College graduate/some postgrad | Steered N=20 | 486 | 1.090 | [-0.011, +0.184] | 0.0412 |
| socioeco | Less than $30,000 vs $100,000 or more | Declared | 486 | 1.211 | [+0.087, +0.297] | 0.000178 |
| socioeco | Less than $30,000 vs $100,000 or more | Neutral (N=0) | 486 | 1.000 | [+nan, +nan] | nan |
| socioeco | Less than $30,000 vs $100,000 or more | Steered N=20 | 486 | 1.051 | [+0.001, +0.097] | 0.022 |

## Q4 directional tests

Negative mean differences and win rates above 0.5 support the intended direction.

| Attribute | Pair | Regime | Target | n | Mean own-minus-competitor | Fractional win rate | p_holm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| gender | Male vs Female | Declared | Male | 121 | +0.002 | 0.508 | 0.71 |
| gender | Male vs Female | Neutral (N=0) | Male | 121 | +0.000 | 0.500 | nan |
| gender | Male vs Female | Steered N=20 | Male | 121 | +0.012 | 0.463 | 0.745 |
| gender | Male vs Female | Declared | Female | 121 | -0.024 | 0.558 | 0.184 |
| gender | Male vs Female | Neutral (N=0) | Female | 121 | +0.000 | 0.500 | nan |
| gender | Male vs Female | Steered N=20 | Female | 121 | -0.011 | 0.521 | 0.745 |
| age | 30-49 vs 65+ | Declared | 30-49 | 121 | -0.015 | 0.475 | 0.575 |
| age | 30-49 vs 65+ | Neutral (N=0) | 30-49 | 121 | +0.000 | 0.500 | nan |
| age | 30-49 vs 65+ | Steered N=20 | 30-49 | 121 | -0.128 | 0.777 | 4.32e-08 |
| age | 30-49 vs 65+ | Declared | 65+ | 121 | -0.079 | 0.591 | 0.0571 |
| age | 30-49 vs 65+ | Neutral (N=0) | 65+ | 121 | +0.000 | 0.500 | nan |
| age | 30-49 vs 65+ | Steered N=20 | 65+ | 121 | +0.098 | 0.306 | 1 |
| education | Less than high school vs High school graduate | Declared | Less than high school | 122 | +0.032 | 0.566 | 1 |
| education | Less than high school vs High school graduate | Neutral (N=0) | Less than high school | 122 | +0.000 | 0.500 | nan |
| education | Less than high school vs High school graduate | Steered N=20 | Less than high school | 122 | +0.009 | 0.467 | 1 |
| education | Less than high school vs High school graduate | Declared | High school graduate | 122 | -0.036 | 0.426 | 1 |
| education | Less than high school vs High school graduate | Neutral (N=0) | High school graduate | 122 | +0.000 | 0.500 | nan |
| education | Less than high school vs High school graduate | Steered N=20 | High school graduate | 122 | -0.017 | 0.549 | 0.982 |
| education | Less than high school vs College graduate/some postgrad | Declared | Less than high school | 122 | +0.097 | 0.451 | 1 |
| education | Less than high school vs College graduate/some postgrad | Neutral (N=0) | Less than high school | 122 | +0.000 | 0.500 | nan |
| education | Less than high school vs College graduate/some postgrad | Steered N=20 | Less than high school | 122 | -0.006 | 0.475 | 1 |
| education | Less than high school vs College graduate/some postgrad | Declared | College graduate/some postgrad | 122 | -0.171 | 0.582 | 0.014 |
| education | Less than high school vs College graduate/some postgrad | Neutral (N=0) | College graduate/some postgrad | 122 | +0.000 | 0.500 | nan |
| education | Less than high school vs College graduate/some postgrad | Steered N=20 | College graduate/some postgrad | 122 | -0.032 | 0.557 | 0.461 |
| education | High school graduate vs College graduate/some postgrad | Declared | High school graduate | 122 | +0.039 | 0.406 | 1 |
| education | High school graduate vs College graduate/some postgrad | Neutral (N=0) | High school graduate | 122 | +0.000 | 0.500 | nan |
| education | High school graduate vs College graduate/some postgrad | Steered N=20 | High school graduate | 122 | +0.048 | 0.451 | 1 |
| education | High school graduate vs College graduate/some postgrad | Declared | College graduate/some postgrad | 122 | -0.069 | 0.619 | 0.446 |
| education | High school graduate vs College graduate/some postgrad | Neutral (N=0) | College graduate/some postgrad | 122 | +0.000 | 0.500 | nan |
| education | High school graduate vs College graduate/some postgrad | Steered N=20 | College graduate/some postgrad | 122 | -0.045 | 0.508 | 1 |
| socioeco | Less than $30,000 vs $100,000 or more | Declared | Less than $30,000 | 122 | +0.007 | 0.525 | 0.361 |
| socioeco | Less than $30,000 vs $100,000 or more | Neutral (N=0) | Less than $30,000 | 122 | +0.000 | 0.500 | nan |
| socioeco | Less than $30,000 vs $100,000 or more | Steered N=20 | Less than $30,000 | 122 | +0.009 | 0.447 | 0.741 |
| socioeco | Less than $30,000 vs $100,000 or more | Declared | $100,000 or more | 122 | -0.231 | 0.607 | 0.000373 |
| socioeco | Less than $30,000 vs $100,000 or more | Neutral (N=0) | $100,000 or more | 122 | +0.000 | 0.500 | nan |
| socioeco | Less than $30,000 vs $100,000 or more | Steered N=20 | $100,000 or more | 122 | -0.034 | 0.602 | 0.072 |

## Files

- [Predictions with human-gap strata](predictions_by_human_gap.csv)
- [Quartile accuracy and bootstrap CIs](accuracy_by_human_gap_quartile.csv)
- [Continuous-gap logistic trends](accuracy_gap_trends.csv)
- [Directional comparisons by quartile](directional_tests_by_human_gap_quartile.csv)
- [Quartile lift plot](lift_by_human_gap_quartile.png)
