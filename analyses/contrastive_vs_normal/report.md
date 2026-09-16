# Raw versus contrastive demographic steering

All comparisons use the same mapped model classes and are paired by question, attribute, class, and magnitude. Outcomes are changes from each run's own N=0. For target distance and specificity, negative is better; for nearest-group credit, positive is better.

## Baseline agreement

| Model | n | Mean abs(Δ own) | Max abs(Δ own) | Mean abs(Δ other) | Max abs(Δ other) |
| --- | ---: | ---: | ---: | ---: | ---: |
| llama | 4362 | 0 | 0 | 0 | 0 |
| mistral | 4362 | 0.00918529 | 0.285717 | 0.00892076 | 0.285717 |
| qwen | 4362 | 0.0115589 | 0.464413 | 0.0113941 | 0.464413 |

## Comparison at the paper magnitudes

| Model | N | Mode | Δ target | Δ others | Δ specificity | Δ nearest |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| llama | 20 | contrastive | -0.0844 | -0.0896 | +0.0052 | -0.0070 |
| llama | 20 | normal | -0.1131 | -0.1158 | +0.0027 | +0.0005 |
| mistral | 2 | contrastive | -0.2408 | -0.2385 | -0.0023 | +0.0040 |
| mistral | 2 | normal | -0.2633 | -0.2626 | -0.0007 | +0.0000 |
| qwen | 20 | contrastive | -0.1283 | -0.1098 | -0.0185 | +0.0543 |
| qwen | 20 | normal | -0.0503 | -0.0485 | -0.0019 | +0.0034 |

## Direct contrastive-minus-normal differences

Negative target/specificity differences and positive nearest differences favor contrastive.

| Model | N | ΔΔ target | ΔΔ specificity | ΔΔ nearest | p target | p specificity | p nearest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen | 20 | -0.0780 | -0.0167 | +0.0509 | 1.48e-30 | 2.25e-19 | 1.94e-15 |
| llama | 20 | +0.0286 | +0.0025 | -0.0075 | 1.11e-18 | 0.0698 | 0.25 |
| mistral | 2 | +0.0225 | -0.0016 | +0.0040 | 2.82e-21 | 0.154 | 0.376 |

### Differences by attribute at the paper magnitudes

| Model | Attribute | ΔΔ target | ΔΔ specificity | ΔΔ nearest |
| --- | --- | ---: | ---: | ---: |
| qwen | age | -0.0406 | -0.0051 | +0.0187 |
| qwen | education | -0.0749 | -0.0204 | +0.0538 |
| qwen | gender | -0.1567 | -0.0369 | +0.1333 |
| qwen | socioeco | -0.0412 | -0.0023 | -0.0036 |
| llama | age | +0.0169 | -0.0013 | -0.0062 |
| llama | education | +0.0145 | +0.0044 | -0.0062 |
| llama | gender | +0.0526 | +0.0110 | -0.0403 |
| llama | socioeco | +0.0374 | -0.0053 | +0.0221 |
| mistral | age | -0.0086 | +0.0020 | +0.0021 |
| mistral | education | +0.0346 | +0.0019 | -0.0161 |
| mistral | gender | +0.0384 | -0.0089 | +0.0320 |
| mistral | socioeco | +0.0194 | -0.0032 | +0.0082 |

## Descriptive best magnitudes

These optima are descriptive and were not selected on an independent validation set.

| Model | Mode | Criterion | Best N | Value |
| --- | --- | --- | ---: | ---: |
| llama | contrastive | delta_nearest | 8 | +0.0038 |
| llama | contrastive | delta_specificity | 8 | +0.0026 |
| llama | contrastive | delta_target | 20 | -0.0844 |
| llama | normal | delta_nearest | 16 | +0.0037 |
| llama | normal | delta_specificity | 16 | +0.0020 |
| llama | normal | delta_target | 20 | -0.1131 |
| mistral | contrastive | delta_nearest | 2 | +0.0040 |
| mistral | contrastive | delta_specificity | 2 | -0.0023 |
| mistral | contrastive | delta_target | 2 | -0.2408 |
| mistral | normal | delta_nearest | 2 | +0.0000 |
| mistral | normal | delta_specificity | 8 | -0.0010 |
| mistral | normal | delta_target | 2 | -0.2633 |
| qwen | contrastive | delta_nearest | 20 | +0.0543 |
| qwen | contrastive | delta_specificity | 20 | -0.0185 |
| qwen | contrastive | delta_target | 20 | -0.1283 |
| qwen | normal | delta_nearest | 20 | +0.0034 |
| qwen | normal | delta_specificity | 20 | -0.0019 |
| qwen | normal | delta_target | 20 | -0.0503 |

## Class-level failures at the paper magnitudes

A class is listed when target distance or specificity worsens relative to N=0.

| Model | Mode | Attribute/class | Δ target | Δ specificity |
| --- | --- | --- | ---: | ---: |
| qwen | normal | gender/female | -0.0510 | +0.0095 |
| qwen | normal | age/older_adult | -0.0030 | +0.0085 |
| qwen | normal | education/college and more | -0.0585 | +0.0104 |
| qwen | normal | socioeco/low | +0.0085 | -0.0023 |
| qwen | contrastive | gender/female | -0.1172 | +0.0173 |
| qwen | contrastive | age/older_adult | +0.0269 | +0.0215 |
| qwen | contrastive | education/college and more | -0.1370 | +0.0193 |
| qwen | contrastive | socioeco/high | -0.0711 | +0.0060 |
| llama | normal | gender/female | -0.1741 | +0.0164 |
| llama | normal | age/older_adult | -0.0529 | +0.0118 |
| llama | normal | education/high school | -0.0171 | +0.0262 |
| llama | normal | education/college and more | -0.1976 | +0.0317 |
| llama | normal | socioeco/low | +0.0680 | -0.0343 |
| llama | normal | socioeco/high | -0.1795 | +0.0466 |
| llama | contrastive | gender/female | -0.1192 | +0.0289 |
| llama | contrastive | age/older_adult | -0.0180 | +0.0153 |
| llama | contrastive | education/high school | +0.0246 | +0.0290 |
| llama | contrastive | education/college and more | -0.2061 | +0.0370 |
| llama | contrastive | socioeco/low | +0.0982 | -0.0386 |
| llama | contrastive | socioeco/high | -0.1348 | +0.0403 |
| mistral | normal | gender/female | -0.1907 | +0.0571 |
| mistral | normal | age/older_adult | -0.2559 | +0.0263 |
| mistral | normal | education/college and more | -0.2368 | +0.0922 |
| mistral | normal | socioeco/high | -0.2712 | +0.0739 |
| mistral | contrastive | gender/female | -0.1092 | +0.0694 |
| mistral | contrastive | age/older_adult | -0.2614 | +0.0328 |
| mistral | contrastive | education/college and more | -0.2160 | +0.1023 |
| mistral | contrastive | socioeco/high | -0.2581 | +0.0652 |

## Interpretation

At the paper magnitudes, contrastive steering clearly improves Qwen: it reduces target distance more, improves the specificity margin, and raises nearest-group recovery relative to raw steering. For Llama it performs worse on all three aggregate outcomes. For Mistral it produces a much smaller change: target-distance improvement is weaker, while specificity and nearest-group differences are close to zero. Thus the contrast is a model-specific improvement, not a generally superior steering rule.

## Input audit

- `qwen/normal`: 47982 complete conditions; 0 incomplete conditions excluded; input `C:\Users\mario\OneDrive - csr.ufmg.br\Documents\GitHub\internal_LLM_bias\results\demographic_opinionqa_qwen.csv`.
- `qwen/contrastive`: 47982 complete conditions; 0 incomplete conditions excluded; input `C:\Users\mario\OneDrive - csr.ufmg.br\Documents\GitHub\internal_LLM_bias\results\demographic_opinionqa_qwen_contrastive.csv`.
- `llama/normal`: 47982 complete conditions; 0 incomplete conditions excluded; input `C:\Users\mario\OneDrive - csr.ufmg.br\Documents\GitHub\internal_LLM_bias\results\demographic_opinionqa_llama.csv`.
- `llama/contrastive`: 47982 complete conditions; 0 incomplete conditions excluded; input `C:\Users\mario\OneDrive - csr.ufmg.br\Documents\GitHub\internal_LLM_bias\results\demographic_opinionqa_llama_contrastive_evaluable.csv`.
- `mistral/normal`: 47982 complete conditions; 0 incomplete conditions excluded; input `C:\Users\mario\OneDrive - csr.ufmg.br\Documents\GitHub\internal_LLM_bias\results\demographic_opinionqa_mistral.csv`.
- `mistral/contrastive`: 47982 complete conditions; 0 incomplete conditions excluded; input `C:\Users\mario\OneDrive - csr.ufmg.br\Documents\GitHub\internal_LLM_bias\results\demographic_opinionqa_mistral_contrastive_evaluable.csv`.

## Output tables

- `mode_summary_by_magnitude.csv`
- `paired_mode_differences.csv`
- `paired_comparison_summary.csv`
- `summary_by_attribute_class.csv`
- `fixed_magnitude_summary.csv`
- `best_magnitudes_descriptive.csv`
- `baseline_agreement.csv`
- `comparison_curves.png`
