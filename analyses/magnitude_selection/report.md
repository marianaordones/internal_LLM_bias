# Steering-magnitude selection

- Split seed: `42`
- Tuning questions: `243`
- Evaluation questions: `243`
- Selection objective: lowest tuning-set matching-group Wasserstein distance, macro-averaged with equal weight per attribute.
- Tie rule: smallest magnitude.

| Model | Selected N | Tuning distance | Grid | Boundary? |
| --- | ---: | ---: | --- | --- |
| qwen | 20 | 0.703 | 0--20 | yes |
| llama | 20 | 0.517 | 0--20 | yes |
| mistral | 2 | 0.403 | 0--20 | no |

## Held-out distance check

| Model | N | Neutral | Declared | Steered | Declared - neutral | Steered - neutral |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llama | 20 | 0.608 | 0.646 | 0.502 | +0.038 | -0.107 |
| mistral | 2 | 0.635 | 0.662 | 0.379 | +0.027 | -0.256 |
| qwen | 20 | 0.737 | 0.716 | 0.676 | -0.021 | -0.062 |

A boundary selection is the best tested magnitude, not evidence that the global optimum has been located.
