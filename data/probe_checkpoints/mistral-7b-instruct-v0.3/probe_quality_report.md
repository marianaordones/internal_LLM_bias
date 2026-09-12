# Probe quality report

- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Hidden size: `4096`
- Decoder layers: `32`
- Attributes: `gender, age, education, socioeconomic`
- Channels: `reading, controlling`

Validation accuracy measures demographic classification on the held-out split. It does not by itself establish that steering will improve human alignment.

## Best layer per probe

| Attribute | Channel | Checkpoint layer | Decoder layer | Accuracy | Balanced acc. | Macro-F1 | Best epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gender | reading | 23 | 22 | 0.9187 | 0.9187 | 0.9187 | 48 |
| age | reading | 32 | 31 | 0.9862 | 0.9862 | 0.9863 | 37 |
| education | reading | 30 | 29 | 0.9700 | 0.9700 | 0.9701 | 47 |
| socioeconomic | reading | 32 | 31 | 0.9733 | 0.9733 | 0.9733 | 37 |
| gender | controlling | 31 | 30 | 0.8979 | 0.8979 | 0.8979 | 50 |
| age | controlling | 32 | 31 | 0.9738 | 0.9738 | 0.9737 | 44 |
| education | controlling | 32 | 31 | 0.9333 | 0.9333 | 0.9337 | 48 |
| socioeconomic | controlling | 32 | 31 | 0.9617 | 0.9617 | 0.9619 | 50 |

## Reading: layers averaged across attributes

| Rank | Checkpoint layer | Decoder layer | Mean accuracy |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 31 | 0.9621 |
| 2 | 28 | 27 | 0.9570 |
| 3 | 29 | 28 | 0.9568 |
| 4 | 26 | 25 | 0.9568 |
| 5 | 30 | 29 | 0.9566 |
| 6 | 31 | 30 | 0.9566 |
| 7 | 23 | 22 | 0.9556 |
| 8 | 27 | 26 | 0.9555 |
| 9 | 25 | 24 | 0.9551 |
| 10 | 22 | 21 | 0.9535 |

Best contiguous windows (directly usable as steering layer ranges):

| Layers | Checkpoints | `--from-idx` | `--to-idx` | Mean accuracy |
| ---: | --- | ---: | ---: | ---: |
| 8 | 25–32 | 24 | 32 | 0.9571 |
| 9 | 24–32 | 23 | 32 | 0.9566 |
| 10 | 23–32 | 22 | 32 | 0.9565 |

## Controlling: layers averaged across attributes

| Rank | Checkpoint layer | Decoder layer | Mean accuracy |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 31 | 0.9417 |
| 2 | 31 | 30 | 0.9364 |
| 3 | 30 | 29 | 0.9345 |
| 4 | 29 | 28 | 0.9309 |
| 5 | 28 | 27 | 0.9276 |
| 6 | 27 | 26 | 0.9271 |
| 7 | 25 | 24 | 0.9236 |
| 8 | 26 | 25 | 0.9232 |
| 9 | 24 | 23 | 0.9186 |
| 10 | 23 | 22 | 0.9170 |

Best contiguous windows (directly usable as steering layer ranges):

| Layers | Checkpoints | `--from-idx` | `--to-idx` | Mean accuracy |
| ---: | --- | ---: | ---: | ---: |
| 8 | 25–32 | 24 | 32 | 0.9306 |
| 9 | 24–32 | 23 | 32 | 0.9293 |
| 10 | 23–32 | 22 | 32 | 0.9281 |
