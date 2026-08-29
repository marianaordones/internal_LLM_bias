# Internal LLM Bias

This repository studies how demographic user profiles influence language-model responses to public-opinion questions from [OpinionQA](https://github.com/tatsu-lab/opinions_qa).

It evaluates three experimental settings:

- **Declared:** the user explicitly states a demographic profile in the prompt.
- **Steered:** pretrained demographic directions are added to internal model activations.
- **Inferred:** reading probes estimate a profile from contextual cues, such as a first name.

The evaluated attributes are gender, age, education, and socioeconomic status. Model response distributions can be compared with the corresponding human subgroup distributions in OpinionQA.

## Repository structure

```text
experiments/   Experiment runners
src/           Shared loading, prompting, steering, and probe utilities
analyses/      Analysis notebooks, derived tables, and plots
data/          OpinionQA data, names, and probe checkpoints
results/       Raw experiment outputs
```

## Setup

The main dependencies are:

```bash
python -m pip install torch transformers datasets numpy pandas matplotlib scipy jupyter
```

The default model is `meta-llama/Llama-2-13b-chat-hf`. The included probes were trained for this model and are not directly compatible with arbitrary architectures.

The TalkTuner checkpoints are stored as:

```text
data/probe_checkpoints/controlling_probe.zip
data/probe_checkpoints/reading_probe.zip
```

Extract both archives before running the experiments. The resulting checkpoints should be available under:

```text
data/probe_checkpoints/controlling_probe/
data/probe_checkpoints/reading_probe/
```

## Training probes for another model

The repository includes TalkTuner's compressed synthetic conversations and a
model-agnostic trainer. For Qwen2.5-7B-Instruct, first run a small end-to-end test:

```bash
python training/train_demographic_probes.py \
  --attributes gender \
  --channels reading \
  --limit 100 \
  --epochs 2 \
  --output-dir data/probe_checkpoints/qwen2.5-7b-instruct-smoke
```

Then train all reading and controlling probes:

```bash
python training/train_demographic_probes.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --attributes all \
  --channels reading,controlling
```

The extraction stage caches final-token hidden states before fitting one linear
probe per layer. To run the expensive model pass and the lightweight fitting step
separately, use `--stage extract` followed by `--stage train` with the same paths.
The defaults reproduce TalkTuner's sigmoid/BCE objective, stratified 80/20 split,
50 epochs, and seed 12345. Checkpoints and per-attribute metadata are written below
`data/probe_checkpoints/qwen2.5-7b-instruct/`.
See `training/README.md` for the two-stage workflow and output layout.


## Experiments

### Declared and steered demographic profiles

```bash
python experiments/demographic_opinionqa_experiment.py \
  --attributes all \
  --channels both \
  --magnitudes 0,1,3,5,7,8,9,13 \
  --out results/demo_full.csv
```

Available classes:

| Attribute | Classes |
| --- | --- |
| Gender | `male`, `female` |
| Age | `child`, `adolescent`, `adult`, `older_adult` |
| Education | `some schooling`, `high school`, `college and more` |
| Socioeconomic status | `low`, `mid`, `high` |

Use `--channels declared` or `--channels steered` to run only one channel. Use `--attributes gender,age`, for example, to select specific attributes.

### Opposed profiles

This experiment combines a declared profile with steering toward the opposite class.

```bash
python experiments/opposed_demographic_opinionqa_experiment.py \
  --magnitudes 1,3,5,7,9,13 \
  --out results/opposed_full.csv
```

### Gender inferred from first names

This experiment prepends `Hi! My name is NAME.` to each question and evaluates gender reading probes after the name sentence and at the final prompt token.

```bash
python experiments/inferred_gender_names_opinionqa.py \
  --out results/inferred_gender_names_full.csv
```

### SubPOP demographic profiles

The SubPOP runner evaluates the extreme sex, education, and income groups using
neutral (`steered`, magnitude 0), declared, and steered prompts. By default,
`--split all` combines `subpop_train.jsonl` and `subpop_eval.jsonl`. Use
`--split train` or `--split test` to select only one source file.

For Llama 2 with the original TalkTuner probes:

```bash
python experiments/demographic_subpop_experiment.py \
  --model-profile llama \
  --attributes all \
  --channels both \
  --out results/demographic_subpop_llama.csv
```

For Qwen2.5-7B-Instruct with the newly trained probes:

```bash
python experiments/demographic_subpop_experiment.py \
  --model-profile qwen \
  --attributes all \
  --channels both \
  --out results/demographic_subpop_qwen.csv
```

The gated dataset is loaded from `jjssuh/subpop` through the local Hugging Face
account. A downloaded JSONL can instead be supplied with `--dataset-file`. To
avoid redistributing gated content, result CSVs contain question IDs and model
distributions, but not question text, options, or human response distributions.


## Analysis

Generate general summaries with:

```bash
python analyses/analyze_demographic_opinionqa_results.py \
  --input results/demo_full.csv \
  --out-dir analyses/data \
  --prefix demographic
```

The analysis notebooks are:

- `analyses/gender_focus_analyses.ipynb`
- `analyses/additional_analyses.ipynb`
- `analyses/inferred_gender_names_analysis.ipynb`

Raw experiment outputs belong in `results/`; derived tables and plots belong in `analyses/`.

## Acknowledgments

The reading and controlling probes were trained and released by the [TalkTuner project](https://github.com/yc015/TalkTuner-chatbot-llm-dashboard), associated with [*Designing a Dashboard for Transparency and Control of Conversational AI*](https://arxiv.org/abs/2406.07882). Parts of the prompting and intervention procedure were adapted from its MIT-licensed implementation.

The public-opinion questions and human subgroup distributions are derived from [OpinionQA](https://github.com/tatsu-lab/opinions_qa), introduced in [*Whose Opinions Do Language Models Reflect?*](https://arxiv.org/abs/2303.17548).

Please cite both upstream projects when using this repository. Probe outputs are model predictions and should not be treated as verified demographic facts about individuals.

SubPOP experiments use the gated [SubPOP dataset](https://huggingface.co/datasets/jjssuh/subpop),
derived from Pew Research Center's American Trends Panel and the General Social
Survey. Use of SubPOP is subject to its non-commercial license and access terms.
The opinions expressed herein, including any implications for policy, are those
of the author and not of the survey research centers.
