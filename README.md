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

Install the project dependencies with:

```bash
python -m pip install -r requirements.txt
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
model-agnostic trainer with built-in Qwen and Mistral profiles. For
Qwen2.5-7B-Instruct, first run a small end-to-end test:

```bash
python training/train_demographic_probes.py \
  --model-profile qwen \
  --attributes gender \
  --channels reading \
  --limit 100 \
  --epochs 2 \
  --output-dir data/probe_checkpoints/qwen2.5-7b-instruct-smoke
```

Then train all reading and controlling probes:

```bash
python training/train_demographic_probes.py \
  --model-profile qwen \
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

To train all probes for Mistral-7B-Instruct-v0.3:

```bash
python training/train_demographic_probes.py \
  --model-profile mistral \
  --attributes all \
  --channels reading,controlling
```

Each model directory receives a compact Markdown quality report and CSV tables
with per-probe metrics and layer rankings.


## Experiments

### Declared and steered demographic profiles

```bash
python experiments/demographic_opinionqa_experiment.py \
  --model-profile llama \
  --attributes all \
  --channels both \
  --magnitudes 0,2,4,6,8,10,12,14,16,18,20 \
  --from-idx 19 \
  --to-idx 29 \
  --out results/demographic_opinionqa_llama.csv
```

Available classes:

| Attribute | Classes |
| --- | --- |
| Gender | `male`, `female` |
| Age | `child`, `adolescent`, `adult`, `older_adult` |
| Education | `some schooling`, `high school`, `college and more` |
| Socioeconomic status | `low`, `mid`, `high` |

Use `--channels declared` or `--channels steered` to run only one channel. Use `--attributes gender,age`, for example, to select specific attributes.

To run the same OpinionQA experiment with Qwen2.5-7B-Instruct and its matching
probes:

```bash
python experiments/demographic_opinionqa_experiment.py \
  --model-profile qwen \
  --attributes all \
  --channels both \
  --magnitudes 0,2,4,6,8,10,12,14,16,18,20 \
  --from-idx 18 \
  --to-idx 28 \
  --out results/demographic_opinionqa_qwen.csv
```

The Qwen profile uses `Qwen/Qwen2.5-7B-Instruct`, bfloat16, its native chat
template, and checkpoints under
`data/probe_checkpoints/qwen2.5-7b-instruct/controlling_probe/`. Explicit
`--model`, `--probe-dir`, and layer arguments override profile defaults.

For Mistral-7B-Instruct-v0.3 and its matching probes:

```bash
python experiments/demographic_opinionqa_experiment.py \
  --model-profile mistral \
  --attributes all \
  --channels both \
  --magnitudes 0,2,4,6,8,10,12,14,16,18,20 \
  --from-idx 22 \
  --to-idx 32 \
  --out results/demographic_opinionqa_mistral.csv
```

The layer ranges above match the experiment CSVs used in the paper.

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

Generate the human-alignment reports at the comparison magnitudes used in the
paper:

```bash
python analyses/analyze_opinionqa_model.py \
  --input results/demographic_opinionqa_qwen.csv \
  --comparison-magnitude 20

python analyses/analyze_opinionqa_model.py \
  --input results/demographic_opinionqa_llama.csv \
  --comparison-magnitude 20

python analyses/analyze_opinionqa_model.py \
  --input results/demographic_opinionqa_mistral.csv \
  --comparison-magnitude 2
```

Each model receives a separate directory under `analyses/model_reports/`. Edit
`CLASS_TO_HUMAN` and `GAP_PAIRS` near the top of the analysis file to change the
mapping between probe classes and OpinionQA groups.

To test whether alignment is specific to the intended subgroup rather than a
uniform movement toward every human group, run:

```bash
python analyses/analyze_cross_group_specificity.py \
  --input results/demographic_opinionqa_qwen.csv \
  --comparison-magnitude 20
```

This produces cross-group distances, nearest-human-group confusion matrices,
own-versus-other specificity margins, and one-sided paired Wilcoxon tests under
`analyses/cross_group_reports/<model>/`.

Measure the human-only separability ceiling without loading model results:

```bash
python analyses/analyze_human_separability.py
```

The two tables under `analyses/human_separability/` contain per-question human
subgroup gaps and attribute/pair summaries with quartiles, near-zero fractions,
and exact and threshold-aware human self-retrieval ceilings.

After producing both the human-separability tables and a model's cross-group
report, test whether profile recovery improves on questions with larger human
subgroup gaps:

```bash
python analyses/analyze_specificity_by_human_gap.py \
  --predictions analyses/cross_group_reports/qwen/nearest_group_predictions.csv \
  --comparison-magnitude 20
```

The report includes pair-specific human-gap quartiles, question-clustered
bootstrap intervals for nearest-group accuracy, continuous-gap logistic trends,
and quartile-stratified directional Wilcoxon tests.

Select one magnitude per model on tuning questions, then generate the held-out
publication figures:

```bash
python analyses/select_steering_magnitude.py
python analyses/paper_figs_heldout.py
```

The first script creates a fixed question-level tuning/evaluation split and
selects `Qwen=20`, `Llama=20`, and `Mistral=2` under the current results. The
second script uses evaluation questions only and writes PNG/PDF figures plus
their source tables under `analyses/paper_figures_heldout/`.

## Acknowledgments

The reading and controlling probes were trained and released by the [TalkTuner project](https://github.com/yc015/TalkTuner-chatbot-llm-dashboard), associated with [*Designing a Dashboard for Transparency and Control of Conversational AI*](https://arxiv.org/abs/2406.07882). Parts of the prompting and intervention procedure were adapted from its MIT-licensed implementation.

The public-opinion questions and human subgroup distributions are derived from [OpinionQA](https://github.com/tatsu-lab/opinions_qa), introduced in [*Whose Opinions Do Language Models Reflect?*](https://arxiv.org/abs/2303.17548).

Please cite both upstream projects when using this repository. Probe outputs are model predictions and should not be treated as verified demographic facts about individuals.
