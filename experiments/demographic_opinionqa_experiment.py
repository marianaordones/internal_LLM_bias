#!/usr/bin/env python
"""Run declared and/or steered demographic experiments on OpinionQA.

The script records response distributions only; aggregation and comparison with
human OpinionQA distributions are handled by the analysis code.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Allow direct execution with ``python experiments/<script>.py`` from any directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (  # noqa: E402
    ATTRIBUTE_CONFIG,
    DEFAULT_CONTROLLING_PROBE_DIR,
    DEFAULT_MODEL,
    DEFAULT_QKEY_DICT,
)
from src.experiment_utils import (  # noqa: E402
    make_demographic_row,
    parse_attributes,
    parse_magnitudes,
)
from src.modeling import (  # noqa: E402
    llama_v2_prompt,
    option_token_ids,
    option_token_ids_for_prompt,
    set_seed,
)
from src.opinionqa import build_user_message, load_questions  # noqa: E402
from src.steering import (  # noqa: E402
    load_steer_vectors,
    option_distribution,
    resolve_controlling_probe_dir,
    steered_distribution,
)


MODEL_PROFILES = {
    "llama": {
        "model_id": DEFAULT_MODEL,
        "probe_dir": DEFAULT_CONTROLLING_PROBE_DIR,
        "from_idx": 19,
        "to_idx": 29,
        "dtype": "float16",
        "trained_probe_names": False,
    },
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "probe_dir": REPO_ROOT / "data/probe_checkpoints/qwen2.5-7b-instruct/controlling_probe",
        "from_idx": 18,
        "to_idx": 28,
        "dtype": "bfloat16",
        "trained_probe_names": True,
    },
    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "probe_dir": REPO_ROOT / "data/probe_checkpoints/mistral-7b-instruct-v0.3/controlling_probe",
        # Use all decoder layers by default. Prefer the best window reported by
        # probe_quality_report.md once the model's probes have been trained.
        "from_idx": 0,
        "to_idx": 32,
        "dtype": "bfloat16",
        "trained_probe_names": True,
    },
}


def format_prompt(tokenizer, model_profile, message):
    """Use the selected instruction model's native conversation format."""
    if model_profile == "llama":
        return llama_v2_prompt([{"role": "user", "content": message}])
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
    )


def candidate_ids(tokenizer, prompt, letters, model_profile, llama_letter_ids):
    # Preserve the original Llama behavior; derive native-model tokens in context.
    if model_profile == "llama":
        return [llama_letter_ids[letter] for letter in letters]
    contextual = option_token_ids_for_prompt(tokenizer, prompt, letters)
    return [contextual[letter] for letter in letters]


def add_run_metadata(row, model_profile, model_id, from_idx, to_idx):
    row.update({
        "model_profile": model_profile,
        "model_id": model_id,
        "from_idx": from_idx,
        "to_idx_exclusive": to_idx,
    })
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-profile", choices=MODEL_PROFILES, default="llama",
        help="Select model, probes, prompt format, dtype, and default steering layers.",
    )
    parser.add_argument("--attributes", default="all", help="Comma-separated: gender, age, education, socioeco, or all")
    parser.add_argument("--channels", choices=["declared", "steered", "both"], default="both")
    parser.add_argument("--magnitudes", default="0,1,3,5,7,8,9,13", help="Comma-separated steering magnitudes")
    parser.add_argument("--qkey-dict", type=Path, default=DEFAULT_QKEY_DICT)
    parser.add_argument("--probe-dir", type=Path, help="Override the profile's controlling-probe directory")
    parser.add_argument("--model", help="Override the profile's Hugging Face model ID or local path")
    parser.add_argument("--from-idx", type=int, help="First decoder layer to steer")
    parser.add_argument("--to-idx", type=int, help="Exclusive last decoder layer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="0 means all questions")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate the pipeline using uniform distributions")
    args = parser.parse_args()

    profile = MODEL_PROFILES[args.model_profile]
    model_id = args.model or profile["model_id"]
    probe_dir_arg = args.probe_dir or profile["probe_dir"]
    from_idx = profile["from_idx"] if args.from_idx is None else args.from_idx
    to_idx = profile["to_idx"] if args.to_idx is None else args.to_idx
    default_outputs = {
        "llama": REPO_ROOT / "demographic_opinionqa_results.csv",
        "qwen": REPO_ROOT / "results/demographic_opinionqa_qwen.csv",
        "mistral": REPO_ROOT / "results/demographic_opinionqa_mistral.csv",
    }
    output = args.out or default_outputs[args.model_profile]

    attributes = parse_attributes(args.attributes)
    magnitudes = parse_magnitudes(args.magnitudes)
    run_declared = args.channels in {"declared", "both"}
    run_steered = args.channels in {"steered", "both"}
    if from_idx < 0 or to_idx <= from_idx:
        parser.error("Require 0 <= from-idx < to-idx.")
    if args.limit < 0:
        parser.error("--limit must be zero or positive.")

    questions = load_questions(args.qkey_dict)
    if args.limit:
        questions = questions[:args.limit]
    set_seed(args.seed, use_torch=not args.dry_run)

    model = tokenizer = probe_dir = None
    steer_vectors = {}
    llama_letter_ids = {}
    if run_steered and not args.dry_run:
        probe_dir = resolve_controlling_probe_dir(probe_dir_arg)

    if not args.dry_run:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = getattr(torch, profile["dtype"]) if args.device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype
        ).to(args.device).eval()
        if to_idx > model.config.num_hidden_layers:
            parser.error(
                f"--to-idx={to_idx} exceeds the model's "
                f"{model.config.num_hidden_layers} decoder layers."
            )
        if args.model_profile == "llama":
            llama_letter_ids = option_token_ids(tokenizer)
        if run_steered:
            for attribute in attributes:
                config = ATTRIBUTE_CONFIG[attribute]
                probe_name = config["probe_name"]
                if profile["trained_probe_names"] and attribute == "socioeco":
                    probe_name = "socioeconomic"
                steer_vectors[attribute] = load_steer_vectors(
                    probe_dir,
                    attribute,
                    from_idx,
                    to_idx,
                    args.device,
                    probe_name=probe_name,
                    class_indices=list(range(len(config["classes"]))),
                    expected_num_classes=len(config["classes"]),
                    expected_hidden_size=model.config.hidden_size,
                )

    print(
        f"[info] questions={len(questions)} model={args.model_profile}:{model_id} "
        f"attributes={attributes} channels={args.channels} "
        f"magnitudes={magnitudes if run_steered else 'n/a'} layers={from_idx}:{to_idx} "
        f"seed={args.seed} dry_run={args.dry_run}"
    )
    if probe_dir:
        print(f"[info] probe_dir={probe_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    # Neutral responses are identical across attributes and classes for a question.
    neutral_cache = {}
    with open(output, "w", newline="", encoding="utf-8") as stream:
        writer = None
        for question_idx, question in enumerate(questions, start=1):
            for attribute in attributes:
                config = ATTRIBUTE_CONFIG[attribute]
                for class_idx, declaration in enumerate(config["declarations"]):
                    # Declared and steered channels share questions but use separate prompts.
                    if run_declared:
                        message, letters = build_user_message(
                            question["question"], question["options"], declaration
                        )
                        if args.dry_run:
                            distribution = np.full(len(letters), 1.0 / len(letters))
                        else:
                            prompt = format_prompt(tokenizer, args.model_profile, message)
                            input_ids = tokenizer(
                                prompt, return_tensors="pt", add_special_tokens=False
                            ).input_ids.to(args.device)
                            distribution = option_distribution(
                                model,
                                input_ids,
                                candidate_ids(
                                    tokenizer, prompt, letters, args.model_profile,
                                    llama_letter_ids,
                                ),
                            )
                        row = make_demographic_row(
                            question, attribute, "declared", class_idx, None,
                            declaration, distribution, args.seed
                        )
                        row = add_run_metadata(
                            row, args.model_profile, model_id, from_idx, to_idx
                        )
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        row_count += 1

                    if run_steered:
                        message, letters = build_user_message(question["question"], question["options"])
                        input_ids = candidates = None
                        if not args.dry_run:
                            prompt = format_prompt(tokenizer, args.model_profile, message)
                            input_ids = tokenizer(
                                prompt, return_tensors="pt", add_special_tokens=False
                            ).input_ids.to(args.device)
                            candidates = candidate_ids(
                                tokenizer, prompt, letters, args.model_profile,
                                llama_letter_ids,
                            )
                        for magnitude in magnitudes:
                            if args.dry_run:
                                distribution = np.full(len(letters), 1.0 / len(letters))
                            elif magnitude == 0 and question["qkey"] in neutral_cache:
                                distribution = neutral_cache[question["qkey"]]
                            else:
                                distribution = steered_distribution(
                                    model, input_ids, candidates, steer_vectors[attribute],
                                    class_idx, magnitude
                                )
                                if magnitude == 0:
                                    neutral_cache[question["qkey"]] = distribution
                            row = make_demographic_row(
                                question, attribute, "steered", class_idx, magnitude,
                                None, distribution, args.seed
                            )
                            row = add_run_metadata(
                                row, args.model_profile, model_id, from_idx, to_idx
                            )
                            if writer is None:
                                writer = csv.DictWriter(stream, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            row_count += 1

            stream.flush()
            if question_idx % 25 == 0 or question_idx == len(questions):
                print(f"[progress] {question_idx}/{len(questions)} questions | rows={row_count}")

    if row_count == 0:
        raise RuntimeError("The experiment produced no rows.")
    print(f"[ok] wrote {row_count} rows to {output.resolve()}")


if __name__ == "__main__":
    main()
