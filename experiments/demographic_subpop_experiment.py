#!/usr/bin/env python3
"""Run neutral, declared, and steered demographic experiments on SubPOP."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment_utils import parse_magnitudes  # noqa: E402
from src.modeling import (  # noqa: E402
    llama_v2_prompt,
    option_token_ids_for_prompt,
    serialize_distribution,
    set_seed,
)
from src.opinionqa import build_user_message  # noqa: E402
from src.steering import (  # noqa: E402
    load_steer_vectors,
    option_distribution,
    resolve_controlling_probe_dir,
    steered_distribution,
)
from src.subpop import (  # noqa: E402
    SUBPOP_ATTRIBUTE_CONFIG,
    load_subpop_questions,
    parse_subpop_attributes,
)


MODEL_PROFILES = {
    "llama": {
        "model_id": "meta-llama/Llama-2-13b-chat-hf",
        "probe_dir": REPO_ROOT / "data/probe_checkpoints/controlling_probe",
        "from_idx": 19,
        "to_idx": 29,
        "dtype": "float16",
    },
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "probe_dir": REPO_ROOT / "data/probe_checkpoints/qwen2.5-7b-instruct/controlling_probe",
        # This covers approximately the same relative depth as Llama layers 19:29.
        "from_idx": 13,
        "to_idx": 20,
        "dtype": "bfloat16",
    },
}


def format_prompt(tokenizer, model_profile: str, message: str) -> str:
    if model_profile == "llama":
        return llama_v2_prompt([{"role": "user", "content": message}])
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
    )


def make_row(
    *, question, attribute, class_config, channel, magnitude, declaration,
    distribution, model_profile, model_id, split, seed, from_idx, to_idx,
):
    dataset_attribute = SUBPOP_ATTRIBUTE_CONFIG[attribute]["dataset_attribute"]
    letters = [chr(ord("A") + index) for index in range(len(question["options"]))]
    return {
        "qkey": question["qkey"],
        "dataset": "jjssuh/subpop",
        "dataset_split": question.get("source_split", split),
        "model_profile": model_profile,
        "model_id": model_id,
        "attribute": attribute,
        "subpop_attribute": dataset_attribute,
        "class_label": class_config["label"],
        "probe_class_idx": class_config["probe_class_idx"],
        "human_group": class_config["group"],
        "channel": channel,
        "magnitude": "" if magnitude is None else magnitude,
        "declaration": declaration or "",
        "from_idx": "" if channel != "steered" else from_idx,
        "to_idx_exclusive": "" if channel != "steered" else to_idx,
        "n_options": len(question["options"]),
        "option_letters": json.dumps(letters),
        "response_distribution": serialize_distribution(distribution),
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-profile", choices=MODEL_PROFILES, required=True)
    parser.add_argument("--model-id", help="Override the Hugging Face model in the selected profile.")
    parser.add_argument("--attributes", default="all", help="Comma-separated: sex, education, income, or all")
    parser.add_argument("--channels", choices=("declared", "steered", "both"), default="both")
    parser.add_argument("--magnitudes", default="0,1,3,5,7,8,9,13")
    parser.add_argument("--dataset-id", default="jjssuh/subpop")
    parser.add_argument(
        "--split", choices=("all", "train", "test"), default="all",
        help="Use both Hugging Face files, only SubPOP-Train, or only SubPOP-Eval.",
    )
    parser.add_argument("--dataset-file", type=Path, help="Optional local SubPOP JSONL instead of Hugging Face.")
    parser.add_argument("--probe-dir", type=Path)
    parser.add_argument("--from-idx", type=int)
    parser.add_argument("--to-idx", type=int, help="Exclusive decoder-layer endpoint.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile = MODEL_PROFILES[args.model_profile]
    model_id = args.model_id or profile["model_id"]
    probe_dir_arg = args.probe_dir or profile["probe_dir"]
    from_idx = profile["from_idx"] if args.from_idx is None else args.from_idx
    to_idx = profile["to_idx"] if args.to_idx is None else args.to_idx
    output = args.out or REPO_ROOT / "results" / f"demographic_subpop_{args.model_profile}.csv"
    attributes = parse_subpop_attributes(args.attributes)
    magnitudes = parse_magnitudes(args.magnitudes)
    run_declared = args.channels in {"declared", "both"}
    run_steered = args.channels in {"steered", "both"}
    if from_idx < 0 or to_idx <= from_idx:
        parser.error("Require 0 <= from-idx < to-idx.")
    if args.limit < 0:
        parser.error("--limit must be zero or positive.")

    questions = load_subpop_questions(
        dataset_id=args.dataset_id,
        split=args.split,
        attributes=attributes,
        dataset_file=args.dataset_file,
    )
    if args.limit:
        questions = questions[: args.limit]
    set_seed(args.seed, use_torch=not args.dry_run)

    model = tokenizer = probe_dir = None
    steer_vectors = {}
    if run_steered and not args.dry_run:
        probe_dir = resolve_controlling_probe_dir(probe_dir_arg)

    if not args.dry_run:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = getattr(torch, profile["dtype"])
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(args.device).eval()
        if to_idx > model.config.num_hidden_layers:
            parser.error(
                f"--to-idx={to_idx} exceeds the model's {model.config.num_hidden_layers} decoder layers."
            )
        if run_steered:
            for attribute in attributes:
                config = SUBPOP_ATTRIBUTE_CONFIG[attribute]
                full_class_count = 2 if attribute == "sex" else 3
                steer_vectors[attribute] = load_steer_vectors(
                    probe_dir,
                    config["probe_attribute"],
                    from_idx,
                    to_idx,
                    args.device,
                    probe_name=config[f"{args.model_profile}_probe_name"],
                    class_indices=[item["probe_class_idx"] for item in config["classes"]],
                    expected_num_classes=full_class_count,
                    expected_hidden_size=model.config.hidden_size,
                )

    print(
        f"[info] dataset={args.dataset_id}/{args.split} questions={len(questions)} "
        f"model={args.model_profile}:{model_id} attributes={attributes} channels={args.channels} "
        f"magnitudes={magnitudes if run_steered else 'n/a'} layers={from_idx}:{to_idx} "
        f"seed={args.seed} dry_run={args.dry_run}",
        flush=True,
    )
    if probe_dir:
        print(f"[info] probe_dir={probe_dir}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    neutral_cache = {}
    row_count = 0
    with open(output, "w", newline="", encoding="utf-8") as stream:
        writer = None
        for question_idx, question in enumerate(questions, start=1):
            for attribute in attributes:
                config = SUBPOP_ATTRIBUTE_CONFIG[attribute]
                for local_class_idx, class_config in enumerate(config["classes"]):
                    if run_declared:
                        message, letters = build_user_message(
                            question["question"], question["options"], class_config["declaration"]
                        )
                        if args.dry_run:
                            distribution = np.full(len(letters), 1.0 / len(letters))
                        else:
                            prompt = format_prompt(tokenizer, args.model_profile, message)
                            input_ids = tokenizer(
                                prompt, return_tensors="pt", add_special_tokens=False
                            ).input_ids.to(args.device)
                            letter_ids = option_token_ids_for_prompt(tokenizer, prompt, letters)
                            distribution = option_distribution(
                                model, input_ids, [letter_ids[letter] for letter in letters]
                            )
                        row = make_row(
                            question=question, attribute=attribute, class_config=class_config,
                            channel="declared", magnitude=None,
                            declaration=class_config["declaration"], distribution=distribution,
                            model_profile=args.model_profile, model_id=model_id, split=args.split,
                            seed=args.seed, from_idx=from_idx, to_idx=to_idx,
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
                            letter_ids = option_token_ids_for_prompt(tokenizer, prompt, letters)
                            candidates = [letter_ids[letter] for letter in letters]
                        for magnitude in magnitudes:
                            cache_key = (question.get("source_split", args.split), question["qkey"])
                            if args.dry_run:
                                distribution = np.full(len(letters), 1.0 / len(letters))
                            elif magnitude == 0 and cache_key in neutral_cache:
                                distribution = neutral_cache[cache_key]
                            else:
                                distribution = steered_distribution(
                                    model, input_ids, candidates, steer_vectors[attribute],
                                    local_class_idx, magnitude,
                                )
                                if magnitude == 0:
                                    neutral_cache[cache_key] = distribution
                            row = make_row(
                                question=question, attribute=attribute, class_config=class_config,
                                channel="steered", magnitude=magnitude, declaration=None,
                                distribution=distribution, model_profile=args.model_profile,
                                model_id=model_id, split=args.split, seed=args.seed,
                                from_idx=from_idx, to_idx=to_idx,
                            )
                            if writer is None:
                                writer = csv.DictWriter(stream, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            row_count += 1
            stream.flush()
            if question_idx % 25 == 0 or question_idx == len(questions):
                print(f"[progress] {question_idx}/{len(questions)} questions | rows={row_count}", flush=True)

    if not row_count:
        raise RuntimeError("The experiment produced no rows.")
    print(f"[ok] wrote {row_count} rows to {output.resolve()}")


if __name__ == "__main__":
    main()
