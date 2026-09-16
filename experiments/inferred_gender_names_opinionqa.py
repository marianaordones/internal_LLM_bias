#!/usr/bin/env python
"""Test demographic inference from strongly gender-associated names on OpinionQA.

Each forward pass records the answer distribution and readings from the gender,
age, education, and socioeconomic probes after the name and at the final token.
"""

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

from src.config import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_NAMES,
    DEFAULT_QKEY_DICT,
    DEFAULT_READING_PROBE_DIR,
)
from src.modeling import (  # noqa: E402
    llama_v2_prompt,
    option_token_ids,
    option_token_ids_for_prompt,
    serialize_distribution,
    set_seed,
)
from src.opinionqa import build_named_user_message, load_questions  # noqa: E402
from src.reading_probes import (  # noqa: E402
    add_probe_readings,
    load_reading_probes,
    load_names,
    locate_name_end_token,
    make_dry_readings,
    read_all_probes,
    resolve_reading_probe_dir,
)


MODEL_PROFILES = {
    "llama": {
        "model_id": DEFAULT_MODEL,
        "probe_dir": DEFAULT_READING_PROBE_DIR,
        "dtype": "float16",
        "num_hidden_layers": 40,
    },
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "probe_dir": (
            REPO_ROOT
            / "data/probe_checkpoints/qwen2.5-7b-instruct/reading_probe"
        ),
        "dtype": "bfloat16",
        "num_hidden_layers": 28,
    },
    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "probe_dir": (
            REPO_ROOT
            / "data/probe_checkpoints/mistral-7b-instruct-v0.3/reading_probe"
        ),
        "dtype": "bfloat16",
        "num_hidden_layers": 32,
    },
}


def format_prompt(tokenizer, model_profile, message):
    """Preserve Llama-2 formatting and use native templates for newer models."""
    if model_profile == "llama":
        return llama_v2_prompt([{"role": "user", "content": message}])
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
    )


def candidate_ids(tokenizer, prompt, letters, model_profile, llama_letter_ids):
    if model_profile == "llama":
        return [llama_letter_ids[letter] for letter in letters]
    contextual = option_token_ids_for_prompt(tokenizer, prompt, letters)
    return [contextual[letter] for letter in letters]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-profile", choices=MODEL_PROFILES, default="llama")
    parser.add_argument("--names", type=Path, default=DEFAULT_NAMES)
    parser.add_argument("--min-dominance", type=float, default=0.95)
    parser.add_argument("--qkey-dict", type=Path, default=DEFAULT_QKEY_DICT)
    parser.add_argument("--reading-probe-dir", type=Path)
    parser.add_argument("--model", help="Override the profile's model ID or local path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Question limit; 0 means all")
    parser.add_argument("--name-limit", type=int, default=0, help="Per-label name limit; 0 means all")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 0.95 <= args.min_dominance <= 1:
        parser.error("--min-dominance must be in [0.95, 1].")
    if args.limit < 0 or args.name_limit < 0:
        parser.error("--limit and --name-limit must be zero or positive.")

    profile = MODEL_PROFILES[args.model_profile]
    model_id = args.model or profile["model_id"]
    probe_dir_arg = args.reading_probe_dir or profile["probe_dir"]
    output_path = args.out or (
        REPO_ROOT / f"results/inferred_gender_names_{args.model_profile}.csv"
    )

    questions = load_questions(args.qkey_dict)
    if args.limit:
        questions = questions[:args.limit]
    names = load_names(args.names, args.min_dominance)
    if args.name_limit:
        # Apply the limit independently so male and female names remain balanced.
        names = [
            name
            for label in ("male", "female")
            for name in [
                item for item in names if item["sex_label"] == label
            ][:args.name_limit]
        ]
    set_seed(args.seed, use_torch=not args.dry_run)

    probe_dir = None
    tokenizer = model = probes = None
    llama_letter_ids = {}
    probe_layers = list(range(profile["num_hidden_layers"] + 1))
    if not args.dry_run:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        probe_dir = resolve_reading_probe_dir(probe_dir_arg)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        dtype = (
            getattr(torch, profile["dtype"])
            if args.device.startswith("cuda")
            else torch.float32
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype
        ).to(args.device).eval()
        probes, probe_layers = load_reading_probes(
            probe_dir, args.device, expected_hidden_size=model.config.hidden_size
        )
        expected_layers = set(range(model.config.num_hidden_layers + 1))
        if set(probe_layers) != expected_layers:
            missing = sorted(expected_layers - set(probe_layers))
            extra = sorted(set(probe_layers) - expected_layers)
            raise ValueError(
                "Reading-probe layers do not match the model hidden states: "
                f"missing={missing}, extra={extra}"
            )
        if args.model_profile == "llama":
            llama_letter_ids = option_token_ids(tokenizer)

    print(
        f"[info] questions={len(questions)} names={len(names)} "
        f"model={args.model_profile}:{model_id} layers={probe_layers[0]}..{probe_layers[-1]} "
        f"forwards={len(questions) * len(names)} min_dominance={args.min_dominance} "
        f"seed={args.seed} dry_run={args.dry_run}"
    )
    print(f"[info] reading_probe_dir={probe_dir or probe_dir_arg}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as stream:
        writer = None
        for question_idx, question in enumerate(questions, start=1):
            for name_record in names:
                message, name_sentence, letters = build_named_user_message(
                    name_record["name"], question["question"], question["options"]
                )
                prompt = (
                    llama_v2_prompt([{"role": "user", "content": message}])
                    if args.dry_run and args.model_profile == "llama"
                    else None
                )

                if args.dry_run:
                    name_token_idx = final_token_idx = -1
                    response = np.full(len(letters), 1.0 / len(letters))
                    name_reading = make_dry_readings(
                        name_record["sex_label"], probe_layers
                    )
                    final_reading = make_dry_readings(
                        name_record["sex_label"], probe_layers
                    )
                else:
                    import torch

                    prompt = format_prompt(tokenizer, args.model_profile, message)
                    # One forward pass supplies logits and both probe-reading positions.
                    name_token_idx, full_ids = locate_name_end_token(
                        tokenizer, prompt, name_sentence
                    )
                    input_ids = torch.tensor([full_ids], device=args.device)
                    final_token_idx = input_ids.shape[1] - 1
                    with torch.inference_mode():
                        model_output = model(
                            input_ids, output_hidden_states=True, use_cache=False
                        )
                    option_ids = torch.tensor(
                        candidate_ids(
                            tokenizer, prompt, letters, args.model_profile,
                            llama_letter_ids,
                        ),
                        device=args.device,
                    )
                    response = torch.softmax(
                        model_output.logits[0, -1, option_ids].float(), dim=-1
                    ).cpu().numpy()
                    name_reading = read_all_probes(
                        model_output.hidden_states,
                        name_token_idx,
                        probes,
                        probe_layers,
                    )
                    final_reading = read_all_probes(
                        model_output.hidden_states,
                        final_token_idx,
                        probes,
                        probe_layers,
                    )

                row = {
                    "qkey": question["qkey"],
                    "name": name_record["name"],
                    "ssa_sex_label": name_record["sex_label"],
                    "ssa_dominance": name_record["dominance"],
                    "ssa_aggregate_share": name_record["aggregate_share"],
                    "ssa_source_start_year": name_record["source_start_year"],
                    "ssa_source_end_year": name_record["source_end_year"],
                    "name_sentence": name_sentence,
                    "name_end_token_idx": name_token_idx,
                    "final_token_idx": final_token_idx,
                    "probe_layers": json.dumps(probe_layers),
                    "probe_attributes": json.dumps(
                        ["gender", "age", "education", "socioeco"]
                    ),
                    "n_options": len(question["options"]),
                    "option_letters": json.dumps(list(letters)),
                    "options": json.dumps(question["options"], ensure_ascii=False),
                    "response_distribution": serialize_distribution(response),
                    "question": question["question"],
                    "model_profile": args.model_profile,
                    "model_id": model_id,
                    "seed": args.seed,
                }
                # Store all layers in each row to keep one row per question-name pair.
                add_probe_readings(
                    row,
                    "after_name",
                    name_reading,
                    probe_layers,
                    expected_gender=name_record["sex_label"],
                )
                add_probe_readings(
                    row,
                    "final_prompt",
                    final_reading,
                    probe_layers,
                    expected_gender=name_record["sex_label"],
                )

                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                row_count += 1

            stream.flush()
            if question_idx % 10 == 0 or question_idx == len(questions):
                print(f"[progress] {question_idx}/{len(questions)} questions | rows={row_count}")

    if row_count == 0:
        raise RuntimeError("The experiment produced no rows.")
    print(f"[ok] wrote {row_count} rows to {output_path.resolve()}")


if __name__ == "__main__":
    main()
