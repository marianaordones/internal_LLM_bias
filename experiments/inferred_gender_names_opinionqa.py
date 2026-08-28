#!/usr/bin/env python
"""Test inferred gender from first names on OpinionQA with TalkTuner probes.

Each model forward pass records the response distribution and gender-probe
readings after the name sentence and at the final prompt token.
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
    PROBE_CLASSES,
    PROBE_LAYERS,
)
from src.modeling import (  # noqa: E402
    llama_v2_prompt,
    option_token_ids,
    serialize_distribution,
    set_seed,
)
from src.opinionqa import build_named_user_message, load_questions  # noqa: E402
from src.reading_probes import (  # noqa: E402
    add_probe_readings,
    load_gender_reading_probes,
    load_names,
    locate_name_end_token,
    make_dry_reading,
    read_all_probes,
    resolve_reading_probe_dir,
)


DEFAULT_OUTPUT = REPO_ROOT / "inferred_gender_names_opinionqa_results.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", type=Path, default=DEFAULT_NAMES)
    parser.add_argument("--min-dominance", type=float, default=0.95)
    parser.add_argument("--qkey-dict", type=Path, default=DEFAULT_QKEY_DICT)
    parser.add_argument("--reading-probe-dir", type=Path, default=DEFAULT_READING_PROBE_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Question limit; 0 means all")
    parser.add_argument("--name-limit", type=int, default=0, help="Per-label name limit; 0 means all")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 0.5 < args.min_dominance <= 1:
        parser.error("--min-dominance must be in (0.5, 1].")
    if args.limit < 0 or args.name_limit < 0:
        parser.error("--limit and --name-limit must be zero or positive.")

    questions = load_questions(args.qkey_dict)
    if args.limit:
        questions = questions[:args.limit]
    names = load_names(args.names, args.min_dominance)
    if args.name_limit:
        # Apply the limit independently so male and female names remain balanced.
        names = [
            name
            for label in PROBE_CLASSES
            for name in [
                item for item in names if item["sex_label"] == label
            ][:args.name_limit]
        ]
    set_seed(args.seed, use_torch=not args.dry_run)

    probe_dir = resolve_reading_probe_dir(args.reading_probe_dir)
    tokenizer = model = probes = None
    letter_ids = {}
    if not args.dry_run:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype
        ).to(args.device).eval()
        probes = load_gender_reading_probes(probe_dir, args.device)
        letter_ids = option_token_ids(tokenizer)

    print(
        f"[info] questions={len(questions)} names={len(names)} layers=0..40 "
        f"forwards={len(questions) * len(names)} seed={args.seed} dry_run={args.dry_run}"
    )
    print(f"[info] reading_probe_dir={probe_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with open(args.out, "w", newline="", encoding="utf-8") as stream:
        writer = None
        for question_idx, question in enumerate(questions, start=1):
            for name_record in names:
                message, name_sentence, letters = build_named_user_message(
                    name_record["name"], question["question"], question["options"]
                )
                prompt = llama_v2_prompt([{"role": "user", "content": message}])

                if args.dry_run:
                    name_token_idx = final_token_idx = -1
                    response = np.full(len(letters), 1.0 / len(letters))
                    name_reading = make_dry_reading(name_record["sex_label"])
                    final_reading = make_dry_reading(name_record["sex_label"])
                else:
                    import torch

                    # One forward pass supplies logits and both probe-reading positions.
                    name_token_idx, full_ids = locate_name_end_token(
                        tokenizer, prompt, name_sentence
                    )
                    input_ids = torch.tensor([full_ids], device=args.device)
                    final_token_idx = input_ids.shape[1] - 1
                    with torch.inference_mode():
                        output = model(input_ids, output_hidden_states=True, use_cache=False)
                    candidate_ids = torch.tensor(
                        [letter_ids[letter] for letter in letters], device=args.device
                    )
                    response = torch.softmax(
                        output.logits[0, -1, candidate_ids].float(), dim=-1
                    ).cpu().numpy()
                    name_reading = read_all_probes(
                        output.hidden_states, name_token_idx, probes
                    )
                    final_reading = read_all_probes(
                        output.hidden_states, final_token_idx, probes
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
                    "probe_layers": json.dumps(PROBE_LAYERS),
                    "n_options": len(question["options"]),
                    "option_letters": json.dumps(list(letters)),
                    "options": json.dumps(question["options"], ensure_ascii=False),
                    "response_distribution": serialize_distribution(response),
                    "question": question["question"],
                    "seed": args.seed,
                }
                # Store all layers in each row to keep one row per question-name pair.
                add_probe_readings(row, "after_name", name_reading)
                add_probe_readings(row, "final_prompt", final_reading)

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
    print(f"[ok] wrote {row_count} rows to {args.out.resolve()}")


if __name__ == "__main__":
    main()
