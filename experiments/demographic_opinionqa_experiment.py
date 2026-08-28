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
from src.modeling import llama_v2_prompt, option_token_ids, set_seed  # noqa: E402
from src.opinionqa import build_user_message, load_questions  # noqa: E402
from src.steering import (  # noqa: E402
    load_steer_vectors,
    option_distribution,
    resolve_controlling_probe_dir,
    steered_distribution,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attributes", default="all", help="Comma-separated: gender, age, education, socioeco, or all")
    parser.add_argument("--channels", choices=["declared", "steered", "both"], default="both")
    parser.add_argument("--magnitudes", default="0,1,3,5,7,8,9,13", help="Comma-separated steering magnitudes")
    parser.add_argument("--qkey-dict", type=Path, default=DEFAULT_QKEY_DICT)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_CONTROLLING_PROBE_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--from-idx", type=int, default=19, help="First decoder layer to steer")
    parser.add_argument("--to-idx", type=int, default=29, help="Exclusive last decoder layer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="0 means all questions")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "demographic_opinionqa_results.csv")
    parser.add_argument("--dry-run", action="store_true", help="Validate the pipeline using uniform distributions")
    args = parser.parse_args()

    attributes = parse_attributes(args.attributes)
    magnitudes = parse_magnitudes(args.magnitudes)
    run_declared = args.channels in {"declared", "both"}
    run_steered = args.channels in {"steered", "both"}
    if args.from_idx < 0 or args.to_idx <= args.from_idx:
        parser.error("Require 0 <= from-idx < to-idx.")
    if args.limit < 0:
        parser.error("--limit must be zero or positive.")

    questions = load_questions(args.qkey_dict)
    if args.limit:
        questions = questions[:args.limit]
    set_seed(args.seed, use_torch=not args.dry_run)

    model = tokenizer = probe_dir = None
    steer_vectors = {}
    letter_ids = {}
    if run_steered:
        probe_dir = resolve_controlling_probe_dir(args.probe_dir)

    if not args.dry_run:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype
        ).to(args.device).eval()
        letter_ids = option_token_ids(tokenizer)
        if run_steered:
            for attribute in attributes:
                steer_vectors[attribute] = load_steer_vectors(
                    probe_dir, attribute, args.from_idx, args.to_idx, args.device
                )

    print(
        f"[info] questions={len(questions)} attributes={attributes} channels={args.channels} "
        f"magnitudes={magnitudes if run_steered else 'n/a'} seed={args.seed} dry_run={args.dry_run}"
    )
    if probe_dir:
        print(f"[info] probe_dir={probe_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    # Neutral responses are identical across attributes and classes for a question.
    neutral_cache = {}
    with open(args.out, "w", newline="", encoding="utf-8") as stream:
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
                            prompt = llama_v2_prompt([{"role": "user", "content": message}])
                            input_ids = tokenizer(
                                prompt, return_tensors="pt", add_special_tokens=False
                            ).input_ids.to(args.device)
                            distribution = option_distribution(
                                model, input_ids, [letter_ids[letter] for letter in letters]
                            )
                        row = make_demographic_row(
                            question, attribute, "declared", class_idx, None,
                            declaration, distribution, args.seed
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
                            prompt = llama_v2_prompt([{"role": "user", "content": message}])
                            input_ids = tokenizer(
                                prompt, return_tensors="pt", add_special_tokens=False
                            ).input_ids.to(args.device)
                            candidates = [letter_ids[letter] for letter in letters]
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
    print(f"[ok] wrote {row_count} rows to {args.out.resolve()}")


if __name__ == "__main__":
    main()
