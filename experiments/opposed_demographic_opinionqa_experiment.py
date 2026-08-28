#!/usr/bin/env python
"""Run declared-demographic prompts with steering toward the opposite class."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

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
    OPPOSED_CONFIGS,
    make_opposed_row,
    parse_magnitudes,
    validate_opposed_configs,
)
from src.modeling import llama_v2_prompt, option_token_ids, set_seed  # noqa: E402
from src.opinionqa import build_user_message, load_questions  # noqa: E402
from src.steering import (  # noqa: E402
    load_steer_vectors,
    resolve_controlling_probe_dir,
    steered_distribution,
)


DEFAULT_MAGNITUDES = "1,3,5,7,9,13"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--magnitudes", default=DEFAULT_MAGNITUDES)
    parser.add_argument("--qkey-dict", type=Path, default=DEFAULT_QKEY_DICT)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_CONTROLLING_PROBE_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--from-idx", type=int, default=19)
    parser.add_argument("--to-idx", type=int, default=29)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="0 means all questions")
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "opposed_demographic_opinionqa_results.csv",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate rows and prompts using uniform distributions without loading the model",
    )
    args = parser.parse_args()

    if args.from_idx < 0 or args.to_idx <= args.from_idx:
        parser.error("Require 0 <= from-idx < to-idx.")
    if args.limit < 0:
        parser.error("--limit must be zero or positive.")

    validate_opposed_configs()
    magnitudes = parse_magnitudes(args.magnitudes, require_positive=True)
    questions = load_questions(args.qkey_dict)
    if args.limit:
        questions = questions[:args.limit]
    set_seed(args.seed, use_torch=not args.dry_run)

    probe_dir = resolve_controlling_probe_dir(args.probe_dir)
    model = tokenizer = None
    vectors = {}
    letter_ids = {}
    if not args.dry_run:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype
        ).to(args.device).eval()
        letter_ids = option_token_ids(tokenizer)
        for attribute in ATTRIBUTE_CONFIG:
            vectors[attribute] = load_steer_vectors(
                probe_dir, attribute, args.from_idx, args.to_idx, args.device
            )

    print(
        f"[info] questions={len(questions)} configurations={len(OPPOSED_CONFIGS)} "
        f"magnitudes={magnitudes} seed={args.seed} dry_run={args.dry_run}"
    )
    print(f"[info] probe_dir={probe_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with open(args.out, "w", newline="", encoding="utf-8") as stream:
        writer = None
        for question_idx, question in enumerate(questions, start=1):
            for config in OPPOSED_CONFIGS:
                attribute = config["attribute"]
                classes = ATTRIBUTE_CONFIG[attribute]["classes"]
                # The declared class affects the prompt; the opposite class selects the vector.
                steer_idx = classes.index(config["steer_class"])
                message, letters = build_user_message(
                    question["question"], question["options"], config["declaration"]
                )

                input_ids = candidates = None
                if not args.dry_run:
                    prompt = llama_v2_prompt([{"role": "user", "content": message}])
                    input_ids = tokenizer(
                        prompt, return_tensors="pt", add_special_tokens=False
                    ).input_ids.to(args.device)
                    candidates = [letter_ids[letter] for letter in letters]

                for magnitude in magnitudes:
                    # Every magnitude reuses the same tokenized declared-profile prompt.
                    if args.dry_run:
                        distribution = np.full(len(letters), 1.0 / len(letters))
                    else:
                        distribution = steered_distribution(
                            model, input_ids, candidates, vectors[attribute],
                            steer_idx, magnitude
                        )
                    row = make_opposed_row(
                        question, config, magnitude, distribution, args.seed
                    )
                    row["from_idx"] = args.from_idx
                    row["to_idx_exclusive"] = args.to_idx
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
