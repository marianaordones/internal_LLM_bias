#!/usr/bin/env python
"""Test socioeconomic inference from occupation cues on OpinionQA.

Each forward pass records the answer distribution and readings from all four
demographic probes immediately after the occupation cue and at the final prompt
token. Occupations are selected using precomputed BLS wage, education, and sex
composition data.
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
from src.opinionqa import build_cued_user_message, load_questions  # noqa: E402
from src.reading_probes import (  # noqa: E402
    add_probe_readings,
    add_target_probe_summary,
    load_reading_probes,
    locate_cue_end_token,
    make_targeted_dry_readings,
    read_all_probes,
    resolve_reading_probe_dir,
)


DEFAULT_OCCUPATIONS = REPO_ROOT / "data/occupation_income_selection_candidates.csv"

MODEL_PROFILES = {
    "llama": {
        "model_id": DEFAULT_MODEL,
        "probe_dir": DEFAULT_READING_PROBE_DIR,
        "dtype": "float16",
        "num_hidden_layers": 40,
    },
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "probe_dir": REPO_ROOT / "data/probe_checkpoints/qwen2.5-7b-instruct/reading_probe",
        "dtype": "bfloat16",
        "num_hidden_layers": 28,
    },
    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "probe_dir": REPO_ROOT / "data/probe_checkpoints/mistral-7b-instruct-v0.3/reading_probe",
        "dtype": "bfloat16",
        "num_hidden_layers": 32,
    },
}


def load_occupations(path, selection):
    """Load BLS-derived cues and apply the preregistered extreme-group rule."""
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "occupation_cue", "occupation_title", "soc_code",
        "median_annual_wage_usd", "typical_entry_education", "percent_women",
    }
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Occupation CSV is missing columns: {sorted(missing)}")

    occupations = []
    for row in rows:
        if not row["occupation_cue"].strip():
            raise ValueError(f"Missing prompt cue for occupation {row['occupation_title']!r}")
        item = dict(row)
        item["median_annual_wage_usd"] = float(row["median_annual_wage_usd"])
        item["percent_women"] = float(row["percent_women"])
        occupations.append(item)
    occupations.sort(key=lambda item: (item["median_annual_wage_usd"], item["soc_code"]))

    if selection == "all":
        low_cutoff = occupations[9]["median_annual_wage_usd"]
        high_cutoff = occupations[-10]["median_annual_wage_usd"]
        for item in occupations:
            wage = item["median_annual_wage_usd"]
            item["selection_group"] = (
                "low" if wage <= low_cutoff else "high" if wage >= high_cutoff else "middle"
            )
        return occupations

    per_group = int(selection.removeprefix("extreme"))
    if len(occupations) < 2 * per_group:
        raise ValueError(
            f"Need at least {2 * per_group} eligible occupations; found {len(occupations)}."
        )
    low = occupations[:per_group]
    high = occupations[-per_group:]
    for item in low:
        item["selection_group"] = "low"
    for item in high:
        item["selection_group"] = "high"
    return low + high


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
    parser.add_argument("--occupations", type=Path, default=DEFAULT_OCCUPATIONS)
    parser.add_argument(
        "--selection", choices=("extreme5", "extreme10", "all"), default="extreme5",
        help="Primary 5x5 contrast, 10x10 sensitivity analysis, or all eligible cues",
    )
    parser.add_argument("--qkey-dict", type=Path, default=DEFAULT_QKEY_DICT)
    parser.add_argument("--reading-probe-dir", type=Path)
    parser.add_argument("--model", help="Override the profile's model ID or local path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Question limit; 0 means all")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be zero or positive.")

    profile = MODEL_PROFILES[args.model_profile]
    model_id = args.model or profile["model_id"]
    probe_dir_arg = args.reading_probe_dir or profile["probe_dir"]
    output_path = args.out or (
        REPO_ROOT / f"results/inferred_income_occupations_{args.model_profile}_{args.selection}.csv"
    )

    questions = load_questions(args.qkey_dict)
    if args.limit:
        questions = questions[:args.limit]
    occupations = load_occupations(args.occupations, args.selection)
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
        dtype = getattr(torch, profile["dtype"]) if args.device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(args.device).eval()
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
        f"[info] questions={len(questions)} occupations={len(occupations)} "
        f"selection={args.selection} model={args.model_profile}:{model_id} "
        f"layers={probe_layers[0]}..{probe_layers[-1]} "
        f"forwards={len(questions) * len(occupations)} seed={args.seed} "
        f"dry_run={args.dry_run}"
    )
    print(f"[info] reading_probe_dir={probe_dir or probe_dir_arg}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as stream:
        writer = None
        for question_idx, question in enumerate(questions, start=1):
            for occupation in occupations:
                cue_sentence = f"I work as a {occupation['occupation_cue']}."
                message, letters = build_cued_user_message(
                    cue_sentence, question["question"], question["options"]
                )
                expected_class = (
                    occupation["selection_group"]
                    if occupation["selection_group"] in {"low", "high"}
                    else None
                )

                if args.dry_run:
                    cue_token_idx = final_token_idx = -1
                    response = np.full(len(letters), 1.0 / len(letters))
                    cue_reading = make_targeted_dry_readings(
                        "socioeco", expected_class, probe_layers
                    )
                    final_reading = make_targeted_dry_readings(
                        "socioeco", expected_class, probe_layers
                    )
                else:
                    import torch

                    prompt = format_prompt(tokenizer, args.model_profile, message)
                    cue_token_idx, full_ids = locate_cue_end_token(
                        tokenizer, prompt, cue_sentence
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
                    cue_reading = read_all_probes(
                        model_output.hidden_states, cue_token_idx, probes, probe_layers
                    )
                    final_reading = read_all_probes(
                        model_output.hidden_states, final_token_idx, probes, probe_layers
                    )

                row = {
                    "qkey": question["qkey"],
                    "occupation_cue": occupation["occupation_cue"],
                    "occupation_title": occupation["occupation_title"],
                    "soc_code": occupation["soc_code"],
                    "income_group": occupation["selection_group"],
                    "median_annual_wage_usd": occupation["median_annual_wage_usd"],
                    "typical_entry_education": occupation["typical_entry_education"],
                    "related_work_experience": occupation.get("related_work_experience", ""),
                    "percent_women": occupation["percent_women"],
                    "occupation_sentence": cue_sentence,
                    "occupation_end_token_idx": cue_token_idx,
                    "final_token_idx": final_token_idx,
                    "selection": args.selection,
                    "probe_layers": json.dumps(probe_layers),
                    "probe_attributes": json.dumps(["gender", "age", "education", "socioeco"]),
                    "n_options": len(question["options"]),
                    "option_letters": json.dumps(list(letters)),
                    "options": json.dumps(question["options"], ensure_ascii=False),
                    "response_distribution": serialize_distribution(response),
                    "question": question["question"],
                    "model_profile": args.model_profile,
                    "model_id": model_id,
                    "seed": args.seed,
                }
                add_probe_readings(row, "after_occupation", cue_reading, probe_layers)
                add_target_probe_summary(
                    row, "after_occupation", cue_reading, probe_layers,
                    "socioeco", expected_class,
                )
                add_probe_readings(row, "final_prompt", final_reading, probe_layers)
                add_target_probe_summary(
                    row, "final_prompt", final_reading, probe_layers,
                    "socioeco", expected_class,
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
