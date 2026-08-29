#!/usr/bin/env python3
"""Extract activations and train TalkTuner-style probes for another chat model."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # noqa: E402

from src.probe_training import (  # noqa: E402
    ATTRIBUTE_SPECS,
    extract_activations,
    load_causal_model,
    load_examples,
    set_seed,
    train_probes_from_cache,
)


def comma_list(value: str, choices: set[str]) -> list[str]:
    values = list(choices) if value == "all" else [item.strip() for item in value.split(",")]
    invalid = set(values) - choices
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid values: {sorted(invalid)}")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--stage", choices=("all", "extract", "train"), default="all")
    parser.add_argument("--attributes", default="all")
    parser.add_argument("--channels", default="reading,controlling")
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=REPO_ROOT / "data" / "probe_training_datasets",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=REPO_ROOT / "data" / "probe_activation_cache" / "qwen2.5-7b-instruct",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "data" / "probe_checkpoints" / "qwen2.5-7b-instruct",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--extraction-batch-size", type=int, default=1)
    parser.add_argument("--training-batch-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--limit", type=int, help="Use only this many examples per attribute (smoke tests).")
    return parser


def stratified_limit(examples, limit: int | None, seed: int):
    """Create a representative smoke-test subset without dropping rare classes."""
    if limit is None or limit >= len(examples):
        return examples
    by_class = {}
    for example in examples:
        by_class.setdefault(example.class_name, []).append(example)
    if limit < len(by_class):
        raise ValueError(f"--limit must be at least {len(by_class)} for this attribute")
    rng = random.Random(seed)
    for group in by_class.values():
        rng.shuffle(group)
    selected = []
    groups = list(by_class.values())
    while len(selected) < limit:
        made_progress = False
        for group in groups:
            if group and len(selected) < limit:
                selected.append(group.pop())
                made_progress = True
        if not made_progress:
            break
    rng.shuffle(selected)
    return selected


def main() -> None:
    args = build_parser().parse_args()
    attributes = comma_list(args.attributes, set(ATTRIBUTE_SPECS))
    channels = comma_list(args.channels, {"reading", "controlling"})
    set_seed(args.seed)

    model = tokenizer = None
    if args.stage in {"all", "extract"}:
        tokenizer, model = load_causal_model(args.model, args.device, args.dtype)
        print(
            f"[model] {args.model} hidden={model.config.hidden_size} "
            f"decoder_layers={model.config.num_hidden_layers}",
            flush=True,
        )

    for attribute in attributes:
        examples, archive_stats = load_examples(args.dataset_dir, attribute)
        examples = stratified_limit(examples, args.limit, args.seed)
        counts = {name: sum(ex.class_name == name for ex in examples) for name in ATTRIBUTE_SPECS[attribute]["classes"]}
        print(f"[data] {attribute}: {len(examples)} examples {counts}", flush=True)

        for channel in channels:
            cache_path = args.cache_dir / f"{attribute}_{channel}.pt"
            if args.stage in {"all", "extract"}:
                extract_activations(
                    model=model,
                    tokenizer=tokenizer,
                    examples=examples,
                    attribute=attribute,
                    channel=channel,
                    device=args.device,
                    batch_size=args.extraction_batch_size,
                    max_length=args.max_length,
                    output_path=cache_path,
                    model_name=args.model,
                    archive_stats=archive_stats,
                )
            if args.stage in {"all", "train"}:
                if not cache_path.is_file():
                    raise FileNotFoundError(f"Activation cache not found: {cache_path}")
                train_probes_from_cache(
                    cache_path=cache_path,
                    output_dir=args.output_dir / f"{channel}_probe",
                    device=args.device,
                    epochs=args.epochs,
                    batch_size=args.training_batch_size,
                    learning_rate=args.learning_rate,
                    validation_fraction=args.validation_fraction,
                    seed=args.seed,
                )


if __name__ == "__main__":
    main()
