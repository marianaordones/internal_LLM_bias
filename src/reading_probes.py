"""Utilities for evaluating TalkTuner gender reading probes."""

import csv
from pathlib import Path

from src.config import PROBE_CLASSES, PROBE_LAYERS
from src.modeling import json_array


def load_names(path, min_dominance=0.95):
    names = []
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "name", "sex_label", "dominance", "aggregate_share",
            "source_start_year", "source_end_year",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Names CSV is missing columns: {sorted(missing)}")
        for row in reader:
            dominance = float(row["dominance"])
            if dominance < min_dominance:
                continue
            if row["sex_label"] not in PROBE_CLASSES:
                raise ValueError(f"Invalid sex_label for {row['name']}: {row['sex_label']}")
            names.append({
                "name": row["name"],
                "sex_label": row["sex_label"],
                "dominance": dominance,
                "aggregate_share": float(row["aggregate_share"]),
                "source_start_year": int(row["source_start_year"]),
                "source_end_year": int(row["source_end_year"]),
            })
    if not names:
        raise ValueError("No names passed the dominance threshold.")
    return names


def resolve_reading_probe_dir(path):
    path = Path(path)
    for candidate in (path, path / "reading_probe"):
        if candidate.is_dir() and all(
            (candidate / f"gender_probe_at_layer_{layer}.pth").is_file()
            for layer in PROBE_LAYERS
        ):
            return candidate
    raise FileNotFoundError(
        f"Could not find all gender reading probes 0..40 in {path} "
        f"or {path / 'reading_probe'}"
    )


def load_gender_reading_probes(probe_dir, device):
    """Load sigmoid-linear gender probes for checkpoints 0 through 40."""
    import torch

    probes = []
    for layer in PROBE_LAYERS:
        path = probe_dir / f"gender_probe_at_layer_{layer}.pth"
        state = torch.load(path, map_location="cpu")
        weight = state["proj.0.weight"].to(device=device, dtype=torch.float32)
        bias = state["proj.0.bias"].to(device=device, dtype=torch.float32)
        if tuple(weight.shape) != (2, 5120) or tuple(bias.shape) != (2,):
            raise ValueError(
                f"Unexpected gender probe shape at layer {layer}: "
                f"weight={tuple(weight.shape)}, bias={tuple(bias.shape)}"
            )
        probes.append((weight, bias))
    return probes


def locate_name_end_token(tokenizer, prompt, name_sentence):
    """Return the token covering the final character of the name sentence."""
    if not getattr(tokenizer, "is_fast", False):
        raise TypeError(
            "A fast tokenizer is required to locate the name sentence exactly. "
            "Load AutoTokenizer with use_fast=True."
        )
    start = prompt.find(name_sentence)
    if start < 0:
        raise ValueError(f"Name sentence not found in formatted prompt: {name_sentence!r}")
    final_character = start + len(name_sentence) - 1
    # Offset mappings avoid guessing how a particular name is split into tokens.
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    for token_idx, (left, right) in enumerate(encoded["offset_mapping"]):
        if left <= final_character < right:
            return token_idx, encoded["input_ids"]
    raise ValueError("Could not map the end of the name sentence to a prompt token.")


def read_all_probes(hidden_states, token_idx, probes):
    """Return raw and normalized probe scores, margins, and predictions."""
    import torch

    readings = {
        "male_score": [],
        "female_score": [],
        "male_normalized": [],
        "female_normalized": [],
        "female_minus_male": [],
        "prediction": [],
    }
    for layer, (weight, bias) in enumerate(probes):
        # hidden_states[0] is the embedding output, matching probe checkpoint 0.
        activation = hidden_states[layer][0, token_idx].to(torch.float32)
        raw = torch.sigmoid(weight @ activation + bias)
        normalized = raw / raw.sum().clamp_min(1e-12)
        male = float(raw[0].item())
        female = float(raw[1].item())
        male_norm = float(normalized[0].item())
        female_norm = float(normalized[1].item())
        readings["male_score"].append(male)
        readings["female_score"].append(female)
        readings["male_normalized"].append(male_norm)
        readings["female_normalized"].append(female_norm)
        readings["female_minus_male"].append(female_norm - male_norm)
        readings["prediction"].append(PROBE_CLASSES[int(torch.argmax(raw).item())])
    return readings


def add_probe_readings(row, prefix, readings):
    for field, values in readings.items():
        row[f"{prefix}_{field}_by_layer"] = json_array(values)


def make_dry_reading(label):
    male, female = (0.75, 0.25) if label == "male" else (0.25, 0.75)
    return {
        "male_score": [male] * len(PROBE_LAYERS),
        "female_score": [female] * len(PROBE_LAYERS),
        "male_normalized": [male] * len(PROBE_LAYERS),
        "female_normalized": [female] * len(PROBE_LAYERS),
        "female_minus_male": [female - male] * len(PROBE_LAYERS),
        "prediction": [label] * len(PROBE_LAYERS),
    }
