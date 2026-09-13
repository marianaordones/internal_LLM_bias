"""Utilities for evaluating demographic reading probes on named prompts."""

import csv
import re
from pathlib import Path

from src.modeling import json_array


# Canonical labels follow the class order used to train the TalkTuner probes.
READING_PROBE_SPECS = {
    "gender": {
        "checkpoint_names": ("gender",),
        "classes": ("male", "female"),
    },
    "age": {
        "checkpoint_names": ("age",),
        "classes": ("child", "adolescent", "adult", "older adult"),
    },
    "education": {
        "checkpoint_names": ("education",),
        "classes": ("someschool", "highschool", "collegemore"),
    },
    "socioeco": {
        "checkpoint_names": ("socioeconomic", "socioeco"),
        "classes": ("low", "middle", "high"),
    },
}


def load_names(path, min_dominance=0.95):
    """Load frequent names whose SSA records meet the requested sex dominance."""
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
            if row["sex_label"] not in {"male", "female"}:
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
    """Accept either a model checkpoint root or its reading_probe directory."""
    path = Path(path)
    for candidate in (path, path / "reading_probe"):
        if candidate.is_dir() and any(candidate.glob("*_probe_at_layer_*.pth")):
            return candidate
    raise FileNotFoundError(
        f"No reading probe checkpoints found in {path} or {path / 'reading_probe'}"
    )


def _attribute_checkpoints(probe_dir, attribute):
    spec = READING_PROBE_SPECS[attribute]
    pattern = re.compile(r"_probe_at_layer_(\d+)\.pth$")
    for checkpoint_name in spec["checkpoint_names"]:
        paths = {}
        for path in probe_dir.glob(f"{checkpoint_name}_probe_at_layer_*.pth"):
            match = pattern.search(path.name)
            if match:
                paths[int(match.group(1))] = path
        if paths:
            return checkpoint_name, paths
    expected = ", ".join(spec["checkpoint_names"])
    raise FileNotFoundError(
        f"No checkpoints for {attribute!r} ({expected}) in {probe_dir}"
    )


def load_reading_probes(probe_dir, device, expected_hidden_size=None):
    """Load all demographic probes and require a shared set of checkpoint layers."""
    import torch

    probe_dir = Path(probe_dir)
    discovered = {}
    common_layers = None
    for attribute in READING_PROBE_SPECS:
        checkpoint_name, paths = _attribute_checkpoints(probe_dir, attribute)
        discovered[attribute] = (checkpoint_name, paths)
        attribute_layers = set(paths)
        common_layers = (
            attribute_layers if common_layers is None
            else common_layers & attribute_layers
        )

    if not common_layers:
        raise ValueError("The demographic reading probes have no checkpoint layers in common.")
    layers = sorted(common_layers)
    probes = {}
    for attribute, (checkpoint_name, paths) in discovered.items():
        classes = READING_PROBE_SPECS[attribute]["classes"]
        parameters = {}
        for layer in layers:
            state = torch.load(paths[layer], map_location="cpu")
            weight = state["proj.0.weight"].to(device=device, dtype=torch.float32)
            bias = state["proj.0.bias"].to(device=device, dtype=torch.float32)
            if weight.shape[0] != len(classes) or tuple(bias.shape) != (len(classes),):
                raise ValueError(
                    f"Unexpected {attribute} probe shape at layer {layer}: "
                    f"weight={tuple(weight.shape)}, bias={tuple(bias.shape)}"
                )
            if expected_hidden_size is not None and weight.shape[1] != expected_hidden_size:
                raise ValueError(
                    f"{paths[layer].name} has hidden size {weight.shape[1]}; "
                    f"the model uses {expected_hidden_size}."
                )
            parameters[layer] = (weight, bias)
        probes[attribute] = {
            "checkpoint_name": checkpoint_name,
            "classes": classes,
            "parameters": parameters,
        }
    return probes, layers


def locate_name_end_token(tokenizer, prompt, name_sentence):
    """Return the token covering the final character of the name sentence."""
    return locate_cue_end_token(tokenizer, prompt, name_sentence)


def locate_cue_end_token(tokenizer, prompt, cue_sentence):
    """Return the token covering the final character of an identity cue."""
    if not getattr(tokenizer, "is_fast", False):
        raise TypeError(
            "A fast tokenizer is required to locate the cue sentence exactly. "
            "Load AutoTokenizer with use_fast=True."
        )
    start = prompt.find(cue_sentence)
    if start < 0:
        raise ValueError(f"Cue sentence not found in formatted prompt: {cue_sentence!r}")
    final_character = start + len(cue_sentence) - 1
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    for token_idx, (left, right) in enumerate(encoded["offset_mapping"]):
        if left <= final_character < right:
            return token_idx, encoded["input_ids"]
    raise ValueError("Could not map the end of the cue sentence to a prompt token.")


def read_all_probes(hidden_states, token_idx, probes, layers):
    """Evaluate every attribute at one token across all shared probe layers.

    ``activation_strength`` is chance-corrected top-class confidence. It is more
    comparable across attributes than raw sigmoid values when class counts differ.
    """
    import torch

    readings = {}
    for attribute, probe in probes.items():
        classes = probe["classes"]
        n_classes = len(classes)
        result = {
            "top_score": [],
            "normalized_confidence": [],
            "activation_strength": [],
            "margin": [],
            "prediction": [],
        }
        if attribute == "gender":
            result.update({
                "male_score": [],
                "female_score": [],
                "male_normalized": [],
                "female_normalized": [],
                "female_minus_male": [],
            })

        for layer in layers:
            if layer >= len(hidden_states):
                raise ValueError(
                    f"Probe layer {layer} is unavailable; model returned "
                    f"{len(hidden_states)} hidden-state tensors."
                )
            weight, bias = probe["parameters"][layer]
            activation = hidden_states[layer][0, token_idx].to(torch.float32)
            raw = torch.sigmoid(weight @ activation + bias)
            normalized = raw / raw.sum().clamp_min(1e-12)
            ordered = torch.sort(normalized, descending=True).values
            confidence = float(ordered[0].item())
            chance = 1.0 / n_classes
            strength = max(0.0, (confidence - chance) / (1.0 - chance))
            prediction_idx = int(torch.argmax(raw).item())

            result["top_score"].append(float(raw.max().item()))
            result["normalized_confidence"].append(confidence)
            result["activation_strength"].append(strength)
            result["margin"].append(float((ordered[0] - ordered[1]).item()))
            result["prediction"].append(classes[prediction_idx])

            if attribute == "gender":
                male = float(raw[0].item())
                female = float(raw[1].item())
                male_norm = float(normalized[0].item())
                female_norm = float(normalized[1].item())
                result["male_score"].append(male)
                result["female_score"].append(female)
                result["male_normalized"].append(male_norm)
                result["female_normalized"].append(female_norm)
                result["female_minus_male"].append(female_norm - male_norm)
        readings[attribute] = result
    return readings


def add_probe_readings(row, prefix, readings, layers, expected_gender=None):
    """Serialize generic readings plus legacy gender fields into one CSV row."""
    strengths = {}
    for attribute, result in readings.items():
        row[f"{prefix}_{attribute}_classes"] = json_array(
            list(READING_PROBE_SPECS[attribute]["classes"])
        )
        for field in (
            "top_score", "normalized_confidence", "activation_strength",
            "margin", "prediction",
        ):
            row[f"{prefix}_{attribute}_{field}_by_layer"] = json_array(result[field])
        strengths[attribute] = result["activation_strength"]

    # Preserve the original gender columns for the existing analysis notebook.
    gender = readings["gender"]
    for field in (
        "male_score", "female_score", "male_normalized", "female_normalized",
        "female_minus_male", "prediction",
    ):
        row[f"{prefix}_{field}_by_layer"] = json_array(gender[field])

    other_attributes = [name for name in readings if name != "gender"]
    advantage = [
        strengths["gender"][index]
        - max(strengths[name][index] for name in other_attributes)
        for index in range(len(layers))
    ]
    row[f"{prefix}_gender_minus_max_other_activation_by_layer"] = json_array(advantage)
    row[f"{prefix}_gender_mean_activation_strength"] = sum(
        strengths["gender"]
    ) / len(layers)
    row[f"{prefix}_gender_mean_advantage_over_max_other"] = sum(advantage) / len(
        layers
    )

    if expected_gender is not None:
        matches = [
            prediction == expected_gender
            for prediction in gender["prediction"]
        ]
        row[f"{prefix}_gender_matches_name_by_layer"] = json_array(matches)
        row[f"{prefix}_gender_match_rate"] = sum(matches) / len(matches)


def add_target_probe_summary(
    row, prefix, readings, layers, target_attribute, expected_class=None
):
    """Add target-vs-placebo activation and optional class-match summaries."""
    if target_attribute not in readings:
        raise KeyError(f"Unknown target probe attribute: {target_attribute}")
    target = readings[target_attribute]
    other_attributes = [name for name in readings if name != target_attribute]
    advantage = [
        target["activation_strength"][index]
        - max(readings[name]["activation_strength"][index] for name in other_attributes)
        for index in range(len(layers))
    ]
    row[f"{prefix}_{target_attribute}_minus_max_other_activation_by_layer"] = (
        json_array(advantage)
    )
    row[f"{prefix}_{target_attribute}_mean_activation_strength"] = sum(
        target["activation_strength"]
    ) / len(layers)
    row[f"{prefix}_{target_attribute}_mean_advantage_over_max_other"] = sum(
        advantage
    ) / len(layers)

    if expected_class is not None:
        matches = [prediction == expected_class for prediction in target["prediction"]]
        row[f"{prefix}_{target_attribute}_matches_expected_by_layer"] = json_array(
            matches
        )
        row[f"{prefix}_{target_attribute}_match_rate"] = sum(matches) / len(matches)


def make_dry_readings(label, layers):
    """Create deterministic placeholder readings for pipeline validation."""
    return make_targeted_dry_readings("gender", label, layers)


def make_targeted_dry_readings(target_attribute, expected_class, layers):
    """Create deterministic placeholder readings for any target attribute."""
    readings = {}
    for attribute, spec in READING_PROBE_SPECS.items():
        classes = spec["classes"]
        if attribute == target_attribute and expected_class in classes:
            prediction = expected_class
            confidence = 0.75
            strength = 0.5
        else:
            prediction = classes[0]
            confidence = 1.0 / len(classes)
            strength = 0.0
        result = {
            "top_score": [confidence] * len(layers),
            "normalized_confidence": [confidence] * len(layers),
            "activation_strength": [strength] * len(layers),
            "margin": [strength] * len(layers),
            "prediction": [prediction] * len(layers),
        }
        if attribute == "gender":
            male, female = (
                (0.75, 0.25) if prediction == "male" else (0.25, 0.75)
            )
            result.update({
                "male_score": [male] * len(layers),
                "female_score": [female] * len(layers),
                "male_normalized": [male] * len(layers),
                "female_normalized": [female] * len(layers),
                "female_minus_male": [female - male] * len(layers),
            })
        readings[attribute] = result
    return readings
