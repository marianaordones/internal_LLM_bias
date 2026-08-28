"""Validation and row-building helpers for experiment entry points."""

import json
import string

from src.config import ATTRIBUTE_ALIASES, ATTRIBUTE_CONFIG
from src.modeling import serialize_distribution


OPPOSED_CONFIGS = [
    {"attribute": "gender", "declared_class": "male", "declaration": "I am a man.", "steer_class": "female"},
    {"attribute": "gender", "declared_class": "female", "declaration": "I am a woman.", "steer_class": "male"},
    {"attribute": "age", "declared_class": "adult", "declaration": "I am an adult.", "steer_class": "older_adult"},
    {"attribute": "age", "declared_class": "older_adult", "declaration": "I am an older adult.", "steer_class": "adult"},
    {"attribute": "education", "declared_class": "some schooling", "declaration": "I did not complete high school.", "steer_class": "college and more"},
    {"attribute": "education", "declared_class": "college and more", "declaration": "I have a college education or higher.", "steer_class": "some schooling"},
    {"attribute": "socioeco", "declared_class": "low", "declaration": "I have low socioeconomic status.", "steer_class": "high"},
    {"attribute": "socioeco", "declared_class": "high", "declaration": "I have high socioeconomic status.", "steer_class": "low"},
]


def parse_attributes(raw):
    requested = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if requested == ["all"]:
        return list(ATTRIBUTE_CONFIG)
    parsed = []
    for item in requested:
        if item not in ATTRIBUTE_ALIASES:
            valid = ", ".join(ATTRIBUTE_CONFIG)
            raise ValueError(f"Unknown attribute {item!r}. Valid attributes: {valid}, all")
        canonical = ATTRIBUTE_ALIASES[item]
        if canonical not in parsed:
            parsed.append(canonical)
    if not parsed:
        raise ValueError("Select at least one attribute.")
    return parsed


def parse_magnitudes(raw, require_positive=False):
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Select at least one steering magnitude.")
    if require_positive and any(value <= 0 for value in values):
        raise ValueError("Opposed steering magnitudes must be greater than zero.")
    return values


def validate_opposed_configs():
    for config in OPPOSED_CONFIGS:
        classes = ATTRIBUTE_CONFIG[config["attribute"]]["classes"]
        for key in ("declared_class", "steer_class"):
            if config[key] not in classes:
                raise ValueError(
                    f"{config[key]!r} is not a TalkTuner class for "
                    f"{config['attribute']}: {classes}"
                )
        if config["declared_class"] == config["steer_class"]:
            raise ValueError(f"Configuration is not opposed: {config}")


def make_demographic_row(
    question, attribute, channel, class_idx, magnitude, declaration, distribution, seed
):
    config = ATTRIBUTE_CONFIG[attribute]
    # Preserve option order because Wasserstein analyses use the same ordering.
    letters = list(string.ascii_uppercase[:len(question["options"])])
    return {
        "qkey": question["qkey"],
        "attribute": attribute,
        "channel": channel,
        "class_idx": class_idx,
        "class_label": config["classes"][class_idx],
        "magnitude": "" if magnitude is None else magnitude,
        "talktuner_default_magnitude": config["talktuner_default_magnitude"],
        "declaration": declaration or "",
        "n_options": len(question["options"]),
        "option_letters": json.dumps(letters),
        "options": json.dumps(question["options"], ensure_ascii=False),
        "response_distribution": serialize_distribution(distribution),
        "question": question["question"],
        "seed": seed,
    }


def make_opposed_row(question, config, magnitude, distribution, seed):
    attribute = config["attribute"]
    classes = ATTRIBUTE_CONFIG[attribute]["classes"]
    declared_idx = classes.index(config["declared_class"])
    steer_idx = classes.index(config["steer_class"])
    letters = list(string.ascii_uppercase[:len(question["options"])])
    return {
        "qkey": question["qkey"],
        "attribute": attribute,
        "configuration": f"declare_{config['declared_class']}__steer_{config['steer_class']}",
        "declared_class_idx": declared_idx,
        "declared_class": config["declared_class"],
        "declaration": config["declaration"],
        "steer_class_idx": steer_idx,
        "steer_class": config["steer_class"],
        "magnitude": magnitude,
        "from_idx": None,
        "to_idx_exclusive": None,
        "seed": seed,
        "n_options": len(question["options"]),
        "option_letters": json.dumps(letters),
        "options": json.dumps(question["options"], ensure_ascii=False),
        "response_distribution": serialize_distribution(distribution),
        "question": question["question"],
    }
