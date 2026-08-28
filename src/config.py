"""Project paths and demographic definitions shared by all experiments."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
# Defaults stay anchored to the repository, regardless of the current directory.
DEFAULT_QKEY_DICT = REPO_ROOT / "data/political_questions/OpinionQA/refined_qkey_dict.json"
DEFAULT_CONTROLLING_PROBE_DIR = REPO_ROOT / "data/probe_checkpoints/controlling_probe"
DEFAULT_READING_PROBE_DIR = REPO_ROOT / "data/probe_checkpoints/reading_probe"
DEFAULT_NAMES = REPO_ROOT / "data/ssa_gender_names_1989_2008.csv"
DEFAULT_MODEL = "meta-llama/Llama-2-13b-chat-hf"

# Class order must match the output order of the trained TalkTuner probes.
ATTRIBUTE_CONFIG = {
    "gender": {
        "probe_name": "gender",
        "classes": ["male", "female"],
        "declarations": ["I am a man.", "I am a woman."],
        "talktuner_default_magnitude": 7,
    },
    "age": {
        "probe_name": "age",
        "classes": ["child", "adolescent", "adult", "older_adult"],
        "declarations": [
            "I am a child.",
            "I am an adolescent.",
            "I am an adult.",
            "I am an older adult.",
        ],
        "talktuner_default_magnitude": 7,
    },
    "education": {
        "probe_name": "education",
        "classes": ["some schooling", "high school", "college and more"],
        "declarations": [
            "I did not complete high school.",
            "I have a high school education.",
            "I have a college education or higher.",
        ],
        "talktuner_default_magnitude": 7,
    },
    "socioeco": {
        "probe_name": "socioeco",
        "classes": ["low", "mid", "high"],
        "declarations": [
            "I have low socioeconomic status.",
            "I have middle socioeconomic status.",
            "I have high socioeconomic status.",
        ],
        "talktuner_default_magnitude": 8,
    },
}

ATTRIBUTE_ALIASES = {
    # Aliases affect CLI parsing only; CSV outputs always use canonical names.
    "gender": "gender",
    "sex": "gender",
    "age": "age",
    "education": "education",
    "edu": "education",
    "socioeco": "socioeco",
    "socioeconomic": "socioeco",
    "socioeconomic_status": "socioeco",
}

PROBE_LAYERS = list(range(41))
# Reading-probe output indices follow this fixed class order.
PROBE_CLASSES = ["male", "female"]
