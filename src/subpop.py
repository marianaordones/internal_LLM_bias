"""Loading and demographic mappings for the gated SubPOP dataset."""

from __future__ import annotations

import json
import re
from pathlib import Path


# Probe indices follow the class order used by the TalkTuner training data.
SUBPOP_ATTRIBUTE_CONFIG = {
    "sex": {
        "dataset_attribute": "SEX",
        "probe_attribute": "gender",
        "llama_probe_name": "gender",
        "qwen_probe_name": "gender",
        "classes": [
            {
                "label": "male",
                "probe_class_idx": 0,
                "group": "Male",
                "declaration": "I am a man.",
            },
            {
                "label": "female",
                "probe_class_idx": 1,
                "group": "Female",
                "declaration": "I am a woman.",
            },
        ],
    },
    "education": {
        "dataset_attribute": "EDUCATION",
        "probe_attribute": "education",
        "llama_probe_name": "education",
        "qwen_probe_name": "education",
        "classes": [
            {
                "label": "some schooling",
                "probe_class_idx": 0,
                "group": "Less than high school",
                "declaration": "I did not complete high school.",
            },
            {
                "label": "college and more",
                "probe_class_idx": 2,
                "group": "College graduate/some postgrad",
                "declaration": "I have a college education or higher.",
            },
        ],
    },
    "income": {
        "dataset_attribute": "INCOME",
        "probe_attribute": "socioeco",
        "llama_probe_name": "socioeco",
        "qwen_probe_name": "socioeconomic",
        "classes": [
            {
                "label": "low",
                "probe_class_idx": 0,
                "group": "Less than $30,000",
                "declaration": "I have low socioeconomic status.",
            },
            {
                "label": "high",
                "probe_class_idx": 2,
                "group": "$100,000 or more",
                "declaration": "I have high socioeconomic status.",
            },
        ],
    },
}


def parse_subpop_attributes(raw: str) -> list[str]:
    aliases = {
        "sex": "sex", "gender": "sex",
        "education": "education", "edu": "education",
        "income": "income", "socioeco": "income", "socioeconomic": "income",
    }
    requested = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if requested == ["all"]:
        return list(SUBPOP_ATTRIBUTE_CONFIG)
    parsed = []
    for value in requested:
        if value not in aliases:
            raise ValueError(f"Unknown SubPOP attribute {value!r}; use sex, education, income, or all.")
        canonical = aliases[value]
        if canonical not in parsed:
            parsed.append(canonical)
    if not parsed:
        raise ValueError("Select at least one attribute.")
    return parsed


def _read_jsonl(path: Path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


# Refusal / "don't know" / "no answer" markers. These are used ONLY to sanity
# check the trailing options that positional alignment against `responses`
# drops -- they never decide the alignment itself (see load_subpop_questions).
_REFUSAL_SUBSTRINGS = (
    "refus",        # refused / refuse
    "don't know",
    "dont know",
    "do not know",
    "no answer",
    "no response",
    "not sure",
    "no opinion",
    "can't choose",
    "cant choose",
    "cannot choose",
    "skipped",
    "declined",
)
_REFUSAL_TOKEN_SETS = (
    {"dk"}, {"na"}, {"nan"}, {"ref"},
    {"dk", "na"}, {"dk", "ref"}, {"dk", "no"}, {"dont", "know"},
)


def _is_refusal_option(option) -> bool:
    """Heuristic: does this option read as a Refused / Don't-know / No-answer
    choice? Used ONLY to sanity-check the trailing options that positional
    alignment against `responses` drops -- never to decide the alignment.
    """
    text = " ".join(str(option).strip().lower().split())
    if not text:
        return True
    if any(sub in text for sub in _REFUSAL_SUBSTRINGS):
        return True
    tokens = set(re.sub(r"[^a-z0-9]+", " ", text).split())
    return any(tokens == marker for marker in _REFUSAL_TOKEN_SETS)


def _load_rows(dataset_id: str, split: str, dataset_file: Path | None):
    if dataset_file is not None:
        return _read_jsonl(dataset_file)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Loading SubPOP from Hugging Face requires `pip install datasets`."
        ) from exc
    if split != "all":
        return load_dataset(dataset_id, split=split)

    # Preserve the source split while combining every file exposed by the Hub.
    dataset_dict = load_dataset(dataset_id)
    rows = []
    for split_name, dataset in dataset_dict.items():
        for row in dataset:
            tagged = dict(row)
            tagged["_subpop_source_split"] = split_name
            rows.append(tagged)
    return rows


def load_subpop_questions(
    *,
    dataset_id: str,
    split: str,
    attributes: list[str],
    dataset_file: Path | None = None,
) -> list[dict]:
    """Return unique questions with complete human data for selected contrasts."""
    rows = _load_rows(dataset_id, split, dataset_file)
    required = {
        (config["dataset_attribute"], item["group"])
        for attribute in attributes
        for item in SUBPOP_ATTRIBUTE_CONFIG[attribute]["classes"]
        for config in [SUBPOP_ATTRIBUTE_CONFIG[attribute]]
    }
    questions = {}
    human = {}
    suspicious_trailing = {}

    for row in rows:
        key = (str(row["attribute"]), str(row["group"]))
        if key not in required:
            continue
        qkey = str(row["qkey"])
        source_split = str(row.get("_subpop_source_split", split))
        record_key = (source_split, qkey)
        raw_options = list(row["options"])
        responses = [float(value) for value in row["responses"]]
        # SubPOP follows OpinionQA's convention: `responses` is a POSITIONAL
        # distribution over the substantive options only. The refusal /
        # "don't know" choice is excluded from `responses` (its mass lives in
        # `refusal_rate`) and is listed LAST in `options`. run_inference.py
        # encodes the same convention by dropping the final option with
        # `output_dist[:-1]`. We therefore align by POSITION -- keep the first
        # len(responses) options and treat any trailing options as refusal --
        # instead of string-matching refusal labels, which keeps missing the
        # many spellings surveys use ("Refused", "DK, DK/No", "No answer", ...).
        n_substantive = len(responses)
        if n_substantive > len(raw_options):
            raise ValueError(
                f"SubPOP {qkey}/{key} has more response probabilities "
                f"({n_substantive}) than options ({len(raw_options)}). "
                f"Raw options: {raw_options!r}"
            )
        options = raw_options[:n_substantive]
        for dropped in raw_options[n_substantive:]:
            if not _is_refusal_option(dropped):
                suspicious_trailing.setdefault(str(dropped), (qkey, key))
        question = {
            "qkey": qkey,
            "question": str(row["question"]),
            "options": options,
            "source_split": source_split,
        }
        if record_key in questions and questions[record_key] != question:
            raise ValueError(
                f"Inconsistent question or options for qkey {qkey} in {source_split}."
            )
        questions[record_key] = question
        human[(*record_key, *key)] = {
            "responses": responses,
            "ordinal": [float(value) for value in (row.get("ordinal") or [])],
            "refusal_rate": float(row.get("refusal_rate", 0.0)),
        }

    if suspicious_trailing:
        preview = ", ".join(
            f"{option!r} (e.g. {qkey}/{key})"
            for option, (qkey, key) in list(suspicious_trailing.items())[:10]
        )
        print(
            f"[warn] Dropped {len(suspicious_trailing)} distinct trailing option(s) "
            "that do not look like refusal choices while aligning options to "
            f"`responses`. Verify the option order for: {preview}",
            flush=True,
        )

    complete = []
    for (source_split, qkey), question in questions.items():
        if not all((source_split, qkey, *key) in human for key in required):
            continue
        question["human"] = {
            f"{attribute}|{group}": human[(source_split, qkey, attribute, group)]
            for attribute, group in required
        }
        complete.append(question)
    complete.sort(key=lambda item: (item["source_split"], item["qkey"]))
    if not complete:
        raise ValueError(
            f"No complete questions found for {sorted(required)} in split {split!r}. "
            "Check the split and exact SubPOP group labels."
        )
    return complete