#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze CSV output from demographic_opinionqa_experiment.py.

The script computes per-question distances and compact aggregate summaries. It
does not call an LLM and does not modify the original experiment result file.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


ANALYSES_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSES_DIR.parent
DEFAULT_QKEY_DICT = REPO_ROOT / "data/political_questions/OpinionQA/refined_qkey_dict.json"


def parse_distribution(raw):
    values = np.asarray(json.loads(raw), dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError(f"Invalid response distribution: {raw!r}")
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"Distribution has invalid total: {raw!r}")
    return values / total


def load_ordinals(path):
    with open(path, encoding="utf-8") as stream:
        records = json.load(stream)
    ordinals = {}
    for qkey, record in records.items():
        options = list(record["options"])
        if options and options[-1] == "Refused":
            options = options[:-1]
        ordinal = record.get("ordinal", list(range(1, len(options) + 1)))
        ordinals[qkey] = np.asarray(ordinal[:len(options)], dtype=float)
    return ordinals


def load_results(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "qkey", "attribute", "channel", "class_idx", "class_label",
            "magnitude", "response_distribution",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
        for source in reader:
            rows.append({
                "qkey": source["qkey"],
                "attribute": source["attribute"],
                "channel": source["channel"],
                "class_idx": int(source["class_idx"]),
                "class_label": source["class_label"],
                "magnitude": None if source["magnitude"] == "" else float(source["magnitude"]),
                "distribution": parse_distribution(source["response_distribution"]),
            })
    if not rows:
        raise ValueError("Input CSV contains no result rows.")
    return rows


def tv_distance(p, q):
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def wasserstein_ordinal(p, q, ordinal):
    """Exact 1D Wasserstein distance, matching the earlier experiment logic."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    ordinal = np.asarray(ordinal, dtype=float)
    if not (len(p) == len(q) == len(ordinal)):
        raise ValueError(
            f"Distribution/ordinal length mismatch: {len(p)}, {len(q)}, {len(ordinal)}"
        )
    order = np.argsort(ordinal, kind="stable")
    x = ordinal[order]
    cumulative_gap = np.abs(np.cumsum(p[order]) - np.cumsum(q[order]))
    return float(np.sum(cumulative_gap[:-1] * np.diff(x)))


def distances(p, q, ordinal):
    return {
        "wasserstein": wasserstein_ordinal(p, q, ordinal),
        "total_variation": tv_distance(p, q),
    }


def compute_pairwise(rows, ordinals):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["qkey"], row["attribute"], row["channel"], row["magnitude"])].append(row)

    output = []
    for (qkey, attribute, channel, magnitude), group in grouped.items():
        group = sorted(group, key=lambda row: row["class_idx"])
        for first, second in itertools.combinations(group, 2):
            metric = distances(first["distribution"], second["distribution"], ordinals[qkey])
            output.append({
                "qkey": qkey,
                "attribute": attribute,
                "channel": channel,
                "magnitude": "" if magnitude is None else magnitude,
                "class_a_idx": first["class_idx"],
                "class_a": first["class_label"],
                "class_b_idx": second["class_idx"],
                "class_b": second["class_label"],
                **metric,
            })
    return output


def compute_declared_vs_steered(rows, ordinals):
    declared = {}
    for row in rows:
        if row["channel"] == "declared":
            declared[(row["qkey"], row["attribute"], row["class_idx"])] = row

    output = []
    for steered in rows:
        if steered["channel"] != "steered":
            continue
        key = (steered["qkey"], steered["attribute"], steered["class_idx"])
        matching_declared = declared.get(key)
        if matching_declared is None:
            continue
        metric = distances(
            matching_declared["distribution"], steered["distribution"],
            ordinals[steered["qkey"]],
        )
        output.append({
            "qkey": steered["qkey"],
            "attribute": steered["attribute"],
            "class_idx": steered["class_idx"],
            "class_label": steered["class_label"],
            "magnitude": steered["magnitude"],
            **metric,
        })
    return output


def compute_dose_shift(rows, ordinals):
    baselines = {}
    for row in rows:
        if row["channel"] == "steered" and row["magnitude"] == 0:
            baselines[(row["qkey"], row["attribute"], row["class_idx"])] = row

    output = []
    for row in rows:
        if row["channel"] != "steered" or row["magnitude"] in {None, 0}:
            continue
        baseline = baselines.get((row["qkey"], row["attribute"], row["class_idx"]))
        if baseline is None:
            continue
        metric = distances(baseline["distribution"], row["distribution"], ordinals[row["qkey"]])
        output.append({
            "qkey": row["qkey"],
            "attribute": row["attribute"],
            "class_idx": row["class_idx"],
            "class_label": row["class_label"],
            "magnitude": row["magnitude"],
            **metric,
        })
    return output


def summarize(rows, group_fields, analysis):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)

    output = []
    for key, group in grouped.items():
        result = {"analysis": analysis}
        result.update(dict(zip(group_fields, key)))
        result["n_questions"] = len(group)
        for metric in ("wasserstein", "total_variation"):
            values = [float(row[metric]) for row in group]
            result[f"{metric}_mean"] = statistics.fmean(values)
            result[f"{metric}_median"] = statistics.median(values)
            result[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
            result[f"{metric}_max"] = max(values)
        output.append(result)
    return output


def write_csv(path, rows):
    if not rows:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV created by the initial experiment runner")
    parser.add_argument("--qkey-dict", type=Path, default=DEFAULT_QKEY_DICT)
    parser.add_argument("--out-dir", type=Path, default=ANALYSES_DIR / "data")
    parser.add_argument("--prefix", default="demographic")
    args = parser.parse_args()

    rows = load_results(args.input)
    ordinals = load_ordinals(args.qkey_dict)
    unknown = sorted({row["qkey"] for row in rows} - set(ordinals))
    if unknown:
        raise ValueError(f"Input contains qkeys absent from qkey dictionary: {unknown[:5]}")

    pairwise = compute_pairwise(rows, ordinals)
    alignment = compute_declared_vs_steered(rows, ordinals)
    dose_shift = compute_dose_shift(rows, ordinals)

    summaries = []
    summaries.extend(summarize(
        pairwise,
        ["attribute", "channel", "magnitude", "class_a", "class_b"],
        "pairwise_classes",
    ))
    summaries.extend(summarize(
        alignment,
        ["attribute", "class_label", "magnitude"],
        "declared_vs_same_class_steered",
    ))
    summaries.extend(summarize(
        dose_shift,
        ["attribute", "class_label", "magnitude"],
        "steered_vs_neutral_n0",
    ))

    outputs = {
        "pairwise": (args.out_dir / f"{args.prefix}_pairwise_distances.csv", pairwise),
        "alignment": (args.out_dir / f"{args.prefix}_declared_vs_steered.csv", alignment),
        "dose": (args.out_dir / f"{args.prefix}_dose_shift.csv", dose_shift),
        "summary": (args.out_dir / f"{args.prefix}_summary.csv", summaries),
    }
    for label, (path, data) in outputs.items():
        if write_csv(path, data):
            print(f"[ok] {label}: {len(data)} rows -> {path.resolve()}")
        else:
            print(f"[skip] {label}: no compatible rows in the input")


if __name__ == "__main__":
    main()
