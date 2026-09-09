#!/usr/bin/env python
"""Measure the human-only separability ceiling in OpinionQA.

For each ordinal question and configured demographic attribute, this script
computes pairwise Wasserstein distances between human subgroup distributions.
It uses the same mappings, refusal handling, and ordinal distance as the
model-to-human analyses, but never loads model results.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_opinionqa_model import (
    CLASS_TO_HUMAN,
    DEFAULT_OQA,
    DEFAULT_QKEY_DICT,
    REPO_ROOT,
    align_distribution,
    load_human,
    load_question_metadata,
    wasserstein_ordinal,
)


def configured_groups(human):
    """Return unique configured human groups that exist in OpinionQA."""
    available = {(attribute, group) for _, attribute, group in human}
    mappings = {}
    missing = []
    for attribute, (human_attribute, class_map) in CLASS_TO_HUMAN.items():
        groups = list(dict.fromkeys(class_map.values()))
        present = [group for group in groups if (human_attribute, group) in available]
        missing.extend(
            (attribute, human_attribute, group)
            for group in groups
            if group not in present
        )
        if len(present) >= 2:
            mappings[attribute] = (human_attribute, present)
    return mappings, missing


def build_pairwise_table(human, question_meta, mappings):
    """Compute every configured human-group pair for every eligible question."""
    rows = []
    audit = {
        "candidate_question_attributes": 0,
        "complete_question_attributes": 0,
        "incomplete_question_attributes": 0,
        "invalid_distributions": 0,
        "trailing_values_removed": 0,
    }

    for attribute, (human_attribute, groups) in mappings.items():
        qkeys = sorted({
            qkey for qkey, oqa_attribute, group in human
            if oqa_attribute == human_attribute and group in groups and qkey in question_meta
        })
        for qkey in qkeys:
            audit["candidate_question_attributes"] += 1
            meta = question_meta[qkey]
            aligned = {}
            for group in groups:
                distribution = human.get((qkey, human_attribute, group))
                if distribution is None:
                    continue
                try:
                    aligned[group], removed = align_distribution(
                        distribution, len(meta["ordinal"])
                    )
                    audit["trailing_values_removed"] += removed
                except ValueError:
                    audit["invalid_distributions"] += 1
            if len(aligned) < len(groups):
                audit["incomplete_question_attributes"] += 1
            else:
                audit["complete_question_attributes"] += 1
            if len(aligned) < 2:
                continue

            question_rows = []
            for group_a, group_b in itertools.combinations(aligned, 2):
                distance = wasserstein_ordinal(
                    aligned[group_a], aligned[group_b], meta["ordinal"]
                )
                question_rows.append({
                    "qkey": qkey,
                    "attribute": attribute,
                    "group_a": group_a,
                    "group_b": group_b,
                    "wasserstein": distance,
                })

            mean_gap = float(np.mean([row["wasserstein"] for row in question_rows]))
            max_gap = float(np.max([row["wasserstein"] for row in question_rows]))
            for row in question_rows:
                row["mean_pairwise_wasserstein"] = mean_gap
                row["max_pairwise_wasserstein"] = max_gap
                rows.append(row)

    return pd.DataFrame(rows), audit


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half_width = (
        float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
        if len(values) > 1 else 0.0
    )
    return mean, half_width


def distribution_summary(values, *, near_zero_threshold):
    values = np.asarray(values, dtype=float)
    mean, ci95 = mean_ci(values)
    return {
        "n_observations": len(values),
        "mean_gap": mean,
        "ci95": ci95,
        "ci95_low": mean - ci95,
        "ci95_high": mean + ci95,
        "min_gap": float(np.min(values)),
        "q1_gap": float(np.quantile(values, 0.25)),
        "median_gap": float(np.median(values)),
        "q3_gap": float(np.quantile(values, 0.75)),
        "max_gap": float(np.max(values)),
        "fraction_near_zero": float(np.mean(values <= near_zero_threshold)),
        "near_zero_threshold": near_zero_threshold,
    }


def oracle_ceiling(pairwise, groups, *, equivalence_threshold):
    """Compute exact and practically-equivalent human self-retrieval ceilings.

    The exact oracle includes the distribution being classified among its
    references, so its own distance is zero. Its only errors are exact ties.
    The threshold-aware score additionally treats all groups within the chosen
    Wasserstein distance from zero as indistinguishable and assigns 1/k credit.
    """
    exact_credits = []
    threshold_credits = []
    exact_ties = []
    threshold_ties = []
    pair_lookup = {}
    for row in pairwise.itertuples(index=False):
        pair_lookup[(row.qkey, row.group_a, row.group_b)] = row.wasserstein
        pair_lookup[(row.qkey, row.group_b, row.group_a)] = row.wasserstein

    expected_pairs = len(groups) * (len(groups) - 1) // 2
    pair_counts = pairwise.groupby("qkey").size()
    complete_qkeys = pair_counts[pair_counts.eq(expected_pairs)].index
    for qkey in complete_qkeys:
        for own_group in groups:
            distances = np.asarray([
                0.0 if candidate == own_group
                else pair_lookup[(qkey, own_group, candidate)]
                for candidate in groups
            ])
            exact_nearest = np.flatnonzero(np.isclose(distances, distances.min()))
            threshold_nearest = np.flatnonzero(
                distances <= distances.min() + equivalence_threshold
            )
            own_index = groups.index(own_group)
            exact_credits.append(
                1.0 / len(exact_nearest) if own_index in exact_nearest else 0.0
            )
            threshold_credits.append(
                1.0 / len(threshold_nearest) if own_index in threshold_nearest else 0.0
            )
            exact_ties.append(len(exact_nearest) > 1)
            threshold_ties.append(len(threshold_nearest) > 1)

    if not exact_credits:
        return {
            "oracle_n_classifications": 0,
            "oracle_exact_fractional_accuracy": np.nan,
            "oracle_exact_tie_rate": np.nan,
            "oracle_threshold_fractional_accuracy": np.nan,
            "oracle_threshold_tie_rate": np.nan,
            "oracle_equivalence_threshold": equivalence_threshold,
        }
    return {
        "oracle_n_classifications": len(exact_credits),
        "oracle_exact_fractional_accuracy": float(np.mean(exact_credits)),
        "oracle_exact_tie_rate": float(np.mean(exact_ties)),
        "oracle_threshold_fractional_accuracy": float(np.mean(threshold_credits)),
        "oracle_threshold_tie_rate": float(np.mean(threshold_ties)),
        "oracle_equivalence_threshold": equivalence_threshold,
    }


def build_summary(pairwise, mappings, *, near_zero_threshold, equivalence_threshold):
    rows = []
    for attribute, (_, groups) in mappings.items():
        subset = pairwise[pairwise.attribute.eq(attribute)]
        if subset.empty:
            continue

        oracle = oracle_ceiling(
            subset, groups, equivalence_threshold=equivalence_threshold
        )
        base = {
            "attribute": attribute,
            "n_groups": len(groups),
            "chance_accuracy": 1.0 / len(groups),
            **oracle,
        }

        rows.append({
            "summary_level": "all_pairs",
            "group_a": "ALL",
            "group_b": "ALL",
            "n_questions": subset.qkey.nunique(),
            **base,
            **distribution_summary(
                subset.wasserstein, near_zero_threshold=near_zero_threshold
            ),
        })

        # For binary attributes, these equal the single pairwise gap.
        per_question = subset.groupby("qkey", as_index=False).agg(
            question_mean=("wasserstein", "mean"),
            question_max=("wasserstein", "max"),
        )
        for level, column in (
            ("question_mean", "question_mean"),
            ("question_max", "question_max"),
        ):
            rows.append({
                "summary_level": level,
                "group_a": "ALL",
                "group_b": "ALL",
                "n_questions": len(per_question),
                **base,
                **distribution_summary(
                    per_question[column], near_zero_threshold=near_zero_threshold
                ),
            })

        # Pair-specific rows expose heterogeneous education effects.
        for (group_a, group_b), pair in subset.groupby(["group_a", "group_b"]):
            rows.append({
                "summary_level": "pair",
                "group_a": group_a,
                "group_b": group_b,
                "n_questions": pair.qkey.nunique(),
                **base,
                **distribution_summary(
                    pair.wasserstein, near_zero_threshold=near_zero_threshold
                ),
            })

    columns = [
        "summary_level", "attribute", "group_a", "group_b", "n_groups",
        "n_questions", "n_observations", "mean_gap", "ci95", "ci95_low",
        "ci95_high", "min_gap", "q1_gap", "median_gap", "q3_gap", "max_gap",
        "fraction_near_zero", "near_zero_threshold", "chance_accuracy",
        "oracle_n_classifications", "oracle_exact_fractional_accuracy",
        "oracle_exact_tie_rate", "oracle_threshold_fractional_accuracy",
        "oracle_threshold_tie_rate", "oracle_equivalence_threshold",
    ]
    return pd.DataFrame(rows).reindex(columns=columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opinionqa", type=Path, default=DEFAULT_OQA)
    parser.add_argument("--qkey-dict", type=Path, default=DEFAULT_QKEY_DICT)
    parser.add_argument(
        "--out-dir", type=Path,
        default=REPO_ROOT / "analyses/human_separability",
    )
    parser.add_argument(
        "--near-zero-threshold", type=float, default=0.01,
        help="Gap counted as approximately zero in summary statistics.",
    )
    parser.add_argument(
        "--oracle-equivalence-threshold", type=float, default=0.01,
        help="Distances within this value are tied in the threshold-aware oracle.",
    )
    args = parser.parse_args()

    for path in (args.opinionqa, args.qkey_dict):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.near_zero_threshold < 0 or args.oracle_equivalence_threshold < 0:
        raise ValueError("Distance thresholds must be non-negative.")

    human = load_human(args.opinionqa)
    question_meta = load_question_metadata(args.qkey_dict)
    mappings, missing = configured_groups(human)
    if not mappings:
        raise ValueError("No configured attribute has at least two OpinionQA groups.")

    pairwise, audit = build_pairwise_table(human, question_meta, mappings)
    if pairwise.empty:
        raise ValueError("No complete human subgroup pairs could be computed.")
    summary = build_summary(
        pairwise,
        mappings,
        near_zero_threshold=args.near_zero_threshold,
        equivalence_threshold=args.oracle_equivalence_threshold,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_question_path = args.out_dir / "human_separability_by_question.csv"
    summary_path = args.out_dir / "human_separability_summary.csv"
    pairwise.to_csv(by_question_path, index=False)
    summary.to_csv(summary_path, index=False)

    if missing:
        labels = ", ".join(f"{attribute}:{group}" for attribute, _, group in missing)
        print(f"[warn] configured groups absent from OpinionQA: {labels}")
    print(
        "[info] "
        f"complete question/attributes={audit['complete_question_attributes']} "
        f"incomplete={audit['incomplete_question_attributes']} "
        f"pairwise rows={len(pairwise)}"
    )
    print(f"[ok] wrote {by_question_path.resolve()}")
    print(f"[ok] wrote {summary_path.resolve()}")


if __name__ == "__main__":
    main()
