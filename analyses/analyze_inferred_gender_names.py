#!/usr/bin/env python
"""Analyze gender inference from names and its demographic specificity.

The input is one CSV produced by inferred_gender_names_opinionqa.py. The script
also uses the matching demographic OpinionQA experiment as the neutral,
declared, and steered reference. No language model is loaded.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analyze_cross_group_specificity import (
    build_cross_group_matrix,
    complete_choice_sets,
    mapping_audit,
    nearest_accuracy_summary,
    nearest_neighbor_results,
)
from analyze_opinionqa_model import (
    DEFAULT_OQA,
    DEFAULT_QKEY_DICT,
    REPO_ROOT,
    build_distance_table,
    identify_model,
    load_human,
    load_model,
    load_question_metadata,
    mean_ci,
    parse_sequence,
    wasserstein_ordinal,
)
from analyze_specificity_by_human_gap import (
    accuracy_summary,
    add_pairwise_quartiles,
    build_directional_rows,
    directional_summary,
    trend_tests,
)


POSITIONS = ("after_name", "final_prompt")
PROBE_ATTRIBUTES = ("gender", "age", "education", "socioeco")
GENDER_MAPPING = {"gender": ("SEX", {"male": "Male", "female": "Female"})}
DEFAULT_MAGNITUDES = {"qwen": 20.0, "llama": 20.0, "mistral": 2.0}
REGIME_COLORS = {
    "Neutral (N=0)": "tab:blue",
    "Declared": "tab:orange",
    "Steered": "tab:green",
    "Inferred from name": "tab:purple",
}


def csv_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def require_columns(path: Path, required: set[str]) -> list[str]:
    columns = csv_columns(path)
    missing = required - set(columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return columns


def safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "model"


def load_inferred_profiles(path: Path):
    """Average response distributions across names within each labeled sex."""
    required = {
        "qkey", "name", "ssa_sex_label", "ssa_dominance",
        "response_distribution", "probe_layers",
    }
    columns = require_columns(path, required)
    optional = [name for name in ("model_profile", "model_id") if name in columns]
    frame = pd.read_csv(path, usecols=sorted(required | set(optional)))
    frame["dist"] = frame.response_distribution.map(
        lambda value: np.asarray(parse_sequence(value), dtype=float)
    )

    records = []
    for (qkey, label), group in frame.groupby(["qkey", "ssa_sex_label"], sort=False):
        lengths = {len(value) for value in group.dist}
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent response lengths for {qkey}/{label}: {lengths}")
        records.append({
            "qkey": qkey,
            "attribute": "gender",
            "channel": "inferred",
            "class_label": label,
            "magnitude": np.nan,
            "dist": np.mean(np.stack(group.dist.to_list()), axis=0),
            "n_names": group.name.nunique(),
        })
    profiles = pd.DataFrame(records)

    model_profile = (
        str(frame.model_profile.dropna().iloc[0])
        if "model_profile" in frame and frame.model_profile.notna().any()
        else ""
    )
    model_id = (
        str(frame.model_id.dropna().iloc[0])
        if "model_id" in frame and frame.model_id.notna().any()
        else ""
    )
    layers = [int(value) for value in parse_sequence(frame.probe_layers.iloc[0])]
    identity = {
        "rows": len(frame),
        "questions": frame.qkey.nunique(),
        "names": frame.name.nunique(),
        "male_names": frame.loc[frame.ssa_sex_label.eq("male"), "name"].nunique(),
        "female_names": frame.loc[frame.ssa_sex_label.eq("female"), "name"].nunique(),
        "minimum_dominance": float(frame.ssa_dominance.min()),
        "mean_dominance": float(frame.ssa_dominance.mean()),
        "model_profile": model_profile,
        "model_id": model_id,
        "layers": layers,
    }
    return profiles, identity


def majority_vote(values: list[str]) -> str:
    counts = Counter(values)
    highest = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == highest)
    return " | ".join(winners)


def analyze_probes(path: Path, layers: list[int], chunksize: int):
    """Summarize probe selectivity without loading the wide CSV all at once."""
    required = {"qkey", "name", "ssa_sex_label"}
    for position in POSITIONS:
        required.add(f"{position}_gender_prediction_by_layer")
        for attribute in PROBE_ATTRIBUTES:
            required.add(f"{position}_{attribute}_activation_strength_by_layer")
            required.add(f"{position}_{attribute}_normalized_confidence_by_layer")
    require_columns(path, required)

    cases = []
    layer_totals = defaultdict(lambda: {"n": 0, "strength": 0.0, "confidence": 0.0,
                                        "gender_correct": 0.0})
    seen_after_name = set()

    for chunk in pd.read_csv(path, usecols=sorted(required), chunksize=chunksize):
        for row in chunk.itertuples(index=False):
            row_data = row._asdict()
            for position in POSITIONS:
                case_key = row_data["name"] if position == "after_name" else (
                    row_data["qkey"], row_data["name"]
                )
                if position == "after_name" and case_key in seen_after_name:
                    continue
                if position == "after_name":
                    seen_after_name.add(case_key)

                predictions = parse_sequence(
                    row_data[f"{position}_gender_prediction_by_layer"]
                )
                if len(predictions) != len(layers):
                    raise ValueError(
                        f"{position} gender predictions have {len(predictions)} layers; "
                        f"expected {len(layers)}."
                    )
                label = row_data["ssa_sex_label"]
                record = {
                    "qkey": row_data["qkey"],
                    "name": row_data["name"],
                    "ssa_sex_label": label,
                    "position": position,
                    "gender_layer_accuracy": np.mean(
                        [prediction == label for prediction in predictions]
                    ),
                    "gender_majority_prediction": majority_vote(predictions),
                }
                record["gender_majority_correct"] = (
                    record["gender_majority_prediction"] == label
                )

                for attribute in PROBE_ATTRIBUTES:
                    strength = np.asarray(parse_sequence(
                        row_data[f"{position}_{attribute}_activation_strength_by_layer"]
                    ), dtype=float)
                    confidence = np.asarray(parse_sequence(
                        row_data[f"{position}_{attribute}_normalized_confidence_by_layer"]
                    ), dtype=float)
                    if len(strength) != len(layers) or len(confidence) != len(layers):
                        raise ValueError(
                            f"{position}/{attribute} probe arrays do not match probe_layers."
                        )
                    record[f"{attribute}_mean_strength"] = float(strength.mean())
                    record[f"{attribute}_mean_confidence"] = float(confidence.mean())
                    for index, layer in enumerate(layers):
                        total = layer_totals[(position, attribute, layer)]
                        total["n"] += 1
                        total["strength"] += float(strength[index])
                        total["confidence"] += float(confidence[index])
                        if attribute == "gender":
                            total["gender_correct"] += predictions[index] == label
                cases.append(record)

    case_frame = pd.DataFrame(cases)
    layer_rows = []
    for (position, attribute, layer), values in layer_totals.items():
        layer_rows.append({
            "position": position,
            "attribute": attribute,
            "layer": layer,
            "n": values["n"],
            "mean_activation_strength": values["strength"] / values["n"],
            "mean_normalized_confidence": values["confidence"] / values["n"],
            "gender_accuracy": (
                values["gender_correct"] / values["n"]
                if attribute == "gender" else np.nan
            ),
        })
    layer_frame = pd.DataFrame(layer_rows).sort_values(
        ["position", "attribute", "layer"]
    )

    summary_rows = []
    for position, group in case_frame.groupby("position", sort=False):
        for attribute in PROBE_ATTRIBUTES:
            values = group[f"{attribute}_mean_strength"].to_numpy(dtype=float)
            mean, ci = mean_ci(values)
            summary_rows.append({
                "position": position,
                "attribute": attribute,
                "n_cases": len(group),
                "mean_activation_strength": mean,
                "ci95": ci,
                "mean_normalized_confidence": group[
                    f"{attribute}_mean_confidence"
                ].mean(),
                "gender_layer_accuracy": (
                    group.gender_layer_accuracy.mean() if attribute == "gender" else np.nan
                ),
                "gender_majority_accuracy": (
                    group.gender_majority_correct.mean() if attribute == "gender" else np.nan
                ),
            })
    probe_summary = pd.DataFrame(summary_rows)

    comparison_rows = []
    for position, group in case_frame.groupby("position", sort=False):
        gender = group.gender_mean_strength.to_numpy(dtype=float)
        for attribute in PROBE_ATTRIBUTES[1:]:
            other = group[f"{attribute}_mean_strength"].to_numpy(dtype=float)
            difference = gender - other
            p_value = np.nan
            if len(difference) and not np.allclose(difference, 0):
                p_value = float(wilcoxon(difference, alternative="greater").pvalue)
            comparison_rows.append({
                "position": position,
                "comparison": f"gender vs {attribute}",
                "n_cases": len(group),
                "mean_gender_strength": gender.mean(),
                "mean_other_strength": other.mean(),
                "mean_difference": difference.mean(),
                "fraction_gender_stronger": np.mean(difference > 0),
                "p_value_one_sided": p_value,
            })
    return case_frame, layer_frame, probe_summary, pd.DataFrame(comparison_rows)


def plot_probe_accuracy(layer_summary: pd.DataFrame, out_path: Path):
    gender = layer_summary[layer_summary.attribute.eq("gender")]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = {"after_name": "After name", "final_prompt": "Final prompt token"}
    for position, group in gender.groupby("position", sort=False):
        ax.plot(group.layer, group.gender_accuracy, marker="o", ms=3,
                label=labels[position])
    ax.axhline(0.5, color="gray", ls=":", label="Chance")
    ax.set(xlabel="Layer", ylabel="Gender classification accuracy",
           title="Can the gender probe recover the name-associated identity?")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_probe_strength(probe_summary: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    labels = {"after_name": "After name", "final_prompt": "Final prompt token"}
    colors = ["tab:purple", "tab:blue", "tab:orange", "tab:green"]
    for ax, position in zip(axes, POSITIONS):
        values = probe_summary[probe_summary.position.eq(position)].set_index(
            "attribute"
        ).reindex(PROBE_ATTRIBUTES)
        x = np.arange(len(values))
        ax.bar(x, values.mean_activation_strength, yerr=values.ci95,
               capsize=3, color=colors)
        ax.set_xticks(x, values.index, rotation=20)
        ax.set_title(labels[position])
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("Chance-corrected probe selectivity (95% CI)")
    fig.suptitle("Gender selectivity versus unrelated demographic probes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def human_gender_pairs(human, question_meta):
    rows = []
    for qkey, meta in question_meta.items():
        male = human.get((qkey, "SEX", "Male"))
        female = human.get((qkey, "SEX", "Female"))
        if male is None or female is None:
            continue
        try:
            gap = wasserstein_ordinal(male, female, meta["ordinal"])
        except ValueError:
            continue
        rows.append({
            "qkey": qkey,
            "attribute": "gender",
            "group_a": "Male",
            "group_b": "Female",
            "model_class_a": "male",
            "model_class_b": "female",
            "n_human_groups": 2,
            "chance_accuracy": 0.5,
            "human_gap": gap,
        })
    return add_pairwise_quartiles(pd.DataFrame(rows))


def inferred_cross_group(profiles, human, question_meta):
    mappings, unmapped = mapping_audit(profiles, human)
    if unmapped:
        raise ValueError(f"Unexpected inferred classes: {unmapped}")
    cross, audit = build_cross_group_matrix(profiles, human, question_meta, mappings)
    cross["regime"] = "Inferred from name"
    complete, incomplete = complete_choice_sets(cross, mappings)
    if complete.empty:
        raise ValueError("No complete inferred-name cross-group choice sets were found.")
    nearest = nearest_neighbor_results(complete)
    nearest_accuracy = nearest_accuracy_summary(nearest, mappings)
    return complete, nearest, nearest_accuracy, audit, incomplete


def find_baseline(path: Path | None, profile: str) -> Path:
    if path is not None:
        return path
    if not profile:
        raise ValueError(
            "Could not identify the model profile; provide --baseline-results."
        )
    candidate = REPO_ROOT / "results" / f"demographic_opinionqa_{profile}.csv"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Automatic baseline not found at {candidate}. "
            "Provide it with --baseline-results."
        )
    return candidate


def select_magnitude(baseline: pd.DataFrame, requested: float | None, profile: str):
    available = sorted(
        float(value) for value in baseline.loc[
            baseline.channel.eq("steered"), "magnitude"
        ].dropna().unique() if float(value) > 0
    )
    if not available:
        raise ValueError("The baseline CSV contains no positive steering magnitude.")
    target = requested
    if target is None:
        target = DEFAULT_MAGNITUDES.get(profile, available[-1])
    selected = min(available, key=lambda value: (abs(value - target), value))
    return selected, selected != target


def distance_comparison(inferred_cross, baseline, human, question_meta,
                        selected_magnitude):
    baseline = baseline[baseline.attribute.eq("gender")].copy()
    baseline_distance, audit = build_distance_table(
        baseline, human, question_meta
    )
    frames = []
    conditions = (
        ("Neutral (N=0)", baseline_distance[
            baseline_distance.channel.eq("steered")
            & baseline_distance.magnitude.eq(0)
        ]),
        ("Declared", baseline_distance[baseline_distance.channel.eq("declared")]),
        (f"Steered N={selected_magnitude:g}", baseline_distance[
            baseline_distance.channel.eq("steered")
            & baseline_distance.magnitude.eq(selected_magnitude)
        ]),
    )
    for regime, frame in conditions:
        part = frame[["qkey", "class_label", "wasserstein"]].copy()
        part["regime"] = regime
        frames.append(part)
    inferred = inferred_cross[inferred_cross.is_own_group][
        ["qkey", "model_class", "wasserstein"]
    ].rename(columns={"model_class": "class_label"})
    inferred["regime"] = "Inferred from name"
    frames.append(inferred)
    distances = pd.concat(frames, ignore_index=True)

    neutral = distances[distances.regime.eq("Neutral (N=0)")][
        ["qkey", "class_label", "wasserstein"]
    ].rename(columns={"wasserstein": "neutral_wasserstein"})
    paired = distances.merge(neutral, on=["qkey", "class_label"], how="inner")
    paired["delta_vs_neutral"] = paired.wasserstein - paired.neutral_wasserstein

    rows = []
    for class_label in ("male", "female", "all"):
        subset = paired if class_label == "all" else paired[
            paired.class_label.eq(class_label)
        ]
        for regime, group in subset.groupby("regime", sort=False):
            # For the combined result, average the two gender endpoints within
            # each question before inference so qkey remains the sampling unit.
            units = (
                group.groupby("qkey", as_index=False)[
                    ["wasserstein", "delta_vs_neutral"]
                ].mean()
                if class_label == "all" else group
            )
            mean_distance, distance_ci = mean_ci(units.wasserstein)
            mean_delta, delta_ci = mean_ci(units.delta_vs_neutral)
            p_value = np.nan
            if not np.allclose(units.delta_vs_neutral, 0):
                p_value = float(wilcoxon(units.delta_vs_neutral).pvalue)
            rows.append({
                "class_label": class_label,
                "regime": regime,
                "n": len(units),
                "mean_wasserstein": mean_distance,
                "wasserstein_ci95": distance_ci,
                "mean_delta_vs_neutral": mean_delta,
                "delta_ci95": delta_ci,
                "paired_wilcoxon_p": p_value,
            })
    return paired, pd.DataFrame(rows), audit


def plot_distance_delta(summary: pd.DataFrame, selected_magnitude: float, out_path: Path):
    regimes = ["Declared", f"Steered N={selected_magnitude:g}", "Inferred from name"]
    classes = ["male", "female", "all"]
    x, width = np.arange(len(classes)), 0.24
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, regime in enumerate(regimes):
        values = summary[summary.regime.eq(regime)].set_index("class_label").reindex(classes)
        color = REGIME_COLORS["Steered"] if regime.startswith("Steered") else REGIME_COLORS[regime]
        ax.bar(x + (index - 1) * width, values.mean_delta_vs_neutral, width,
               yerr=values.delta_ci95, capsize=3, label=regime, color=color)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, ["Male", "Female", "Overall"])
    ax.set_ylabel(r"$\Delta W$ relative to neutral (95% CI)")
    ax.set_title("Distance to the matching human gender group")
    ax.text(0.01, 0.02, "negative = closer", transform=ax.transAxes, fontsize=9)
    ax.grid(axis="y", alpha=0.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_nearest_confusion(nearest: pd.DataFrame, out_path: Path):
    labels = ["Male", "Female"]
    if nearest.predicted_human_group.str.contains(r"\|", regex=True).any():
        labels.append("TIE")
    plotted = nearest.assign(
        plotted_prediction=np.where(
            nearest.n_tied_nearest.gt(1), "TIE", nearest.predicted_human_group
        )
    )
    matrix = pd.crosstab(plotted.model_class, plotted.plotted_prediction).reindex(
        index=["male", "female"], columns=labels, fill_value=0
    )
    fractions = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues")
    for row in range(2):
        for column in range(len(labels)):
            value = fractions.iloc[row, column]
            ax.text(column, row, "NA" if not np.isfinite(value) else f"{value:.2f}",
                    ha="center", va="center",
                    color="white" if np.isfinite(value) and value > 0.55 else "black")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(2), ["male-name profile", "female-name profile"])
    ax.set(xlabel="Nearest human group", ylabel="Inferred model profile",
           title="Cross-group nearest-neighbor classification")
    fig.colorbar(image, ax=ax, label="Row fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_gap_recovery(accuracy: pd.DataFrame, out_path: Path):
    quartiles = ["Q1", "Q2", "Q3", "Q4"]
    values = accuracy[
        accuracy.regime.eq("Inferred from name") & accuracy.quartile.isin(quartiles)
    ].set_index("quartile").reindex(quartiles)
    lower = values.lift_over_chance - values.lift_ci95_low
    upper = values.lift_ci95_high - values.lift_over_chance
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(quartiles, values.lift_over_chance,
                yerr=np.vstack([lower, upper]), marker="o", capsize=4,
                color=REGIME_COLORS["Inferred from name"])
    ax.axhline(0, color="black", lw=1)
    ax.set(xlabel="Human gender-gap quartile",
           ylabel="Nearest-group lift over chance",
           title="Does recovery increase where human gender groups diverge?")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_gap_trend(trends: pd.DataFrame, out_path: Path):
    row = trends[trends.regime.eq("Inferred from name")].iloc[0]
    odds = row.odds_ratio_per_gap_sd
    low = row.odds_ratio_ci95_low
    high = row.odds_ratio_ci95_high
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.errorbar(
        odds, 0,
        xerr=np.asarray([[odds - low], [high - odds]]),
        marker="o", markersize=8, capsize=4,
        color=REGIME_COLORS["Inferred from name"],
    )
    ax.axvline(1, color="black", lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_yticks([0], ["Inferred from name"])
    ax.set_xlabel("Odds ratio per SD of human gender gap (95% CI)")
    ax.set_title("Continuous trend in nearest-group recovery")
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def directional_q4_table(directional_rows, directional):
    q4_summary = directional[
        directional.quartile.eq("Q4")
        & directional.regime.eq("Inferred from name")
    ].copy()
    q4_rows = directional_rows[
        directional_rows.quartile.eq("Q4")
        & directional_rows.regime.eq("Inferred from name")
    ]
    ci_rows = []
    for target, group in q4_rows.groupby("target_human_group"):
        mean, ci = mean_ci(group.difference)
        ci_rows.append({"target_human_group": target, "difference_ci95": ci})
    return q4_summary.merge(pd.DataFrame(ci_rows), on="target_human_group", how="left")


def plot_directional_q4(q4: pd.DataFrame, out_path: Path):
    order = ["Male", "Female"]
    values = q4.set_index("target_human_group").reindex(order)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["tab:blue", "tab:red"]
    ax.bar(x, values.mean_difference, yerr=values.difference_ci95,
           capsize=4, color=colors)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, order)
    ax.set_ylabel("Own − competitor Wasserstein distance (95% CI)")
    ax.set_title("Directional specificity in high-human-gap questions (Q4)")
    ax.text(0.01, 0.02, "negative = correct direction", transform=ax.transAxes, fontsize=9)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fmt(value, digits=3):
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def fmt_p(value):
    return "NA" if not np.isfinite(value) else f"{value:.3g}"


def write_report(path: Path, *, input_path: Path, baseline_path: Path, identity: dict,
                 selected_magnitude: float, fallback: bool, probe_summary: pd.DataFrame,
                 activation: pd.DataFrame, distance: pd.DataFrame,
                 nearest_accuracy: pd.DataFrame, quartile_accuracy: pd.DataFrame,
                 trends: pd.DataFrame, q4_directional: pd.DataFrame,
                 audit: dict, incomplete: int, files: dict[str, Path]):
    inferred_accuracy = nearest_accuracy.iloc[0]
    quartile = quartile_accuracy[quartile_accuracy.regime.eq("Inferred from name")]
    q4 = quartile[quartile.quartile.eq("Q4")].iloc[0]
    trend = trends[trends.regime.eq("Inferred from name")].iloc[0]
    distance_all = distance[distance.class_label.eq("all")]
    inferred_distance = distance_all[
        distance_all.regime.eq("Inferred from name")
    ].iloc[0]
    after_name = probe_summary[
        probe_summary.attribute.eq("gender")
        & probe_summary.position.eq("after_name")
    ].iloc[0]
    weakest_activation = activation.loc[activation.mean_difference.idxmin()]

    lines = [
        f"# Inferred gender from names: {identity['model_profile'] or 'model'}", "",
        f"- Inferred-results input: `{input_path.resolve()}`",
        f"- Baseline experiment: `{baseline_path.resolve()}`",
        f"- Model: `{identity['model_id'] or identity['model_profile'] or 'not recorded'}`",
        f"- Questions: `{identity['questions']}`; names: `{identity['names']}` "
        f"({identity['male_names']} male, {identity['female_names']} female)",
        f"- Minimum SSA sex-label dominance: `{identity['minimum_dominance']:.3f}`",
        f"- Steering reference magnitude: `{selected_magnitude:g}`"
        + (" (nearest available)" if fallback else ""), "",
        "The response distribution is averaged across names within each sex label before "
        "human-alignment analyses. This keeps the unit comparable to the declared and "
        "steered profiles and prevents individual names from becoming pseudo-replicates.", "",
        "## Headline", "",
        f"The across-layer majority vote recovers the name-associated gender with "
        f"**{after_name.gender_majority_accuracy:.1%} accuracy immediately after the name**. "
        f"Gender selectivity exceeds every unrelated probe on average, although its smallest "
        f"advantage is only **{weakest_activation.mean_difference:+.3f}** "
        f"({weakest_activation.comparison}, {weakest_activation.position}).", "",
        f"At the response level, inferred identity changes matching-group distance by "
        f"**Delta W={inferred_distance.mean_delta_vs_neutral:+.3f}** relative to neutral. "
        f"Nearest-group lift is **{inferred_accuracy.lift_over_chance:+.3f}** overall and "
        f"**{q4.lift_over_chance:+.3f}** in Q4; the continuous trend is "
        f"**OR={fmt(trend.odds_ratio_per_gap_sd)}** per SD of human divergence.", "",
        "## 1. Gender inference by the probes", "",
        "`Layer accuracy` averages correctness across layers. `Majority accuracy` asks whether "
        "the majority of layers predicts the SSA-associated gender.", "",
        "| Position | Layer accuracy | Majority accuracy | Cases |",
        "| --- | ---: | ---: | ---: |",
    ]
    gender_probe = probe_summary[probe_summary.attribute.eq("gender")]
    for row in gender_probe.itertuples(index=False):
        lines.append(
            f"| {row.position} | {row.gender_layer_accuracy:.3f} | "
            f"{row.gender_majority_accuracy:.3f} | {row.n_cases} |"
        )

    lines.extend(["", "## 2. Gender versus unrelated probe selectivity", "",
                  "Selectivity is the normalized top-class confidence corrected for chance "
                  "under each probe's number of classes. Positive differences favor gender. "
                  "Because independently trained probes need not be calibrated identically, "
                  "this is a relative diagnostic rather than a neuronal activation measure.", "",
                  "| Position | Comparison | Mean difference | Gender stronger | p (one-sided) |",
                  "| --- | --- | ---: | ---: | ---: |"])
    for row in activation.itertuples(index=False):
        lines.append(
            f"| {row.position} | {row.comparison} | {row.mean_difference:+.3f} | "
            f"{row.fraction_gender_stronger:.3f} | {fmt_p(row.p_value_one_sided)} |"
        )

    lines.extend(["", "## 3. Distance to the matching human group", "",
                  "Negative delta values mean that the regime is closer to the matching human "
                  "gender distribution than the neutral model.", "",
                  "| Regime | Mean W | Delta W vs neutral | 95% CI of delta | paired p |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for row in distance_all.itertuples(index=False):
        lines.append(
            f"| {row.regime} | {row.mean_wasserstein:.3f} | "
            f"{row.mean_delta_vs_neutral:+.3f} | ±{row.delta_ci95:.3f} | "
            f"{fmt_p(row.paired_wilcoxon_p)} |"
        )

    lines.extend(["", "## 4. Cross-group specificity", "",
                  f"Nearest-human-group strict accuracy is **{inferred_accuracy.strict_accuracy:.3f}**; "
                  f"fractional-tie accuracy is **{inferred_accuracy.fractional_tie_accuracy:.3f}** "
                  f"against chance **{inferred_accuracy.chance_accuracy:.3f}** "
                  f"(lift **{inferred_accuracy.lift_over_chance:+.3f}**).", "",
                  "## 5. Recovery as a function of human divergence", "",
                  f"In Q4, fractional accuracy is **{q4.fractional_accuracy:.3f}**, with lift "
                  f"**{q4.lift_over_chance:+.3f}** and bootstrap 95% CI "
                  f"[{q4.lift_ci95_low:+.3f}, {q4.lift_ci95_high:+.3f}].", "",
                  f"The continuous logistic trend gives OR per human-gap SD = "
                  f"**{fmt(trend.odds_ratio_per_gap_sd)}** (95% CI "
                  f"[{fmt(trend.odds_ratio_ci95_low)}, {fmt(trend.odds_ratio_ci95_high)}]; "
                  f"one-sided cluster-robust p = "
                  f"{fmt_p(trend.cluster_robust_p_slope_le_zero)}).", "",
                  "## 6. Directional test in Q4", "",
                  "Negative own-minus-competitor values support gender-specific recovery; "
                  "opposite signs across the two targets indicate collapse toward one pole.", "",
                  "| Human target | n | Mean own-minus-competitor | Win rate | Holm p |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for row in q4_directional.itertuples(index=False):
        lines.append(
            f"| {row.target_human_group} | {row.n_questions} | "
            f"{row.mean_difference:+.3f} | {row.fractional_win_rate:.3f} | "
            f"{fmt_p(row.p_value_holm)} |"
        )

    lines.extend(["", "## Audit", "",
                  f"- Cross-group cells: `{audit['cross_group_rows']}`",
                  f"- Incomplete nearest-neighbor sets excluded: `{incomplete}`",
                  f"- Model trailing non-ordinal values removed: "
                  f"`{audit['trailing_model_values_removed']}`",
                  f"- Human trailing non-ordinal values removed: "
                  f"`{audit['trailing_human_values_removed']}`", "",
                  "## Output files", ""])
    lines.extend(f"- [{label}]({file.name})" for label, file in files.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument("--opinionqa", type=Path, default=DEFAULT_OQA)
    parser.add_argument("--qkey-dict", type=Path, default=DEFAULT_QKEY_DICT)
    parser.add_argument("--comparison-magnitude", type=float)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=500)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    for path in (args.input, args.opinionqa, args.qkey_dict):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.bootstrap_iterations < 100:
        raise ValueError("Use at least 100 bootstrap iterations.")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be positive.")

    profiles, identity = load_inferred_profiles(args.input)
    baseline_path = find_baseline(args.baseline_results, identity["model_profile"])
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    baseline = load_model(baseline_path)
    baseline_label, baseline_model_id = identify_model(baseline, baseline_path)
    if identity["model_id"] and baseline_model_id and identity["model_id"] != baseline_model_id:
        raise ValueError(
            f"Model mismatch: inferred CSV uses {identity['model_id']!r}, but baseline "
            f"uses {baseline_model_id!r}."
        )
    selected_magnitude, fallback = select_magnitude(
        baseline, args.comparison_magnitude, identity["model_profile"] or baseline_label
    )

    human = load_human(args.opinionqa)
    question_meta = load_question_metadata(args.qkey_dict)
    cross, nearest, nearest_accuracy, cross_audit, incomplete = inferred_cross_group(
        profiles, human, question_meta
    )
    human_pairs = human_gender_pairs(human, question_meta)
    predictions = nearest.merge(
        human_pairs[["qkey", "attribute", "group_a", "group_b", "model_class_a",
                     "model_class_b", "n_human_groups", "chance_accuracy", "human_gap",
                     "quartile"]],
        on=["qkey", "attribute"], how="inner",
    )
    quartile_accuracy = accuracy_summary(
        predictions, iterations=args.bootstrap_iterations, seed=args.seed
    )
    trends = trend_tests(predictions)
    trends["odds_ratio_ci95_low"] = np.exp(trends.slope_ci95_low)
    trends["odds_ratio_ci95_high"] = np.exp(trends.slope_ci95_high)
    directional_rows = build_directional_rows(cross, human_pairs)
    directional = directional_summary(directional_rows)
    q4_directional = directional_q4_table(directional_rows, directional)

    paired_distance, distance_summary, baseline_audit = distance_comparison(
        cross, baseline, human, question_meta, selected_magnitude
    )
    case_summary, layer_summary, probe_summary, activation = analyze_probes(
        args.input, identity["layers"], args.chunksize
    )

    model_label = safe_label(identity["model_profile"] or baseline_label)
    out_dir = args.out_dir or REPO_ROOT / "analyses/inferred_gender_reports" / model_label
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "Probe case summary": out_dir / "probe_case_summary.csv",
        "Probe results by layer": out_dir / "probe_by_layer.csv",
        "Probe summary": out_dir / "probe_summary.csv",
        "Probe activation comparisons": out_dir / "probe_activation_comparisons.csv",
        "Inferred profiles": out_dir / "inferred_profiles.csv",
        "Matching-group paired distances": out_dir / "distance_to_matching_human.csv",
        "Distance summary": out_dir / "distance_vs_neutral_summary.csv",
        "Cross-group distances": out_dir / "cross_group_distances.csv",
        "Nearest-group predictions": out_dir / "nearest_group_predictions.csv",
        "Nearest-group accuracy": out_dir / "nearest_group_accuracy.csv",
        "Accuracy by human-gap quartile": out_dir / "accuracy_by_human_gap_quartile.csv",
        "Continuous human-gap trends": out_dir / "accuracy_gap_trends.csv",
        "Directional rows": out_dir / "directional_by_question.csv",
        "Directional Q4 tests": out_dir / "directional_tests_q4.csv",
        "Gender-probe accuracy plot": out_dir / "gender_probe_accuracy_by_layer.png",
        "Probe selectivity plot": out_dir / "probe_selectivity_by_attribute.png",
        "Distance delta plot": out_dir / "distance_delta_vs_neutral.png",
        "Nearest-group confusion plot": out_dir / "nearest_group_confusion.png",
        "Human-gap recovery plot": out_dir / "lift_by_human_gap_quartile.png",
        "Human-gap odds-ratio plot": out_dir / "human_gap_odds_ratio.png",
        "Directional Q4 plot": out_dir / "directional_specificity_q4.png",
    }

    case_summary.to_csv(files["Probe case summary"], index=False)
    layer_summary.to_csv(files["Probe results by layer"], index=False)
    probe_summary.to_csv(files["Probe summary"], index=False)
    activation.to_csv(files["Probe activation comparisons"], index=False)
    serializable_profiles = profiles.copy()
    serializable_profiles["response_distribution"] = serializable_profiles.dist.map(
        lambda values: json.dumps([round(float(value), 8) for value in values])
    )
    serializable_profiles.drop(columns="dist").to_csv(files["Inferred profiles"], index=False)
    paired_distance.to_csv(files["Matching-group paired distances"], index=False)
    distance_summary.to_csv(files["Distance summary"], index=False)
    cross.to_csv(files["Cross-group distances"], index=False)
    nearest.to_csv(files["Nearest-group predictions"], index=False)
    nearest_accuracy.to_csv(files["Nearest-group accuracy"], index=False)
    quartile_accuracy.to_csv(files["Accuracy by human-gap quartile"], index=False)
    trends.to_csv(files["Continuous human-gap trends"], index=False)
    directional_rows.to_csv(files["Directional rows"], index=False)
    q4_directional.to_csv(files["Directional Q4 tests"], index=False)

    plot_probe_accuracy(layer_summary, files["Gender-probe accuracy plot"])
    plot_probe_strength(probe_summary, files["Probe selectivity plot"])
    plot_distance_delta(distance_summary, selected_magnitude, files["Distance delta plot"])
    plot_nearest_confusion(nearest, files["Nearest-group confusion plot"])
    plot_gap_recovery(quartile_accuracy, files["Human-gap recovery plot"])
    plot_gap_trend(trends, files["Human-gap odds-ratio plot"])
    plot_directional_q4(q4_directional, files["Directional Q4 plot"])

    report_path = out_dir / "report.md"
    write_report(
        report_path,
        input_path=args.input,
        baseline_path=baseline_path,
        identity=identity,
        selected_magnitude=selected_magnitude,
        fallback=fallback,
        probe_summary=probe_summary,
        activation=activation,
        distance=distance_summary,
        nearest_accuracy=nearest_accuracy,
        quartile_accuracy=quartile_accuracy,
        trends=trends,
        q4_directional=q4_directional,
        audit=cross_audit,
        incomplete=incomplete,
        files=files,
    )
    print(f"[ok] model={model_label} comparison_magnitude={selected_magnitude:g}")
    print(f"[ok] report={report_path.resolve()}")


if __name__ == "__main__":
    main()
