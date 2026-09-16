#!/usr/bin/env python
"""Analyze socioeconomic inference from BLS-derived occupation cues.

The input is one CSV produced by inferred_income_occupations_opinionqa.py. The
script reads all demographic probes and compares occupation-conditioned answers
with the matching OpinionQA income groups and with the neutral, declared, and
steered conditions from the same model. No language model is loaded.
"""

from __future__ import annotations

import argparse
import json
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
from analyze_inferred_gender_names import (
    find_baseline,
    fmt,
    fmt_p,
    require_columns,
    safe_label,
    select_magnitude,
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


POSITIONS = ("after_occupation", "final_prompt")
PROBE_ATTRIBUTES = ("socioeco", "education", "gender", "age")
REGIME_INFERRED = "Inferred from occupation"
INCOME_MAPPING = {
    "socioeco": (
        "INCOME",
        {"low": "Less than $30,000", "high": "$100,000 or more"},
    )
}
REGIME_COLORS = {
    "Neutral (N=0)": "tab:blue",
    "Declared": "tab:orange",
    "Steered": "tab:green",
    REGIME_INFERRED: "tab:purple",
}


def majority_vote(values: list[str]) -> str:
    counts = Counter(values)
    highest = max(counts.values())
    return " | ".join(sorted(label for label, count in counts.items() if count == highest))


def load_inferred_profiles(path: Path):
    """Average answer distributions across occupations in each wage group."""
    required = {
        "qkey", "occupation_cue", "income_group", "median_annual_wage_usd",
        "response_distribution", "probe_layers",
    }
    columns = require_columns(path, required)
    optional = [name for name in ("model_profile", "model_id", "selection") if name in columns]
    frame = pd.read_csv(path, usecols=sorted(required | set(optional)))
    frame = frame[frame.income_group.isin(["low", "high"])].copy()
    if frame.empty:
        raise ValueError("The inferred-occupation CSV has no low/high profiles.")
    frame["dist"] = frame.response_distribution.map(
        lambda value: np.asarray(parse_sequence(value), dtype=float)
    )

    rows = []
    for (qkey, label), group in frame.groupby(["qkey", "income_group"], sort=False):
        lengths = {len(value) for value in group.dist}
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent response lengths for {qkey}/{label}: {lengths}")
        rows.append({
            "qkey": qkey,
            "attribute": "socioeco",
            "channel": "inferred",
            "class_label": label,
            "magnitude": np.nan,
            "dist": np.mean(np.stack(group.dist.to_list()), axis=0),
            "n_occupations": group.occupation_cue.nunique(),
        })
    profiles = pd.DataFrame(rows)
    layers = [int(value) for value in parse_sequence(frame.probe_layers.iloc[0])]
    identity = {
        "rows": len(frame),
        "questions": frame.qkey.nunique(),
        "occupations": frame.occupation_cue.nunique(),
        "low_occupations": frame.loc[frame.income_group.eq("low"), "occupation_cue"].nunique(),
        "high_occupations": frame.loc[frame.income_group.eq("high"), "occupation_cue"].nunique(),
        "low_mean_wage": frame.loc[frame.income_group.eq("low"), "median_annual_wage_usd"].mean(),
        "high_mean_wage": frame.loc[frame.income_group.eq("high"), "median_annual_wage_usd"].mean(),
        "model_profile": str(frame.model_profile.dropna().iloc[0]) if "model_profile" in frame and frame.model_profile.notna().any() else "",
        "model_id": str(frame.model_id.dropna().iloc[0]) if "model_id" in frame and frame.model_id.notna().any() else "",
        "selection": str(frame.selection.dropna().iloc[0]) if "selection" in frame and frame.selection.notna().any() else "",
        "layers": layers,
    }
    return profiles, identity


def analyze_probes(path: Path, layers: list[int], chunksize: int):
    """Measure income-probe recovery and target-versus-placebo selectivity."""
    required = {"qkey", "occupation_cue", "income_group"}
    for position in POSITIONS:
        required.add(f"{position}_socioeco_prediction_by_layer")
        for attribute in PROBE_ATTRIBUTES:
            required.add(f"{position}_{attribute}_activation_strength_by_layer")
            required.add(f"{position}_{attribute}_normalized_confidence_by_layer")
    require_columns(path, required)

    cases = []
    layer_totals = defaultdict(lambda: {
        "n": 0, "strength": 0.0, "confidence": 0.0, "target_correct": 0.0,
    })
    seen_after_cue = set()
    for chunk in pd.read_csv(path, usecols=sorted(required), chunksize=chunksize):
        chunk = chunk[chunk.income_group.isin(["low", "high"])]
        for row in chunk.itertuples(index=False):
            values = row._asdict()
            for position in POSITIONS:
                case_key = values["occupation_cue"] if position == "after_occupation" else (
                    values["qkey"], values["occupation_cue"]
                )
                if position == "after_occupation" and case_key in seen_after_cue:
                    continue
                if position == "after_occupation":
                    seen_after_cue.add(case_key)

                predictions = parse_sequence(
                    values[f"{position}_socioeco_prediction_by_layer"]
                )
                if len(predictions) != len(layers):
                    raise ValueError(f"{position} predictions do not match probe_layers.")
                expected = values["income_group"]
                record = {
                    "qkey": values["qkey"],
                    "occupation_cue": values["occupation_cue"],
                    "income_group": expected,
                    "position": position,
                    "socioeco_layer_accuracy": np.mean([value == expected for value in predictions]),
                    "socioeco_majority_prediction": majority_vote(predictions),
                }
                record["socioeco_majority_correct"] = (
                    record["socioeco_majority_prediction"] == expected
                )
                for attribute in PROBE_ATTRIBUTES:
                    strength = np.asarray(parse_sequence(
                        values[f"{position}_{attribute}_activation_strength_by_layer"]
                    ), dtype=float)
                    confidence = np.asarray(parse_sequence(
                        values[f"{position}_{attribute}_normalized_confidence_by_layer"]
                    ), dtype=float)
                    if len(strength) != len(layers) or len(confidence) != len(layers):
                        raise ValueError(f"{position}/{attribute} arrays do not match probe_layers.")
                    record[f"{attribute}_mean_strength"] = float(strength.mean())
                    record[f"{attribute}_mean_confidence"] = float(confidence.mean())
                    for index, layer in enumerate(layers):
                        total = layer_totals[(position, attribute, layer)]
                        total["n"] += 1
                        total["strength"] += float(strength[index])
                        total["confidence"] += float(confidence[index])
                        if attribute == "socioeco":
                            total["target_correct"] += predictions[index] == expected
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
            "socioeco_accuracy": values["target_correct"] / values["n"] if attribute == "socioeco" else np.nan,
        })
    layer_frame = pd.DataFrame(layer_rows).sort_values(["position", "attribute", "layer"])

    summary_rows = []
    for (position, income_group), group in case_frame.groupby(["position", "income_group"], sort=False):
        for attribute in PROBE_ATTRIBUTES:
            strengths = group[f"{attribute}_mean_strength"].to_numpy(float)
            mean, ci = mean_ci(strengths)
            summary_rows.append({
                "position": position,
                "income_group": income_group,
                "attribute": attribute,
                "n_cases": len(group),
                "mean_activation_strength": mean,
                "ci95": ci,
                "mean_normalized_confidence": group[f"{attribute}_mean_confidence"].mean(),
                "socioeco_layer_accuracy": group.socioeco_layer_accuracy.mean() if attribute == "socioeco" else np.nan,
                "socioeco_majority_accuracy": group.socioeco_majority_correct.mean() if attribute == "socioeco" else np.nan,
            })
    summary = pd.DataFrame(summary_rows)

    comparison_rows = []
    for (position, income_group), group in case_frame.groupby(["position", "income_group"], sort=False):
        target = group.socioeco_mean_strength.to_numpy(float)
        for attribute in PROBE_ATTRIBUTES[1:]:
            other = group[f"{attribute}_mean_strength"].to_numpy(float)
            difference = target - other
            p_value = np.nan
            if len(difference) and not np.allclose(difference, 0):
                p_value = float(wilcoxon(difference, alternative="greater").pvalue)
            comparison_rows.append({
                "position": position,
                "income_group": income_group,
                "comparison": f"socioeco vs {attribute}",
                "n_cases": len(group),
                "mean_socioeco_strength": target.mean(),
                "mean_other_strength": other.mean(),
                "mean_difference": difference.mean(),
                "fraction_socioeco_stronger": np.mean(difference > 0),
                "p_value_one_sided": p_value,
            })
    return case_frame, layer_frame, summary, pd.DataFrame(comparison_rows)


def human_income_pair(human, question_meta):
    rows = []
    for qkey, meta in question_meta.items():
        low = human.get((qkey, "INCOME", "Less than $30,000"))
        high = human.get((qkey, "INCOME", "$100,000 or more"))
        if low is None or high is None:
            continue
        try:
            gap = wasserstein_ordinal(low, high, meta["ordinal"])
        except ValueError:
            continue
        rows.append({
            "qkey": qkey,
            "attribute": "socioeco",
            "group_a": "Less than $30,000",
            "group_b": "$100,000 or more",
            "model_class_a": "low",
            "model_class_b": "high",
            "n_human_groups": 2,
            "chance_accuracy": 0.5,
            "human_gap": gap,
        })
    return add_pairwise_quartiles(pd.DataFrame(rows))


def inferred_cross_group(profiles, human, question_meta):
    # Temporarily restrict the global mapping product to the two income endpoints.
    mappings = INCOME_MAPPING
    cross, audit = build_cross_group_matrix(profiles, human, question_meta, mappings)
    cross["regime"] = REGIME_INFERRED
    complete, incomplete = complete_choice_sets(cross, mappings)
    if complete.empty:
        raise ValueError("No complete inferred-occupation choice sets were found.")
    nearest = nearest_neighbor_results(complete)
    accuracy = nearest_accuracy_summary(nearest, mappings)
    return complete, nearest, accuracy, audit, incomplete


def distance_comparison(inferred_cross, baseline, human, question_meta, magnitude):
    baseline = baseline[baseline.attribute.eq("socioeco")].copy()
    baseline_distance, audit = build_distance_table(baseline, human, question_meta)
    frames = []
    for regime, frame in (
        ("Neutral (N=0)", baseline_distance[baseline_distance.channel.eq("steered") & baseline_distance.magnitude.eq(0)]),
        ("Declared", baseline_distance[baseline_distance.channel.eq("declared")]),
        (f"Steered N={magnitude:g}", baseline_distance[baseline_distance.channel.eq("steered") & baseline_distance.magnitude.eq(magnitude)]),
    ):
        part = frame[["qkey", "class_label", "wasserstein"]].copy()
        part["regime"] = regime
        frames.append(part)
    inferred = inferred_cross[inferred_cross.is_own_group][
        ["qkey", "model_class", "wasserstein"]
    ].rename(columns={"model_class": "class_label"})
    inferred["regime"] = REGIME_INFERRED
    frames.append(inferred)
    distances = pd.concat(frames, ignore_index=True)

    neutral = distances[distances.regime.eq("Neutral (N=0)")][
        ["qkey", "class_label", "wasserstein"]
    ].rename(columns={"wasserstein": "neutral_wasserstein"})
    paired = distances.merge(neutral, on=["qkey", "class_label"], how="inner")
    paired["delta_vs_neutral"] = paired.wasserstein - paired.neutral_wasserstein
    rows = []
    for label in ("low", "high", "all"):
        subset = paired if label == "all" else paired[paired.class_label.eq(label)]
        for regime, group in subset.groupby("regime", sort=False):
            units = group.groupby("qkey", as_index=False)[["wasserstein", "delta_vs_neutral"]].mean() if label == "all" else group
            mean_w, ci_w = mean_ci(units.wasserstein)
            mean_delta, ci_delta = mean_ci(units.delta_vs_neutral)
            p_value = np.nan
            if len(units) and not np.allclose(units.delta_vs_neutral, 0):
                p_value = float(wilcoxon(units.delta_vs_neutral).pvalue)
            rows.append({
                "class_label": label, "regime": regime, "n": len(units),
                "mean_wasserstein": mean_w, "wasserstein_ci95": ci_w,
                "mean_delta_vs_neutral": mean_delta, "delta_ci95": ci_delta,
                "paired_wilcoxon_p": p_value,
            })
    return paired, pd.DataFrame(rows), audit


def directional_q4(directional_rows, directional):
    summary = directional[
        directional.quartile.eq("Q4") & directional.regime.eq(REGIME_INFERRED)
    ].copy()
    raw = directional_rows[
        directional_rows.quartile.eq("Q4") & directional_rows.regime.eq(REGIME_INFERRED)
    ]
    cis = []
    for target, group in raw.groupby("target_human_group"):
        _, ci = mean_ci(group.difference)
        cis.append({"target_human_group": target, "difference_ci95": ci})
    return summary.merge(pd.DataFrame(cis), on="target_human_group", how="left")


def plot_probe_accuracy(layer_summary, out_path):
    values = layer_summary[layer_summary.attribute.eq("socioeco")]
    fig, ax = plt.subplots(figsize=(9, 5))
    for position, group in values.groupby("position", sort=False):
        label = "After occupation" if position == "after_occupation" else "Final prompt token"
        ax.plot(group.layer, group.socioeco_accuracy, marker="o", ms=3, label=label)
    ax.axhline(1 / 3, color="gray", ls=":", label="Three-class chance")
    ax.set(xlabel="Layer", ylabel="Income-group classification accuracy",
           title="Does the socioeconomic probe recover the occupation wage group?", ylim=(0, 1))
    ax.grid(alpha=.15); ax.legend(); fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def plot_probe_strength(cases, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = ["tab:purple", "tab:orange", "tab:blue", "tab:green"]
    for ax, position in zip(axes, POSITIONS):
        group = cases[cases.position.eq(position)]
        rows = []
        for attribute in PROBE_ATTRIBUTES:
            mean, ci = mean_ci(group[f"{attribute}_mean_strength"])
            rows.append((attribute, mean, ci))
        values = pd.DataFrame(
            rows, columns=["attribute", "mean_activation_strength", "ci95"]
        ).set_index("attribute")
        x = np.arange(len(values))
        ax.bar(x, values.mean_activation_strength, yerr=values.ci95, capsize=3, color=colors)
        ax.set_xticks(x, values.index, rotation=20)
        ax.set_title("After occupation" if position == "after_occupation" else "Final prompt token")
        ax.grid(axis="y", alpha=.15)
    axes[0].set_ylabel("Chance-corrected probe selectivity (95% CI)")
    fig.suptitle("Socioeconomic selectivity versus placebo demographic probes")
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def plot_distance(summary, magnitude, out_path):
    regimes = ["Declared", f"Steered N={magnitude:g}", REGIME_INFERRED]
    labels = ["low", "high", "all"]
    x, width = np.arange(3), .24
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, regime in enumerate(regimes):
        values = summary[summary.regime.eq(regime)].set_index("class_label").reindex(labels)
        color = REGIME_COLORS["Steered"] if regime.startswith("Steered") else REGIME_COLORS[regime]
        ax.bar(x + (index - 1) * width, values.mean_delta_vs_neutral, width,
               yerr=values.delta_ci95, capsize=3, label=regime, color=color)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, ["Lower-wage", "Higher-wage", "Overall"])
    ax.set(ylabel=r"$\Delta W$ relative to neutral (95% CI)",
           title="Distance to the matching human income group")
    ax.text(.01, .02, "negative = closer", transform=ax.transAxes, fontsize=9)
    ax.grid(axis="y", alpha=.15); ax.legend(); fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def plot_confusion(nearest, out_path):
    labels = ["Less than $30,000", "$100,000 or more"]
    plotted = nearest.assign(prediction=np.where(nearest.n_tied_nearest.gt(1), "TIE", nearest.predicted_human_group))
    if plotted.prediction.eq("TIE").any():
        labels.append("TIE")
    matrix = pd.crosstab(plotted.model_class, plotted.prediction).reindex(index=["low", "high"], columns=labels, fill_value=0)
    fractions = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(7, 4.8)); image = ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues")
    for row in range(2):
        for column in range(len(labels)):
            value = fractions.iloc[row, column]
            ax.text(column, row, "NA" if not np.isfinite(value) else f"{value:.2f}", ha="center", va="center",
                    color="white" if np.isfinite(value) and value > .55 else "black")
    ax.set_xticks(range(len(labels)), [label.replace("$", r"\$") for label in labels], rotation=15)
    ax.set_yticks(range(2), ["lower-wage profile", "higher-wage profile"])
    ax.set(xlabel="Nearest human group", ylabel="Occupation-conditioned profile",
           title="Cross-group nearest-neighbor classification")
    fig.colorbar(image, ax=ax, label="Row fraction"); fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def plot_quartile(accuracy, out_path):
    order = ["Q1", "Q2", "Q3", "Q4"]
    values = accuracy[accuracy.regime.eq(REGIME_INFERRED) & accuracy.quartile.isin(order)].set_index("quartile").reindex(order)
    lower = values.lift_over_chance - values.lift_ci95_low
    upper = values.lift_ci95_high - values.lift_over_chance
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(order, values.lift_over_chance, yerr=np.vstack([lower, upper]), marker="o", capsize=4, color=REGIME_COLORS[REGIME_INFERRED])
    ax.axhline(0, color="black", lw=1)
    ax.set(xlabel="Human income-gap quartile", ylabel="Nearest-group lift over chance",
           title="Recovery as human income groups diverge")
    ax.grid(alpha=.15); fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def plot_odds_ratio(trends, out_path):
    row = trends[trends.regime.eq(REGIME_INFERRED)].iloc[0]
    odds, low, high = row.odds_ratio_per_gap_sd, row.odds_ratio_ci95_low, row.odds_ratio_ci95_high
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.errorbar(odds, 0, xerr=np.asarray([[odds-low], [high-odds]]), marker="o", ms=8, capsize=4, color=REGIME_COLORS[REGIME_INFERRED])
    ax.axvline(1, color="black", lw=1, ls=":"); ax.set_xscale("log")
    ax.set_yticks([0], [REGIME_INFERRED]); ax.set_xlabel("Odds ratio per SD of human income gap (95% CI)")
    ax.set_title("Continuous trend in nearest-group recovery"); ax.grid(axis="x", alpha=.15)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def plot_directional(q4, out_path):
    order = ["Less than $30,000", "$100,000 or more"]
    values = q4.set_index("target_human_group").reindex(order)
    x = np.arange(2); fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, values.mean_difference, yerr=values.difference_ci95, capsize=4, color=["tab:blue", "tab:red"])
    ax.axhline(0, color="black", lw=1); ax.set_xticks(x, ["< $30,000", "$100,000+"])
    ax.set(ylabel="Own − competitor Wasserstein distance (95% CI)",
           title="Directional specificity in high-human-gap questions (Q4)")
    ax.text(.01, .02, "negative = correct direction", transform=ax.transAxes, fontsize=9)
    ax.grid(axis="y", alpha=.15); fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def write_report(path, *, input_path, baseline_path, identity, magnitude, fallback,
                 probe_summary, activation, distance, nearest_accuracy,
                 quartile_accuracy, trends, q4, audit, incomplete, files):
    probe = probe_summary[probe_summary.attribute.eq("socioeco")]
    after = probe[probe.position.eq("after_occupation")]
    final = probe[probe.position.eq("final_prompt")]
    nearest_row = nearest_accuracy.iloc[0]
    q4_accuracy = quartile_accuracy[quartile_accuracy.regime.eq(REGIME_INFERRED) & quartile_accuracy.quartile.eq("Q4")].iloc[0]
    trend = trends[trends.regime.eq(REGIME_INFERRED)].iloc[0]
    distance_all = distance[distance.class_label.eq("all")]
    inferred_distance = distance_all[distance_all.regime.eq(REGIME_INFERRED)].iloc[0]
    weakest = activation.loc[activation.mean_difference.idxmin()]
    lines = [
        f"# Inferred income from occupations: {identity['model_profile'] or 'model'}", "",
        f"- Input: `{input_path.resolve()}`",
        f"- Baseline: `{baseline_path.resolve()}`",
        f"- Model: `{identity['model_id'] or identity['model_profile']}`",
        f"- Questions: `{identity['questions']}`; occupations: `{identity['occupations']}` "
        f"({identity['low_occupations']} lower-wage, {identity['high_occupations']} higher-wage)",
        f"- Mean occupational wages: `${identity['low_mean_wage']:,.0f}` and `${identity['high_mean_wage']:,.0f}`",
        f"- Steering comparison magnitude: `{magnitude:g}`" + (" (nearest available)" if fallback else ""), "",
        "Answer distributions are averaged across occupations within wage group before comparison with humans. "
        "The labels are relative occupational-wage groups; they are not direct observations of the OpinionQA household-income endpoints.", "",
        "## Headline", "",
        f"Across-layer majority accuracy is **{after.socioeco_majority_accuracy.mean():.1%} after the occupation cue** "
        f"and **{final.socioeco_majority_accuracy.mean():.1%} at the final prompt token**. The weakest target-probe advantage is "
        f"**{weakest.mean_difference:+.3f}** ({weakest.comparison}, {weakest.position}, {weakest.income_group}).", "",
        f"At the answer level, occupation inference changes matching-group distance by **Delta W={inferred_distance.mean_delta_vs_neutral:+.3f}**. "
        f"Nearest-group lift is **{nearest_row.lift_over_chance:+.3f}** overall and **{q4_accuracy.lift_over_chance:+.3f}** in Q4. "
        f"The continuous trend is **OR={fmt(trend.odds_ratio_per_gap_sd)}** per SD of human divergence.", "",
        "## 1. Socioeconomic probe recovery", "",
        "| Position | Wage group | Layer accuracy | Majority accuracy | Cases |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in probe.itertuples(index=False):
        lines.append(f"| {row.position} | {row.income_group} | {row.socioeco_layer_accuracy:.3f} | {row.socioeco_majority_accuracy:.3f} | {row.n_cases} |")
    lines += ["", "## 2. Socioeconomic probe versus placebo probes", "",
              "Positive differences favor the socioeconomic probe. Chance-corrected selectivity is a relative diagnostic; independently trained probes may differ in calibration.", "",
              "| Position | Wage group | Comparison | Mean difference | Socioeconomic stronger | p (one-sided) |",
              "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in activation.itertuples(index=False):
        lines.append(f"| {row.position} | {row.income_group} | {row.comparison} | {row.mean_difference:+.3f} | {row.fraction_socioeco_stronger:.3f} | {fmt_p(row.p_value_one_sided)} |")
    lines += ["", "## 3. Distance to matching human income group", "",
              "Negative Delta W means closer than the neutral model.", "",
              "| Regime | Mean W | Delta W | 95% CI of Delta | paired p |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in distance_all.itertuples(index=False):
        lines.append(f"| {row.regime} | {row.mean_wasserstein:.3f} | {row.mean_delta_vs_neutral:+.3f} | ±{row.delta_ci95:.3f} | {fmt_p(row.paired_wilcoxon_p)} |")
    lines += ["", "## 4. Cross-group specificity", "",
              f"Strict nearest-group accuracy is **{nearest_row.strict_accuracy:.3f}** and fractional-tie accuracy is **{nearest_row.fractional_tie_accuracy:.3f}**, against chance **0.500** (lift **{nearest_row.lift_over_chance:+.3f}**).", "",
              "## 5. Human-gap trend", "",
              f"Q4 fractional accuracy is **{q4_accuracy.fractional_accuracy:.3f}**, lift **{q4_accuracy.lift_over_chance:+.3f}**, bootstrap 95% CI **[{q4_accuracy.lift_ci95_low:+.3f}, {q4_accuracy.lift_ci95_high:+.3f}]**.", "",
              f"OR per gap SD is **{fmt(trend.odds_ratio_per_gap_sd)}** (95% CI [{fmt(trend.odds_ratio_ci95_low)}, {fmt(trend.odds_ratio_ci95_high)}], one-sided cluster-robust p={fmt_p(trend.cluster_robust_p_slope_le_zero)}).", "",
              "## 6. Directional test in Q4", "",
              "Negative own-minus-competitor values indicate the correct direction.", "",
              "| Human target | n | Mean difference | Win rate | Holm p |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in q4.itertuples(index=False):
        lines.append(f"| {row.target_human_group} | {row.n_questions} | {row.mean_difference:+.3f} | {row.fractional_win_rate:.3f} | {fmt_p(row.p_value_holm)} |")
    lines += ["", "## Audit", "",
              f"- Cross-group cells: `{audit['cross_group_rows']}`",
              f"- Incomplete choice sets excluded: `{incomplete}`",
              f"- Model trailing values removed: `{audit['trailing_model_values_removed']}`",
              f"- Human trailing values removed: `{audit['trailing_human_values_removed']}`", "",
              "## Output files", ""]
    lines += [f"- [{label}]({file.name})" for label, file in files.items()]
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
    for input_path in (args.input, args.opinionqa, args.qkey_dict):
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
    if args.bootstrap_iterations < 100 or args.chunksize < 1:
        raise ValueError("Use at least 100 bootstrap iterations and a positive chunksize.")

    profiles, identity = load_inferred_profiles(args.input)
    baseline_path = find_baseline(args.baseline_results, identity["model_profile"])
    baseline = load_model(baseline_path)
    baseline_label, baseline_model_id = identify_model(baseline, baseline_path)
    if identity["model_id"] and baseline_model_id and identity["model_id"] != baseline_model_id:
        raise ValueError(f"Model mismatch: {identity['model_id']!r} vs {baseline_model_id!r}.")
    magnitude, fallback = select_magnitude(
        baseline, args.comparison_magnitude, identity["model_profile"] or baseline_label
    )
    human = load_human(args.opinionqa)
    question_meta = load_question_metadata(args.qkey_dict)
    cross, nearest, nearest_accuracy, audit, incomplete = inferred_cross_group(profiles, human, question_meta)
    human_pairs = human_income_pair(human, question_meta)
    predictions = nearest.merge(
        human_pairs[["qkey", "attribute", "group_a", "group_b", "model_class_a", "model_class_b", "n_human_groups", "chance_accuracy", "human_gap", "quartile"]],
        on=["qkey", "attribute"], how="inner",
    )
    quartile_accuracy = accuracy_summary(predictions, iterations=args.bootstrap_iterations, seed=args.seed)
    trends = trend_tests(predictions)
    trends["odds_ratio_ci95_low"] = np.exp(trends.slope_ci95_low)
    trends["odds_ratio_ci95_high"] = np.exp(trends.slope_ci95_high)
    directional_rows = build_directional_rows(cross, human_pairs)
    directional = directional_summary(directional_rows)
    q4 = directional_q4(directional_rows, directional)
    paired_distance, distance_summary, _ = distance_comparison(cross, baseline, human, question_meta, magnitude)
    cases, layers, probe_summary, activation = analyze_probes(args.input, identity["layers"], args.chunksize)

    model_label = safe_label(identity["model_profile"] or baseline_label)
    out_dir = args.out_dir or REPO_ROOT / "analyses/inferred_income_occupation_reports" / model_label
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "Probe cases": out_dir / "probe_case_summary.csv",
        "Probe accuracy by occupation": out_dir / "probe_accuracy_by_occupation.csv",
        "Probe results by layer": out_dir / "probe_by_layer.csv",
        "Probe summary": out_dir / "probe_summary.csv",
        "Probe placebo comparisons": out_dir / "probe_activation_comparisons.csv",
        "Inferred profiles": out_dir / "inferred_profiles.csv",
        "Matching-group distances": out_dir / "distance_to_matching_human.csv",
        "Distance summary": out_dir / "distance_vs_neutral_summary.csv",
        "Cross-group distances": out_dir / "cross_group_distances.csv",
        "Nearest-group predictions": out_dir / "nearest_group_predictions.csv",
        "Nearest-group accuracy": out_dir / "nearest_group_accuracy.csv",
        "Accuracy by human-gap quartile": out_dir / "accuracy_by_human_gap_quartile.csv",
        "Continuous human-gap trends": out_dir / "accuracy_gap_trends.csv",
        "Directional rows": out_dir / "directional_by_question.csv",
        "Directional Q4 tests": out_dir / "directional_tests_q4.csv",
        "Probe accuracy plot": out_dir / "socioeco_probe_accuracy_by_layer.png",
        "Probe placebo plot": out_dir / "probe_selectivity_by_attribute.png",
        "Distance plot": out_dir / "distance_delta_vs_neutral.png",
        "Nearest-group confusion": out_dir / "nearest_group_confusion.png",
        "Human-gap recovery": out_dir / "lift_by_human_gap_quartile.png",
        "Human-gap odds ratio": out_dir / "human_gap_odds_ratio.png",
        "Directional Q4": out_dir / "directional_specificity_q4.png",
    }
    cases.to_csv(files["Probe cases"], index=False)
    occupation_summary = cases.groupby(
        ["position", "occupation_cue", "income_group"], as_index=False
    ).agg(
        n_cases=("qkey", "size"),
        mean_layer_accuracy=("socioeco_layer_accuracy", "mean"),
        majority_accuracy=("socioeco_majority_correct", "mean"),
        mean_socioeco_strength=("socioeco_mean_strength", "mean"),
        mean_education_strength=("education_mean_strength", "mean"),
        mean_gender_strength=("gender_mean_strength", "mean"),
        mean_age_strength=("age_mean_strength", "mean"),
    )
    occupation_summary.to_csv(files["Probe accuracy by occupation"], index=False)
    layers.to_csv(files["Probe results by layer"], index=False)
    probe_summary.to_csv(files["Probe summary"], index=False)
    activation.to_csv(files["Probe placebo comparisons"], index=False)
    serializable = profiles.copy()
    serializable["response_distribution"] = serializable.dist.map(lambda values: json.dumps([round(float(value), 8) for value in values]))
    serializable.drop(columns="dist").to_csv(files["Inferred profiles"], index=False)
    paired_distance.to_csv(files["Matching-group distances"], index=False)
    distance_summary.to_csv(files["Distance summary"], index=False)
    cross.to_csv(files["Cross-group distances"], index=False)
    nearest.to_csv(files["Nearest-group predictions"], index=False)
    nearest_accuracy.to_csv(files["Nearest-group accuracy"], index=False)
    quartile_accuracy.to_csv(files["Accuracy by human-gap quartile"], index=False)
    trends.to_csv(files["Continuous human-gap trends"], index=False)
    directional_rows.to_csv(files["Directional rows"], index=False)
    q4.to_csv(files["Directional Q4 tests"], index=False)
    plot_probe_accuracy(layers, files["Probe accuracy plot"])
    plot_probe_strength(cases, files["Probe placebo plot"])
    plot_distance(distance_summary, magnitude, files["Distance plot"])
    plot_confusion(nearest, files["Nearest-group confusion"])
    plot_quartile(quartile_accuracy, files["Human-gap recovery"])
    plot_odds_ratio(trends, files["Human-gap odds ratio"])
    plot_directional(q4, files["Directional Q4"])
    report = out_dir / "report.md"
    write_report(report, input_path=args.input, baseline_path=baseline_path,
                 identity=identity, magnitude=magnitude, fallback=fallback,
                 probe_summary=probe_summary, activation=activation,
                 distance=distance_summary, nearest_accuracy=nearest_accuracy,
                 quartile_accuracy=quartile_accuracy, trends=trends, q4=q4,
                 audit=audit, incomplete=incomplete, files=files)
    print(f"[ok] model={model_label} comparison_magnitude={magnitude:g}")
    print(f"[ok] report={report.resolve()}")


if __name__ == "__main__":
    main()
