#!/usr/bin/env python
"""Generate held-out paper figures after tuning steering magnitude.

This script only reads existing analysis CSVs. It uses the qkey split and the
model-specific magnitudes written by select_steering_magnitude.py; no language
model is loaded or evaluated here.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np
import pandas as pd

from analyze_specificity_by_human_gap import (
    add_pairwise_quartiles,
    build_directional_rows,
    canonicalize_human_pairs,
    directional_summary,
    mapped_pairs,
    trend_tests,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ["qwen", "llama", "mistral"]
MODEL_LABELS = {"qwen": "Qwen", "llama": "Llama", "mistral": "Mistral"}
MODEL_MARKERS = {"qwen": "o", "llama": "s", "mistral": "^"}
ATTRIBUTE_ORDER = ["gender", "age", "education", "socioeco"]
ATTRIBUTE_LABELS = {
    "gender": "Gender", "age": "Age", "education": "Education",
    "socioeco": "Socioeconomic status",
}
DISTANCE_SUFFIXES = {"qwen": "", "llama": "_llama", "mistral": "_mistral"}
SPECIFICITY_SUFFIXES = {"qwen": "", "llama": "_llama", "mistral": ""}
COLORS = {"Declared": "#E6862A", "Steered": "#2878B5"}

# The contrasts used in the main text. Education's extreme contrast is used
# in Figures 2 and 4; Figure 3 intentionally shows all three education pairs.
PAPER_PAIRS = {
    "gender": ("Male", "Female"),
    "age": ("30-49", "65+"),
    "education": ("Less than high school", "College graduate/some postgrad"),
    "socioeco": ("Less than $30,000", "$100,000 or more"),
}


def resolve_csv(directory: Path, stem: str, suffix: str = "") -> Path:
    candidates = [directory / f"{stem}{suffix}.csv", directory / f"{stem}.csv"]
    for path in dict.fromkeys(candidates):
        if path.is_file():
            return path
    raise FileNotFoundError("Expected one of: " + ", ".join(map(str, candidates)))


def load_inputs(analyses_dir: Path, selection_dir: Path):
    split = pd.read_csv(selection_dir / "qkey_split.csv")
    selected_table = pd.read_csv(selection_dir / "selected_magnitudes.csv")
    selected = dict(zip(selected_table.model, selected_table.selected_magnitude))
    missing_models = set(MODEL_ORDER) - set(selected)
    if missing_models:
        raise ValueError(f"Magnitude selection is missing models: {sorted(missing_models)}")

    distances, predictions, cross = {}, {}, {}
    for model in MODEL_ORDER:
        distances[model] = pd.read_csv(resolve_csv(
            analyses_dir / "model_reports" / model,
            "distance_to_human", DISTANCE_SUFFIXES[model],
        ))
        predictions[model] = pd.read_csv(resolve_csv(
            analyses_dir / "human_gap_specificity_reports" / model,
            "predictions_by_human_gap", SPECIFICITY_SUFFIXES[model],
        ))
        cross[model] = pd.read_csv(resolve_csv(
            analyses_dir / "cross_group_reports" / model,
            "cross_group_distances",
        ))
        for frame in (distances[model], predictions[model], cross[model]):
            if "magnitude" in frame:
                frame["magnitude"] = pd.to_numeric(frame.magnitude, errors="coerce")
    human_gap = pd.read_csv(
        analyses_dir / "human_separability" / "human_separability_by_question.csv"
    )
    evaluation = set(split.loc[split.split.eq("evaluation"), "qkey"])
    return selected, evaluation, distances, predictions, cross, human_gap


def condition_rows(frame: pd.DataFrame, magnitude: float, regime: str):
    if regime == "Declared":
        return frame[frame.channel.eq("declared")]
    if regime == "Steered":
        return frame[frame.channel.eq("steered") & frame.magnitude.eq(magnitude)]
    if regime == "Neutral":
        return frame[frame.channel.eq("steered") & frame.magnitude.eq(0)]
    raise ValueError(regime)


def bootstrap_mean(values_by_qkey: pd.Series, *, iterations: int,
                   rng: np.random.Generator):
    values = values_by_qkey.dropna().to_numpy(dtype=float)
    if not len(values):
        return np.nan, np.nan, np.nan
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    samples = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def pair_filter(frame: pd.DataFrame, attribute: str):
    group_a, group_b = PAPER_PAIRS[attribute]
    return frame[
        frame.attribute.eq(attribute)
        & frame.group_a.eq(group_a)
        & frame.group_b.eq(group_b)
    ]


def metric_reversal_data(selected, evaluation, distances, predictions,
                         *, iterations: int, seed: int):
    rng = np.random.default_rng(seed)
    left_rows, right_rows = [], []
    keys = ["qkey", "attribute", "class_label"]
    for model in MODEL_ORDER:
        magnitude = float(selected[model])
        distance = distances[model]
        distance = distance[distance.qkey.isin(evaluation)]
        neutral = condition_rows(distance, magnitude, "Neutral")[keys + ["wasserstein"]]
        for attribute in ATTRIBUTE_ORDER:
            baseline = neutral[neutral.attribute.eq(attribute)]
            for regime in ("Declared", "Steered"):
                current = condition_rows(distance, magnitude, regime)
                current = current[current.attribute.eq(attribute)][keys + ["wasserstein"]]
                paired = current.merge(baseline, on=keys, suffixes=("_condition", "_neutral"))
                paired["delta"] = paired.wasserstein_condition - paired.wasserstein_neutral
                by_qkey = paired.groupby("qkey").delta.mean()
                mean, low, high = bootstrap_mean(by_qkey, iterations=iterations, rng=rng)
                left_rows.append({
                    "model": model, "attribute": attribute, "regime": regime,
                    "selected_magnitude": magnitude, "n_questions": len(by_qkey),
                    "mean": mean, "ci95_low": low, "ci95_high": high,
                })

        pred = predictions[model]
        pred = pred[pred.qkey.isin(evaluation)]
        for attribute in ATTRIBUTE_ORDER:
            pair = pair_filter(pred, attribute)
            pair = pair[pair.quartile.eq("Q4")]
            for regime in ("Declared", "Steered"):
                current = condition_rows(pair, magnitude, regime)
                by_qkey = current.groupby("qkey").fractional_credit.mean()
                chance = float(current.chance_accuracy.iloc[0]) if len(current) else np.nan
                lift = by_qkey - chance
                mean, low, high = bootstrap_mean(lift, iterations=iterations, rng=rng)
                right_rows.append({
                    "model": model, "attribute": attribute, "regime": regime,
                    "selected_magnitude": magnitude, "quartile": "Q4",
                    "n_questions": len(by_qkey), "chance_accuracy": chance,
                    "mean": mean, "ci95_low": low, "ci95_high": high,
                })
    return pd.DataFrame(left_rows), pd.DataFrame(right_rows)


def draw_metric_reversal(left: pd.DataFrame, right: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 7.2), sharey=True)
    rows = [(model, attribute) for model in MODEL_ORDER for attribute in ATTRIBUTE_ORDER]
    y = np.arange(len(rows))[::-1]
    offsets = {"Declared": .14, "Steered": -.14}
    for ax, data, title, xlabel in (
        (axes[0], left, "A. Proximity to the matching human group",
         r"$\Delta W = W_{regime}-W_{neutral}$" + "\ncloser to matching group  ←"),
        (axes[1], right, "B. Demographic-group recovery",
         "Nearest-group lift over chance in Q4\n→  better demographic recovery"),
    ):
        ax.axvline(0, color="0.35", lw=1, ls="--")
        for row_index, (model, attribute) in enumerate(rows):
            for regime in ("Declared", "Steered"):
                value = data[
                    data.model.eq(model) & data.attribute.eq(attribute)
                    & data.regime.eq(regime)
                ]
                if value.empty:
                    continue
                record = value.iloc[0]
                ax.errorbar(
                    record["mean"], y[row_index] + offsets[regime],
                    xerr=[[record["mean"] - record.ci95_low],
                          [record.ci95_high - record["mean"]]],
                    fmt=MODEL_MARKERS[model], ms=6.3, capsize=2.5,
                    color=COLORS[regime], mec=COLORS[regime], lw=1.4,
                )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=.18)
    labels = [f"{MODEL_LABELS[m]}  ·  {ATTRIBUTE_LABELS[a]}" for m, a in rows]
    axes[0].set_yticks(y, labels)
    axes[0].tick_params(axis="y", labelsize=9)
    legend = [Line2D([0], [0], marker="o", color=COLORS[r], lw=2,
                     label=r, markersize=6) for r in ("Declared", "Steered")]
    fig.legend(handles=legend, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, .985))
    fig.suptitle("Metric reversal: proximity improves without reliable group recovery",
                 y=1.02, fontsize=15)
    fig.text(.5, .012,
             "Steering typically wins on proximity   ←   metric boundary   →   declaration typically wins on specificity",
             ha="center", fontsize=9.3, fontweight="bold", color="0.2")
    fig.text(.5, -.006,
             "Magnitudes selected on tuning questions; estimates and 95% bootstrap CIs use evaluation questions only.",
             ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0, .055, 1, .955), w_pad=2.4)
    for extension in ("png", "pdf"):
        fig.savefig(out_dir / f"figure5_metric_reversal_simplified_heldout.{extension}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def heldout_trends(selected, evaluation, predictions):
    frames = []
    for model in MODEL_ORDER:
        magnitude = float(selected[model])
        data = predictions[model]
        data = data[data.qkey.isin(evaluation)]
        data = pd.concat([
            condition_rows(data, magnitude, "Declared"),
            condition_rows(data, magnitude, "Steered"),
        ], ignore_index=True)
        result = trend_tests(data)
        result.insert(0, "model", model)
        result["display_regime"] = np.where(result.channel.eq("declared"), "Declared", "Steered")
        result["selected_magnitude"] = magnitude
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def short_pair(attribute: str, group_a: str, group_b: str):
    replacements = {
        "Less than high school": "<HS",
        "High school graduate": "HS graduate",
        "College graduate/some postgrad": "College+",
        "Less than $30,000": "<30k",
        "$100,000 or more": "100k+",
    }
    return f"{replacements.get(group_a, group_a)} vs {replacements.get(group_b, group_b)}"


def draw_or_forest(trends: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.6), sharex=True)
    axes = axes.ravel()
    finite = trends.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["slope_ci95_low", "slope_ci95_high"]
    )
    global_low = np.exp(finite.slope_ci95_low).min()
    global_high = np.exp(finite.slope_ci95_high).max()
    margin = .12
    log_low, log_high = np.log(global_low), np.log(global_high)
    xlim = (np.exp(log_low - margin * (log_high - log_low)),
            np.exp(log_high + margin * (log_high - log_low)))
    regime_dodge = {"Declared": .13, "Steered": -.13}

    for ax, attribute in zip(axes, ATTRIBUTE_ORDER):
        subset = trends[trends.attribute.eq(attribute)].copy()
        pair_order = list(dict.fromkeys(zip(subset.group_a, subset.group_b)))
        rows = [(a, b, model) for a, b in pair_order for model in MODEL_ORDER]
        positions = np.arange(len(rows))[::-1]
        ax.axvline(1, color="0.35", ls="--", lw=1)
        for index, (group_a, group_b, model) in enumerate(rows):
            for regime in ("Declared", "Steered"):
                value = subset[
                    subset.group_a.eq(group_a) & subset.group_b.eq(group_b)
                    & subset.model.eq(model) & subset.display_regime.eq(regime)
                ]
                if value.empty:
                    continue
                record = value.iloc[0]
                odds = float(record.odds_ratio_per_gap_sd)
                low, high = np.exp([record.slope_ci95_low, record.slope_ci95_high])
                significant = record.cluster_robust_p_slope_le_zero < .05
                ax.errorbar(
                    odds, positions[index] + regime_dodge[regime],
                    xerr=[[odds - low], [high - odds]], fmt=MODEL_MARKERS[model],
                    color=COLORS[regime], mec=COLORS[regime],
                    mfc=COLORS[regime] if significant else "white",
                    ms=6.5, mew=1.3, capsize=2.2, lw=1.2,
                )
        labels = [
            f"{MODEL_LABELS[model]} — {short_pair(attribute, a, b)}"
            for a, b, model in rows
        ]
        ax.set_yticks(positions, labels, fontsize=8.2)
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ticks = [value for value in (.8, .9, 1.0, 1.1, 1.2, 1.3, 1.4)
                 if xlim[0] <= value <= xlim[1]]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{value:g}" for value in ticks]))
        ax.tick_params(axis="x", which="both", labelbottom=True)
        ax.grid(axis="x", which="both", alpha=.16)
        ax.set_title(ATTRIBUTE_LABELS[attribute], loc="left", fontweight="bold")
        ax.set_xlabel("Odds ratio per 1 SD increase in human gap")
    legend = [
        Line2D([0], [0], marker="o", color=COLORS[regime], lw=1.5,
               markerfacecolor=COLORS[regime], label=regime)
        for regime in ("Declared", "Steered")
    ]
    legend.extend([
        Line2D([0], [0], marker="o", color="0.25", lw=0, mfc="0.25", label="p < .05"),
        Line2D([0], [0], marker="o", color="0.25", lw=0, mfc="white", label="n.s."),
    ])
    fig.legend(handles=legend, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, .985))
    fig.suptitle("Declared profiles track human divergence; steering generally does not",
                 fontsize=15, y=1.02)
    fig.text(.5, .008,
             "Cluster-robust 95% CIs; one-sided directional significance. Evaluation questions only.",
             ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0, .03, 1, .95), h_pad=2.0, w_pad=1.6)
    for extension in ("png", "pdf"):
        fig.savefig(out_dir / f"figure3_or_forest_heldout.{extension}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def heldout_directional(selected, evaluation, cross, human_gap):
    pairs = mapped_pairs()
    human_pairs = add_pairwise_quartiles(canonicalize_human_pairs(human_gap, pairs))
    frames = []
    for model in MODEL_ORDER:
        magnitude = float(selected[model])
        directional = build_directional_rows(cross[model], human_pairs)
        directional = directional[directional.qkey.isin(evaluation)]
        directional = pd.concat([
            condition_rows(directional, magnitude, "Declared"),
            condition_rows(directional, magnitude, "Steered"),
        ], ignore_index=True)
        summary = directional_summary(directional)
        summary.insert(0, "model", model)
        summary["display_regime"] = np.where(summary.channel.eq("declared"), "Declared", "Steered")
        summary["selected_magnitude"] = magnitude
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def stars(p_value: float):
    if not np.isfinite(p_value):
        return ""
    if p_value < .001:
        return "***"
    if p_value < .01:
        return "**"
    if p_value < .05:
        return "*"
    return ""


def short_target(value: str):
    return {
        "Less than high school": "<HS",
        "College graduate/some postgrad": "College+",
        "Less than $30,000": "<30k",
        "$100,000 or more": "100k+",
    }.get(value, value)


def draw_directional_heatmap(summary: pd.DataFrame, out_dir: Path):
    data = summary[summary.quartile.eq("Q4")].copy()
    data = pd.concat([pair_filter(data, attribute) for attribute in ATTRIBUTE_ORDER],
                     ignore_index=True)
    rows = [(model, attribute) for model in MODEL_ORDER for attribute in ATTRIBUTE_ORDER]
    values = np.full((len(rows), 2, 2), np.nan)
    annotations = np.full((len(rows), 2, 2), "", dtype=object)
    regimes = ["Declared", "Steered"]
    for row_index, (model, attribute) in enumerate(rows):
        group_a, group_b = PAPER_PAIRS[attribute]
        for regime_index, regime in enumerate(regimes):
            for pole_index, target in enumerate((group_a, group_b)):
                record = data[
                    data.model.eq(model) & data.attribute.eq(attribute)
                    & data.display_regime.eq(regime)
                    & data.target_human_group.eq(target)
                ]
                if record.empty:
                    continue
                record = record.iloc[0]
                values[row_index, regime_index, pole_index] = record.mean_difference
                annotations[row_index, regime_index, pole_index] = (
                    f"{short_target(target)}\n{record.mean_difference:+.2f}{stars(record.p_value_holm)}"
                )
    finite = np.abs(values[np.isfinite(values)])
    limit = float(finite.max()) if len(finite) else 1.0
    limit = max(limit, .05)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 8.7), sharey=True)
    images = []
    for regime_index, (ax, regime) in enumerate(zip(axes, regimes)):
        image = ax.imshow(values[:, regime_index, :], cmap="RdBu_r", aspect="auto",
                          vmin=-limit, vmax=limit)
        images.append(image)
        ax.set_xticks([0, 1], ["Pole A", "Pole B"])
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_yticks(np.arange(len(rows)), [
            f"{MODEL_LABELS[m]}  ·  {ATTRIBUTE_LABELS[a]}" for m, a in rows
        ], fontsize=9)
        ax.set_title(regime, fontweight="bold")
        for row_index in range(len(rows)):
            for pole_index in range(2):
                value = values[row_index, regime_index, pole_index]
                if not np.isfinite(value):
                    continue
                text_color = "white" if abs(value) > .58 * limit else "black"
                ax.text(pole_index, row_index,
                        annotations[row_index, regime_index, pole_index],
                        ha="center", va="center", fontsize=7.5, color=text_color)
        ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", bottom=False, left=False)
    colorbar_axis = fig.add_axes([.30, .067, .48, .022])
    cbar = fig.colorbar(images[-1], cax=colorbar_axis, orientation="horizontal")
    cbar.set_label("Own-profile minus competitor distance   (negative = correct specificity)")
    fig.suptitle("Steering often makes one demographic pole specific at the expense of the other",
                 fontsize=14.5, y=.995)
    fig.subplots_adjust(left=.25, right=.98, top=.91, bottom=.12, wspace=.12)
    for extension in ("pdf", "png"):
        fig.savefig(out_dir / f"figure4_directional_heatmap_heldout.{extension}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyses-dir", type=Path, default=REPO_ROOT / "analyses")
    parser.add_argument("--selection-dir", type=Path,
                        default=REPO_ROOT / "analyses/magnitude_selection")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "analyses/paper_figures_heldout")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    selected, evaluation, distances, predictions, cross, human_gap = load_inputs(
        args.analyses_dir, args.selection_dir
    )
    left, right = metric_reversal_data(
        selected, evaluation, distances, predictions,
        iterations=args.bootstrap_iterations, seed=args.seed,
    )
    left.to_csv(args.out_dir / "figure5_distance_delta_data.csv", index=False)
    right.to_csv(args.out_dir / "figure5_q4_lift_data.csv", index=False)
    draw_metric_reversal(left, right, args.out_dir)

    trends = heldout_trends(selected, evaluation, predictions)
    trends.to_csv(args.out_dir / "figure3_or_trends_heldout.csv", index=False)
    draw_or_forest(trends, args.out_dir)

    directional = heldout_directional(selected, evaluation, cross, human_gap)
    directional_q4 = directional[directional.quartile.eq("Q4")].copy()
    directional_q4 = pd.concat(
        [pair_filter(directional_q4, attribute) for attribute in ATTRIBUTE_ORDER],
        ignore_index=True,
    )
    directional_q4.to_csv(
        args.out_dir / "figure4_directional_q4_heldout.csv", index=False
    )
    draw_directional_heatmap(directional, args.out_dir)
    print(f"[ok] figures and source tables={args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
