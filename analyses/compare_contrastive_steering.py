#!/usr/bin/env python
"""Compare raw and contrastive demographic steering on matched OpinionQA rows.

The comparison is paired by question, attribute, model class, and magnitude.
Every outcome is expressed as a change from that run's own neutral condition
(N=0), which prevents small baseline differences from being attributed to the
steering vector. Distances are computed against every mapped human group.
"""

from __future__ import annotations

import argparse
import math
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
    margin_results,
    nearest_neighbor_results,
)
from analyze_opinionqa_model import (
    DEFAULT_OQA,
    DEFAULT_QKEY_DICT,
    REPO_ROOT,
    load_human,
    load_model,
    load_question_metadata,
)


MODEL_FILES = {
    "qwen": {
        "normal": "demographic_opinionqa_qwen.csv",
        "contrastive": "demographic_opinionqa_qwen_contrastive.csv",
        "paper_magnitude": 20.0,
    },
    "llama": {
        "normal": "demographic_opinionqa_llama.csv",
        "contrastive": "demographic_opinionqa_llama_contrastive_evaluable.csv",
        "paper_magnitude": 20.0,
    },
    "mistral": {
        "normal": "demographic_opinionqa_mistral.csv",
        "contrastive": "demographic_opinionqa_mistral_contrastive_evaluable.csv",
        "paper_magnitude": 2.0,
    },
}

METRICS = {
    "delta_target": ("Δ distance to own group", "lower"),
    "delta_others": ("Δ distance to other groups", "diagnostic"),
    "delta_specificity": ("Δ specificity margin", "lower"),
    "delta_nearest": ("Δ nearest-group credit", "higher"),
}


def safe_wilcoxon(values: pd.Series) -> float:
    values = values.dropna().to_numpy(float)
    if not len(values) or np.allclose(values, 0):
        return np.nan
    try:
        return float(wilcoxon(values, alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def mean_ci(values: pd.Series) -> tuple[float, float]:
    values = values.dropna().to_numpy(float)
    if not len(values):
        return np.nan, np.nan
    mean = float(values.mean())
    ci = float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, ci


def build_mode_metrics(path, mode, human, metadata):
    demo = load_model(path)
    demo = demo[demo.channel.eq("steered")].copy()
    mappings, unmapped = mapping_audit(demo, human)
    cross, audit = build_cross_group_matrix(demo, human, metadata, mappings)
    complete, incomplete = complete_choice_sets(cross, mappings)
    margins = margin_results(complete)
    nearest = nearest_neighbor_results(complete)
    keys = ["qkey", "attribute", "channel", "magnitude", "regime", "model_class"]
    metrics = margins.merge(
        nearest[keys + ["fractional_credit"]], on=keys, how="inner", validate="one_to_one"
    )
    metrics = metrics.rename(columns={
        "distance_own": "target_distance",
        "mean_distance_others": "others_distance",
        "specificity_margin": "specificity",
        "fractional_credit": "nearest_credit",
    })
    baseline = metrics[metrics.magnitude.eq(0)][
        ["qkey", "attribute", "model_class", "target_distance", "others_distance",
         "specificity", "nearest_credit"]
    ].rename(columns={
        "target_distance": "baseline_target",
        "others_distance": "baseline_others",
        "specificity": "baseline_specificity",
        "nearest_credit": "baseline_nearest",
    })
    baseline_keys = ["qkey", "attribute", "model_class"]
    if baseline.duplicated(baseline_keys).any():
        raise ValueError(f"Duplicate N=0 rows in {path}")
    metrics = metrics.merge(baseline, on=baseline_keys, how="inner", validate="many_to_one")
    metrics["delta_target"] = metrics.target_distance - metrics.baseline_target
    metrics["delta_others"] = metrics.others_distance - metrics.baseline_others
    metrics["delta_specificity"] = metrics.specificity - metrics.baseline_specificity
    metrics["delta_nearest"] = metrics.nearest_credit - metrics.baseline_nearest
    metrics["mode"] = mode
    return metrics, {
        "input": str(path),
        "model_rows": len(demo),
        "cross_group_rows": audit["cross_group_rows"],
        "complete_conditions": len(metrics),
        "incomplete_conditions": incomplete,
        "unmapped_classes": unmapped,
    }


def pair_modes(normal, contrastive, model):
    keys = ["qkey", "attribute", "model_class", "magnitude"]
    columns = keys + [
        "target_distance", "others_distance", "specificity", "nearest_credit",
        "baseline_target", "baseline_others", "baseline_specificity", "baseline_nearest",
        *METRICS,
    ]
    paired = normal[columns].merge(
        contrastive[columns], on=keys, suffixes=("_normal", "_contrastive"),
        how="inner", validate="one_to_one",
    )
    paired["model"] = model
    for metric in METRICS:
        paired[f"contrastive_minus_normal_{metric}"] = (
            paired[f"{metric}_contrastive"] - paired[f"{metric}_normal"]
        )
    return paired


def mode_summary(all_metrics):
    rows = []
    fields = ["model", "mode", "magnitude"]
    for values, group in all_metrics.groupby(fields, sort=False):
        record = dict(zip(fields, values))
        record.update({"n": len(group), "n_qkeys": group.qkey.nunique()})
        for metric in METRICS:
            record[f"mean_{metric}"] = group[metric].mean()
            record[f"ci95_{metric}"] = mean_ci(group[metric])[1]
        rows.append(record)
    return pd.DataFrame(rows)


def by_attribute_class(all_metrics):
    fields = ["model", "mode", "attribute", "model_class", "magnitude"]
    rows = []
    for values, group in all_metrics.groupby(fields, sort=False):
        record = dict(zip(fields, values)) | {"n": len(group)}
        for metric in METRICS:
            record[f"mean_{metric}"] = group[metric].mean()
            record[f"median_{metric}"] = group[metric].median()
        rows.append(record)
    return pd.DataFrame(rows)


def paired_summary(paired):
    rows = []
    grouping_sets = [
        ["model", "magnitude"],
        ["model", "attribute", "magnitude"],
        ["model", "attribute", "model_class", "magnitude"],
    ]
    for fields in grouping_sets:
        for values, group in paired.groupby(fields, sort=False):
            if not isinstance(values, tuple):
                values = (values,)
            record = dict(zip(fields, values))
            record["summary_level"] = (
                "overall" if len(fields) == 2 else "attribute" if len(fields) == 3 else "class"
            )
            record.update({"n": len(group), "n_qkeys": group.qkey.nunique()})
            for metric, (_, direction) in METRICS.items():
                column = f"contrastive_minus_normal_{metric}"
                mean, ci = mean_ci(group[column])
                # Questions are the independent sampling units. Aggregate over
                # demographic classes before inference to avoid pseudoreplication.
                question_differences = group.groupby("qkey")[column].mean()
                record[f"mean_difference_{metric}"] = mean
                record[f"ci95_difference_{metric}"] = ci
                record[f"p_two_sided_{metric}"] = safe_wilcoxon(question_differences)
                if direction == "lower":
                    record[f"contrastive_better_rate_{metric}"] = question_differences.lt(0).mean()
                elif direction == "higher":
                    record[f"contrastive_better_rate_{metric}"] = question_differences.gt(0).mean()
            rows.append(record)
    return pd.DataFrame(rows)


def baseline_audit(paired):
    n0 = paired[paired.magnitude.eq(0)].copy()
    rows = []
    for model, group in n0.groupby("model"):
        record = {"model": model, "n": len(group)}
        for metric in ["target", "others", "specificity", "nearest"]:
            diff = group[f"baseline_{metric}_contrastive"] - group[f"baseline_{metric}_normal"]
            record[f"mean_abs_difference_{metric}"] = diff.abs().mean()
            record[f"max_abs_difference_{metric}"] = diff.abs().max()
        rows.append(record)
    return pd.DataFrame(rows)


def best_magnitudes(summary):
    positive = summary[summary.magnitude.gt(0)].copy()
    rows = []
    for (model, mode), group in positive.groupby(["model", "mode"], sort=False):
        for metric, (_, direction) in METRICS.items():
            if direction == "diagnostic":
                continue
            column = f"mean_{metric}"
            index = group[column].idxmin() if direction == "lower" else group[column].idxmax()
            best = group.loc[index]
            rows.append({
                "model": model,
                "mode": mode,
                "criterion": metric,
                "best_magnitude": best.magnitude,
                "criterion_value": best[column],
            })
    return pd.DataFrame(rows)


def fixed_magnitude_summary(summary):
    rows = []
    for model, config in MODEL_FILES.items():
        magnitude = config["paper_magnitude"]
        selected = summary[summary.model.eq(model) & summary.magnitude.eq(magnitude)]
        for row in selected.itertuples(index=False):
            rows.append(row._asdict() | {"selection": "paper magnitude"})
    return pd.DataFrame(rows)


def plot_curves(summary, path):
    models = list(MODEL_FILES)
    plot_metrics = ["delta_target", "delta_specificity", "delta_nearest"]
    fig, axes = plt.subplots(len(models), len(plot_metrics), figsize=(15, 12), sharex=True)
    colors = {"normal": "tab:blue", "contrastive": "tab:orange"}
    for row_index, model in enumerate(models):
        for column_index, metric in enumerate(plot_metrics):
            ax = axes[row_index, column_index]
            for mode in ["normal", "contrastive"]:
                values = summary[
                    summary.model.eq(model) & summary["mode"].eq(mode)
                ].sort_values("magnitude")
                ax.plot(
                    values.magnitude, values[f"mean_{metric}"], marker="o",
                    color=colors[mode], label=mode.capitalize(),
                )
            ax.axhline(0, color="black", lw=1, alpha=0.55)
            ax.grid(alpha=0.15)
            ax.set_title(f"{model} — {METRICS[metric][0]}")
            ax.set_xlabel("Steering magnitude N")
            ax.set_ylabel("Change from N=0")
            if row_index == 0 and column_index == 0:
                ax.legend()
    fig.suptitle("Raw versus contrastive demographic steering", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_value(value, signed=True):
    if pd.isna(value):
        return "NA"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def write_report(path, summary, fixed, best, by_class, baseline, paired_stats, audits):
    lines = [
        "# Raw versus contrastive demographic steering", "",
        "All comparisons use the same mapped model classes and are paired by question, "
        "attribute, class, and magnitude. Outcomes are changes from each run's own N=0. "
        "For target distance and specificity, negative is better; for nearest-group credit, "
        "positive is better.", "",
        "## Baseline agreement", "",
        "| Model | n | Mean abs(Δ own) | Max abs(Δ own) | Mean abs(Δ other) | Max abs(Δ other) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in baseline.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.n} | {row.mean_abs_difference_target:.6g} | "
            f"{row.max_abs_difference_target:.6g} | {row.mean_abs_difference_others:.6g} | "
            f"{row.max_abs_difference_others:.6g} |"
        )
    lines += ["", "## Comparison at the paper magnitudes", "",
              "| Model | N | Mode | Δ target | Δ others | Δ specificity | Δ nearest |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in fixed.sort_values(["model", "mode"]).itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.magnitude:g} | {row.mode} | "
            f"{format_value(row.mean_delta_target)} | {format_value(row.mean_delta_others)} | "
            f"{format_value(row.mean_delta_specificity)} | {format_value(row.mean_delta_nearest)} |"
        )
    lines += ["", "## Direct contrastive-minus-normal differences", "",
              "Negative target/specificity differences and positive nearest differences favor contrastive.", "",
              "| Model | N | ΔΔ target | ΔΔ specificity | ΔΔ nearest | p target | p specificity | p nearest |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    overall = paired_stats[paired_stats.summary_level.eq("overall")]
    for model, config in MODEL_FILES.items():
        values = overall[overall.model.eq(model) & overall.magnitude.eq(config["paper_magnitude"])]
        if values.empty:
            continue
        row = values.iloc[0]
        lines.append(
            f"| {model} | {row.magnitude:g} | {format_value(row.mean_difference_delta_target)} | "
            f"{format_value(row.mean_difference_delta_specificity)} | "
            f"{format_value(row.mean_difference_delta_nearest)} | "
            f"{row.p_two_sided_delta_target:.3g} | {row.p_two_sided_delta_specificity:.3g} | "
            f"{row.p_two_sided_delta_nearest:.3g} |"
        )
    lines += ["", "### Differences by attribute at the paper magnitudes", "",
              "| Model | Attribute | ΔΔ target | ΔΔ specificity | ΔΔ nearest |",
              "| --- | --- | ---: | ---: | ---: |"]
    attributes = paired_stats[paired_stats.summary_level.eq("attribute")]
    for model, config in MODEL_FILES.items():
        selected = attributes[
            attributes.model.eq(model) & attributes.magnitude.eq(config["paper_magnitude"])
        ]
        for row in selected.sort_values("attribute").itertuples(index=False):
            lines.append(
                f"| {model} | {row.attribute} | "
                f"{format_value(row.mean_difference_delta_target)} | "
                f"{format_value(row.mean_difference_delta_specificity)} | "
                f"{format_value(row.mean_difference_delta_nearest)} |"
            )
    lines += ["", "## Descriptive best magnitudes", "",
              "These optima are descriptive and were not selected on an independent validation set.", "",
              "| Model | Mode | Criterion | Best N | Value |",
              "| --- | --- | --- | ---: | ---: |"]
    for row in best.sort_values(["model", "mode", "criterion"]).itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.mode} | {row.criterion} | {row.best_magnitude:g} | "
            f"{format_value(row.criterion_value)} |"
        )
    lines += ["", "## Class-level failures at the paper magnitudes", "",
              "A class is listed when target distance or specificity worsens relative to N=0.", ""]
    failures = []
    for model, config in MODEL_FILES.items():
        selected = by_class[
            by_class.model.eq(model) & by_class.magnitude.eq(config["paper_magnitude"])
        ]
        for row in selected.itertuples(index=False):
            if row.mean_delta_target > 0 or row.mean_delta_specificity > 0:
                failures.append(row)
    if failures:
        lines += ["| Model | Mode | Attribute/class | Δ target | Δ specificity |",
                  "| --- | --- | --- | ---: | ---: |"]
        for row in failures:
            lines.append(
                f"| {row.model} | {row.mode} | {row.attribute}/{row.model_class} | "
                f"{format_value(row.mean_delta_target)} | {format_value(row.mean_delta_specificity)} |"
            )
    else:
        lines.append("No class-level failures under this definition.")
    lines += ["", "## Interpretation", "",
              "At the paper magnitudes, contrastive steering clearly improves Qwen: it "
              "reduces target distance more, improves the specificity margin, and raises "
              "nearest-group recovery relative to raw steering. For Llama it performs worse "
              "on all three aggregate outcomes. For Mistral it produces a much smaller change: "
              "target-distance improvement is weaker, while specificity and nearest-group "
              "differences are close to zero. Thus the contrast is a model-specific improvement, "
              "not a generally superior steering rule."]
    lines += ["", "## Input audit", ""]
    for model, mode_audits in audits.items():
        for mode, audit in mode_audits.items():
            lines.append(
                f"- `{model}/{mode}`: {audit['complete_conditions']} complete conditions; "
                f"{audit['incomplete_conditions']} incomplete conditions excluded; input `{audit['input']}`."
            )
    lines += ["", "## Output tables", "",
              "- `mode_summary_by_magnitude.csv`", "- `paired_mode_differences.csv`",
              "- `paired_comparison_summary.csv`", "- `summary_by_attribute_class.csv`",
              "- `fixed_magnitude_summary.csv`", "- `best_magnitudes_descriptive.csv`",
              "- `baseline_agreement.csv`", "- `comparison_curves.png`"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--opinionqa", type=Path, default=DEFAULT_OQA)
    parser.add_argument("--qkey-dict", type=Path, default=DEFAULT_QKEY_DICT)
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "analyses/contrastive_vs_normal"
    )
    args = parser.parse_args()

    human = load_human(args.opinionqa)
    metadata = load_question_metadata(args.qkey_dict)
    all_metrics, all_paired, audits = [], [], {}
    for model, config in MODEL_FILES.items():
        audits[model] = {}
        mode_frames = {}
        for mode in ["normal", "contrastive"]:
            input_path = args.results_dir / config[mode]
            if not input_path.is_file():
                raise FileNotFoundError(input_path)
            metrics, audit = build_mode_metrics(input_path, mode, human, metadata)
            metrics["model"] = model
            all_metrics.append(metrics)
            mode_frames[mode] = metrics
            audits[model][mode] = audit
            print(f"[info] {model}/{mode}: complete conditions={len(metrics)}")
        paired = pair_modes(mode_frames["normal"], mode_frames["contrastive"], model)
        all_paired.append(paired)
        print(f"[info] {model}: paired conditions={len(paired)}")

    metrics = pd.concat(all_metrics, ignore_index=True)
    paired = pd.concat(all_paired, ignore_index=True)
    summary = mode_summary(metrics)
    class_summary = by_attribute_class(metrics)
    comparison = paired_summary(paired)
    baseline = baseline_audit(paired)
    best = best_magnitudes(summary)
    fixed = fixed_magnitude_summary(summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir / "mode_metrics_by_question.csv", index=False)
    paired.to_csv(args.out_dir / "paired_mode_differences.csv", index=False)
    summary.to_csv(args.out_dir / "mode_summary_by_magnitude.csv", index=False)
    class_summary.to_csv(args.out_dir / "summary_by_attribute_class.csv", index=False)
    comparison.to_csv(args.out_dir / "paired_comparison_summary.csv", index=False)
    fixed.to_csv(args.out_dir / "fixed_magnitude_summary.csv", index=False)
    best.to_csv(args.out_dir / "best_magnitudes_descriptive.csv", index=False)
    baseline.to_csv(args.out_dir / "baseline_agreement.csv", index=False)
    plot_curves(summary, args.out_dir / "comparison_curves.png")
    write_report(
        args.out_dir / "report.md", summary, fixed, best, class_summary,
        baseline, comparison, audits,
    )
    print(f"[ok] report={args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
