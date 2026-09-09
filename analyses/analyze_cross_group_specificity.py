#!/usr/bin/env python
"""Test whether demographic steering is group-specific rather than uniform.

The input is one raw CSV produced by demographic_opinionqa_experiment.py. For
each model profile, this script measures ordinal Wasserstein distance to every
configured human target group for that attribute, rather than only to the
matching group.
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

from analyze_opinionqa_model import (
    CLASS_TO_HUMAN,
    DEFAULT_OQA,
    DEFAULT_QKEY_DICT,
    REPO_ROOT,
    align_distribution,
    identify_model,
    load_human,
    load_model,
    load_question_metadata,
    mean_ci,
    wasserstein_ordinal,
)


def regime_name(channel, magnitude) -> str:
    if channel == "declared":
        return "Declared"
    if float(magnitude) == 0:
        return "Neutral (N=0)"
    return f"Steered N={float(magnitude):g}"


def mapping_audit(demo, human):
    available_human = {(attribute, group) for _, attribute, group in human}
    mapped, unmapped, missing = {}, [], []
    for attribute in list(dict.fromkeys(demo.attribute)):
        config = CLASS_TO_HUMAN.get(attribute)
        if config is None:
            unmapped.extend(
                (attribute, model_class)
                for model_class in demo.loc[demo.attribute.eq(attribute), "class_label"].unique()
            )
            continue
        human_attribute, class_map = config
        present_classes = set(demo.loc[demo.attribute.eq(attribute), "class_label"])
        usable_map = {
            model_class: human_group
            for model_class, human_group in class_map.items()
            if model_class in present_classes
        }
        unmapped.extend(
            (attribute, model_class) for model_class in sorted(present_classes - set(usable_map))
        )
        for model_class, human_group in usable_map.items():
            if (human_attribute, human_group) not in available_human:
                missing.append((attribute, model_class, human_attribute, human_group))
        if usable_map:
            mapped[attribute] = (human_attribute, usable_map)
    if missing:
        details = ", ".join(f"{a}/{c} -> {ha}/{hg}" for a, c, ha, hg in missing)
        raise ValueError(f"Configured human groups are absent from OpinionQA: {details}")
    return mapped, unmapped


def build_cross_group_matrix(demo, human, question_meta, mappings):
    rows = []
    audit = {
        "model_rows": len(demo),
        "eligible_model_rows": 0,
        "cross_group_rows": 0,
        "unmapped_model_rows": 0,
        "nonordinal_model_rows": 0,
        "missing_human_cells": 0,
        "invalid_distribution_cells": 0,
        "trailing_model_values_removed": 0,
        "trailing_human_values_removed": 0,
    }
    for model_row in demo.itertuples(index=False):
        mapping = mappings.get(model_row.attribute)
        if mapping is None or model_row.class_label not in mapping[1]:
            audit["unmapped_model_rows"] += 1
            continue
        meta = question_meta.get(model_row.qkey)
        if meta is None:
            audit["nonordinal_model_rows"] += 1
            continue
        human_attribute, class_map = mapping
        own_group = class_map[model_row.class_label]
        try:
            model_aligned, model_removed = align_distribution(
                model_row.dist, len(meta["ordinal"])
            )
        except ValueError:
            audit["invalid_distribution_cells"] += len(class_map)
            continue
        audit["eligible_model_rows"] += 1
        audit["trailing_model_values_removed"] += model_removed
        for human_group in dict.fromkeys(class_map.values()):
            human_dist = human.get((model_row.qkey, human_attribute, human_group))
            if human_dist is None:
                audit["missing_human_cells"] += 1
                continue
            try:
                human_aligned, human_removed = align_distribution(
                    human_dist, len(meta["ordinal"])
                )
                distance = wasserstein_ordinal(
                    model_aligned, human_aligned, meta["ordinal"]
                )
            except ValueError:
                audit["invalid_distribution_cells"] += 1
                continue
            audit["trailing_human_values_removed"] += human_removed
            audit["cross_group_rows"] += 1
            rows.append({
                "qkey": model_row.qkey,
                "attribute": model_row.attribute,
                "channel": model_row.channel,
                "magnitude": model_row.magnitude,
                "regime": regime_name(model_row.channel, model_row.magnitude),
                "model_class": model_row.class_label,
                "human_attribute": human_attribute,
                "human_group": human_group,
                "own_human_group": own_group,
                "is_own_group": human_group == own_group,
                "wasserstein": distance,
            })
    return pd.DataFrame(rows), audit


def complete_choice_sets(cross, mappings):
    """Keep model conditions for which every configured human group is present."""
    keys = ["qkey", "attribute", "channel", "magnitude", "regime", "model_class"]
    expected = {attribute: len(set(class_map.values())) for attribute, (_, class_map) in mappings.items()}
    counts = cross.groupby(keys, dropna=False).human_group.nunique().rename("available_groups")
    annotated = cross.merge(counts.reset_index(), on=keys, how="left")
    annotated["expected_groups"] = annotated.attribute.map(expected)
    complete = annotated[annotated.available_groups.eq(annotated.expected_groups)].copy()
    n_total = annotated[keys].drop_duplicates().shape[0]
    n_complete = complete[keys].drop_duplicates().shape[0]
    return complete, n_total - n_complete


def nearest_neighbor_results(cross):
    keys = ["qkey", "attribute", "channel", "magnitude", "regime", "model_class"]
    rows = []
    for values, group in cross.groupby(keys, dropna=False, sort=False):
        minimum = group.wasserstein.min()
        tied = sorted(group.loc[np.isclose(group.wasserstein, minimum), "human_group"])
        own_group = group.own_human_group.iloc[0]
        rows.append(dict(zip(keys, values)) | {
            "own_human_group": own_group,
            "predicted_human_group": " | ".join(tied),
            "n_tied_nearest": len(tied),
            "correct": own_group in tied and len(tied) == 1,
            "correct_in_tie": own_group in tied,
            "fractional_credit": (1.0 / len(tied)) if own_group in tied else 0.0,
            "nearest_distance": minimum,
        })
    return pd.DataFrame(rows)


def nearest_accuracy_summary(nearest, mappings):
    group_fields = ["attribute", "channel", "magnitude", "regime"]
    rows = []
    for values, group in nearest.groupby(group_fields, dropna=False, sort=False):
        attribute, channel, magnitude, regime = values
        n_groups = len(set(mappings[attribute][1].values()))
        rows.append({
            "attribute": attribute,
            "channel": channel,
            "magnitude": magnitude,
            "regime": regime,
            "n": len(group),
            "n_human_groups": n_groups,
            "chance_accuracy": 1.0 / n_groups,
            "strict_accuracy": group.correct.mean(),
            "tie_aware_accuracy": group.correct_in_tie.mean(),
            "fractional_tie_accuracy": group.fractional_credit.mean(),
            "lift_over_chance": group.fractional_credit.mean() - 1.0 / n_groups,
            "tie_rate": group.n_tied_nearest.gt(1).mean(),
        })
    return pd.DataFrame(rows)


def confusion_table(nearest):
    rows = []
    # Tied predictions remain an explicit column instead of being broken arbitrarily.
    fields = ["attribute", "channel", "magnitude", "regime", "model_class",
              "own_human_group", "predicted_human_group"]
    for values, group in nearest.groupby(fields, dropna=False, sort=False):
        record = dict(zip(fields, values))
        record["count"] = len(group)
        rows.append(record)
    result = pd.DataFrame(rows)
    denominators = result.groupby(
        ["attribute", "channel", "magnitude", "regime", "model_class"], dropna=False
    )["count"].transform("sum")
    result["row_fraction"] = result["count"] / denominators
    return result


def margin_results(cross):
    keys = ["qkey", "attribute", "channel", "magnitude", "regime", "model_class"]
    rows = []
    for values, group in cross.groupby(keys, dropna=False, sort=False):
        own = group.loc[group.is_own_group, "wasserstein"]
        others = group.loc[~group.is_own_group, "wasserstein"]
        if len(own) != 1 or not len(others):
            continue
        rows.append(dict(zip(keys, values)) | {
            "own_human_group": group.own_human_group.iloc[0],
            "distance_own": own.iloc[0],
            "mean_distance_others": others.mean(),
            "specificity_margin": own.iloc[0] - others.mean(),
        })
    return pd.DataFrame(rows)


def margin_summary(margins):
    rows = []
    fields = ["attribute", "channel", "magnitude", "regime", "model_class"]
    for values, group in margins.groupby(fields, dropna=False, sort=False):
        mean, ci = mean_ci(group.specificity_margin)
        rows.append(dict(zip(fields, values)) | {
            "n": len(group),
            "mean_margin": mean,
            "ci95": ci,
            "median_margin": group.specificity_margin.median(),
            "fraction_negative": group.specificity_margin.lt(0).mean(),
        })
    return pd.DataFrame(rows)


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    valid = np.flatnonzero(np.isfinite(p_values))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(p_values[valid])]
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(order) - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def directional_tests(cross, mappings):
    """Test own-profile distance < competitor-profile distance to the same group."""
    rows = []
    condition_fields = ["attribute", "channel", "magnitude", "regime"]
    for condition, subset in cross.groupby(condition_fields, dropna=False, sort=False):
        attribute, channel, magnitude, regime = condition
        class_map = mappings[attribute][1]
        for target_class, target_group in class_map.items():
            target = subset[subset.human_group.eq(target_group)]
            own = target[target.model_class.eq(target_class)][["qkey", "wasserstein"]].rename(
                columns={"wasserstein": "distance_own_profile"}
            )
            for competitor_class in class_map:
                if competitor_class == target_class:
                    continue
                competitor = target[target.model_class.eq(competitor_class)][
                    ["qkey", "wasserstein"]
                ].rename(columns={"wasserstein": "distance_competitor_profile"})
                paired = own.merge(competitor, on="qkey", how="inner")
                difference = (
                    paired.distance_own_profile - paired.distance_competitor_profile
                )
                p_value = np.nan
                if len(difference) and not np.allclose(difference, 0):
                    try:
                        p_value = float(wilcoxon(
                            paired.distance_own_profile,
                            paired.distance_competitor_profile,
                            alternative="less",
                        ).pvalue)
                    except ValueError:
                        pass
                rows.append({
                    "attribute": attribute,
                    "channel": channel,
                    "magnitude": magnitude,
                    "regime": regime,
                    "target_human_group": target_group,
                    "expected_model_class": target_class,
                    "competitor_model_class": competitor_class,
                    "n": len(difference),
                    "mean_difference": difference.mean() if len(difference) else np.nan,
                    "median_difference": difference.median() if len(difference) else np.nan,
                    "own_profile_closer_rate": difference.lt(0).mean() if len(difference) else np.nan,
                    "tie_rate": np.isclose(difference, 0).mean() if len(difference) else np.nan,
                    "fractional_win_rate": (
                        difference.lt(0).mean() + 0.5 * np.isclose(difference, 0).mean()
                        if len(difference) else np.nan
                    ),
                    "p_value_one_sided": p_value,
                })
    tests = pd.DataFrame(rows)
    tests["p_value_holm"] = np.nan
    family = ["attribute", "channel", "magnitude"]
    for _, indices in tests.groupby(family, dropna=False).groups.items():
        tests.loc[indices, "p_value_holm"] = holm_adjust(
            tests.loc[indices, "p_value_one_sided"]
        )
    tests["significant_holm_0.05"] = tests.p_value_holm.lt(0.05)
    return tests


def select_comparison_magnitude(cross, requested):
    available = sorted(cross.loc[cross.channel.eq("steered"), "magnitude"].dropna().unique())
    positive = [value for value in available if value > 0]
    if not positive:
        raise ValueError("No positive steering magnitude is available.")
    if requested in positive:
        return float(requested), False
    return float(min(positive, key=lambda value: (abs(value - requested), value))), True


def plot_nearest_accuracy(summary, out_path):
    attributes = list(dict.fromkeys(summary.attribute))
    ncols, nrows = 2, math.ceil(len(attributes) / 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, attribute in zip(axes, attributes):
        values = summary[summary.attribute.eq(attribute)]
        steered = values[values.channel.eq("steered")].sort_values("magnitude")
        declared = values[values.channel.eq("declared")].fractional_tie_accuracy
        chance = values.chance_accuracy.iloc[0]
        ax.plot(steered.magnitude, steered.strict_accuracy, marker="o",
                label="Steered")
        if len(declared):
            declared_accuracy = values.loc[
                values.channel.eq("declared"), "strict_accuracy"
            ].iloc[0]
            ax.axhline(declared_accuracy, color="tab:orange", ls="--", label="Declared")
        ax.axhline(chance, color="gray", ls=":", label=f"Chance ({chance:.2f})")
        ax.set(title=attribute, xlabel="Steering magnitude N",
               ylabel="Nearest-human-group accuracy")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8)
    for ax in axes[len(attributes):]:
        ax.axis("off")
    fig.suptitle("Cross-group nearest-neighbor accuracy", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_margins(summary, out_path):
    attributes = list(dict.fromkeys(summary.attribute))
    ncols, nrows = 2, math.ceil(len(attributes) / 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.8 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, attribute in zip(axes, attributes):
        values = summary[summary.attribute.eq(attribute)]
        for index, (model_class, class_rows) in enumerate(values.groupby("model_class")):
            color = f"C{index}"
            steered = class_rows[class_rows.channel.eq("steered")].sort_values("magnitude")
            declared = class_rows[class_rows.channel.eq("declared")]
            ax.plot(steered.magnitude, steered.mean_margin, marker="o", color=color,
                    label=f"steered {model_class}")
            if len(declared):
                ax.axhline(declared.mean_margin.iloc[0], color=color, ls="--",
                           label=f"declared {model_class}")
        ax.axhline(0, color="black", lw=1, alpha=0.6)
        ax.set(title=attribute, xlabel="Steering magnitude N",
               ylabel="Own − mean(other) Wasserstein distance")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=7)
    for ax in axes[len(attributes):]:
        ax.axis("off")
    fig.suptitle("Group-specificity margin (more negative is more specific)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_confusions(nearest, selected_magnitude, mappings, out_path):
    attributes = list(mappings)
    regimes = ["Declared", "Neutral (N=0)", f"Steered N={selected_magnitude:g}"]
    fig, axes = plt.subplots(
        len(attributes), len(regimes),
        figsize=(4.8 * len(regimes), 4.8 * len(attributes)), squeeze=False,
    )
    for row_index, attribute in enumerate(attributes):
        class_map = mappings[attribute][1]
        classes = list(class_map)
        human_groups = list(dict.fromkeys(class_map.values()))
        for column_index, regime in enumerate(regimes):
            ax = axes[row_index, column_index]
            subset = nearest[nearest.attribute.eq(attribute) & nearest.regime.eq(regime)]
            subset = subset.assign(
                confusion_prediction=np.where(
                    subset.n_tied_nearest.gt(1), "TIE", subset.predicted_human_group
                )
            )
            columns = human_groups + (["TIE"] if subset.n_tied_nearest.gt(1).any() else [])
            matrix = pd.crosstab(subset.model_class, subset.confusion_prediction).reindex(
                index=classes, columns=columns, fill_value=0
            )
            fractions = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0)
            ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues", aspect="auto")
            for i in range(len(classes)):
                for j in range(len(human_groups)):
                    value = fractions.iloc[i, j]
                    ax.text(j, i, "NA" if not np.isfinite(value) else f"{value:.2f}",
                            ha="center", va="center",
                            color="white" if np.isfinite(value) and value > 0.55 else "black",
                            fontsize=8)
            ax.set_xticks(range(len(columns)), columns, rotation=35, ha="right", fontsize=7)
            ax.set_yticks(range(len(classes)), classes, fontsize=8)
            ax.set_title(f"{attribute} — {regime}", fontsize=10)
            ax.set_xlabel("Nearest human group")
            ax.set_ylabel("Model profile")
    fig.suptitle("Nearest-human-group confusion matrices", fontsize=14)
    fig.subplots_adjust(top=0.96, bottom=0.06, hspace=0.95, wspace=0.4)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(path, *, model_label, model_id, input_path, selected_magnitude,
                 fallback, audit, incomplete_sets, unmapped, accuracy, margins,
                 tests, files):
    headline = accuracy[
        accuracy.regime.isin(["Declared", "Neutral (N=0)", f"Steered N={selected_magnitude:g}"])
    ].copy()
    lines = [
        f"# Cross-group specificity report: {model_label}", "",
        f"- Input: `{input_path.resolve()}`",
        f"- Model ID: `{model_id or 'not recorded in CSV'}`",
        f"- Comparison magnitude: `{selected_magnitude:g}`"
        + (" (nearest available)" if fallback else ""),
        f"- Cross-group distance cells: `{audit['cross_group_rows']}`",
        f"- Incomplete model/question choice sets excluded from classification: `{incomplete_sets}`",
        f"- Unmapped model rows: `{audit['unmapped_model_rows']}`", "",
        "The nearest-neighbor result is correct only when the uniquely closest human group "
        "matches the configured group for that model profile. Fractional accuracy assigns "
        "1/k credit when the correct group is among k tied nearest groups. The specificity "
        "margin is own-group distance minus the mean distance to other groups; negative values "
        "indicate profile-specific alignment.", "",
        "## Headline nearest-group accuracy", "",
        "| Attribute | Regime | n | Strict accuracy | Fractional-tie accuracy | Chance | Lift* | Tie rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in headline.itertuples(index=False):
        lines.append(
            f"| {row.attribute} | {row.regime} | {row.n} | "
            f"{row.strict_accuracy:.3f} | {row.fractional_tie_accuracy:.3f} | "
            f"{row.chance_accuracy:.3f} | "
            f"{row.lift_over_chance:+.3f} | {row.tie_rate:.3f} |"
        )
    lines.extend(["", "\\* Lift uses fractional credit for tied nearest groups."])
    lines.extend(["", "## Directional tests at the comparison regimes", "",
                  "Negative differences support specificity. `p_holm` is the one-sided "
                  "Wilcoxon p-value corrected within attribute and condition.", "",
                  "| Attribute | Regime | Target | Expected vs competitor | n | Mean Δ | Fractional win rate | p_holm |",
                  "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"])
    selected_tests = tests[
        tests.regime.isin(["Declared", "Neutral (N=0)", f"Steered N={selected_magnitude:g}"])
    ]
    for row in selected_tests.itertuples(index=False):
        lines.append(
            f"| {row.attribute} | {row.regime} | {row.target_human_group} | "
            f"{row.expected_model_class} vs {row.competitor_model_class} | {row.n} | "
            f"{row.mean_difference:+.3f} | {row.fractional_win_rate:.3f} | "
            f"{row.p_value_holm:.3g} |"
        )
    if unmapped:
        lines.extend(["", "## Unmapped classes", ""])
        lines.extend(f"- `{attribute}/{model_class}`" for attribute, model_class in unmapped)
    lines.extend(["", "## Output files", ""])
    lines.extend(f"- [{label}]({file.name})" for label, file in files.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--opinionqa", type=Path, default=DEFAULT_OQA)
    parser.add_argument("--qkey-dict", type=Path, default=DEFAULT_QKEY_DICT)
    parser.add_argument("--comparison-magnitude", type=float, default=7.0)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    for file in (args.input, args.opinionqa, args.qkey_dict):
        if not file.is_file():
            raise FileNotFoundError(file)

    demo = load_model(args.input)
    human = load_human(args.opinionqa)
    question_meta = load_question_metadata(args.qkey_dict)
    model_label, model_id = identify_model(demo, args.input)
    out_dir = args.out_dir or REPO_ROOT / "analyses/cross_group_reports" / model_label
    out_dir.mkdir(parents=True, exist_ok=True)

    mappings, unmapped = mapping_audit(demo, human)
    cross, audit = build_cross_group_matrix(demo, human, question_meta, mappings)
    if cross.empty:
        raise ValueError("No cross-group distances could be computed.")
    complete, incomplete_sets = complete_choice_sets(cross, mappings)
    nearest = nearest_neighbor_results(complete)
    accuracy = nearest_accuracy_summary(nearest, mappings)
    confusion = confusion_table(nearest)
    margins = margin_results(complete)
    margin_stats = margin_summary(margins)
    tests = directional_tests(complete, mappings)
    selected_magnitude, fallback = select_comparison_magnitude(
        complete, args.comparison_magnitude
    )

    files = {
        "Cross-group distances": out_dir / "cross_group_distances.csv",
        "Nearest predictions": out_dir / "nearest_group_predictions.csv",
        "Nearest accuracy": out_dir / "nearest_group_accuracy.csv",
        "Confusion matrices": out_dir / "nearest_group_confusions.csv",
        "Specificity margins": out_dir / "specificity_margins.csv",
        "Specificity margin summary": out_dir / "specificity_margin_summary.csv",
        "Directional Wilcoxon tests": out_dir / "directional_tests.csv",
        "Accuracy plot": out_dir / "nearest_group_accuracy.png",
        "Margin plot": out_dir / "specificity_margin.png",
        "Confusion plot": out_dir / "nearest_group_confusions.png",
    }
    cross.to_csv(files["Cross-group distances"], index=False)
    nearest.to_csv(files["Nearest predictions"], index=False)
    accuracy.to_csv(files["Nearest accuracy"], index=False)
    confusion.to_csv(files["Confusion matrices"], index=False)
    margins.to_csv(files["Specificity margins"], index=False)
    margin_stats.to_csv(files["Specificity margin summary"], index=False)
    tests.to_csv(files["Directional Wilcoxon tests"], index=False)
    plot_nearest_accuracy(accuracy, files["Accuracy plot"])
    plot_margins(margin_stats, files["Margin plot"])
    plot_confusions(nearest, selected_magnitude, mappings, files["Confusion plot"])

    report = out_dir / "report.md"
    write_report(
        report,
        model_label=model_label,
        model_id=model_id,
        input_path=args.input,
        selected_magnitude=selected_magnitude,
        fallback=fallback,
        audit=audit,
        incomplete_sets=incomplete_sets,
        unmapped=unmapped,
        accuracy=accuracy,
        margins=margin_stats,
        tests=tests,
        files=files,
    )
    print(f"[ok] cross-group rows={len(cross)} complete conditions={len(nearest)}")
    print(f"[ok] report={report.resolve()}")


if __name__ == "__main__":
    main()
