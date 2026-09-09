#!/usr/bin/env python
"""Test model-profile specificity across human-separability quartiles.

This analysis consumes outputs already produced by
analyze_cross_group_specificity.py and analyze_human_separability.py. It does
not load or run a language model. Human-gap quartiles are calculated separately
for each attribute and human-group pair, preserving multiclass contrasts such
as the three distinct education pairs.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon
from sklearn.linear_model import LogisticRegression

from analyze_cross_group_specificity import holm_adjust
from analyze_opinionqa_model import CLASS_TO_HUMAN, REPO_ROOT


QUARTILES = ["Q1", "Q2", "Q3", "Q4"]


def read_csv_required(path: Path, columns: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false", "1", "0"})
    if invalid.any():
        examples = sorted(normalized[invalid].unique())[:5]
        raise ValueError(f"Could not parse boolean values: {examples}")
    return normalized.isin({"true", "1"})


def mapped_pairs() -> pd.DataFrame:
    rows = []
    for attribute, (_, class_map) in CLASS_TO_HUMAN.items():
        human_to_model = {human: model for model, human in class_map.items()}
        if len(human_to_model) != len(class_map):
            raise ValueError(f"{attribute} mapping is not one-to-one.")
        groups = list(human_to_model)
        for index, group_a in enumerate(groups):
            for group_b in groups[index + 1:]:
                rows.append({
                    "attribute": attribute,
                    "group_a": group_a,
                    "group_b": group_b,
                    "model_class_a": human_to_model[group_a],
                    "model_class_b": human_to_model[group_b],
                    "n_human_groups": len(groups),
                    "chance_accuracy": 1.0 / len(groups),
                })
    return pd.DataFrame(rows)


def canonicalize_human_pairs(human_gap: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Match pair orientation to CLASS_TO_HUMAN and attach unrelated pairs."""
    records = []
    lookup = {
        (row.attribute, frozenset((row.group_a, row.group_b))): row
        for row in pairs.itertuples(index=False)
    }
    for row in human_gap.itertuples(index=False):
        config = lookup.get((row.attribute, frozenset((row.group_a, row.group_b))))
        if config is None:
            continue
        records.append({
            "qkey": row.qkey,
            "attribute": row.attribute,
            "group_a": config.group_a,
            "group_b": config.group_b,
            "model_class_a": config.model_class_a,
            "model_class_b": config.model_class_b,
            "n_human_groups": config.n_human_groups,
            "chance_accuracy": config.chance_accuracy,
            "human_gap": float(row.wasserstein),
        })
    result = pd.DataFrame(records)
    duplicates = result.duplicated(["qkey", "attribute", "group_a", "group_b"])
    if duplicates.any():
        raise ValueError("Human separability input contains duplicated question/pair rows.")
    return result


def add_pairwise_quartiles(human_pairs: pd.DataFrame) -> pd.DataFrame:
    """Assign Q1--Q4 within every attribute and exact human contrast."""
    result = human_pairs.copy()
    result["quartile"] = ""
    fields = ["attribute", "group_a", "group_b"]
    for _, indices in result.groupby(fields, sort=False).groups.items():
        values = result.loc[indices, "human_gap"]
        # Average ranks keep identical gaps in the same stratum.
        percentile = values.rank(method="average", pct=True)
        labels = np.ceil(percentile * 4).clip(1, 4).astype(int).map(lambda q: f"Q{q}")
        result.loc[indices, "quartile"] = labels
    return result


def attach_gaps_to_predictions(nearest: pd.DataFrame, human_pairs: pd.DataFrame) -> pd.DataFrame:
    """Expand predictions only across pairs containing their own human group."""
    merged = nearest.merge(human_pairs, on=["qkey", "attribute"], how="inner")
    endpoint = (
        merged.own_human_group.eq(merged.group_a)
        | merged.own_human_group.eq(merged.group_b)
    )
    merged = merged[endpoint].copy()
    expected_class = np.where(
        merged.own_human_group.eq(merged.group_a),
        merged.model_class_a,
        merged.model_class_b,
    )
    merged = merged[merged.model_class.eq(expected_class)].copy()
    return merged


def clustered_bootstrap_accuracy(group: pd.DataFrame, *, iterations: int,
                                 rng: np.random.Generator):
    qkeys = group.qkey.unique()
    if not len(qkeys):
        return (np.nan,) * 5
    by_question = group.groupby("qkey").agg(
        strict=("correct", "mean"), fractional=("fractional_credit", "mean")
    )
    sampled_indices = rng.integers(
        0, len(by_question), size=(iterations, len(by_question))
    )
    strict_samples = by_question.strict.to_numpy()[sampled_indices].mean(axis=1)
    fractional_samples = by_question.fractional.to_numpy()[sampled_indices].mean(axis=1)
    chance = float(group.chance_accuracy.iloc[0])
    lifts = fractional_samples - chance
    return (
        float(np.quantile(strict_samples, 0.025)),
        float(np.quantile(strict_samples, 0.975)),
        float(np.quantile(lifts, 0.025)),
        float(np.quantile(lifts, 0.975)),
        float((1 + np.count_nonzero(lifts <= 0)) / (iterations + 1)),
    )


def accuracy_summary(predictions: pd.DataFrame, *, iterations: int, seed: int):
    rows = []
    rng = np.random.default_rng(seed)
    fields = [
        "attribute", "group_a", "group_b", "channel", "magnitude", "regime"
    ]
    for condition, subset in predictions.groupby(fields, dropna=False, sort=False):
        strata = [("All", subset)]
        strata.extend((quartile, subset[subset.quartile.eq(quartile)]) for quartile in QUARTILES)
        for quartile, group in strata:
            if group.empty:
                continue
            chance = float(group.chance_accuracy.iloc[0])
            fractional = float(group.fractional_credit.mean())
            strict_ci_low, strict_ci_high, ci_low, ci_high, p_boot = clustered_bootstrap_accuracy(
                group, iterations=iterations, rng=rng
            )
            rows.append(dict(zip(fields, condition)) | {
                "quartile": quartile,
                "n_questions": group.qkey.nunique(),
                "n_predictions": len(group),
                "mean_human_gap": group.human_gap.mean(),
                "strict_accuracy": group.correct.mean(),
                "strict_accuracy_ci95_low": strict_ci_low,
                "strict_accuracy_ci95_high": strict_ci_high,
                "tie_aware_accuracy": group.correct_in_tie.mean(),
                "fractional_accuracy": fractional,
                "chance_accuracy": chance,
                "lift_over_chance": fractional - chance,
                "lift_ci95_low": ci_low,
                "lift_ci95_high": ci_high,
                "bootstrap_p_lift_le_zero": p_boot,
            })
    return pd.DataFrame(rows)


def logistic_trend(group: pd.DataFrame):
    """Fit correctness on gap and return a qkey-clustered slope inference."""
    y = group.correct.astype(int).to_numpy()
    x = group.human_gap.to_numpy(dtype=float)
    scale = float(x.std(ddof=0))
    if len(np.unique(y)) < 2 or scale <= 0:
        return np.nan, np.nan, np.nan, np.nan
    standardized = (x - x.mean()) / scale
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(standardized.reshape(-1, 1), y)
    slope = float(model.coef_[0, 0])

    # Cluster-robust sandwich covariance keeps the two endpoint profiles from
    # the same question from being treated as independent observations.
    design = np.column_stack([np.ones(len(group)), standardized])
    probabilities = model.predict_proba(standardized.reshape(-1, 1))[:, 1]
    weights = probabilities * (1 - probabilities)
    bread = np.linalg.pinv(design.T @ (weights[:, None] * design))
    scores = design * (y - probabilities)[:, None]
    score_frame = pd.DataFrame(scores, columns=["intercept", "slope"])
    score_frame["qkey"] = group.qkey.to_numpy()
    cluster_scores = score_frame.groupby("qkey")[["intercept", "slope"]].sum().to_numpy()
    meat = cluster_scores.T @ cluster_scores
    covariance = bread @ meat @ bread
    n_clusters = len(cluster_scores)
    n_observations, n_parameters = design.shape
    if n_clusters > 1 and n_observations > n_parameters:
        covariance *= (
            n_clusters / (n_clusters - 1)
            * (n_observations - 1) / (n_observations - n_parameters)
        )
    standard_error = float(np.sqrt(max(covariance[1, 1], 0)))
    if standard_error <= 0 or not np.isfinite(standard_error):
        return slope, np.nan, np.nan, np.nan
    ci_low = slope - 1.96 * standard_error
    ci_high = slope + 1.96 * standard_error
    p_positive = float(norm.sf(slope / standard_error))
    return slope, ci_low, ci_high, p_positive


def trend_tests(predictions: pd.DataFrame):
    rows = []
    fields = [
        "attribute", "group_a", "group_b", "channel", "magnitude", "regime"
    ]
    for condition, group in predictions.groupby(fields, dropna=False, sort=False):
        slope, ci_low, ci_high, p_positive = logistic_trend(group)
        qkeys = group.qkey.unique()
        rows.append(dict(zip(fields, condition)) | {
            "n_questions": len(qkeys),
            "n_predictions": len(group),
            "strict_accuracy": group.correct.mean(),
            "log_odds_slope_per_gap_sd": slope,
            "odds_ratio_per_gap_sd": math.exp(slope) if np.isfinite(slope) else np.nan,
            "slope_ci95_low": ci_low,
            "slope_ci95_high": ci_high,
            "cluster_robust_p_slope_le_zero": p_positive,
        })
    return pd.DataFrame(rows)


def build_directional_rows(cross: pd.DataFrame, human_pairs: pd.DataFrame):
    """Rebuild per-question own-profile versus competitor comparisons."""
    rows = []
    condition = ["qkey", "attribute", "channel", "magnitude", "regime"]
    pair_fields = ["attribute", "group_a", "group_b", "model_class_a", "model_class_b"]
    for pair_values, pair_questions in human_pairs.groupby(pair_fields, sort=False):
        attribute, group_a, group_b, model_class_a, model_class_b = pair_values
        attribute_cross = cross[cross.attribute.eq(attribute)]
        for target_group, expected_class, competitor_class in (
            (group_a, model_class_a, model_class_b),
            (group_b, model_class_b, model_class_a),
        ):
            target = attribute_cross[attribute_cross.human_group.eq(target_group)]
            own = target[target.model_class.eq(expected_class)][condition + ["wasserstein"]].rename(
                columns={"wasserstein": "distance_own_profile"}
            )
            competitor = target[target.model_class.eq(competitor_class)][
                condition + ["wasserstein"]
            ].rename(columns={"wasserstein": "distance_competitor_profile"})
            paired = own.merge(competitor, on=condition, how="inner").merge(
                pair_questions[["qkey", "human_gap", "quartile"]],
                on="qkey", how="inner",
            )
            for record in paired.itertuples(index=False):
                difference = record.distance_own_profile - record.distance_competitor_profile
                rows.append({
                    "qkey": record.qkey,
                    "attribute": attribute,
                    "group_a": group_a,
                    "group_b": group_b,
                    "human_gap": record.human_gap,
                    "quartile": record.quartile,
                    "channel": record.channel,
                    "magnitude": record.magnitude,
                    "regime": record.regime,
                    "target_human_group": target_group,
                    "expected_model_class": expected_class,
                    "competitor_model_class": competitor_class,
                    "distance_own_profile": record.distance_own_profile,
                    "distance_competitor_profile": record.distance_competitor_profile,
                    "difference": difference,
                })
    return pd.DataFrame(rows)


def directional_summary(directional: pd.DataFrame):
    rows = []
    fields = [
        "attribute", "group_a", "group_b", "channel", "magnitude", "regime",
        "target_human_group", "expected_model_class", "competitor_model_class",
    ]
    for condition, subset in directional.groupby(fields, dropna=False, sort=False):
        strata = [("All", subset)]
        strata.extend((quartile, subset[subset.quartile.eq(quartile)]) for quartile in QUARTILES)
        for quartile, group in strata:
            if group.empty:
                continue
            difference = group.difference.to_numpy(dtype=float)
            p_value = np.nan
            if not np.allclose(difference, 0):
                try:
                    p_value = float(wilcoxon(difference, alternative="less").pvalue)
                except ValueError:
                    pass
            tied = np.isclose(difference, 0)
            rows.append(dict(zip(fields, condition)) | {
                "quartile": quartile,
                "n_questions": group.qkey.nunique(),
                "mean_human_gap": group.human_gap.mean(),
                "mean_difference": difference.mean(),
                "median_difference": np.median(difference),
                "own_profile_closer_rate": np.mean(difference < 0),
                "tie_rate": tied.mean(),
                "fractional_win_rate": np.mean(difference < 0) + 0.5 * tied.mean(),
                "p_value_one_sided": p_value,
            })
    result = pd.DataFrame(rows)
    result["p_value_holm"] = np.nan
    family = ["attribute", "channel", "magnitude", "quartile"]
    for _, indices in result.groupby(family, dropna=False).groups.items():
        result.loc[indices, "p_value_holm"] = holm_adjust(
            result.loc[indices, "p_value_one_sided"]
        )
    result["significant_holm_0.05"] = result.p_value_holm.lt(0.05)
    return result


def choose_magnitude(frame: pd.DataFrame, requested: float):
    available = sorted(frame.loc[frame.channel.eq("steered"), "magnitude"].dropna().unique())
    positive = [float(value) for value in available if value > 0]
    if not positive:
        raise ValueError("No positive steering magnitude is available.")
    if requested in positive:
        return float(requested), False
    return min(positive, key=lambda value: (abs(value - requested), value)), True


def plot_quartile_lift(summary: pd.DataFrame, selected_magnitude: float, out_path: Path):
    selected_regimes = ["Declared", "Neutral (N=0)", f"Steered N={selected_magnitude:g}"]
    pairs = summary[["attribute", "group_a", "group_b"]].drop_duplicates()
    ncols = 2
    nrows = math.ceil(len(pairs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.6 * nrows), squeeze=False)
    axes = axes.ravel()
    colors = {"Neutral (N=0)": "tab:blue", "Declared": "tab:orange",
              f"Steered N={selected_magnitude:g}": "tab:green"}
    for ax, pair in zip(axes, pairs.itertuples(index=False)):
        subset = summary[
            summary.attribute.eq(pair.attribute)
            & summary.group_a.eq(pair.group_a)
            & summary.group_b.eq(pair.group_b)
            & summary.quartile.isin(QUARTILES)
            & summary.regime.isin(selected_regimes)
        ]
        for regime in selected_regimes:
            values = subset[subset.regime.eq(regime)].set_index("quartile").reindex(QUARTILES)
            if values.lift_over_chance.notna().any():
                yerr = np.vstack([
                    values.lift_over_chance - values.lift_ci95_low,
                    values.lift_ci95_high - values.lift_over_chance,
                ])
                ax.errorbar(
                    QUARTILES, values.lift_over_chance, yerr=yerr, marker="o",
                    capsize=3, color=colors[regime], label=regime,
                )
        ax.axhline(0, color="black", lw=1, alpha=0.6)
        group_a = str(pair.group_a).replace("$", r"\$")
        group_b = str(pair.group_b).replace("$", r"\$")
        ax.set(
            title=f"{pair.attribute}: {group_a} vs {group_b}",
            xlabel="Human separability quartile",
            ylabel="Nearest-group lift over chance",
        )
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8)
    for ax in axes[len(pairs):]:
        ax.axis("off")
    fig.suptitle("Profile recovery increases where human groups diverge?", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def safe_label(path: Path) -> str:
    parent = path.resolve().parent.name
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", parent).strip("-") or "model"


def write_report(path: Path, *, predictions_path: Path, selected_magnitude: float,
                 fallback: bool, accuracy: pd.DataFrame, trends: pd.DataFrame,
                 directional: pd.DataFrame, files: dict[str, Path]):
    regimes = ["Declared", "Neutral (N=0)", f"Steered N={selected_magnitude:g}"]
    q4 = accuracy[accuracy.quartile.eq("Q4") & accuracy.regime.isin(regimes)]
    q1 = accuracy[accuracy.quartile.eq("Q1") & accuracy.regime.isin(regimes)]
    all_rows = accuracy[accuracy.quartile.eq("All") & accuracy.regime.isin(regimes)]
    comparison = q4.merge(
        q1[["attribute", "group_a", "group_b", "regime", "lift_over_chance"]],
        on=["attribute", "group_a", "group_b", "regime"], suffixes=("_q4", "_q1"),
    ).merge(
        all_rows[["attribute", "group_a", "group_b", "regime", "lift_over_chance"]],
        on=["attribute", "group_a", "group_b", "regime"],
    )
    lines = [
        "# Specificity conditioned on human separability", "",
        f"- Nearest predictions: `{predictions_path.resolve()}`",
        f"- Steering comparison magnitude: `{selected_magnitude:g}`"
        + (" (nearest available)" if fallback else ""),
        "- Bootstrap unit: `qkey`", "",
        "Quartiles are computed within each attribute and exact human-group pair. "
        "For education, each of the three pairwise contrasts is therefore analyzed "
        "separately. Accuracy remains multiclass and chance is `1 / n_groups`.", "",
        "Because a pair-specific education panel retains only the two endpoint model "
        "profiles from an original three-group classification, its empirical neutral "
        "accuracy need not equal theoretical chance. Read lift together with the neutral "
        "curve; the all-class cross-group report remains the calibrated global baseline.", "",
        "## Q4 headline", "",
        "| Attribute | Human pair | Regime | n questions | Q1 lift | Q4 lift | All lift | Q4 95% CI | p(Q4 lift <= 0) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.attribute} | {row.group_a} vs {row.group_b} | {row.regime} | "
            f"{row.n_questions} | {row.lift_over_chance_q1:+.3f} | "
            f"{row.lift_over_chance_q4:+.3f} | {row.lift_over_chance:+.3f} | "
            f"[{row.lift_ci95_low:+.3f}, {row.lift_ci95_high:+.3f}] | "
            f"{row.bootstrap_p_lift_le_zero:.3g} |"
        )
    lines.extend(["", "## Continuous-gap trend", "",
                  "Positive slopes mean strict nearest-group accuracy increases with human gap.", "",
                  "| Attribute | Human pair | Regime | n | OR per gap SD | 95% slope CI | p(slope <= 0) |",
                  "| --- | --- | --- | ---: | ---: | --- | ---: |"])
    selected_trends = trends[trends.regime.isin(regimes)]
    for row in selected_trends.itertuples(index=False):
        lines.append(
            f"| {row.attribute} | {row.group_a} vs {row.group_b} | {row.regime} | "
            f"{row.n_questions} | {row.odds_ratio_per_gap_sd:.3f} | "
            f"[{row.slope_ci95_low:+.3f}, {row.slope_ci95_high:+.3f}] | "
            f"{row.cluster_robust_p_slope_le_zero:.3g} |"
        )
    lines.extend(["", "## Q4 directional tests", "",
                  "Negative mean differences and win rates above 0.5 support the intended direction.", "",
                  "| Attribute | Pair | Regime | Target | n | Mean own-minus-competitor | Fractional win rate | p_holm |",
                  "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"])
    selected_directional = directional[
        directional.quartile.eq("Q4") & directional.regime.isin(regimes)
    ]
    for row in selected_directional.itertuples(index=False):
        lines.append(
            f"| {row.attribute} | {row.group_a} vs {row.group_b} | {row.regime} | "
            f"{row.target_human_group} | {row.n_questions} | {row.mean_difference:+.3f} | "
            f"{row.fractional_win_rate:.3f} | {row.p_value_holm:.3g} |"
        )
    lines.extend(["", "## Files", ""])
    lines.extend(f"- [{name}]({file.name})" for name, file in files.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="nearest_group_predictions.csv from cross-group analysis")
    parser.add_argument("--cross-group", type=Path,
                        help="cross_group_distances.csv (default: beside predictions)")
    parser.add_argument(
        "--human-separability", type=Path,
        default=REPO_ROOT / "analyses/human_separability/human_separability_by_question.csv",
    )
    parser.add_argument("--comparison-magnitude", type=float, default=7.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    cross_path = args.cross_group or args.predictions.with_name("cross_group_distances.csv")
    for path in (args.predictions, cross_path, args.human_separability):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.bootstrap_iterations < 100:
        raise ValueError("Use at least 100 bootstrap iterations.")

    nearest = read_csv_required(args.predictions, {
        "qkey", "attribute", "channel", "magnitude", "regime", "model_class",
        "own_human_group", "correct", "correct_in_tie", "fractional_credit",
    })
    nearest["correct"] = parse_bool(nearest["correct"])
    nearest["correct_in_tie"] = parse_bool(nearest["correct_in_tie"])
    nearest["magnitude"] = pd.to_numeric(nearest["magnitude"], errors="coerce")
    cross = read_csv_required(cross_path, {
        "qkey", "attribute", "channel", "magnitude", "regime", "model_class",
        "human_group", "wasserstein",
    })
    cross["magnitude"] = pd.to_numeric(cross["magnitude"], errors="coerce")
    human_gap = read_csv_required(args.human_separability, {
        "qkey", "attribute", "group_a", "group_b", "wasserstein",
    })

    pairs = mapped_pairs()
    human_pairs = add_pairwise_quartiles(canonicalize_human_pairs(human_gap, pairs))
    predictions = attach_gaps_to_predictions(nearest, human_pairs)
    if predictions.empty:
        raise ValueError("No nearest predictions matched the human separability pairs.")

    accuracy = accuracy_summary(
        predictions, iterations=args.bootstrap_iterations, seed=args.seed
    )
    trends = trend_tests(predictions)
    directional_rows = build_directional_rows(cross, human_pairs)
    if directional_rows.empty:
        raise ValueError("No directional model-profile pairs could be reconstructed.")
    directional = directional_summary(directional_rows)
    selected_magnitude, fallback = choose_magnitude(accuracy, args.comparison_magnitude)

    output_dir = args.out_dir or (
        REPO_ROOT / "analyses/human_gap_specificity_reports" / safe_label(args.predictions)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "Predictions with human-gap strata": output_dir / "predictions_by_human_gap.csv",
        "Quartile accuracy and bootstrap CIs": output_dir / "accuracy_by_human_gap_quartile.csv",
        "Continuous-gap logistic trends": output_dir / "accuracy_gap_trends.csv",
        "Directional comparisons by quartile": output_dir / "directional_tests_by_human_gap_quartile.csv",
        "Quartile lift plot": output_dir / "lift_by_human_gap_quartile.png",
    }
    predictions.to_csv(files["Predictions with human-gap strata"], index=False)
    accuracy.to_csv(files["Quartile accuracy and bootstrap CIs"], index=False)
    trends.to_csv(files["Continuous-gap logistic trends"], index=False)
    directional.to_csv(files["Directional comparisons by quartile"], index=False)
    plot_quartile_lift(accuracy, selected_magnitude, files["Quartile lift plot"])
    report_path = output_dir / "report.md"
    write_report(
        report_path,
        predictions_path=args.predictions,
        selected_magnitude=selected_magnitude,
        fallback=fallback,
        accuracy=accuracy,
        trends=trends,
        directional=directional,
        files=files,
    )
    print(
        f"[ok] matched predictions={len(predictions)} "
        f"directional question-level comparisons={len(directional_rows)}"
    )
    print(f"[ok] report={report_path.resolve()}")


if __name__ == "__main__":
    main()
