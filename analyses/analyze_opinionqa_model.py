#!/usr/bin/env python
"""Create a unified report for one demographic OpinionQA experiment CSV.

Pass one or more model results in the same run. Human-group mappings and
subgroup contrasts are intentionally kept near the top of the file so they can
be reviewed or changed without touching the analysis functions.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OQA = REPO_ROOT / "data/political_questions/OpinionQA/opinionqa.csv"
DEFAULT_QKEY_DICT = (
    REPO_ROOT / "data/political_questions/OpinionQA/refined_qkey_dict.json"
)

# Edit this mapping when a probe class should be compared with another
# OpinionQA subgroup. Classes absent from the mapping are reported and skipped.
# Format: model attribute -> (OpinionQA attribute, model class -> human group).
CLASS_TO_HUMAN = {
    "gender": ("SEX", {"male": "Male", "female": "Female"}),
    "age": ("AGE", {"adult": "30-49", "older_adult": "65+"}),
    "education": (
        "EDUCATION",
        {
            "some schooling": "Less than high school",
            "high school": "High school graduate",
            "college and more": "College graduate/some postgrad",
        },
    ),
    "socioeco": (
        "INCOME",
        {
            "low": "Less than $30,000",
            # OpinionQA splits the middle-income range into three groups. This
            # direct mapping can be changed or removed if only extremes are used.
            #"mid": "$50,000-$75,000",
            "high": "$100,000 or more",
        },
    ),
}

# Contrasts used in the model-gap versus human-gap scatter plots.
GAP_PAIRS = {
    "gender": ("SEX", ("male", "female"), ("Male", "Female")),
    "age": ("AGE", ("adult", "older_adult"), ("30-49", "65+")),
    "education": (
        "EDUCATION",
        ("some schooling", "college and more"),
        ("Less than high school", "College graduate/some postgrad"),
    ),
    "socioeco": (
        "INCOME",
        ("low", "high"),
        ("Less than $30,000", "$100,000 or more"),
    ),
}

GENDER_PATTERN = re.compile(
    r"\b(gender|sex|male|female|man|men|woman|women|boy|girl|"
    r"transgender|nonbinary|non-binary)\b",
    re.IGNORECASE,
)


def parse_sequence(value) -> list:
    """Read JSON/Python-list fields used by the experiment and OpinionQA CSVs."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        parsed = ast.literal_eval(value)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Expected a sequence, got {type(parsed).__name__}: {value!r}")
    return list(parsed)


def normalize(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Distribution must be a finite, non-empty vector.")
    if (values < 0).any():
        raise ValueError("Distribution contains negative probabilities.")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Distribution has zero probability mass.")
    return values / total


def align_distribution(values, ordinal_size: int) -> tuple[np.ndarray, int]:
    """Drop trailing non-ordinal choices and condition on ordinal responses.

    OpinionQA stores one ordinal value per substantive choice. Extra trailing
    probabilities correspond to refusal, don't-know, or other non-ordinal
    choices. They are removed and the remaining probabilities are renormalized.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < ordinal_size:
        raise ValueError(
            f"Distribution has {len(values)} entries but needs {ordinal_size} ordinal entries."
        )
    return normalize(values[:ordinal_size]), len(values) - ordinal_size


def wasserstein_ordinal(left, right, ordinal) -> float:
    ordinal = np.asarray(ordinal, dtype=float)
    left, _ = align_distribution(left, len(ordinal))
    right, _ = align_distribution(right, len(ordinal))
    order = np.argsort(ordinal, kind="stable")
    positions = ordinal[order]
    cumulative_gap = np.abs(np.cumsum(left[order]) - np.cumsum(right[order]))
    return float(np.sum(cumulative_gap[:-1] * np.diff(positions)))


def load_question_metadata(path: Path):
    records = json.loads(path.read_text(encoding="utf-8"))
    metadata = {}
    for qkey, record in records.items():
        ordinal = np.asarray(record.get("ordinal", []), dtype=float)
        if len(ordinal) < 2 or not np.isfinite(ordinal).all() or len(np.unique(ordinal)) < 2:
            continue
        options = list(record.get("options", []))
        if len(options) < len(ordinal):
            continue
        metadata[qkey] = {
            "ordinal": ordinal,
            "ordinal_options": options[: len(ordinal)],
            "trailing_options": options[len(ordinal) :],
            "question": record.get("original_qbody", record.get("refined_qbody", "")),
        }
    return metadata


def load_human(path: Path):
    human = {}
    for row in pd.read_csv(path).itertuples(index=False):
        human[(row.qkey, row.attribute, row.group)] = np.asarray(
            parse_sequence(row.responses), dtype=float
        )
    return human


def load_model(path: Path):
    demo = pd.read_csv(path)
    required = {
        "qkey", "attribute", "channel", "class_label", "magnitude",
        "response_distribution",
    }
    missing = required - set(demo.columns)
    if missing:
        raise ValueError(f"Result CSV is missing columns: {sorted(missing)}")
    demo["dist"] = demo["response_distribution"].map(
        lambda value: np.asarray(parse_sequence(value), dtype=float)
    )
    demo["magnitude"] = pd.to_numeric(demo["magnitude"], errors="coerce")
    return demo


def identify_model(demo: pd.DataFrame, result_path: Path) -> tuple[str, str]:
    profile = ""
    model_id = ""
    if "model_profile" in demo and demo["model_profile"].notna().any():
        profile = str(demo["model_profile"].dropna().iloc[0])
    if "model_id" in demo and demo["model_id"].notna().any():
        model_id = str(demo["model_id"].dropna().iloc[0])
    label = profile or result_path.stem.replace("demographic_opinionqa_", "")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-"), model_id


def validate_mapping(demo, human):
    human_groups = {(attribute, group) for _, attribute, group in human}
    present = set(demo[["attribute", "class_label"]].itertuples(index=False, name=None))
    unmapped, missing_human = [], []
    for attribute, model_class in sorted(present):
        mapping = CLASS_TO_HUMAN.get(attribute)
        if mapping is None or model_class not in mapping[1]:
            unmapped.append((attribute, model_class))
            continue
        human_attribute, groups = mapping
        human_group = groups[model_class]
        if (human_attribute, human_group) not in human_groups:
            missing_human.append((attribute, model_class, human_attribute, human_group))
    if missing_human:
        details = ", ".join(f"{a}/{c} -> {ha}/{hg}" for a, c, ha, hg in missing_human)
        raise ValueError(f"Configured human groups are absent from OpinionQA: {details}")
    return unmapped


def build_distance_table(demo, human, question_meta):
    rows, audit = [], {
        "model_rows": len(demo),
        "matched_rows": 0,
        "unmapped_rows": 0,
        "missing_human_rows": 0,
        "nonordinal_question_rows": 0,
        "invalid_distribution_rows": 0,
        "model_trailing_values_removed": 0,
        "human_trailing_values_removed": 0,
    }
    for row in demo.itertuples(index=False):
        mapping = CLASS_TO_HUMAN.get(row.attribute)
        if mapping is None or row.class_label not in mapping[1]:
            audit["unmapped_rows"] += 1
            continue
        meta = question_meta.get(row.qkey)
        if meta is None:
            audit["nonordinal_question_rows"] += 1
            continue
        human_attribute, group_map = mapping
        human_group = group_map[row.class_label]
        human_dist = human.get((row.qkey, human_attribute, human_group))
        if human_dist is None:
            audit["missing_human_rows"] += 1
            continue
        try:
            model_aligned, model_removed = align_distribution(row.dist, len(meta["ordinal"]))
            human_aligned, human_removed = align_distribution(human_dist, len(meta["ordinal"]))
            distance = wasserstein_ordinal(model_aligned, human_aligned, meta["ordinal"])
        except ValueError:
            audit["invalid_distribution_rows"] += 1
            continue
        audit["matched_rows"] += 1
        audit["model_trailing_values_removed"] += model_removed
        audit["human_trailing_values_removed"] += human_removed
        rows.append({
            "qkey": row.qkey,
            "attribute": row.attribute,
            "channel": row.channel,
            "class_label": row.class_label,
            "magnitude": row.magnitude,
            "human_attribute": human_attribute,
            "human_group": human_group,
            "wasserstein": distance,
        })
    return pd.DataFrame(rows), audit


def choose_magnitude(steered, requested: float):
    available = sorted(steered["magnitude"].dropna().unique())
    if not available:
        raise ValueError("The input has no steered rows with numeric magnitudes.")
    if requested in available:
        return float(requested), False
    return float(min(available, key=lambda value: (abs(value - requested), value))), True


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    ci = float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, ci


def plot_regime_bars(distance, selected_magnitude, out_path):
    declared = distance[distance.channel.eq("declared")]
    steered = distance[distance.channel.eq("steered")]
    regimes = [
        ("Neutral (N=0)", steered[steered.magnitude.eq(0)]),
        ("Declared", declared),
        (f"Steered N={selected_magnitude:g}", steered[steered.magnitude.eq(selected_magnitude)]),
    ]
    rows = []
    for (attribute, model_class), _ in declared.groupby(["attribute", "class_label"]):
        for regime, frame in regimes:
            values = frame[
                frame.attribute.eq(attribute) & frame.class_label.eq(model_class)
            ].wasserstein
            if len(values):
                rows.append({
                    "attribute": attribute,
                    "class_label": model_class,
                    "group": f"{attribute}\n{model_class}",
                    "regime": regime,
                    "mean": mean_ci(values)[0],
                    "ci95": mean_ci(values)[1],
                    "n": len(values),
                })
    summary = pd.DataFrame(rows)
    labels = list(dict.fromkeys(summary.group))
    x, width = np.arange(len(labels)), 0.27
    fig, ax = plt.subplots(figsize=(max(11, len(labels) * 1.15), 6))
    for index, (regime, _) in enumerate(regimes):
        values = summary[summary.regime.eq(regime)].set_index("group").reindex(labels)
        ax.bar(x + (index - 1) * width, values["mean"], width,
               yerr=values["ci95"], capsize=3, label=regime)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("Mean ordinal Wasserstein distance to humans (95% CI)")
    ax.set_title("Distance to matching human groups by regime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return summary


def paired_significance(distance, selected_magnitude):
    neutral = distance[
        distance.channel.eq("steered") & distance.magnitude.eq(0)
    ]
    conditions = [
        ("Declared", distance[distance.channel.eq("declared")]),
        (
            f"Steered N={selected_magnitude:g}",
            distance[
                distance.channel.eq("steered")
                & distance.magnitude.eq(selected_magnitude)
            ],
        ),
    ]
    rows = []
    keys = ["qkey", "attribute", "class_label"]
    for attribute in sorted(distance.attribute.unique()):
        baseline = neutral[neutral.attribute.eq(attribute)]
        for regime, frame in conditions:
            condition = frame[frame.attribute.eq(attribute)]
            merged = condition.merge(baseline, on=keys, suffixes=("_condition", "_neutral"))
            differences = merged.wasserstein_condition - merged.wasserstein_neutral
            p_value = np.nan
            if len(differences) and not np.allclose(differences, 0):
                try:
                    p_value = float(wilcoxon(differences).pvalue)
                except ValueError:
                    pass
            delta = float(differences.mean()) if len(differences) else np.nan
            rows.append({
                "attribute": attribute,
                "regime": regime,
                "n": len(differences),
                "delta_vs_neutral": delta,
                "p_value": p_value,
                "direction": "closer" if delta < 0 else ("farther" if delta > 0 else "unchanged"),
            })
    return pd.DataFrame(rows)


def plot_dose_curve(distance, selected_magnitude, out_path):
    steered = distance[distance.channel.eq("steered")]
    curve = steered.groupby(["attribute", "magnitude"], as_index=False).wasserstein.mean()
    rows = []
    fig, ax = plt.subplots(figsize=(9, 5))
    for attribute, values in curve.groupby("attribute"):
        values = values.sort_values("magnitude")
        ax.plot(values.magnitude, values.wasserstein, marker="o", label=attribute)
        optimum = values.loc[values.wasserstein.idxmin()]
        selected = values[values.magnitude.eq(selected_magnitude)]
        rows.append({
            "attribute": attribute,
            "best_magnitude": optimum.magnitude,
            "distance_at_best": optimum.wasserstein,
            "selected_magnitude": selected_magnitude,
            "distance_at_selected": selected.wasserstein.iloc[0] if len(selected) else np.nan,
        })
    ax.axvline(selected_magnitude, ls=":", color="gray", label="comparison magnitude")
    ax.set(xlabel="Steering magnitude N", ylabel="Mean ordinal Wasserstein distance to humans",
           title="Distance to humans by steering magnitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return pd.DataFrame(rows)


def model_pair_gap(group, class_a, class_b, question_meta):
    distributions = {row.class_label: row.dist for row in group.itertuples(index=False)}
    qkey = group.iloc[0].qkey
    if class_a not in distributions or class_b not in distributions or qkey not in question_meta:
        return np.nan
    try:
        return wasserstein_ordinal(
            distributions[class_a], distributions[class_b], question_meta[qkey]["ordinal"]
        )
    except ValueError:
        return np.nan


def plot_gender_separation(demo, question_meta, out_path):
    gender = demo[demo.attribute.eq("gender")]
    rows = []
    for (qkey, channel, magnitude), group in gender.groupby(
        ["qkey", "channel", "magnitude"], dropna=False
    ):
        question = question_meta.get(qkey, {}).get("question", "")
        rows.append({
            "qkey": qkey,
            "channel": channel,
            "magnitude": magnitude,
            "gender_question": bool(GENDER_PATTERN.search(question or "")),
            "gap": model_pair_gap(group, "male", "female", question_meta),
        })
    gaps = pd.DataFrame(rows).dropna(subset=["gap"])
    styles = {
        True: ("gender-related questions", "tab:blue", "tab:purple"),
        False: ("other questions", "tab:orange", "tab:red"),
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    for flag, (label, steered_color, declared_color) in styles.items():
        subset = gaps[gaps.gender_question.eq(flag)]
        steered = subset[subset.channel.eq("steered")].groupby("magnitude").gap.mean()
        declared = subset[subset.channel.eq("declared")].gap.mean()
        if len(steered):
            ax.plot(steered.index, steered.values, marker="o", color=steered_color,
                    label=f"Steered — {label}")
        if np.isfinite(declared):
            ax.axhline(declared, ls="--", color=declared_color,
                       label=f"Declared — {label}")
    ax.set(xlabel="Steering magnitude N", ylabel="Mean male–female Wasserstein gap",
           title="Male–female separation by question type")
    ax.grid(alpha=0.15)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return gaps


def plot_gender_topic_regimes(distance, question_meta, selected_magnitude, out_path):
    """Compare gender-to-human distances inside and outside gender topics."""
    gender = distance[distance.attribute.eq("gender")].copy()
    gender["gender_question"] = gender.qkey.map(
        lambda qkey: bool(
            GENDER_PATTERN.search(question_meta.get(qkey, {}).get("question", "") or "")
        )
    )
    regimes = [
        ("Neutral (N=0)", "steered", 0.0),
        ("Declared", "declared", np.nan),
        (f"Steered N={selected_magnitude:g}", "steered", selected_magnitude),
    ]
    records = []
    for topic_flag in (True, False):
        topic = "Gender-related questions" if topic_flag else "Non-gender questions"
        topic_rows = gender[gender.gender_question.eq(topic_flag)]
        for model_class in ("male", "female"):
            for regime, channel, magnitude in regimes:
                subset = topic_rows[
                    topic_rows.channel.eq(channel) & topic_rows.class_label.eq(model_class)
                ]
                if channel == "steered":
                    subset = subset[subset.magnitude.eq(magnitude)]
                values = subset.wasserstein
                if len(values):
                    mean, ci = mean_ci(values)
                    records.append({
                        "question_type": topic,
                        "gender_question": topic_flag,
                        "class_label": model_class,
                        "regime": regime,
                        "mean": mean,
                        "ci95": ci,
                        "n": len(values),
                    })

    summary = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    x, width = np.arange(2), 0.25
    for ax, (topic_flag, title) in zip(
        axes,
        ((True, "Gender-related questions"), (False, "Non-gender questions")),
    ):
        panel = summary[summary.gender_question.eq(topic_flag)]
        for index, ((regime, _, _), color) in enumerate(zip(regimes, colors)):
            values = panel[panel.regime.eq(regime)].set_index("class_label").reindex(
                ["male", "female"]
            )
            ax.bar(
                x + (index - 1) * width,
                values["mean"],
                width,
                yerr=values["ci95"],
                capsize=3,
                color=color,
                label=regime,
            )
        ax.set_xticks(x, ["male", "female"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.12)
    axes[0].set_ylabel("Mean ordinal Wasserstein distance to matching human group (95% CI)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Gender-profile distance to humans by question type and regime", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return summary


def human_pair_gaps(human, question_meta, oqa_attribute, group_a, group_b):
    output = {}
    for qkey, meta in question_meta.items():
        left = human.get((qkey, oqa_attribute, group_a))
        right = human.get((qkey, oqa_attribute, group_b))
        if left is None or right is None:
            continue
        try:
            output[qkey] = wasserstein_ordinal(left, right, meta["ordinal"])
        except ValueError:
            continue
    return output


def plot_gap_correlations(demo, human, question_meta, selected_magnitude, out_path):
    pairs = [(name, value) for name, value in GAP_PAIRS.items() if name in set(demo.attribute)]
    fig, axes = plt.subplots(len(pairs), 2, figsize=(13, 4.6 * len(pairs)), squeeze=False)
    rows = []
    regimes = [("declared", np.nan, "Declared"),
               ("steered", selected_magnitude, f"Steered N={selected_magnitude:g}")]
    for row_index, (attribute, (oqa_attribute, classes, groups)) in enumerate(pairs):
        human_gaps = human_pair_gaps(human, question_meta, oqa_attribute, *groups)
        for column_index, (channel, magnitude, title) in enumerate(regimes):
            subset = demo[demo.attribute.eq(attribute) & demo.channel.eq(channel)]
            subset = subset[subset.magnitude.isna()] if channel == "declared" else subset[
                subset.magnitude.eq(magnitude)
            ]
            model_gaps = {
                qkey: model_pair_gap(group, *classes, question_meta)
                for qkey, group in subset.groupby("qkey")
            }
            common = [qkey for qkey, value in model_gaps.items()
                      if qkey in human_gaps and np.isfinite(value)]
            x = np.asarray([human_gaps[qkey] for qkey in common])
            y = np.asarray([model_gaps[qkey] for qkey in common])
            pearson = (float(pearsonr(x, y).statistic)
                       if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else np.nan)
            spearman = (float(spearmanr(x, y).statistic)
                        if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else np.nan)
            mean_delta = float(np.mean(y - x)) if len(x) else np.nan
            ax = axes[row_index, column_index]
            ax.scatter(x, y, alpha=0.5, s=18)
            limit = max(x.max() if len(x) else 1, y.max() if len(y) else 1)
            ax.plot([0, limit], [0, limit], "k--", alpha=0.6)
            ax.set(
                title=f"{attribute} — {title}\nr={pearson:.2f}, ρ={spearman:.2f}, "
                      f"mean Δ={mean_delta:+.3f}, n={len(x)}",
                xlabel=f"Human gap: {groups[0]} vs {groups[1]}",
                ylabel=f"Model gap: {classes[0]} vs {classes[1]}",
            )
            rows.append({"attribute": attribute, "channel": channel, "magnitude": magnitude,
                         "n": len(x), "pearson": pearson, "spearman": spearman,
                         "mean_delta": mean_delta})
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_distance_panels(distance, out_path):
    declared = distance[distance.channel.eq("declared")]
    steered = distance[distance.channel.eq("steered")]
    attributes = list(dict.fromkeys(distance.attribute))
    ncols, nrows = 2, math.ceil(len(attributes) / 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.8 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, attribute in zip(axes, attributes):
        classes = list(dict.fromkeys(steered[steered.attribute.eq(attribute)].class_label))
        for index, model_class in enumerate(classes):
            color = f"C{index}"
            curve = steered[
                steered.attribute.eq(attribute) & steered.class_label.eq(model_class)
            ].groupby("magnitude").wasserstein.mean().sort_index()
            ax.plot(curve.index, curve.values, marker="o", color=color,
                    label=f"steered {model_class}")
            declared_values = declared[
                declared.attribute.eq(attribute) & declared.class_label.eq(model_class)
            ].wasserstein
            if len(declared_values):
                ax.axhline(declared_values.mean(), ls="--", color=color, alpha=0.85,
                           label=f"declared {model_class}")
        ax.set(title=attribute, xlabel="Steering magnitude N",
               ylabel="Mean ordinal Wasserstein distance to humans")
        ax.grid(alpha=0.12)
        ax.legend(fontsize=8)
    for ax in axes[len(attributes):]:
        ax.axis("off")
    fig.suptitle("Distance to human groups: steered curves and declared baselines", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_report(path, *, model_label, model_id, result_path, selected_magnitude,
                 used_fallback, audit, unmapped, tables, figures):
    lines = [
        f"# OpinionQA demographic report: {model_label}", "",
        f"- Result CSV: `{result_path}`",
        f"- Model ID: `{model_id or 'not recorded in CSV'}`",
        f"- Comparison magnitude: `{selected_magnitude:g}`"
        + (" (nearest available magnitude)" if used_fallback else ""),
        f"- Matched model/human rows: `{audit['matched_rows']}` / `{audit['model_rows']}`",
        f"- Rows without a mapped probe class: `{audit['unmapped_rows']}`",
        f"- Rows without the matching human distribution: `{audit['missing_human_rows']}`",
        f"- Rows excluded because the question was not usefully ordinal: "
        f"`{audit['nonordinal_question_rows']}`",
        f"- Rows excluded because distributions could not be aligned: "
        f"`{audit['invalid_distribution_rows']}`",
        f"- Trailing non-ordinal model values removed: "
        f"`{audit['model_trailing_values_removed']}`",
        f"- Trailing non-ordinal human values removed: "
        f"`{audit['human_trailing_values_removed']}`",
        "",
        "Distances are ordinal Wasserstein distances after conditioning both model and human "
        "distributions on the options that have ordinal positions. Trailing refusal, "
        "don't-know, or other non-ordinal choices are excluded and the retained mass is "
        "renormalized.", "",
    ]
    if unmapped:
        lines.extend(["## Unmapped model classes", ""])
        lines.extend(f"- `{attribute}/{model_class}`" for attribute, model_class in unmapped)
        lines.append("")
    lines.extend(["## Tables", ""])
    lines.extend(f"- [{name}]({file.name})" for name, file in tables.items())
    lines.extend(["", "## Figures", ""])
    lines.extend(f"- [{name}]({file.name})" for name, file in figures.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_one(result_path, *, opinionqa_path, qkey_dict_path, output_dir,
                comparison_magnitude):
    """Analyze one model CSV and return the generated report path."""
    demo = load_model(result_path)
    model_label, model_id = identify_model(demo, result_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    question_meta = load_question_metadata(qkey_dict_path)
    human = load_human(opinionqa_path)
    unmapped = validate_mapping(demo, human)
    distance, audit = build_distance_table(demo, human, question_meta)
    if distance.empty:
        raise ValueError("No model rows could be matched to OpinionQA human distributions.")

    selected_magnitude, used_fallback = choose_magnitude(
        distance[distance.channel.eq("steered")], comparison_magnitude
    )
    if used_fallback:
        print(f"[warn] {model_label}: magnitude {comparison_magnitude:g} is unavailable; "
              f"using nearest magnitude {selected_magnitude:g}.")

    tables = {
        "Per-question distances": output_dir / "distance_to_human.csv",
        "Regime means and 95% CIs": output_dir / "regime_summary_ci.csv",
        "Paired tests against neutral": output_dir / "paired_significance_tests.csv",
        "Best steering magnitudes": output_dir / "best_magnitudes.csv",
        "Gender separation by question": output_dir / "gender_separation_by_question.csv",
        "Gender distance by topic and regime": output_dir / "gender_topic_regime_summary.csv",
        "Model/human gap correlations": output_dir / "gap_correlations.csv",
    }
    figures = {
        "Distance to human groups with 95% CIs": output_dir / "distance_by_regime_ci.png",
        "Distance by steering magnitude": output_dir / "distance_by_magnitude.png",
        "Male–female separation by question type": output_dir / "gender_separation.png",
        "Gender distance by topic and regime": output_dir / "gender_topic_regimes.png",
        "Model subgroup gaps versus human subgroup gaps": output_dir / "gap_correlations.png",
        "Steered and declared distance panels": output_dir / "distance_panels.png",
    }

    distance.to_csv(tables["Per-question distances"], index=False)
    plot_regime_bars(distance, selected_magnitude, figures["Distance to human groups with 95% CIs"]).to_csv(
        tables["Regime means and 95% CIs"], index=False
    )
    paired_significance(distance, selected_magnitude).to_csv(
        tables["Paired tests against neutral"], index=False
    )
    plot_dose_curve(distance, selected_magnitude, figures["Distance by steering magnitude"]).to_csv(
        tables["Best steering magnitudes"], index=False
    )
    plot_gender_separation(demo, question_meta, figures["Male–female separation by question type"]).to_csv(
        tables["Gender separation by question"], index=False
    )
    plot_gender_topic_regimes(
        distance,
        question_meta,
        selected_magnitude,
        figures["Gender distance by topic and regime"],
    ).to_csv(tables["Gender distance by topic and regime"], index=False)
    plot_gap_correlations(
        demo, human, question_meta, selected_magnitude,
        figures["Model subgroup gaps versus human subgroup gaps"],
    ).to_csv(tables["Model/human gap correlations"], index=False)
    plot_distance_panels(distance, figures["Steered and declared distance panels"])

    report_path = output_dir / "report.md"
    write_report(
        report_path, model_label=model_label, model_id=model_id,
        result_path=result_path.resolve(), selected_magnitude=selected_magnitude,
        used_fallback=used_fallback, audit=audit, unmapped=unmapped,
        tables=tables, figures=figures,
    )
    print(f"[ok] model={model_label} matched_rows={audit['matched_rows']}")
    print(f"[ok] report={report_path.resolve()}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True,
                        help="One or more CSVs produced by demographic_opinionqa_experiment.py")
    parser.add_argument("--opinionqa", type=Path, default=DEFAULT_OQA)
    parser.add_argument("--qkey-dict", type=Path, default=DEFAULT_QKEY_DICT)
    parser.add_argument("--out-dir", type=Path,
                        help="Base output directory (default: analyses/model_reports)")
    parser.add_argument("--comparison-magnitude", type=float, default=7.0)
    args = parser.parse_args()

    for path in (*args.input, args.opinionqa, args.qkey_dict):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_base = args.out_dir or REPO_ROOT / "analyses/model_reports"
    for result_path in args.input:
        demo_header = pd.read_csv(result_path, nrows=1)
        model_label, _ = identify_model(demo_header, result_path)
        analyze_one(
            result_path,
            opinionqa_path=args.opinionqa,
            qkey_dict_path=args.qkey_dict,
            output_dir=output_base / model_label,
            comparison_magnitude=args.comparison_magnitude,
        )


if __name__ == "__main__":
    main()
