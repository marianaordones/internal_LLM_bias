#!/usr/bin/env python
"""Select one steering magnitude per model on held-out tuning questions.

The script uses existing model-to-human distance CSVs and never runs a model.
Questions are split once at qkey level. Magnitude selection minimizes the
macro-average matching-group Wasserstein distance across demographic attributes
on the tuning split; evaluation rows remain untouched for paper figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = {
    "qwen": "",
    "llama": "_llama",
    "mistral": "_mistral",
}
MODEL_ORDER = list(MODEL_FILES)
ATTRIBUTE_ORDER = ["gender", "age", "education", "socioeco"]


def resolve_distance_file(analyses_dir: Path, model: str, suffix: str) -> Path:
    directory = analyses_dir / "model_reports" / model
    candidates = [directory / f"distance_to_human{suffix}.csv",
                  directory / "distance_to_human.csv"]
    for path in dict.fromkeys(candidates):
        if path.is_file():
            return path
    raise FileNotFoundError("Expected one of: " + ", ".join(map(str, candidates)))


def load_distances(analyses_dir: Path):
    frames = {}
    for model, suffix in MODEL_FILES.items():
        path = resolve_distance_file(analyses_dir, model, suffix)
        frame = pd.read_csv(path)
        required = {"qkey", "attribute", "channel", "class_label",
                    "magnitude", "wasserstein"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame["magnitude"] = pd.to_numeric(frame["magnitude"], errors="coerce")
        frames[model] = frame
    return frames


def split_qkeys(frames, *, tuning_fraction: float, seed: int):
    qkey_sets = [set(frame.qkey.unique()) for frame in frames.values()]
    common = sorted(set.intersection(*qkey_sets))
    if len(common) < 2:
        raise ValueError("Fewer than two qkeys are shared across model distance tables.")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(common, dtype=object)
    rng.shuffle(shuffled)
    n_tuning = round(len(shuffled) * tuning_fraction)
    n_tuning = min(max(n_tuning, 1), len(shuffled) - 1)
    tuning = set(shuffled[:n_tuning])
    return pd.DataFrame({
        "qkey": common,
        "split": ["tuning" if qkey in tuning else "evaluation" for qkey in common],
        "seed": seed,
        "tuning_fraction": tuning_fraction,
    })


def macro_curve(frame: pd.DataFrame, qkeys: set[str]):
    """Average classes/questions within attribute, then weight attributes equally."""
    steered = frame[
        frame.channel.eq("steered") & frame.qkey.isin(qkeys)
    ].copy()
    attribute_means = (
        steered.groupby(["magnitude", "attribute"], as_index=False).wasserstein.mean()
    )
    curve = (
        attribute_means.groupby("magnitude", as_index=False).wasserstein.mean()
        .rename(columns={"wasserstein": "macro_mean_distance"})
        .sort_values("magnitude")
    )
    return curve, attribute_means


def select_magnitudes(frames, split):
    tuning = set(split.loc[split.split.eq("tuning"), "qkey"])
    evaluation = set(split.loc[split.split.eq("evaluation"), "qkey"])
    selections, curves, heldout = [], [], []
    for model in MODEL_ORDER:
        frame = frames[model]
        tuning_curve, _ = macro_curve(frame, tuning)
        minimum = tuning_curve.macro_mean_distance.min()
        # A stable tie rule avoids preferring unnecessarily large interventions.
        selected = float(
            tuning_curve.loc[
                np.isclose(tuning_curve.macro_mean_distance, minimum), "magnitude"
            ].min()
        )
        available = sorted(tuning_curve.magnitude.astype(float))
        at_boundary = selected in {available[0], available[-1]}
        selections.append({
            "model": model,
            "selected_magnitude": selected,
            "selection_metric": "macro_mean_matching_group_wasserstein",
            "tuning_distance": minimum,
            "grid_min": min(available),
            "grid_max": max(available),
            "selected_at_grid_boundary": at_boundary,
            "tie_break": "smallest_magnitude",
        })
        tuning_curve.insert(0, "model", model)
        tuning_curve["split"] = "tuning"
        curves.append(tuning_curve)

        evaluation_frame = frame[frame.qkey.isin(evaluation)]
        for attribute in ATTRIBUTE_ORDER:
            values = evaluation_frame[evaluation_frame.attribute.eq(attribute)]
            neutral = values[
                values.channel.eq("steered") & values.magnitude.eq(0)
            ].wasserstein.mean()
            declared = values[values.channel.eq("declared")].wasserstein.mean()
            steered = values[
                values.channel.eq("steered") & values.magnitude.eq(selected)
            ].wasserstein.mean()
            heldout.append({
                "model": model,
                "attribute": attribute,
                "selected_magnitude": selected,
                "n_questions": values.qkey.nunique(),
                "neutral_distance": neutral,
                "declared_distance": declared,
                "steered_distance": steered,
                "declared_minus_neutral": declared - neutral,
                "steered_minus_neutral": steered - neutral,
            })
        evaluation_curve, _ = macro_curve(frame, evaluation)
        evaluation_curve.insert(0, "model", model)
        evaluation_curve["split"] = "evaluation"
        curves.append(evaluation_curve)
    return pd.DataFrame(selections), pd.concat(curves, ignore_index=True), pd.DataFrame(heldout)


def write_report(path: Path, split, selections, heldout):
    overall = (
        heldout.groupby(["model", "selected_magnitude"], as_index=False)[
            ["neutral_distance", "declared_distance", "steered_distance",
             "declared_minus_neutral", "steered_minus_neutral"]
        ].mean()
    )
    lines = [
        "# Steering-magnitude selection", "",
        f"- Split seed: `{int(split.seed.iloc[0])}`",
        f"- Tuning questions: `{split.split.eq('tuning').sum()}`",
        f"- Evaluation questions: `{split.split.eq('evaluation').sum()}`",
        "- Selection objective: lowest tuning-set matching-group Wasserstein distance, "
        "macro-averaged with equal weight per attribute.",
        "- Tie rule: smallest magnitude.", "",
        "| Model | Selected N | Tuning distance | Grid | Boundary? |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in selections.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.selected_magnitude:g} | {row.tuning_distance:.3f} | "
            f"{row.grid_min:g}--{row.grid_max:g} | "
            f"{'yes' if row.selected_at_grid_boundary else 'no'} |"
        )
    lines.extend(["", "## Held-out distance check", "",
                  "| Model | N | Neutral | Declared | Steered | Declared - neutral | Steered - neutral |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in overall.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.selected_magnitude:g} | {row.neutral_distance:.3f} | "
            f"{row.declared_distance:.3f} | {row.steered_distance:.3f} | "
            f"{row.declared_minus_neutral:+.3f} | {row.steered_minus_neutral:+.3f} |"
        )
    lines.extend(["", "A boundary selection is the best tested magnitude, not evidence "
                  "that the global optimum has been located.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyses-dir", type=Path, default=REPO_ROOT / "analyses")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "analyses/magnitude_selection")
    parser.add_argument("--tuning-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.tuning_fraction < 1:
        raise ValueError("--tuning-fraction must be strictly between 0 and 1.")

    frames = load_distances(args.analyses_dir)
    split = split_qkeys(frames, tuning_fraction=args.tuning_fraction, seed=args.seed)
    selections, curves, heldout = select_magnitudes(frames, split)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split.to_csv(args.out_dir / "qkey_split.csv", index=False)
    selections.to_csv(args.out_dir / "selected_magnitudes.csv", index=False)
    curves.to_csv(args.out_dir / "distance_curves_by_split.csv", index=False)
    heldout.to_csv(args.out_dir / "heldout_distance_by_attribute.csv", index=False)
    write_report(args.out_dir / "report.md", split, selections, heldout)
    print("[ok] " + ", ".join(
        f"{row.model}=N{row.selected_magnitude:g}"
        for row in selections.itertuples(index=False)
    ))
    print(f"[ok] selection report={args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
