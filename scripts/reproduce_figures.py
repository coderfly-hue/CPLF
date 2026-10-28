#!/usr/bin/env python3
"""Regenerate published-display versions of empirical Figures 4-8 from CSVs.

These are plot reproductions using deposited summaries or clearly labelled
transcriptions, not independent reruns of the original experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "CPLF LLM protocols io complete package" / "CPLF LLM protocols io FORMAL RELEASE"
PLOT_VALUES = ROOT / "paper" / "plot_values"
SUMMARY = FORMAL / "release_data" / "seed_level_summary.csv"
FEEDBACK = FORMAL / "reported_results" / "feedback_generation_summary.csv"
VALIDITY = FORMAL / "reported_results" / "profile_validity_summary.csv"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot4(output: Path) -> list[Path]:
    source = pd.read_csv(SUMMARY)
    source = source[(source.analysis == "overall_prediction") & source.metric.isin(["AUC", "F1"])]
    methods = ["NeuralCDM", "RCD", "HierCDF", "QIKT", "QRCL", "LaMP", "OPPU",
               "DMC-CDM", "DSMKT", "GCAKT", "LearnLens", "Adaptive G-UKT", "MCSKT", "CPLF-LLM"]
    datasets = ["OULAD", "ASSIST2009", "EdNet"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    y = np.arange(len(methods))
    for ax, metric in zip(axes, ["AUC", "F1"]):
        for j, (dataset, color) in enumerate(zip(datasets, colors)):
            subset = source[(source.metric == metric) & (source.dataset == dataset)].set_index("method")
            missing = set(methods) - set(subset.index)
            if missing:
                raise ValueError(f"Figure 4 {metric}/{dataset} lacks {sorted(missing)}")
            means = subset.loc[methods, "mean"].to_numpy(float)
            ax.barh(y + (j - 1) * .23, means, height=.21, color=color, label=dataset)
        ax.set_xlabel(metric)
        ax.grid(axis="x", alpha=.25)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, methods)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right", ncol=3)
    fig.suptitle("Prediction benchmark comparison (reported five-seed means)")
    save(fig, output)
    return [SUMMARY]


def plot5(output: Path) -> list[Path]:
    src = PLOT_VALUES / "profile_heatmap_display_transcription.csv"
    frame = pd.read_csv(src)
    if frame.shape != (12, 11):
        raise ValueError("Figure 5 transcription must be 12 students x 10 concepts")
    matrix = frame.iloc[:, 1:].to_numpy(dtype=float)
    if not np.isfinite(matrix).all() or not ((matrix >= 0) & (matrix <= 1)).all():
        raise ValueError("Figure 5 profile values outside [0,1]")
    validity = pd.read_csv(VALIDITY)
    fig, (ax, note) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [8, 2]})
    image = ax.imshow(matrix, cmap="GnBu", vmin=.25, vmax=.95, aspect="auto")
    ax.set_xticks(range(10), frame.columns[1:], rotation=35, ha="right")
    ax.set_yticks(range(12), frame.student)
    ax.set_xlabel("Selected concepts")
    ax.set_ylabel("Representative students")
    for row in range(12):
        for col in range(10):
            value = matrix[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if value > .72 else "black")
    fig.colorbar(image, ax=ax, fraction=.026, pad=.02, label="Displayed estimated mastery")
    note.axis("off")
    strings = ["Held-out profile validity", "Spearman rho (95% CI)"]
    for row in validity.itertuples():
        strings.append(f"{row.dataset}: {row.spearman_rho:.2f} [{row.ci_low:.2f}, {row.ci_high:.2f}]")
    note.text(.04, .55, "\n\n".join(strings), va="center", fontsize=11)
    fig.suptitle("Figure 5 display-value transcription (not raw learner records)", fontsize=13)
    save(fig, output)
    return [src, VALIDITY]


def plot6(output: Path) -> list[Path]:
    src = PLOT_VALUES / "sequence_length_table_transcription.csv"
    frame = pd.read_csv(src)
    xs = np.array([50, 100, 150, 200])
    fig, ax = plt.subplots(figsize=(9, 5))
    marks = ["o", "s", "^", "D", "p", "*"]
    for i, row in enumerate(frame.itertuples(index=False)):
        values = np.asarray(row[1:], float)
        ax.plot(xs, values, marker=marks[i % len(marks)], label=row[0], linewidth=2,
                markersize=9 if row[0] == "CPLF-LLM" else 6)
    ax.set_xticks(xs)
    ax.set_xlabel("Maximum sequence length")
    ax.set_ylabel("AUC")
    ax.set_ylim(.78, .87)
    ax.grid(alpha=.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Sequence-length sensitivity (manuscript table means)")
    save(fig, output)
    return [src]


def plot7(output: Path) -> list[Path]:
    frame = pd.read_csv(FEEDBACK)
    axes = ["Relevance", "Diagnostic consistency", "Actionability", "Personalization", "Leakage control"]
    cols = ["relevance_mean", "diagnostic_mean", "actionability_mean", "personalization_mean", "leakage_mean"]
    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False)
    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw={"polar": True})
    for row in frame.itertuples(index=False):
        entry = frame.loc[frame.method == row.method].iloc[0]
        scores = [float(entry[col]) for col in cols]
        scores[-1] = 1 - scores[-1]
        ax.plot(np.r_[angles, angles[0]], scores + [scores[0]], marker="o", label=row.method)
        ax.fill(np.r_[angles, angles[0]], scores + [scores[0]], alpha=.07)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles, axes)
    ax.set_ylim(.6, 1.0)
    ax.set_yticks([.6, .7, .8, .9, 1.0])
    ax.legend(bbox_to_anchor=(1.25, 1), loc="upper left")
    fig.suptitle("Feedback quality by method (published means)")
    save(fig, output)
    return [FEEDBACK]


def plot8(output: Path) -> list[Path]:
    src = PLOT_VALUES / "feedback_error_table_transcription.csv"
    frame = pd.read_csv(src)
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    before = frame.before_verification.to_numpy(float)
    after = frame.residual_after_verification.to_numpy(float)
    ax.bar(x - .18, before, .36, color="#4C78A8", label="Before verification")
    ax.bar(x + .18, after, .36, color="#2A9D75", label="Residual after verification")
    for pos, value in zip(x - .18, before):
        ax.text(pos, value + .4, f"{value:.1f}%", ha="center", fontsize=9)
    for pos, value in zip(x + .18, after):
        ax.text(pos, value + .4, f"{value:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x, frame.error_type, rotation=12, ha="right")
    ax.set_ylabel("Rate (% of original low-quality cases)")
    ax.set_ylim(0, 35)
    ax.grid(axis="y", alpha=.25)
    ax.set_axisbelow(True)
    ax.legend()
    fig.suptitle("Feedback error reduction (manuscript table values)")
    save(fig, output)
    return [src]


PLOTS = {4: (plot4, "archived five-seed summary"),
         5: (plot5, "two-decimal Figure 5 display transcription plus published validity summary"),
         6: (plot6, "manuscript sequence-length table transcription"),
         7: (plot7, "archived feedback-summary table"),
         8: (plot8, "manuscript feedback-error table transcription")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--figures", nargs="+", type=int, choices=sorted(PLOTS))
    parser.add_argument("--output-dir", default="outputs/reproduced_figures")
    args = parser.parse_args()
    selected = sorted(PLOTS) if args.all else args.figures
    target = Path(args.output_dir)
    report = {"scope": "published-display reproduction; not independent raw-experiment replay", "figures": {}}
    for number in selected:
        plotter, evidence = PLOTS[number]
        output = target / f"figure{number}.png"
        inputs = plotter(output)
        report["figures"][str(number)] = {
            "evidence_class": evidence,
            "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in inputs],
            "output": str(output), "sha256": digest(output),
        }
        print(f"Rebuilt displayed Figure {number}: {output}")
    target.mkdir(parents=True, exist_ok=True)
    (target / "figure_provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
