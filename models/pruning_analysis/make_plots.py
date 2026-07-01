#!/usr/bin/env python3
"""
Pruning convergence figures: training error and RDF deviation vs retention × α.

  cd models/pruning_analysis
  python make_plots.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from load_data import (
    ALPHAS,
    discover_completed_training,
    discover_md_force_metrics,
    discover_rdf_metrics,
    load_statepoints,
)
from plot_style import ALPHA_COLORS, alpha_label, apply_style

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
FIG_DIR = OUTPUT_DIR / "figures"


def aggregate_training(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["fps_alpha", "retention_fraction", "retention_pct"])["training_rmse"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["fps_alpha", "retention_pct"])
    )


def aggregate_rdf(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            ["fps_alpha", "retention_fraction", "retention_pct", "case", "statepoint_id"]
        )["rdf_deviation_pct"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["case", "fps_alpha", "retention_pct"])
    )


def fig_training_rmse_vs_retention(train: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for alpha in ALPHAS:
        sub = train[train["fps_alpha"] == alpha].sort_values("retention_pct")
        if sub.empty:
            continue
        ax.errorbar(
            sub["retention_pct"],
            sub["mean"],
            yerr=sub["std"].fillna(0),
            marker="o",
            capsize=3,
            lw=1.5,
            color=ALPHA_COLORS[alpha],
            label=f"α={alpha_label(alpha)} (n={int(sub['count'].max())})",
        )
    ax.set_xlabel("Training set retention (%)")
    ax.set_ylabel("In-sample force RMSE (eV/Å)")
    ax.set_title("CN ChIMES training error vs pruning (completed models)")
    ax.legend(loc="best", fontsize=7)
    ax.set_xticks([1, 10])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_training_rmse_heatmap(train: pd.DataFrame, out_path: Path) -> None:
    pivot = train.pivot(index="fps_alpha", columns="retention_pct", values="mean").reindex(
        list(ALPHAS)
    )
    counts = train.pivot(index="fps_alpha", columns="retention_pct", values="count").reindex(
        list(ALPHAS)
    )
    pivot.index = [f"α={alpha_label(a)}" for a in pivot.index]
    annot = pivot.copy().astype(str)
    for i in range(len(pivot.index)):
        for j, col in enumerate(pivot.columns):
            v = pivot.iloc[i, j]
            n = counts.iloc[i, j] if col in counts.columns else 0
            annot.iloc[i, j] = f"{v:.3f}\n(n={int(n)})" if pd.notna(v) else ""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        pivot,
        annot=annot,
        fmt="",
        cmap="YlOrRd",
        cbar_kws={"label": "Force RMSE (eV/Å)"},
        ax=ax,
    )
    ax.set_xlabel("Retention (%)")
    ax.set_ylabel("FPS α")
    ax.set_title("Training error heatmap")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_rdf_deviation_vs_retention(rdf: pd.DataFrame, out_path: Path) -> None:
    """Mean RDF deviation (all cases) vs retention."""
    agg = (
        rdf.groupby(["fps_alpha", "retention_pct"])["rdf_deviation_pct"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for alpha in ALPHAS:
        sub = agg[agg["fps_alpha"] == alpha].sort_values("retention_pct")
        if sub.empty:
            continue
        ax.errorbar(
            sub["retention_pct"],
            sub["mean"],
            yerr=sub["std"].fillna(0),
            marker="o",
            capsize=3,
            lw=1.5,
            color=ALPHA_COLORS[alpha],
            label=f"α={alpha_label(alpha)}",
        )
    ax.set_xlabel("Training set retention (%)")
    ax.set_ylabel("RDF RMS deviation from full model (%)")
    ax.set_title("Mean RDF deviation vs pruning (all statepoints)")
    ax.legend(loc="best", fontsize=7)
    ax.set_xticks([1, 10])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_rdf_deviation_by_case(rdf: pd.DataFrame, out_path: Path) -> None:
    """2×5 panels: each statepoint, deviation vs retention per α."""
    sp_meta = {sp["case"]: sp for sp in load_statepoints()}
    agg = (
        rdf.groupby(["case", "fps_alpha", "retention_pct"])["rdf_deviation_pct"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    cases = sorted(agg["case"].unique())
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.5), sharey=True)
    for ax, case in zip(axes.flat, cases):
        sub_case = agg[agg["case"] == case]
        sp = sp_meta.get(case, {})
        title = f"#{case}  {int(sp.get('temperature_k', 0))}K"
        for alpha in ALPHAS:
            sub = sub_case[sub_case["fps_alpha"] == alpha].sort_values("retention_pct")
            if sub.empty:
                continue
            ax.errorbar(
                sub["retention_pct"],
                sub["mean"],
                yerr=sub["std"].fillna(0),
                marker="o",
                capsize=2,
                lw=1.2,
                color=ALPHA_COLORS[alpha],
                label=f"α={alpha_label(alpha)}",
            )
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Retention (%)")
        ax.set_xticks([1, 10])
        if ax in axes[:, 0]:
            ax.set_ylabel("RDF dev. from full (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("RDF deviation from full model per statepoint", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_rdf_heatmap_by_case(rdf: pd.DataFrame, out_path: Path) -> None:
    """Heatmap: statepoint × retention at each α (mean over replicates)."""
    agg = (
        rdf.groupby(["fps_alpha", "case", "retention_pct"])["rdf_deviation_pct"]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(1, len(ALPHAS), figsize=(3.2 * len(ALPHAS), 5), squeeze=False)
    vmax = agg["rdf_deviation_pct"].max() if not agg.empty else 1.0
    for ax, alpha in zip(axes[0], ALPHAS):
        sub = agg[agg["fps_alpha"] == alpha]
        if sub.empty:
            ax.set_visible(False)
            continue
        pivot = sub.pivot(index="case", columns="retention_pct", values="rdf_deviation_pct")
        pivot = pivot.reindex(sorted(agg["case"].unique()))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
            cbar=ax == axes[0][-1],
            ax=ax,
        )
        ax.set_title(f"α={alpha_label(alpha)}")
        ax.set_xlabel("Retention (%)")
        if ax == axes[0][0]:
            ax.set_ylabel("Statepoint case")
        else:
            ax.set_ylabel("")
    fig.suptitle("RDF RMS deviation from full model (%)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_rdf_cn_by_case(rdf_long: pd.DataFrame, out_path: Path) -> None:
    """CN pair only — deviation by case (bar summary at 1% and 10%)."""
    if "rdf_deviation_cn_pct" not in rdf_long.columns:
        return
    agg = (
        rdf_long.groupby(["case", "retention_pct", "fps_alpha"])["rdf_deviation_cn_pct"]
        .mean()
        .reset_index()
    )
    retentions = sorted(agg["retention_pct"].unique())
    fig, axes = plt.subplots(1, len(retentions), figsize=(5 * len(retentions), 4), squeeze=False)
    for ax, pct in zip(axes[0], retentions):
        sub = agg[agg["retention_pct"] == pct]
        pivot = sub.pivot(index="case", columns="fps_alpha", values="rdf_deviation_cn_pct")
        pivot = pivot.reindex(columns=list(ALPHAS))
        pivot.columns = [alpha_label(c) for c in pivot.columns]
        pivot.plot(kind="bar", ax=ax, color=[ALPHA_COLORS[a] for a in ALPHAS], width=0.8)
        ax.set_xlabel("Statepoint case")
        ax.set_ylabel("C–N RDF deviation (%)")
        ax.set_title(f"{pct}% retention")
        ax.legend(title="FPS α", fontsize=6, title_fontsize=7)
    fig.suptitle("C–N RDF deviation from full model by statepoint", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_md_force_vs_retention(md: pd.DataFrame, out_path: Path) -> None:
    agg = (
        md.groupby(["fps_alpha", "retention_pct"])["force_deviation_ev_a"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for alpha in ALPHAS:
        sub = agg[agg["fps_alpha"] == alpha].sort_values("retention_pct")
        if sub.empty:
            continue
        ax.errorbar(
            sub["retention_pct"],
            sub["mean"],
            yerr=sub["std"].fillna(0),
            label=alpha_label(alpha),
            color=ALPHA_COLORS[alpha],
            marker="o",
            capsize=3,
        )
    ax.set_xlabel("Training set retention (%)")
    ax.set_ylabel("Force RMSE vs full model (eV/Å)")
    ax.set_title("MD hold-out: pruned vs full ChIMES forces")
    ax.legend(title="FPS α", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_md_force_by_case(md: pd.DataFrame, out_path: Path) -> None:
    statepoints = load_statepoints()
    cases = sorted({sp["case"] for sp in statepoints if sp["id"] != "3.20.3percN_4.0gcc"})
    ncols = 5
    nrows = int(np.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), squeeze=False)
    agg = (
        md.groupby(["case", "fps_alpha", "retention_pct"])["force_deviation_ev_a"]
        .mean()
        .reset_index()
    )
    for ax, case in zip(axes.flat, cases):
        sub_case = agg[agg["case"] == case]
        for alpha in ALPHAS:
            sub = sub_case[sub_case["fps_alpha"] == alpha].sort_values("retention_pct")
            if sub.empty:
                continue
            ax.plot(
                sub["retention_pct"],
                sub["force_deviation_ev_a"],
                color=ALPHA_COLORS[alpha],
                marker="o",
                markersize=3,
                label=alpha_label(alpha),
            )
        ax.set_title(f"Case {case}")
        ax.set_xlabel("Retention (%)")
        ax.set_ylabel("ΔF RMSE")
    for ax in axes.flat[len(cases) :]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=7, title="FPS α")
    fig.suptitle("MD hold-out force deviation by statepoint", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run(output_dir: Path) -> None:
    apply_style()
    metrics_dir = output_dir / "metrics"
    fig_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    train_long = pd.DataFrame(discover_completed_training())
    rdf_long = pd.DataFrame(discover_rdf_metrics())
    train_long.to_csv(metrics_dir / "training_rmse_long.csv", index=False)
    rdf_long.to_csv(metrics_dir / "rdf_deviation_long.csv", index=False)

    if train_long.empty:
        print("No completed training runs found.")
        return

    train_agg = aggregate_training(train_long)
    train_agg.to_csv(metrics_dir / "training_rmse_vs_retention.csv", index=False)

    print(f"Training: {len(train_long)} completed models")
    print(train_long.groupby(["retention_pct", "fps_alpha"]).size().unstack(fill_value=0).to_string())

    fig_training_rmse_vs_retention(train_agg, fig_dir / "training_rmse_vs_retention.png")
    fig_training_rmse_heatmap(train_agg, fig_dir / "training_rmse_heatmap.png")

    if rdf_long.empty:
        print("No RDF comparisons available (need full + pruned rdf.dat).")
    else:
        rdf_agg = aggregate_rdf(rdf_long)
        rdf_agg.to_csv(metrics_dir / "rdf_deviation_vs_retention.csv", index=False)
        print(f"\nRDF pairs: {len(rdf_long)} model×statepoint comparisons")

        fig_rdf_deviation_vs_retention(rdf_long, fig_dir / "rdf_deviation_vs_retention.png")
        fig_rdf_deviation_by_case(rdf_long, fig_dir / "rdf_deviation_by_case.png")
        fig_rdf_heatmap_by_case(rdf_long, fig_dir / "rdf_deviation_heatmap_by_case.png")
        fig_rdf_cn_by_case(rdf_long, fig_dir / "rdf_cn_deviation_by_case.png")

    md_long = pd.DataFrame(discover_md_force_metrics())
    if not md_long.empty:
        md_long.to_csv(metrics_dir / "md_force_deviation_long.csv", index=False)
        print(f"\nMD hold-out force rows: {len(md_long)}")
        fig_md_force_vs_retention(md_long, fig_dir / "md_force_deviation_vs_retention.png")
        fig_md_force_by_case(md_long, fig_dir / "md_force_deviation_by_case.png")
    else:
        print("\nNo MD hold-out force metrics (run models/holdout/submit_eval.sh).")

    if not list(fig_dir.glob("*.png")):
        return

    print(f"\nWrote figures to {fig_dir}:")
    for p in sorted(fig_dir.glob("*.png")):
        print(f"  {p.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    run(parse_args().output_dir)


if __name__ == "__main__":
    main()
