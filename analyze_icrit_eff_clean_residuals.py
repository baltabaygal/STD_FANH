#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results_icrit_eff"
OUTDIR = ROOT / "results_icrit_eff_clean_analysis"
DPI = 220


def load_inputs():
    df = pd.read_csv(BASE / "inferred_icrit_eff.csv")
    df = df[df["H"] >= 1.5].copy()
    model = json.loads((BASE / "model_comparison_cleanH15H20.json").read_text())["models"][0]
    return df, model


def apply_model(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    work = df.copy()
    Xcols = [np.ones(len(work), dtype=np.float64)]
    for pred in model["predictors"]:
        Xcols.append(np.log(work[pred].to_numpy(dtype=np.float64)))
    X = np.column_stack(Xcols)
    coef = np.asarray(model["coef"], dtype=np.float64)
    work["I_eff_fit"] = np.exp(X @ coef)
    work["rel_resid"] = (work["I_eff_fit"] - work["I_eff"]) / np.maximum(work["I_eff"], 1.0e-18)
    return work


def plot_heatmap(df: pd.DataFrame, columns: str, title: str, path: Path):
    pivot = df.pivot_table(index="theta", columns=columns, values="rel_resid", aggfunc="mean").sort_index()
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    pcm = ax.pcolormesh(
        np.arange(len(pivot.columns) + 1),
        np.arange(len(pivot.index) + 1),
        pivot.to_numpy(dtype=np.float64),
        cmap="coolwarm",
        vmin=-0.5,
        vmax=0.5,
        shading="auto",
    )
    ax.set_xticks(np.arange(len(pivot.columns)) + 0.5)
    labels = []
    for col in pivot.columns:
        if isinstance(col, (float, np.floating)):
            labels.append(f"{float(col):.3g}")
        else:
            labels.append(str(col))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(pivot.index)) + 0.5)
    ax.set_yticklabels([f"{float(t):.3f}" for t in pivot.index])
    ax.set_xlabel(columns)
    ax.set_ylabel(r"$\theta$")
    ax.set_title(title)
    fig.colorbar(pcm, ax=ax, label="relative residual")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_by_theta(df: pd.DataFrame, path: Path):
    stats = (
        df.groupby("theta", as_index=False)
        .agg(
            mean_rel_resid=("rel_resid", "mean"),
            mean_abs_rel_resid=("rel_resid", lambda s: float(np.mean(np.abs(s)))),
            std_rel_resid=("rel_resid", "std"),
        )
        .sort_values("theta")
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    theta = stats["theta"].to_numpy(dtype=np.float64)
    mean = stats["mean_rel_resid"].to_numpy(dtype=np.float64)
    std = stats["std_rel_resid"].fillna(0.0).to_numpy(dtype=np.float64)
    ax.errorbar(theta, mean, yerr=std, fmt="o-", color="tab:blue", capsize=3, label="mean ± std")
    ax.plot(theta, stats["mean_abs_rel_resid"], "s--", color="tab:red", label="mean |resid|")
    ax.axhline(0.0, color="black", lw=1.0, alpha=0.7)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("relative residual")
    ax.set_title(r"Best clean-$H_*$ $I_{\rm crit,eff}$ model residuals vs $\theta$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return stats


def plot_pred_vs_obs(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    sc = ax.scatter(df["I_eff"], df["I_eff_fit"], c=df["theta"], cmap="viridis", s=18)
    lo = min(df["I_eff"].min(), df["I_eff_fit"].min())
    hi = max(df["I_eff"].max(), df["I_eff_fit"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$I_{\rm crit,eff}$ inferred")
    ax.set_ylabel(r"$I_{\rm crit,eff}$ model")
    ax.set_title("Predicted vs inferred")
    fig.colorbar(sc, ax=ax, label=r"$\theta$")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df, model = load_inputs()
    df = apply_model(df, model)
    df.to_csv(OUTDIR / "clean_predictions.csv", index=False)

    plot_heatmap(df, "vw", r"Clean $H_*=1.5,2.0$ residuals vs $v_w$", OUTDIR / "residual_heatmap_theta_vw.png")
    plot_heatmap(df, "H", r"Clean residuals vs $H_*$", OUTDIR / "residual_heatmap_theta_H.png")
    plot_heatmap(df, "beta_over_H", r"Clean residuals vs $\beta/H_*$", OUTDIR / "residual_heatmap_theta_betaH.png")
    theta_stats = plot_by_theta(df, OUTDIR / "residual_vs_theta.png")
    plot_pred_vs_obs(df, OUTDIR / "predicted_vs_inferred.png")

    summary = {
        "status": "ok",
        "best_model": model,
        "mean_abs_rel_resid_overall": float(np.mean(np.abs(df["rel_resid"]))),
        "theta_worst_mean_abs_resid": theta_stats.sort_values("mean_abs_rel_resid", ascending=False).iloc[0].to_dict(),
        "theta_best_mean_abs_resid": theta_stats.sort_values("mean_abs_rel_resid", ascending=True).iloc[0].to_dict(),
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
