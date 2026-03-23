#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "results_ode_fitready_hstar"
TABLES = [
    ROOT / "ode/analysis/data/dm_tp_fitready_H0p500.txt",
    ROOT / "ode/analysis/data/dm_tp_fitready_H1p000.txt",
    ROOT / "ode/analysis/data/dm_tp_fitready_H1p500.txt",
    ROOT / "ode/analysis/data/dm_tp_fitready_H2p000.txt",
]
DPI = 220


def load_fitready(path: Path) -> pd.DataFrame:
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    cols = [
        "H",
        "t_star",
        "theta",
        "tp",
        "tp_over_tosc",
        "Ea3_PT",
        "Ea3_noPT",
        "fanh_PT",
        "fanh_noPT",
        "xi",
        "nsteps_PT",
        "nsteps_noPT",
    ]
    ncol = min(arr.shape[1], len(cols))
    df = pd.DataFrame(arr[:, :ncol], columns=cols[:ncol])
    return df


def interp_loglog(x: np.ndarray, y: np.ndarray, xgrid: np.ndarray) -> np.ndarray:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    x = np.asarray(x[mask], dtype=np.float64)
    y = np.asarray(y[mask], dtype=np.float64)
    if len(x) < 2:
        return np.full_like(xgrid, np.nan)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    return np.exp(np.interp(np.log(xgrid), np.log(x), np.log(y), left=np.nan, right=np.nan))


def gamma_vs_h(hvals: np.ndarray, yvals: np.ndarray) -> float:
    mask = np.isfinite(hvals) & np.isfinite(yvals) & (hvals > 0.0) & (yvals > 0.0)
    if np.sum(mask) < 3:
        return np.nan
    slope, _ = np.polyfit(np.log(hvals[mask]), np.log(yvals[mask]), deg=1)
    return float(slope)


def summarize_theta(sub: pd.DataFrame) -> dict:
    hvals = np.array(sorted(sub["H"].unique()), dtype=np.float64)
    curves = []
    lo = -np.inf
    hi = np.inf
    for h in hvals:
        hsub = sub[np.isclose(sub["H"], h, atol=1.0e-12)].sort_values("tp")
        x = hsub["tp"].to_numpy(dtype=np.float64)
        y = hsub["fanh_PT"].to_numpy(dtype=np.float64)
        curves.append((h, x, y))
        lo = max(lo, float(np.min(x)))
        hi = min(hi, float(np.max(x)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return {"mean_relstd": np.nan, "max_relstd": np.nan, "gamma_med": np.nan, "gamma_p16": np.nan, "gamma_p84": np.nan}
    tp_grid = np.geomspace(lo, hi, 180)
    arr = np.asarray([interp_loglog(x, y, tp_grid) for _, x, y in curves], dtype=np.float64)
    valid = np.sum(np.isfinite(arr), axis=0) >= 3
    if not np.any(valid):
        return {"mean_relstd": np.nan, "max_relstd": np.nan, "gamma_med": np.nan, "gamma_p16": np.nan, "gamma_p84": np.nan}
    arr = arr[:, valid]
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=1)
    relstd = std / np.maximum(mean, 1.0e-18)
    gammas = np.array([gamma_vs_h(hvals, arr[:, j]) for j in range(arr.shape[1])], dtype=np.float64)
    gammas = gammas[np.isfinite(gammas)]
    return {
        "mean_relstd": float(np.nanmean(relstd)),
        "max_relstd": float(np.nanmax(relstd)),
        "gamma_med": float(np.nanmedian(gammas)) if gammas.size else np.nan,
        "gamma_p16": float(np.nanpercentile(gammas, 16.0)) if gammas.size else np.nan,
        "gamma_p84": float(np.nanpercentile(gammas, 84.0)) if gammas.size else np.nan,
    }


def plot_overlay(df: pd.DataFrame, outpath: Path):
    theta_values = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sorted(df["H"].unique()))))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, theta in zip(axes.flat, theta_values):
        sub = df[np.isclose(df["theta"], theta, atol=5.0e-4)].copy()
        for color, (h, hsub) in zip(colors, sub.groupby("H", sort=True)):
            ax.plot(hsub["tp"], hsub["fanh_PT"], "o-", ms=2.6, lw=1.0, color=color, label=rf"$H={h:g}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$f_{\rm anh,PT}$")
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def plot_heatmap(summary_df: pd.DataFrame, value_col: str, title: str, outpath: Path):
    vals = summary_df.sort_values("theta")
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    bar = ax.bar(np.arange(len(vals)), vals[value_col].to_numpy(dtype=np.float64), color="tab:blue")
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels([f"{th:.3f}" for th in vals["theta"]], rotation=45)
    ax.set_xlabel(r"$\theta$")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in TABLES:
        if not path.exists():
            raise FileNotFoundError(f"Missing direct ODE fit-ready table: {path}")
        frames.append(load_fitready(path))
    df = pd.concat(frames, ignore_index=True)
    df = df[np.isfinite(df["theta"]) & np.isfinite(df["tp"]) & np.isfinite(df["fanh_PT"]) & np.isfinite(df["H"])].copy()
    df = df[df["fanh_PT"] > 0.0].copy()
    plot_overlay(df, OUTDIR / "fanh_overlay_vs_tp.png")

    rows = []
    for theta, sub in df.groupby("theta", sort=True):
        stats = summarize_theta(sub)
        rows.append({"theta": float(theta), **stats})
    summary_df = pd.DataFrame(rows).sort_values("theta").reset_index(drop=True)
    summary_df.to_csv(OUTDIR / "summary.csv", index=False)
    plot_heatmap(summary_df, "mean_relstd", r"Direct ODE mean rel. std across $H_*$", OUTDIR / "mean_relstd_by_theta.png")
    plot_heatmap(summary_df, "gamma_med", r"Direct ODE median $\gamma_H$ at fixed $t_p$", OUTDIR / "gamma_med_by_theta.png")

    summary = {
        "status": "ok",
        "worst_relstd": summary_df.sort_values("mean_relstd", ascending=False).iloc[0][["theta", "mean_relstd"]].to_dict(),
        "worst_gamma_abs": summary_df.assign(abs_gamma=lambda d: np.abs(d["gamma_med"])).sort_values("abs_gamma", ascending=False).iloc[0][["theta", "gamma_med"]].to_dict(),
        "mean_relstd_overall": float(np.nanmean(summary_df["mean_relstd"])),
        "max_relstd_overall": float(np.nanmax(summary_df["max_relstd"])),
        "max_abs_gamma_overall": float(np.nanmax(np.abs(summary_df["gamma_med"]))),
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
