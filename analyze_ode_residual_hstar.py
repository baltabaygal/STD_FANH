#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import collapse_and_fit_fanh_tosc as cf


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "results_ode_hstar_residual"
VW_VALUES = [0.3, 0.5, 0.7, 0.9]
TARGET_H = [0.5, 1.0, 1.5, 2.0]
GLOBAL_LATTICE_BETA = -0.0976177385713417
DPI = 220


def compute_fanh(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fanh"] = out["xi"].to_numpy(dtype=np.float64) * out["F0"].to_numpy(dtype=np.float64) / np.power(
        out["tp"].to_numpy(dtype=np.float64) / cf.T_OSC, 1.5
    )
    return out


def interp_loglog(x: np.ndarray, y: np.ndarray, xgrid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.full_like(xgrid, np.nan)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    lx = np.log(x)
    ly = np.log(y)
    return np.exp(np.interp(np.log(xgrid), lx, ly, left=np.nan, right=np.nan))


def fit_gamma_vs_h(hvals: np.ndarray, yvals: np.ndarray) -> float:
    mask = np.isfinite(hvals) & np.isfinite(yvals) & (hvals > 0.0) & (yvals > 0.0)
    if np.sum(mask) < 3:
        return np.nan
    x = np.log(hvals[mask])
    y = np.log(yvals[mask])
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope)


def summarize_theta(sub: pd.DataFrame) -> dict:
    hvals = np.array(sorted(sub["H"].unique()), dtype=np.float64)
    curves = []
    lo = -np.inf
    hi = np.inf
    for h in hvals:
        hsub = sub[np.isclose(sub["H"], h, atol=1.0e-12)].sort_values("x")
        x = hsub["x"].to_numpy(dtype=np.float64)
        y = hsub["fanh"].to_numpy(dtype=np.float64)
        curves.append((h, x, y))
        lo = max(lo, float(np.min(x)))
        hi = min(hi, float(np.max(x)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return {"mean_relstd": np.nan, "max_relstd": np.nan, "gamma_median": np.nan, "gamma_p16": np.nan, "gamma_p84": np.nan}
    xgrid = np.geomspace(lo, hi, 120)
    arr = []
    for _, x, y in curves:
        arr.append(interp_loglog(x, y, xgrid))
    arr = np.asarray(arr, dtype=np.float64)
    valid = np.sum(np.isfinite(arr), axis=0) >= 3
    if not np.any(valid):
        return {"mean_relstd": np.nan, "max_relstd": np.nan, "gamma_median": np.nan, "gamma_p16": np.nan, "gamma_p84": np.nan}
    arr = arr[:, valid]
    xgrid = xgrid[valid]
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=1)
    relstd = std / np.maximum(mean, 1.0e-18)
    gammas = np.array([fit_gamma_vs_h(hvals, arr[:, j]) for j in range(arr.shape[1])], dtype=np.float64)
    gammas = gammas[np.isfinite(gammas)]
    return {
        "xgrid": xgrid,
        "arr": arr,
        "mean_relstd": float(np.nanmean(relstd)),
        "max_relstd": float(np.nanmax(relstd)),
        "gamma_median": float(np.nanmedian(gammas)) if gammas.size else np.nan,
        "gamma_p16": float(np.nanpercentile(gammas, 16.0)) if gammas.size else np.nan,
        "gamma_p84": float(np.nanpercentile(gammas, 84.0)) if gammas.size else np.nan,
    }


def plot_overlay(df: pd.DataFrame, mode_label: str, outpath: Path):
    theta_values = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sorted(df["H"].unique()))))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, theta in zip(axes.flat, theta_values):
        sub = df[np.isclose(df["theta"], theta, atol=5.0e-4)].copy()
        for color, (h, hsub) in zip(colors, sub.groupby("H", sort=True)):
            ax.plot(hsub["x"], hsub["fanh"], "o-", ms=2.8, lw=1.0, color=color, label=rf"$H={h:g}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(mode_label)
        ax.set_ylabel("fanh")
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def plot_heatmap(summary_df: pd.DataFrame, value_col: str, title: str, outpath: Path):
    pivot = summary_df.pivot(index="theta", columns="vw", values=value_col).sort_index()
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    pcm = ax.pcolormesh(
        np.arange(len(pivot.columns) + 1),
        np.arange(len(pivot.index) + 1),
        pivot.to_numpy(dtype=np.float64),
        shading="auto",
        cmap="coolwarm",
    )
    ax.set_xticks(np.arange(len(pivot.columns)) + 0.5)
    ax.set_xticklabels([f"{float(c):.1f}" for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)) + 0.5)
    ax.set_yticklabels([f"{float(y):.3f}" for y in pivot.index])
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$\theta$")
    ax.set_title(title)
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def run_mode(ode_all: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = []
    for vw in VW_VALUES:
        df = ode_all[np.isclose(ode_all["vw"], vw, atol=1.0e-12)].copy()
        beta = 0.0
        if mode == "beta_best":
            beta = float(cf.find_best_beta(df, 120)["beta"])
            df = cf.compute_x(df, beta)
        elif mode == "beta_lattice":
            beta = GLOBAL_LATTICE_BETA
            df = cf.compute_x(df, beta)
        else:
            df = df.copy()
            df["x"] = df["tp"].to_numpy(dtype=np.float64)
        plot_overlay(df, r"$x=t_p$" if mode == "tp" else rf"$x=t_p H^{{{beta:.3f}}}$", OUTDIR / f"overlay_{mode}_vw{vw:.1f}.png")
        for theta, sub in df.groupby("theta", sort=True):
            stats = summarize_theta(sub)
            rows.append(
                {
                    "mode": mode,
                    "vw": vw,
                    "beta": beta,
                    "theta": float(theta),
                    "mean_relstd": stats["mean_relstd"],
                    "max_relstd": stats["max_relstd"],
                    "gamma_median": stats["gamma_median"],
                    "gamma_p16": stats["gamma_p16"],
                    "gamma_p84": stats["gamma_p84"],
                }
            )
    return pd.DataFrame(rows).sort_values(["vw", "theta"]).reset_index(drop=True)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rho_path = cf.resolve_first_existing(cf.RHO_CANDIDATES, "")
    ode_path = cf.resolve_first_existing([ROOT / "ode/xi_DM_ODE_results.txt"], "")
    if rho_path is None or ode_path is None:
        raise FileNotFoundError("Missing rho_noPT_data.txt or ode/xi_DM_ODE_results.txt.")
    f0_table = cf.load_f0_table(rho_path, TARGET_H)
    ode_frames = []
    for vw in VW_VALUES:
        sub = cf.load_optional_ode(ode_path, vw, TARGET_H)
        if sub is None or sub.empty:
            continue
        sub = cf.merge_f0(sub, f0_table)
        sub = compute_fanh(sub)
        ode_frames.append(sub)
    ode_all = pd.concat(ode_frames, ignore_index=True)

    tp_summary = run_mode(ode_all, "tp")
    beta_best_summary = run_mode(ode_all, "beta_best")
    beta_lattice_summary = run_mode(ode_all, "beta_lattice")
    tp_summary.to_csv(OUTDIR / "summary_tp.csv", index=False)
    beta_best_summary.to_csv(OUTDIR / "summary_beta_best.csv", index=False)
    beta_lattice_summary.to_csv(OUTDIR / "summary_beta_lattice.csv", index=False)

    plot_heatmap(tp_summary, "mean_relstd", r"ODE mean rel. std across $H_*$, $x=t_p$", OUTDIR / "heat_relstd_tp.png")
    plot_heatmap(tp_summary, "gamma_median", r"ODE median $\gamma_H$, $x=t_p$", OUTDIR / "heat_gamma_tp.png")
    plot_heatmap(beta_best_summary, "mean_relstd", r"ODE mean rel. std across $H_*$, $x=t_p H^{\beta_{\rm best}}$", OUTDIR / "heat_relstd_beta_best.png")
    plot_heatmap(beta_best_summary, "gamma_median", r"ODE median $\gamma_H$, $x=t_p H^{\beta_{\rm best}}$", OUTDIR / "heat_gamma_beta_best.png")
    plot_heatmap(beta_lattice_summary, "mean_relstd", r"ODE mean rel. std across $H_*$, $x=t_p H^{\beta_{\rm lattice}}$", OUTDIR / "heat_relstd_beta_lattice.png")

    summary = {
        "status": "ok",
        "main_findings": {
            "tp": {
                "worst_relstd": tp_summary.sort_values("mean_relstd", ascending=False).iloc[0][["vw", "theta", "mean_relstd"]].to_dict(),
                "worst_gamma_abs": tp_summary.assign(abs_gamma=lambda d: np.abs(d["gamma_median"])).sort_values("abs_gamma", ascending=False).iloc[0][["vw", "theta", "gamma_median"]].to_dict(),
            },
            "beta_best": {
                "worst_relstd": beta_best_summary.sort_values("mean_relstd", ascending=False).iloc[0][["vw", "theta", "mean_relstd", "beta"]].to_dict(),
                "worst_gamma_abs": beta_best_summary.assign(abs_gamma=lambda d: np.abs(d["gamma_median"])).sort_values("abs_gamma", ascending=False).iloc[0][["vw", "theta", "gamma_median", "beta"]].to_dict(),
            },
            "beta_lattice": {
                "worst_relstd": beta_lattice_summary.sort_values("mean_relstd", ascending=False).iloc[0][["vw", "theta", "mean_relstd"]].to_dict(),
            },
        },
        "global_lattice_beta": GLOBAL_LATTICE_BETA,
    }
    cf.save_json(OUTDIR / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
