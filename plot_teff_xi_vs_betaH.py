#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from refit_fanh_with_teff_model import load_clean_predictions, build_teff, fit_universal_model, T_OSC


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "results_fanh_teff_refit"
DPI = 220


def slug(x: float) -> str:
    return f"{x:.10f}".replace("-", "m").replace(".", "p")


def add_predictions(df: pd.DataFrame, fit_result: dict) -> pd.DataFrame:
    theta_values = fit_result["theta_values"]
    params = np.asarray(fit_result["params"], dtype=np.float64)
    t_c = float(params[0])
    r = float(params[1])
    finf = np.asarray(params[2:], dtype=np.float64)
    theta_index = np.array(
        [int(np.argmin(np.abs(theta_values - float(th)))) for th in df["theta"].to_numpy(dtype=np.float64)],
        dtype=np.int64,
    )
    f0 = df["F0"].to_numpy(dtype=np.float64)
    teff = df["t_eff_model"].to_numpy(dtype=np.float64)
    xi_fit = np.power(teff / T_OSC, 1.5) * finf[theta_index] / np.maximum(f0 * f0, 1.0e-18) + 1.0 / (
        1.0 + np.power(teff / max(t_c, 1.0e-12), r)
    )
    out = df.copy()
    out["xi_fit"] = xi_fit
    return out


def plot_grid(df: pd.DataFrame, H_value: float, outpath: Path):
    theta_values = sorted(df["theta"].unique())
    vw_values = sorted(df["vw"].unique())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(vw_values)))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, theta in zip(axes.flat, theta_values):
        sub = df[np.isclose(df["H"], H_value, atol=1.0e-12) & np.isclose(df["theta"], theta, atol=5.0e-4)].copy()
        sub = sub.sort_values(["vw", "beta_over_H"])
        for color, vw in zip(colors, vw_values):
            vsub = sub[np.isclose(sub["vw"], vw, atol=1.0e-12)].copy().sort_values("beta_over_H")
            if vsub.empty:
                continue
            ax.plot(vsub["beta_over_H"], vsub["xi"], "o", ms=3.0, color=color, label=rf"$v_w={vw:.1f}$")
            ax.plot(vsub["beta_over_H"], vsub["xi_fit"], "-", lw=1.5, color=color)
        ax.set_xlabel(r"$\beta/H_*$")
        ax.set_ylabel(r"$\xi$")
        ax.set_title(rf"$\theta={float(theta):.3f}$")
        ax.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def plot_separate(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    vw_values = sorted(df["vw"].unique())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(vw_values)))
    for H_value in sorted(df["H"].unique()):
        for theta in sorted(df["theta"].unique()):
            sub = df[np.isclose(df["H"], H_value, atol=1.0e-12) & np.isclose(df["theta"], theta, atol=5.0e-4)].copy()
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(6.0, 4.4))
            for color, vw in zip(colors, vw_values):
                vsub = sub[np.isclose(sub["vw"], vw, atol=1.0e-12)].copy().sort_values("beta_over_H")
                if vsub.empty:
                    continue
                ax.plot(vsub["beta_over_H"], vsub["xi"], "o", ms=3.2, color=color, label=rf"$v_w={vw:.1f}$")
                ax.plot(vsub["beta_over_H"], vsub["xi_fit"], "-", lw=1.6, color=color)
            ax.set_xlabel(r"$\beta/H_*$")
            ax.set_ylabel(r"$\xi$")
            ax.set_title(rf"$H_*={float(H_value):.1f},\ \theta={float(theta):.3f}$")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            name = f"xi_vs_betaH_teff_H{str(float(H_value)).replace('.','p')}_theta_{slug(float(theta))}.png"
            path = outdir / name
            fig.savefig(path, dpi=DPI)
            plt.close(fig)
            rows.append({"H": float(H_value), "theta": float(theta), "path": str(path)})
    return pd.DataFrame(rows)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    clean = load_clean_predictions()
    teff_df = build_teff(clean)
    fit_result = fit_universal_model(teff_df)
    pred = add_predictions(teff_df, fit_result)
    pred.to_csv(OUTDIR / "teff_xi_predictions.csv", index=False)

    plot_grid(pred, 1.5, OUTDIR / "xi_vs_betaH_teff_H1p5.png")
    plot_grid(pred, 2.0, OUTDIR / "xi_vs_betaH_teff_H2p0.png")
    index_df = plot_separate(pred, OUTDIR / "xi_vs_betaH_teff_separate")
    index_df.to_csv(OUTDIR / "xi_vs_betaH_teff_index.csv", index=False)

    summary = {
        "outdir": str(OUTDIR),
        "main_pngs": [
            str(OUTDIR / "xi_vs_betaH_teff_H1p5.png"),
            str(OUTDIR / "xi_vs_betaH_teff_H2p0.png"),
        ],
        "separate_dir": str(OUTDIR / "xi_vs_betaH_teff_separate"),
        "index_csv": str(OUTDIR / "xi_vs_betaH_teff_index.csv"),
        "fit": {
            "t_c": float(fit_result["params"][0]),
            "r": float(fit_result["params"][1]),
            "rel_rmse": float(fit_result["rel_rmse"]),
        },
    }
    (OUTDIR / "xi_vs_betaH_teff_plot_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
