#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fit_vw_timewarp as tw
from infer_icrit_eff_dependencies import PercolationICache


ROOT = Path(__file__).resolve().parent
FITDIR = ROOT / "results_vw_fullshift_allvars"
OUTDIR = ROOT / "results_vw_fullshift_allvars_ieff"


def save_json(path: Path, payload):
    path.write_text(json.dumps(tw.to_native(payload), indent=2, sort_keys=True))


def make_args():
    return SimpleNamespace(
        rho="",
        vw_folders=["v3", "v5", "v7", "v9"],
        h_values=[1.5, 2.0],
        option="B",
        fix_tc=True,
        t_osc=1.5,
        tc0=1.5,
        tp_min=None,
        tp_max=None,
        nboot=0,
        n_jobs=1,
        outdir=str(FITDIR),
        plot=False,
        use_analytic_f0=False,
        reg_Finf=1.0e-3,
    )


def load_fit():
    p = FITDIR / "final_summary.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing fit summary: {p}")
    return json.loads(p.read_text())["fit"]


def prepare_data():
    args = make_args()
    df, _, _, _ = tw.prepare_dataframe(args, FITDIR)
    return df


def scale_for_row(row: pd.Series, fit: dict) -> float:
    refs = fit["refs"]
    log_scale = (
        fit["log_s0"]
        + fit["a_vw"] * math.log(max(float(row["v_w"]) / float(refs["vw_ref"]), 1.0e-18))
        + fit["b_H"] * math.log(max(float(row["H"]) / float(refs["H_ref"]), 1.0e-18))
        + fit["c_beta_over_H"] * math.log(max(float(row["beta_over_H"]) / float(refs["beta_ref"]), 1.0e-18))
    )
    return math.exp(log_scale)


def add_teff_ieff(df: pd.DataFrame, fit: dict) -> pd.DataFrame:
    icache = PercolationICache()
    out = df.copy()
    out["s_eff"] = out.apply(lambda row: scale_for_row(row, fit), axis=1)
    out["t_eff"] = out["tp"] * out["s_eff"]
    out["I_eff"] = [
        icache.get(float(h), float(bh), float(vw)).eval(float(teff))
        for h, bh, vw, teff in zip(
            out["H"].to_numpy(dtype=np.float64),
            out["beta_over_H"].to_numpy(dtype=np.float64),
            out["v_w"].to_numpy(dtype=np.float64),
            out["t_eff"].to_numpy(dtype=np.float64),
        )
    ]
    out["F_false_eff"] = np.exp(-out["I_eff"].to_numpy(dtype=np.float64))
    out = out[np.isfinite(out["I_eff"]) & (out["I_eff"] > 0.0)].copy()
    out["teff_over_tp"] = out["t_eff"] / out["tp"]
    return out


def monotonicity_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (vw, h), sub in df.groupby(["v_w", "H"]):
        sub = sub.sort_values("beta_over_H").copy()
        for theta, thsub in sub.groupby("theta"):
            thsub = thsub.sort_values("beta_over_H")
            beta = thsub["beta_over_H"].to_numpy(dtype=np.float64)
            ieff = thsub["I_eff"].to_numpy(dtype=np.float64)
            teff = thsub["t_eff"].to_numpy(dtype=np.float64)
            if len(beta) < 2:
                continue
            dI = np.diff(ieff)
            dt = np.diff(teff)
            rows.append(
                {
                    "v_w": float(vw),
                    "H": float(h),
                    "theta": float(theta),
                    "I_monotone_increasing": bool(np.all(dI >= -1.0e-12)),
                    "I_monotone_decreasing": bool(np.all(dI <= 1.0e-12)),
                    "t_eff_monotone_increasing": bool(np.all(dt >= -1.0e-12)),
                    "t_eff_monotone_decreasing": bool(np.all(dt <= 1.0e-12)),
                    "I_eff_min": float(np.min(ieff)),
                    "I_eff_max": float(np.max(ieff)),
                    "teff_over_tp_min": float(np.min(thsub["teff_over_tp"])),
                    "teff_over_tp_max": float(np.max(thsub["teff_over_tp"])),
                }
            )
    return pd.DataFrame(rows)


def plot_scatter(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    sc = axes[0].scatter(df["beta_over_H"], df["I_eff"], c=df["v_w"], cmap="viridis", s=22)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\beta/H_*$")
    axes[0].set_ylabel(r"$I_{\rm eff}$")
    axes[0].set_title(r"Implied $I_{\rm eff}$ vs $\beta/H_*$")
    axes[1].scatter(df["beta_over_H"], df["teff_over_tp"], c=df["v_w"], cmap="viridis", s=22)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$\beta/H_*$")
    axes[1].set_ylabel(r"$t_{\rm eff}/t_p$")
    axes[1].set_title(r"$t_{\rm eff}/t_p$ vs $\beta/H_*$")
    axes[2].scatter(df["beta_over_H"], df["F_false_eff"], c=df["v_w"], cmap="viridis", s=22)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"$\beta/H_*$")
    axes[2].set_ylabel(r"$F_{\rm false}(t_{\rm eff})$")
    axes[2].set_title(r"Implied false-vacuum fraction")
    for ax in axes:
        ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.03)
    cbar.set_label(r"$v_w$")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_by_vw_h(df: pd.DataFrame, value_col: str, ylabel: str, title: str, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True, sharey=False)
    axes = axes.flatten()
    combos = sorted(df[["v_w", "H"]].drop_duplicates().itertuples(index=False, name=None))
    theta_vals = sorted(df["theta"].unique())
    cmap = plt.get_cmap("plasma")
    colors = {theta: cmap(i / max(len(theta_vals) - 1, 1)) for i, theta in enumerate(theta_vals)}
    for ax, (vw, h) in zip(axes, combos):
        sub = df[np.isclose(df["v_w"], vw) & np.isclose(df["H"], h)].copy()
        for theta in theta_vals:
            thsub = sub[np.isclose(sub["theta"], theta)].sort_values("beta_over_H")
            if thsub.empty:
                continue
            ax.plot(thsub["beta_over_H"], thsub[value_col], "o-", ms=3.2, lw=1.2, color=colors[theta], label=rf"$\theta={theta:.3f}$")
        ax.set_xscale("log")
        if np.all(sub[value_col] > 0.0):
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$v_w={vw:.1f},\ H_*={h:.1f}$")
        ax.set_xlabel(r"$\beta/H_*$")
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fit = load_fit()
    df = prepare_data()
    df = add_teff_ieff(df, fit)
    df.to_csv(OUTDIR / "implied_ieff_dataset.csv", index=False)

    mono = monotonicity_summary(df)
    mono.to_csv(OUTDIR / "monotonicity_summary.csv", index=False)

    plot_scatter(df, OUTDIR / "ieff_scatter.png")
    plot_by_vw_h(df, "I_eff", r"$I_{\rm eff}$", r"Implied $I_{\rm eff}$ from direct $t_{\rm eff}$ fit", OUTDIR / "ieff_by_vw_h.png")
    plot_by_vw_h(df, "teff_over_tp", r"$t_{\rm eff}/t_p$", r"Direct fitted $t_{\rm eff}/t_p$", OUTDIR / "teff_over_tp_by_vw_h.png")
    plot_by_vw_h(df, "F_false_eff", r"$F_{\rm false}(t_{\rm eff})$", r"Implied false-vacuum fraction at $t_{\rm eff}$", OUTDIR / "ffalse_by_vw_h.png")

    old_summary_path = ROOT / "results_icrit_eff" / "final_summary.json"
    old_summary = json.loads(old_summary_path.read_text()) if old_summary_path.exists() else None

    summary = {
        "status": "ok",
        "n_points_used": int(len(df)),
        "teff_over_tp_range": [float(df["teff_over_tp"].min()), float(df["teff_over_tp"].max())],
        "I_eff_range": [float(df["I_eff"].min()), float(df["I_eff"].max())],
        "F_false_eff_range": [float(df["F_false_eff"].min()), float(df["F_false_eff"].max())],
        "I_eff_median": float(np.median(df["I_eff"])),
        "F_false_eff_median": float(np.median(df["F_false_eff"])),
        "I_monotone_increasing_fraction": float(np.mean(mono["I_monotone_increasing"].astype(float))) if not mono.empty else None,
        "I_monotone_decreasing_fraction": float(np.mean(mono["I_monotone_decreasing"].astype(float))) if not mono.empty else None,
        "I_monotone_either_fraction": float(np.mean((mono["I_monotone_increasing"] | mono["I_monotone_decreasing"]).astype(float))) if not mono.empty else None,
        "teff_monotone_increasing_fraction": float(np.mean(mono["t_eff_monotone_increasing"].astype(float))) if not mono.empty else None,
        "teff_monotone_decreasing_fraction": float(np.mean(mono["t_eff_monotone_decreasing"].astype(float))) if not mono.empty else None,
        "teff_monotone_either_fraction": float(np.mean((mono["t_eff_monotone_increasing"] | mono["t_eff_monotone_decreasing"]).astype(float))) if not mono.empty else None,
        "direct_teff_fit": {
            "s0": fit["s0"],
            "a_vw": fit["a_vw"],
            "b_H": fit["b_H"],
            "c_beta_over_H": fit["c_beta_over_H"],
            "r": fit["r"],
            "rel_rmse": fit["rel_rmse"],
        },
    }
    if old_summary is not None:
        summary["previous_inverse_icrit"] = {
            "icrit_eff_range": old_summary.get("icrit_eff_range"),
            "teff_over_tp_range": old_summary.get("teff_over_tp_range"),
            "best_model_cleanH15H20": old_summary.get("best_model_cleanH15H20"),
        }
    save_json(OUTDIR / "final_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
        save_json(OUTDIR / "_error.json", payload)
        print(json.dumps(payload, sort_keys=True))
        raise
