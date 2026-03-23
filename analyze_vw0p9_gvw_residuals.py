#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_PRED = ROOT / "results_vw0p9_model_with_gvw" / "predictions_with_gvw.csv"
DEFAULT_FIT = ROOT / "results_tosc_lattice_vw0p9_H1p0H1p5H2p0" / "collapse_and_fit_fanh" / "global_fit.json"
DEFAULT_OUTDIR = ROOT / "results_vw0p9_model_with_gvw_residuals"


def parse_args():
    p = argparse.ArgumentParser(description="Analyze residual clustering for the g(v_w) transport fit.")
    p.add_argument("--predictions", type=str, default=str(DEFAULT_PRED))
    p.add_argument("--fit-json", type=str, default=str(DEFAULT_FIT))
    p.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def to_native(obj):
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_native(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def save_json(path: Path, payload):
    path.write_text(json.dumps(to_native(payload), indent=2, sort_keys=True))


def build_dataframe(pred_path: Path, fit_path: Path) -> tuple[pd.DataFrame, float]:
    df = pd.read_csv(pred_path)
    fit = json.loads(fit_path.read_text())
    tc = float(fit["t_c"])
    df = df.copy()
    resid = df["xi_fit_gvw"].to_numpy(dtype=np.float64) - df["xi"].to_numpy(dtype=np.float64)
    sem = np.maximum(df["xi_sem"].to_numpy(dtype=np.float64), 1.0e-18)
    xi = np.maximum(df["xi"].to_numpy(dtype=np.float64), 1.0e-18)
    x = np.maximum(df["x"].to_numpy(dtype=np.float64), 1.0e-18)
    df["resid"] = resid
    df["frac_resid"] = resid / xi
    df["abs_frac_resid"] = np.abs(df["frac_resid"])
    df["pull"] = resid / sem
    df["log10_x_over_tc"] = np.log10(x / tc)
    df["x_over_tc"] = x / tc
    df["transition_region"] = np.abs(df["log10_x_over_tc"]) <= 0.3
    return df, tc


def grouped_summary(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(by, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: float(val) if isinstance(val, (int, float, np.floating, np.integer)) else val for col, val in zip(by, keys)}
        row.update(
            {
                "n_points": int(len(g)),
                "mean_frac_resid": float(np.mean(g["frac_resid"])),
                "mean_abs_frac_resid": float(np.mean(g["abs_frac_resid"])),
                "median_abs_frac_resid": float(np.median(g["abs_frac_resid"])),
                "rmse_frac": float(np.sqrt(np.mean(np.square(g["frac_resid"]))))
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_by_theta(df: pd.DataFrame, outdir: Path, dpi: int):
    agg = grouped_summary(df, ["theta"]).sort_values("theta")
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(agg["theta"], agg["mean_abs_frac_resid"], marker="o", lw=1.8, label="mean |frac resid|")
    ax.plot(agg["theta"], np.abs(agg["mean_frac_resid"]), marker="s", lw=1.5, label="|mean frac resid|")
    ax.grid(alpha=0.25)
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel("residual level")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "residual_vs_theta.png", dpi=dpi)
    plt.close(fig)
    return agg


def plot_by_beta(df: pd.DataFrame, outdir: Path, dpi: int):
    agg = grouped_summary(df, ["beta_over_H"]).sort_values("beta_over_H")
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(agg["beta_over_H"], agg["mean_abs_frac_resid"], marker="o", lw=1.8, label="mean |frac resid|")
    ax.plot(agg["beta_over_H"], np.abs(agg["mean_frac_resid"]), marker="s", lw=1.5, label="|mean frac resid|")
    ax.set_xscale("log")
    ax.grid(alpha=0.25)
    ax.set_xlabel(r"$\beta/H_*$")
    ax.set_ylabel("residual level")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "residual_vs_betaH.png", dpi=dpi)
    plt.close(fig)
    return agg


def plot_vs_transition(df: pd.DataFrame, outdir: Path, dpi: int):
    bins = np.linspace(df["log10_x_over_tc"].min(), df["log10_x_over_tc"].max(), 15)
    cats = pd.cut(df["log10_x_over_tc"], bins=bins, include_lowest=True)
    rows = []
    for interval, g in df.groupby(cats, observed=False):
        if len(g) == 0:
            continue
        rows.append(
            {
                "bin_center": float(0.5 * (interval.left + interval.right)),
                "n_points": int(len(g)),
                "mean_frac_resid": float(np.mean(g["frac_resid"])),
                "mean_abs_frac_resid": float(np.mean(g["abs_frac_resid"])),
                "median_abs_frac_resid": float(np.median(g["abs_frac_resid"])),
            }
        )
    agg = pd.DataFrame(rows).sort_values("bin_center")
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.plot(agg["bin_center"], agg["mean_abs_frac_resid"], marker="o", lw=1.8, label="mean |frac resid|")
    ax.plot(agg["bin_center"], np.abs(agg["mean_frac_resid"]), marker="s", lw=1.5, label="|mean frac resid|")
    ax.axvspan(-0.3, 0.3, color="grey", alpha=0.15, label=r"transition window $0.5<x/t_c<2$")
    ax.grid(alpha=0.25)
    ax.set_xlabel(r"$\log_{10}(x/t_c)$")
    ax.set_ylabel("residual level")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "residual_vs_logx_over_tc.png", dpi=dpi)
    plt.close(fig)
    return agg


def plot_heatmap(df: pd.DataFrame, row_col: str, col_col: str, value_col: str, outpath: Path, dpi: int):
    pivot = (
        df.groupby([row_col, col_col])[value_col]
        .mean()
        .reset_index()
        .pivot(index=row_col, columns=col_col, values=value_col)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    im = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{float(v):.3g}" for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{float(v):.3f}" for v in pivot.index])
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return pivot


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df, tc = build_dataframe(Path(args.predictions).resolve(), Path(args.fit_json).resolve())
    df.to_csv(outdir / "residual_table.csv", index=False)

    by_theta = plot_by_theta(df, outdir, args.dpi)
    by_beta = plot_by_beta(df, outdir, args.dpi)
    by_trans = plot_vs_transition(df, outdir, args.dpi)
    by_theta.to_csv(outdir / "residual_by_theta.csv", index=False)
    by_beta.to_csv(outdir / "residual_by_betaH.csv", index=False)
    by_trans.to_csv(outdir / "residual_by_logx_over_tc.csv", index=False)

    heat_theta_beta = plot_heatmap(
        df, "theta", "beta_over_H", "abs_frac_resid", outdir / "residual_heatmap_theta_betaH.png", args.dpi
    )
    heat_theta_vw = plot_heatmap(
        df, "theta", "v_w", "abs_frac_resid", outdir / "residual_heatmap_theta_vw.png", args.dpi
    )

    in_trans = df[df["transition_region"]]
    out_trans = df[~df["transition_region"]]
    transition_compare = {
        "tc": tc,
        "window_definition": "|log10(x/t_c)| <= 0.3",
        "in_transition_mean_abs_frac_resid": float(np.mean(in_trans["abs_frac_resid"])),
        "out_transition_mean_abs_frac_resid": float(np.mean(out_trans["abs_frac_resid"])),
        "in_transition_median_abs_frac_resid": float(np.median(in_trans["abs_frac_resid"])),
        "out_transition_median_abs_frac_resid": float(np.median(out_trans["abs_frac_resid"])),
        "in_transition_n": int(len(in_trans)),
        "out_transition_n": int(len(out_trans)),
    }

    summary = {
        "status": "ok",
        "source_predictions": str(Path(args.predictions).resolve()),
        "source_fit_json": str(Path(args.fit_json).resolve()),
        "transition_compare": transition_compare,
        "worst_theta_by_mean_abs_frac_resid": by_theta.sort_values("mean_abs_frac_resid", ascending=False).head(3).to_dict(orient="records"),
        "worst_betaH_by_mean_abs_frac_resid": by_beta.sort_values("mean_abs_frac_resid", ascending=False).head(5).to_dict(orient="records"),
        "outputs": {
            "residual_table": str(outdir / "residual_table.csv"),
            "residual_by_theta": str(outdir / "residual_by_theta.csv"),
            "residual_by_betaH": str(outdir / "residual_by_betaH.csv"),
            "residual_by_logx_over_tc": str(outdir / "residual_by_logx_over_tc.csv"),
            "residual_vs_theta": str(outdir / "residual_vs_theta.png"),
            "residual_vs_betaH": str(outdir / "residual_vs_betaH.png"),
            "residual_vs_logx_over_tc": str(outdir / "residual_vs_logx_over_tc.png"),
            "residual_heatmap_theta_betaH": str(outdir / "residual_heatmap_theta_betaH.png"),
            "residual_heatmap_theta_vw": str(outdir / "residual_heatmap_theta_vw.png"),
        },
    }
    save_json(outdir / "final_summary.json", summary)
    print(json.dumps(to_native(summary), sort_keys=True))


if __name__ == "__main__":
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        main()
    except Exception as exc:
        payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
        save_json(outdir / "_error.json", payload)
        print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True))
        raise
