#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fit_vw_amplitude as base_fit
import fit_vw_smearing_tests as smear
import plot_lattice_xi_vs_x_all_vw_quadwarped as quadwarp


OUTDIR = ROOT / "results_vw_quadwarped_fit"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(description="Plot raw xi(beta/H*) using the fitted quadratic log-time warp model.")
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--reference-vw", type=float, default=0.9)
    p.add_argument("--outdir", type=str, default=str(OUTDIR))
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def load_lattice_dataframe(outdir: Path, vw_tags, h_values):
    args = SimpleNamespace(
        rho="",
        vw_folders=vw_tags,
        h_values=h_values,
        tp_min=None,
        tp_max=None,
        bootstrap=0,
        n_jobs=1,
        reg_Finf=0.0,
        tc0=1.5,
        fix_tc=True,
        dpi=220,
        outdir=str(outdir),
    )
    outdir.mkdir(parents=True, exist_ok=True)
    df, _, _ = base_fit.prepare_dataframe(args, outdir)
    return df.sort_values(["v_w", "H", "theta", "tp"]).reset_index(drop=True)


def make_reference_map(df: pd.DataFrame, ref_vw: float, beta: float):
    out = {}
    ref = df[np.isclose(df["v_w"], float(ref_vw), atol=1.0e-12, rtol=0.0)].copy()
    for (h, theta), sub in ref.groupby(["H", "theta"], sort=True):
        sub = sub.sort_values("tp").copy()
        x_ref = sub["tp"].to_numpy(dtype=np.float64) * np.power(float(h), float(beta))
        out[(float(h), float(theta))] = {
            "beta_over_H": sub["beta_over_H"].to_numpy(dtype=np.float64),
            "tp": sub["tp"].to_numpy(dtype=np.float64),
            "x": x_ref,
            "xi": sub["xi"].to_numpy(dtype=np.float64),
        }
    return out


def predict_curve(cur: pd.DataFrame, params: dict, beta: float, ref_series: dict):
    tp = cur["tp"].to_numpy(dtype=np.float64)
    h = float(cur["H"].iloc[0])
    x_cur = quadwarp.warp_tp(tp, params["log_s"], params["b"], params["c"]) * np.power(h, float(beta))
    pred = smear.log_interp(ref_series["x"], ref_series["xi"], x_cur)
    return pred


def rel_rmse(y, y_fit):
    y = np.asarray(y, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(y_fit) & (y > 0.0)
    if np.count_nonzero(mask) == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square((y_fit[mask] - y[mask]) / np.maximum(y[mask], 1.0e-12)))))


def plot_main(df: pd.DataFrame, params: dict, ref_vw: float, beta: float, outdir: Path, dpi: int):
    ref_map = make_reference_map(df, ref_vw, beta)
    vw_values = np.sort(df["v_w"].unique())
    theta_values = np.sort(df["theta"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    rows = []

    for h_value in np.sort(df["H"].unique()):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
        axes = axes.ravel()
        sub_h = df[np.isclose(df["H"], float(h_value), atol=1.0e-12, rtol=0.0)].copy()
        for ax, theta in zip(axes, theta_values):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
            ref_series = ref_map.get((float(h_value), float(theta)))
            if ref_series is None:
                ax.axis("off")
                continue
            for vw in vw_values:
                cur = sub[np.isclose(sub["v_w"], float(vw), atol=1.0e-12, rtol=0.0)].sort_values("beta_over_H").copy()
                if cur.empty:
                    continue
                rec = params.get(f"{float(vw):.1f}", {"log_s": 0.0, "b": 1.0, "c": 0.0})
                pred = predict_curve(cur, rec, beta, ref_series)
                ax.scatter(cur["beta_over_H"], cur["xi"], s=22, color=colors[float(vw)], alpha=0.85)
                mask = np.isfinite(pred)
                if np.count_nonzero(mask) >= 2:
                    ax.plot(cur["beta_over_H"].to_numpy(dtype=np.float64)[mask], pred[mask], color=colors[float(vw)], lw=1.8)
                rows.append(
                    {
                        "H": float(h_value),
                        "theta": float(theta),
                        "v_w": float(vw),
                        "rel_rmse": rel_rmse(cur["xi"].to_numpy(dtype=np.float64), pred),
                    }
                )
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$\beta/H_*$")
            ax.set_ylabel(r"$\xi$")
        for ax in axes[len(theta_values) :]:
            ax.axis("off")
        handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
        labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
        fig.suptitle(
            rf"Quadratic-warp fit in raw $\xi(\beta/H_*)$, $H_*={h_value:.1f}$, $\beta={beta:.4f}$",
            y=0.995,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        tag = str(float(h_value)).replace(".", "p")
        fig.savefig(outdir / f"xi_vs_betaH_quadwarped_H{tag}.png", dpi=dpi)
        plt.close(fig)

    return pd.DataFrame(rows)


def plot_separate(df: pd.DataFrame, params: dict, ref_vw: float, beta: float, outdir: Path, dpi: int):
    ref_map = make_reference_map(df, ref_vw, beta)
    outdir.mkdir(parents=True, exist_ok=True)
    vw_values = np.sort(df["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    rows = []
    for h_value in np.sort(df["H"].unique()):
        sub_h = df[np.isclose(df["H"], float(h_value), atol=1.0e-12, rtol=0.0)].copy()
        for theta in np.sort(sub_h["theta"].unique()):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
            ref_series = ref_map.get((float(h_value), float(theta)))
            if ref_series is None:
                continue
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            for vw in vw_values:
                cur = sub[np.isclose(sub["v_w"], float(vw), atol=1.0e-12, rtol=0.0)].sort_values("beta_over_H").copy()
                if cur.empty:
                    continue
                rec = params.get(f"{float(vw):.1f}", {"log_s": 0.0, "b": 1.0, "c": 0.0})
                pred = predict_curve(cur, rec, beta, ref_series)
                ax.scatter(cur["beta_over_H"], cur["xi"], s=24, color=colors[float(vw)], alpha=0.9, label=rf"data $v_w={vw:.1f}$")
                mask = np.isfinite(pred)
                if np.count_nonzero(mask) >= 2:
                    ax.plot(cur["beta_over_H"].to_numpy(dtype=np.float64)[mask], pred[mask], color=colors[float(vw)], lw=2.0, label=rf"fit $v_w={vw:.1f}$")
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_xlabel(r"$\beta/H_*$")
            ax.set_ylabel(r"$\xi$")
            ax.set_title(rf"$H_*={h_value:.1f}$, $\theta={theta:.3f}$, quad warp")
            ax.legend(frameon=False, fontsize=7, ncol=2)
            fig.tight_layout()
            h_tag = f"H{h_value:.1f}".replace(".", "p")
            theta_tag = f"theta_{theta:.10f}".replace(".", "p")
            filename = f"xi_vs_betaH_quadwarped_{h_tag}_{theta_tag}.png"
            filepath = outdir / filename
            fig.savefig(filepath, dpi=dpi)
            plt.close(fig)
            rows.append({"H": float(h_value), "theta": float(theta), "file": str(filepath)})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_lattice_dataframe(outdir, args.vw_folders, args.h_values)
    params, fit_rows = quadwarp.fit_quadwarps(df, args.reference_vw, args.beta)

    rmse_df = plot_main(df, params, args.reference_vw, args.beta, outdir, args.dpi)
    separate_dir = outdir / "xi_vs_betaH_quadwarped_separate"
    index_df = plot_separate(df, params, args.reference_vw, args.beta, separate_dir, args.dpi)
    index_df.to_csv(outdir / "xi_vs_betaH_quadwarped_index.csv", index=False)

    summary = {
        "status": "ok",
        "beta": float(args.beta),
        "reference_vw": float(args.reference_vw),
        "fit_rows": fit_rows,
        "rmse_by_group": rmse_df.groupby("v_w", as_index=False)["rel_rmse"].mean().to_dict(orient="records"),
        "separate_dir": str(separate_dir),
        "index_csv": str(outdir / "xi_vs_betaH_quadwarped_index.csv"),
    }
    (outdir / "xi_vs_betaH_quadwarped_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        payload = {"status": "error", "message": str(exc)}
        outdir = Path(parse_args().outdir).resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "xi_vs_betaH_quadwarped_error.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(json.dumps(payload, sort_keys=True))
        sys.exit(1)
