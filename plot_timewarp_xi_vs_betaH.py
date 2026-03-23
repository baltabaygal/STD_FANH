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

import fit_vw_timewarp as tw_fit


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "results_vw_timewarp"


def save_json(path: Path, payload):
    path.write_text(json.dumps(tw_fit.to_native(payload), indent=2, sort_keys=True))


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
        outdir=str(OUTDIR),
        plot=False,
        use_analytic_f0=False,
        reg_Finf=1.0e-3,
    )


def load_fit():
    path = OUTDIR / "params_optionB.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing time-warp fit JSON: {path}")
    return json.loads(path.read_text())


def prepare_data():
    args = make_args()
    df, _, theta_values, _ = tw_fit.prepare_dataframe(args, OUTDIR)
    return df, theta_values, args


def warp_scale(vw, fit_payload):
    key = f"{float(vw):.1f}"
    scales = fit_payload["scales"]
    if key in scales:
        return float(scales[key])
    for k, v in scales.items():
        if abs(float(k) - float(vw)) < 1.0e-8:
            return float(v)
    raise KeyError(f"Missing scale for v_w={vw}")


def xi_curve(beta_grid, theta, h_value, vw, f0, fit_payload, t_osc):
    perc = tw_fit.base_fit.PercolationCache()
    tp_grid = np.asarray([perc.get(float(h_value), float(beta), float(vw)) for beta in beta_grid], dtype=np.float64)
    tp_grid = np.maximum(tp_grid, 1.0e-18)
    scale = warp_scale(vw, fit_payload)
    tp_scaled = tp_grid * scale
    finf = float(fit_payload["F_inf"][f"{theta:.10f}"])
    plateau = np.power(tp_grid / float(t_osc), 1.5) * finf / max(float(f0) ** 2, 1.0e-18)
    transient = np.power(max(scale, 1.0e-18), -1.5) / (
        1.0 + np.power(tp_scaled / max(float(fit_payload["t_c"]), 1.0e-18), float(fit_payload["r"]))
    )
    return plateau + transient


def plot_grid_for_H(df, theta_values, fit_payload, H_value, t_osc, outpath):
    reps = np.asarray(theta_values, dtype=np.float64)
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex=True, sharey=False)
    axes = axes.flatten()
    vw_values = np.sort(df["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}

    sub_h = df[np.isclose(df["H"], float(H_value))].copy()
    for ax, theta in zip(axes, reps):
        sub = sub_h[np.isclose(sub_h["theta"], float(theta))].copy()
        if sub.empty:
            ax.axis("off")
            continue
        f0 = float(sub["F0"].iloc[0])
        for vw in vw_values:
            sub_vw = sub[np.isclose(sub["v_w"], float(vw))].sort_values("beta_over_H").copy()
            if sub_vw.empty:
                continue
            ax.scatter(
                sub_vw["beta_over_H"],
                sub_vw["xi"],
                s=28,
                color=colors[vw],
                alpha=0.9,
                label=rf"data $v_w={vw:.1f}$",
            )
            beta_grid = np.geomspace(sub_vw["beta_over_H"].min(), sub_vw["beta_over_H"].max(), 250)
            xi_fit = xi_curve(beta_grid, float(theta), float(H_value), float(vw), f0, fit_payload, t_osc)
            ax.plot(beta_grid, xi_fit, color=colors[vw], lw=2.0, alpha=0.95, label=rf"fit $v_w={vw:.1f}$")
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$\beta/H_*$")
        ax.set_ylabel(r"$\xi$")

    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        rf"Time-warp-only fit vs data, $H_*={H_value:.1f}$"
        + "\n"
        + rf"$r={fit_payload['r']:.3f}$, $t_c={fit_payload['t_c']:.3f}$",
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_separate(df, theta_values, fit_payload, t_osc, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    vw_values = np.sort(df["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    rows = []
    for H_value in np.sort(df["H"].unique()):
        sub_h = df[np.isclose(df["H"], float(H_value))]
        for theta in np.sort(theta_values):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta))].copy()
            if sub.empty:
                continue
            f0 = float(sub["F0"].iloc[0])
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            for vw in vw_values:
                sub_vw = sub[np.isclose(sub["v_w"], float(vw))].sort_values("beta_over_H").copy()
                if sub_vw.empty:
                    continue
                ax.scatter(
                    sub_vw["beta_over_H"],
                    sub_vw["xi"],
                    s=30,
                    color=colors[vw],
                    alpha=0.9,
                    label=rf"data $v_w={vw:.1f}$",
                )
                beta_grid = np.geomspace(sub_vw["beta_over_H"].min(), sub_vw["beta_over_H"].max(), 250)
                xi_fit = xi_curve(beta_grid, float(theta), float(H_value), float(vw), f0, fit_payload, t_osc)
                ax.plot(beta_grid, xi_fit, color=colors[vw], lw=2.0, label=rf"fit $v_w={vw:.1f}$")
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_xlabel(r"$\beta/H_*$")
            ax.set_ylabel(r"$\xi$")
            ax.set_title(rf"Time-warp only: $H_*={H_value:.1f}$, $\theta={theta:.3f}$")
            ax.legend(frameon=False, fontsize=7, ncol=2)
            fig.tight_layout()

            h_tag = f"H{H_value:.1f}".replace(".", "p")
            theta_tag = f"theta_{theta:.10f}".replace(".", "p")
            filepath = outdir / f"xi_vs_betaH_timewarp_{h_tag}_{theta_tag}.png"
            fig.savefig(filepath, dpi=220)
            plt.close(fig)
            rows.append({"H": float(H_value), "theta": float(theta), "file": str(filepath)})
    return pd.DataFrame(rows)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df, theta_values, args = prepare_data()
    fit_payload = load_fit()

    files = []
    for H_value in [1.5, 2.0]:
        out = OUTDIR / f"xi_vs_betaH_timewarp_H{str(H_value).replace('.', 'p')}.png"
        plot_grid_for_H(df, theta_values, fit_payload, H_value, args.t_osc, out)
        files.append(str(out))

    separate_dir = OUTDIR / "xi_vs_betaH_timewarp_separate"
    index_df = plot_separate(df, theta_values, fit_payload, args.t_osc, separate_dir)
    index_df.to_csv(OUTDIR / "xi_vs_betaH_timewarp_index.csv", index=False)

    summary = {
        "status": "ok",
        "files": files,
        "separate_dir": str(separate_dir),
        "index_csv": str(OUTDIR / "xi_vs_betaH_timewarp_index.csv"),
        "rel_rmse": fit_payload["rel_rmse"],
        "r": fit_payload["r"],
        "t_c": fit_payload["t_c"],
        "scales": fit_payload["scales"],
    }
    save_json(OUTDIR / "xi_vs_betaH_timewarp_plot_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
        save_json(OUTDIR / "xi_vs_betaH_timewarp_plot_error.json", payload)
        print(json.dumps(payload, sort_keys=True))
        raise
