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

import fit_final_lattice_universal_H15H20 as final_fit


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "results_final_lattice_H15H20"


def save_json(path: Path, payload):
    path.write_text(json.dumps(final_fit.to_native(payload), indent=2, sort_keys=True))


def load_summary():
    path = OUTDIR / "final_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing final fit summary: {path}")
    return json.loads(path.read_text())


def build_args():
    return SimpleNamespace(
        outdir=str(OUTDIR),
        nboot=0,
        n_jobs=1,
        seed=42,
        fix_beta=True,
        t_osc=1.5,
        use_analytic_h=True,
        use_stage2_calibration=True,
        plot=False,
        validation_H1p0=False,
        tp_min=None,
        tp_max=None,
        vw_folders=["v3", "v5", "v7", "v9"],
        rho_file="",
        fit_table="",
        best_beta="",
    )


def curve_from_betaH(beta_over_h_grid, theta, H_value, vw, coeffs, stage_payload, t_osc):
    perc = final_fit.PercolationCache()
    tp = np.asarray(
        [perc.get(float(H_value), float(beta_over_h), float(vw)) for beta_over_h in beta_over_h_grid],
        dtype=np.float64,
    )
    tp = np.maximum(tp, 1.0e-18)
    h_val = float(final_fit.h_theta(np.asarray([theta], dtype=np.float64))[0])
    f0 = coeffs["A0"] * np.power(h_val, coeffs["gamma0"])
    finf = coeffs["Ainf_used"] * np.power(h_val, coeffs["gammainf"])
    beta = float(stage_payload["beta"])
    x = tp * np.power(float(H_value), beta)
    transient = f0 / (np.power(x, 1.5) * (1.0 + np.power(x / max(float(stage_payload["t_c"]), 1.0e-12), float(stage_payload["r"]))))
    f_univ = finf / f0 + transient
    c_calib = float(stage_payload.get("c_calib", 1.0))
    return c_calib * np.power(tp / float(t_osc), 1.5) / f0 * f_univ


def load_training_df(summary):
    args = build_args()
    args.t_osc = float(summary["t_osc"])
    args.fix_beta = bool(summary["beta_fixed"])
    raw_df, _ = final_fit.build_dataframe(args)
    train_df_raw, _, _ = final_fit.split_datasets(raw_df, False)

    analytic = summary["analytic_amplitudes"]
    coeffs = {
        "A0": float(analytic["A0"]),
        "gamma0": float(analytic["gamma0"]),
        "Ainf": float(analytic["Ainf_used"]),
        "Ainf_used": float(analytic["Ainf_used"]),
        "gammainf": float(analytic["gammainf"]),
    }
    f0_table = final_fit.load_f0_table(final_fit.resolve_existing_file(final_fit.RHO_CANDIDATES))
    train_df = final_fit.attach_amplitudes(train_df_raw, coeffs, f0_table, None)
    return train_df.sort_values(["H", "theta", "v_w", "beta_over_H"]).reset_index(drop=True), coeffs


def plot_grid_for_H(df, coeffs, stage_payload, stage_name, H_value, outpath, t_osc):
    theta_values = np.sort(df["theta"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex=True, sharey=False)
    axes = axes.flatten()
    vw_values = np.sort(df["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}

    sub_h = df[np.isclose(df["H"], float(H_value))].copy()
    for ax, theta in zip(axes, theta_values):
        sub = sub_h[np.isclose(sub_h["theta"], float(theta))].copy()
        if sub.empty:
            ax.axis("off")
            continue
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
            xi_fit = curve_from_betaH(beta_grid, float(theta), float(H_value), float(vw), coeffs, stage_payload, t_osc)
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
        rf"Final lattice-only {stage_name} fit vs data, $H_*={H_value:.1f}$"
        + "\n"
        + rf"$\beta={stage_payload['beta']:.4f}$, $t_c={stage_payload['t_c']:.3f}$, $r={stage_payload['r']:.3f}$"
        + (rf", $c_{{\rm calib}}={stage_payload['c_calib']:.3f}$" if "c_calib" in stage_payload else ""),
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_separate(df, coeffs, stage_payload, stage_name, outdir, t_osc):
    outdir.mkdir(parents=True, exist_ok=True)
    vw_values = np.sort(df["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    rows = []
    for H_value in np.sort(df["H"].unique()):
        sub_h = df[np.isclose(df["H"], float(H_value))]
        for theta in np.sort(sub_h["theta"].unique()):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta))].copy()
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
                xi_fit = curve_from_betaH(beta_grid, float(theta), float(H_value), float(vw), coeffs, stage_payload, t_osc)
                ax.plot(beta_grid, xi_fit, color=colors[vw], lw=2.0, label=rf"fit $v_w={vw:.1f}$")
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_xlabel(r"$\beta/H_*$")
            ax.set_ylabel(r"$\xi$")
            ax.set_title(
                rf"{stage_name}: $H_*={H_value:.1f}$, $\theta={theta:.3f}$"
            )
            ax.legend(frameon=False, fontsize=7, ncol=2)
            fig.tight_layout()
            h_tag = f"H{H_value:.1f}".replace(".", "p")
            theta_tag = f"theta_{theta:.10f}".replace(".", "p")
            filepath = outdir / f"xi_vs_betaH_{stage_name}_{h_tag}_{theta_tag}.png"
            fig.savefig(filepath, dpi=220)
            plt.close(fig)
            rows.append({"stage": stage_name, "H": float(H_value), "theta": float(theta), "file": str(filepath)})
    return rows


if __name__ == "__main__":
    try:
        summary = load_summary()
        df, coeffs = load_training_df(summary)

        stage1 = summary["stage1"]
        stage2 = summary.get("stage2")

        files = []
        for H_value in [1.5, 2.0]:
            out = OUTDIR / f"xi_vs_betaH_stage1_H{str(H_value).replace('.', 'p')}.png"
            plot_grid_for_H(df, coeffs, stage1, "Stage 1", H_value, out, summary["t_osc"])
            files.append(str(out))
        stage1_rows = plot_separate(df, coeffs, stage1, "stage1", OUTDIR / "xi_vs_betaH_stage1_separate", summary["t_osc"])

        stage2_rows = []
        if stage2:
            for H_value in [1.5, 2.0]:
                out = OUTDIR / f"xi_vs_betaH_stage2_H{str(H_value).replace('.', 'p')}.png"
                plot_grid_for_H(df, coeffs, stage2, "Stage 2", H_value, out, summary["t_osc"])
                files.append(str(out))
            stage2_rows = plot_separate(df, coeffs, stage2, "stage2", OUTDIR / "xi_vs_betaH_stage2_separate", summary["t_osc"])

        index_df = pd.DataFrame(stage1_rows + stage2_rows)
        index_df.to_csv(OUTDIR / "xi_vs_betaH_final_index.csv", index=False)

        payload = {
            "status": "ok",
            "outdir": str(OUTDIR),
            "files": files,
            "index_csv": str(OUTDIR / "xi_vs_betaH_final_index.csv"),
            "stage1_rel_rmse": stage1["rel_rmse"],
            "stage2_rel_rmse": None if not stage2 else stage2["rel_rmse"],
        }
        save_json(OUTDIR / "xi_vs_betaH_final_plot_summary.json", payload)
        print(json.dumps(payload, sort_keys=True))
    except Exception as exc:
        trace = traceback.format_exc()
        payload = {"status": "error", "message": str(exc), "traceback": trace}
        save_json(OUTDIR / "_error_plot_final_betaH.json", payload)
        print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True))
        raise
