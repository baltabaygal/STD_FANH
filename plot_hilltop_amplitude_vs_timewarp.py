#!/usr/bin/env python3
from __future__ import annotations

import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

import fit_vw_amplitude as amp_fit


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "results_model_compare"
H_VALUES = [1.5, 2.0]
THETA_VALUES = [2.3561944902, 2.8797932658]
VW_TAGS = ["v3", "v5", "v7", "v9"]


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(amp_fit.to_native(payload), indent=2, sort_keys=True))


def build_args():
    return SimpleNamespace(
        rho="",
        vw_folders=VW_TAGS,
        h_values=H_VALUES,
        tp_min=None,
        tp_max=None,
        bootstrap=0,
        n_jobs=1,
        reg_Finf=0.0,
        tc0=1.5,
        fix_tc=True,
        dpi=220,
        outdir=str(OUTDIR),
    )


def load_fits():
    amp_path = ROOT / "results_vw_amp/global_fit_optionB.json"
    warp_path = ROOT / "results_vw_timewarp/params_optionB.json"
    if not amp_path.exists():
        raise FileNotFoundError(f"Missing amplitude fit: {amp_path}")
    if not warp_path.exists():
        raise FileNotFoundError(f"Missing time-warp fit: {warp_path}")
    return json.loads(amp_path.read_text()), json.loads(warp_path.read_text())


def amplitude_curve(beta_grid, theta, H_value, vw, f0, fit_payload):
    perc = amp_fit.PercolationCache()
    tp_grid = np.asarray([perc.get(float(H_value), float(beta), float(vw)) for beta in beta_grid], dtype=np.float64)
    tp_grid = np.maximum(tp_grid, 1.0e-18)
    amp = fit_payload["A0"] + fit_payload["A1"] * np.power(float(vw), float(fit_payload["alpha"]))
    finf = float(fit_payload["F_inf"][f"{theta:.10f}"])
    plateau = np.power(tp_grid, 1.5) * finf / max(float(f0) ** 2, 1.0e-18)
    transient = 1.0 / (1.0 + np.power(tp_grid / max(float(fit_payload["t_c"]), 1.0e-18), float(fit_payload["r"])))
    return amp * (plateau + transient)


def timewarp_curve(beta_grid, theta, H_value, vw, f0, fit_payload, t_osc=1.5):
    perc = amp_fit.PercolationCache()
    tp_grid = np.asarray([perc.get(float(H_value), float(beta), float(vw)) for beta in beta_grid], dtype=np.float64)
    tp_grid = np.maximum(tp_grid, 1.0e-18)
    scale = float(fit_payload["scales"][f"{float(vw):.1f}"])
    tp_scaled = tp_grid * scale
    finf = float(fit_payload["F_inf"][f"{theta:.10f}"])
    plateau = np.power(tp_grid / t_osc, 1.5) * finf / max(float(f0) ** 2, 1.0e-18)
    transient = np.power(max(scale, 1.0e-18), -1.5) / (
        1.0 + np.power(tp_scaled / max(float(fit_payload["t_c"]), 1.0e-18), float(fit_payload["r"]))
    )
    return plateau + transient


def nearest_present(value, choices):
    arr = np.asarray(list(choices), dtype=np.float64)
    return float(arr[np.argmin(np.abs(arr - float(value)))])


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    args = build_args()
    df, _, _ = amp_fit.prepare_dataframe(args, OUTDIR)
    amp_payload, warp_payload = load_fits()

    cmap = plt.get_cmap("viridis")
    vw_values = np.sort(df["v_w"].unique())
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}

    theta_plot = [nearest_present(theta, df["theta"].unique()) for theta in THETA_VALUES]
    fig, axes = plt.subplots(2, 4, figsize=(18.0, 8.5), sharex=False, sharey=False)

    summary_rows = []
    for row_idx, theta in enumerate(theta_plot):
        for col_base, H_value in enumerate(H_VALUES):
            sub = df[(np.isclose(df["theta"], theta)) & (np.isclose(df["H"], float(H_value)))].copy()
            f0 = float(sub["F0"].iloc[0])

            ax_amp = axes[row_idx, 2 * col_base]
            ax_warp = axes[row_idx, 2 * col_base + 1]
            for vw in vw_values:
                sub_vw = sub[np.isclose(sub["v_w"], float(vw))].sort_values("beta_over_H").copy()
                if sub_vw.empty:
                    continue
                beta_grid = np.geomspace(sub_vw["beta_over_H"].min(), sub_vw["beta_over_H"].max(), 250)

                ax_amp.scatter(sub_vw["beta_over_H"], sub_vw["xi"], s=28, color=colors[vw], alpha=0.9)
                ax_amp.plot(beta_grid, amplitude_curve(beta_grid, theta, H_value, vw, f0, amp_payload), color=colors[vw], lw=2.0)

                ax_warp.scatter(sub_vw["beta_over_H"], sub_vw["xi"], s=28, color=colors[vw], alpha=0.9)
                ax_warp.plot(beta_grid, timewarp_curve(beta_grid, theta, H_value, vw, f0, warp_payload, 1.5), color=colors[vw], lw=2.0)

                summary_rows.append({"theta": float(theta), "H": float(H_value), "v_w": float(vw)})

            for ax in (ax_amp, ax_warp):
                ax.set_xscale("log")
                ax.grid(alpha=0.25)
                ax.set_xlabel(r"$\beta/H_*$")
                ax.set_ylabel(r"$\xi$")

            ax_amp.set_title(rf"Amp-only: $H_*={H_value:.1f}$, $\theta={theta:.3f}$")
            ax_warp.set_title(rf"Time-warp: $H_*={H_value:.1f}$, $\theta={theta:.3f}$")

    legend_handles = [
        plt.Line2D([0], [0], color=colors[vw], marker="o", linestyle="-", label=rf"$v_w={vw:.1f}$")
        for vw in vw_values
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=4, frameon=False)
    fig.suptitle(
        "Hilltop comparison: amplitude-only vs time-warp-only\n"
        + rf"Amplitude rel-RMSE={amp_payload['rel_rmse']:.4f}, Time-warp rel-RMSE={warp_payload['rel_rmse']:.4f}",
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    outfile = OUTDIR / "hilltop_amplitude_vs_timewarp.png"
    fig.savefig(outfile, dpi=220)
    plt.close(fig)

    payload = {
        "status": "ok",
        "file": str(outfile),
        "amp_rel_rmse": float(amp_payload["rel_rmse"]),
        "warp_rel_rmse": float(warp_payload["rel_rmse"]),
        "theta_values": [float(theta) for theta in theta_plot],
        "H_values": H_VALUES,
    }
    save_json(OUTDIR / "hilltop_amplitude_vs_timewarp.json", payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
        save_json(OUTDIR / "_error.json", payload)
        print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True))
        raise
