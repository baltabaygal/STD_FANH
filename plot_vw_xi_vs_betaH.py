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
from scipy.optimize import least_squares

import fit_vw_amplitude as base_fit


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "results_vw_amp"


def save_json(path: Path, payload):
    path.write_text(json.dumps(base_fit.to_native(payload), indent=2, sort_keys=True))


def make_args():
    return SimpleNamespace(
        rho="",
        vw_folders=["v3", "v5", "v7", "v9"],
        h_values=[1.5, 2.0],
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


def load_existing_fit(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing fit JSON: {path}")
    return json.loads(path.read_text())


def fit_constrained_a1eq1(df, theta_values, tc_fixed=1.5):
    meta = {
        "xi": df["xi"].to_numpy(dtype=np.float64),
        "tp": df["tp"].to_numpy(dtype=np.float64),
        "F0_sq": np.square(df["F0"].to_numpy(dtype=np.float64)),
        "theta_idx": df["theta_idx"].to_numpy(dtype=np.int64),
        "v_w": df["v_w"].to_numpy(dtype=np.float64),
        "n_theta": len(theta_values),
    }

    finf0 = base_fit.estimate_finf_init(df, theta_values)
    option_b = load_existing_fit(OUTDIR / "global_fit_optionB.json")
    x0 = np.concatenate(
        [[float(option_b["r"]), max(float(option_b.get("A0", 0.0)), 0.0), float(option_b["alpha"])], finf0]
    )
    lower = np.concatenate([[0.1, 0.0, -3.0], np.full(len(theta_values), 1.0e-8)])
    upper = np.concatenate([[50.0, 10.0, 3.0], np.full(len(theta_values), 1.0e3)])

    def unpack(params):
        idx = 0
        r = float(params[idx])
        idx += 1
        a0 = float(params[idx])
        idx += 1
        alpha = float(params[idx])
        idx += 1
        finf = np.asarray(params[idx : idx + len(theta_values)], dtype=np.float64)
        return r, a0, alpha, finf

    def model(params):
        r, a0, alpha, finf = unpack(params)
        amp = a0 + np.power(meta["v_w"], alpha)
        base = np.power(meta["tp"], 1.5) * finf[meta["theta_idx"]] / meta["F0_sq"]
        transient = 1.0 / (1.0 + np.power(meta["tp"] / max(tc_fixed, 1.0e-12), r))
        return amp * (base + transient)

    def resid(params):
        return (model(params) - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)

    huber = least_squares(resid, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=20000)
    final = least_squares(resid, huber.x, bounds=(lower, upper), loss="linear", max_nfev=20000)
    y_model = model(final.x)
    rel = base_fit.rel_rmse(meta["xi"], y_model)
    aic, bic = base_fit.aic_bic((y_model - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12), len(final.x))
    r, a0, alpha, finf = unpack(final.x)
    payload = {
        "model": "A1eq1",
        "A0": a0,
        "A1_fixed": 1.0,
        "alpha": alpha,
        "r": r,
        "t_c": float(tc_fixed),
        "rel_rmse": rel,
        "AIC": float(aic),
        "BIC": float(bic),
        "F_inf": {f"{theta:.10f}": float(val) for theta, val in zip(theta_values, finf)},
        "theta_values": [float(theta) for theta in theta_values],
        "success": bool(final.success),
        "message": str(final.message),
    }
    save_json(OUTDIR / "fixed_A1eq1_fit.json", payload)
    return payload


def xi_curve(beta_grid, theta, h_value, vw, f0, fit_payload, mode):
    perc = base_fit.PercolationCache()
    tp_grid = np.asarray([perc.get(float(h_value), float(beta), float(vw)) for beta in beta_grid], dtype=np.float64)
    tp_grid = np.maximum(tp_grid, 1.0e-18)

    if mode == "optionB":
        amp = fit_payload["A0"] + fit_payload["A1"] * np.power(vw, fit_payload["alpha"])
    elif mode == "A1eq1":
        amp = fit_payload["A0"] + np.power(vw, fit_payload["alpha"])
    else:
        raise ValueError(f"Unsupported mode {mode}")

    finf = fit_payload["F_inf"][f"{theta:.10f}"]
    base = np.power(tp_grid, 1.5) * float(finf) / max(float(f0) ** 2, 1.0e-18)
    transient = 1.0 / (1.0 + np.power(tp_grid / max(float(fit_payload["t_c"]), 1.0e-12), float(fit_payload["r"])))
    return amp * (base + transient)


def representative_thetas(theta_values):
    theta_values = np.asarray(theta_values, dtype=np.float64)
    if len(theta_values) <= 6:
        return theta_values
    idx = np.linspace(0, len(theta_values) - 1, 6).round().astype(int)
    return theta_values[np.unique(idx)]


def plot_model(df, theta_values, fit_payload, mode, outpath: Path):
    reps = representative_thetas(theta_values)
    ncols = 3
    nrows = int(math.ceil(len(reps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    cmap = plt.get_cmap("viridis")
    vw_values = np.sort(df["v_w"].unique())
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    marker_map = {1.5: "o", 2.0: "s"}

    for ax, theta in zip(axes, reps):
        sub_theta = df[df["theta"] == float(theta)].copy()
        f0 = float(sub_theta["F0"].iloc[0])
        for vw in vw_values:
            sub_vw = sub_theta[sub_theta["v_w"] == float(vw)].copy()
            if sub_vw.empty:
                continue
            for h_value, sub_h in sub_vw.groupby("H"):
                sub_h = sub_h.sort_values("beta_over_H")
                ax.scatter(
                    sub_h["beta_over_H"],
                    sub_h["xi"],
                    s=24,
                    color=colors[vw],
                    marker=marker_map.get(float(h_value), "o"),
                    alpha=0.85,
                )
                beta_grid = np.geomspace(sub_h["beta_over_H"].min(), sub_h["beta_over_H"].max(), 250)
                xi_fit = xi_curve(beta_grid, float(theta), float(h_value), float(vw), f0, fit_payload, mode)
                ax.plot(beta_grid, xi_fit, color=colors[vw], lw=1.8, alpha=0.95)

        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$\beta/H_*$")
        ax.set_ylabel(r"$\xi$")

    for ax in axes[len(reps) :]:
        ax.axis("off")

    handles = []
    labels = []
    for vw in vw_values:
        handles.append(plt.Line2D([0], [0], color=colors[vw], marker="o", linestyle="-"))
        labels.append(rf"$v_w={vw:.1f}$")
    handles.extend(
        [
            plt.Line2D([0], [0], color="black", marker="o", linestyle="None"),
            plt.Line2D([0], [0], color="black", marker="s", linestyle="None"),
        ]
    )
    labels.extend([r"$H_*=1.5$", r"$H_*=2.0$"])
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False)
    fig.suptitle(
        rf"$\xi(\beta/H_*)$ data vs {mode} fit, "
        + rf"$r={fit_payload['r']:.3f}$, $t_c={fit_payload['t_c']:.3f}$",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_model_separate(df, theta_values, fit_payload, mode, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis")
    vw_values = np.sort(df["v_w"].unique())
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    marker_map = {1.5: "o", 2.0: "s"}
    rows = []

    for h_value in np.sort(df["H"].unique()):
        sub_h = df[df["H"] == float(h_value)].copy()
        for theta in np.sort(theta_values):
            sub = sub_h[sub_h["theta"] == float(theta)].copy()
            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            f0 = float(sub["F0"].iloc[0])
            for vw in vw_values:
                sub_vw = sub[sub["v_w"] == float(vw)].sort_values("beta_over_H").copy()
                if sub_vw.empty:
                    continue
                ax.scatter(
                    sub_vw["beta_over_H"],
                    sub_vw["xi"],
                    s=26,
                    color=colors[vw],
                    marker=marker_map.get(float(h_value), "o"),
                    alpha=0.9,
                    label=rf"data $v_w={vw:.1f}$",
                )
                beta_grid = np.geomspace(sub_vw["beta_over_H"].min(), sub_vw["beta_over_H"].max(), 250)
                xi_fit = xi_curve(beta_grid, float(theta), float(h_value), float(vw), f0, fit_payload, mode)
                ax.plot(beta_grid, xi_fit, color=colors[vw], lw=2.0, alpha=0.95, label=rf"fit $v_w={vw:.1f}$")

            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_xlabel(r"$\beta/H_*$")
            ax.set_ylabel(r"$\xi$")
            ax.set_title(rf"$H_*={h_value:.1f}$, $\theta={theta:.3f}$, {mode}")
            ax.legend(frameon=False, fontsize=7, ncol=2)
            fig.tight_layout()

            h_tag = f"H{h_value:.1f}".replace(".", "p")
            theta_tag = f"theta_{theta:.10f}".replace(".", "p")
            filename = f"xi_vs_betaH_{mode}_{h_tag}_{theta_tag}.png"
            filepath = outdir / filename
            fig.savefig(filepath, dpi=220)
            plt.close(fig)
            rows.append(
                {
                    "mode": mode,
                    "H": float(h_value),
                    "theta": float(theta),
                    "file": str(filepath),
                }
            )

    return pd.DataFrame(rows)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    args = make_args()
    df, _, theta_values = base_fit.prepare_dataframe(args, OUTDIR)
    option_b = load_existing_fit(OUTDIR / "global_fit_optionB.json")
    constrained = fit_constrained_a1eq1(df, theta_values, tc_fixed=1.5)
    separate_optionb_dir = OUTDIR / "xi_vs_betaH_optionB_separate"
    separate_a1eq1_dir = OUTDIR / "xi_vs_betaH_A1eq1_separate"

    plot_model(df, theta_values, option_b, "optionB", OUTDIR / "xi_vs_betaH_by_theta_vw_optionB.png")
    plot_model(df, theta_values, constrained, "A1eq1", OUTDIR / "xi_vs_betaH_by_theta_vw_A1eq1.png")
    idx_b = plot_model_separate(df, theta_values, option_b, "optionB", separate_optionb_dir)
    idx_a = plot_model_separate(df, theta_values, constrained, "A1eq1", separate_a1eq1_dir)
    index_df = pd.concat([idx_b, idx_a], ignore_index=True)
    index_df.to_csv(OUTDIR / "xi_vs_betaH_separate_index.csv", index=False)

    summary = {
        "status": "ok",
        "files": {
            "optionB_plot": str(OUTDIR / "xi_vs_betaH_by_theta_vw_optionB.png"),
            "A1eq1_plot": str(OUTDIR / "xi_vs_betaH_by_theta_vw_A1eq1.png"),
            "A1eq1_fit": str(OUTDIR / "fixed_A1eq1_fit.json"),
            "optionB_separate_dir": str(separate_optionb_dir),
            "A1eq1_separate_dir": str(separate_a1eq1_dir),
            "separate_index": str(OUTDIR / "xi_vs_betaH_separate_index.csv"),
        },
        "optionB": {
            "rel_rmse": option_b["rel_rmse"],
            "A0": option_b["A0"],
            "A1": option_b["A1"],
            "alpha": option_b["alpha"],
            "r": option_b["r"],
            "t_c": option_b["t_c"],
        },
        "A1eq1": {
            "rel_rmse": constrained["rel_rmse"],
            "A0": constrained["A0"],
            "A1_fixed": 1.0,
            "alpha": constrained["alpha"],
            "r": constrained["r"],
            "t_c": constrained["t_c"],
        },
    }
    save_json(OUTDIR / "xi_vs_betaH_plot_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
        save_json(OUTDIR / "xi_vs_betaH_plot_error.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise
