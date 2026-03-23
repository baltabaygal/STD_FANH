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


ROOT = Path(__file__).resolve().parent
FITDIR = ROOT / "results_vw_fullshift_allvars"
OUTDIR = ROOT / "results_vw_fullshift_allvars_H1p0"


def save_json(path: Path, payload):
    path.write_text(json.dumps(tw.to_native(payload), indent=2, sort_keys=True))


def make_args():
    return SimpleNamespace(
        rho="",
        vw_folders=["v3", "v5", "v7", "v9"],
        h_values=[1.0],
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
    p = FITDIR / "final_summary.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing fitted model summary: {p}")
    return json.loads(p.read_text())["fit"]


def prepare_data():
    args = make_args()
    df, _, theta_values, _ = tw.prepare_dataframe(args, OUTDIR)
    return df, theta_values, args


def scale_for_triplet(vw, h_value, beta_over_h, payload):
    refs = payload["refs"]
    log_scale = (
        payload["log_s0"]
        + payload["a_vw"] * math.log(max(float(vw) / refs["vw_ref"], 1.0e-18))
        + payload["b_H"] * math.log(max(float(h_value) / refs["H_ref"], 1.0e-18))
        + payload["c_beta_over_H"] * math.log(max(float(beta_over_h) / refs["beta_ref"], 1.0e-18))
    )
    return math.exp(log_scale)


def predict_df(df: pd.DataFrame, payload: dict, t_osc: float) -> pd.DataFrame:
    out = df.copy()
    out["s_eff"] = [
        scale_for_triplet(vw, h, bh, payload)
        for vw, h, bh in zip(
            out["v_w"].to_numpy(dtype=np.float64),
            out["H"].to_numpy(dtype=np.float64),
            out["beta_over_H"].to_numpy(dtype=np.float64),
        )
    ]
    s = out["s_eff"].to_numpy(dtype=np.float64)
    tp = out["tp"].to_numpy(dtype=np.float64)
    f0 = out["F0"].to_numpy(dtype=np.float64)
    finf = np.asarray(
        [payload["F_inf"][f"{theta:.10f}"] for theta in out["theta"].to_numpy(dtype=np.float64)],
        dtype=np.float64,
    )
    plateau = np.power(np.maximum(s, 1.0e-18), 1.5) * np.power(tp / float(t_osc), 1.5) * finf / np.maximum(f0 * f0, 1.0e-18)
    transient = 1.0 / (
        1.0 + np.power(np.maximum(s * tp, 1.0e-18) / max(float(payload["t_c"]), 1.0e-18), float(payload["r"]))
    )
    out["xi_model"] = plateau + transient
    out["frac_resid"] = (out["xi_model"] - out["xi"]) / np.maximum(out["xi"], 1.0e-12)
    out["data_over_model"] = out["xi"] / np.maximum(out["xi_model"], 1.0e-12)
    return out


def rel_rmse(y, y_fit):
    y = np.asarray(y, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square((y_fit - y) / np.maximum(y, 1.0e-12)))))


def xi_curve(beta_grid, theta, h_value, vw, f0, payload, t_osc):
    perc = tw.base_fit.PercolationCache()
    tp_grid = np.asarray([perc.get(float(h_value), float(beta), float(vw)) for beta in beta_grid], dtype=np.float64)
    tp_grid = np.maximum(tp_grid, 1.0e-18)
    scale = np.asarray([scale_for_triplet(vw, h_value, beta, payload) for beta in beta_grid], dtype=np.float64)
    tp_scaled = tp_grid * scale
    finf = float(payload["F_inf"][f"{theta:.10f}"])
    plateau = np.power(np.maximum(scale, 1.0e-18), 1.5) * np.power(tp_grid / float(t_osc), 1.5) * finf / max(float(f0) ** 2, 1.0e-18)
    transient = 1.0 / (1.0 + np.power(tp_scaled / max(float(payload["t_c"]), 1.0e-18), float(payload["r"])))
    return plateau + transient


def plot_grid(df, theta_values, payload, t_osc, outpath):
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex=True, sharey=False)
    axes = axes.flatten()
    vw_values = np.sort(df["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}

    for ax, theta in zip(axes, np.sort(theta_values)):
        sub = df[np.isclose(df["theta"], float(theta))].copy()
        if sub.empty:
            ax.axis("off")
            continue
        f0 = float(sub["F0"].iloc[0])
        for vw in vw_values:
            sub_vw = sub[np.isclose(sub["v_w"], float(vw))].sort_values("beta_over_H").copy()
            if sub_vw.empty:
                continue
            ax.scatter(sub_vw["beta_over_H"], sub_vw["xi"], s=28, color=colors[vw], alpha=0.9)
            beta_grid = np.geomspace(sub_vw["beta_over_H"].min(), sub_vw["beta_over_H"].max(), 250)
            ax.plot(beta_grid, xi_curve(beta_grid, float(theta), 1.0, float(vw), f0, payload, t_osc), color=colors[vw], lw=2.0)
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$\beta/H_*$")
        ax.set_ylabel(r"$\xi$")

    fig.suptitle(r"Validation on $H_*=1.0$ with direct $t_{\rm eff}$ fit trained on $H_*=1.5,2.0$", y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_separate(df, theta_values, payload, t_osc, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    vw_values = np.sort(df["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    rows = []
    for theta in np.sort(theta_values):
        sub = df[np.isclose(df["theta"], float(theta))].copy()
        if sub.empty:
            continue
        f0 = float(sub["F0"].iloc[0])
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for vw in vw_values:
            sub_vw = sub[np.isclose(sub["v_w"], float(vw))].sort_values("beta_over_H").copy()
            if sub_vw.empty:
                continue
            ax.scatter(sub_vw["beta_over_H"], sub_vw["xi"], s=30, color=colors[vw], alpha=0.9, label=rf"data $v_w={vw:.1f}$")
            beta_grid = np.geomspace(sub_vw["beta_over_H"].min(), sub_vw["beta_over_H"].max(), 250)
            ax.plot(beta_grid, xi_curve(beta_grid, float(theta), 1.0, float(vw), f0, payload, t_osc), color=colors[vw], lw=2.0, label=rf"fit $v_w={vw:.1f}$")
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"$\beta/H_*$")
        ax.set_ylabel(r"$\xi$")
        ax.set_title(rf"Validation $H_*=1.0$, $\theta={theta:.3f}$")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        theta_tag = f"theta_{theta:.10f}".replace(".", "p")
        filepath = outdir / f"xi_vs_betaH_H1p0_{theta_tag}.png"
        fig.savefig(filepath, dpi=220)
        plt.close(fig)
        rows.append({"theta": float(theta), "file": str(filepath)})
    return pd.DataFrame(rows)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    payload = load_fit()
    df, theta_values, args = prepare_data()
    pred = predict_df(df, payload, args.t_osc)
    pred.to_csv(OUTDIR / "predictions.csv", index=False)
    plot_grid(pred, theta_values, payload, args.t_osc, OUTDIR / "xi_vs_betaH_H1p0.png")
    index_df = plot_separate(pred, theta_values, payload, args.t_osc, OUTDIR / "xi_vs_betaH_H1p0_separate")
    index_df.to_csv(OUTDIR / "xi_vs_betaH_H1p0_index.csv", index=False)

    theta_stats = (
        pred.groupby("theta", as_index=False)
        .agg(rel_rmse=("frac_resid", lambda x: float(np.sqrt(np.mean(np.square(x))))), mean_ratio=("data_over_model", "mean"))
        .sort_values("theta")
    )
    theta_stats.to_csv(OUTDIR / "theta_stats.csv", index=False)

    summary = {
        "status": "ok",
        "validation_H": 1.0,
        "global_rel_rmse": rel_rmse(pred["xi"], pred["xi_model"]),
        "per_vw_rel_rmse": {
            f"{vw:.1f}": rel_rmse(
                pred.loc[np.isclose(pred["v_w"], float(vw), atol=1.0e-8), "xi"],
                pred.loc[np.isclose(pred["v_w"], float(vw), atol=1.0e-8), "xi_model"],
            )
            for vw in sorted(pred["v_w"].unique())
        },
        "per_theta_rel_rmse": {
            f"{row.theta:.10f}": float(row.rel_rmse)
            for row in theta_stats.itertuples(index=False)
        },
        "plot_file": str(OUTDIR / "xi_vs_betaH_H1p0.png"),
        "separate_dir": str(OUTDIR / "xi_vs_betaH_H1p0_separate"),
        "index_csv": str(OUTDIR / "xi_vs_betaH_H1p0_index.csv"),
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
