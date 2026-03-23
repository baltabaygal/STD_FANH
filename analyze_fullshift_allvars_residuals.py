#!/usr/bin/env python3
from __future__ import annotations

import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fit_vw_timewarp as tw
import fit_vw_fullshift_allvars as fs


ROOT = Path(__file__).resolve().parent
FITDIR = ROOT / "results_vw_fullshift_allvars"
OUTDIR = ROOT / "results_vw_fullshift_allvars_residuals"


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
    df, _, theta_values, _ = tw.prepare_dataframe(args, FITDIR)
    return df, theta_values, args


def parameter_vector(payload: dict, theta_values) -> np.ndarray:
    parts = [payload["log_s0"], payload["a_vw"], payload["b_H"], payload["c_beta_over_H"], payload["r"]]
    parts.extend([payload["F_inf"][f"{theta:.10f}"] for theta in theta_values])
    return np.asarray(parts, dtype=np.float64)


def add_predictions(df: pd.DataFrame, payload: dict, args, theta_values) -> tuple[pd.DataFrame, np.ndarray]:
    meta = fs.make_meta(df, theta_values, args.t_osc)
    params = parameter_vector(payload, theta_values)
    y_fit = fs.model_eval(params, meta, args.fix_tc, args.tc0)
    out = df.copy()
    out["xi_model"] = y_fit
    out["frac_resid"] = (out["xi_model"] - out["xi"]) / np.maximum(out["xi"], 1.0e-12)
    out["data_over_model"] = out["xi"] / np.maximum(out["xi_model"], 1.0e-12)
    out["abs_frac_resid"] = np.abs(out["frac_resid"])
    return out, y_fit


def plot_theta_summary(theta_stats: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.errorbar(
        theta_stats["theta"],
        theta_stats["mean_abs_frac_resid"],
        yerr=theta_stats["std_frac_resid"],
        fmt="o-",
        lw=1.6,
        capsize=3,
    )
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"mean $|(\xi_{\rm fit}-\xi)/\xi|$")
    ax.set_title("Residual level by angle")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, value_col: str, x_col: str, path: Path, title: str, x_is_log=False):
    if x_col == "beta_over_H":
        bins = np.array([4, 6, 8, 10, 12, 16, 20, 30, 40], dtype=np.float64)
        labels = [f"{bins[i]:g}-{bins[i+1]:g}" for i in range(len(bins) - 1)]
        work = df.copy()
        work["xbin"] = pd.cut(work[x_col], bins=bins, labels=labels, include_lowest=True)
        pivot = work.pivot_table(index="theta", columns="xbin", values=value_col, aggfunc="mean").sort_index()
    else:
        pivot = df.pivot_table(index="theta", columns=x_col, values=value_col, aggfunc="mean").sort_index()

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    pcm = ax.pcolormesh(
        np.arange(len(pivot.columns) + 1),
        np.arange(len(pivot.index) + 1),
        pivot.to_numpy(dtype=np.float64),
        cmap="coolwarm",
        vmin=-0.08,
        vmax=0.08,
        shading="auto",
    )
    ax.set_xticks(np.arange(len(pivot.columns)) + 0.5)
    ax.set_xticklabels([str(v) for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)) + 0.5)
    ax.set_yticklabels([f"{float(v):.3f}" for v in pivot.index])
    ax.set_xlabel(r"$\beta/H_*$ bins" if x_col == "beta_over_H" else x_col)
    ax.set_ylabel(r"$\theta_0$")
    ax.set_title(title)
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_ratio_panels(df: pd.DataFrame, path: Path):
    theta_values = np.sort(df["theta"].unique())
    vw_values = np.sort(df["v_w"].unique())
    h_values = np.sort(df["H"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    markers = {float(h_values[0]): "o", float(h_values[1]): "s"}

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, theta in zip(axes, theta_values):
        sub = df[np.isclose(df["theta"], float(theta), atol=5.0e-4)].copy()
        for h in h_values:
            for vw in vw_values:
                hsub = sub[np.isclose(sub["H"], float(h), atol=1.0e-8) & np.isclose(sub["v_w"], float(vw), atol=1.0e-8)].sort_values("beta_over_H")
                if hsub.empty:
                    continue
                ax.plot(
                    hsub["beta_over_H"],
                    hsub["data_over_model"],
                    marker=markers[float(h)],
                    ms=3.5,
                    lw=1.0,
                    color=colors[vw],
                    alpha=0.95,
                )
        ax.axhline(1.0, color="black", lw=1.0, alpha=0.7)
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$\beta/H_*$")
        ax.set_ylabel(r"data / fit")

    handles = []
    labels = []
    for vw in vw_values:
        handles.append(plt.Line2D([], [], color=colors[vw], lw=1.8))
        labels.append(rf"$v_w={vw:.1f}$")
    for h in h_values:
        handles.append(plt.Line2D([], [], color="gray", marker=markers[float(h)], lw=0))
        labels.append(rf"$H_*={h:.1f}$")
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle("Full-shift residual ratio by angle", y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_hsplit(df: pd.DataFrame, path: Path):
    rows = []
    for (vw, theta, beta), sub in df.groupby(["v_w", "theta", "beta_over_H"]):
        if sub["H"].nunique() != 2:
            continue
        sub = sub.sort_values("H")
        vals = sub["data_over_model"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "v_w": float(vw),
                "theta": float(theta),
                "beta_over_H": float(beta),
                "hsplit_ratio": float(vals[-1] - vals[0]),
            }
        )
    split = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    if not split.empty:
        pivot = split.pivot_table(index="theta", values="hsplit_ratio", aggfunc="mean").sort_index()
        ax.plot(pivot.index.to_numpy(dtype=np.float64), pivot["hsplit_ratio"].to_numpy(dtype=np.float64), "o-", lw=1.6)
    ax.axhline(0.0, color="black", lw=1.0, alpha=0.7)
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"mean $(\mathrm{data/fit})_{H=2}-(\mathrm{data/fit})_{H=1.5}$")
    ax.set_title(r"Residual $H_*$ split by angle")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return split


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    payload = load_fit()
    df, theta_values, args = prepare_data()
    pred, y_fit = add_predictions(df, payload, args, theta_values)
    pred.to_csv(OUTDIR / "predictions.csv", index=False)

    theta_stats = (
        pred.groupby("theta", as_index=False)
        .agg(
            mean_abs_frac_resid=("abs_frac_resid", "mean"),
            mean_frac_resid=("frac_resid", "mean"),
            std_frac_resid=("frac_resid", "std"),
            mean_ratio=("data_over_model", "mean"),
        )
        .sort_values("theta")
    )
    theta_stats.to_csv(OUTDIR / "theta_stats.csv", index=False)

    hsplit = plot_hsplit(pred, OUTDIR / "hsplit_vs_theta.png")
    hsplit.to_csv(OUTDIR / "hsplit.csv", index=False)

    plot_theta_summary(theta_stats, OUTDIR / "residual_vs_theta.png")
    plot_heatmap(pred, "frac_resid", "beta_over_H", OUTDIR / "residual_heatmap_theta_betaH.png", r"Mean fractional residual vs $\theta_0$ and $\beta/H_*$")
    plot_heatmap(pred, "frac_resid", "v_w", OUTDIR / "residual_heatmap_theta_vw.png", r"Mean fractional residual vs $\theta_0$ and $v_w$")
    plot_heatmap(pred, "frac_resid", "H", OUTDIR / "residual_heatmap_theta_H.png", r"Mean fractional residual vs $\theta_0$ and $H_*$")
    plot_ratio_panels(pred, OUTDIR / "ratio_vs_betaH_by_theta.png")

    worst_theta = float(theta_stats.sort_values("mean_abs_frac_resid", ascending=False).iloc[0]["theta"])
    best_theta = float(theta_stats.sort_values("mean_abs_frac_resid", ascending=True).iloc[0]["theta"])

    summary = {
        "status": "ok",
        "n_points": int(len(pred)),
        "global_rel_rmse": float(np.sqrt(np.mean(np.square(pred["frac_resid"].to_numpy(dtype=np.float64))))),
        "worst_theta": worst_theta,
        "best_theta": best_theta,
        "theta_mean_abs_residuals": {
            f"{row.theta:.10f}": float(row.mean_abs_frac_resid)
            for row in theta_stats.itertuples(index=False)
        },
        "mean_hsplit_by_theta": {
            f"{theta:.10f}": float(val)
            for theta, val in (
                hsplit.groupby("theta")["hsplit_ratio"].mean().sort_index().items() if not hsplit.empty else []
            )
        },
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
