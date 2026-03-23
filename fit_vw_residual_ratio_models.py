#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "results_vw_factorization_tests" / "diagnostic_table.csv"
DEFAULT_OUTDIR = ROOT / "results_vw_residual_ratio_models"
VW_REF = 0.9


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit simple v_w laws to the residual ratio R = (xi/tp^(3/2)) / ref(vw=0.9)."
    )
    p.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
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


def model_power(vw: np.ndarray, alpha: float) -> np.ndarray:
    return np.power(vw / VW_REF, alpha)


def model_log(vw: np.ndarray, a: float) -> np.ndarray:
    return 1.0 + a * np.log(vw / VW_REF)


def fit_one_param(vw: np.ndarray, y: np.ndarray, kind: str):
    if kind == "power":
        fun = lambda p: model_power(vw, p[0]) - y
        x0 = np.array([0.1], dtype=np.float64)
    elif kind == "log":
        fun = lambda p: model_log(vw, p[0]) - y
        x0 = np.array([0.1], dtype=np.float64)
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    res = least_squares(fun, x0=x0, method="trf")
    return res


def fit_stats(y_true: np.ndarray, y_pred: np.ndarray, n_params: int):
    resid = y_pred - y_true
    rss = float(np.sum(resid**2))
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    n = int(y_true.size)
    if rss <= 0.0:
        aic = float("-inf")
        bic = float("-inf")
    else:
        aic = float(n * np.log(rss / n) + 2 * n_params)
        bic = float(n * np.log(rss / n) + n_params * np.log(n))
    return {"rss": rss, "rmse": rmse, "mae": mae, "aic": aic, "bic": bic, "n_points": n}


def grouped_stats(df: pd.DataFrame, pred_col: str):
    rows = []
    for (h, theta), g in df.groupby(["H", "theta"], sort=True):
        y = g["R"].to_numpy(dtype=np.float64)
        yp = g[pred_col].to_numpy(dtype=np.float64)
        st = fit_stats(y, yp, 1)
        rows.append({"H": float(h), "theta": float(theta), **st})
    return pd.DataFrame(rows)


def plot_main(df: pd.DataFrame, outdir: Path, dpi: int):
    theta_values = np.sort(df["theta"].unique())
    h_values = np.sort(df["H"].unique())
    beta_vals = np.sort(df["beta_over_H"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {float(b): cmap(i / max(len(beta_vals) - 1, 1)) for i, b in enumerate(beta_vals)}
    vw_grid = np.linspace(0.3, 0.9, 400)

    for h in h_values:
        sub_h = df[np.isclose(df["H"], float(h), atol=1.0e-12, rtol=0.0)].copy()
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
        axes = axes.ravel()
        for ax, theta in zip(axes, theta_values):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
            for beta_val, cur in sub.groupby("beta_over_H", sort=True):
                cur = cur.sort_values("v_w")
                ax.plot(
                    cur["v_w"],
                    cur["R"],
                    color=colors[float(beta_val)],
                    marker="o",
                    lw=1.2,
                    ms=3.0,
                    alpha=0.75,
                )
            ax.plot(vw_grid, model_power(vw_grid, float(df["alpha_power"].iloc[0])), color="black", lw=1.8, label="power")
            ax.plot(vw_grid, model_log(vw_grid, float(df["a_log"].iloc[0])), color="crimson", lw=1.6, ls="--", label="log")
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$v_w$")
            ax.set_ylabel(r"$R = [\xi/t_p^{3/2}] / [\xi/t_p^{3/2}]_{v_w=0.9}$")
        for ax in axes[len(theta_values):]:
            ax.axis("off")
        handles = [
            plt.Line2D([0], [0], color="black", lw=1.8, label="power"),
            plt.Line2D([0], [0], color="crimson", lw=1.6, ls="--", label="log"),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
        fig.suptitle(rf"Residual ratio models vs $v_w$, $H_*={float(h):.1f}$", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        tag = str(float(h)).replace(".", "p")
        fig.savefig(outdir / f"residual_ratio_models_H{tag}.png", dpi=dpi)
        plt.close(fig)


def plot_aggregate(df: pd.DataFrame, outdir: Path, dpi: int):
    agg = (
        df.groupby("v_w", as_index=False)["R"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "R_mean", "std": "R_std"})
    )
    vw_grid = np.linspace(0.3, 0.9, 400)
    alpha = float(df["alpha_power"].iloc[0])
    a_log = float(df["a_log"].iloc[0])
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.errorbar(
        agg["v_w"],
        agg["R_mean"],
        yerr=agg["R_std"],
        color="tab:blue",
        marker="o",
        lw=1.4,
        ms=5.0,
        capsize=3.0,
        label="mean ± std",
    )
    ax.plot(vw_grid, model_power(vw_grid, alpha), color="black", lw=2.0, label=rf"power: $(v_w/0.9)^{{{alpha:.3f}}}$")
    ax.plot(vw_grid, model_log(vw_grid, a_log), color="crimson", lw=1.8, ls="--", label=rf"log: $1 + {a_log:.3f}\ln(v_w/0.9)$")
    ax.grid(alpha=0.25)
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$R = [\xi/t_p^{3/2}] / [\xi/t_p^{3/2}]_{v_w=0.9}$")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "aggregate_ratio_fit.png", dpi=dpi)
    plt.close(fig)
    return agg


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(args.input).resolve())
    df = df.copy()
    df["R"] = df["xi_over_tp32_ratio_to_vw09"].to_numpy(dtype=np.float64)
    vw = df["v_w"].to_numpy(dtype=np.float64)
    y = df["R"].to_numpy(dtype=np.float64)

    res_power = fit_one_param(vw, y, "power")
    alpha = float(res_power.x[0])
    y_power = model_power(vw, alpha)
    stats_power = fit_stats(y, y_power, 1)

    res_log = fit_one_param(vw, y, "log")
    a_log = float(res_log.x[0])
    y_log = model_log(vw, a_log)
    stats_log = fit_stats(y, y_log, 1)

    df["pred_power"] = y_power
    df["pred_log"] = y_log
    df["alpha_power"] = alpha
    df["a_log"] = a_log
    df.to_csv(outdir / "residual_ratio_with_predictions.csv", index=False)

    grouped_power = grouped_stats(df, "pred_power")
    grouped_log = grouped_stats(df, "pred_log")
    grouped_power.to_csv(outdir / "grouped_stats_power.csv", index=False)
    grouped_log.to_csv(outdir / "grouped_stats_log.csv", index=False)

    plot_main(df, outdir, args.dpi)
    agg = plot_aggregate(df, outdir, args.dpi)
    agg.to_csv(outdir / "aggregate_by_vw.csv", index=False)

    summary = {
        "status": "ok",
        "input": str(Path(args.input).resolve()),
        "vw_reference": VW_REF,
        "power_model": {
            "formula": "R(v_w) = (v_w / 0.9)^alpha",
            "alpha": alpha,
            **stats_power,
        },
        "log_model": {
            "formula": "R(v_w) = 1 + a * log(v_w / 0.9)",
            "a": a_log,
            **stats_log,
        },
        "preferred_by_rmse": "power" if stats_power["rmse"] < stats_log["rmse"] else "log",
        "preferred_by_aic": "power" if stats_power["aic"] < stats_log["aic"] else "log",
        "outputs": {
            "aggregate_plot": str(outdir / "aggregate_ratio_fit.png"),
            "aggregate_table": str(outdir / "aggregate_by_vw.csv"),
            "pred_table": str(outdir / "residual_ratio_with_predictions.csv"),
            "grouped_power": str(outdir / "grouped_stats_power.csv"),
            "grouped_log": str(outdir / "grouped_stats_log.csv"),
            **{
                f"H{str(float(h)).replace('.', 'p')}": str(
                    outdir / f"residual_ratio_models_H{str(float(h)).replace('.', 'p')}.png"
                )
                for h in np.sort(df["H"].unique())
            },
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
