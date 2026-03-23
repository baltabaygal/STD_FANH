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
DEFAULT_PRED = ROOT / "results_vw0p9_model_applied_all_vw" / "predictions.csv"
DEFAULT_OUTDIR = ROOT / "results_vw0p9_model_with_gvw"
VW_REF = 0.9


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot frozen v_w=0.9 baseline model multiplied by a fitted scalar g(v_w)."
    )
    p.add_argument("--predictions", type=str, default=str(DEFAULT_PRED))
    p.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    p.add_argument("--model", choices=["power", "log"], default="power")
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


def fit_g(df: pd.DataFrame, model_kind: str):
    vw = df["v_w"].to_numpy(dtype=np.float64)
    ratio = (df["xi"] / np.maximum(df["xi_fit_vw0p9_model"], 1.0e-18)).to_numpy(dtype=np.float64)

    if model_kind == "power":
        res = least_squares(lambda p: model_power(vw, p[0]) - ratio, x0=np.array([-0.1]))
        param = float(res.x[0])
        pred = model_power(vw, param)
        formula = "g(v_w) = (v_w / 0.9)^alpha"
        payload = {"alpha": param}
    else:
        res = least_squares(lambda p: model_log(vw, p[0]) - ratio, x0=np.array([-0.1]))
        param = float(res.x[0])
        pred = model_log(vw, param)
        formula = "g(v_w) = 1 + a * log(v_w / 0.9)"
        payload = {"a": param}
    return pred, formula, payload


def rel_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.mean(np.square(y_true)), 1.0e-18)
    return float(np.sqrt(np.mean(np.square(y_pred - y_true)) / denom))


def plot_main(df: pd.DataFrame, outdir: Path, dpi: int):
    theta_values = np.sort(df["theta"].unique())
    h_values = np.sort(df["H"].unique())
    vw_values = np.sort(df["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {float(vw): cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    summary_rows = []

    for h in h_values:
        sub_h = df[np.isclose(df["H"], float(h), atol=1.0e-12, rtol=0.0)].copy()
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
        axes = axes.ravel()
        for ax, theta in zip(axes, theta_values):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
            for vw, cur in sub.groupby("v_w", sort=True):
                cur = cur.sort_values("beta_over_H")
                ax.scatter(
                    cur["beta_over_H"],
                    cur["xi"],
                    color=colors[float(vw)],
                    s=18,
                    alpha=0.9,
                )
                ax.plot(
                    cur["beta_over_H"],
                    cur["xi_fit_gvw"],
                    color=colors[float(vw)],
                    lw=1.8,
                    alpha=0.95,
                )
                summary_rows.append(
                    {
                        "H": float(h),
                        "theta": float(theta),
                        "v_w": float(vw),
                        "rel_rmse": rel_rmse(
                            cur["xi"].to_numpy(dtype=np.float64),
                            cur["xi_fit_gvw"].to_numpy(dtype=np.float64),
                        ),
                    }
                )
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$\beta/H_*$")
            ax.set_ylabel(r"$\xi$")
        for ax in axes[len(theta_values):]:
            ax.axis("off")
        handles = [plt.Line2D([0], [0], color=colors[float(vw)], lw=2.0) for vw in vw_values]
        labels = [rf"$v_w={float(vw):.1f}$" for vw in vw_values]
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        tag = str(float(h)).replace(".", "p")
        fig.savefig(outdir / f"xi_vs_betaH_gvw_H{tag}.png", dpi=dpi)
        plt.close(fig)
    return summary_rows


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(args.predictions).resolve())
    gpred, formula, params = fit_g(df, args.model)
    df["g_vw"] = gpred
    df["xi_fit_gvw"] = df["xi_fit_vw0p9_model"] * df["g_vw"]
    df.to_csv(outdir / "predictions_with_gvw.csv", index=False)

    rows = plot_main(df, outdir, args.dpi)
    y = df["xi"].to_numpy(dtype=np.float64)
    y0 = df["xi_fit_vw0p9_model"].to_numpy(dtype=np.float64)
    y1 = df["xi_fit_gvw"].to_numpy(dtype=np.float64)

    per_vw = []
    for vw, cur in df.groupby("v_w", sort=True):
        per_vw.append(
            {
                "v_w": float(vw),
                "baseline_rel_rmse": rel_rmse(cur["xi"].to_numpy(dtype=np.float64), cur["xi_fit_vw0p9_model"].to_numpy(dtype=np.float64)),
                "gvw_rel_rmse": rel_rmse(cur["xi"].to_numpy(dtype=np.float64), cur["xi_fit_gvw"].to_numpy(dtype=np.float64)),
                "g_vw_mean": float(np.mean(cur["g_vw"].to_numpy(dtype=np.float64))),
            }
        )

    summary = {
        "status": "ok",
        "source_predictions": str(Path(args.predictions).resolve()),
        "g_model": {
            "kind": args.model,
            "formula": formula,
            **params,
        },
        "global_baseline_rel_rmse": rel_rmse(y, y0),
        "global_gvw_rel_rmse": rel_rmse(y, y1),
        "per_vw": per_vw,
        "outputs": {
            **{
                f"H{str(float(h)).replace('.', 'p')}": str(
                    outdir / f"xi_vs_betaH_gvw_H{str(float(h)).replace('.', 'p')}.png"
                )
                for h in np.sort(df["H"].unique())
            },
            "predictions": str(outdir / "predictions_with_gvw.csv"),
        },
        "summary_rows": rows,
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
