#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results_pointwise_shift_from_vw0p9_baseline" / "pointwise_shift_table.csv"
OUTDIR = ROOT / "results_pointwise_shift_powerlaw"
REF_VW = 0.9


def parse_args():
    p = argparse.ArgumentParser(description="Fit compact power-law models to the inferred pointwise shift s(tp,vw).")
    p.add_argument("--input-table", type=str, default=str(INPUT))
    p.add_argument("--outdir", type=str, default=str(OUTDIR))
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


def rel_rmse(y, yfit):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square((yfit - y) / np.maximum(y, 1.0e-12)))))


def fit_models(df: pd.DataFrame):
    tp = df["tp"].to_numpy(dtype=np.float64)
    vw = df["v_w"].to_numpy(dtype=np.float64)
    s = df["s_pointwise"].to_numpy(dtype=np.float64)

    out = {}

    # s = c * tp^a * vw^b
    def model_plain(par):
        c, a, b = par
        return np.exp(np.log(np.maximum(c, 1.0e-12)) + a * np.log(tp) + b * np.log(vw))

    sol = least_squares(
        lambda par: np.log(model_plain(par)) - np.log(s),
        x0=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        bounds=([1.0e-6, -5.0, -5.0], [10.0, 5.0, 5.0]),
    )
    pred = model_plain(sol.x)
    out["plain_power"] = {"params": sol.x, "rel_rmse": rel_rmse(s, pred)}

    # s = 1 + A * ((0.9/vw)^m - 1) * tp^p
    def model_anchor_diff(par):
        A, m, p = par
        return 1.0 + A * (np.power(REF_VW / vw, m) - 1.0) * np.power(tp, p)

    sol = least_squares(
        lambda par: (model_anchor_diff(par) - s) / np.maximum(s, 1.0e-12),
        x0=np.array([0.6, 0.3, -1.3], dtype=np.float64),
        bounds=([-10.0, -5.0, -5.0], [10.0, 5.0, 5.0]),
    )
    pred = model_anchor_diff(sol.x)
    out["anchored_diff_power"] = {"params": sol.x, "rel_rmse": rel_rmse(s, pred)}

    # s = 1 + A * log(0.9/vw) * tp^p
    def model_anchor_log(par):
        A, p = par
        return 1.0 + A * np.log(REF_VW / vw) * np.power(tp, p)

    sol = least_squares(
        lambda par: (model_anchor_log(par) - s) / np.maximum(s, 1.0e-12),
        x0=np.array([0.2, -1.3], dtype=np.float64),
        bounds=([-10.0, -5.0], [10.0, 5.0]),
    )
    pred = model_anchor_log(sol.x)
    out["anchored_log_power"] = {"params": sol.x, "rel_rmse": rel_rmse(s, pred)}

    return out


def predict(name: str, params, tp, vw):
    tp = np.asarray(tp, dtype=np.float64)
    vw = np.asarray(vw, dtype=np.float64)
    if name == "plain_power":
        c, a, b = params
        return np.exp(np.log(np.maximum(c, 1.0e-12)) + a * np.log(tp) + b * np.log(vw))
    if name == "anchored_diff_power":
        A, m, p = params
        return 1.0 + A * (np.power(REF_VW / vw, m) - 1.0) * np.power(tp, p)
    if name == "anchored_log_power":
        A, p = params
        return 1.0 + A * np.log(REF_VW / vw) * np.power(tp, p)
    raise KeyError(name)


def plot_all(df: pd.DataFrame, best_name: str, best_params, outdir: Path):
    vw_values = np.sort(df["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    for vw in vw_values:
        cur = df[np.isclose(df["v_w"], float(vw), atol=1.0e-12, rtol=0.0)].copy()
        ax.scatter(cur["tp"], cur["s_pointwise"], s=18, color=colors[float(vw)], alpha=0.35)
        xfit = np.geomspace(float(cur["tp"].min()), float(cur["tp"].max()), 300)
        yfit = predict(best_name, best_params, xfit, np.full_like(xfit, float(vw)))
        ax.plot(xfit, yfit, color=colors[float(vw)], lw=2.0, label=rf"$v_w={vw:.1f}$")
    ax.axhline(1.0, color="black", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$t_p$")
    ax.set_ylabel(r"$s$")
    ax.set_title("Pointwise shift and best anchored power-law fit")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "s_vs_tp_powerlaw_fit.png", dpi=220)
    plt.close(fig)


def plot_binned(df: pd.DataFrame, best_name: str, best_params, outdir: Path):
    vw_values = np.sort(df["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    for vw in vw_values:
        cur = df[np.isclose(df["v_w"], float(vw), atol=1.0e-12, rtol=0.0)].copy()
        bins = np.quantile(cur["tp"], np.linspace(0.0, 1.0, 7))
        bins = np.unique(bins)
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (cur["tp"] >= lo) & ((cur["tp"] <= hi) if hi == bins[-1] else (cur["tp"] < hi))
            sub = cur.loc[mask]
            if sub.empty:
                continue
            rows.append({"tp_med": float(sub["tp"].median()), "s_med": float(sub["s_pointwise"].median())})
        bdf = pd.DataFrame(rows)
        ax.plot(bdf["tp_med"], bdf["s_med"], "o", ms=4.5, color=colors[float(vw)])
        xfit = np.geomspace(float(cur["tp"].min()), float(cur["tp"].max()), 300)
        yfit = predict(best_name, best_params, xfit, np.full_like(xfit, float(vw)))
        ax.plot(xfit, yfit, color=colors[float(vw)], lw=2.0, label=rf"$v_w={vw:.1f}$")
    ax.axhline(1.0, color="black", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$t_p$")
    ax.set_ylabel(r"median pointwise $s$")
    ax.set_title("Binned pointwise shift with anchored power-law fit")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "s_vs_tp_powerlaw_fit_binned.png", dpi=220)
    plt.close(fig)


def plot_residuals(df: pd.DataFrame, best_name: str, best_params, outdir: Path):
    work = df.copy()
    work["s_fit"] = predict(best_name, best_params, work["tp"].to_numpy(), work["v_w"].to_numpy())
    work["frac_resid"] = (work["s_fit"] - work["s_pointwise"]) / np.maximum(work["s_pointwise"], 1.0e-12)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].scatter(work["tp"], work["frac_resid"], c=work["v_w"], s=14, cmap="viridis", alpha=0.4)
    axes[0].axhline(0.0, color="black", lw=1.0, ls="--")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$t_p$")
    axes[0].set_ylabel("fractional residual")
    axes[0].set_title("Residual vs $t_p$")
    axes[0].grid(alpha=0.25)
    axes[1].scatter(work["theta"], work["frac_resid"], c=work["v_w"], s=14, cmap="viridis", alpha=0.4)
    axes[1].axhline(0.0, color="black", lw=1.0, ls="--")
    axes[1].set_xlabel(r"$\theta_0$")
    axes[1].set_ylabel("fractional residual")
    axes[1].set_title("Residual vs $\\theta_0$")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "s_powerlaw_residuals.png", dpi=220)
    plt.close(fig)

    return work


def main():
    args = parse_args()
    input_path = Path(args.input_table).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    df = df[df["in_range"]].copy()

    fits = fit_models(df)
    best_name = min(fits, key=lambda k: fits[k]["rel_rmse"])
    best_params = np.asarray(fits[best_name]["params"], dtype=np.float64)

    plot_all(df, best_name, best_params, outdir)
    plot_binned(df, best_name, best_params, outdir)
    work = plot_residuals(df, best_name, best_params, outdir)
    work.to_csv(outdir / "pointwise_shift_with_fit.csv", index=False)

    summary = {
        "status": "ok",
        "input_table": str(input_path),
        "best_model": best_name,
        "fit_models": {name: {"params": list(map(float, rec["params"])), "rel_rmse": float(rec["rel_rmse"])} for name, rec in fits.items()},
        "best_formula": (
            r"s = 1 + A[(0.9/v_w)^m - 1] t_p^p"
            if best_name == "anchored_diff_power"
            else (r"s = 1 + A \log(0.9/v_w) t_p^p" if best_name == "anchored_log_power" else r"s = c t_p^a v_w^b")
        ),
        "best_params_named": (
            {"A": float(best_params[0]), "m": float(best_params[1]), "p": float(best_params[2])}
            if best_name == "anchored_diff_power"
            else ({"A": float(best_params[0]), "p": float(best_params[1])} if best_name == "anchored_log_power" else {"c": float(best_params[0]), "a": float(best_params[1]), "b": float(best_params[2])})
        ),
        "outputs": {
            "all_points": str((outdir / "s_vs_tp_powerlaw_fit.png").resolve()),
            "binned": str((outdir / "s_vs_tp_powerlaw_fit_binned.png").resolve()),
            "residuals": str((outdir / "s_powerlaw_residuals.png").resolve()),
            "table": str((outdir / "pointwise_shift_with_fit.csv").resolve()),
        },
        "residual_by_vw": [
            {
                "v_w": float(vw),
                "rel_rmse": rel_rmse(sub["s_pointwise"], sub["s_fit"]),
                "median_abs_frac_resid": float(np.median(np.abs(sub["frac_resid"]))),
            }
            for vw, sub in work.groupby("v_w", sort=True)
        ],
    }
    save_json(outdir / "final_summary.json", summary)
    print(json.dumps(to_native(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
