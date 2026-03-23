#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from refit_fanh_with_teff_model import ICurve, T_OSC
import collapse_and_fit_fanh_tosc as cf


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results_icrit_eff"
OUTDIR = ROOT / "results_icrit_eff_theta_term"
TARGET_H = [1.5, 2.0]
DPI = 220


def save_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_clean_data():
    df = pd.read_csv(BASE / "inferred_icrit_eff.csv")
    return df[df["H"].isin(TARGET_H)].copy()


def theta_features(theta: np.ndarray) -> dict[str, np.ndarray]:
    theta = np.asarray(theta, dtype=np.float64)
    h = np.log(np.e / np.maximum(np.cos(theta / 2.0) ** 2, 1.0e-15))
    return {
        "none": np.zeros_like(theta),
        "log_h": np.log(np.maximum(h, 1.0e-15)),
        "h": h,
        "theta": theta,
        "cos_theta": np.cos(theta),
        "theta_pi_minus": theta * (np.pi - theta),
        "inv_hilltop": 1.0 / np.maximum(np.pi - theta, 1.0e-6),
    }


def fit_log_model(df: pd.DataFrame, theta_term: str):
    feats = theta_features(df["theta"].to_numpy(dtype=np.float64))
    y = np.log(df["I_eff"].to_numpy(dtype=np.float64))
    cols = [
        np.ones(len(df), dtype=np.float64),
        np.log(df["vw"].to_numpy(dtype=np.float64)),
        np.log(df["H"].to_numpy(dtype=np.float64)),
        np.log(df["beta_over_H"].to_numpy(dtype=np.float64)),
    ]
    names = ["const", "log_vw", "log_H", "log_beta_over_H"]
    if theta_term != "none":
        cols.append(feats[theta_term])
        names.append(theta_term)
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yfit = X @ coef
    rss = float(np.sum((y - yfit) ** 2))
    n = len(y)
    k = len(coef)
    aic = float(n * math.log(max(rss, 1.0e-18) / n) + 2.0 * k)
    bic = float(n * math.log(max(rss, 1.0e-18) / n) + k * math.log(n))
    rel_rmse = float(np.sqrt(np.mean(((np.exp(yfit) - np.exp(y)) / np.maximum(np.exp(y), 1.0e-18)) ** 2)))
    out = df.copy()
    out["I_eff_fit"] = np.exp(yfit)
    out["rel_resid"] = (out["I_eff_fit"] - out["I_eff"]) / np.maximum(out["I_eff"], 1.0e-18)
    return {
        "theta_term": theta_term,
        "coef_names": names,
        "coef": coef.tolist(),
        "rss": rss,
        "aic": aic,
        "bic": bic,
        "rel_rmse": rel_rmse,
        "predictions": out,
    }


def build_teff(df: pd.DataFrame) -> pd.DataFrame:
    cache: dict[tuple[float, float, float], ICurve] = {}
    teff = []
    for row in df.itertuples(index=False):
        key = (float(row.H), float(row.beta_over_H), float(row.vw))
        if key not in cache:
            cache[key] = ICurve(*key)
        teff.append(cache[key].inverse(float(row.I_eff_fit)))
    out = df.copy()
    out["t_eff_model"] = np.asarray(teff, dtype=np.float64)
    out = out[np.isfinite(out["t_eff_model"]) & (out["t_eff_model"] > 0.0)].copy()
    return out


def fit_universal_fanh(df: pd.DataFrame):
    theta_values = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    theta_index = np.array([cf.nearest_theta(theta_values, th) for th in df["theta"].to_numpy(dtype=np.float64)], dtype=np.int64)
    f0 = df["F0"].to_numpy(dtype=np.float64)
    xi_data = df["xi"].to_numpy(dtype=np.float64)
    teff = df["t_eff_model"].to_numpy(dtype=np.float64)

    finf0 = []
    for th in theta_values:
        sub = df[np.isclose(df["theta"], th, atol=5.0e-4)]
        guess = np.median(sub["I_eff_fit"].to_numpy(dtype=np.float64))
        finf0.append(max(float(guess), 1.0e-8))
    x0 = np.concatenate([[1.5, 2.0], np.asarray(finf0, dtype=np.float64)])
    lower = np.concatenate([[0.1, 0.1], np.full(len(theta_values), 1.0e-8)])
    upper = np.concatenate([[20.0, 50.0], np.full(len(theta_values), 1.0e3)])

    def model(par):
        t_c = float(par[0])
        r = float(par[1])
        finf = np.asarray(par[2:], dtype=np.float64)
        return np.power(teff / T_OSC, 1.5) * finf[theta_index] / np.maximum(f0 * f0, 1.0e-18) + 1.0 / (
            1.0 + np.power(teff / max(t_c, 1.0e-12), r)
        )

    def resid(par):
        xi_fit = model(par)
        return (xi_fit - xi_data) / np.maximum(xi_data, 1.0e-12)

    res0 = least_squares(resid, x0=x0, bounds=(lower, upper), loss="soft_l1", f_scale=0.05, max_nfev=8000)
    res = least_squares(resid, x0=res0.x, bounds=(lower, upper), loss="linear", max_nfev=8000)
    xi_fit = model(res.x)
    rel_rmse = float(np.sqrt(np.mean(np.square((xi_fit - xi_data) / np.maximum(xi_data, 1.0e-12)))))
    pred = df.copy()
    pred["xi_fit"] = xi_fit
    rows = []
    for theta, sub in pred.groupby("theta", sort=True):
        rel = np.sqrt(np.mean(np.square((sub["xi_fit"] - sub["xi"]) / np.maximum(sub["xi"], 1.0e-12))))
        rows.append({"theta": float(theta), "rel_rmse": float(rel)})
    return {
        "params": res.x.tolist(),
        "t_c": float(res.x[0]),
        "r": float(res.x[1]),
        "rel_rmse": rel_rmse,
        "theta_rmse": pd.DataFrame(rows).sort_values("theta").reset_index(drop=True),
        "predictions": pred,
    }


def plot_theta_term_comparison(models: list[dict], path: Path):
    names = [m["theta_term"] for m in models]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].bar(names, [m["rel_rmse"] for m in models], color="tab:blue")
    axes[0].set_title(r"$I_{\rm eff}$ rel-RMSE")
    axes[1].bar(names, [m["aic"] for m in models], color="tab:orange")
    axes[1].set_title("AIC")
    axes[2].bar(names, [m["bic"] for m in models], color="tab:green")
    axes[2].set_title("BIC")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_best_residual_vs_theta(theta_rmse: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(theta_rmse["theta"], theta_rmse["rel_rmse"], "o-", color="tab:red")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("rel-RMSE")
    ax.set_title(r"Universal fanh refit after best $\theta$-term $I_{\rm eff}$ model")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_best_teff_overlay(df: pd.DataFrame, path: Path):
    theta_values = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sorted(df["H"].unique()))))
    for ax, theta in zip(axes.flat, theta_values):
        sub = df[np.isclose(df["theta"], theta, atol=5.0e-4)].copy().sort_values("t_eff_model")
        for color, (H, hsub) in zip(colors, sub.groupby("H", sort=True)):
            fanh = hsub["xi"].to_numpy(dtype=np.float64) * hsub["F0"].to_numpy(dtype=np.float64) / np.power(hsub["t_eff_model"].to_numpy(dtype=np.float64) / T_OSC, 1.5)
            ax.plot(hsub["t_eff_model"], fanh, "o", ms=2.8, color=color, label=rf"$H={H:g}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$t_{\rm eff}$")
        ax.set_ylabel("fanh")
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = load_clean_data()
    candidates = ["none", "log_h", "h", "theta", "cos_theta", "theta_pi_minus", "inv_hilltop"]
    models = [fit_log_model(df, name) for name in candidates]
    models_sorted = sorted(models, key=lambda rec: rec["bic"])
    best = models_sorted[0]

    teff_df = build_teff(best["predictions"])
    fanh_fit = fit_universal_fanh(teff_df)

    pd.DataFrame(
        [{k: v for k, v in rec.items() if k != "predictions"} for rec in models_sorted]
    ).to_csv(OUTDIR / "theta_term_model_comparison.csv", index=False)
    save_json(
        OUTDIR / "theta_term_model_comparison.json",
        {"models": [{k: v for k, v in rec.items() if k != "predictions"} for rec in models_sorted]},
    )
    fanh_fit["theta_rmse"].to_csv(OUTDIR / "per_theta_rmse.csv", index=False)
    teff_df.to_csv(OUTDIR / "best_teff_dataset.csv", index=False)

    plot_theta_term_comparison(models_sorted, OUTDIR / "theta_term_model_comparison.png")
    plot_best_residual_vs_theta(fanh_fit["theta_rmse"], OUTDIR / "best_theta_term_residual_vs_theta.png")
    plot_best_teff_overlay(teff_df, OUTDIR / "best_theta_term_fanh_overlay.png")

    summary = {
        "status": "ok",
        "best_theta_term_model": {k: v for k, v in best.items() if k != "predictions"},
        "fanh_refit": {
            "t_c": fanh_fit["t_c"],
            "r": fanh_fit["r"],
            "rel_rmse": fanh_fit["rel_rmse"],
            "theta_worst": fanh_fit["theta_rmse"].sort_values("rel_rmse", ascending=False).iloc[0].to_dict(),
            "theta_best": fanh_fit["theta_rmse"].sort_values("rel_rmse", ascending=True).iloc[0].to_dict(),
        },
    }
    save_json(OUTDIR / "final_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
        save_json(OUTDIR / "_error.json", payload)
        print(json.dumps(payload, indent=2))
        raise
