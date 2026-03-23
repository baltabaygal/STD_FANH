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

import collapse_and_fit_fanh_tosc as cf


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "results_fanh_teff_refit"
TARGET_H = [1.5, 2.0]
T_OSC = 1.5
DPI = 220


def save_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def cumtrapz_same(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    out = np.zeros_like(y, dtype=np.float64)
    out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dx, dtype=np.float64)
    return out


class ICurve:
    def __init__(self, H_star: float, beta_over_H: float, vw: float):
        H = float(H_star)
        bH = float(beta_over_H)
        vw = float(vw)
        beta = bH * H
        Gamma0 = H**4
        t_PT = 1.0 / (2.0 * H)
        N = 4000
        tmax = 200.0 / H
        eta_max = math.sqrt(2.0 * tmax / H)
        eta = np.linspace(1e-12 / max(H, 1e-300), eta_max, N, dtype=np.float64)
        a = H * eta
        t = 0.5 * H * eta**2
        logG = math.log(max(Gamma0, 1e-300)) + beta * (t - t_PT)
        logG = np.minimum(logG, 250.0)
        Gamma = np.exp(logG)
        w = Gamma * (a**4)
        M0 = cumtrapz_same(w, eta)
        M1 = cumtrapz_same(w * eta, eta)
        M2 = cumtrapz_same(w * eta**2, eta)
        M3 = cumtrapz_same(w * eta**3, eta)
        K = (eta**3) * M0 - 3.0 * (eta**2) * M1 + 3.0 * eta * M2 - M3
        K = np.maximum(K, 0.0)
        I = (4.0 * np.pi / 3.0) * (vw**3) * K
        self.t = t
        self.I = np.maximum.accumulate(I)

    def inverse(self, Ieff: float) -> float:
        if not np.isfinite(Ieff) or Ieff < self.I[0] or Ieff > self.I[-1]:
            return np.nan
        return float(np.interp(Ieff, self.I, self.t))


def load_clean_predictions():
    p = ROOT / "results_icrit_eff_clean_analysis" / "clean_predictions.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing clean predictions file: {p}")
    df = pd.read_csv(p)
    df = df[df["H"].isin(TARGET_H)].copy()
    return df


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


def fit_universal_model(df: pd.DataFrame):
    theta_values = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    f0_map = {float(th): float(np.median(df.loc[np.isclose(df["theta"], th, atol=5.0e-4), "F0"])) for th in theta_values}
    finf0 = np.array([max(1.0e-8, float(np.median(df.loc[np.isclose(df["theta"], th, atol=5.0e-4), "I_eff"]))) for th in theta_values])
    x0 = np.concatenate([[1.5, 2.0], finf0])
    lower = np.concatenate([[0.1, 0.1], np.full(len(theta_values), 1.0e-8)])
    upper = np.concatenate([[20.0, 50.0], np.full(len(theta_values), 1.0e3)])
    theta_index = np.array([cf.nearest_theta(theta_values, th) for th in df["theta"].to_numpy(dtype=np.float64)], dtype=np.int64)
    f0 = df["F0"].to_numpy(dtype=np.float64)
    teff = df["t_eff_model"].to_numpy(dtype=np.float64)
    xi_data = df["xi"].to_numpy(dtype=np.float64)

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
    return {
        "params": res.x,
        "theta_values": theta_values,
        "theta_index": theta_index,
        "xi_fit": xi_fit,
        "rel_rmse": rel_rmse,
        "success": bool(res.success),
        "message": res.message,
    }


def per_theta_rmse(df: pd.DataFrame, xi_fit: np.ndarray) -> pd.DataFrame:
    rows = []
    work = df.copy()
    work["xi_fit"] = xi_fit
    for theta, sub in work.groupby("theta", sort=True):
        rel = np.sqrt(np.mean(np.square((sub["xi_fit"] - sub["xi"]) / np.maximum(sub["xi"], 1.0e-12))))
        rows.append({"theta": float(theta), "rel_rmse": float(rel)})
    return pd.DataFrame(rows).sort_values("theta").reset_index(drop=True)


def plot_overlay(df: pd.DataFrame, fit_result: dict, path: Path):
    theta_values = fit_result["theta_values"]
    params = fit_result["params"]
    t_c = float(params[0])
    r = float(params[1])
    finf = np.asarray(params[2:], dtype=np.float64)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sorted(df["H"].unique()))))
    for ax, theta in zip(axes.flat, theta_values):
        sub = df[np.isclose(df["theta"], theta, atol=5.0e-4)].copy().sort_values("t_eff_model")
        idx = cf.nearest_theta(theta_values, theta)
        for color, (H, hsub) in zip(colors, sub.groupby("H", sort=True)):
            ax.plot(hsub["t_eff_model"], hsub["fanh"], "o", ms=2.8, color=color, label=rf"$H={H:g}$")
        xfit = np.geomspace(float(sub["t_eff_model"].min()), float(sub["t_eff_model"].max()), 250)
        f0 = float(np.median(sub["F0"]))
        fanh_fit = finf[idx] / max(f0, 1.0e-18) + f0 / (np.power(xfit / T_OSC, 1.5) * (1.0 + np.power(xfit / max(t_c, 1.0e-12), r)))
        ax.plot(xfit, fanh_fit, color="black", lw=1.8)
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


def plot_residual_vs_theta(theta_rmse: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(theta_rmse["theta"], theta_rmse["rel_rmse"], "o-", color="tab:red")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("rel-RMSE")
    ax.set_title(r"Universal refit using inferred $t_{\rm eff}$")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_teff_ratio(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    sc = ax.scatter(df["beta_over_H"], df["t_eff_model"] / df["tp"], c=df["vw"], cmap="viridis", s=18)
    ax.set_xlabel(r"$\beta/H_*$")
    ax.set_ylabel(r"$t_{\rm eff}/t_p$")
    ax.set_title(r"Inferred $t_{\rm eff}/t_p$")
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, label=r"$v_w$")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    clean = load_clean_predictions()
    df = build_teff(clean)
    df["fanh"] = df["xi"].to_numpy(dtype=np.float64) * df["F0"].to_numpy(dtype=np.float64) / np.power(df["t_eff_model"].to_numpy(dtype=np.float64) / T_OSC, 1.5)
    fit_result = fit_universal_model(df)
    theta_rmse = per_theta_rmse(df, fit_result["xi_fit"])
    theta_rmse.to_csv(OUTDIR / "per_theta_rmse.csv", index=False)
    df.to_csv(OUTDIR / "teff_dataset.csv", index=False)
    plot_overlay(df, fit_result, OUTDIR / "fanh_teff_overlay.png")
    plot_residual_vs_theta(theta_rmse, OUTDIR / "residual_vs_theta.png")
    plot_teff_ratio(df, OUTDIR / "teff_over_tp.png")
    payload = {
        "status": "ok",
        "fit": {
            "t_c": float(fit_result["params"][0]),
            "r": float(fit_result["params"][1]),
            "rel_rmse": float(fit_result["rel_rmse"]),
            "success": bool(fit_result["success"]),
            "message": fit_result["message"],
        },
        "teff_over_tp_range": [float(df["t_eff_model"].div(df["tp"]).min()), float(df["t_eff_model"].div(df["tp"]).max())],
        "theta_worst": theta_rmse.sort_values("rel_rmse", ascending=False).iloc[0].to_dict(),
        "theta_best": theta_rmse.sort_values("rel_rmse", ascending=True).iloc[0].to_dict(),
    }
    save_json(OUTDIR / "final_summary.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
        save_json(OUTDIR / "_error.json", payload)
        print(json.dumps(payload, indent=2))
        raise
