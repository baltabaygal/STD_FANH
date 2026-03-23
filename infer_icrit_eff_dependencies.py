#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import collapse_and_fit_fanh_tosc as cf


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "results_icrit_eff"
VW_MAP = {"v3": 0.3, "v5": 0.5, "v7": 0.7, "v9": 0.9}
ODE_TABLES = [
    ROOT / "ode/analysis/data/dm_tp_fitready_H0p500.txt",
    ROOT / "ode/analysis/data/dm_tp_fitready_H1p000.txt",
    ROOT / "ode/analysis/data/dm_tp_fitready_H1p500.txt",
    ROOT / "ode/analysis/data/dm_tp_fitready_H2p000.txt",
]
TARGET_H = [0.5, 1.0, 1.5, 2.0]
DPI = 220


def save_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def potential(theta):
    theta = np.asarray(theta, dtype=np.float64)
    return 1.0 - np.cos(theta)


def cumtrapz_same(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    out = np.zeros_like(y, dtype=np.float64)
    out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dx, dtype=np.float64)
    return out


def load_direct_ode_table(path: Path) -> pd.DataFrame:
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 10:
        raise RuntimeError(f"Expected at least 10 columns in {path}, got {arr.shape[1]}")
    df = pd.DataFrame(
        {
            "H": arr[:, 0].astype(np.float64),
            "t_star": arr[:, 1].astype(np.float64),
            "theta": arr[:, 2].astype(np.float64),
            "tp": arr[:, 3].astype(np.float64),
            "tp_over_tosc": arr[:, 4].astype(np.float64),
            "Ea3_PT": arr[:, 5].astype(np.float64),
            "Ea3_noPT": arr[:, 6].astype(np.float64),
            "fanh_PT": arr[:, 7].astype(np.float64),
            "fanh_noPT": arr[:, 8].astype(np.float64),
            "xi": arr[:, 9].astype(np.float64),
        }
    )
    return df


def load_ode_all() -> pd.DataFrame:
    frames = []
    for path in ODE_TABLES:
        if not path.exists():
            raise FileNotFoundError(f"Missing direct ODE fit-ready table: {path}")
        frames.append(load_direct_ode_table(path))
    df = pd.concat(frames, ignore_index=True)
    df = df[np.isfinite(df["H"]) & np.isfinite(df["theta"]) & np.isfinite(df["tp"]) & np.isfinite(df["xi"])].copy()
    return df


def build_inverse_curves(ode_df: pd.DataFrame):
    inverse = {}
    for (H, theta), sub in ode_df.groupby(["H", "theta"], sort=True):
        sub = sub.sort_values("tp").copy()
        tp = sub["tp"].to_numpy(dtype=np.float64)
        xi = sub["xi"].to_numpy(dtype=np.float64)
        # Enforce monotonicity against tiny numerical wiggles.
        xi_mono = np.maximum.accumulate(xi)
        keep = np.concatenate([[True], np.diff(xi_mono) > 1.0e-12])
        xi_u = xi_mono[keep]
        tp_u = tp[keep]
        if len(tp_u) < 3:
            continue
        inverse[(float(H), float(theta))] = {
            "tp": tp_u,
            "xi": xi_u,
            "xi_min": float(np.min(xi_u)),
            "xi_max": float(np.max(xi_u)),
        }
    return inverse


def infer_t_eff(row, inverse_map):
    H = float(row.H)
    theta = float(row.theta)
    theta_keys = np.array(sorted(k[1] for k in inverse_map.keys() if np.isclose(k[0], H, atol=1.0e-12)), dtype=np.float64)
    if theta_keys.size == 0:
        return np.nan
    idx = int(np.argmin(np.abs(theta_keys - theta)))
    if abs(theta_keys[idx] - theta) > 5.0e-4:
        return np.nan
    key = (H, float(theta_keys[idx]))
    rec = inverse_map[key]
    xi = float(row.xi)
    if xi < rec["xi_min"] or xi > rec["xi_max"]:
        return np.nan
    return float(np.interp(xi, rec["xi"], rec["tp"]))


@dataclass
class ICurve:
    t: np.ndarray
    I: np.ndarray

    def eval(self, t_eff: float) -> float:
        if not np.isfinite(t_eff) or t_eff < self.t[0] or t_eff > self.t[-1]:
            return np.nan
        return float(np.interp(t_eff, self.t, self.I))


class PercolationICache:
    def __init__(self):
        self.cache: dict[tuple[float, float, float], ICurve] = {}

    def build_curve(self, H_star: float, beta_over_H: float, vw: float) -> ICurve:
        H = float(H_star)
        bH = float(beta_over_H)
        vw = float(vw)
        beta = bH * H
        Gamma0 = H**4
        t_PT = 1.0 / (2.0 * H)
        tmax = cf.TMAX_FAC_default / H if hasattr(cf, "TMAX_FAC_default") else 200.0 / H
        N = cf.NTAU_PERC_default if hasattr(cf, "NTAU_PERC_default") else 4000
        LOGG_CLIP = cf.LOGG_CLIP_default if hasattr(cf, "LOGG_CLIP_default") else 250.0
        eta_max = math.sqrt(2.0 * tmax / H)
        eta = np.linspace(1e-12 / max(H, 1e-300), eta_max, N, dtype=np.float64)
        a = H * eta
        t = 0.5 * H * eta**2
        logG = math.log(max(Gamma0, 1e-300)) + beta * (t - t_PT)
        logG = np.minimum(logG, LOGG_CLIP)
        Gamma = np.exp(logG).astype(np.float64)
        w = Gamma * (a**4)
        M0 = cumtrapz_same(w, eta)
        M1 = cumtrapz_same(w * eta, eta)
        M2 = cumtrapz_same(w * eta**2, eta)
        M3 = cumtrapz_same(w * eta**3, eta)
        K = (eta**3) * M0 - 3.0 * (eta**2) * M1 + 3.0 * eta * M2 - M3
        K = np.maximum(K, 0.0)
        I = (4.0 * np.pi / 3.0) * (vw**3) * K
        I = np.maximum.accumulate(I)
        return ICurve(t=t, I=I)

    def get(self, H_star: float, beta_over_H: float, vw: float) -> ICurve:
        key = (float(H_star), float(beta_over_H), float(vw))
        if key not in self.cache:
            self.cache[key] = self.build_curve(*key)
        return self.cache[key]


def load_lattice_all() -> pd.DataFrame:
    frames = []
    rho_path = cf.resolve_first_existing(cf.RHO_CANDIDATES, "")
    f0_table = cf.load_f0_table(rho_path, TARGET_H)
    for tag, vw in VW_MAP.items():
        raw = ROOT / "lattice_data" / "data" / f"energy_ratio_by_theta_data_{tag}.txt"
        if not raw.exists():
            raise FileNotFoundError(f"Missing lattice raw file: {raw}")
        df = cf.load_lattice_data(raw, vw, TARGET_H)
        df = cf.merge_f0(df, f0_table)
        df["vw"] = float(vw)
        df["vw_tag"] = tag
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[np.isfinite(df["H"]) & np.isfinite(df["theta"]) & np.isfinite(df["tp"]) & np.isfinite(df["xi"])].copy()
    return df


def fit_log_model(df: pd.DataFrame, predictors: list[str]):
    work = df.copy()
    y = np.log(work["I_eff"].to_numpy(dtype=np.float64))
    cols = [np.ones(len(work), dtype=np.float64)]
    names = ["const"]
    for pred in predictors:
        vals = np.log(work[pred].to_numpy(dtype=np.float64))
        cols.append(vals)
        names.append(f"log_{pred}")
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yfit = X @ coef
    rss = float(np.sum((y - yfit) ** 2))
    n = len(y)
    k = len(coef)
    aic = float(n * math.log(max(rss, 1.0e-18) / n) + 2.0 * k)
    bic = float(n * math.log(max(rss, 1.0e-18) / n) + k * math.log(n))
    rel_rmse = float(np.sqrt(np.mean(((np.exp(yfit) - np.exp(y)) / np.maximum(np.exp(y), 1.0e-18)) ** 2)))
    return {
        "predictors": predictors,
        "coef_names": names,
        "coef": coef.tolist(),
        "rss": rss,
        "aic": aic,
        "bic": bic,
        "rel_rmse": rel_rmse,
    }


def add_model_predictions(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    work = df.copy()
    Xcols = [np.ones(len(work), dtype=np.float64)]
    for pred in model["predictors"]:
        Xcols.append(np.log(work[pred].to_numpy(dtype=np.float64)))
    X = np.column_stack(Xcols)
    coef = np.asarray(model["coef"], dtype=np.float64)
    work["I_eff_fit"] = np.exp(X @ coef)
    work["rel_resid"] = (work["I_eff_fit"] - work["I_eff"]) / np.maximum(work["I_eff"], 1.0e-18)
    return work


def plot_model_comparison(models: list[dict], path: Path):
    names = ["const" if not m["predictors"] else "+".join(m["predictors"]) for m in models]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].bar(names, [m["rel_rmse"] for m in models], color="tab:blue")
    axes[0].set_title("rel-RMSE")
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


def plot_scatter(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    sc = axes[0].scatter(df["beta_over_H"], df["I_eff"], c=df["vw"], cmap="viridis", s=26)
    axes[0].set_xlabel(r"$\beta/H_*$")
    axes[0].set_ylabel(r"$I_{\rm crit,eff}$")
    axes[0].set_title(r"$I_{\rm crit,eff}$ vs $\beta/H_*$")
    axes[1].scatter(df["H"], df["I_eff"], c=df["vw"], cmap="viridis", s=26)
    axes[1].set_xlabel(r"$H_*$")
    axes[1].set_ylabel(r"$I_{\rm crit,eff}$")
    axes[1].set_title(r"$I_{\rm crit,eff}$ vs $H_*$")
    axes[2].scatter(df["vw"], df["I_eff"], c=df["beta_over_H"], cmap="plasma", s=26)
    axes[2].set_xlabel(r"$v_w$")
    axes[2].set_ylabel(r"$I_{\rm crit,eff}$")
    axes[2].set_title(r"$I_{\rm crit,eff}$ vs $v_w$")
    for ax in axes:
        ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=axes[:2], fraction=0.03, pad=0.04)
    cbar.set_label(r"$v_w$")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_residual_heatmap(df: pd.DataFrame, path: Path):
    pivot = df.pivot_table(index="theta", columns="vw", values="rel_resid", aggfunc="mean").sort_index()
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    pcm = ax.pcolormesh(
        np.arange(len(pivot.columns) + 1),
        np.arange(len(pivot.index) + 1),
        pivot.to_numpy(dtype=np.float64),
        cmap="coolwarm",
        vmin=-0.3,
        vmax=0.3,
        shading="auto",
    )
    ax.set_xticks(np.arange(len(pivot.columns)) + 0.5)
    ax.set_xticklabels([f"{float(v):.1f}" for v in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)) + 0.5)
    ax.set_yticklabels([f"{float(t):.3f}" for t in pivot.index])
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$\theta$")
    ax.set_title("Best-model residuals in inferred $I_{\\rm crit,eff}$")
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    lattice = load_lattice_all()
    ode = load_ode_all()
    inv = build_inverse_curves(ode)
    lattice["t_eff"] = lattice.apply(lambda row: infer_t_eff(row, inv), axis=1)
    lattice = lattice[np.isfinite(lattice["t_eff"])].copy()
    if lattice.empty:
        raise RuntimeError("No lattice rows could be matched to direct ODE xi(tp) curves.")
    icache = PercolationICache()
    lattice["I_eff"] = [
        icache.get(float(h), float(bh), float(vw)).eval(float(teff))
        for h, bh, vw, teff in zip(
            lattice["H"].to_numpy(dtype=np.float64),
            lattice["beta_over_H"].to_numpy(dtype=np.float64),
            lattice["vw"].to_numpy(dtype=np.float64),
            lattice["t_eff"].to_numpy(dtype=np.float64),
        )
    ]
    lattice["F_false_eff"] = np.exp(-lattice["I_eff"].to_numpy(dtype=np.float64))
    lattice = lattice[np.isfinite(lattice["I_eff"]) & (lattice["I_eff"] > 0.0)].copy()
    lattice["teff_over_tp"] = lattice["t_eff"] / lattice["tp"]
    lattice.to_csv(OUTDIR / "inferred_icrit_eff.csv", index=False)

    model_specs = [
        [],
        ["vw"],
        ["H"],
        ["beta_over_H"],
        ["vw", "H"],
        ["vw", "beta_over_H"],
        ["H", "beta_over_H"],
        ["vw", "H", "beta_over_H"],
    ]
    models = [fit_log_model(lattice, spec) for spec in model_specs]
    models = sorted(models, key=lambda rec: rec["bic"])
    save_json(OUTDIR / "model_comparison.json", {"models": models})
    best = models[0]
    lattice_best = add_model_predictions(lattice, best)
    lattice_best.to_csv(OUTDIR / "best_model_predictions.csv", index=False)

    plot_model_comparison(models, OUTDIR / "model_comparison.png")
    plot_scatter(lattice, OUTDIR / "icrit_eff_scatter.png")
    plot_residual_heatmap(lattice_best, OUTDIR / "best_model_residual_heatmap.png")

    clean = lattice[lattice["H"] >= 1.5].copy()
    clean_models = [fit_log_model(clean, spec) for spec in model_specs]
    clean_models = sorted(clean_models, key=lambda rec: rec["bic"])
    save_json(OUTDIR / "model_comparison_cleanH15H20.json", {"models": clean_models})

    summary = {
        "status": "ok",
        "n_points_used": int(len(lattice)),
        "best_model_allH": best,
        "best_model_cleanH15H20": clean_models[0],
        "icrit_eff_range": [float(lattice["I_eff"].min()), float(lattice["I_eff"].max())],
        "teff_over_tp_range": [float(lattice["teff_over_tp"].min()), float(lattice["teff_over_tp"].max())],
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
