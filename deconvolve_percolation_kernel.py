#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.optimize import minimize, minimize_scalar
from statsmodels.tools.eval_measures import aic as sm_aic
from statsmodels.tools.eval_measures import bic as sm_bic
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fit_vw_amplitude as base_fit


OUTDIR_DEFAULT = "results_deconv"
XI_ALL_DEFAULT = ROOT / "data/xi_all.csv"
XI_ODE_DEFAULT = ROOT / "ode/xi_DM_ODE_results.txt"
THETA_TARGETS = np.array([0.262, 0.785, 1.309, 1.833, 2.356, 2.880], dtype=np.float64)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_native(payload), indent=2, sort_keys=True))


def error_exit(outdir: Path, exc: Exception):
    payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "_error.json", payload)
    print(json.dumps(payload, sort_keys=True))


def parse_args():
    p = argparse.ArgumentParser(description="Regularized non-negative deconvolution of the percolation kernel.")
    p.add_argument("--xi-latt", type=str, default=str(XI_ALL_DEFAULT))
    p.add_argument("--xi-ode", type=str, default=str(XI_ODE_DEFAULT))
    p.add_argument("--rho", type=str, default="")
    p.add_argument("--vw-list", type=str, default="0.3,0.5,0.7,0.9")
    p.add_argument("--theta-list", type=float, nargs="*", default=None)
    p.add_argument("--h-values", type=float, nargs="+", default=[1.5, 2.0])
    p.add_argument("--tp-min", type=float, default=None)
    p.add_argument("--tp-max", type=float, default=None)
    p.add_argument("--grid-size", type=int, default=120)
    p.add_argument("--lambda-grid", type=float, nargs="*", default=None)
    p.add_argument("--nboot", type=int, default=200)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--outdir", type=str, default=OUTDIR_DEFAULT)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_vw_list(text: str):
    vals = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals.append(float(chunk))
    if not vals:
        raise ValueError("vw-list must contain at least one value.")
    return np.array(sorted(set(vals)), dtype=np.float64)


def resolve_lambda_grid(values):
    if values:
        arr = np.array(values, dtype=np.float64)
    else:
        arr = np.logspace(-6.0, 0.0, 13)
    arr = arr[np.isfinite(arr) & (arr >= 0.0)]
    if arr.size == 0:
        raise ValueError("lambda-grid must contain at least one non-negative value.")
    return np.unique(arr)


def log_interp(x, y, xq):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if np.count_nonzero(mask) < 2:
        return np.full_like(xq, np.nan, dtype=np.float64)
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x, idx = np.unique(x, return_index=True)
    y = y[idx]
    if x.size < 2:
        return np.full_like(xq, np.nan, dtype=np.float64)
    f = interp1d(np.log(x), np.log(y), bounds_error=False, fill_value=np.nan, assume_sorted=True)
    out = np.exp(f(np.log(np.maximum(xq, 1.0e-300))))
    out[(xq < x[0]) | (xq > x[-1])] = np.nan
    return out


def load_lattice_dataframe(args, outdir: Path):
    xi_latt_path = Path(args.xi_latt).resolve()
    if xi_latt_path.exists():
        print(f"[load] lattice unified table {xi_latt_path}")
        df = pd.read_csv(xi_latt_path)
        cols = {str(c).lower(): c for c in df.columns}
        needed = ["theta", "tp", "xi", "h", "v_w"]
        if not all(col in cols for col in needed):
            raise RuntimeError(f"{xi_latt_path} must contain columns theta,tp,xi,H,v_w.")
        out = pd.DataFrame(
            {
                "theta": df[cols["theta"]].astype(float),
                "tp": df[cols["tp"]].astype(float),
                "xi": df[cols["xi"]].astype(float),
                "H": df[cols["h"]].astype(float),
                "v_w": df[cols["v_w"]].astype(float),
            }
        )
        return out

    print("[load] lattice unified table missing, autodiscovering raw v* runs")
    tags = [f"v{int(round(vw * 10))}" for vw in parse_vw_list(args.vw_list)]
    load_args = argparse.Namespace(
        rho=args.rho,
        vw_folders=tags,
        h_values=args.h_values,
        tp_min=args.tp_min,
        tp_max=args.tp_max,
        bootstrap=0,
        n_jobs=1,
        reg_Finf=0.0,
        tc0=1.5,
        fix_tc=True,
        dpi=220,
        outdir=str(outdir),
    )
    df, _, _ = base_fit.prepare_dataframe(load_args, outdir)
    return df.copy()


def load_ode_dataframe(path: Path, h_values, vw_values):
    if not path.exists():
        raise FileNotFoundError(f"Missing ODE file: {path}")
    df = pd.read_csv(path, sep=r"\s+|,", engine="python", comment="#", header=None)
    if df.shape[1] < 6:
        raise RuntimeError(f"Could not parse ODE file {path}; expected at least 6 columns.")
    df = df.iloc[:, :6].copy()
    df.columns = ["v_w", "theta", "H", "beta_over_H", "tp", "xi"]
    out = df.astype(float)
    out = out[
        np.isfinite(out["v_w"])
        & np.isfinite(out["theta"])
        & np.isfinite(out["H"])
        & np.isfinite(out["tp"])
        & np.isfinite(out["xi"])
    ].copy()
    out = out[
        out["H"].isin(list(np.asarray(h_values, dtype=np.float64)))
        & out["v_w"].isin(list(np.asarray(vw_values, dtype=np.float64)))
        & (out["tp"] > 0.0)
        & (out["xi"] > 0.0)
    ].copy()
    if out.empty:
        raise RuntimeError(f"No matching ODE rows found in {path}.")
    return out.sort_values(["v_w", "H", "theta", "tp"]).reset_index(drop=True)


def resolve_theta_list(theta_values, requested):
    theta_values = np.asarray(sorted(theta_values), dtype=np.float64)
    if requested:
        out = []
        for target in requested:
            idx = int(np.argmin(np.abs(theta_values - float(target))))
            val = float(theta_values[idx])
            if val not in out:
                out.append(val)
        return np.asarray(out, dtype=np.float64)
    out = []
    for target in THETA_TARGETS:
        idx = int(np.argmin(np.abs(theta_values - float(target))))
        val = float(theta_values[idx])
        if val not in out:
            out.append(val)
    return np.asarray(out, dtype=np.float64)


def snap_theta_values(df, theta_list, atol=5.0e-3):
    df = df.copy()
    ref = np.asarray(theta_list, dtype=np.float64)
    theta = df["theta"].to_numpy(dtype=np.float64)
    idx = np.argmin(np.abs(theta[:, None] - ref[None, :]), axis=1)
    snapped = ref[idx]
    keep = np.abs(theta - snapped) <= float(atol)
    return pd.DataFrame(
        {
            **{col: df[col].to_numpy()[keep] for col in df.columns if col != "theta"},
            "theta": snapped[keep],
        }
    )


def filter_dataframe(df, vw_values, theta_list, h_values, tp_min, tp_max):
    df = snap_theta_values(df, theta_list)
    df = df.copy()
    h_arr = np.asarray(h_values, dtype=np.float64)
    vw_arr = np.asarray(vw_values, dtype=np.float64)
    h_mask = np.min(np.abs(df["H"].to_numpy(dtype=np.float64)[:, None] - h_arr[None, :]), axis=1) <= 1.0e-8
    vw_mask = np.min(np.abs(df["v_w"].to_numpy(dtype=np.float64)[:, None] - vw_arr[None, :]), axis=1) <= 1.0e-8
    df = df[
        vw_mask
        & h_mask
        & np.isfinite(df["tp"])
        & np.isfinite(df["xi"])
        & (df["tp"] > 0.0)
        & (df["xi"] > 0.0)
    ].copy()
    if tp_min is not None:
        df = df[df["tp"] >= float(tp_min)].copy()
    if tp_max is not None:
        df = df[df["tp"] <= float(tp_max)].copy()
    if df.empty:
        raise RuntimeError("No valid rows remained after filtering.")
    return df.sort_values(["v_w", "H", "theta", "tp"]).reset_index(drop=True)


def pooled_curve(df, grid, theta_list, h_values):
    curves = []
    keys = []
    for h in h_values:
        for theta in theta_list:
            sub = df[np.isclose(df["H"], float(h)) & np.isclose(df["theta"], float(theta), atol=5.0e-4, rtol=0.0)].sort_values("tp")
            if len(sub) < 2:
                continue
            yi = log_interp(sub["tp"].to_numpy(dtype=np.float64), sub["xi"].to_numpy(dtype=np.float64), grid)
            curves.append(yi)
            keys.append((float(h), float(theta)))
    if not curves:
        return None
    arr = np.vstack(curves)
    coverage = np.sum(np.isfinite(arr), axis=0)
    pooled = np.full(arr.shape[1], np.nan, dtype=np.float64)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        col = col[np.isfinite(col)]
        if col.size:
            pooled[j] = float(np.median(col))
    return {
        "grid": np.asarray(grid, dtype=np.float64),
        "pooled": pooled,
        "coverage": coverage,
        "matrix": arr,
        "keys": keys,
    }


def build_common_grid(latt_df, ode_df, grid_size):
    all_tp = np.concatenate([latt_df["tp"].to_numpy(dtype=np.float64), ode_df["tp"].to_numpy(dtype=np.float64)])
    all_tp = all_tp[np.isfinite(all_tp) & (all_tp > 0.0)]
    if all_tp.size < 10:
        raise RuntimeError("Not enough time points to build a common grid.")
    return np.linspace(float(np.min(all_tp)), float(np.max(all_tp)), int(grid_size))


def finalize_pooled_curves(vw, latt_df, ode_df, theta_list, h_values, grid_size):
    grid = build_common_grid(latt_df, ode_df, grid_size)
    pooled_latt = pooled_curve(latt_df, grid, theta_list, h_values)
    pooled_ode = pooled_curve(ode_df, grid, theta_list, h_values)
    if pooled_latt is None or pooled_ode is None:
        raise RuntimeError(f"Could not pool curves for v_w={vw:.1f}.")
    min_cov_latt = max(2, int(math.ceil(0.5 * len(pooled_latt["keys"]))))
    min_cov_ode = max(2, int(math.ceil(0.5 * len(pooled_ode["keys"]))))
    mask = (
        np.isfinite(pooled_latt["pooled"])
        & np.isfinite(pooled_ode["pooled"])
        & (pooled_latt["coverage"] >= min_cov_latt)
        & (pooled_ode["coverage"] >= min_cov_ode)
    )
    if np.count_nonzero(mask) < 20:
        mask = np.isfinite(pooled_latt["pooled"]) & np.isfinite(pooled_ode["pooled"])
    if np.count_nonzero(mask) < 20:
        raise RuntimeError(f"Too few overlapping pooled nodes for v_w={vw:.1f}.")
    grid = grid[mask]
    y = pooled_latt["pooled"][mask]
    r = pooled_ode["pooled"][mask]
    good = np.isfinite(grid) & np.isfinite(y) & np.isfinite(r) & (y > 0.0) & (r > 0.0)
    if np.count_nonzero(good) < 20:
        raise RuntimeError(f"Too few positive overlapping nodes for v_w={vw:.1f}.")
    return {
        "t": grid[good],
        "y": y[good],
        "r": r[good],
        "coverage_latt": pooled_latt["coverage"][mask][good],
        "coverage_ode": pooled_ode["coverage"][mask][good],
        "keys_latt": pooled_latt["keys"],
        "keys_ode": pooled_ode["keys"],
    }


def build_convolution_matrix(r):
    r = np.asarray(r, dtype=np.float64)
    n = len(r)
    k = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        k[i, : i + 1] = r[i::-1]
    return k


def second_difference_matrix(n):
    if n < 3:
        return np.zeros((0, n), dtype=np.float64)
    d = np.zeros((n - 2, n), dtype=np.float64)
    for i in range(n - 2):
        d[i, i : i + 3] = np.array([1.0, -2.0, 1.0], dtype=np.float64)
    return d


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z)
    e = np.exp(np.clip(z, -700.0, 700.0))
    return e / np.sum(e)


def weights_to_pdf(weights, dt):
    weights = np.asarray(weights, dtype=np.float64)
    return weights / max(float(dt), 1.0e-18)


def effective_k(weights):
    weights = np.asarray(weights, dtype=np.float64)
    thresh = max(1.0e-3 / max(len(weights), 1), 1.0e-6)
    return max(int(np.count_nonzero(weights > thresh)), 1)


def fit_deconvolution_for_lambda(y, k_mat, d2, lam, z0=None):
    y = np.asarray(y, dtype=np.float64)
    k_mat = np.asarray(k_mat, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)
    n = len(y)
    if z0 is None:
        z0 = np.zeros(n, dtype=np.float64)

    def objective(z):
        w = softmax(z)
        resid = k_mat @ w - y
        reg = d2 @ w if d2.size else np.zeros(0, dtype=np.float64)
        return float(np.dot(resid, resid) + float(lam) * np.dot(reg, reg))

    def gradient(z):
        w = softmax(z)
        resid = k_mat @ w - y
        g_w = 2.0 * (k_mat.T @ resid)
        if d2.size:
            reg = d2 @ w
            g_w = g_w + 2.0 * float(lam) * (d2.T @ reg)
        dot = float(np.dot(w, g_w))
        return w * (g_w - dot)

    res = minimize(objective, z0, method="L-BFGS-B", jac=gradient)
    w = softmax(res.x)
    y_hat = k_mat @ w
    resid = y_hat - y
    rss = float(np.dot(resid, resid))
    reg_norm = float(np.dot(d2 @ w, d2 @ w)) if d2.size else 0.0
    k_eff = effective_k(w)
    nobs = len(y)
    sigma2 = max(rss / max(nobs, 1), 1.0e-18)
    llf = -0.5 * nobs * (math.log(2.0 * math.pi) + math.log(sigma2) + 1.0)
    return {
        "lambda": float(lam),
        "success": bool(res.success),
        "message": str(res.message),
        "z": res.x,
        "weights": w,
        "y_hat": y_hat,
        "rss": rss,
        "reg_norm": reg_norm,
        "k_eff": int(k_eff),
        "AIC": float(sm_aic(llf, nobs, k_eff)),
        "BIC": float(sm_bic(llf, nobs, k_eff)),
        "rel_rmse": float(np.sqrt(np.mean(np.square(resid / np.maximum(y, 1.0e-12))))),
    }


def fit_lambda_path(y, r, lambda_grid):
    k_mat = build_convolution_matrix(r)
    d2 = second_difference_matrix(len(y))
    z_prev = None
    fits = []
    for lam in lambda_grid:
        fit = fit_deconvolution_for_lambda(y, k_mat, d2, float(lam), z0=z_prev)
        z_prev = fit["z"]
        fits.append(fit)
    idx = int(np.argmin([fit["AIC"] for fit in fits]))
    return fits, fits[idx], k_mat, d2


def fit_warp_only(t, y, r):
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)

    def objective(scale):
        pred = log_interp(t, r, float(scale) * t)
        mask = np.isfinite(pred) & np.isfinite(y)
        if np.count_nonzero(mask) < 10:
            return 1.0e9
        resid = (pred[mask] - y[mask]) / np.maximum(y[mask], 1.0e-12)
        return float(np.mean(np.square(resid)))

    res = minimize_scalar(objective, bounds=(0.2, 5.0), method="bounded", options={"xatol": 1.0e-4})
    scale = float(res.x)
    pred = log_interp(t, r, scale * t)
    mask = np.isfinite(pred) & np.isfinite(y)
    resid = (pred[mask] - y[mask]) / np.maximum(y[mask], 1.0e-12)
    rss = float(np.dot(resid, resid))
    nobs = int(np.count_nonzero(mask))
    sigma2 = max(rss / max(nobs, 1), 1.0e-18)
    llf = -0.5 * nobs * (math.log(2.0 * math.pi) + math.log(sigma2) + 1.0)
    return {
        "scale": scale,
        "rel_rmse": float(np.sqrt(np.mean(np.square(resid)))) if resid.size else np.nan,
        "AIC": float(sm_aic(llf, nobs, 1)) if resid.size else np.nan,
        "BIC": float(sm_bic(llf, nobs, 1)) if resid.size else np.nan,
        "y_hat": pred,
    }


def bootstrap_deconvolution(vw, t, y, r, lam, nboot, n_jobs, seed):
    k_mat = build_convolution_matrix(r)
    d2 = second_difference_matrix(len(y))
    best = fit_deconvolution_for_lambda(y, k_mat, d2, lam)
    resid = y - best["y_hat"]
    z0 = best["z"]

    def one(i):
        rng = np.random.default_rng(seed + 1000 * int(round(vw * 10)) + i)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=len(resid))
        y_boot = np.maximum(best["y_hat"] + resid * signs, 1.0e-12)
        fit = fit_deconvolution_for_lambda(y_boot, k_mat, d2, lam, z0=z0)
        return fit["weights"], fit["rel_rmse"]

    jobs = (delayed(one)(i) for i in range(nboot))
    results = Parallel(n_jobs=n_jobs, prefer="threads")(jobs)
    weights = np.vstack([item[0] for item in results]) if results else np.zeros((0, len(y)))
    rmses = np.array([item[1] for item in results], dtype=np.float64) if results else np.array([], dtype=np.float64)
    return weights, rmses


def plot_kernel(vw, tau, pdf, pdf_band, outpath: Path):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(tau, pdf, lw=2.0, label="best-fit $p(\\tau)$")
    if pdf_band is not None:
        ax.fill_between(tau, pdf_band["p16"], pdf_band["p84"], alpha=0.25, label="68% bootstrap")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$p(\tau)$")
    ax.set_title(rf"$v_w={vw:.1f}$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_reconstruction(vw, t, y, y_hat, y_warp, outpath: Path):
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]})
    axes[0].plot(t, y, "o", ms=3.4, label="lattice pooled")
    axes[0].plot(t, y_hat, lw=2.0, label="deconvolution reconstruction")
    if y_warp is not None:
        axes[0].plot(t, y_warp, lw=1.6, linestyle="--", label="warp-only")
    axes[0].set_ylabel(r"$\xi$")
    axes[0].set_title(rf"$v_w={vw:.1f}$ pooled reconstruction")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    resid = (y_hat - y) / np.maximum(y, 1.0e-12)
    axes[1].plot(t, resid, lw=1.5)
    axes[1].axhline(0.0, color="black", lw=1.0, alpha=0.4)
    axes[1].set_xlabel(r"$t_p$")
    axes[1].set_ylabel(r"$(\hat\xi-\xi)/\xi$")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_lcurve(vw, fits, outpath: Path):
    lambdas = np.array([fit["lambda"] for fit in fits], dtype=np.float64)
    rss = np.array([fit["rss"] for fit in fits], dtype=np.float64)
    reg = np.array([fit["reg_norm"] for fit in fits], dtype=np.float64)
    aics = np.array([fit["AIC"] for fit in fits], dtype=np.float64)
    best_idx = int(np.argmin(aics))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].plot(rss, reg, "o-")
    axes[0].scatter(rss[best_idx], reg[best_idx], color="red", zorder=5, label="AIC minimum")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\mathrm{RSS}$")
    axes[0].set_ylabel(r"$\|D p\|^2$")
    axes[0].set_title(rf"$v_w={vw:.1f}$ L-curve")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(lambdas, aics, "o-", label="AIC")
    axes[1].plot(lambdas, [fit["BIC"] for fit in fits], "s--", label="BIC")
    axes[1].axvline(lambdas[best_idx], color="red", lw=1.2, alpha=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel("information criterion")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_rmse_comparison(summary_rows, outpath: Path):
    df = pd.DataFrame(summary_rows).sort_values("v_w")
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    x = np.arange(len(df), dtype=np.float64)
    width = 0.36
    ax.bar(x - width / 2.0, df["deconv_rel_rmse"], width=width, label="deconvolution")
    ax.bar(x + width / 2.0, df["warp_rel_rmse"], width=width, label="warp-only")
    ax.set_xticks(x, [f"{vw:.1f}" for vw in df["v_w"]])
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel("rel-RMSE")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = (ROOT / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    vw_values = parse_vw_list(args.vw_list)
    lambda_grid = resolve_lambda_grid(args.lambda_grid)

    print("[1/6] loading lattice and ODE data")
    lattice_df = load_lattice_dataframe(args, outdir)
    theta_list = resolve_theta_list(np.sort(lattice_df["theta"].unique()), args.theta_list)
    lattice_df = filter_dataframe(lattice_df, vw_values, theta_list, args.h_values, args.tp_min, args.tp_max)
    ode_df = load_ode_dataframe(Path(args.xi_ode).resolve(), args.h_values, vw_values)
    ode_df = filter_dataframe(ode_df, vw_values, theta_list, args.h_values, args.tp_min, args.tp_max)

    save_json(
        outdir / "input_summary.json",
        {
            "vw_values": vw_values,
            "theta_list": theta_list,
            "h_values": args.h_values,
            "lambda_grid": lambda_grid,
            "n_lattice_rows": int(len(lattice_df)),
            "n_ode_rows": int(len(ode_df)),
        },
    )

    print("[2/6] pooling curves and fitting lambda paths")
    results = {}
    summary_rows = []
    for vw in tqdm(vw_values, desc="deconv"):
        latt_v = lattice_df[np.isclose(lattice_df["v_w"], float(vw), atol=1.0e-12, rtol=0.0)].copy()
        ode_v = ode_df[np.isclose(ode_df["v_w"], float(vw), atol=1.0e-12, rtol=0.0)].copy()
        pooled = finalize_pooled_curves(float(vw), latt_v, ode_v, theta_list, args.h_values, args.grid_size)
        fits, best, _, _ = fit_lambda_path(pooled["y"], pooled["r"], lambda_grid)
        warp = fit_warp_only(pooled["t"], pooled["y"], pooled["r"])
        dt = float(np.median(np.diff(pooled["t"])))
        tau = np.arange(len(best["weights"]), dtype=np.float64) * dt
        pdf = weights_to_pdf(best["weights"], dt)

        print(f"[vw={vw:.1f}] best lambda={best['lambda']:.3e}, rel_rmse={best['rel_rmse']:.4e}, warp={warp['rel_rmse']:.4e}")
        plot_kernel(float(vw), tau, pdf, None, outdir / f"p_vw{vw:.1f}.png")
        plot_reconstruction(float(vw), pooled["t"], pooled["y"], best["y_hat"], warp["y_hat"], outdir / f"reconstruction_vw{vw:.1f}.png")
        plot_lcurve(float(vw), fits, outdir / f"lambda_diagnostic_vw{vw:.1f}.png")

        pd.DataFrame(
            {
                "tau": tau,
                "weight": best["weights"],
                "pdf": pdf,
            }
        ).to_csv(outdir / f"p_vw{vw:.1f}.csv", index=False)
        pd.DataFrame(
            {
                "t": pooled["t"],
                "xi_lattice_pooled": pooled["y"],
                "xi_ode_pooled": pooled["r"],
                "xi_hat_deconv": best["y_hat"],
                "xi_hat_warp": warp["y_hat"],
                "coverage_lattice": pooled["coverage_latt"],
                "coverage_ode": pooled["coverage_ode"],
            }
        ).to_csv(outdir / f"pooled_reconstruction_vw{vw:.1f}.csv", index=False)

        results[f"{vw:.1f}"] = {
            "best_lambda": float(best["lambda"]),
            "rel_rmse": float(best["rel_rmse"]),
            "warp_rel_rmse": float(warp["rel_rmse"]),
            "AIC": float(best["AIC"]),
            "BIC": float(best["BIC"]),
            "warp_AIC": float(warp["AIC"]),
            "warp_BIC": float(warp["BIC"]),
            "warp_scale": float(warp["scale"]),
            "k_eff": int(best["k_eff"]),
            "lambda_path": [
                {
                    "lambda": float(item["lambda"]),
                    "rel_rmse": float(item["rel_rmse"]),
                    "rss": float(item["rss"]),
                    "reg_norm": float(item["reg_norm"]),
                    "AIC": float(item["AIC"]),
                    "BIC": float(item["BIC"]),
                    "k_eff": int(item["k_eff"]),
                }
                for item in fits
            ],
            "t_grid": pooled["t"],
            "weights": best["weights"],
        }
        summary_rows.append(
            {
                "v_w": float(vw),
                "deconv_rel_rmse": float(best["rel_rmse"]),
                "warp_rel_rmse": float(warp["rel_rmse"]),
                "best_lambda": float(best["lambda"]),
                "warp_scale": float(warp["scale"]),
            }
        )

    print("[3/6] bootstrap uncertainty estimation")
    for vw in tqdm(vw_values, desc="bootstrap"):
        item = results[f"{vw:.1f}"]
        pooled_path = outdir / f"pooled_reconstruction_vw{vw:.1f}.csv"
        pooled_df = pd.read_csv(pooled_path)
        weights_boot, rmses_boot = bootstrap_deconvolution(
            float(vw),
            pooled_df["t"].to_numpy(dtype=np.float64),
            pooled_df["xi_lattice_pooled"].to_numpy(dtype=np.float64),
            pooled_df["xi_ode_pooled"].to_numpy(dtype=np.float64),
            float(item["best_lambda"]),
            int(args.nboot),
            int(args.n_jobs),
            int(args.seed),
        )
        dt = float(np.median(np.diff(pooled_df["t"].to_numpy(dtype=np.float64))))
        pdf_boot = weights_boot / max(dt, 1.0e-18) if weights_boot.size else np.zeros((0, len(pooled_df)))
        band = None
        if pdf_boot.size:
            band = {
                "p16": np.percentile(pdf_boot, 16, axis=0),
                "p50": np.percentile(pdf_boot, 50, axis=0),
                "p84": np.percentile(pdf_boot, 84, axis=0),
            }
            plot_kernel(float(vw), pooled_df["t"].to_numpy(dtype=np.float64) - pooled_df["t"].iloc[0], item["weights"] / max(dt, 1.0e-18), band, outdir / f"p_vw{vw:.1f}.png")
        item["bootstrap"] = {
            "rel_rmse_p16": float(np.percentile(rmses_boot, 16)) if rmses_boot.size else np.nan,
            "rel_rmse_p50": float(np.percentile(rmses_boot, 50)) if rmses_boot.size else np.nan,
            "rel_rmse_p84": float(np.percentile(rmses_boot, 84)) if rmses_boot.size else np.nan,
            "pdf_p16": band["p16"] if band is not None else None,
            "pdf_p50": band["p50"] if band is not None else None,
            "pdf_p84": band["p84"] if band is not None else None,
        }
        if band is not None:
            pd.DataFrame(
                {
                    "tau": np.arange(len(item["weights"]), dtype=np.float64) * dt,
                    "pdf_best": item["weights"] / max(dt, 1.0e-18),
                    "pdf_p16": band["p16"],
                    "pdf_p50": band["p50"],
                    "pdf_p84": band["p84"],
                }
            ).to_csv(outdir / f"p_vw{vw:.1f}.csv", index=False)

    print("[4/6] saving aggregate artifacts")
    with open(outdir / "bootstrap_results.pkl", "wb") as handle:
        pickle.dump(to_native(results), handle)

    summary_df = pd.DataFrame(summary_rows).sort_values("v_w")
    summary_df.to_csv(outdir / "summary_table.csv", index=False)
    plot_rmse_comparison(summary_rows, outdir / "rmse_comparison.png")

    summary = {
        "status": "ok",
        "xi_latt_source": str(Path(args.xi_latt).resolve()) if Path(args.xi_latt).exists() else "autodiscovered_raw_vw_runs",
        "xi_ode_source": str(Path(args.xi_ode).resolve()),
        "vw_values": vw_values,
        "theta_list": theta_list,
        "h_values": args.h_values,
        "lambda_grid": lambda_grid,
        "per_vw": results,
    }
    save_json(outdir / "summary.json", summary)

    final_summary = {
        "status": "ok",
        "best_by_vw": {
            key: {
                "best_lambda": value["best_lambda"],
                "deconv_rel_rmse": value["rel_rmse"],
                "warp_rel_rmse": value["warp_rel_rmse"],
                "delta_AIC_deconv_minus_warp": value["AIC"] - value["warp_AIC"],
                "delta_BIC_deconv_minus_warp": value["BIC"] - value["warp_BIC"],
                "warp_scale": value["warp_scale"],
            }
            for key, value in results.items()
        },
        "median_rel_rmse_deconv": float(np.median([value["rel_rmse"] for value in results.values()])),
        "median_rel_rmse_warp": float(np.median([value["warp_rel_rmse"] for value in results.values()])),
    }
    save_json(outdir / "final_summary.json", final_summary)
    print(json.dumps(to_native(final_summary), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error_exit((ROOT / OUTDIR_DEFAULT).resolve(), exc)
        sys.exit(1)
