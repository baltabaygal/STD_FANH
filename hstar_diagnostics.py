#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ode.analysis.run_phase1_phase2_fanh_model import (
    contour_tceff_r,
    compute_fanh_and_save,
    estimate_finf_tail,
    estimate_smalltp_exponents,
    evaluate_model_A,
    fit_model_A,
    fit_model_B,
    fit_tail_amplitude,
    load_lattice_scan,
    load_ode_nopt,
    load_ode_scan,
    merge_f0,
    model_A_transient,
    model_B_tpeff,
    nearest_theta,
    profile_tceff,
    reconstruct_f0_table,
    rel_rmse,
    save_json,
)


EXPECTED = {
    "rho_noPT_data.txt": [
        ROOT / "rho_noPT_data.txt",
        ROOT / "lattice_data/data/rho_noPT_data.txt",
    ],
    "xi_lattice_scan_H1p5.txt": [
        ROOT / "xi_lattice_scan_H1p5.txt",
        ROOT / "lattice_data/data/energy_ratio_by_theta_data_v9.txt",
    ],
    "xi_lattice_scan_H2p0.txt": [
        ROOT / "xi_lattice_scan_H2p0.txt",
        ROOT / "lattice_data/data/energy_ratio_by_theta_data_v9.txt",
    ],
    "xi_ode_scan.txt": [
        ROOT / "xi_ode_scan.txt",
        ROOT / "ode/xi_DM_ODE_results.txt",
    ],
    "ode_nopt_reference.txt": [
        ROOT / "ode/analysis/results/noPT_reference_dm_tp_fitready_H1p000.txt",
    ],
}

H0_TREND = 1.5
LATTICE_H = [1.5, 2.0]
ODE_H = [0.5, 1.0, 1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(
        description="Run H_* diagnostics and global fits for the xi/fanh model."
    )
    p.add_argument("--rho-nopt", type=str, default="")
    p.add_argument("--xi-h1p5", type=str, default="")
    p.add_argument("--xi-h2p0", type=str, default="")
    p.add_argument("--xi-ode", type=str, default="")
    p.add_argument("--ode-nopt", type=str, default="")
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--bootstrap-jobs", type=int, default=6)
    p.add_argument("--bootstrap-seed", type=int, default=12345)
    p.add_argument("--profile-grid-n", type=int, default=80)
    p.add_argument("--huber-fscale", type=float, default=0.05)
    p.add_argument("--outdir", type=str, default="results_hstar")
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def error_json(message: str):
    payload = {"status": "error", "message": message}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def resolve_input(user_value: str, logical_name: str, required: bool = True) -> Path | None:
    if user_value:
        p = Path(user_value).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Missing required input: {p}")
    for candidate in EXPECTED[logical_name]:
        if candidate.exists():
            return candidate.resolve()
    if required:
        raise FileNotFoundError(f"Missing required input for {logical_name}. Tried: {EXPECTED[logical_name]}")
    return None


def dataset_tag(hstar: float) -> str:
    return f"H{str(hstar).replace('.', 'p')}"


def robust_linear_loglog(x, y, fscale):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def resid(par):
        return par[0] + par[1] * x - y

    res = least_squares(
        resid,
        x0=np.array([float(np.median(y)), 1.5], dtype=np.float64),
        loss="soft_l1",
        f_scale=fscale,
    )
    rss = float(np.sum(np.square(res.fun)))
    dof = max(len(y) - 2, 1)
    try:
        cov = (rss / dof) * np.linalg.inv(res.jac.T @ res.jac)
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.nan, dtype=np.float64)
    return {
        "intercept": float(res.x[0]),
        "slope": float(res.x[1]),
        "intercept_err": float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else np.nan,
        "slope_err": float(np.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) else np.nan,
        "cov": cov,
        "success": bool(res.success),
    }


def fit_tail_plots(df, tail_df, outdir, prefix, dpi, tail_frac):
    theta_vals = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    for theta in theta_vals:
        sub = df[np.isclose(df["theta"], theta, rtol=0.0, atol=5.0e-4)].copy().sort_values("tp")
        fit = fit_tail_amplitude(sub, huber_fscale=0.05, tail_frac=tail_frac)
        c = fit["C"]
        ntail = fit["tail_n"]
        tail = sub.tail(ntail)
        fig, ax = plt.subplots(figsize=(5.8, 4.4))
        ax.plot(sub["tp"], sub["xi"], "ko", ms=3.4, label="data")
        ax.plot(tail["tp"], tail["xi"], "o", color="tab:blue", ms=4.0, label="tail points")
        ax.plot(sub["tp"], c * np.power(sub["tp"], 1.5), color="tab:red", lw=1.8, label=r"tail fit $C t_p^{3/2}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$\xi$")
        ax.set_title(rf"Tail fit, $\theta={theta:.3f}$, {prefix}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"tail_fit_theta_{theta:.3f}_{prefix}.png", dpi=dpi)
        plt.close(fig)


def inversion_single_theta(group, finf_tail, fscale):
    group = group.sort_values("tp").copy()
    f0 = float(group["F0"].iloc[0])
    tp = group["tp"].to_numpy(dtype=np.float64)
    xi = group["xi"].to_numpy(dtype=np.float64)
    T = np.power(tp, 1.5) * finf_tail / (f0 * f0)
    D = xi - T
    mask = (D > 0.05) & (D < 0.95)
    work = group.loc[mask].copy()
    if len(work) < 3:
        return None
    D = work["xi"].to_numpy(dtype=np.float64) - np.power(work["tp"].to_numpy(dtype=np.float64), 1.5) * finf_tail / (f0 * f0)
    Y = 1.0 / D - 1.0
    valid = Y > 0.0
    work = work.loc[valid].copy()
    Y = Y[valid]
    if len(work) < 3:
        return None
    fit = robust_linear_loglog(np.log(work["tp"].to_numpy(dtype=np.float64)), np.log(Y), fscale)
    r = fit["slope"]
    a = fit["intercept"]
    s_over_tc = float(np.exp(a / r))
    cov = fit["cov"]
    if np.all(np.isfinite(cov)) and abs(r) > 1.0e-12:
        da = 1.0 / r
        dr = -a / (r * r)
        var = (s_over_tc ** 2) * (da * da * cov[0, 0] + dr * dr * cov[1, 1] + 2.0 * da * dr * cov[0, 1])
        s_over_tc_err = float(np.sqrt(max(var, 0.0)))
    else:
        s_over_tc_err = np.nan
    return {
        "theta": float(group["theta"].iloc[0]),
        "r": float(r),
        "r_err": float(fit["slope_err"]),
        "s_over_tc": s_over_tc,
        "s_over_tc_err": s_over_tc_err,
        "n_points_used": int(len(work)),
        "tp_fit": work["tp"].to_numpy(dtype=np.float64),
        "logY_fit": np.log(Y),
        "intercept": float(a),
    }


def fit_inversion_global(inversion_rows, fscale):
    data = [row for row in inversion_rows if row is not None and row["n_points_used"] >= 3]
    if not data:
        return {"success": False, "message": "No valid inversion rows"}
    theta_vals = np.array([row["theta"] for row in data], dtype=np.float64)
    uniq = np.array(sorted(np.unique(theta_vals)), dtype=np.float64)
    x = np.concatenate([np.log(row["tp_fit"]) for row in data])
    y = np.concatenate([row["logY_fit"] for row in data])
    theta_index = np.concatenate([np.full(len(row["tp_fit"]), nearest_theta(uniq, row["theta"]), dtype=np.int64) for row in data])

    def resid(par):
        intercepts = par[:-1]
        r = par[-1]
        return intercepts[theta_index] + r * x - y

    x0 = np.zeros(len(uniq) + 1, dtype=np.float64)
    for i, th in enumerate(uniq):
        row = next(row for row in data if np.isclose(row["theta"], th, atol=5.0e-4))
        x0[i] = row["intercept"]
    x0[-1] = float(np.nanmedian([row["r"] for row in data]))
    res = least_squares(resid, x0=x0, loss="soft_l1", f_scale=fscale)
    rss = float(np.sum(np.square(res.fun)))
    dof = max(len(y) - len(x0), 1)
    try:
        cov = (rss / dof) * np.linalg.inv(res.jac.T @ res.jac)
    except np.linalg.LinAlgError:
        cov = np.full((len(x0), len(x0)), np.nan, dtype=np.float64)
    out = {
        "success": bool(res.success),
        "r_global": float(res.x[-1]),
        "r_global_err": float(np.sqrt(cov[-1, -1])) if np.isfinite(cov[-1, -1]) else np.nan,
        "theta_intercepts": {f"{float(th):.10f}": float(res.x[i]) for i, th in enumerate(uniq)},
        "rss_frac": rss,
        "n_points": int(len(y)),
    }
    return out


def make_logY_plots(rows, prefix, outdir, dpi):
    for row in rows:
        if row is None:
            continue
        x = np.log(row["tp_fit"])
        y = row["logY_fit"]
        yfit = row["intercept"] + row["r"] * x
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.plot(x, y, "ko", ms=3.6, label="data")
        ax.plot(x, yfit, color="tab:red", lw=1.8, label=rf"fit, $r={row['r']:.3f}$")
        ax.set_xlabel(r"$\log t_p$")
        ax.set_ylabel(r"$\log Y$")
        ax.set_title(rf"$\log Y$ vs $\log t_p$, $\theta={row['theta']:.3f}$, {prefix}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"logY_vs_logtp_theta_{row['theta']:.3f}_{prefix}.png", dpi=dpi)
        plt.close(fig)


def smalltp_plot(df, prefix, outdir, dpi):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.errorbar(df["theta"], df["p"], yerr=df["p_err"], marker="o", ms=4.5, lw=1.2)
    ax.axhline(1.5, color="black", lw=1.0, ls="--", alpha=0.6)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("small-$t_p$ exponent")
    ax.set_title(f"Small-tp exponent vs theta, {prefix}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f"p_vs_theta_{prefix}.png", dpi=dpi)
    plt.close(fig)


def tceff_from_halfway(group, finf_tail, row):
    group = group.sort_values("tp").copy()
    f0 = float(group["F0"].iloc[0])
    tp = group["tp"].to_numpy(dtype=np.float64)
    xi = group["xi"].to_numpy(dtype=np.float64)
    T = np.power(tp, 1.5) * finf_tail / (f0 * f0)
    D = xi - T
    center = np.nan
    center_err = np.nan
    if np.any(D <= 0.5) and np.any(D >= 0.5):
        order = np.argsort(D)
        D_sorted = D[order]
        tp_sorted = tp[order]
        if np.all(np.diff(D_sorted) >= 0.0):
            center = float(np.interp(0.5, D_sorted, tp_sorted))
            idx = int(np.argmin(np.abs(tp - center)))
            if 0 < idx < len(tp) - 1:
                center_err = 0.5 * abs(tp[idx + 1] - tp[idx - 1])
            else:
                center_err = abs(tp[min(idx + 1, len(tp) - 1)] - tp[max(idx - 1, 0)])
    t_c_eff_inv = float(1.0 / row["s_over_tc"]) if np.isfinite(row["s_over_tc"]) and row["s_over_tc"] > 0 else np.nan
    return {
        "theta": float(group["theta"].iloc[0]),
        "tceff_halfway": center,
        "tceff_halfway_err": center_err,
        "tceff_from_inversion": t_c_eff_inv,
        "delta_halfway_minus_inversion": float(center - t_c_eff_inv) if np.isfinite(center) and np.isfinite(t_c_eff_inv) else np.nan,
    }


def slope_curvature_series(group):
    group = group.sort_values("tp").copy()
    x = np.log(group["tp"].to_numpy(dtype=np.float64))
    y = np.log(group["xi"].to_numpy(dtype=np.float64))
    n = len(x)
    if n < 7:
        slope = np.gradient(y, x)
        curv = np.gradient(slope, x)
    else:
        win = min(n if n % 2 == 1 else n - 1, 9)
        if win < 5:
            win = 5
        grid = np.linspace(x.min(), x.max(), n)
        y_uni = np.interp(grid, x, y)
        dx = float(np.mean(np.diff(grid)))
        slope_uni = savgol_filter(y_uni, window_length=win, polyorder=min(3, win - 2), deriv=1, delta=dx)
        curv_uni = savgol_filter(y_uni, window_length=win, polyorder=min(3, win - 2), deriv=2, delta=dx)
        slope = np.interp(x, grid, slope_uni)
        curv = np.interp(x, grid, curv_uni)
    idx = int(np.argmax(np.abs(curv)))
    return pd.DataFrame(
        {
            "theta": group["theta"].to_numpy(dtype=np.float64),
            "tp": group["tp"].to_numpy(dtype=np.float64),
            "xi": group["xi"].to_numpy(dtype=np.float64),
            "dlnxi_dln_tp": slope,
            "d2lnxi_dln_tp2": curv,
        }
    ), {
        "theta": float(group["theta"].iloc[0]),
        "tp_max_curvature": float(group["tp"].iloc[idx]),
        "max_abs_curvature": float(abs(curv[idx])),
    }


def make_slope_curvature_plots(series_df, prefix, outdir, dpi):
    theta = float(series_df["theta"].iloc[0])
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(series_df["tp"], series_df["dlnxi_dln_tp"], "o-", color="tab:blue", ms=3.2)
    ax.set_xscale("log")
    ax.set_xlabel(r"$t_p$")
    ax.set_ylabel(r"$d\ln\xi / d\ln t_p$")
    ax.set_title(rf"Slope, $\theta={theta:.3f}$, {prefix}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f"slopes_theta_{theta:.3f}_{prefix}.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(series_df["tp"], series_df["d2lnxi_dln_tp2"], "o-", color="tab:red", ms=3.2)
    ax.set_xscale("log")
    ax.set_xlabel(r"$t_p$")
    ax.set_ylabel(r"$d^2\ln\xi / d(\ln t_p)^2$")
    ax.set_title(rf"Curvature, $\theta={theta:.3f}$, {prefix}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f"curvature_theta_{theta:.3f}_{prefix}.png", dpi=dpi)
    plt.close(fig)


def build_priors(f0_table, tail_df, exp_df):
    priors = {"F_inf": {}, "r": None, "t_c_eff": {"center": 1.5, "sigma": 1.5}}
    for row in tail_df.itertuples(index=False):
        sigma = max(0.30 * row.F_inf_tail, row.err if np.isfinite(row.err) else 0.0, 1.0e-8)
        priors["F_inf"][float(row.theta)] = {"center": float(row.F_inf_tail), "sigma": float(sigma)}
    low = exp_df[np.isfinite(exp_df["p"]) & (exp_df["p"] < 1.4)]
    if len(low) >= max(3, len(exp_df) // 2):
        center = float(np.median(low["p"]))
        sigma = max(0.30 * center, float(np.nanmedian(low["p_err"])) if np.any(np.isfinite(low["p_err"])) else 0.5)
        priors["r"] = {"center": center, "sigma": sigma}
    return priors


def fit_dataset_global(df, tag, outdir, bootstrap_n, bootstrap_jobs, bootstrap_seed, huber_fscale, profile_grid_n, dpi):
    theta_values = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    priors = build_priors(None, pd.read_csv(outdir / f"Finf_tail_{tag}_table.csv"), pd.read_csv(outdir / f"smalltp_exponent_{tag}.csv"))
    fitA = fit_model_A(df, priors, huber_fscale, profile_grid_n // 2)
    fitB = fit_model_B(df, priors, huber_fscale, max(12, profile_grid_n // 3))

    def bootstrap_worker(seed):
        rng = np.random.default_rng(seed)
        boot_df = df.copy()
        resid = fitA["data_resid"]
        fitted = fitA["xi_fit"]
        xi_boot = fitted.copy()
        for theta in theta_values:
            mask = np.isclose(df["theta"].to_numpy(dtype=np.float64), float(theta), rtol=0.0, atol=5.0e-4)
            sample = rng.choice(resid[mask], size=np.sum(mask), replace=True)
            xi_boot[mask] = fitted[mask] * (1.0 + sample)
        boot_df["xi"] = np.maximum(xi_boot, 1.0e-8)
        rec = fit_model_A(boot_df, priors, huber_fscale, max(12, profile_grid_n // 3))
        return {
            "t_c_eff": rec["t_c_eff"],
            "r": rec["r"],
            "F_inf": rec["F_inf"].tolist(),
        }

    seeds = [int(bootstrap_seed + i) for i in range(bootstrap_n)]
    boot = Parallel(n_jobs=bootstrap_jobs)(delayed(bootstrap_worker)(seed) for seed in seeds)
    boot_summary = {
        "t_c_eff": {
            "p16": float(np.percentile([b["t_c_eff"] for b in boot], 16.0)),
            "p50": float(np.percentile([b["t_c_eff"] for b in boot], 50.0)),
            "p84": float(np.percentile([b["t_c_eff"] for b in boot], 84.0)),
        },
        "r": {
            "p16": float(np.percentile([b["r"] for b in boot], 16.0)),
            "p50": float(np.percentile([b["r"] for b in boot], 50.0)),
            "p84": float(np.percentile([b["r"] for b in boot], 84.0)),
        },
        "F_inf": {},
    }
    for i, th in enumerate(theta_values):
        vals = [b["F_inf"][i] for b in boot]
        boot_summary["F_inf"][f"{float(th):.10f}"] = {
            "p16": float(np.percentile(vals, 16.0)),
            "p50": float(np.percentile(vals, 50.0)),
            "p84": float(np.percentile(vals, 84.0)),
        }

    payloadA = {
        "dataset": tag,
        "t_c_eff": fitA["t_c_eff"],
        "t_c_eff_err": fitA["param_err"]["t_c_eff"],
        "r": fitA["r"],
        "r_err": fitA["param_err"]["r"],
        "rel_rmse": fitA["rel_rmse"],
        "rss_frac": fitA["rss_frac"],
        "success": fitA["success"],
        "covariance": fitA["cov"].tolist(),
        "F_inf": {f"{float(th):.10f}": {"F_inf": float(val)} for th, val in zip(fitA["theta_values"], fitA["F_inf"])},
    }
    save_json(outdir / f"global_fit_{tag}.json", payloadA)
    save_json(outdir / f"bootstrap_{tag}.json", boot_summary)

    payloadB = {
        "dataset": tag,
        "t_c_eff": fitB["t_c_eff"],
        "t_c_eff_err": fitB["param_err"]["t_c_eff"],
        "r": fitB["r"],
        "r_err": fitB["param_err"]["r"],
        "tau_p": fitB["tau_p"],
        "tau_p_err": fitB["param_err"]["tau_p"],
        "rel_rmse": fitB["rel_rmse"],
        "rss_frac": fitB["rss_frac"],
        "success": fitB["success"],
        "covariance": fitB["cov"].tolist(),
        "F_inf": {f"{float(th):.10f}": {"F_inf": float(val)} for th, val in zip(fitB["theta_values"], fitB["F_inf"])},
    }
    save_json(outdir / f"tau_fit_{tag}.json", payloadB)

    profile_df = profile_tceff(df, priors, huber_fscale, profile_grid_n)
    profile_df.to_csv(outdir / f"profile_tceff_{tag}.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(profile_df["t_c_eff"], profile_df["rel_rmse"], color="tab:blue", lw=1.8)
    ax.set_xlabel(r"$t_{c,\rm eff}$")
    ax.set_ylabel("rel-RMSE")
    ax.set_title(f"Profile t_c_eff, {tag}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f"profile_tceff_vs_relRMSE_{tag}.png", dpi=dpi)
    plt.close(fig)

    t_grid, r_grid, Z = contour_tceff_r(df, priors, max(30, profile_grid_n // 2))
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    levels = np.linspace(float(np.min(Z)), float(np.percentile(Z, 95.0)), 16)
    cs = ax.contourf(t_grid, r_grid, Z, levels=levels, cmap="viridis")
    ax.contour(t_grid, r_grid, Z, levels=levels[::2], colors="white", linewidths=0.5, alpha=0.55)
    ax.set_xlabel(r"$t_{c,\rm eff}$")
    ax.set_ylabel(r"$r$")
    ax.set_title(f"t_c_eff-r contour, {tag}")
    fig.colorbar(cs, ax=ax, pad=0.02, label="rel-RMSE")
    fig.tight_layout()
    fig.savefig(outdir / f"tceff_r_contour_{tag}.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    theta_vals = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    tp_vals = np.array(sorted(df["tp"].unique()), dtype=np.float64)
    grid = np.full((len(theta_vals), len(tp_vals)), np.nan, dtype=np.float64)
    resid_map = (df["xi"].to_numpy(dtype=np.float64) - fitA["xi_fit"]) / df["xi"].to_numpy(dtype=np.float64)
    for i, th in enumerate(theta_vals):
        for j, tp in enumerate(tp_vals):
            mask = np.isclose(df["theta"], th, atol=5.0e-4) & np.isclose(df["tp"], tp, atol=1.0e-12)
            if np.any(mask):
                grid[i, j] = float(np.median(resid_map[mask]))
    vmax = np.nanmax(np.abs(grid))
    mesh = ax.pcolormesh(tp_vals, np.arange(len(theta_vals) + 1), np.vstack([grid, grid[-1:]]), cmap="coolwarm", shading="auto", vmin=-vmax, vmax=vmax)
    ax.set_xscale("log")
    ax.set_xlabel(r"$t_p$")
    ax.set_ylabel(r"$\theta$")
    ax.set_yticks(np.arange(len(theta_vals)) + 0.5)
    ax.set_yticklabels([f"{th:.3f}" for th in theta_vals])
    ax.set_title(f"Residual heatmap, {tag}")
    fig.colorbar(mesh, ax=ax, pad=0.02, label=r"$(\xi_{\rm data}-\xi_{\rm model})/\xi_{\rm data}$")
    fig.tight_layout()
    fig.savefig(outdir / f"residual_heatmap_{tag}.png", dpi=dpi)
    plt.close(fig)

    for theta in theta_values:
        sub = df[np.isclose(df["theta"], theta, atol=5.0e-4)].copy().sort_values("tp")
        idx = nearest_theta(fitA["theta_values"], theta)
        transient = model_A_transient(sub["tp"].to_numpy(dtype=np.float64), fitA["t_c_eff"], fitA["r"])
        fanh_model = fitA["F_inf"][idx] / sub["F0"].to_numpy(dtype=np.float64) + sub["F0"].to_numpy(dtype=np.float64) * transient / np.power(sub["tp"].to_numpy(dtype=np.float64), 1.5)
        fig, ax = plt.subplots(figsize=(5.8, 4.4))
        ax.plot(sub["tp"], sub["fanh_data"], "ko", ms=3.6, label="data")
        ax.plot(sub["tp"], fanh_model, color="tab:red", lw=1.8, label="fit")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel("fanh")
        ax.set_title(rf"$\theta={theta:.3f}$, {tag}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"fanh_theta_{theta:.3f}_{tag}.png", dpi=dpi)
        plt.close(fig)

    n = int(len(df))
    kA = 2 + len(theta_values)
    kB = 3 + len(theta_values)
    aicA = n * math.log(max(fitA["rss_frac"], 1.0e-18) / n) + 2.0 * kA
    bicA = n * math.log(max(fitA["rss_frac"], 1.0e-18) / n) + kA * math.log(n)
    aicB = n * math.log(max(fitB["rss_frac"], 1.0e-18) / n) + 2.0 * kB
    bicB = n * math.log(max(fitB["rss_frac"], 1.0e-18) / n) + kB * math.log(n)
    tau_required = bool((fitB["rel_rmse"] < 0.95 * fitA["rel_rmse"]) and (aicB < aicA - 2.0) and (bicB < bicA - 2.0))
    model_comp = {
        "dataset": tag,
        "ModelA": {"t_c_eff": fitA["t_c_eff"], "r": fitA["r"], "rel_rmse": fitA["rel_rmse"], "AIC": aicA, "BIC": bicA},
        "ModelB": {"t_c_eff": fitB["t_c_eff"], "r": fitB["r"], "tau_p": fitB["tau_p"], "rel_rmse": fitB["rel_rmse"], "AIC": aicB, "BIC": bicB},
        "delta_rel_rmse": float(fitB["rel_rmse"] - fitA["rel_rmse"]),
        "delta_AIC": float(aicB - aicA),
        "delta_BIC": float(bicB - bicA),
        "tau_required": tau_required,
    }
    return fitA, fitB, boot_summary, model_comp


def powerlaw_gamma_fit(hvals, yvals, yerr=None):
    hvals = np.asarray(hvals, dtype=np.float64)
    yvals = np.asarray(yvals, dtype=np.float64)
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=np.float64)
    mask = np.isfinite(hvals) & np.isfinite(yvals) & (hvals > 0.0) & (yvals > 0.0)
    if yerr is not None:
        mask &= np.isfinite(yerr)
    hvals = hvals[mask]
    yvals = yvals[mask]
    if yerr is not None:
        yerr = yerr[mask]
    if len(hvals) < 2:
        return {
            "success": False,
            "message": "Insufficient positive finite points for power-law fit",
            "param0": np.nan,
            "gamma": np.nan,
            "rss": np.nan,
            "AIC_powerlaw": None,
            "BIC_powerlaw": None,
            "AIC_constant": None,
            "BIC_constant": None,
        }
    x = np.log(hvals / H0_TREND)
    y = np.log(np.maximum(yvals, 1.0e-18))
    if yerr is None or np.any(~np.isfinite(yerr)) or np.all(yerr <= 0):
        w = np.ones_like(y)
    else:
        w = 1.0 / np.square(np.maximum(yerr / np.maximum(yvals, 1.0e-18), 1.0e-12))
    A = np.vstack([np.ones_like(x), x]).T
    Aw = A * np.sqrt(w[:, None])
    yw = y * np.sqrt(w)
    try:
        coeff, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    except np.linalg.LinAlgError:
        return {
            "success": False,
            "message": "Linear least-squares failed to converge",
            "param0": np.nan,
            "gamma": np.nan,
            "rss": np.nan,
            "AIC_powerlaw": None,
            "BIC_powerlaw": None,
            "AIC_constant": None,
            "BIC_constant": None,
        }
    residual = y - (A @ coeff)
    rss = float(np.sum(np.square(residual)))
    if len(y) > 2:
        aic = len(y) * math.log(max(rss, 1.0e-18) / len(y)) + 2.0 * 2
        bic = len(y) * math.log(max(rss, 1.0e-18) / len(y)) + 2.0 * math.log(len(y))
        const = np.average(y, weights=w)
        rss0 = float(np.sum(np.square(y - const)))
        aic0 = len(y) * math.log(max(rss0, 1.0e-18) / len(y)) + 2.0
        bic0 = len(y) * math.log(max(rss0, 1.0e-18) / len(y)) + math.log(len(y))
    else:
        aic = bic = aic0 = bic0 = None
    return {
        "success": True,
        "param0": float(np.exp(coeff[0])),
        "gamma": float(coeff[1]),
        "rss": rss,
        "AIC_powerlaw": aic,
        "BIC_powerlaw": bic,
        "AIC_constant": aic0,
        "BIC_constant": bic0,
    }


def bootstrap_gamma(hvals, yvals, yerr, nboot, seed):
    hvals = np.asarray(hvals, dtype=np.float64)
    yvals = np.asarray(yvals, dtype=np.float64)
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=np.float64)
    mask = np.isfinite(hvals) & np.isfinite(yvals) & (hvals > 0.0) & (yvals > 0.0)
    if yerr is not None:
        mask &= np.isfinite(yerr)
    hvals = hvals[mask]
    yvals = yvals[mask]
    if yerr is not None:
        yerr = yerr[mask]
    if len(hvals) < 2:
        return {"gamma": np.nan, "gamma_ci95": [np.nan, np.nan], "p_value_two_sided": np.nan}
    rng = np.random.default_rng(seed)
    gammas = []
    for _ in range(nboot):
        draw = yvals.copy()
        if yerr is not None:
            draw += rng.normal(0.0, yerr)
        draw = np.maximum(draw, 1.0e-18)
        fit = powerlaw_gamma_fit(hvals, draw, yerr=None)
        if np.isfinite(fit["gamma"]):
            gammas.append(fit["gamma"])
    if not gammas:
        return {"gamma": np.nan, "gamma_ci95": [np.nan, np.nan], "p_value_two_sided": np.nan}
    gammas = np.asarray(gammas, dtype=np.float64)
    return {
        "gamma": float(np.median(gammas)),
        "gamma_ci95": [float(np.percentile(gammas, 2.5)), float(np.percentile(gammas, 97.5))],
        "p_value_two_sided": float(2.0 * min(np.mean(gammas <= 0.0), np.mean(gammas >= 0.0))),
    }


def summarize_hstar_trends(lattice_products, ode_products, global_fit_map, bootstrap_map, tau_comp_map, nboot, seed):
    summary = {"status": "ok", "lattice": {}, "ode": {}, "global_fits": {}, "tau_model_comparison": tau_comp_map}

    for tag, fit in global_fit_map.items():
        boot = bootstrap_map[tag]
        summary["global_fits"][tag] = {
            "t_c_eff": fit["t_c_eff"],
            "t_c_eff_ci68": [boot["t_c_eff"]["p16"], boot["t_c_eff"]["p84"]],
            "r": fit["r"],
            "r_ci68": [boot["r"]["p16"], boot["r"]["p84"]],
            "rel_rmse": fit["rel_rmse"],
        }

    # Lattice H trends with two H points.
    for theta in sorted(lattice_products["Finf"]["H1p5"]["theta"].unique()):
        hvals = np.array(LATTICE_H, dtype=np.float64)
        yvals = np.array(
            [
                float(lattice_products["Finf"]["H1p5"].loc[np.isclose(lattice_products["Finf"]["H1p5"]["theta"], theta), "F_inf_tail"].iloc[0]),
                float(lattice_products["Finf"]["H2p0"].loc[np.isclose(lattice_products["Finf"]["H2p0"]["theta"], theta), "F_inf_tail"].iloc[0]),
            ],
            dtype=np.float64,
        )
        yerr = np.array(
            [
                float(lattice_products["Finf"]["H1p5"].loc[np.isclose(lattice_products["Finf"]["H1p5"]["theta"], theta), "err"].iloc[0]),
                float(lattice_products["Finf"]["H2p0"].loc[np.isclose(lattice_products["Finf"]["H2p0"]["theta"], theta), "err"].iloc[0]),
            ],
            dtype=np.float64,
        )
        fit = powerlaw_gamma_fit(hvals, yvals, yerr)
        boot = bootstrap_gamma(hvals, yvals, yerr, nboot, seed + int(theta * 1e4))
        summary["lattice"].setdefault("F_inf_tail", {})[f"{float(theta):.10f}"] = {**fit, **boot}

    # ODE H trends with four H points if available.
    if ode_products:
        for param_name in ["F_inf_tail", "r_inversion", "tceff_halfway"]:
            summary["ode"][param_name] = {}
        for theta in sorted(ode_products["Finf"]["theta"].unique()):
            hvals = np.array(sorted(ode_products["Finf"]["Hstar"].unique()), dtype=np.float64)
            finf = []
            ferr = []
            rinv = []
            rerr = []
            tce = []
            terr = []
            for h in hvals:
                finf_row = ode_products["Finf"][(np.isclose(ode_products["Finf"]["Hstar"], h, atol=1.0e-12)) & (np.isclose(ode_products["Finf"]["theta"], theta, atol=5.0e-4))].iloc[0]
                inv_row = ode_products["Inversion"][(np.isclose(ode_products["Inversion"]["Hstar"], h, atol=1.0e-12)) & (np.isclose(ode_products["Inversion"]["theta"], theta, atol=5.0e-4))].iloc[0]
                tc_row = ode_products["Tceff"][(np.isclose(ode_products["Tceff"]["Hstar"], h, atol=1.0e-12)) & (np.isclose(ode_products["Tceff"]["theta"], theta, atol=5.0e-4))].iloc[0]
                finf.append(float(finf_row["F_inf_tail"]))
                ferr.append(float(finf_row["err"]) if np.isfinite(finf_row["err"]) else 0.0)
                rinv.append(float(inv_row["r"]))
                rerr.append(float(inv_row["r_err"]) if np.isfinite(inv_row["r_err"]) else 0.0)
                tce.append(float(tc_row["tceff_halfway"]))
                terr.append(float(tc_row["tceff_halfway_err"]) if np.isfinite(tc_row["tceff_halfway_err"]) else 0.0)
            summary["ode"]["F_inf_tail"][f"{float(theta):.10f}"] = {**powerlaw_gamma_fit(hvals, finf, ferr), **bootstrap_gamma(hvals, finf, ferr, nboot, seed + int(theta * 1e4) + 100)}
            summary["ode"]["r_inversion"][f"{float(theta):.10f}"] = {**powerlaw_gamma_fit(hvals, rinv, rerr), **bootstrap_gamma(hvals, rinv, rerr, nboot, seed + int(theta * 1e4) + 200)}
            summary["ode"]["tceff_halfway"][f"{float(theta):.10f}"] = {**powerlaw_gamma_fit(hvals, tce, terr), **bootstrap_gamma(hvals, tce, terr, nboot, seed + int(theta * 1e4) + 300)}

    return summary


def fit_summary_text(tag, fitA, bootA, tau_comp):
    lines = [
        f"{tag}: best-fit global r = {bootA['r']['p50']:.4f} [{bootA['r']['p16']:.4f}, {bootA['r']['p84']:.4f}], "
        f"best-fit t_c_eff = {bootA['t_c_eff']['p50']:.4f} [{bootA['t_c_eff']['p16']:.4f}, {bootA['t_c_eff']['p84']:.4f}], "
        f"rel-RMSE = {fitA['rel_rmse']:.4e}.",
        (
            f"Tau significantly improved fit: {'yes' if tau_comp['tau_required'] else 'no'}; "
            f"delta rel-RMSE = {tau_comp['delta_rel_rmse']:.4e}, "
            f"delta AIC = {tau_comp['delta_AIC']:.4f}, delta BIC = {tau_comp['delta_BIC']:.4f}."
        ),
        (
            "Recommendation: "
            + (
                "prefer the tau-extended model for this H* slice."
                if tau_comp["tau_required"]
                else "keep the no-tau model; added delay is not statistically justified."
            )
        ),
    ]
    return "\n".join(lines) + "\n"


def run_dataset_diagnostics(df, prefix, outdir, dpi, huber_fscale, is_ode=False):
    fanh_csv = outdir / f"fanh_{prefix}.csv"
    df = compute_fanh_and_save(df, fanh_csv)

    exp_csv = outdir / f"smalltp_exponent_{prefix}.csv"
    exp_df = estimate_smalltp_exponents(df, "ODE" if is_ode else prefix, exp_csv, huber_fscale, 5, 10)
    smalltp_plot(exp_df, prefix, outdir, dpi)

    finf_csv = outdir / f"Finf_tail_{prefix}_table.csv"
    tail_df = estimate_finf_tail(df, "ODE" if is_ode else prefix, finf_csv, huber_fscale, 0.10)
    tail_df.to_csv(outdir / f"Finf_tail_{prefix}.csv", index=False)
    if is_ode:
        for hstar, sub in df.groupby("Hstar", sort=True):
            fit_tail_plots(sub, tail_df[tail_df["Hstar"].eq(hstar)], outdir, f"{prefix}_H{str(hstar).replace('.', 'p')}", dpi, 0.10)
    else:
        fit_tail_plots(df, tail_df, outdir, prefix, dpi, 0.10)

    inversion_rows = []
    tceff_rows = []
    curvature_rows = []
    group_keys = ["Hstar", "theta"] if is_ode else ["theta"]
    tail_merge_cols = ["theta"] + (["Hstar"] if is_ode else [])
    df_inv = df.merge(tail_df[tail_merge_cols + ["F_inf_tail"]], on=tail_merge_cols, how="left")
    for keys, group in df_inv.groupby(group_keys, sort=True):
        row = inversion_single_theta(group, float(group["F_inf_tail"].iloc[0]), huber_fscale)
        if row is not None and is_ode:
            row["Hstar"] = float(group["Hstar"].iloc[0])
        inversion_rows.append(row)
        if row is not None:
            tc_row = tceff_from_halfway(group, float(group["F_inf_tail"].iloc[0]), row)
            if is_ode:
                tc_row["Hstar"] = float(group["Hstar"].iloc[0])
            tceff_rows.append(tc_row)
        series_df, curv = slope_curvature_series(group)
        if is_ode:
            curv["Hstar"] = float(group["Hstar"].iloc[0])
            save_name = f"{prefix}_H{str(group['Hstar'].iloc[0]).replace('.', 'p')}"
        else:
            save_name = prefix
        curvature_rows.append(curv)
        series_df.to_csv(outdir / f"slopes_theta_{float(group['theta'].iloc[0]):.3f}_{save_name}.csv", index=False)
        make_slope_curvature_plots(series_df, save_name, outdir, dpi)

    make_logY_plots(inversion_rows, prefix, outdir, dpi)
    inv_df = pd.DataFrame([row for row in inversion_rows if row is not None])
    inv_csv = outdir / f"inversion_{prefix}.csv"
    inv_df.to_csv(inv_csv, index=False)
    tc_df = pd.DataFrame(tceff_rows)
    tc_df.to_csv(outdir / f"tceff_theta_{prefix}.csv", index=False)
    save_json(outdir / f"curvature_centers_{prefix}.json", curvature_rows)
    if is_ode:
        global_json = {}
        for hstar, sub in inv_df.groupby("Hstar", sort=True):
            rows = []
            for _, row in sub.iterrows():
                rows.append(
                    {
                        "theta": float(row["theta"]),
                        "r": float(row["r"]),
                        "intercept": float(row["intercept"]),
                        "n_points_used": int(row["n_points_used"]),
                        "tp_fit": np.array([], dtype=np.float64),
                        "logY_fit": np.array([], dtype=np.float64),
                    }
                )
            global_json[f"H{str(hstar).replace('.', 'p')}"] = {"note": "per-H global inversion requires raw fit arrays; saved per-theta table only"}
        save_json(outdir / f"inversion_global_{prefix}.json", global_json)
    else:
        global_inv = fit_inversion_global([row for row in inversion_rows if row is not None], huber_fscale)
        save_json(outdir / f"inversion_global_{prefix}.json", global_inv)

    return {
        "df": df,
        "Exponent": exp_df,
        "Finf": tail_df,
        "Inversion": inv_df,
        "Tceff": tc_df,
        "Curvature": curvature_rows,
    }


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        rho_path = resolve_input(args.rho_nopt, "rho_noPT_data.txt")
        h15_path = resolve_input(args.xi_h1p5, "xi_lattice_scan_H1p5.txt")
        h20_path = resolve_input(args.xi_h2p0, "xi_lattice_scan_H2p0.txt")
        ode_path = resolve_input(args.xi_ode, "xi_ode_scan.txt", required=False)
        ode_nopt_path = resolve_input(args.ode_nopt, "ode_nopt_reference.txt")
    except FileNotFoundError as exc:
        save_json(outdir / "hstar_trends.json", error_json(str(exc)))
        return 1

    ode_nopt_df = load_ode_nopt(ode_nopt_path)
    f0_table, _ = reconstruct_f0_table(rho_path, ode_nopt_df, outdir, use_all_h=False)

    lat_h15 = merge_f0(load_lattice_scan(h15_path, 1.5, args.fixed_vw), f0_table)
    lat_h20 = merge_f0(load_lattice_scan(h20_path, 2.0, args.fixed_vw), f0_table)
    if lat_h15.empty or lat_h20.empty:
        msg = "Lattice xi scans could not be constructed for H*=1.5 and H*=2.0."
        save_json(outdir / "hstar_trends.json", error_json(msg))
        return 1

    datasets = {
        "H1p5": run_dataset_diagnostics(lat_h15, "H1p5", outdir, args.dpi, args.huber_fscale, is_ode=False),
        "H2p0": run_dataset_diagnostics(lat_h20, "H2p0", outdir, args.dpi, args.huber_fscale, is_ode=False),
    }

    ode_products = {}
    if ode_path is not None:
        ode_df = merge_f0(load_ode_scan(ode_path, args.fixed_vw), f0_table)
        if not ode_df.empty:
            ode_products = run_dataset_diagnostics(ode_df, "ODE", outdir, args.dpi, args.huber_fscale, is_ode=True)

    fitA_h15, fitB_h15, boot_h15, tau_h15 = fit_dataset_global(
        datasets["H1p5"]["df"], "H1p5", outdir, args.bootstrap, args.bootstrap_jobs, args.bootstrap_seed + 15, args.huber_fscale, args.profile_grid_n, args.dpi
    )
    fitA_h20, fitB_h20, boot_h20, tau_h20 = fit_dataset_global(
        datasets["H2p0"]["df"], "H2p0", outdir, args.bootstrap, args.bootstrap_jobs, args.bootstrap_seed + 20, args.huber_fscale, args.profile_grid_n, args.dpi
    )

    tau_model_comparison = {"H1p5": tau_h15, "H2p0": tau_h20}
    save_json(outdir / "tau_model_comparison.json", tau_model_comparison)

    (outdir / "fit_summary_H1p5.txt").write_text(fit_summary_text("H1p5", fitA_h15, boot_h15, tau_h15))
    (outdir / "fit_summary_H2p0.txt").write_text(fit_summary_text("H2p0", fitA_h20, boot_h20, tau_h20))

    hstar_summary = summarize_hstar_trends(
        {
            "Finf": {"H1p5": datasets["H1p5"]["Finf"], "H2p0": datasets["H2p0"]["Finf"]},
        },
        ode_products,
        {"H1p5": fitA_h15, "H2p0": fitA_h20},
        {"H1p5": boot_h15, "H2p0": boot_h20},
        tau_model_comparison,
        args.bootstrap,
        args.bootstrap_seed,
    )
    save_json(outdir / "hstar_trends.json", hstar_summary)
    print(json.dumps(hstar_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
