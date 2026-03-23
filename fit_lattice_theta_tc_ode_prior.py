#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

import collapse_and_fit_fanh_tosc as latmod


ROOT = Path(__file__).resolve().parent
T_OSC = latmod.T_OSC
BASELINE_RUN = ROOT / "results_tosc_lattice_vw0p9_H1p0H1p5H2p0" / "collapse_and_fit_fanh" / "global_fit.json"
ODE_ANCHOR_TABLE = ROOT / "results_ode_anchored_finf_vw0p9_H1p0H1p5H2p0" / "ode_anchor_table.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit clean lattice vw=0.9 data with per-theta t_c(theta), global r, and soft ODE-anchor penalty on F_inf(theta)."
    )
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--h-values", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0])
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--outdir", type=str, default="results_theta_tc_ode_prior_vw0p9_H1p0H1p5H2p0")
    return p.parse_args()


def load_collapsed_dataset(h_values: list[float]) -> tuple[pd.DataFrame, float]:
    with open(BASELINE_RUN) as f:
        baseline = json.load(f)
    beta = float(baseline["beta"])
    rho_path = latmod.resolve_first_existing(latmod.RHO_CANDIDATES, None)
    raw_lattice_path = latmod.resolve_first_existing(latmod.LATTICE_RAW_CANDIDATES, None)
    f0_table = latmod.load_f0_table(rho_path, h_values)
    lattice_df = latmod.load_lattice_data(raw_lattice_path, 0.9, h_values)
    lattice_df = latmod.merge_f0(lattice_df, f0_table)
    lattice_df = lattice_df[np.isfinite(lattice_df["F0"]) & (lattice_df["F0"] > 0.0)].copy()
    collapsed = latmod.compute_x(lattice_df, beta)
    return collapsed, beta


def load_baseline_payload() -> dict:
    with open(BASELINE_RUN) as f:
        return json.load(f)


def load_ode_anchor_table() -> pd.DataFrame:
    df = pd.read_csv(ODE_ANCHOR_TABLE).copy()
    df["theta"] = df["theta"].astype(float)
    return df


def h_alt(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    return np.log(np.e / np.maximum(1.0 - np.square(theta / np.pi), 1.0e-12))


def xi_model(df: pd.DataFrame, theta_values: np.ndarray, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = float(params[0])
    ntheta = len(theta_values)
    tc_theta = np.asarray(params[1 : 1 + ntheta], dtype=np.float64)
    finf = np.asarray(params[1 + ntheta : 1 + 2 * ntheta], dtype=np.float64)
    theta_index = np.array([latmod.nearest_theta(theta_values, th) for th in df["theta"].to_numpy(dtype=np.float64)], dtype=np.int64)
    x = df["x"].to_numpy(dtype=np.float64)
    f0 = df["F0"].to_numpy(dtype=np.float64)
    transient = 1.0 / (1.0 + np.power(x / np.maximum(tc_theta[theta_index], 1.0e-12), r))
    xi_fit = np.power(x / T_OSC, 1.5) * finf[theta_index] / np.maximum(f0 * f0, 1.0e-18) + transient
    return xi_fit, theta_index


def fit_lambda(df: pd.DataFrame, theta_values: np.ndarray, init_r: float, init_tc: float, init_finf: np.ndarray, finf_anchor: np.ndarray, lam: float) -> dict:
    ntheta = len(theta_values)
    x0 = np.concatenate([[init_r], np.full(ntheta, init_tc, dtype=np.float64), init_finf]).astype(np.float64)
    lower = np.concatenate([[0.1], np.full(ntheta, 1.0e-3, dtype=np.float64), np.full(ntheta, 1.0e-8, dtype=np.float64)]).astype(np.float64)
    upper = np.concatenate([[20.0], np.full(ntheta, 20.0, dtype=np.float64), np.full(ntheta, 1.0e4, dtype=np.float64)]).astype(np.float64)
    sqrt_lam = math.sqrt(max(lam, 0.0))

    def resid(par: np.ndarray) -> np.ndarray:
        xi_fit, _ = xi_model(df, theta_values, par)
        xi = df["xi"].to_numpy(dtype=np.float64)
        data_resid = (xi_fit - xi) / np.maximum(xi, 1.0e-12)
        if lam <= 0.0:
            return data_resid
        finf = np.asarray(par[1 + ntheta : 1 + 2 * ntheta], dtype=np.float64)
        pen = sqrt_lam * (finf - finf_anchor) / np.maximum(finf_anchor, 1.0e-18)
        return np.concatenate([data_resid, pen])

    res0 = least_squares(resid, x0=x0, bounds=(lower, upper), loss="soft_l1", f_scale=0.05, max_nfev=12000)
    res = least_squares(resid, x0=res0.x, bounds=(lower, upper), loss="linear", max_nfev=12000)
    xi_fit, theta_index = xi_model(df, theta_values, res.x)
    xi = df["xi"].to_numpy(dtype=np.float64)
    data_resid = (xi_fit - xi) / np.maximum(xi, 1.0e-12)
    data_rss = float(np.sum(np.square(data_resid)))
    n = int(len(df))
    k = int(len(res.x))
    dof = max(n - k, 1)
    try:
        cov = (data_rss / dof) * np.linalg.inv(res.jac[:n].T @ res.jac[:n])
    except np.linalg.LinAlgError:
        cov = np.full((len(res.x), len(res.x)), np.nan, dtype=np.float64)
    aic = float(n * math.log(max(data_rss, 1.0e-18) / n) + 2.0 * k)
    bic = float(n * math.log(max(data_rss, 1.0e-18) / n) + k * math.log(n))
    ntheta = len(theta_values)
    tc_theta = np.asarray(res.x[1 : 1 + ntheta], dtype=np.float64)
    finf = np.asarray(res.x[1 + ntheta : 1 + 2 * ntheta], dtype=np.float64)
    penalty_rel = (finf - finf_anchor) / np.maximum(finf_anchor, 1.0e-18)
    return {
        "success": bool(res.success),
        "message": res.message,
        "params": np.asarray(res.x, dtype=np.float64),
        "covariance": cov,
        "xi_fit": xi_fit,
        "theta_index": theta_index,
        "rel_rmse": float(latmod.rel_rmse(xi, xi_fit)),
        "rss_frac": data_rss,
        "AIC": aic,
        "BIC": bic,
        "dof": dof,
        "tc_theta": tc_theta,
        "finf": finf,
        "r": float(res.x[0]),
        "penalty_rms": float(np.sqrt(np.mean(np.square(penalty_rel)))),
        "penalty_max_abs": float(np.max(np.abs(penalty_rel))),
        "mean_abs_frac_resid": float(np.mean(np.abs((xi_fit - xi) / np.maximum(xi, 1.0e-12)))),
    }


def save_fit_payload(outdir: Path, lam: float, theta_values: np.ndarray, fit: dict) -> None:
    ntheta = len(theta_values)
    cov = np.asarray(fit["covariance"], dtype=np.float64)
    payload = {
        "lambda": float(lam),
        "success": bool(fit["success"]),
        "message": fit["message"],
        "r": float(fit["r"]),
        "r_err": float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else np.nan,
        "rel_rmse": float(fit["rel_rmse"]),
        "rss_frac": float(fit["rss_frac"]),
        "AIC": float(fit["AIC"]),
        "BIC": float(fit["BIC"]),
        "dof": int(fit["dof"]),
        "penalty_rms": float(fit["penalty_rms"]),
        "penalty_max_abs": float(fit["penalty_max_abs"]),
        "t_c_theta": {},
        "F_inf": {},
    }
    for i, theta in enumerate(theta_values):
        tc_err = float(np.sqrt(cov[1 + i, 1 + i])) if np.isfinite(cov[1 + i, 1 + i]) else np.nan
        finf_err = float(np.sqrt(cov[1 + ntheta + i, 1 + ntheta + i])) if np.isfinite(cov[1 + ntheta + i, 1 + ntheta + i]) else np.nan
        payload["t_c_theta"][f"{theta:.10f}"] = {"value": float(fit["tc_theta"][i]), "err": tc_err}
        payload["F_inf"][f"{theta:.10f}"] = {"value": float(fit["finf"][i]), "err": finf_err}
    with open(outdir / f"fit_lambda_{lam:g}.json", "w") as f:
        json.dump(payload, f, indent=2)


def plot_collapse_overlay(df: pd.DataFrame, fit: dict, outpath: Path, dpi: int) -> None:
    theta_values = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sorted(df["H"].unique()))))
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex=False, sharey=False)
    for ax, theta in zip(axes.flat, theta_values):
        sub = df[np.isclose(df["theta"], theta, atol=5.0e-4, rtol=0.0)].copy()
        for color, (h, hsub) in zip(colors, sub.groupby("H", sort=True)):
            ax.plot(hsub["x"], hsub["xi"], "o", ms=3.2, color=color, label=rf"$H={h:g}$")
            mask = np.isclose(sub["H"], h, atol=1.0e-12, rtol=0.0)
            hfit = fit["xi_fit"][sub.index.to_numpy()][mask]
            xfit = hsub["x"].to_numpy(dtype=np.float64)
            order = np.argsort(xfit)
            ax.plot(xfit[order], hfit[order], "-", lw=1.5, color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$x = t_p H^\beta$")
        ax.set_ylabel(r"$\xi$")
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_theta_parameters(theta_values: np.ndarray, finf_anchor: np.ndarray, baseline_finf: np.ndarray, results: dict[float, dict], outdir: Path, dpi: int) -> None:
    hvals = h_alt(theta_values)
    # t_c(theta)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
    for lam, fit in results.items():
        label = rf"$\lambda={lam:g}$"
        axes[0].plot(theta_values, fit["tc_theta"], "o-", ms=4, lw=1.3, label=label)
        axes[1].plot(hvals, fit["tc_theta"], "o-", ms=4, lw=1.3, label=label)
    axes[0].set_xlabel(r"$\theta_0$")
    axes[0].set_ylabel(r"$t_c(\theta_0)$")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel(r"$h(\theta_0)=\log\!\left(e/[1-(\theta/\pi)^2]\right)$")
    axes[1].set_ylabel(r"$t_c(\theta_0)$")
    axes[1].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(outdir / "tc_theta_scan.png", dpi=dpi)
    plt.close(fig)

    # F_inf(theta)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
    axes[0].plot(theta_values, finf_anchor, "k--", lw=1.5, label="ODE anchor")
    axes[0].plot(theta_values, baseline_finf, "k:", lw=1.5, label="old free fit")
    axes[1].plot(hvals, finf_anchor, "k--", lw=1.5, label="ODE anchor")
    axes[1].plot(hvals, baseline_finf, "k:", lw=1.5, label="old free fit")
    for lam, fit in results.items():
        label = rf"$\lambda={lam:g}$"
        axes[0].plot(theta_values, fit["finf"], "o-", ms=4, lw=1.3, label=label)
        axes[1].plot(hvals, fit["finf"], "o-", ms=4, lw=1.3, label=label)
    axes[0].set_xlabel(r"$\theta_0$")
    axes[0].set_ylabel(r"$F_\infty(\theta_0)$")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel(r"$h(\theta_0)=\log\!\left(e/[1-(\theta/\pi)^2]\right)$")
    axes[1].set_ylabel(r"$F_\infty(\theta_0)$")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(outdir / "Finf_theta_scan.png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df, beta = load_collapsed_dataset([float(h) for h in args.h_values])
    theta_values = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    baseline = load_baseline_payload()
    baseline_finf = np.array([float(baseline["F_inf"][f"{theta:.10f}"]["value"]) for theta in theta_values], dtype=np.float64)
    baseline_tc = float(baseline["t_c"])
    baseline_r = float(baseline["r"])

    anchor_df = load_ode_anchor_table()
    anchor_theta = anchor_df["theta"].to_numpy(dtype=np.float64)
    finf_anchor = np.array(
        [float(anchor_df.iloc[latmod.nearest_theta(anchor_theta, theta)]["F_inf_ode_anchor_lattice_units"]) for theta in theta_values],
        dtype=np.float64,
    )

    summary_rows = []
    fit_results: dict[float, dict] = {}
    init_r = baseline_r
    init_tc = baseline_tc
    init_finf = baseline_finf.copy()

    for lam in args.lambdas:
        fit = fit_lambda(df, theta_values, init_r, init_tc, init_finf, finf_anchor, float(lam))
        fit_results[float(lam)] = fit
        save_fit_payload(outdir, float(lam), theta_values, fit)
        label = f"lambda_{lam:g}".replace(".", "p")
        plot_collapse_overlay(df, fit, outdir / f"collapse_overlay_{label}.png", args.dpi)

        tc_spearman = spearmanr(theta_values, fit["tc_theta"]).statistic if len(theta_values) >= 2 else np.nan
        h_spearman = spearmanr(h_alt(theta_values), fit["tc_theta"]).statistic if len(theta_values) >= 2 else np.nan
        summary_rows.append(
            {
                "lambda": float(lam),
                "r": float(fit["r"]),
                "rel_rmse": float(fit["rel_rmse"]),
                "AIC": float(fit["AIC"]),
                "BIC": float(fit["BIC"]),
                "penalty_rms": float(fit["penalty_rms"]),
                "penalty_max_abs": float(fit["penalty_max_abs"]),
                "tc_theta_min": float(np.min(fit["tc_theta"])),
                "tc_theta_max": float(np.max(fit["tc_theta"])),
                "tc_theta_spearman_vs_theta": float(tc_spearman) if np.isfinite(tc_spearman) else np.nan,
                "tc_theta_spearman_vs_h": float(h_spearman) if np.isfinite(h_spearman) else np.nan,
            }
        )

        init_r = float(fit["r"])
        init_tc = float(np.median(fit["tc_theta"]))
        init_finf = fit["finf"].copy()

    summary_df = pd.DataFrame(summary_rows).sort_values("lambda").reset_index(drop=True)
    summary_df.to_csv(outdir / "lambda_scan_summary.csv", index=False)
    with open(outdir / "lambda_scan_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    plot_theta_parameters(theta_values, finf_anchor, baseline_finf, fit_results, outdir, args.dpi)

    lam0 = fit_results[min(fit_results.keys(), key=lambda x: abs(x - 0.0))]
    lam0_df = pd.DataFrame(
        {
            "theta": theta_values,
            "h_alt": h_alt(theta_values),
            "tc_theta_lambda0": lam0["tc_theta"],
            "Finf_lambda0": lam0["finf"],
            "Finf_anchor": finf_anchor,
            "Finf_old_free": baseline_finf,
        }
    )
    lam0_df.to_csv(outdir / "lambda0_theta_table.csv", index=False)

    best_rmse_row = summary_df.iloc[summary_df["rel_rmse"].argmin()]
    final_summary = {
        "beta_fixed_from_baseline": float(beta),
        "baseline": {
            "t_c": baseline_tc,
            "r": baseline_r,
            "rel_rmse": float(baseline["rel_rmse"]),
        },
        "best_data_rmse_lambda": float(best_rmse_row["lambda"]),
        "best_data_rmse": float(best_rmse_row["rel_rmse"]),
        "scan_rows": summary_rows,
    }
    with open(outdir / "final_summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
