#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import gammaincc

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import collapse_and_fit_fanh_tosc as collapse
import fit_lattice_quadwarp_universal as uq
import plot_lattice_xi_vs_x_all_vw_quadwarped as quadwarp


OUTDIR = ROOT / "results_gamma_survival_quadwarp_beta0"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit a universal lattice curve using quadratic t_eff and a Gamma survival transition kernel."
    )
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--reference-vw", type=float, default=0.9)
    p.add_argument("--dpi", type=int, default=220)
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
    mask = np.isfinite(y) & np.isfinite(yfit) & (y > 0.0)
    if np.count_nonzero(mask) == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square((yfit[mask] - y[mask]) / np.maximum(y[mask], 1.0e-12)))))


def theta_index(theta_values, theta_array):
    idx_map = {float(th): i for i, th in enumerate(np.asarray(theta_values, dtype=np.float64))}
    return np.array([idx_map[float(th)] for th in np.asarray(theta_array, dtype=np.float64)], dtype=np.int64)


def gamma_survival(x, tc, k):
    x = np.asarray(x, dtype=np.float64)
    tc = max(float(tc), 1.0e-12)
    k = max(float(k), 1.0e-12)
    z = k * np.maximum(x, 0.0) / tc
    return gammaincc(k, z)


def xi_model_gamma(df, theta_values, params):
    tc = float(params[0])
    k = float(params[1])
    finf = np.asarray(params[2:], dtype=np.float64)
    idx = theta_index(theta_values, df["theta"].to_numpy(dtype=np.float64))
    x = df["x"].to_numpy(dtype=np.float64)
    f0 = df["F0"].to_numpy(dtype=np.float64)
    plateau = np.power(np.maximum(x / collapse.T_OSC, 1.0e-18), 1.5) * finf[idx] / np.maximum(f0 * f0, 1.0e-18)
    transition = gamma_survival(x, tc, k)
    return plateau + transition, idx


def fit_gamma_model(df, finf_tail_df):
    theta_values = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    tail_map = {float(row.theta): float(row.F_inf_tail) for row in finf_tail_df.itertuples(index=False)}
    default_tail = float(np.nanmedian(finf_tail_df["F_inf_tail"].to_numpy(dtype=np.float64)))
    finf0 = np.array([max(tail_map.get(float(th), default_tail), 1.0e-8) for th in theta_values], dtype=np.float64)

    x0 = np.concatenate([np.array([2.5, 2.0], dtype=np.float64), finf0])
    lower = np.concatenate([np.array([1.0e-3, 0.1], dtype=np.float64), np.full(len(theta_values), 1.0e-8, dtype=np.float64)])
    upper = np.concatenate([np.array([30.0, 100.0], dtype=np.float64), np.full(len(theta_values), 1.0e4, dtype=np.float64)])
    y = df["xi"].to_numpy(dtype=np.float64)

    def resid(par):
        yfit, _ = xi_model_gamma(df, theta_values, par)
        return (yfit - y) / np.maximum(y, 1.0e-12)

    res0 = least_squares(resid, x0=x0, bounds=(lower, upper), loss="soft_l1", f_scale=0.05, max_nfev=8000)
    res = least_squares(resid, x0=res0.x, bounds=(lower, upper), loss="linear", max_nfev=8000)

    yfit, idx = xi_model_gamma(df, theta_values, res.x)
    rss_frac = float(np.sum(np.square((yfit - y) / np.maximum(y, 1.0e-12))))
    n = int(len(df))
    npar = int(len(res.x))
    dof = max(n - npar, 1)
    try:
        cov = (rss_frac / dof) * np.linalg.inv(res.jac.T @ res.jac)
    except np.linalg.LinAlgError:
        cov = np.full((npar, npar), np.nan, dtype=np.float64)
    aic = float(n * math.log(max(rss_frac, 1.0e-18) / n) + 2.0 * npar)
    bic = float(n * math.log(max(rss_frac, 1.0e-18) / n) + npar * math.log(n))
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "params": np.asarray(res.x, dtype=np.float64),
        "theta_values": theta_values,
        "theta_index": idx,
        "xi_fit": yfit,
        "covariance": cov,
        "dof": dof,
        "rel_rmse": rel_rmse(y, yfit),
        "rss_frac": rss_frac,
        "AIC": aic,
        "BIC": bic,
    }


def save_gamma_fit(result, beta, outdir: Path):
    params = np.asarray(result["params"], dtype=np.float64)
    cov = np.asarray(result["covariance"], dtype=np.float64)
    tc_err = float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else np.nan
    k_err = float(np.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) else np.nan
    finf_err = [
        float(np.sqrt(cov[2 + i, 2 + i])) if np.isfinite(cov[2 + i, 2 + i]) else np.nan
        for i in range(len(result["theta_values"]))
    ]
    payload = {
        "success": result["success"],
        "message": result["message"],
        "beta": float(beta),
        "t_c": float(params[0]),
        "t_c_err": tc_err,
        "k_gamma": float(params[1]),
        "k_gamma_err": k_err,
        "F_inf": {
            f"{float(th):.10f}": {"value": float(params[2 + i]), "err": finf_err[i]}
            for i, th in enumerate(result["theta_values"])
        },
        "dof": int(result["dof"]),
        "rel_rmse": float(result["rel_rmse"]),
        "rss_frac": float(result["rss_frac"]),
        "AIC": float(result["AIC"]),
        "BIC": float(result["BIC"]),
    }
    save_json(outdir / "global_fit_gamma.json", payload)
    return payload


def choose_theta_subset(theta_values):
    targets = np.array([0.262, 0.785, 1.309, 1.833, 2.356, 2.880], dtype=np.float64)
    theta_values = np.asarray(sorted(theta_values), dtype=np.float64)
    out = []
    for target in targets:
        idx = int(np.argmin(np.abs(theta_values - target)))
        val = float(theta_values[idx])
        if val not in out:
            out.append(val)
    return np.asarray(out, dtype=np.float64)


def plot_collapse_overlay(df, pred_col: str, outdir: Path, dpi: int, beta: float, title: str, filename: str):
    theta_values = choose_theta_subset(df["theta"].unique())
    vw_values = np.sort(df["v_w"].unique())
    h_values = np.sort(df["H"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    marker_map = {1.0: "s", 1.5: "^", 2.0: "D", 0.5: "o"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, theta in zip(axes, theta_values):
        sub = df[np.isclose(df["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
        for vw in vw_values:
            for h in h_values:
                cur = sub[
                    np.isclose(sub["v_w"], float(vw), atol=1.0e-12, rtol=0.0)
                    & np.isclose(sub["H"], float(h), atol=1.0e-12, rtol=0.0)
                ].sort_values("x")
                if cur.empty:
                    continue
                ax.scatter(cur["x"], cur["xi"], s=20, color=colors[float(vw)], marker=marker_map.get(float(h), "o"), alpha=0.85)
                ax.plot(cur["x"], cur[pred_col], color=colors[float(vw)], lw=1.6, alpha=0.95)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$x=t_{\rm eff} H_*^\beta$")
        ax.set_ylabel(r"$\xi$")
    for ax in axes[len(theta_values):]:
        ax.axis("off")
    vw_handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
    vw_labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
    h_handles = [plt.Line2D([0], [0], color="black", marker=marker_map[h], linestyle="None") for h in h_values]
    h_labels = [rf"$H_*={h:g}$" for h in h_values]
    fig.legend(vw_handles + h_handles, vw_labels + h_labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(title + rf", $\beta={beta:.4f}$", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outdir / filename, dpi=dpi)
    plt.close(fig)


def plot_raw_xi_vs_betaH(df, pred_col: str, outdir: Path, dpi: int, beta: float, tag: str):
    vw_values = np.sort(df["v_w"].unique())
    theta_values = choose_theta_subset(df["theta"].unique())
    h_values = np.sort(df["H"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    rows = []
    for h_value in h_values:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
        axes = axes.ravel()
        sub_h = df[np.isclose(df["H"], float(h_value), atol=1.0e-12, rtol=0.0)].copy()
        for ax, theta in zip(axes, theta_values):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
            for vw in vw_values:
                cur = sub[np.isclose(sub["v_w"], float(vw), atol=1.0e-12, rtol=0.0)].sort_values("beta_over_H").copy()
                if cur.empty:
                    continue
                ax.scatter(cur["beta_over_H"], cur["xi"], s=22, color=colors[float(vw)], alpha=0.85)
                ax.plot(cur["beta_over_H"], cur[pred_col], color=colors[float(vw)], lw=1.8)
                rows.append({"H": float(h_value), "theta": float(theta), "v_w": float(vw), "rel_rmse": rel_rmse(cur["xi"], cur[pred_col])})
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$\beta/H_*$")
            ax.set_ylabel(r"$\xi$")
        for ax in axes[len(theta_values):]:
            ax.axis("off")
        handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
        labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
        fig.suptitle(rf"{tag} in raw $\xi(\beta/H_*)$, $H_*={h_value:.1f}$, $\beta={beta:.4f}$", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        htag = str(float(h_value)).replace(".", "p")
        fig.savefig(outdir / f"xi_vs_betaH_{tag}_H{htag}.png", dpi=dpi)
        plt.close(fig)
    return rows


def metric_tables(df, pred_col: str):
    by_theta = [{"theta": float(theta), "rel_rmse": rel_rmse(sub["xi"], sub[pred_col])} for theta, sub in df.groupby("theta", sort=True)]
    by_vw = [{"v_w": float(vw), "rel_rmse": rel_rmse(sub["xi"], sub[pred_col])} for vw, sub in df.groupby("v_w", sort=True)]
    by_h = [{"H": float(h), "rel_rmse": rel_rmse(sub["xi"], sub[pred_col])} for h, sub in df.groupby("H", sort=True)]
    return by_theta, by_vw, by_h


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("[load] reading lattice dataframe")
    df = uq.load_lattice_dataframe(outdir, args.vw_folders, args.h_values)

    print("[warp] fitting per-vw quadratic time warp")
    warp_params, warp_rows = quadwarp.fit_quadwarps(df, args.reference_vw, args.beta)

    print("[warp] applying warp")
    warped_df = uq.apply_quadwarp(df, warp_params, args.beta)

    print("[tail] estimating initial F_inf from warped tails")
    finf_tail_df = collapse.fit_tail(warped_df, outdir, args.dpi)

    print("[baseline] fitting old r-kernel model on the same warped data")
    baseline_result = collapse.fit_global(warped_df, finf_tail_df)
    baseline_payload = collapse.save_global_fit(baseline_result, args.beta, outdir)
    warped_df["xi_baseline"] = baseline_result["xi_fit"]

    print("[gamma] fitting Gamma survival transition model")
    gamma_result = fit_gamma_model(warped_df, finf_tail_df)
    gamma_payload = save_gamma_fit(gamma_result, args.beta, outdir)
    warped_df["xi_gamma"] = gamma_result["xi_fit"]

    print("[plot] writing overlays")
    plot_collapse_overlay(
        warped_df,
        "xi_baseline",
        outdir,
        args.dpi,
        float(args.beta),
        r"Rigid transition kernel $1/(1+(x/t_c)^r)$",
        "collapse_overlay_baseline.png",
    )
    plot_collapse_overlay(
        warped_df,
        "xi_gamma",
        outdir,
        args.dpi,
        float(args.beta),
        r"Gamma survival transition kernel $S_\Gamma(x;t_c,k)$",
        "collapse_overlay_gamma.png",
    )
    raw_baseline = plot_raw_xi_vs_betaH(warped_df, "xi_baseline", outdir, args.dpi, float(args.beta), "baseline")
    raw_gamma = plot_raw_xi_vs_betaH(warped_df, "xi_gamma", outdir, args.dpi, float(args.beta), "gamma")

    theta_baseline, vw_baseline, h_baseline = metric_tables(warped_df, "xi_baseline")
    theta_gamma, vw_gamma, h_gamma = metric_tables(warped_df, "xi_gamma")

    comparison = {
        "baseline_rel_rmse": float(baseline_result["rel_rmse"]),
        "gamma_rel_rmse": float(gamma_result["rel_rmse"]),
        "delta_rel_rmse_gamma_minus_baseline": float(gamma_result["rel_rmse"] - baseline_result["rel_rmse"]),
        "baseline_AIC": float(baseline_result["AIC"]),
        "gamma_AIC": float(gamma_result["AIC"]),
        "delta_AIC_gamma_minus_baseline": float(gamma_result["AIC"] - baseline_result["AIC"]),
        "baseline_BIC": float(baseline_result["BIC"]),
        "gamma_BIC": float(gamma_result["BIC"]),
        "delta_BIC_gamma_minus_baseline": float(gamma_result["BIC"] - baseline_result["BIC"]),
        "preferred_model": "gamma" if gamma_result["AIC"] < baseline_result["AIC"] else "baseline",
    }
    save_json(outdir / "model_comparison.json", comparison)

    warped_df.to_csv(outdir / "predictions.csv", index=False)

    summary = {
        "status": "ok",
        "beta": float(args.beta),
        "reference_vw": float(args.reference_vw),
        "t_osc": float(collapse.T_OSC),
        "warp_params": warp_params,
        "warp_rows": warp_rows,
        "baseline": {
            "t_c": float(baseline_result["params"][0]),
            "r": float(baseline_result["params"][1]),
            "rel_rmse": float(baseline_result["rel_rmse"]),
            "AIC": float(baseline_result["AIC"]),
            "BIC": float(baseline_result["BIC"]),
            "rmse_by_theta": theta_baseline,
            "rmse_by_vw": vw_baseline,
            "rmse_by_H": h_baseline,
        },
        "gamma_survival": {
            "t_c": float(gamma_result["params"][0]),
            "k_gamma": float(gamma_result["params"][1]),
            "rel_rmse": float(gamma_result["rel_rmse"]),
            "AIC": float(gamma_result["AIC"]),
            "BIC": float(gamma_result["BIC"]),
            "rmse_by_theta": theta_gamma,
            "rmse_by_vw": vw_gamma,
            "rmse_by_H": h_gamma,
        },
        "comparison": comparison,
        "outputs": {
            "collapse_baseline": str(outdir / "collapse_overlay_baseline.png"),
            "collapse_gamma": str(outdir / "collapse_overlay_gamma.png"),
            "raw_gamma_H1p0": str(outdir / "xi_vs_betaH_gamma_H1p0.png"),
            "raw_gamma_H1p5": str(outdir / "xi_vs_betaH_gamma_H1p5.png"),
            "raw_gamma_H2p0": str(outdir / "xi_vs_betaH_gamma_H2p0.png"),
            "predictions": str(outdir / "predictions.csv"),
        },
        "raw_plot_rmse_baseline": raw_baseline,
        "raw_plot_rmse_gamma": raw_gamma,
        "baseline_payload": baseline_payload,
        "gamma_payload": gamma_payload,
    }
    save_json(outdir / "final_summary.json", summary)
    print(json.dumps(to_native(summary), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        outdir = OUTDIR.resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "error",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_json(outdir / "_error.json", payload)
        print(json.dumps(payload, sort_keys=True))
        raise
