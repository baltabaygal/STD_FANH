#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import fit_vw_amplitude as base_fit
import plot_lattice_xi_vs_x_all_vw_quadwarped as quadwarp


T_OSC = 1.5
OUTDIR = ROOT / "results_quadwarp_alpha_qshape"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]
DEFAULT_ALPHA = -0.12114099707506708


def parse_args():
    p = argparse.ArgumentParser(
        description="Test a q-extended transition kernel on top of quadratic t_eff and a fixed v_w^alpha amplitude factor."
    )
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--reference-vw", type=float, default=0.9)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--bootstrap", type=int, default=0)
    p.add_argument("--bootstrap-jobs", type=int, default=1)
    p.add_argument("--bootstrap-seed", type=int, default=12345)
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
    return float(np.sqrt(np.mean(np.square((yfit - y) / np.maximum(y, 1.0e-12)))))


def load_lattice_dataframe(outdir: Path, vw_tags, h_values):
    args = SimpleNamespace(
        rho="",
        vw_folders=vw_tags,
        h_values=h_values,
        tp_min=None,
        tp_max=None,
        bootstrap=0,
        n_jobs=1,
        reg_Finf=0.0,
        tc0=1.5,
        fix_tc=True,
        dpi=220,
        outdir=str(outdir),
    )
    outdir.mkdir(parents=True, exist_ok=True)
    df, _, _ = base_fit.prepare_dataframe(args, outdir)
    return df.sort_values(["v_w", "H", "theta", "tp"]).reset_index(drop=True)


def apply_quadwarp(df, warp_params, beta):
    out = df.copy()
    x_vals = []
    teff_vals = []
    for row in out.itertuples(index=False):
        rec = warp_params.get(f"{float(row.v_w):.1f}")
        if rec is None:
            raise RuntimeError(f"Missing warp parameters for v_w={float(row.v_w):.3f}")
        teff = quadwarp.warp_tp(float(row.tp), rec["log_s"], rec["b"], rec["c"])
        teff_vals.append(float(teff))
        x_vals.append(float(teff * np.power(float(row.H), float(beta))))
    out["t_eff"] = np.asarray(teff_vals, dtype=np.float64)
    out["x"] = np.asarray(x_vals, dtype=np.float64)
    out = out[np.isfinite(out["x"]) & (out["x"] > 0.0)].copy()
    return out.sort_values(["v_w", "H", "theta", "beta_over_H"]).reset_index(drop=True)


def nearest_theta(theta_values, theta0, atol=5.0e-4):
    theta_values = np.asarray(theta_values, dtype=np.float64)
    idx = int(np.argmin(np.abs(theta_values - float(theta0))))
    if abs(theta_values[idx] - float(theta0)) > atol:
        raise RuntimeError(f"No theta match for theta={theta0:.10f}")
    return idx


def amplitude(vw_arr, alpha):
    return np.power(np.asarray(vw_arr, dtype=np.float64), float(alpha))


def fit_tail_alpha(df: pd.DataFrame, alpha: float, outdir: Path, dpi: int):
    rows = []
    for theta, group in df.groupby("theta", sort=True):
        group = group.sort_values("x").copy()
        ntail = max(5, int(math.ceil(0.10 * len(group))))
        tail = group.tail(ntail).copy()
        amp = amplitude(tail["v_w"], alpha)
        x = tail["x"].to_numpy(dtype=np.float64)
        y = tail["xi"].to_numpy(dtype=np.float64) / np.maximum(amp, 1.0e-18)
        design = np.power(x / T_OSC, 1.5)
        c = float(np.sum(design * y) / np.maximum(np.sum(design * design), 1.0e-18))
        f0 = float(group["F0"].iloc[0])
        rows.append(
            {
                "theta": float(theta),
                "F0": f0,
                "F_inf_tail": float(max(c * f0 * f0, 1.0e-8)),
                "n_tail": int(ntail),
            }
        )

        fig, ax = plt.subplots(figsize=(5.8, 4.2))
        for vw, sub in group.groupby("v_w", sort=True):
            amp_sub = float(amplitude(np.asarray([vw]), alpha)[0])
            ax.plot(sub["x"], sub["xi"] / amp_sub, "o", ms=3.2, label=rf"$v_w={vw:.1f}$")
        xfit = np.geomspace(float(np.min(group["x"])), float(np.max(group["x"])), 200)
        ax.plot(xfit, c * np.power(xfit / T_OSC, 1.5), color="black", lw=1.8, label=r"tail fit")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$x=t_{\rm eff}H_*^\beta$")
        ax.set_ylabel(r"$\xi / v_w^\alpha$")
        ax.set_title(rf"Tail fit, $\theta={theta:.3f}$")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"tail_fit_theta_{theta:.3f}.png", dpi=dpi)
        plt.close(fig)
    out = pd.DataFrame(rows).sort_values("theta").reset_index(drop=True)
    out.to_csv(outdir / "Finf_tail.csv", index=False)
    return out


def xi_model_from_params(df: pd.DataFrame, theta_values: np.ndarray, params: np.ndarray, alpha: float, free_q: bool):
    t_c = float(params[0])
    r = float(params[1])
    if free_q:
        q = float(params[2])
        finf = np.asarray(params[3:], dtype=np.float64)
    else:
        q = 1.0
        finf = np.asarray(params[2:], dtype=np.float64)
    theta_index = np.array([nearest_theta(theta_values, th) for th in df["theta"].to_numpy(dtype=np.float64)], dtype=np.int64)
    x = df["x"].to_numpy(dtype=np.float64)
    f0 = df["F0"].to_numpy(dtype=np.float64)
    amp = amplitude(df["v_w"].to_numpy(dtype=np.float64), alpha)
    transient = 1.0 / np.power(1.0 + np.power(x / max(t_c, 1.0e-12), r), q)
    xi_fit = amp * (np.power(x / T_OSC, 1.5) * finf[theta_index] / np.maximum(f0 * f0, 1.0e-18) + transient)
    return xi_fit, theta_index, q


def fit_global(df: pd.DataFrame, finf_tail_df: pd.DataFrame, alpha: float, free_q: bool):
    theta_values = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    tail_map = {float(row.theta): float(row.F_inf_tail) for row in finf_tail_df.itertuples(index=False)}
    finf0 = np.array([max(tail_map.get(float(th), np.nanmedian(finf_tail_df["F_inf_tail"])), 1.0e-8) for th in theta_values], dtype=np.float64)

    if free_q:
        x0 = np.concatenate([np.array([2.5, 3.0, 1.0], dtype=np.float64), finf0])
        lower = np.concatenate([np.array([1.0e-3, 0.1, 0.1], dtype=np.float64), np.full(len(theta_values), 1.0e-8, dtype=np.float64)])
        upper = np.concatenate([np.array([20.0, 20.0, 5.0], dtype=np.float64), np.full(len(theta_values), 1.0e4, dtype=np.float64)])
    else:
        x0 = np.concatenate([np.array([2.5, 3.0], dtype=np.float64), finf0])
        lower = np.concatenate([np.array([1.0e-3, 0.1], dtype=np.float64), np.full(len(theta_values), 1.0e-8, dtype=np.float64)])
        upper = np.concatenate([np.array([20.0, 20.0], dtype=np.float64), np.full(len(theta_values), 1.0e4, dtype=np.float64)])

    def resid(par):
        xi_fit, _, _ = xi_model_from_params(df, theta_values, par, alpha, free_q)
        return (xi_fit - df["xi"].to_numpy(dtype=np.float64)) / np.maximum(df["xi"].to_numpy(dtype=np.float64), 1.0e-12)

    res0 = least_squares(resid, x0=x0, bounds=(lower, upper), loss="soft_l1", f_scale=0.05, max_nfev=12000)
    res = least_squares(resid, x0=res0.x, bounds=(lower, upper), loss="linear", max_nfev=12000)

    xi_fit, theta_index, q = xi_model_from_params(df, theta_values, res.x, alpha, free_q)
    rss = float(np.sum(np.square((xi_fit - df["xi"].to_numpy(dtype=np.float64)) / np.maximum(df["xi"].to_numpy(dtype=np.float64), 1.0e-12))))
    n = int(len(df))
    k = int(len(res.x))
    dof = max(n - k, 1)
    try:
        cov = (rss / dof) * np.linalg.inv(res.jac.T @ res.jac)
    except np.linalg.LinAlgError:
        cov = np.full((len(res.x), len(res.x)), np.nan, dtype=np.float64)
    aic = float(n * math.log(max(rss, 1.0e-18) / n) + 2.0 * k)
    bic = float(n * math.log(max(rss, 1.0e-18) / n) + k * math.log(n))
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "params": res.x,
        "theta_values": theta_values,
        "theta_index": theta_index,
        "xi_fit": xi_fit,
        "covariance": cov,
        "dof": dof,
        "rel_rmse": rel_rmse(df["xi"].to_numpy(dtype=np.float64), xi_fit),
        "rss_frac": rss,
        "AIC": aic,
        "BIC": bic,
        "jac": res.jac,
        "q": float(q),
        "free_q": bool(free_q),
    }


def save_global_fit(result: dict, beta: float, alpha: float, outdir: Path):
    params = np.asarray(result["params"], dtype=np.float64)
    cov = np.asarray(result["covariance"], dtype=np.float64)
    payload = {
        "success": result["success"],
        "message": result["message"],
        "beta": float(beta),
        "alpha": float(alpha),
        "t_c": float(params[0]),
        "r": float(params[1]),
        "q": float(result["q"]),
        "free_q": bool(result["free_q"]),
        "F_inf": {},
        "dof": int(result["dof"]),
        "rel_rmse": float(result["rel_rmse"]),
        "rss_frac": float(result["rss_frac"]),
        "AIC": float(result["AIC"]),
        "BIC": float(result["BIC"]),
        "covariance": result["covariance"].tolist(),
    }
    offset = 3 if result["free_q"] else 2
    for i, th in enumerate(result["theta_values"]):
        err = float(np.sqrt(cov[offset + i, offset + i])) if np.isfinite(cov[offset + i, offset + i]) else np.nan
        payload["F_inf"][f"{float(th):.10f}"] = {"value": float(params[offset + i]), "err": err}
    save_json(outdir / "global_fit.json", payload)
    return payload


def plot_collapse_overlay(df, fit_result, outdir: Path, dpi: int, beta: float, alpha: float, tag: str):
    theta_values = np.sort(df["theta"].unique())
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
                pred, _, _ = xi_model_from_params(cur, fit_result["theta_values"], fit_result["params"], alpha, fit_result["free_q"])
                ax.scatter(cur["x"], cur["xi"], s=20, color=colors[float(vw)], marker=marker_map.get(float(h), "o"), alpha=0.85)
                ax.plot(cur["x"], pred, color=colors[float(vw)], lw=1.6, alpha=0.95)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$x = t_{\rm eff} H_*^\beta$")
        ax.set_ylabel(r"$\xi$")
    for ax in axes[len(theta_values):]:
        ax.axis("off")
    vw_handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
    vw_labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
    h_handles = [plt.Line2D([0], [0], color="black", marker=marker_map[h], linestyle="None") for h in h_values]
    h_labels = [rf"$H_*={h:g}$" for h in h_values]
    fig.legend(vw_handles + h_handles, vw_labels + h_labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(rf"Quadratic $t_{{eff}}$ + $v_w^{{{alpha:.3f}}}$, {tag}, $\beta={beta:.4f}$", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outdir / f"collapse_overlay_{tag}.png", dpi=dpi)
    plt.close(fig)


def plot_raw_xi_vs_betaH(df, fit_result, outdir: Path, dpi: int, alpha: float, tag: str):
    vw_values = np.sort(df["v_w"].unique())
    theta_values = np.sort(df["theta"].unique())
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
                pred, _, _ = xi_model_from_params(cur, fit_result["theta_values"], fit_result["params"], alpha, fit_result["free_q"])
                ax.scatter(cur["beta_over_H"], cur["xi"], s=22, color=colors[float(vw)], alpha=0.85)
                ax.plot(cur["beta_over_H"], pred, color=colors[float(vw)], lw=1.8)
                rows.append({"H": float(h_value), "theta": float(theta), "v_w": float(vw), "rel_rmse": rel_rmse(cur["xi"], pred)})
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
        fig.suptitle(rf"Quadratic $t_{{eff}}$ + $v_w^{{{alpha:.3f}}}$, {tag}, $H_*={h_value:.1f}$", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(outdir / f"xi_vs_betaH_{tag}_H{str(float(h_value)).replace('.', 'p')}.png", dpi=dpi)
        plt.close(fig)
    return rows


def plot_residual_heatmap(df: pd.DataFrame, fit_result: dict, outdir: Path, dpi: int, alpha: float, tag: str):
    theta_values = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    x = df["x"].to_numpy(dtype=np.float64)
    pred, _, _ = xi_model_from_params(df, fit_result["theta_values"], fit_result["params"], alpha, fit_result["free_q"])
    resid = (df["xi"].to_numpy(dtype=np.float64) - pred) / np.maximum(df["xi"].to_numpy(dtype=np.float64), 1.0e-12)
    xbins = np.geomspace(float(np.min(x)), float(np.max(x)), 40)
    heat = np.full((len(theta_values), len(xbins) - 1), np.nan, dtype=np.float64)
    for i, theta in enumerate(theta_values):
        mask_theta = np.isclose(df["theta"], theta, atol=5.0e-4, rtol=0.0)
        for j in range(len(xbins) - 1):
            mask_bin = mask_theta & (x >= xbins[j]) & (x < xbins[j + 1])
            if np.any(mask_bin):
                heat[i, j] = float(np.mean(resid[mask_bin]))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    mesh = ax.pcolormesh(
        xbins,
        np.arange(len(theta_values) + 1),
        heat,
        cmap="coolwarm",
        shading="auto",
        vmin=-0.12,
        vmax=0.12,
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$x=t_{\rm eff}H_*^\beta$")
    ax.set_ylabel(r"$\theta$")
    ax.set_yticks(np.arange(len(theta_values)) + 0.5)
    ax.set_yticklabels([f"{theta:.3f}" for theta in theta_values])
    ax.set_title(f"Residual heatmap, {tag}")
    fig.colorbar(mesh, ax=ax, label=r"$(\xi-\xi_{\rm fit})/\xi$")
    fig.tight_layout()
    fig.savefig(outdir / f"residual_heatmap_{tag}.png", dpi=dpi)
    plt.close(fig)


def build_summary(fit_payload, rmse_rows, alpha):
    return {
        "t_c": float(fit_payload["t_c"]),
        "r": float(fit_payload["r"]),
        "q": float(fit_payload["q"]),
        "alpha": float(alpha),
        "rel_rmse": float(fit_payload["rel_rmse"]),
        "AIC": float(fit_payload["AIC"]),
        "BIC": float(fit_payload["BIC"]),
        "rmse_by_vw": pd.DataFrame(rmse_rows).groupby("v_w", as_index=False)["rel_rmse"].mean().to_dict(orient="records"),
        "rmse_by_H": pd.DataFrame(rmse_rows).groupby("H", as_index=False)["rel_rmse"].mean().to_dict(orient="records"),
    }


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("[load] reading lattice dataframe")
    df = load_lattice_dataframe(outdir, args.vw_folders, args.h_values)

    print("[warp] fitting quadratic per-vw t_eff")
    warp_params, warp_rows = quadwarp.fit_quadwarps(df, args.reference_vw, args.beta)
    warped_df = apply_quadwarp(df, warp_params, args.beta)

    print("[fit] estimating tail amplitudes with fixed vw^alpha factor")
    tail_dir = outdir / "tail"
    tail_dir.mkdir(parents=True, exist_ok=True)
    finf_tail_df = fit_tail_alpha(warped_df, float(args.alpha), tail_dir, args.dpi)

    print("[fit] fitting baseline q=1 model")
    q1_dir = outdir / "q1"
    q1_dir.mkdir(parents=True, exist_ok=True)
    q1_result = fit_global(warped_df, finf_tail_df, float(args.alpha), free_q=False)
    q1_payload = save_global_fit(q1_result, float(args.beta), float(args.alpha), q1_dir)
    q1_rmse = plot_raw_xi_vs_betaH(warped_df, q1_result, q1_dir, args.dpi, float(args.alpha), "q1")
    plot_collapse_overlay(warped_df, q1_result, q1_dir, args.dpi, float(args.beta), float(args.alpha), "q1")
    plot_residual_heatmap(warped_df, q1_result, q1_dir, args.dpi, float(args.alpha), "q1")

    print("[fit] fitting q-free model")
    qfree_dir = outdir / "qfree"
    qfree_dir.mkdir(parents=True, exist_ok=True)
    qfree_result = fit_global(warped_df, finf_tail_df, float(args.alpha), free_q=True)
    qfree_payload = save_global_fit(qfree_result, float(args.beta), float(args.alpha), qfree_dir)
    qfree_rmse = plot_raw_xi_vs_betaH(warped_df, qfree_result, qfree_dir, args.dpi, float(args.alpha), "qfree")
    plot_collapse_overlay(warped_df, qfree_result, qfree_dir, args.dpi, float(args.beta), float(args.alpha), "qfree")
    plot_residual_heatmap(warped_df, qfree_result, qfree_dir, args.dpi, float(args.alpha), "qfree")

    comparison = {
        "status": "ok",
        "alpha_fixed": float(args.alpha),
        "beta": float(args.beta),
        "reference_vw": float(args.reference_vw),
        "warp_rows": warp_rows,
        "q1": build_summary(q1_payload, q1_rmse, float(args.alpha)),
        "qfree": build_summary(qfree_payload, qfree_rmse, float(args.alpha)),
        "delta_AIC_qfree_minus_q1": float(qfree_payload["AIC"] - q1_payload["AIC"]),
        "delta_BIC_qfree_minus_q1": float(qfree_payload["BIC"] - q1_payload["BIC"]),
        "rel_rmse_improvement": float(q1_payload["rel_rmse"] - qfree_payload["rel_rmse"]),
        "preferred_model": "qfree" if qfree_payload["AIC"] < q1_payload["AIC"] else "q1",
        "outputs": {
            "q1_summary": str(q1_dir / "global_fit.json"),
            "q1_H1p0": str(q1_dir / "xi_vs_betaH_q1_H1p0.png"),
            "q1_H1p5": str(q1_dir / "xi_vs_betaH_q1_H1p5.png"),
            "q1_H2p0": str(q1_dir / "xi_vs_betaH_q1_H2p0.png"),
            "qfree_summary": str(qfree_dir / "global_fit.json"),
            "qfree_H1p0": str(qfree_dir / "xi_vs_betaH_qfree_H1p0.png"),
            "qfree_H1p5": str(qfree_dir / "xi_vs_betaH_qfree_H1p5.png"),
            "qfree_H2p0": str(qfree_dir / "xi_vs_betaH_qfree_H2p0.png"),
        },
    }
    save_json(outdir / "final_summary.json", comparison)
    print(json.dumps(to_native(comparison), sort_keys=True))


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
