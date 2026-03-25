#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import traceback
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from scipy.optimize import least_squares

import fit_lattice_fixed_ode_amplitudes as shared
import fit_lattice_fixed_ode_amplitudes_theta_tc_shared_r_by_vw as shared_rvw
import fit_lattice_quadwarp_universal as quad


ROOT = Path(__file__).resolve().parent
OUTDIR_DEFAULT = ROOT / "results_lattice_broken_powerlaw_minimal_beta0"
REFERENCE_DIR_DEFAULT = ROOT / "results_lattice_broken_powerlaw_knob_h0_beta0"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]

CASE_CONFIGS = [
    ("allvw_c1_rvw_tcvw", r"broken power, $C=1$, $t_c(v_w)$, $r(v_w)$", "allvw_c1_rvw_tcvw"),
    ("allvw_c1_rlinear_tclinear", r"broken power, $C=1$, $t_c(v_w)=t_{c0}+t_{c1}v_w$, $r(v_w)=r_0+r_1 v_w$", "allvw_c1_rlinear_tclinear"),
    ("allvw_c1_rlinear_tcconst", r"broken power, $C=1$, $t_c=t_c^*$, $r(v_w)=r_0+r_1 v_w$", "allvw_c1_rlinear_tcconst"),
    ("allvw_c1_rvw_h0gammavw_ref", r"broken power ref, $C=1$, $t_c^{(0)}(v_w)(h+h_0)^{\gamma(v_w)}$", "allvw_c1_rvw_h0gammavw_ref"),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Test minimal broken-power-law timing models: free tc(v_w),r(v_w) and smooth tc*=const with linear r(v_w)."
    )
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--fixed-beta", type=float, default=0.0)
    p.add_argument("--t-osc", type=float, default=1.5)
    p.add_argument("--ode-summary", type=str, default=str(shared.ODE_SUMMARY_DEFAULT))
    p.add_argument("--reference-dir", type=str, default=str(REFERENCE_DIR_DEFAULT))
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--outdir", type=str, default=str(OUTDIR_DEFAULT))
    return p.parse_args()


def save_json(path: Path, payload):
    path.write_text(json.dumps(shared.to_native(payload), indent=2, sort_keys=True))


def error_exit(outdir: Path, exc: Exception):
    payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "_error.json", payload)
    print(json.dumps(payload, sort_keys=True))


def xi_broken_power(u: np.ndarray, r: np.ndarray) -> np.ndarray:
    u = np.maximum(np.asarray(u, dtype=np.float64), 1.0e-18)
    r = np.maximum(np.asarray(r, dtype=np.float64), 1.0e-12)
    z = r * np.log(u)
    log_xi = np.where(z > 50.0, z / r, np.log1p(np.exp(np.clip(z, -700.0, 700.0))) / r)
    return np.exp(log_xi)


def build_meta(df, theta_values: np.ndarray, vw_values: np.ndarray, ode: dict, t_osc: float):
    return shared_rvw.build_meta(df, theta_values, vw_values, ode, t_osc)


def parse_r_by_vw(summary: dict, vw_values: np.ndarray):
    table = summary["r_by_vw"]
    return np.asarray([float(table[f"{float(vw):.1f}"]) for vw in vw_values], dtype=np.float64)


def parse_tc_mean_by_vw(summary: dict, theta_values: np.ndarray, vw_values: np.ndarray):
    table = summary["tc_by_vw_theta"]
    vals = []
    for vw in vw_values:
        block = table[f"{float(vw):.1f}"]
        arr = [float(block[f"{float(theta):.10f}"]) for theta in theta_values]
        vals.append(float(np.mean(arr)))
    return np.asarray(vals, dtype=np.float64)


def fit_case_tcvw(meta, fixed_beta: float, init_tc_by_vw: np.ndarray, init_r_by_vw: np.ndarray):
    n_vw = len(meta["vw_values"])
    x0 = np.concatenate([np.asarray(init_tc_by_vw, dtype=np.float64), np.asarray(init_r_by_vw, dtype=np.float64)])
    lower = np.concatenate([np.full(n_vw, 0.01), np.full(n_vw, 0.1)])
    upper = np.concatenate([np.full(n_vw, 300.0), np.full(n_vw, 30.0)])

    def unpack(par):
        tc_by_vw = np.asarray(par[:n_vw], dtype=np.float64)
        r_by_vw = np.asarray(par[n_vw : 2 * n_vw], dtype=np.float64)
        return tc_by_vw, r_by_vw

    def resid(par):
        tc_by_vw, r_by_vw = unpack(par)
        x, xi_scale = shared_rvw.x_and_xi_scale(meta, fixed_beta)
        tc_eff = tc_by_vw[meta["vw_idx"]]
        r_eff = r_by_vw[meta["vw_idx"]]
        u = meta["F_inf"] * xi_scale / np.maximum(meta["F0_sq"] * np.power(tc_eff, 1.5), 1.0e-18)
        xi_fit = xi_broken_power(u, r_eff)
        return (xi_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)

    huber = least_squares(resid, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=60000)
    final = least_squares(resid, huber.x, bounds=(lower, upper), loss="linear", max_nfev=60000)
    tc_by_vw, r_by_vw = unpack(final.x)
    x, xi_scale = shared_rvw.x_and_xi_scale(meta, fixed_beta)
    tc_eff = tc_by_vw[meta["vw_idx"]]
    r_eff = r_by_vw[meta["vw_idx"]]
    u = meta["F_inf"] * xi_scale / np.maximum(meta["F0_sq"] * np.power(tc_eff, 1.5), 1.0e-18)
    xi_fit = xi_broken_power(u, r_eff)
    frac_resid = (xi_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    aic, bic = shared.aic_bic(frac_resid, len(final.x))
    return {
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "mode": "allvw_c1_rvw_tcvw",
        "beta": float(fixed_beta),
        "canonical_xi_form": "xi = [1 + u^r]^(1/r)",
        "xi_model_form": "xi = [1 + (F_inf(theta0) (2 x / (3 t_osc t_c(v_w)))^(3/2) / F0(theta0)^2)^(r(v_w)) ]^(1/r(v_w))",
        "rel_rmse": shared_rvw.rel_rmse(meta["xi"], xi_fit),
        "AIC": float(aic),
        "BIC": float(bic),
        "n_points": int(meta["xi"].size),
        "n_params": int(len(final.x)),
        "result_x": np.asarray(final.x, dtype=np.float64),
        "x_fit": x,
        "u_fit": u,
        "y_fit": xi_fit,
        "f_tilde_fit": xi_fit * meta["F0"] / np.maximum(xi_scale, 1.0e-18),
        "frac_resid": frac_resid,
        "c_by_vw": np.ones(n_vw, dtype=np.float64),
        "tc_by_vw": np.asarray(tc_by_vw, dtype=np.float64),
        "r_by_vw": np.asarray(r_by_vw, dtype=np.float64),
        "per_vw_rel_rmse": {
            f"{float(vw):.1f}": shared_rvw.rel_rmse(
                meta["xi"][np.isclose(meta["v_w"], float(vw), atol=1.0e-12)],
                xi_fit[np.isclose(meta["v_w"], float(vw), atol=1.0e-12)],
            )
            for vw in meta["vw_values"]
        },
    }


def fit_case_rlinear_tcconst(meta, fixed_beta: float, init_tc_star: float, init_r_by_vw: np.ndarray):
    vw_values = np.asarray(meta["vw_values"], dtype=np.float64)
    coeff = np.polyfit(vw_values, np.asarray(init_r_by_vw, dtype=np.float64), deg=1)
    init_r1 = float(coeff[0])
    init_r0 = float(coeff[1])
    x0 = np.asarray([float(init_tc_star), init_r0, init_r1], dtype=np.float64)
    lower = np.asarray([0.01, 0.01, -10.0], dtype=np.float64)
    upper = np.asarray([300.0, 30.0, 10.0], dtype=np.float64)

    def model_terms(par):
        tc_star = float(par[0])
        r0 = float(par[1])
        r1 = float(par[2])
        r_by_vw = r0 + r1 * vw_values
        return tc_star, r0, r1, r_by_vw

    def resid(par):
        tc_star, _, _, r_by_vw = model_terms(par)
        if np.any(r_by_vw <= 0.05):
            return np.full_like(meta["xi"], 1.0e6, dtype=np.float64)
        x, xi_scale = shared_rvw.x_and_xi_scale(meta, fixed_beta)
        r_eff = (r_by_vw)[meta["vw_idx"]]
        u = meta["F_inf"] * xi_scale / np.maximum(meta["F0_sq"] * np.power(tc_star, 1.5), 1.0e-18)
        xi_fit = xi_broken_power(u, r_eff)
        return (xi_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)

    huber = least_squares(resid, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=60000)
    final = least_squares(resid, huber.x, bounds=(lower, upper), loss="linear", max_nfev=60000)
    tc_star, r0, r1, r_by_vw = model_terms(final.x)
    x, xi_scale = shared_rvw.x_and_xi_scale(meta, fixed_beta)
    r_eff = r_by_vw[meta["vw_idx"]]
    u = meta["F_inf"] * xi_scale / np.maximum(meta["F0_sq"] * np.power(tc_star, 1.5), 1.0e-18)
    xi_fit = xi_broken_power(u, r_eff)
    frac_resid = (xi_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    aic, bic = shared.aic_bic(frac_resid, len(final.x))
    return {
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "mode": "allvw_c1_rlinear_tcconst",
        "beta": float(fixed_beta),
        "canonical_xi_form": "xi = [1 + u^r]^(1/r)",
        "xi_model_form": "xi = [1 + (F_inf(theta0) (2 x / (3 t_osc t_c^*))^(3/2) / F0(theta0)^2)^(r0 + r1 v_w) ]^(1/(r0 + r1 v_w))",
        "rel_rmse": shared_rvw.rel_rmse(meta["xi"], xi_fit),
        "AIC": float(aic),
        "BIC": float(bic),
        "n_points": int(meta["xi"].size),
        "n_params": int(len(final.x)),
        "result_x": np.asarray(final.x, dtype=np.float64),
        "x_fit": x,
        "u_fit": u,
        "y_fit": xi_fit,
        "f_tilde_fit": xi_fit * meta["F0"] / np.maximum(xi_scale, 1.0e-18),
        "frac_resid": frac_resid,
        "c_by_vw": np.ones(len(vw_values), dtype=np.float64),
        "tc_star": float(tc_star),
        "r0": float(r0),
        "r1": float(r1),
        "r_by_vw": np.asarray(r_by_vw, dtype=np.float64),
        "per_vw_rel_rmse": {
            f"{float(vw):.1f}": shared_rvw.rel_rmse(
                meta["xi"][np.isclose(meta["v_w"], float(vw), atol=1.0e-12)],
                xi_fit[np.isclose(meta["v_w"], float(vw), atol=1.0e-12)],
            )
            for vw in meta["vw_values"]
        },
    }


def fit_case_rlinear_tclinear(meta, fixed_beta: float, init_tc_by_vw: np.ndarray, init_r_by_vw: np.ndarray):
    vw_values = np.asarray(meta["vw_values"], dtype=np.float64)
    tc_coeff = np.polyfit(vw_values, np.asarray(init_tc_by_vw, dtype=np.float64), deg=1)
    init_tc1 = float(tc_coeff[0])
    init_tc0 = float(tc_coeff[1])
    r_coeff = np.polyfit(vw_values, np.asarray(init_r_by_vw, dtype=np.float64), deg=1)
    init_r1 = float(r_coeff[0])
    init_r0 = float(r_coeff[1])
    x0 = np.asarray([init_tc0, init_tc1, init_r0, init_r1], dtype=np.float64)
    lower = np.asarray([0.01, -20.0, 0.01, -10.0], dtype=np.float64)
    upper = np.asarray([300.0, 20.0, 30.0, 10.0], dtype=np.float64)

    def model_terms(par):
        tc0 = float(par[0])
        tc1 = float(par[1])
        r0 = float(par[2])
        r1 = float(par[3])
        tc_by_vw = tc0 + tc1 * vw_values
        r_by_vw = r0 + r1 * vw_values
        return tc0, tc1, r0, r1, tc_by_vw, r_by_vw

    def resid(par):
        _, _, _, _, tc_by_vw, r_by_vw = model_terms(par)
        if np.any(tc_by_vw <= 0.05) or np.any(r_by_vw <= 0.05):
            return np.full_like(meta["xi"], 1.0e6, dtype=np.float64)
        x, xi_scale = shared_rvw.x_and_xi_scale(meta, fixed_beta)
        tc_eff = tc_by_vw[meta["vw_idx"]]
        r_eff = r_by_vw[meta["vw_idx"]]
        u = meta["F_inf"] * xi_scale / np.maximum(meta["F0_sq"] * np.power(tc_eff, 1.5), 1.0e-18)
        xi_fit = xi_broken_power(u, r_eff)
        return (xi_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)

    huber = least_squares(resid, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=60000)
    final = least_squares(resid, huber.x, bounds=(lower, upper), loss="linear", max_nfev=60000)
    tc0, tc1, r0, r1, tc_by_vw, r_by_vw = model_terms(final.x)
    x, xi_scale = shared_rvw.x_and_xi_scale(meta, fixed_beta)
    tc_eff = tc_by_vw[meta["vw_idx"]]
    r_eff = r_by_vw[meta["vw_idx"]]
    u = meta["F_inf"] * xi_scale / np.maximum(meta["F0_sq"] * np.power(tc_eff, 1.5), 1.0e-18)
    xi_fit = xi_broken_power(u, r_eff)
    frac_resid = (xi_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    aic, bic = shared.aic_bic(frac_resid, len(final.x))
    return {
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "mode": "allvw_c1_rlinear_tclinear",
        "beta": float(fixed_beta),
        "canonical_xi_form": "xi = [1 + u^r]^(1/r)",
        "xi_model_form": "xi = [1 + (F_inf(theta0) (2 x / (3 t_osc (tc0 + tc1 v_w)))^(3/2) / F0(theta0)^2)^(r0 + r1 v_w) ]^(1/(r0 + r1 v_w))",
        "rel_rmse": shared_rvw.rel_rmse(meta["xi"], xi_fit),
        "AIC": float(aic),
        "BIC": float(bic),
        "n_points": int(meta["xi"].size),
        "n_params": int(len(final.x)),
        "result_x": np.asarray(final.x, dtype=np.float64),
        "x_fit": x,
        "u_fit": u,
        "y_fit": xi_fit,
        "f_tilde_fit": xi_fit * meta["F0"] / np.maximum(xi_scale, 1.0e-18),
        "frac_resid": frac_resid,
        "c_by_vw": np.ones(len(vw_values), dtype=np.float64),
        "tc0_lin": float(tc0),
        "tc1_lin": float(tc1),
        "r0": float(r0),
        "r1": float(r1),
        "tc_by_vw": np.asarray(tc_by_vw, dtype=np.float64),
        "r_by_vw": np.asarray(r_by_vw, dtype=np.float64),
        "per_vw_rel_rmse": {
            f"{float(vw):.1f}": shared_rvw.rel_rmse(
                meta["xi"][np.isclose(meta["v_w"], float(vw), atol=1.0e-12)],
                xi_fit[np.isclose(meta["v_w"], float(vw), atol=1.0e-12)],
            )
            for vw in meta["vw_values"]
        },
    }


def build_prediction_frame(df, meta, fit_payload: dict):
    out = df.copy()
    out["x"] = fit_payload["x_fit"]
    out["u_fit"] = fit_payload["u_fit"]
    out["xi_fit"] = fit_payload["y_fit"]
    out["f_tilde_fit"] = fit_payload["f_tilde_fit"]
    return out


def summarize_payload(payload: dict, vw_values: np.ndarray):
    out = {
        k: v
        for k, v in payload.items()
        if k not in {"result_x", "x_fit", "u_fit", "y_fit", "f_tilde_fit", "frac_resid", "r_by_vw", "c_by_vw", "tc_by_vw"}
    }
    out["r_by_vw"] = {f"{float(vw):.1f}": float(payload["r_by_vw"][i]) for i, vw in enumerate(vw_values)}
    out["c_by_vw"] = {f"{float(vw):.1f}": float(payload["c_by_vw"][i]) for i, vw in enumerate(vw_values)}
    if "tc_by_vw" in payload:
        out["tc_by_vw"] = {f"{float(vw):.1f}": float(payload["tc_by_vw"][i]) for i, vw in enumerate(vw_values)}
    return out


def rmse_tables(df_pred):
    rows = []
    by_h = defaultdict(list)
    by_h_vw = defaultdict(list)
    for h_value in sorted(df_pred["H"].unique()):
        sub_h = df_pred[np.isclose(df_pred["H"], float(h_value), atol=1.0e-8)].copy()
        for theta in sorted(sub_h["theta"].unique()):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=1.0e-8)].copy()
            for vw in sorted(sub["v_w"].unique()):
                cur = sub[np.isclose(sub["v_w"], float(vw), atol=1.0e-8)]
                rmse = shared.rel_rmse(cur["xi"], cur["xi_fit"])
                rows.append({"H": float(h_value), "theta": float(theta), "v_w": float(vw), "rel_rmse": rmse})
                by_h[float(h_value)].append(rmse)
                by_h_vw[(float(h_value), float(vw))].append(rmse)
    mean_by_h = {f"{h:.1f}": float(np.mean(vals)) for h, vals in sorted(by_h.items())}
    mean_by_h_vw = {f"H{h:.1f}_vw{vw:.1f}": float(np.mean(vals)) for (h, vw), vals in sorted(by_h_vw.items())}
    return rows, mean_by_h, mean_by_h_vw


def plot_xi_vs_u_by_vw(df_pred, vw_values: np.ndarray, fit_payload: dict, outpath: Path, dpi: int):
    cmap = plt.get_cmap("viridis")
    theta_values = np.sort(df_pred["theta"].unique())
    colors = {float(theta): cmap(i / max(len(theta_values) - 1, 1)) for i, theta in enumerate(theta_values)}
    marker_map = {1.0: "s", 1.5: "^", 2.0: "D", 0.5: "o"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True, constrained_layout=True)
    used_axes = 0
    for ax, vw in zip(axes.ravel(), vw_values):
        sub_vw = df_pred[np.isclose(df_pred["v_w"], float(vw), atol=1.0e-8)].copy()
        for theta in theta_values:
            sub_theta = sub_vw[np.isclose(sub_vw["theta"], float(theta), atol=1.0e-8)].copy()
            for h in sorted(sub_theta["H"].unique()):
                cur = sub_theta[np.isclose(sub_theta["H"], float(h), atol=1.0e-8)].sort_values("u_fit")
                ax.scatter(cur["u_fit"], cur["xi"], s=18, color=colors[float(theta)], marker=marker_map.get(float(h), "o"), alpha=0.85)
        u_min = max(float(sub_vw["u_fit"].min()), 1.0e-6)
        u_max = max(float(sub_vw["u_fit"].max()), u_min * 10.0)
        u_grid = np.logspace(np.log10(u_min), np.log10(u_max), 300)
        i = int(np.argmin(np.abs(vw_values - float(vw))))
        r_eff = float(fit_payload["r_by_vw"][i])
        xi_curve = xi_broken_power(u_grid, np.full_like(u_grid, r_eff))
        ax.plot(u_grid, xi_curve, color="black", lw=1.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$v_w={float(vw):.1f}$")
        ax.set_xlabel(r"$u$")
        ax.set_ylabel(r"$\xi$")
        used_axes += 1
    for ax in axes.ravel()[used_axes:]:
        ax.axis("off")
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_rmse_by_h(case_summaries: dict, outpath: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    h_values = sorted(
        {
            float(k)
            for summary in case_summaries.values()
            if "mean_raw_rmse_by_h" in summary
            for k in summary["mean_raw_rmse_by_h"].keys()
        }
    )
    for mode, label, _ in CASE_CONFIGS:
        summary = case_summaries[mode]
        if "mean_raw_rmse_by_h" not in summary:
            continue
        cur_h = [h for h in h_values if f"{h:.1f}" in summary["mean_raw_rmse_by_h"]]
        vals = [summary["mean_raw_rmse_by_h"][f"{h:.1f}"] for h in cur_h]
        ax.plot(cur_h, vals, "o-", lw=1.8, label=label)
    ax.set_xlabel(r"$H_*$")
    ax.set_ylabel("mean raw rel_rmse")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_r_by_vw(case_summaries: dict, outpath: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    vw_values = [0.3, 0.5, 0.7, 0.9]
    for mode, label, _ in CASE_CONFIGS:
        vals = [case_summaries[mode]["r_by_vw"][f"{vw:.1f}"] for vw in vw_values]
        ax.plot(vw_values, vals, "o-", lw=1.8, label=label)
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$r(v_w)$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_tc_by_vw(case_summaries: dict, outpath: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    vw_values = [0.3, 0.5, 0.7, 0.9]
    summary = case_summaries["allvw_c1_rvw_tcvw"]
    vals = [summary["tc_by_vw"][f"{vw:.1f}"] for vw in vw_values]
    ax.plot(vw_values, vals, "o-", lw=1.8, label=CASE_CONFIGS[0][1])
    summary_mid = case_summaries["allvw_c1_rlinear_tclinear"]
    vals_mid = [summary_mid["tc_by_vw"][f"{vw:.1f}"] for vw in vw_values]
    ax.plot(vw_values, vals_mid, "o-", lw=1.8, label=CASE_CONFIGS[1][1])
    summary2 = case_summaries["allvw_c1_rlinear_tcconst"]
    tc_star = float(summary2["tc_star"])
    ax.plot(vw_values, [tc_star] * len(vw_values), "o--", lw=1.8, label=CASE_CONFIGS[2][1])
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$t_c$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def make_comparison_sheet(case_dirs: dict[str, Path], outpath: Path, dpi: int):
    items = []
    for mode, label, _ in CASE_CONFIGS:
        subdir = case_dirs[mode]
        items.append((f"{label}: collapse", subdir / "collapse_overlay.png"))
        items.append((f"{label}: raw H*=2.0", subdir / "xi_vs_betaH_H2p0.png"))
    fig, axes = plt.subplots(len(CASE_CONFIGS), 2, figsize=(16, 6 * len(CASE_CONFIGS)))
    for ax, (title, path) in zip(axes.ravel(), items):
        img = imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.suptitle(r"Minimal broken-power timing models", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def copy_case(src_root: Path, src_slug: str, outdir: Path, dst_slug: str):
    src = src_root / src_slug
    dst = outdir / dst_slug
    if not src.exists():
        raise FileNotFoundError(f"Missing case directory: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return dst


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        reference_dir = Path(args.reference_dir).resolve()
        ref = json.loads((reference_dir / "final_summary.json").read_text())
        ref_case = ref["cases"]["allvw_c1_rvw_h0gammavw"]

        df = quad.load_lattice_dataframe(outdir, args.vw_folders, args.h_values)
        theta_values = np.sort(df["theta"].unique())
        vw_values = np.sort(df["v_w"].unique())
        h_values = np.sort(df["H"].unique())
        ode = shared.load_ode_amplitudes(Path(args.ode_summary).resolve(), theta_values)
        meta = build_meta(df, theta_values, vw_values, ode, args.t_osc)

        init_r_by_vw = parse_r_by_vw(ref_case, vw_values)
        init_tc_by_vw = parse_tc_mean_by_vw(ref_case, theta_values, vw_values)
        init_tc_star = float(np.mean(init_tc_by_vw))

        fit_tcvw = fit_case_tcvw(meta, args.fixed_beta, init_tc_by_vw, init_r_by_vw)
        fit_tclinear = fit_case_rlinear_tclinear(meta, args.fixed_beta, init_tc_by_vw, fit_tcvw["r_by_vw"])
        fit_linear = fit_case_rlinear_tcconst(meta, args.fixed_beta, init_tc_star, fit_tcvw["r_by_vw"])

        case_payloads = {
            "allvw_c1_rvw_tcvw": fit_tcvw,
            "allvw_c1_rlinear_tclinear": fit_tclinear,
            "allvw_c1_rlinear_tcconst": fit_linear,
        }
        case_summaries = {}
        case_dirs = {}

        for mode, label, slug in CASE_CONFIGS[:3]:
            subdir = outdir / slug
            subdir.mkdir(parents=True, exist_ok=True)
            case_dirs[mode] = subdir
            payload = case_payloads[mode]
            pred = build_prediction_frame(df, meta, payload)
            pred.to_csv(subdir / "predictions.csv", index=False)
            shared_rvw.plot_collapse_overlay(pred, theta_values, vw_values, payload, subdir / "collapse_overlay.png", args.dpi, args.fixed_beta, label)
            shared_rvw.plot_raw(pred, theta_values, vw_values, subdir, "xi_vs_betaH", args.dpi)
            plot_xi_vs_u_by_vw(pred, vw_values, payload, subdir / "xi_vs_u_by_vw.png", args.dpi)
            raw_rows, mean_by_h, mean_by_h_vw = rmse_tables(pred)
            summary = summarize_payload(payload, vw_values)
            summary["label"] = label
            summary["mean_raw_rmse_by_h"] = mean_by_h
            summary["mean_raw_rmse_by_h_vw"] = mean_by_h_vw
            summary["raw_plot_rmse"] = raw_rows
            case_summaries[mode] = summary
            save_json(subdir / "final_summary.json", summary)

        ref_summary = ref_case.copy()
        ref_summary["label"] = CASE_CONFIGS[3][1]
        case_summaries["allvw_c1_rvw_h0gammavw_ref"] = ref_summary
        case_dirs["allvw_c1_rvw_h0gammavw_ref"] = copy_case(reference_dir, "allvw_c1_rvw_h0gammavw", outdir, "allvw_c1_rvw_h0gammavw_ref")

        plot_rmse_by_h(case_summaries, outdir / "rmse_by_h.png", args.dpi)
        plot_r_by_vw(case_summaries, outdir / "r_by_vw.png", args.dpi)
        plot_tc_by_vw(case_summaries, outdir / "tc_by_vw.png", args.dpi)
        make_comparison_sheet(case_dirs, outdir / "comparison_sheet.png", args.dpi)

        summary = {
            "status": "ok",
            "vw_values": [float(v) for v in vw_values],
            "theta_values": [float(v) for v in theta_values],
            "h_values": [float(v) for v in h_values],
            "fixed_beta": float(args.fixed_beta),
            "n_points": int(len(df)),
            "ode_amplitude_source": ode["source"],
            "reference_dir": str(reference_dir),
            "cases": case_summaries,
            "outputs": {
                "comparison_sheet": str(outdir / "comparison_sheet.png"),
                "rmse_by_h": str(outdir / "rmse_by_h.png"),
                "r_by_vw": str(outdir / "r_by_vw.png"),
                "tc_by_vw": str(outdir / "tc_by_vw.png"),
            },
        }
        save_json(outdir / "final_summary.json", summary)
        print(json.dumps(shared.to_native(summary), indent=2, sort_keys=True))
    except Exception as exc:
        error_exit(outdir, exc)
        raise


if __name__ == "__main__":
    main()
