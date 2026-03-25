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
OUTDIR_DEFAULT = ROOT / "results_lattice_broken_powerlaw_knob_h0_beta0"
REFERENCE_DIR_DEFAULT = ROOT / "results_lattice_broken_powerlaw_beta0"
REFERENCE_TIMING_DIR_DEFAULT = ROOT / "knob_tc_h0_weibull"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]

CASE_CONFIGS = [
    ("allvw_c1_rvw", r"broken power, $C=1$, $r(v_w)$", "allvw_c1_rvw"),
    ("allvw_c1_rvw_h0gammavw", r"broken power, $C=1$, $r(v_w)$, $t_c^{(0)}(v_w)(h+h_0)^{\gamma(v_w)}$", "allvw_c1_rvw_h0gammavw"),
    ("allvw_cvw_rvw_ref", r"broken power ref, $C(v_w)$, $r(v_w)$", "allvw_cvw_rvw_ref"),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Test a shifted-h timing knob inside the broken-power xi ansatz. C(v_w) is fixed to 1 in the timed model because it is exactly degenerate with tc0(v_w)."
    )
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--fixed-beta", type=float, default=0.0)
    p.add_argument("--free-beta", action="store_true")
    p.add_argument("--beta-min", type=float, default=-0.5)
    p.add_argument("--beta-max", type=float, default=0.5)
    p.add_argument("--t-osc", type=float, default=1.5)
    p.add_argument("--ode-summary", type=str, default=str(shared.ODE_SUMMARY_DEFAULT))
    p.add_argument("--reference-dir", type=str, default=str(REFERENCE_DIR_DEFAULT))
    p.add_argument("--reference-timing-dir", type=str, default=str(REFERENCE_TIMING_DIR_DEFAULT))
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


def h_theta(theta):
    theta = np.asarray(theta, dtype=np.float64)
    cos_half = np.cos(theta / 2.0)
    return np.log(np.e / np.maximum(cos_half * cos_half, 1.0e-300))


def xi_broken_power(u: np.ndarray, r: np.ndarray) -> np.ndarray:
    u = np.maximum(np.asarray(u, dtype=np.float64), 1.0e-18)
    r = np.maximum(np.asarray(r, dtype=np.float64), 1.0e-12)
    z = r * np.log(u)
    log_xi = np.where(z > 50.0, z / r, np.log1p(np.exp(np.clip(z, -700.0, 700.0))) / r)
    return np.exp(log_xi)


def parse_r_by_vw(summary: dict, vw_values: np.ndarray):
    table = summary["r_by_vw"]
    return np.asarray([float(table[f"{float(vw):.1f}"]) for vw in vw_values], dtype=np.float64)


def parse_tc0_gamma_h0(summary: dict, vw_values: np.ndarray):
    tc0 = np.asarray([float(summary["tc0_by_vw"][f"{float(vw):.1f}"]) for vw in vw_values], dtype=np.float64)
    gamma = np.asarray([float(summary["gamma_by_vw"][f"{float(vw):.1f}"]) for vw in vw_values], dtype=np.float64)
    h0 = float(summary["h0"])
    return tc0, gamma, h0


def build_meta(df, theta_values: np.ndarray, vw_values: np.ndarray, ode: dict, t_osc: float):
    meta = shared_rvw.build_meta(df, theta_values, vw_values, ode, t_osc)
    meta["h_theta_values"] = h_theta(theta_values)
    meta["h_theta_row"] = meta["h_theta_values"][np.asarray(meta["theta_idx"], dtype=np.int64)]
    return meta


def unpack_rvw_only(params: np.ndarray, n_vw: int):
    return np.asarray(params[:n_vw], dtype=np.float64)


def unpack_h0_knob(params: np.ndarray, n_vw: int):
    idx = 0
    r_by_vw = np.asarray(params[idx : idx + n_vw], dtype=np.float64)
    idx += n_vw
    tc0_by_vw = np.asarray(params[idx : idx + n_vw], dtype=np.float64)
    idx += n_vw
    gamma_by_vw = np.asarray(params[idx : idx + n_vw], dtype=np.float64)
    idx += n_vw
    h0 = float(params[idx])
    return r_by_vw, tc0_by_vw, gamma_by_vw, h0


def unpack_rvw_only_with_beta(params: np.ndarray, n_vw: int, fixed_beta: float, free_beta: bool):
    idx = 0
    beta = float(fixed_beta)
    if free_beta:
        beta = float(params[idx])
        idx += 1
    r_by_vw = np.asarray(params[idx : idx + n_vw], dtype=np.float64)
    return beta, r_by_vw


def unpack_h0_knob_with_beta(params: np.ndarray, n_vw: int, fixed_beta: float, free_beta: bool):
    idx = 0
    beta = float(fixed_beta)
    if free_beta:
        beta = float(params[idx])
        idx += 1
    r_by_vw = np.asarray(params[idx : idx + n_vw], dtype=np.float64)
    idx += n_vw
    tc0_by_vw = np.asarray(params[idx : idx + n_vw], dtype=np.float64)
    idx += n_vw
    gamma_by_vw = np.asarray(params[idx : idx + n_vw], dtype=np.float64)
    idx += n_vw
    h0 = float(params[idx])
    return beta, r_by_vw, tc0_by_vw, gamma_by_vw, h0


def fit_rvw_only(meta, fixed_beta: float, init_r_by_vw: np.ndarray, free_beta: bool, beta_min: float, beta_max: float):
    n_vw = len(meta["vw_values"])
    x0_parts = []
    lower_parts = []
    upper_parts = []
    if free_beta:
        x0_parts.append(np.asarray([float(fixed_beta)], dtype=np.float64))
        lower_parts.append(np.asarray([float(beta_min)], dtype=np.float64))
        upper_parts.append(np.asarray([float(beta_max)], dtype=np.float64))
    x0_parts.append(np.asarray(init_r_by_vw, dtype=np.float64))
    lower_parts.append(np.full(n_vw, 0.1, dtype=np.float64))
    upper_parts.append(np.full(n_vw, 30.0, dtype=np.float64))
    x0 = np.concatenate(x0_parts)
    lower = np.concatenate(lower_parts)
    upper = np.concatenate(upper_parts)

    def resid(par: np.ndarray) -> np.ndarray:
        beta, r_by_vw = unpack_rvw_only_with_beta(par, n_vw, fixed_beta, free_beta)
        x, xi_scale = shared_rvw.x_and_xi_scale(meta, beta)
        r_eff = r_by_vw[meta["vw_idx"]]
        u = meta["F_inf"] * xi_scale / np.maximum(meta["F0_sq"], 1.0e-18)
        xi_fit = xi_broken_power(u, r_eff)
        return (xi_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)

    huber = least_squares(resid, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=60000)
    final = least_squares(resid, huber.x, bounds=(lower, upper), loss="linear", max_nfev=60000)
    beta_fit, r_by_vw = unpack_rvw_only_with_beta(final.x, n_vw, fixed_beta, free_beta)
    x, xi_scale = shared_rvw.x_and_xi_scale(meta, beta_fit)
    r_eff = r_by_vw[meta["vw_idx"]]
    u = meta["F_inf"] * xi_scale / np.maximum(meta["F0_sq"], 1.0e-18)
    xi_fit = xi_broken_power(u, r_eff)
    frac_resid = (xi_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    aic, bic = shared.aic_bic(frac_resid, len(final.x))
    return {
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "mode": "allvw_c1_rvw",
        "beta": float(beta_fit),
        "canonical_xi_form": "xi = [1 + u^r]^(1/r)",
        "xi_model_form": "xi = [1 + (F_inf(theta0) (2 x / (3 t_osc))^(3/2) / F0(theta0)^2)^(r(v_w)) ]^(1/r(v_w))",
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
        "r_by_vw": np.asarray(r_by_vw, dtype=np.float64),
        "c_by_vw": np.ones(n_vw, dtype=np.float64),
        "per_vw_rel_rmse": {
            f"{float(vw):.1f}": shared_rvw.rel_rmse(meta["xi"][np.isclose(meta["v_w"], float(vw), atol=1.0e-12)], xi_fit[np.isclose(meta["v_w"], float(vw), atol=1.0e-12)])
            for vw in meta["vw_values"]
        },
    }


def fit_h0_knob(
    meta,
    fixed_beta: float,
    init_beta: float,
    init_r_by_vw: np.ndarray,
    init_tc0_by_vw: np.ndarray,
    init_gamma_by_vw: np.ndarray,
    init_h0: float,
    free_beta: bool,
    beta_min: float,
    beta_max: float,
):
    n_vw = len(meta["vw_values"])
    x0_parts = []
    lower_parts = []
    upper_parts = []
    if free_beta:
        x0_parts.append(np.asarray([float(init_beta)], dtype=np.float64))
        lower_parts.append(np.asarray([float(beta_min)], dtype=np.float64))
        upper_parts.append(np.asarray([float(beta_max)], dtype=np.float64))
    x0_parts.extend(
        [
            np.asarray(init_r_by_vw, dtype=np.float64),
            np.asarray(init_tc0_by_vw, dtype=np.float64),
            np.asarray(init_gamma_by_vw, dtype=np.float64),
            np.asarray([float(init_h0)], dtype=np.float64),
        ]
    )
    lower_parts.extend(
        [
            np.full(n_vw, 0.1, dtype=np.float64),
            np.full(n_vw, 0.01, dtype=np.float64),
            np.full(n_vw, -5.0, dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
        ]
    )
    upper_parts.extend(
        [
            np.full(n_vw, 30.0, dtype=np.float64),
            np.full(n_vw, 300.0, dtype=np.float64),
            np.full(n_vw, 5.0, dtype=np.float64),
            np.asarray([50.0], dtype=np.float64),
        ]
    )
    x0 = np.concatenate(x0_parts)
    lower = np.concatenate(lower_parts)
    upper = np.concatenate(upper_parts)

    def resid(par: np.ndarray) -> np.ndarray:
        beta, r_by_vw, tc0_by_vw, gamma_by_vw, h0 = unpack_h0_knob_with_beta(par, n_vw, fixed_beta, free_beta)
        x, xi_scale = shared_rvw.x_and_xi_scale(meta, beta)
        r_eff = r_by_vw[meta["vw_idx"]]
        gamma_eff = gamma_by_vw[meta["vw_idx"]]
        tc_eff = tc0_by_vw[meta["vw_idx"]] * np.power(np.maximum(meta["h_theta_row"] + float(h0), 1.0e-18), gamma_eff)
        u = meta["F_inf"] * xi_scale / np.maximum(meta["F0_sq"] * np.power(tc_eff, 1.5), 1.0e-18)
        xi_fit = xi_broken_power(u, r_eff)
        return (xi_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)

    huber = least_squares(resid, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=90000)
    final = least_squares(resid, huber.x, bounds=(lower, upper), loss="linear", max_nfev=90000)
    beta_fit, r_by_vw, tc0_by_vw, gamma_by_vw, h0 = unpack_h0_knob_with_beta(final.x, n_vw, fixed_beta, free_beta)
    x, xi_scale = shared_rvw.x_and_xi_scale(meta, beta_fit)
    r_eff = r_by_vw[meta["vw_idx"]]
    gamma_eff = gamma_by_vw[meta["vw_idx"]]
    tc_eff = tc0_by_vw[meta["vw_idx"]] * np.power(np.maximum(meta["h_theta_row"] + float(h0), 1.0e-18), gamma_eff)
    u = meta["F_inf"] * xi_scale / np.maximum(meta["F0_sq"] * np.power(tc_eff, 1.5), 1.0e-18)
    xi_fit = xi_broken_power(u, r_eff)
    frac_resid = (xi_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    aic, bic = shared.aic_bic(frac_resid, len(final.x))
    tc_by_vw_theta = tc0_by_vw[:, None] * np.power(np.maximum(meta["h_theta_values"][None, :] + float(h0), 1.0e-18), gamma_by_vw[:, None])
    return {
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "mode": "allvw_c1_rvw_h0gammavw",
        "beta": float(beta_fit),
        "canonical_xi_form": "xi = [1 + u^r]^(1/r)",
        "time_knob_form": "tc(theta0,v_w) = tc0(v_w) * (h(theta0) + h0)^(gamma(v_w))",
        "xi_model_form": "xi = [1 + (F_inf(theta0) (2 x / (3 t_osc tc(theta0,v_w)))^(3/2) / F0(theta0)^2)^(r(v_w)) ]^(1/r(v_w))",
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
        "r_by_vw": np.asarray(r_by_vw, dtype=np.float64),
        "c_by_vw": np.ones(n_vw, dtype=np.float64),
        "tc0_by_vw": np.asarray(tc0_by_vw, dtype=np.float64),
        "gamma_by_vw": np.asarray(gamma_by_vw, dtype=np.float64),
        "h0": float(h0),
        "tc_by_vw_theta": np.asarray(tc_by_vw_theta, dtype=np.float64),
        "per_vw_rel_rmse": {
            f"{float(vw):.1f}": shared_rvw.rel_rmse(meta["xi"][np.isclose(meta["v_w"], float(vw), atol=1.0e-12)], xi_fit[np.isclose(meta["v_w"], float(vw), atol=1.0e-12)])
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


def summarize_payload(payload: dict, theta_values: np.ndarray, vw_values: np.ndarray):
    out = {
        k: v
        for k, v in payload.items()
        if k
        not in {
            "result_x",
            "x_fit",
            "u_fit",
            "y_fit",
            "f_tilde_fit",
            "frac_resid",
            "r_by_vw",
            "c_by_vw",
            "tc0_by_vw",
            "gamma_by_vw",
            "tc_by_vw_theta",
        }
    }
    out["r_by_vw"] = {f"{float(vw):.1f}": float(payload["r_by_vw"][i]) for i, vw in enumerate(vw_values)}
    out["c_by_vw"] = {f"{float(vw):.1f}": float(payload["c_by_vw"][i]) for i, vw in enumerate(vw_values)}
    if "tc0_by_vw" in payload:
        out["tc0_by_vw"] = {f"{float(vw):.1f}": float(payload["tc0_by_vw"][i]) for i, vw in enumerate(vw_values)}
    if "gamma_by_vw" in payload:
        out["gamma_by_vw"] = {f"{float(vw):.1f}": float(payload["gamma_by_vw"][i]) for i, vw in enumerate(vw_values)}
    if "tc_by_vw_theta" in payload:
        out["tc_by_vw_theta"] = {}
        for i, vw in enumerate(vw_values):
            out["tc_by_vw_theta"][f"{float(vw):.1f}"] = {
                f"{float(theta):.10f}": float(payload["tc_by_vw_theta"][i, j]) for j, theta in enumerate(theta_values)
            }
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
    vw_values = sorted(
        {
            float(vw)
            for summary in case_summaries.values()
            if "r_by_vw" in summary
            for vw in summary["r_by_vw"].keys()
        }
    )
    for mode, label, _ in CASE_CONFIGS:
        summary = case_summaries[mode]
        if "r_by_vw" not in summary:
            continue
        cur_vw = [vw for vw in vw_values if f"{vw:.1f}" in summary["r_by_vw"]]
        vals = [summary["r_by_vw"][f"{vw:.1f}"] for vw in cur_vw]
        ax.plot(cur_vw, vals, "o-", lw=1.8, label=label)
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$r(v_w)$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_gamma_by_vw(case_summaries: dict, outpath: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    summary = case_summaries.get("allvw_c1_rvw_h0gammavw", {})
    if "gamma_by_vw" in summary:
        vw_values = sorted(float(vw) for vw in summary["gamma_by_vw"].keys())
        vals = [summary["gamma_by_vw"][f"{vw:.1f}"] for vw in vw_values]
        ax.plot(vw_values, vals, "o-", lw=1.8, color="tab:purple")
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$\gamma(v_w)$")
    ax.grid(alpha=0.25)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_tc0_by_vw(case_summaries: dict, outpath: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    summary = case_summaries.get("allvw_c1_rvw_h0gammavw", {})
    if "tc0_by_vw" in summary:
        vw_values = sorted(float(vw) for vw in summary["tc0_by_vw"].keys())
        vals = [summary["tc0_by_vw"][f"{vw:.1f}"] for vw in vw_values]
        ax.plot(vw_values, vals, "o-", lw=1.8, color="tab:blue")
        ax.set_title(rf"$h_0={summary.get('h0', float('nan')):.3f}$")
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$t_c^{(0)}(v_w)$")
    ax.grid(alpha=0.25)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_c_by_vw(case_summaries: dict, outpath: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    vw_values = sorted(
        {
            float(vw)
            for summary in case_summaries.values()
            if "c_by_vw" in summary
            for vw in summary["c_by_vw"].keys()
        }
    )
    for mode, label, _ in CASE_CONFIGS:
        summary = case_summaries[mode]
        if "c_by_vw" not in summary:
            continue
        cur_vw = [vw for vw in vw_values if f"{vw:.1f}" in summary["c_by_vw"]]
        vals = [summary["c_by_vw"][f"{vw:.1f}"] for vw in cur_vw]
        ax.plot(cur_vw, vals, "o-", lw=1.8, label=label)
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$C(v_w)$")
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
    fig.suptitle(r"Broken-power timing-knob comparison", fontsize=16)
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
        reference_timing_dir = Path(args.reference_timing_dir).resolve()
        ref_broken = json.loads((reference_dir / "final_summary.json").read_text())
        ref_timing = json.loads((reference_timing_dir / "final_summary.json").read_text())

        df = quad.load_lattice_dataframe(outdir, args.vw_folders, args.h_values)
        theta_values = np.sort(df["theta"].unique())
        vw_values = np.sort(df["v_w"].unique())
        h_values = np.sort(df["H"].unique())
        ode = shared.load_ode_amplitudes(Path(args.ode_summary).resolve(), theta_values)
        meta = build_meta(df, theta_values, vw_values, ode, args.t_osc)

        ref_broken_best = ref_broken["cases"]["allvw_cvw_rvw"]
        init_r_by_vw = parse_r_by_vw(ref_broken_best, vw_values)
        init_tc0_by_vw, init_gamma_by_vw, init_h0 = parse_tc0_gamma_h0(ref_timing["cases"]["rvw_theta_h0_gammavw"], vw_values)

        fit_c1 = fit_rvw_only(meta, args.fixed_beta, init_r_by_vw, args.free_beta, args.beta_min, args.beta_max)
        fit_knob = fit_h0_knob(
            meta,
            args.fixed_beta,
            fit_c1["beta"],
            fit_c1["r_by_vw"],
            init_tc0_by_vw,
            init_gamma_by_vw,
            init_h0,
            args.free_beta,
            args.beta_min,
            args.beta_max,
        )

        case_payloads = {
            "allvw_c1_rvw": fit_c1,
            "allvw_c1_rvw_h0gammavw": fit_knob,
        }
        case_summaries = {}
        case_dirs = {}

        for mode, label, slug in CASE_CONFIGS[:2]:
            subdir = outdir / slug
            subdir.mkdir(parents=True, exist_ok=True)
            case_dirs[mode] = subdir
            payload = case_payloads[mode]
            pred = build_prediction_frame(df, meta, payload)
            pred.to_csv(subdir / "predictions.csv", index=False)
            shared_rvw.plot_collapse_overlay(pred, theta_values, vw_values, payload, subdir / "collapse_overlay.png", args.dpi, payload["beta"], label)
            shared_rvw.plot_raw(pred, theta_values, vw_values, subdir, "xi_vs_betaH", args.dpi)
            plot_xi_vs_u_by_vw(pred, vw_values, payload, subdir / "xi_vs_u_by_vw.png", args.dpi)
            raw_rows, mean_by_h, mean_by_h_vw = rmse_tables(pred)
            summary = summarize_payload(payload, theta_values, vw_values)
            summary["label"] = label
            summary["mean_raw_rmse_by_h"] = mean_by_h
            summary["mean_raw_rmse_by_h_vw"] = mean_by_h_vw
            summary["raw_plot_rmse"] = raw_rows
            case_summaries[mode] = summary
            save_json(subdir / "final_summary.json", summary)

        ref_summary = ref_broken_best.copy()
        ref_summary["label"] = CASE_CONFIGS[2][1]
        case_summaries["allvw_cvw_rvw_ref"] = ref_summary
        case_dirs["allvw_cvw_rvw_ref"] = copy_case(reference_dir, "allvw_cvw_rvw", outdir, "allvw_cvw_rvw_ref")

        plot_rmse_by_h(case_summaries, outdir / "rmse_by_h.png", args.dpi)
        plot_r_by_vw(case_summaries, outdir / "r_by_vw.png", args.dpi)
        plot_gamma_by_vw(case_summaries, outdir / "gamma_by_vw.png", args.dpi)
        plot_tc0_by_vw(case_summaries, outdir / "tc0_by_vw.png", args.dpi)
        plot_c_by_vw(case_summaries, outdir / "c_by_vw.png", args.dpi)
        make_comparison_sheet(case_dirs, outdir / "comparison_sheet.png", args.dpi)

        summary = {
            "status": "ok",
            "vw_values": [float(v) for v in vw_values],
            "theta_values": [float(v) for v in theta_values],
            "h_values": [float(v) for v in h_values],
            "fixed_beta": float(args.fixed_beta),
            "free_beta": bool(args.free_beta),
            "n_points": int(len(df)),
            "ode_amplitude_source": ode["source"],
            "reference_dir": str(reference_dir),
            "reference_timing_dir": str(reference_timing_dir),
            "note": "In the timed broken-power model, C(v_w) is fixed to 1 because C(v_w) and tc0(v_w) are exactly degenerate through the combination C / tc0^(3/2).",
            "cases": case_summaries,
            "outputs": {
                "comparison_sheet": str(outdir / "comparison_sheet.png"),
                "rmse_by_h": str(outdir / "rmse_by_h.png"),
                "r_by_vw": str(outdir / "r_by_vw.png"),
                "gamma_by_vw": str(outdir / "gamma_by_vw.png"),
                "tc0_by_vw": str(outdir / "tc0_by_vw.png"),
                "c_by_vw": str(outdir / "c_by_vw.png"),
            },
        }
        save_json(outdir / "final_summary.json", summary)
        print(json.dumps(shared.to_native(summary), indent=2, sort_keys=True))
    except Exception as exc:
        error_exit(outdir, exc)
        raise


if __name__ == "__main__":
    main()
