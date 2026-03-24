#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from scipy.optimize import least_squares

import fit_lattice_fixed_ode_amplitudes as shared
import fit_lattice_fixed_ode_amplitudes_theta_tc as theta_tc
import fit_lattice_fixed_ode_amplitudes_theta_tc_shared_r_by_vw as shared_rvw
import fit_lattice_quadwarp_universal as quad


ROOT = Path(__file__).resolve().parent
OUTDIR_DEFAULT = "results_lattice_theta_tc_tosc_and_s_tests_beta0_tcmax300"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]
REFERENCE_SUMMARY_DEFAULT = (
    ROOT / "results_lattice_fixed_ode_amplitudes_theta_tc_sharedr_by_vw_beta0_tcmax300" / "final_summary.json"
)

CASE_CONFIGS = [
    ("calib_fixed_tosc", r"ODE-fixed $f_\infty$, $c(v_w)$, fixed $t_{\rm osc}$", "calib_fixed_tosc"),
    ("calib_free_tosc", r"ODE-fixed $f_\infty$, $c(v_w)$, free $t_{\rm osc}$", "calib_free_tosc"),
    ("s_alltp", r"$s(v_w,\theta_0)$ applied to all $t_p$", "s_alltp"),
]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compare canonical shared-r fits for lattice data using ODE-fixed F0,f_infty with "
            "xi = (2 x / (3 t_osc))^(3/2) * f_tilde / F0: "
            "(1) calibrated baseline with fixed t_osc, "
            "(2) calibrated fit with global free t_osc, and "
            "(3) a time-rescaling variant where s(v_w,theta0) multiplies every t_p entering the ansatz."
        )
    )
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--fixed-beta", type=float, default=0.0)
    p.add_argument("--t-osc", type=float, default=1.5)
    p.add_argument("--ode-summary", type=str, default=str(shared.ODE_SUMMARY_DEFAULT))
    p.add_argument("--reference-summary", type=str, default=str(REFERENCE_SUMMARY_DEFAULT))
    p.add_argument("--tc-max", type=float, default=300.0)
    p.add_argument("--tosc-min", type=float, default=0.2)
    p.add_argument("--tosc-max", type=float, default=10.0)
    p.add_argument("--s-min", type=float, default=1.0e-4)
    p.add_argument("--s-max", type=float, default=10.0)
    p.add_argument(
        "--transient-kernel",
        type=str,
        choices=["power_inside", "shifted"],
        default="power_inside",
        help="Use 1 + u^r (`power_inside`) or (1 + u)^r (`shifted`) in the transient denominator.",
    )
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--outdir", type=str, default=OUTDIR_DEFAULT)
    return p.parse_args()


def error_exit(outdir: Path, exc: Exception):
    payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
    outdir.mkdir(parents=True, exist_ok=True)
    shared.save_json(outdir / "_error.json", payload)
    print(json.dumps(payload, sort_keys=True))


def load_json_if_exists(path: Path | None):
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def init_from_reference(summary: dict | None, theta_values: np.ndarray, vw_values: np.ndarray):
    n_vw = len(vw_values)
    n_theta = len(theta_values)
    init_tc = np.full((n_vw, n_theta), 2.0, dtype=np.float64)
    init_c = np.ones(n_vw, dtype=np.float64)
    init_r = 2.0
    if summary is None:
        return init_r, init_tc, init_c
    payload = summary.get("shared_r_calib", {})
    init_r = float(payload.get("r", init_r))
    c_map = payload.get("c_calib_by_vw", {})
    for i, vw in enumerate(vw_values):
        key = f"{float(vw):.1f}"
        if key in c_map:
            init_c[i] = float(c_map[key])
    tc_map = payload.get("tc_by_vw_theta", {})
    for i, vw in enumerate(vw_values):
        vw_key = f"{float(vw):.1f}"
        row = tc_map.get(vw_key, {})
        for j, theta in enumerate(theta_values):
            theta_key = f"{float(theta):.10f}"
            if theta_key in row:
                init_tc[i, j] = float(row[theta_key])
    return init_r, init_tc, init_c


def unpack_params(params: np.ndarray, mode: str, n_vw: int, n_theta: int):
    idx = 0
    r = float(params[idx])
    idx += 1
    grid = np.asarray(params[idx : idx + n_vw * n_theta], dtype=np.float64).reshape(n_vw, n_theta)
    idx += n_vw * n_theta
    c_vw = np.asarray(params[idx : idx + n_vw], dtype=np.float64)
    idx += n_vw
    extra = {}
    if mode == "calib_free_tosc":
        extra["t_osc"] = float(params[idx])
    return r, grid, c_vw, extra


def f_infty_ode_pointwise(meta):
    return meta["F_inf"] / np.maximum(meta["F0"], 1.0e-18)


def model_details(meta, fixed_beta: float, params: np.ndarray, kernel: str, mode: str):
    n_vw = len(meta["vw_values"])
    n_theta = len(meta["theta_values"])
    r, grid, c_vw, extra = unpack_params(params, mode, n_vw, n_theta)
    x_raw = meta["tp"] * np.power(meta["H"], float(fixed_beta))
    f_infty = f_infty_ode_pointwise(meta)
    c_point = c_vw[meta["vw_idx"]]

    if mode in {"calib_fixed_tosc", "calib_free_tosc"}:
        t_osc = float(meta["t_osc"] if mode == "calib_fixed_tosc" else extra["t_osc"])
        tc = grid[meta["vw_idx"], meta["theta_idx"]]
        xi_scale = shared_rvw.xi_scale_from_x(x_raw, t_osc)
        denom = shared_rvw.transition_denom(x_raw, tc, r, kernel)
        f_tilde_base = f_infty + meta["F0"] / np.maximum(xi_scale * denom, 1.0e-18)
        f_tilde = c_point * f_tilde_base
        xi_fit = shared_rvw.xi_from_f_tilde(xi_scale, f_tilde, meta["F0"])
        return {
            "r": r,
            "grid": grid,
            "c_vw": c_vw,
            "x_raw": x_raw,
            "x_plot": x_raw,
            "xi_scale": xi_scale,
            "f_tilde": f_tilde,
            "xi_fit": xi_fit,
            "t_osc": t_osc,
        }

    if mode == "s_alltp":
        # Here s(v_w,theta0) is treated as a direct time-rescaling factor and
        # multiplies every tp entering the canonical ansatz.
        s_grid = grid
        s_point = s_grid[meta["vw_idx"], meta["theta_idx"]]
        x_eff = s_point * x_raw
        xi_scale = shared_rvw.xi_scale_from_x(x_eff, meta["t_osc"])
        denom = shared_rvw.transition_denom(x_eff, np.ones_like(x_eff), r, kernel)
        f_tilde_base = f_infty + meta["F0"] / np.maximum(xi_scale * denom, 1.0e-18)
        f_tilde = c_point * f_tilde_base
        xi_fit = shared_rvw.xi_from_f_tilde(xi_scale, f_tilde, meta["F0"])
        return {
            "r": r,
            "grid": s_grid,
            "c_vw": c_vw,
            "x_raw": x_raw,
            "x_plot": x_eff,
            "xi_scale": xi_scale,
            "f_tilde": f_tilde,
            "xi_fit": xi_fit,
            "t_osc": float(meta["t_osc"]),
        }

    raise ValueError(f"Unknown mode: {mode}")


def fit_case(meta, fixed_beta: float, init_r: float, init_tc: np.ndarray, init_c: np.ndarray, args, mode: str):
    n_vw = len(meta["vw_values"])
    n_theta = len(meta["theta_values"])
    if np.asarray(init_tc).shape != (n_vw, n_theta):
        raise ValueError("Initial tc grid must match (n_vw, n_theta).")
    if np.asarray(init_c).shape != (n_vw,):
        raise ValueError("Initial c(v_w) vector must match n_vw.")

    if mode == "s_alltp":
        init_grid = 1.0 / np.maximum(np.asarray(init_tc, dtype=np.float64), 1.0e-18)
        lower_grid = np.full(n_vw * n_theta, float(args.s_min), dtype=np.float64)
        upper_grid = np.full(n_vw * n_theta, float(args.s_max), dtype=np.float64)
    else:
        init_grid = np.asarray(init_tc, dtype=np.float64)
        lower_grid = np.full(n_vw * n_theta, 0.1, dtype=np.float64)
        upper_grid = np.full(n_vw * n_theta, float(args.tc_max), dtype=np.float64)

    x0 = np.concatenate(
        [
            np.array([float(init_r)], dtype=np.float64),
            np.asarray(init_grid, dtype=np.float64).ravel(),
            np.asarray(init_c, dtype=np.float64),
        ]
    )
    lower = np.concatenate(
        [
            np.array([0.1], dtype=np.float64),
            lower_grid,
            np.full(n_vw, 0.1, dtype=np.float64),
        ]
    )
    upper = np.concatenate(
        [
            np.array([20.0], dtype=np.float64),
            upper_grid,
            np.full(n_vw, 10.0, dtype=np.float64),
        ]
    )
    if mode == "calib_free_tosc":
        x0 = np.concatenate([x0, np.array([float(meta["t_osc"])], dtype=np.float64)])
        lower = np.concatenate([lower, np.array([float(args.tosc_min)], dtype=np.float64)])
        upper = np.concatenate([upper, np.array([float(args.tosc_max)], dtype=np.float64)])

    def resid(par: np.ndarray) -> np.ndarray:
        details = model_details(meta, fixed_beta, par, args.transient_kernel, mode)
        return (details["xi_fit"] - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)

    huber = least_squares(resid, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=50000)
    final = least_squares(resid, huber.x, bounds=(lower, upper), loss="linear", max_nfev=50000)
    details = model_details(meta, fixed_beta, final.x, args.transient_kernel, mode)
    frac_resid = (details["xi_fit"] - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    aic, bic = shared.aic_bic(frac_resid, len(final.x))
    payload = {
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "mode": mode,
        "beta": float(fixed_beta),
        "transient_kernel": str(args.transient_kernel),
        "canonical_xi_form": "xi = (2 x / (3 t_osc))^(3/2) * f_tilde / F0",
        "transition_denominator": shared_rvw.kernel_denom_text(args.transient_kernel),
        "r": float(details["r"]),
        "rel_rmse": shared_rvw.rel_rmse(meta["xi"], details["xi_fit"]),
        "AIC": float(aic),
        "BIC": float(bic),
        "n_points": int(meta["xi"].size),
        "n_params": int(len(final.x)),
        "result_x": np.asarray(final.x, dtype=np.float64),
        "frac_resid": frac_resid,
        "y_fit": details["xi_fit"],
        "f_tilde": details["f_tilde"],
        "x_plot": details["x_plot"],
        "x_raw": details["x_raw"],
        "t_osc": float(details["t_osc"]),
        "c_vw": np.asarray(details["c_vw"], dtype=np.float64),
        "per_vw_rel_rmse": {},
    }
    if mode == "s_alltp":
        payload["s_by_vw_theta"] = np.asarray(details["grid"], dtype=np.float64)
        payload["f_tilde_form"] = (
            "f_tilde = c(v_w) * [f_infty(theta0) + F0(theta0) / ((2 s*x / (3 t_osc))^(3/2) * D)] "
            "with D built from s*x"
        )
    else:
        payload["tc_by_vw_theta"] = np.asarray(details["grid"], dtype=np.float64)
        payload["f_tilde_form"] = "f_tilde = c(v_w) * [f_infty(theta0) + F0(theta0) / ((2 x / (3 t_osc))^(3/2) * D)]"
    for i, vw in enumerate(meta["vw_values"]):
        mask = np.isclose(meta["v_w"], float(vw), atol=1.0e-12)
        payload["per_vw_rel_rmse"][f"{float(vw):.1f}"] = shared_rvw.rel_rmse(meta["xi"][mask], details["xi_fit"][mask])
    return payload


def build_prediction_frame(df, payload: dict):
    out = df.copy()
    out["x_raw"] = np.asarray(payload["x_raw"], dtype=np.float64)
    out["x"] = np.asarray(payload["x_plot"], dtype=np.float64)
    out["xi_fit"] = np.asarray(payload["y_fit"], dtype=np.float64)
    out["f_tilde_fit"] = np.asarray(payload["f_tilde"], dtype=np.float64)
    return out


def summarize_payload(payload: dict, theta_values: np.ndarray, vw_values: np.ndarray, ode: dict):
    out = {k: v for k, v in payload.items() if k not in {"result_x", "frac_resid", "y_fit", "f_tilde", "x_plot", "x_raw", "c_vw", "tc_by_vw_theta", "s_by_vw_theta"}}
    out["c_calib_by_vw"] = {f"{float(vw):.1f}": float(payload["c_vw"][i]) for i, vw in enumerate(vw_values)}
    if "tc_by_vw_theta" in payload:
        out["tc_by_vw_theta"] = {
            f"{float(vw):.1f}": {
                f"{float(theta):.10f}": float(payload["tc_by_vw_theta"][i, j])
                for j, theta in enumerate(theta_values)
            }
            for i, vw in enumerate(vw_values)
        }
    if "s_by_vw_theta" in payload:
        out["s_by_vw_theta"] = {
            f"{float(vw):.1f}": {
                f"{float(theta):.10f}": float(payload["s_by_vw_theta"][i, j])
                for j, theta in enumerate(theta_values)
            }
            for i, vw in enumerate(vw_values)
        }
        out["tc_equiv_by_vw_theta"] = {
            f"{float(vw):.1f}": {
                f"{float(theta):.10f}": float(1.0 / max(float(payload["s_by_vw_theta"][i, j]), 1.0e-18))
                for j, theta in enumerate(theta_values)
            }
            for i, vw in enumerate(vw_values)
        }
    out["F0_ode"] = {f"{float(theta):.10f}": float(val) for theta, val in zip(theta_values, ode["F0"])}
    out["F_infty_raw_ode"] = {f"{float(theta):.10f}": float(val) for theta, val in zip(theta_values, ode["F_inf"])}
    out["f_infty_ode"] = {
        f"{float(theta):.10f}": float(ode["F_inf"][i] / max(float(ode["F0"][i]), 1.0e-18))
        for i, theta in enumerate(theta_values)
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


def plot_grid_by_vw(summary: dict, theta_values: np.ndarray, vw_values: np.ndarray, outpath: Path, dpi: int, key: str, ylabel: str):
    cmap = plt.get_cmap("viridis")
    colors = {float(vw): cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    hvals = theta_tc.h_alt(theta_values)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), constrained_layout=True)
    for i, vw in enumerate(vw_values):
        vals = np.asarray([summary[key][f"{float(vw):.1f}"][f"{float(theta):.10f}"] for theta in theta_values], dtype=np.float64)
        axes[0].plot(theta_values, vals, "o-", ms=4.0, lw=1.4, color=colors[float(vw)], label=rf"$v_w={float(vw):.1f}$")
        axes[1].plot(hvals, vals, "o-", ms=4.0, lw=1.4, color=colors[float(vw)], label=rf"$v_w={float(vw):.1f}$")
    axes[0].set_xlabel(r"$\theta_0$")
    axes[0].set_ylabel(ylabel)
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel(r"$h(\theta_0)=\log\!\left(e/[1-(\theta/\pi)^2]\right)$")
    axes[1].set_ylabel(ylabel)
    axes[1].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_collapse_overlay_case(df_pred, theta_values: np.ndarray, vw_values: np.ndarray, outpath: Path, dpi: int, title: str, x_label: str):
    cmap = plt.get_cmap("viridis")
    colors = {float(vw): cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    marker_map = {1.0: "s", 1.5: "^", 2.0: "D", 0.5: "o"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, theta in zip(axes, theta_values):
        sub_theta = df_pred[np.isclose(df_pred["theta"], float(theta), atol=1.0e-8)].copy()
        for vw in vw_values:
            sub_vw = sub_theta[np.isclose(sub_theta["v_w"], float(vw), atol=1.0e-8)].copy()
            if sub_vw.empty:
                continue
            for h in sorted(sub_vw["H"].unique()):
                cur = sub_vw[np.isclose(sub_vw["H"], float(h), atol=1.0e-8)].sort_values("x")
                ax.scatter(cur["x"], cur["xi"], s=20, color=colors[float(vw)], marker=marker_map.get(float(h), "o"), alpha=0.85)
            curve = sub_vw.sort_values("x")
            ax.plot(curve["x"], curve["xi_fit"], color=colors[float(vw)], lw=1.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"$\xi$")
    for ax in axes[len(theta_values):]:
        ax.axis("off")
    vw_handles = [plt.Line2D([0], [0], color=colors[float(vw)], lw=2.0) for vw in vw_values]
    vw_labels = [rf"$v_w={float(vw):.1f}$" for vw in vw_values]
    h_handles = [plt.Line2D([0], [0], color="black", marker=marker_map[h], linestyle="None") for h in sorted(df_pred["H"].unique())]
    h_labels = [rf"$H_*={h:g}$" for h in sorted(df_pred["H"].unique())]
    fig.legend(vw_handles + h_handles, vw_labels + h_labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_rmse_by_h(case_summaries: dict, outpath: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    h_values = [1.0, 1.5, 2.0]
    for mode, label, _ in CASE_CONFIGS:
        vals = [case_summaries[mode]["mean_raw_rmse_by_h"][f"{h:.1f}"] for h in h_values]
        ax.plot(h_values, vals, "o-", lw=1.8, label=label)
    ax.set_xlabel(r"$H_*$")
    ax.set_ylabel("mean raw rel_rmse")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_rmse_by_h_vw(case_summaries: dict, outpath: Path, dpi: int):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), constrained_layout=True)
    h_values = [1.0, 1.5, 2.0]
    for ax, vw in zip(axes, [0.3, 0.9]):
        for mode, label, _ in CASE_CONFIGS:
            vals = [case_summaries[mode]["mean_raw_rmse_by_h_vw"][f"H{h:.1f}_vw{vw:.1f}"] for h in h_values]
            ax.plot(h_values, vals, "o-", lw=1.8, label=label)
        ax.set_title(rf"$v_w={vw:.1f}$")
        ax.set_xlabel(r"$H_*$")
        ax.set_ylabel("mean raw rel_rmse")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def make_comparison_sheet(case_dirs: dict[str, Path], outpath: Path, dpi: int):
    items = []
    for _, label, slug in CASE_CONFIGS:
        subdir = case_dirs[slug]
        items.append((f"{label}: collapse", subdir / "collapse_overlay.png"))
        items.append((f"{label}: raw H*=2.0", subdir / "xi_vs_betaH_H2p0.png"))
    fig, axes = plt.subplots(len(CASE_CONFIGS), 2, figsize=(16, 6 * len(CASE_CONFIGS)))
    for ax, (title, path) in zip(axes.ravel(), items):
        ax.imshow(imread(path))
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.suptitle("Canonical calibrated comparison: fixed/free t_osc and all-t_p s-rescaling", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        df = quad.load_lattice_dataframe(outdir, args.vw_folders, args.h_values)
        theta_values = np.sort(df["theta"].unique())
        vw_values = np.sort(df["v_w"].unique())
        ode = shared.load_ode_amplitudes(Path(args.ode_summary).resolve(), theta_values)
        meta = shared_rvw.build_meta(df, theta_values, vw_values, ode, args.t_osc)
        ref_summary = load_json_if_exists(Path(args.reference_summary).resolve() if args.reference_summary else None)
        init_r, init_tc, init_c = init_from_reference(ref_summary, theta_values, vw_values)

        case_payloads = {}
        case_summaries = {}
        case_dirs = {}

        for mode, label, slug in CASE_CONFIGS:
            if mode == "calib_fixed_tosc":
                payload = fit_case(meta, args.fixed_beta, init_r, init_tc, init_c, args, mode)
            elif mode == "calib_free_tosc":
                seed = case_payloads["calib_fixed_tosc"]
                payload = fit_case(
                    meta,
                    args.fixed_beta,
                    float(seed["r"]),
                    np.asarray(seed["tc_by_vw_theta"], dtype=np.float64),
                    np.asarray(seed["c_vw"], dtype=np.float64),
                    args,
                    mode,
                )
            elif mode == "s_alltp":
                seed = case_payloads["calib_free_tosc"]
                payload = fit_case(
                    meta,
                    args.fixed_beta,
                    float(seed["r"]),
                    np.asarray(seed["tc_by_vw_theta"], dtype=np.float64),
                    np.asarray(seed["c_vw"], dtype=np.float64),
                    args,
                    mode,
                )
            else:
                raise ValueError(f"Unknown mode {mode}")
            case_payloads[mode] = payload

            subdir = outdir / slug
            subdir.mkdir(parents=True, exist_ok=True)
            case_dirs[slug] = subdir
            pred = build_prediction_frame(df, payload)
            pred.to_csv(subdir / "predictions.csv", index=False)

            x_label = r"$t_p$" if mode != "s_alltp" else r"$x_s=s(v_w,\theta_0)\,t_p$"
            plot_collapse_overlay_case(
                pred,
                theta_values,
                vw_values,
                subdir / "collapse_overlay.png",
                args.dpi,
                (
                    rf"{label}, kernel {shared_rvw.kernel_label(args.transient_kernel)}, "
                    rf"$r={payload['r']:.3f}$, $t_{{osc}}={payload['t_osc']:.3f}$"
                ),
                x_label,
            )
            shared_rvw.plot_raw(pred, theta_values, vw_values, subdir, "xi_vs_betaH", args.dpi)

            summary = summarize_payload(payload, theta_values, vw_values, ode)
            summary["label"] = label
            summary["collapse_x_label"] = x_label
            raw_rows, mean_by_h, mean_by_h_vw = rmse_tables(pred)
            summary["raw_plot_rmse"] = raw_rows
            summary["mean_raw_rmse_by_h"] = mean_by_h
            summary["mean_raw_rmse_by_h_vw"] = mean_by_h_vw
            case_summaries[mode] = summary
            shared.save_json(subdir / "final_summary.json", summary)
            if "tc_by_vw_theta" in summary:
                plot_grid_by_vw(summary, theta_values, vw_values, subdir / "tc_by_vw.png", args.dpi, "tc_by_vw_theta", r"$t_c(\theta_0; v_w)$")
            if "s_by_vw_theta" in summary:
                plot_grid_by_vw(summary, theta_values, vw_values, subdir / "s_by_vw.png", args.dpi, "s_by_vw_theta", r"$s(\theta_0; v_w)$")

        plot_rmse_by_h(case_summaries, outdir / "rmse_by_h.png", args.dpi)
        plot_rmse_by_h_vw(case_summaries, outdir / "rmse_by_h_vw.png", args.dpi)
        make_comparison_sheet(case_dirs, outdir / "comparison_sheet.png", args.dpi)

        summary = {
            "status": "ok",
            "vw_values": [float(v) for v in vw_values],
            "theta_values": [float(v) for v in theta_values],
            "fixed_beta": float(args.fixed_beta),
            "tc_max": float(args.tc_max),
            "t_osc_fixed_for_s_case": float(args.t_osc),
            "transient_kernel": str(args.transient_kernel),
            "n_points": int(len(df)),
            "ode_amplitude_source": ode["source"],
            "ode_fit_summary": ode["ode_fit_summary"],
            "cases": case_summaries,
            "outputs": {
                "comparison_sheet": str(outdir / "comparison_sheet.png"),
                "rmse_by_h": str(outdir / "rmse_by_h.png"),
                "rmse_by_h_vw": str(outdir / "rmse_by_h_vw.png"),
            },
        }
        shared.save_json(outdir / "final_summary.json", summary)
        print(json.dumps(shared.to_native(summary), indent=2, sort_keys=True))
    except Exception as exc:
        error_exit(outdir, exc)
        raise


if __name__ == "__main__":
    main()
