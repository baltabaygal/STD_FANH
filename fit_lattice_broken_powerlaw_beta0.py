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
import fit_lattice_fixed_ode_amplitudes_theta_tc_shared_r_by_vw as shared_rvw
import fit_lattice_quadwarp_universal as quad


ROOT = Path(__file__).resolve().parent
OUTDIR_DEFAULT = "results_lattice_broken_powerlaw_beta0"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]
REFERENCE_VW_DEFAULT = 0.9

CASE_CONFIGS = [
    ("vw0p9_c1_sharedr", r"$v_w=0.9$, $C=1$, shared $r$", "vw0p9_c1_sharedr"),
    ("allvw_cvw_sharedr", r"all $v_w$, $C(v_w)$, shared $r$", "allvw_cvw_sharedr"),
    ("allvw_cvw_rvw", r"all $v_w$, $C(v_w)$, $r(v_w)$", "allvw_cvw_rvw"),
    ("allvw_ctheta_rvw", r"all $v_w$, $C(\theta_0)$, $r(v_w)$", "allvw_ctheta_rvw"),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit the clean broken-power-law xi model xi = (1 + u^r)^(1/r) with u = C(v_w) F_inf x^(3/2) / F0."
    )
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--fixed-beta", type=float, default=0.0)
    p.add_argument("--t-osc", type=float, default=1.5)
    p.add_argument("--ode-summary", type=str, default=str(shared.ODE_SUMMARY_DEFAULT))
    p.add_argument("--reference-vw", type=float, default=REFERENCE_VW_DEFAULT)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--outdir", type=str, default=OUTDIR_DEFAULT)
    return p.parse_args()


def error_exit(outdir: Path, exc: Exception):
    payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
    outdir.mkdir(parents=True, exist_ok=True)
    shared.save_json(outdir / "_error.json", payload)
    print(json.dumps(payload, sort_keys=True))


def xi_broken_power(u: np.ndarray, r: np.ndarray) -> np.ndarray:
    u = np.maximum(np.asarray(u, dtype=np.float64), 1.0e-18)
    r = np.maximum(np.asarray(r, dtype=np.float64), 1.0e-12)
    z = r * np.log(u)
    log_xi = np.where(
        z > 50.0,
        z / r,
        np.log1p(np.exp(np.clip(z, -700.0, 700.0))) / r,
    )
    return np.exp(log_xi)


def unpack_params(params: np.ndarray, n_vw: int, ref_idx: int, n_theta: int, mode: str):
    idx = 0
    if mode in {"vw0p9_c1_sharedr", "allvw_cvw_sharedr"}:
        r_values = np.asarray([float(params[idx])], dtype=np.float64)
        idx += 1
    elif mode in {"allvw_cvw_rvw", "allvw_ctheta_rvw"}:
        r_values = np.asarray(params[idx : idx + n_vw], dtype=np.float64)
        idx += n_vw
    else:
        raise ValueError(f"Unknown mode: {mode}")

    c_values = np.ones(n_vw, dtype=np.float64)
    if mode in {"allvw_cvw_sharedr", "allvw_cvw_rvw"}:
        free_idx = [i for i in range(n_vw) if i != ref_idx]
        c_values[free_idx] = np.asarray(params[idx : idx + len(free_idx)], dtype=np.float64)
        idx += len(free_idx)

    c_theta = np.ones(n_theta, dtype=np.float64)
    if mode == "allvw_ctheta_rvw":
        c_theta = np.asarray(params[idx : idx + n_theta], dtype=np.float64)
        idx += n_theta
    return r_values, c_values, c_theta


def build_param_vector(
    init_r: float,
    init_c_by_vw: np.ndarray | None,
    init_c_by_theta: np.ndarray | None,
    n_vw: int,
    n_theta: int,
    ref_idx: int,
    mode: str,
):
    parts = []
    lower = []
    upper = []
    if mode in {"vw0p9_c1_sharedr", "allvw_cvw_sharedr"}:
        parts.append(float(init_r))
        lower.append(0.1)
        upper.append(30.0)
    else:
        parts.extend(np.full(n_vw, float(init_r), dtype=np.float64).tolist())
        lower.extend([0.1] * n_vw)
        upper.extend([30.0] * n_vw)
    if mode in {"allvw_cvw_sharedr", "allvw_cvw_rvw"}:
        if init_c_by_vw is None or len(init_c_by_vw) != n_vw:
            raise ValueError("init_c_by_vw must match number of vw entries")
        free_idx = [i for i in range(n_vw) if i != ref_idx]
        c_init = np.asarray(init_c_by_vw, dtype=np.float64)
        parts.extend(c_init[free_idx].tolist())
        lower.extend([0.1] * len(free_idx))
        upper.extend([10.0] * len(free_idx))
    if mode == "allvw_ctheta_rvw":
        if init_c_by_theta is None or len(init_c_by_theta) != n_theta:
            raise ValueError("init_c_by_theta must match number of theta entries")
        c_theta_init = np.asarray(init_c_by_theta, dtype=np.float64)
        parts.extend(c_theta_init.tolist())
        lower.extend([0.1] * n_theta)
        upper.extend([10.0] * n_theta)
    return np.asarray(parts, dtype=np.float64), np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def model_details(meta, fixed_beta: float, params: np.ndarray, ref_idx: int, mode: str):
    n_vw = len(meta["vw_values"])
    n_theta = len(meta["theta_values"])
    r_values, c_values, c_theta = unpack_params(params, n_vw, ref_idx, n_theta, mode)
    x, xi_scale = shared_rvw.x_and_xi_scale(meta, fixed_beta)
    if mode in {"vw0p9_c1_sharedr", "allvw_cvw_sharedr"}:
        r_eff = np.full_like(x, float(r_values[0]), dtype=np.float64)
    else:
        r_eff = r_values[meta["vw_idx"]]
    if mode == "allvw_ctheta_rvw":
        c_eff = c_theta[meta["theta_idx"]]
    else:
        c_eff = c_values[meta["vw_idx"]]
    u = c_eff * meta["F_inf"] / np.maximum(meta["F0"], 1.0e-18) * xi_scale
    xi_fit = xi_broken_power(u, r_eff)
    f_tilde = xi_fit * meta["F0"] / np.maximum(xi_scale, 1.0e-18)
    return x, xi_scale, u, xi_fit, f_tilde, r_values, c_values, c_theta


def fit_case(
    meta,
    fixed_beta: float,
    init_r: float,
    init_c_by_vw: np.ndarray | None,
    init_c_by_theta: np.ndarray | None,
    ref_idx: int,
    mode: str,
):
    n_vw = len(meta["vw_values"])
    n_theta = len(meta["theta_values"])
    x0, lower, upper = build_param_vector(init_r, init_c_by_vw, init_c_by_theta, n_vw, n_theta, ref_idx, mode)

    def resid(par: np.ndarray) -> np.ndarray:
        _, _, _, xi_fit, _, _, _, _ = model_details(meta, fixed_beta, par, ref_idx, mode)
        return (xi_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)

    huber = least_squares(resid, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=60000)
    final = least_squares(resid, huber.x, bounds=(lower, upper), loss="linear", max_nfev=60000)
    x, xi_scale, u, xi_fit, f_tilde, r_values, c_values, c_theta = model_details(meta, fixed_beta, final.x, ref_idx, mode)
    frac_resid = (xi_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    aic, bic = shared.aic_bic(frac_resid, len(final.x))
    payload = {
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "mode": str(mode),
        "beta": float(fixed_beta),
        "canonical_xi_form": "xi = (2 x / (3 t_osc))^(3/2) * f_tilde / F0",
        "f_tilde_form": "f_tilde = [ (C(v_w) f_infty(theta0))^r + (F0(theta0) / (2 x / (3 t_osc))^(3/2))^r ]^(1/r)",
        "xi_model_form": "xi = [1 + (C F_inf(theta0) (2 x / (3 t_osc))^(3/2) / F0(theta0))^r]^(1/r)",
        "rel_rmse": shared_rvw.rel_rmse(meta["xi"], xi_fit),
        "AIC": float(aic),
        "BIC": float(bic),
        "n_points": int(meta["xi"].size),
        "n_params": int(len(final.x)),
        "result_x": np.asarray(final.x, dtype=np.float64),
        "x_fit": x,
        "xi_scale_fit": xi_scale,
        "u_fit": u,
        "y_fit": xi_fit,
        "f_tilde_fit": f_tilde,
        "frac_resid": frac_resid,
        "per_vw_rel_rmse": {},
        "theta_values": np.asarray(meta["theta_values"], dtype=np.float64),
    }
    if mode in {"vw0p9_c1_sharedr", "allvw_cvw_sharedr"}:
        payload["r"] = float(r_values[0])
    else:
        payload["r_by_vw"] = np.asarray(r_values, dtype=np.float64)
    payload["c_by_vw"] = np.asarray(c_values, dtype=np.float64)
    payload["c_by_theta"] = np.asarray(c_theta, dtype=np.float64)
    for i, vw in enumerate(meta["vw_values"]):
        mask = np.isclose(meta["v_w"], float(vw), atol=1.0e-12)
        payload["per_vw_rel_rmse"][f"{float(vw):.1f}"] = shared_rvw.rel_rmse(meta["xi"][mask], xi_fit[mask])
    return payload


def build_prediction_frame(df, fit_payload: dict):
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
        if k
        not in {
            "result_x",
            "x_fit",
            "xi_scale_fit",
            "u_fit",
            "y_fit",
            "f_tilde_fit",
            "frac_resid",
            "c_by_vw",
            "r_by_vw",
        }
    }
    out["c_by_vw"] = {f"{float(vw):.1f}": float(payload["c_by_vw"][i]) for i, vw in enumerate(vw_values)}
    if "r_by_vw" in payload:
        out["r_by_vw"] = {f"{float(vw):.1f}": float(payload["r_by_vw"][i]) for i, vw in enumerate(vw_values)}
    if "c_by_theta" in payload and "theta_values" in payload:
        out["c_by_theta"] = {
            f"{float(theta):.3f}": float(payload["c_by_theta"][i])
            for i, theta in enumerate(payload["theta_values"])
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
        if "r_by_vw" in fit_payload:
            i = int(np.argmin(np.abs(vw_values - float(vw))))
            r_eff = float(fit_payload["r_by_vw"][i])
        else:
            r_eff = float(fit_payload["r"])
        xi_curve = xi_broken_power(u_grid, np.full_like(u_grid, r_eff))
        ax.plot(u_grid, xi_curve, color="black", lw=1.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$v_w={float(vw):.1f}$")
        ax.set_xlabel(r"$u = C(v_w) F_\infty x^{3/2}/F_0$")
        ax.set_ylabel(r"$\xi$")
        used_axes += 1
    for ax in axes.ravel()[used_axes:]:
        ax.axis("off")
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_r_by_vw(case_summaries: dict, outpath: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    base_vws = None
    for mode, label, _ in CASE_CONFIGS:
        if "r_by_vw" not in case_summaries.get(mode, {}):
            continue
        keys = sorted(case_summaries[mode]["r_by_vw"])
        vw_values = [float(k) for k in keys]
        vals = [case_summaries[mode]["r_by_vw"][k] for k in keys]
        base_vws = vw_values
        ax.plot(vw_values, vals, "o-", lw=1.8, label=label)
    if base_vws is not None:
        ax.set_xticks(base_vws)
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$r(v_w)$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_c_by_vw(case_summaries: dict, outpath: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for mode, label, _ in CASE_CONFIGS:
        if "c_by_vw" not in case_summaries.get(mode, {}):
            continue
        keys = sorted(case_summaries[mode]["c_by_vw"])
        vw_values = [float(k) for k in keys]
        vals = [case_summaries[mode]["c_by_vw"][k] for k in keys]
        ax.plot(vw_values, vals, "o-", lw=1.8, label=label)
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$C(v_w)$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_c_by_theta(case_summaries: dict, outpath: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    plotted = False
    for mode, label, _ in CASE_CONFIGS:
        summary = case_summaries.get(mode, {})
        if "c_by_theta" not in summary:
            continue
        keys = sorted(summary["c_by_theta"], key=float)
        theta_values = [float(k) for k in keys]
        vals = [summary["c_by_theta"][k] for k in keys]
        ax.plot(theta_values, vals, "o-", lw=1.8, label=label)
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "no C(theta) data", ha="center", va="center")
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$C(\theta_0)$")
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
    fig.suptitle(r"Broken-power interpolation model comparison", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        df_all = quad.load_lattice_dataframe(outdir, args.vw_folders, args.h_values)
        theta_values = np.sort(df_all["theta"].unique())
        vw_values_all = np.sort(df_all["v_w"].unique())
        h_values = np.sort(df_all["H"].unique())
        ode = shared.load_ode_amplitudes(Path(args.ode_summary).resolve(), theta_values)

        ref_matches = np.where(np.isclose(vw_values_all, float(args.reference_vw), atol=1.0e-12))[0]
        if ref_matches.size != 1:
            raise RuntimeError(f"Could not identify a unique reference v_w={args.reference_vw}")
        ref_idx_all = int(ref_matches[0])

        df_v9 = df_all[np.isclose(df_all["v_w"], float(args.reference_vw), atol=1.0e-12)].copy()
        meta_v9 = shared_rvw.build_meta(df_v9, theta_values, np.asarray([float(args.reference_vw)], dtype=np.float64), ode, args.t_osc)
        fit_v9 = fit_case(
            meta_v9,
            args.fixed_beta,
            2.0,
            np.ones(1, dtype=np.float64),
            None,
            0,
            "vw0p9_c1_sharedr",
        )

        meta_all = shared_rvw.build_meta(df_all, theta_values, vw_values_all, ode, args.t_osc)
        fit_all_sharedr = fit_case(
            meta_all,
            args.fixed_beta,
            float(fit_v9["r"]),
            np.ones(len(vw_values_all), dtype=np.float64),
            None,
            ref_idx_all,
            "allvw_cvw_sharedr",
        )

        c_init = np.asarray(fit_all_sharedr["c_by_vw"], dtype=np.float64)
        fit_all_rvw = fit_case(
            meta_all,
            args.fixed_beta,
            float(fit_all_sharedr["r"]),
            c_init,
            None,
            ref_idx_all,
            "allvw_cvw_rvw",
        )

        fit_all_ctheta_rvw = fit_case(
            meta_all,
            args.fixed_beta,
            float(fit_v9["r"]),
            None,
            np.ones(len(theta_values), dtype=np.float64),
            ref_idx_all,
            "allvw_ctheta_rvw",
        )

        case_payloads = {
            "vw0p9_c1_sharedr": fit_v9,
            "allvw_cvw_sharedr": fit_all_sharedr,
            "allvw_cvw_rvw": fit_all_rvw,
            "allvw_ctheta_rvw": fit_all_ctheta_rvw,
        }
        case_summaries = {}
        case_dirs = {}

        for mode, label, slug in CASE_CONFIGS:
            subdir = outdir / slug
            subdir.mkdir(parents=True, exist_ok=True)
            case_dirs[mode] = subdir
            payload = case_payloads[mode]
            if mode == "vw0p9_c1_sharedr":
                df_case = df_v9
                meta_case = meta_v9
                vw_case = np.asarray([float(args.reference_vw)], dtype=np.float64)
            else:
                df_case = df_all
                meta_case = meta_all
                vw_case = vw_values_all
            pred = build_prediction_frame(df_case, payload)
            pred.to_csv(subdir / "predictions.csv", index=False)
            if "r_by_vw" in payload:
                title = label
            else:
                title = rf"{label}, $r={payload['r']:.3f}$"
            shared_rvw.plot_collapse_overlay(pred, theta_values, vw_case, payload, subdir / "collapse_overlay.png", args.dpi, args.fixed_beta, title)
            shared_rvw.plot_raw(pred, theta_values, vw_case, subdir, "xi_vs_betaH", args.dpi)
            plot_xi_vs_u_by_vw(pred, vw_case, payload, subdir / "xi_vs_u_by_vw.png", args.dpi)
            raw_rows, mean_by_h, mean_by_h_vw = rmse_tables(pred)
            summary = summarize_payload(payload, vw_case)
            summary["label"] = label
            summary["reference_vw"] = float(args.reference_vw)
            summary["mean_raw_rmse_by_h"] = mean_by_h
            summary["mean_raw_rmse_by_h_vw"] = mean_by_h_vw
            summary["raw_plot_rmse"] = raw_rows
            case_summaries[mode] = summary
            shared.save_json(subdir / "final_summary.json", summary)

        plot_r_by_vw(case_summaries, outdir / "r_by_vw.png", args.dpi)
        plot_c_by_vw(case_summaries, outdir / "c_by_vw.png", args.dpi)
        plot_c_by_theta(case_summaries, outdir / "c_by_theta.png", args.dpi)
        make_comparison_sheet(case_dirs, outdir / "comparison_sheet.png", args.dpi)

        summary = {
            "status": "ok",
            "vw_values": [float(v) for v in vw_values_all],
            "theta_values": [float(v) for v in theta_values],
            "h_values": [float(v) for v in h_values],
            "fixed_beta": float(args.fixed_beta),
            "reference_vw": float(args.reference_vw),
            "n_points": int(len(df_all)),
            "ode_amplitude_source": ode["source"],
            "ode_fit_summary": ode["ode_fit_summary"],
            "cases": case_summaries,
            "outputs": {
                "comparison_sheet": str(outdir / "comparison_sheet.png"),
                "r_by_vw": str(outdir / "r_by_vw.png"),
                "c_by_vw": str(outdir / "c_by_vw.png"),
                "c_by_theta": str(outdir / "c_by_theta.png"),
            },
        }
        shared.save_json(outdir / "final_summary.json", summary)
        print(json.dumps(shared.to_native(summary), indent=2, sort_keys=True))
    except Exception as exc:
        error_exit(outdir, exc)
        raise


if __name__ == "__main__":
    main()
