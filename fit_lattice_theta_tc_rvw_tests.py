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
OUTDIR_DEFAULT = "results_lattice_theta_tc_rvw_tests_beta0_tcmax300"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]
BASELINE_INIT_DEFAULT = ROOT / "results_lattice_theta_tc_sharedr_free_finf_by_vw_beta0_tcmax300" / "final_summary.json"


CASE_CONFIGS = [
    ("baseline", "Shared r, tc(theta0,vw)", "baseline"),
    ("rvw_globaltc", r"$r(v_w)$, shared $t_c$", "rvw_globaltc"),
    ("rvw_theta", r"$r(v_w)$, $t_c(\theta_0,v_w)$", "rvw_theta"),
    ("rvw_hgrid", r"$r(v_w)$, $t_c(\theta_0,v_w,H_*)$", "rvw_hgrid"),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Test whether allowing r to vary with v_w fixes the remaining tp-shape problem, with and without H* dependence in tc."
    )
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--fixed-beta", type=float, default=0.0)
    p.add_argument("--t-osc", type=float, default=1.5)
    p.add_argument("--ode-summary", type=str, default=str(shared.ODE_SUMMARY_DEFAULT))
    p.add_argument("--baseline-init-summary", type=str, default=str(BASELINE_INIT_DEFAULT))
    p.add_argument("--tc-max", type=float, default=300.0)
    p.add_argument(
        "--transient-kernel",
        type=str,
        choices=["power_inside", "shifted", "weibull"],
        default="power_inside",
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


def baseline_init_from_summary(summary: dict | None, theta_values: np.ndarray, vw_values: np.ndarray, ode: dict):
    init_r = 2.0
    init_tc = np.full((len(vw_values), len(theta_values)), 2.0, dtype=np.float64)
    init_f_infty = np.asarray(ode["F_inf"], dtype=np.float64) / np.maximum(np.asarray(ode["F0"], dtype=np.float64), 1.0e-18)
    if summary is None:
        return init_r, init_tc, init_f_infty
    payload = summary.get("shared_r_free_finf", {})
    if "r" in payload:
        init_r = float(payload["r"])
    table = payload.get("tc_by_vw_theta", {})
    for i, vw in enumerate(vw_values):
        block = table.get(f"{float(vw):.1f}", {})
        for j, theta in enumerate(theta_values):
            key = f"{float(theta):.10f}"
            if key in block:
                init_tc[i, j] = float(block[key])
    f_inf_table = payload.get("f_infty_lattice", {})
    for j, theta in enumerate(theta_values):
        key = f"{float(theta):.10f}"
        if key in f_inf_table:
            init_f_infty[j] = float(f_inf_table[key])
    return init_r, init_tc, init_f_infty


def unpack_params(params: np.ndarray, n_vw: int, n_theta: int, n_h: int, mode: str):
    idx = 0
    if mode == "baseline":
        r_values = np.asarray([float(params[idx])], dtype=np.float64)
        idx += 1
    else:
        r_values = np.asarray(params[idx : idx + n_vw], dtype=np.float64)
        idx += n_vw
    if mode == "rvw_globaltc":
        tc_values = np.asarray([float(params[idx])], dtype=np.float64)
        idx += 1
    elif mode == "rvw_hgrid":
        tc_values = np.asarray(params[idx : idx + n_vw * n_theta * n_h], dtype=np.float64).reshape(n_vw, n_theta, n_h)
        idx += n_vw * n_theta * n_h
    else:
        tc_values = np.asarray(params[idx : idx + n_vw * n_theta], dtype=np.float64).reshape(n_vw, n_theta)
        idx += n_vw * n_theta
    f_infty_theta = np.asarray(params[idx : idx + n_theta], dtype=np.float64)
    return r_values, tc_values, f_infty_theta


def build_param_vector(init_r: float, init_tc: np.ndarray, init_f_infty: np.ndarray, mode: str, n_vw: int, n_h: int):
    tc_init = np.asarray(init_tc, dtype=np.float64)
    if mode == "rvw_globaltc":
        tc_scalar = float(np.median(tc_init))
        tc_init = np.asarray([tc_scalar], dtype=np.float64)
    elif mode == "rvw_hgrid":
        if tc_init.ndim == 2:
            tc_init = np.repeat(tc_init[:, :, None], n_h, axis=2)
        if tc_init.ndim != 3:
            raise ValueError("rvw_hgrid mode requires a 3D tc initializer or a 2D grid that can be broadcast over H.")
    else:
        if tc_init.ndim != 2:
            raise ValueError("This mode requires a 2D tc(vw,theta) initializer.")
    r_init = np.full(n_vw, float(init_r), dtype=np.float64) if mode != "baseline" else np.asarray([float(init_r)], dtype=np.float64)
    x0 = np.concatenate([r_init, tc_init.ravel(), np.asarray(init_f_infty, dtype=np.float64)])
    lower = np.concatenate(
        [
            np.full(r_init.size, 0.1, dtype=np.float64),
            np.full(tc_init.size, 0.1, dtype=np.float64),
            np.full(init_f_infty.size, 1.0e-8, dtype=np.float64),
        ]
    )
    upper = np.concatenate(
        [
            np.full(r_init.size, 20.0, dtype=np.float64),
            np.full(tc_init.size, np.inf, dtype=np.float64),
            np.full(init_f_infty.size, 1.0e4, dtype=np.float64),
        ]
    )
    return x0, lower, upper, r_init.size, tc_init.size


def model_details(meta, fixed_beta: float, params: np.ndarray, kernel: str, mode: str):
    n_vw = len(meta["vw_values"])
    n_theta = len(meta["theta_values"])
    n_h = len(meta["h_values"])
    r_values, tc_values, f_infty_theta = unpack_params(params, n_vw, n_theta, n_h, mode)
    x, xi_scale = shared_rvw.x_and_xi_scale(meta, fixed_beta)
    if mode == "rvw_globaltc":
        tc = np.full_like(x, float(tc_values[0]), dtype=np.float64)
    elif mode == "rvw_hgrid":
        tc = tc_values[meta["vw_idx"], meta["theta_idx"], meta["h_idx"]]
    else:
        tc = tc_values[meta["vw_idx"], meta["theta_idx"]]
    if mode == "baseline":
        r_eff = np.full_like(x, float(r_values[0]), dtype=np.float64)
    else:
        r_eff = r_values[meta["vw_idx"]]
    denom = shared_rvw.transition_denom(x, tc, r_eff, kernel)
    f_tilde = f_infty_theta[meta["theta_idx"]] + meta["F0"] / np.maximum(xi_scale * denom, 1.0e-18)
    xi_fit = shared_rvw.xi_from_f_tilde(xi_scale, f_tilde, meta["F0"])
    return x, xi_scale, tc, r_eff, f_tilde, xi_fit, r_values, tc_values, f_infty_theta


def fit_case(meta, fixed_beta: float, init_r: float, init_tc: np.ndarray, init_f_infty: np.ndarray, tc_max: float, kernel: str, mode: str):
    n_vw = len(meta["vw_values"])
    x0, lower, upper, r_param_count, tc_param_count = build_param_vector(init_r, init_tc, init_f_infty, mode, n_vw, len(meta["h_values"]))
    tc_slice = slice(r_param_count, r_param_count + tc_param_count)
    lower[tc_slice] = 0.1
    upper[tc_slice] = float(tc_max)

    def resid(par: np.ndarray) -> np.ndarray:
        _, _, _, _, _, xi_fit, *_ = model_details(meta, fixed_beta, par, kernel, mode)
        return (xi_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)

    huber = least_squares(resid, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=70000)
    final = least_squares(resid, huber.x, bounds=(lower, upper), loss="linear", max_nfev=70000)
    x, xi_scale, tc_eff, r_eff, f_tilde, xi_fit, r_values, tc_values, f_infty_theta = model_details(meta, fixed_beta, final.x, kernel, mode)
    frac_resid = (xi_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    aic, bic = shared.aic_bic(frac_resid, len(final.x))
    payload = {
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "mode": str(mode),
        "beta": float(fixed_beta),
        "transient_kernel": str(kernel),
        "canonical_xi_form": "xi = (2 x / (3 t_osc))^(3/2) * f_tilde / F0",
        "transition_denominator": shared_rvw.kernel_denom_text(kernel),
        "rel_rmse": shared_rvw.rel_rmse(meta["xi"], xi_fit),
        "AIC": float(aic),
        "BIC": float(bic),
        "n_points": int(meta["xi"].size),
        "n_params": int(len(final.x)),
        "result_x": np.asarray(final.x, dtype=np.float64),
        "y_fit": xi_fit,
        "f_tilde_fit": f_tilde,
        "x_fit": x,
        "tc_eff_fit": tc_eff,
        "r_eff_fit": r_eff,
        "frac_resid": frac_resid,
        "f_infty": f_infty_theta,
        "per_vw_rel_rmse": {},
    }
    if mode == "baseline":
        payload["r"] = float(r_values[0])
        payload["tc_by_vw_theta"] = tc_values
        payload["f_tilde_form"] = "f_tilde = f_infty(theta0) + F0(theta0) / ((2 x / (3 t_osc))^(3/2) * D)"
    elif mode == "rvw_globaltc":
        payload["r_by_vw"] = np.asarray(r_values, dtype=np.float64)
        payload["t_c_shared"] = float(tc_values[0])
        payload["f_tilde_form"] = "f_tilde = f_infty(theta0) + F0(theta0) / ((2 x / (3 t_osc))^(3/2) * D), with r = r(v_w) and shared tc"
    elif mode == "rvw_theta":
        payload["r_by_vw"] = np.asarray(r_values, dtype=np.float64)
        payload["tc_by_vw_theta"] = tc_values
        payload["f_tilde_form"] = "f_tilde = f_infty(theta0) + F0(theta0) / ((2 x / (3 t_osc))^(3/2) * D), with r = r(v_w)"
    elif mode == "rvw_hgrid":
        payload["r_by_vw"] = np.asarray(r_values, dtype=np.float64)
        payload["tc_by_vw_theta_h"] = tc_values
        payload["h_values"] = np.asarray(meta["h_values"], dtype=np.float64)
        payload["f_tilde_form"] = "f_tilde = f_infty(theta0) + F0(theta0) / ((2 x / (3 t_osc))^(3/2) * D), with r = r(v_w) and tc = tc(theta0,v_w,H)"
    for i, vw in enumerate(meta["vw_values"]):
        mask = np.isclose(meta["v_w"], float(vw), atol=1.0e-12)
        payload["per_vw_rel_rmse"][f"{float(vw):.1f}"] = shared_rvw.rel_rmse(meta["xi"][mask], xi_fit[mask])
    return payload


def build_prediction_frame(df, meta, fit_payload: dict, fixed_beta: float, kernel: str, mode: str):
    out = df.copy()
    out["x"] = fit_payload["x_fit"]
    out["xi_fit"] = fit_payload["y_fit"]
    out["f_tilde_fit"] = fit_payload["f_tilde_fit"]
    out["tc_eff_fit"] = fit_payload["tc_eff_fit"]
    out["r_eff_fit"] = fit_payload["r_eff_fit"]
    n_vw = len(meta["vw_values"])
    n_theta = len(meta["theta_values"])
    n_h = len(meta["h_values"])
    _, _, f_infty_theta = unpack_params(fit_payload["result_x"], n_vw, n_theta, n_h, mode)
    out["f_infty_fit"] = f_infty_theta[np.asarray(meta["theta_idx"], dtype=np.int64)]
    return out


def summarize_payload(payload: dict, theta_values: np.ndarray, vw_values: np.ndarray, ode: dict):
    out = {
        k: v
        for k, v in payload.items()
        if k not in {
            "result_x",
            "y_fit",
            "f_tilde_fit",
            "x_fit",
            "tc_eff_fit",
            "r_eff_fit",
            "frac_resid",
            "tc_by_vw_theta",
            "tc_by_vw_theta_h",
            "f_infty",
            "h_values",
            "r_by_vw",
        }
    }
    if "r_by_vw" in payload:
        out["r_by_vw"] = {f"{float(vw):.1f}": float(payload["r_by_vw"][i]) for i, vw in enumerate(vw_values)}
    if "t_c_shared" in payload:
        out["t_c_shared"] = float(payload["t_c_shared"])
    elif "tc_by_vw_theta_h" in payload:
        h_values = np.asarray(payload["h_values"], dtype=np.float64)
        out["tc_by_vw_theta_h"] = {}
        for i, vw in enumerate(vw_values):
            out["tc_by_vw_theta_h"][f"{float(vw):.1f}"] = {}
            for j, theta in enumerate(theta_values):
                out["tc_by_vw_theta_h"][f"{float(vw):.1f}"][f"{float(theta):.10f}"] = {
                    f"{float(h):.1f}": float(payload["tc_by_vw_theta_h"][i, j, k])
                    for k, h in enumerate(h_values)
                }
    else:
        out["tc_by_vw_theta"] = {}
        for i, vw in enumerate(vw_values):
            out["tc_by_vw_theta"][f"{float(vw):.1f}"] = {
                f"{float(theta):.10f}": float(payload["tc_by_vw_theta"][i, j])
                for j, theta in enumerate(theta_values)
            }
    out["f_infty_lattice"] = {f"{float(theta):.10f}": float(payload["f_infty"][i]) for i, theta in enumerate(theta_values)}
    out["f_infty_ode"] = {
        f"{float(theta):.10f}": float(ode["F_inf"][i] / max(float(ode["F0"][i]), 1.0e-18))
        for i, theta in enumerate(theta_values)
    }
    out["F0_ode"] = {f"{float(theta):.10f}": float(val) for theta, val in zip(theta_values, ode["F0"])}
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


def plot_r_by_vw(case_summaries: dict, outpath: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    vw_values = [0.3, 0.5, 0.7, 0.9]
    for mode, label, _ in CASE_CONFIGS:
        if "r_by_vw" not in case_summaries[mode]:
            continue
        vals = [case_summaries[mode]["r_by_vw"][f"{vw:.1f}"] for vw in vw_values]
        ax.plot(vw_values, vals, "o-", lw=1.8, label=label)
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$r(v_w)$")
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
    fig.suptitle(r"$r(v_w)$ comparison", fontsize=16)
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
        h_values = np.sort(df["H"].unique())
        ode = shared.load_ode_amplitudes(Path(args.ode_summary).resolve(), theta_values)
        meta = shared_rvw.build_meta(df, theta_values, vw_values, ode, args.t_osc)
        h_index = {float(h): i for i, h in enumerate(h_values)}
        meta["h_values"] = np.asarray(h_values, dtype=np.float64)
        meta["h_idx"] = np.array([h_index[float(h)] for h in df["H"].to_numpy(dtype=np.float64)], dtype=np.int64)

        baseline_ref = load_json_if_exists(Path(args.baseline_init_summary).resolve() if args.baseline_init_summary else None)
        init_r, init_tc, init_f_infty = baseline_init_from_summary(baseline_ref, theta_values, vw_values, ode)

        case_payloads = {}
        case_summaries = {}
        case_dirs = {}

        baseline = fit_case(meta, args.fixed_beta, init_r, init_tc, init_f_infty, args.tc_max, args.transient_kernel, "baseline")
        case_payloads["baseline"] = baseline

        rvw_globaltc = fit_case(
            meta,
            args.fixed_beta,
            float(baseline["r"]),
            np.asarray(baseline["tc_by_vw_theta"], dtype=np.float64),
            np.asarray(baseline["f_infty"], dtype=np.float64),
            args.tc_max,
            args.transient_kernel,
            "rvw_globaltc",
        )
        case_payloads["rvw_globaltc"] = rvw_globaltc

        rvw_theta = fit_case(
            meta,
            args.fixed_beta,
            float(baseline["r"]),
            np.asarray(baseline["tc_by_vw_theta"], dtype=np.float64),
            np.asarray(baseline["f_infty"], dtype=np.float64),
            args.tc_max,
            args.transient_kernel,
            "rvw_theta",
        )
        case_payloads["rvw_theta"] = rvw_theta

        rvw_hgrid = fit_case(
            meta,
            args.fixed_beta,
            float(baseline["r"]),
            np.asarray(baseline["tc_by_vw_theta"], dtype=np.float64),
            np.asarray(baseline["f_infty"], dtype=np.float64),
            args.tc_max,
            args.transient_kernel,
            "rvw_hgrid",
        )
        case_payloads["rvw_hgrid"] = rvw_hgrid

        for mode, label, slug in CASE_CONFIGS:
            subdir = outdir / slug
            subdir.mkdir(parents=True, exist_ok=True)
            case_dirs[mode] = subdir
            payload = case_payloads[mode]
            pred = build_prediction_frame(df, meta, payload, args.fixed_beta, args.transient_kernel, mode)
            pred.to_csv(subdir / "predictions.csv", index=False)
            title = rf"{label}, kernel {shared_rvw.kernel_label(args.transient_kernel)}"
            if "r" in payload:
                title += rf", $r={payload['r']:.3f}$"
            elif "r_by_vw" in payload:
                title += r", $r=r(v_w)$"
            shared_rvw.plot_collapse_overlay(pred, theta_values, vw_values, payload, subdir / "collapse_overlay.png", args.dpi, args.fixed_beta, title)
            shared_rvw.plot_raw(pred, theta_values, vw_values, subdir, "xi_vs_betaH", args.dpi)
            raw_rows, mean_by_h, mean_by_h_vw = rmse_tables(pred)
            summary = summarize_payload(payload, theta_values, vw_values, ode)
            summary["label"] = label
            summary["mean_raw_rmse_by_h"] = mean_by_h
            summary["mean_raw_rmse_by_h_vw"] = mean_by_h_vw
            summary["raw_plot_rmse"] = raw_rows
            if mode == "rvw_hgrid":
                summary["h_values"] = [float(h) for h in h_values]
            case_summaries[mode] = summary
            shared.save_json(subdir / "final_summary.json", summary)

        plot_rmse_by_h(case_summaries, outdir / "rmse_by_h.png", args.dpi)
        plot_rmse_by_h_vw(case_summaries, outdir / "rmse_by_h_vw.png", args.dpi)
        plot_r_by_vw(case_summaries, outdir / "r_by_vw.png", args.dpi)
        make_comparison_sheet(case_dirs, outdir / "comparison_sheet.png", args.dpi)

        summary = {
            "status": "ok",
            "vw_values": [float(v) for v in vw_values],
            "theta_values": [float(v) for v in theta_values],
            "h_values": [float(v) for v in h_values],
            "fixed_beta": float(args.fixed_beta),
            "tc_max": float(args.tc_max),
            "transient_kernel": str(args.transient_kernel),
            "n_points": int(len(df)),
            "ode_amplitude_source": ode["source"],
            "ode_fit_summary": ode["ode_fit_summary"],
            "cases": case_summaries,
            "outputs": {
                "comparison_sheet": str(outdir / "comparison_sheet.png"),
                "rmse_by_h": str(outdir / "rmse_by_h.png"),
                "rmse_by_h_vw": str(outdir / "rmse_by_h_vw.png"),
                "r_by_vw": str(outdir / "r_by_vw.png"),
            },
        }
        shared.save_json(outdir / "final_summary.json", summary)
        print(json.dumps(shared.to_native(summary), indent=2, sort_keys=True))
    except Exception as exc:
        error_exit(outdir, exc)
        raise


if __name__ == "__main__":
    main()
