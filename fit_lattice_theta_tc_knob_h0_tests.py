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
OUTDIR_DEFAULT = ROOT / "knob_tc_h0_weibull"
REFERENCE_DIR_DEFAULT = ROOT / "results_lattice_theta_tc_rvw_tests_beta0_tcmax300_weibull"
PREVIOUS_KNOB_DIR_DEFAULT = ROOT / "knob_tc_weibull"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]

CASE_CONFIGS = [
    ("rvw_theta_hgamma", r"$r(v_w)$, $t_c^{(0)}(v_w)\,h(\theta_0)^\gamma$", "rvw_theta_hgamma"),
    ("rvw_theta_h0_gammavw", r"$r(v_w)$, $t_c^{(0)}(v_w)\,(h(\theta_0)+h_0)^{\gamma(v_w)}$", "rvw_theta_h0_gammavw"),
    ("rvw_theta", r"$r(v_w)$, $t_c(\theta_0,v_w)$", "rvw_theta"),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Test tc(theta0,vw)=tc0(vw) * (h(theta0)+h0)^(gamma(vw)) against the old h(theta)^gamma knob and the full tc(theta0,vw) table."
    )
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--fixed-beta", type=float, default=0.0)
    p.add_argument("--t-osc", type=float, default=1.5)
    p.add_argument("--ode-summary", type=str, default=str(shared.ODE_SUMMARY_DEFAULT))
    p.add_argument("--reference-dir", type=str, default=str(REFERENCE_DIR_DEFAULT))
    p.add_argument("--previous-knob-dir", type=str, default=str(PREVIOUS_KNOB_DIR_DEFAULT))
    p.add_argument("--tc-max", type=float, default=300.0)
    p.add_argument(
        "--transient-kernel",
        type=str,
        choices=["power_inside", "shifted", "weibull"],
        default="weibull",
    )
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


def tc_grid_from_case(case_summary: dict, theta_values: np.ndarray, vw_values: np.ndarray):
    table = case_summary["tc_by_vw_theta"]
    out = np.empty((len(vw_values), len(theta_values)), dtype=np.float64)
    for i, vw in enumerate(vw_values):
        block = table[f"{float(vw):.1f}"]
        for j, theta in enumerate(theta_values):
            out[i, j] = float(block[f"{float(theta):.10f}"])
    return out


def f_infty_from_case(case_summary: dict, theta_values: np.ndarray):
    table = case_summary["f_infty_lattice"]
    return np.asarray([float(table[f"{float(theta):.10f}"]) for theta in theta_values], dtype=np.float64)


def r_by_vw_from_case(case_summary: dict, vw_values: np.ndarray):
    table = case_summary["r_by_vw"]
    return np.asarray([float(table[f"{float(vw):.1f}"]) for vw in vw_values], dtype=np.float64)


def tc_shifted_init(tc_grid: np.ndarray, theta_values: np.ndarray):
    tc_grid = np.asarray(tc_grid, dtype=np.float64)
    h_vals = np.maximum(h_theta(theta_values), 1.0e-12)
    log_h = np.log(h_vals)
    n_vw = tc_grid.shape[0]
    gamma_by_vw = np.empty(n_vw, dtype=np.float64)
    tc0_by_vw = np.empty(n_vw, dtype=np.float64)
    for i in range(n_vw):
        coeff = np.polyfit(log_h, np.log(np.maximum(tc_grid[i], 1.0e-18)), deg=1)
        gamma_by_vw[i] = float(coeff[0])
        tc0_by_vw[i] = float(np.exp(coeff[1]))
    h0 = 0.0
    return tc0_by_vw, gamma_by_vw, h0


def unpack_params(params: np.ndarray, n_vw: int, n_theta: int):
    idx = 0
    r_by_vw = np.asarray(params[idx : idx + n_vw], dtype=np.float64)
    idx += n_vw
    tc0_by_vw = np.asarray(params[idx : idx + n_vw], dtype=np.float64)
    idx += n_vw
    gamma_by_vw = np.asarray(params[idx : idx + n_vw], dtype=np.float64)
    idx += n_vw
    h0 = float(params[idx])
    idx += 1
    f_infty_theta = np.asarray(params[idx : idx + n_theta], dtype=np.float64)
    return r_by_vw, tc0_by_vw, gamma_by_vw, h0, f_infty_theta


def build_param_vector(init_r_by_vw, init_tc0_by_vw, init_gamma_by_vw, init_h0, init_f_infty):
    r0 = np.asarray(init_r_by_vw, dtype=np.float64)
    tc0 = np.asarray(init_tc0_by_vw, dtype=np.float64)
    gamma0 = np.asarray(init_gamma_by_vw, dtype=np.float64)
    finf0 = np.asarray(init_f_infty, dtype=np.float64)
    x0 = np.concatenate([r0, tc0, gamma0, np.asarray([float(init_h0)]), finf0])
    lower = np.concatenate(
        [
            np.full(r0.size, 0.1, dtype=np.float64),
            np.full(tc0.size, 0.1, dtype=np.float64),
            np.full(gamma0.size, -5.0, dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            np.full(finf0.size, 1.0e-8, dtype=np.float64),
        ]
    )
    upper = np.concatenate(
        [
            np.full(r0.size, 20.0, dtype=np.float64),
            np.full(tc0.size, np.inf, dtype=np.float64),
            np.full(gamma0.size, 5.0, dtype=np.float64),
            np.asarray([50.0], dtype=np.float64),
            np.full(finf0.size, 1.0e4, dtype=np.float64),
        ]
    )
    return x0, lower, upper


def model_details(meta, fixed_beta: float, params: np.ndarray, kernel: str):
    n_vw = len(meta["vw_values"])
    n_theta = len(meta["theta_values"])
    r_by_vw, tc0_by_vw, gamma_by_vw, h0, f_infty_theta = unpack_params(params, n_vw, n_theta)
    x, xi_scale = shared_rvw.x_and_xi_scale(meta, fixed_beta)
    r_eff = r_by_vw[meta["vw_idx"]]
    gamma_eff = gamma_by_vw[meta["vw_idx"]]
    h_shift_row = np.maximum(meta["h_theta_row"] + float(h0), 1.0e-18)
    tc_eff = tc0_by_vw[meta["vw_idx"]] * np.power(h_shift_row, gamma_eff)
    denom = shared_rvw.transition_denom(x, tc_eff, r_eff, kernel)
    f_tilde = f_infty_theta[meta["theta_idx"]] + meta["F0"] / np.maximum(xi_scale * denom, 1.0e-18)
    xi_fit = shared_rvw.xi_from_f_tilde(xi_scale, f_tilde, meta["F0"])
    tc_by_vw_theta = tc0_by_vw[:, None] * np.power(
        np.maximum(meta["h_theta_values"][None, :] + float(h0), 1.0e-18),
        gamma_by_vw[:, None],
    )
    return x, xi_scale, tc_eff, r_eff, f_tilde, xi_fit, r_by_vw, tc0_by_vw, gamma_by_vw, h0, f_infty_theta, tc_by_vw_theta


def fit_case(meta, fixed_beta: float, init_r_by_vw, init_tc0_by_vw, init_gamma_by_vw, init_h0, init_f_infty, tc_max: float, kernel: str):
    x0, lower, upper = build_param_vector(init_r_by_vw, init_tc0_by_vw, init_gamma_by_vw, init_h0, init_f_infty)
    n_vw = len(meta["vw_values"])
    tc_slice = slice(n_vw, 2 * n_vw)
    upper[tc_slice] = float(tc_max)

    def resid(par: np.ndarray) -> np.ndarray:
        _, _, _, _, _, xi_fit, *_ = model_details(meta, fixed_beta, par, kernel)
        return (xi_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)

    huber = least_squares(resid, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=90000)
    final = least_squares(resid, huber.x, bounds=(lower, upper), loss="linear", max_nfev=90000)
    x, xi_scale, tc_eff, r_eff, f_tilde, xi_fit, r_by_vw, tc0_by_vw, gamma_by_vw, h0, f_infty_theta, tc_by_vw_theta = model_details(
        meta, fixed_beta, final.x, kernel
    )
    frac_resid = (xi_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    aic, bic = shared.aic_bic(frac_resid, len(final.x))
    payload = {
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "mode": "rvw_theta_h0_gammavw",
        "beta": float(fixed_beta),
        "transient_kernel": str(kernel),
        "canonical_xi_form": "xi = (2 x / (3 t_osc))^(3/2) * f_tilde / F0",
        "transition_denominator": shared_rvw.kernel_denom_text(kernel),
        "f_tilde_form": "f_tilde = f_infty(theta0) + F0(theta0) / ((2 x / (3 t_osc))^(3/2) * D), with tc(theta0,v_w) = tc0(v_w) * (h(theta0)+h0)^(gamma(v_w)) and r = r(v_w)",
        "h_theta_form": "h(theta0) = log(e / cos(theta0/2)^2)",
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
        "r_by_vw": np.asarray(r_by_vw, dtype=np.float64),
        "tc0_by_vw": np.asarray(tc0_by_vw, dtype=np.float64),
        "gamma_by_vw": np.asarray(gamma_by_vw, dtype=np.float64),
        "h0": float(h0),
        "tc_by_vw_theta": np.asarray(tc_by_vw_theta, dtype=np.float64),
        "per_vw_rel_rmse": {},
    }
    for i, vw in enumerate(meta["vw_values"]):
        mask = np.isclose(meta["v_w"], float(vw), atol=1.0e-12)
        payload["per_vw_rel_rmse"][f"{float(vw):.1f}"] = shared_rvw.rel_rmse(meta["xi"][mask], xi_fit[mask])
    return payload


def build_prediction_frame(df, meta, fit_payload: dict):
    out = df.copy()
    out["x"] = fit_payload["x_fit"]
    out["xi_fit"] = fit_payload["y_fit"]
    out["f_tilde_fit"] = fit_payload["f_tilde_fit"]
    out["tc_eff_fit"] = fit_payload["tc_eff_fit"]
    out["r_eff_fit"] = fit_payload["r_eff_fit"]
    n_vw = len(meta["vw_values"])
    n_theta = len(meta["theta_values"])
    _, _, _, _, f_infty_theta = unpack_params(fit_payload["result_x"], n_vw, n_theta)
    out["f_infty_fit"] = f_infty_theta[np.asarray(meta["theta_idx"], dtype=np.int64)]
    return out


def summarize_payload(payload: dict, theta_values: np.ndarray, vw_values: np.ndarray, ode: dict):
    out = {
        k: v
        for k, v in payload.items()
        if k
        not in {
            "result_x",
            "y_fit",
            "f_tilde_fit",
            "x_fit",
            "tc_eff_fit",
            "r_eff_fit",
            "frac_resid",
            "f_infty",
            "r_by_vw",
            "tc0_by_vw",
            "gamma_by_vw",
            "tc_by_vw_theta",
        }
    }
    out["r_by_vw"] = {f"{float(vw):.1f}": float(payload["r_by_vw"][i]) for i, vw in enumerate(vw_values)}
    out["tc0_by_vw"] = {f"{float(vw):.1f}": float(payload["tc0_by_vw"][i]) for i, vw in enumerate(vw_values)}
    out["gamma_by_vw"] = {f"{float(vw):.1f}": float(payload["gamma_by_vw"][i]) for i, vw in enumerate(vw_values)}
    out["tc_by_vw_theta"] = {}
    for i, vw in enumerate(vw_values):
        out["tc_by_vw_theta"][f"{float(vw):.1f}"] = {
            f"{float(theta):.10f}": float(payload["tc_by_vw_theta"][i, j]) for j, theta in enumerate(theta_values)
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
        vals = [case_summaries[mode]["r_by_vw"][f"{vw:.1f}"] for vw in vw_values]
        ax.plot(vw_values, vals, "o-", lw=1.8, label=label)
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$r(v_w)$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_gamma_by_vw(case_summaries: dict, outpath: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    vw_values = [0.3, 0.5, 0.7, 0.9]
    for mode, label, _ in CASE_CONFIGS:
        summary = case_summaries[mode]
        if "gamma_by_vw" in summary:
            vals = [summary["gamma_by_vw"][f"{vw:.1f}"] for vw in vw_values]
            ax.plot(vw_values, vals, "o-", lw=1.8, label=label)
        elif "gamma" in summary:
            vals = [summary["gamma"]] * len(vw_values)
            ax.plot(vw_values, vals, "o--", lw=1.6, label=label)
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$\gamma$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_tc0_by_vw(case_summaries: dict, outpath: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    vw_values = [0.3, 0.5, 0.7, 0.9]
    for mode, label, _ in CASE_CONFIGS:
        summary = case_summaries[mode]
        if "tc0_by_vw" not in summary:
            continue
        vals = [summary["tc0_by_vw"][f"{vw:.1f}"] for vw in vw_values]
        ax.plot(vw_values, vals, "o-", lw=1.8, label=label)
    new_summary = case_summaries.get("rvw_theta_h0_gammavw", {})
    title = rf"$h_0={new_summary.get('h0', float('nan')):.3f}$" if "h0" in new_summary else ""
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$t_c^{(0)}(v_w)$")
    ax.set_title(title)
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
    fig.suptitle(r"Shifted-$h$ knob comparison", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def copy_case(src_root: Path, outdir: Path, mode: str):
    src = src_root / mode
    dst = outdir / mode
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
        previous_knob_dir = Path(args.previous_knob_dir).resolve()
        reference_summary = json.loads((reference_dir / "final_summary.json").read_text())
        previous_knob_summary = json.loads((previous_knob_dir / "final_summary.json").read_text())

        df = quad.load_lattice_dataframe(outdir, args.vw_folders, args.h_values)
        theta_values = np.sort(df["theta"].unique())
        vw_values = np.sort(df["v_w"].unique())
        h_values = np.sort(df["H"].unique())
        ode = shared.load_ode_amplitudes(Path(args.ode_summary).resolve(), theta_values)
        meta = shared_rvw.build_meta(df, theta_values, vw_values, ode, args.t_osc)
        meta["h_theta_values"] = h_theta(theta_values)
        meta["h_theta_row"] = meta["h_theta_values"][np.asarray(meta["theta_idx"], dtype=np.int64)]

        ref_theta = reference_summary["cases"]["rvw_theta"]
        init_r_by_vw = r_by_vw_from_case(ref_theta, vw_values)
        init_tc_grid = tc_grid_from_case(ref_theta, theta_values, vw_values)
        init_tc0_by_vw, init_gamma_by_vw, init_h0 = tc_shifted_init(init_tc_grid, theta_values)
        init_f_infty = f_infty_from_case(ref_theta, theta_values)

        fit_payload = fit_case(
            meta,
            args.fixed_beta,
            init_r_by_vw,
            init_tc0_by_vw,
            init_gamma_by_vw,
            init_h0,
            init_f_infty,
            args.tc_max,
            args.transient_kernel,
        )

        new_dir = outdir / "rvw_theta_h0_gammavw"
        new_dir.mkdir(parents=True, exist_ok=True)
        pred = build_prediction_frame(df, meta, fit_payload)
        pred.to_csv(new_dir / "predictions.csv", index=False)
        shared_rvw.plot_collapse_overlay(
            pred,
            theta_values,
            vw_values,
            fit_payload,
            new_dir / "collapse_overlay.png",
            args.dpi,
            args.fixed_beta,
            rf"$r(v_w)$, $t_c^{{(0)}}(v_w)\,(h(\theta_0)+h_0)^{{\gamma(v_w)}}$, kernel {shared_rvw.kernel_label(args.transient_kernel)}",
        )
        shared_rvw.plot_raw(pred, theta_values, vw_values, new_dir, "xi_vs_betaH", args.dpi)
        raw_rows, mean_by_h, mean_by_h_vw = rmse_tables(pred)
        new_summary = summarize_payload(fit_payload, theta_values, vw_values, ode)
        new_summary["label"] = r"$r(v_w)$, $t_c^{(0)}(v_w)\,(h(\theta_0)+h_0)^{\gamma(v_w)}$"
        new_summary["mean_raw_rmse_by_h"] = mean_by_h
        new_summary["mean_raw_rmse_by_h_vw"] = mean_by_h_vw
        new_summary["raw_plot_rmse"] = raw_rows
        save_json(new_dir / "final_summary.json", new_summary)

        case_summaries = {
            "rvw_theta_hgamma": previous_knob_summary["cases"]["rvw_theta_hgamma"],
            "rvw_theta_h0_gammavw": new_summary,
            "rvw_theta": reference_summary["cases"]["rvw_theta"],
        }
        case_dirs = {
            "rvw_theta_hgamma": copy_case(previous_knob_dir, outdir, "rvw_theta_hgamma"),
            "rvw_theta_h0_gammavw": new_dir,
            "rvw_theta": copy_case(reference_dir, outdir, "rvw_theta"),
        }

        plot_rmse_by_h(case_summaries, outdir / "rmse_by_h.png", args.dpi)
        plot_rmse_by_h_vw(case_summaries, outdir / "rmse_by_h_vw.png", args.dpi)
        plot_r_by_vw(case_summaries, outdir / "r_by_vw.png", args.dpi)
        plot_gamma_by_vw(case_summaries, outdir / "gamma_by_vw.png", args.dpi)
        plot_tc0_by_vw(case_summaries, outdir / "tc0_by_vw.png", args.dpi)
        make_comparison_sheet(case_dirs, outdir / "comparison_sheet.png", args.dpi)

        summary = {
            "status": "ok",
            "vw_values": [float(v) for v in vw_values],
            "theta_values": [float(v) for v in theta_values],
            "h_values": [float(v) for v in h_values],
            "fixed_beta": float(args.fixed_beta),
            "tc_max": float(args.tc_max),
            "transient_kernel": str(args.transient_kernel),
            "h_theta_form": "h(theta0) = log(e / cos(theta0/2)^2)",
            "shifted_h_form": "tc(theta0,v_w) = tc0(v_w) * (h(theta0) + h0)^(gamma(v_w))",
            "n_points": int(len(df)),
            "ode_amplitude_source": ode["source"],
            "ode_fit_summary": ode["ode_fit_summary"],
            "reference_dir": str(reference_dir),
            "previous_knob_dir": str(previous_knob_dir),
            "cases": case_summaries,
            "outputs": {
                "comparison_sheet": str(outdir / "comparison_sheet.png"),
                "rmse_by_h": str(outdir / "rmse_by_h.png"),
                "rmse_by_h_vw": str(outdir / "rmse_by_h_vw.png"),
                "r_by_vw": str(outdir / "r_by_vw.png"),
                "gamma_by_vw": str(outdir / "gamma_by_vw.png"),
                "tc0_by_vw": str(outdir / "tc0_by_vw.png"),
            },
        }
        save_json(outdir / "final_summary.json", summary)
        print(json.dumps(shared.to_native(summary), indent=2, sort_keys=True))
    except Exception as exc:
        error_exit(outdir, exc)
        raise


if __name__ == "__main__":
    main()
