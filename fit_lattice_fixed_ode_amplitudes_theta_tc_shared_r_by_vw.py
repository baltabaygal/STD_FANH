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

import fit_lattice_fixed_ode_amplitudes as shared
import fit_lattice_fixed_ode_amplitudes_theta_tc as theta_tc
import fit_lattice_quadwarp_universal as quad


ROOT = Path(__file__).resolve().parent
OUTDIR_DEFAULT = "results_lattice_fixed_ode_amplitudes_theta_tc_sharedr_by_vw_beta0_tcmax100"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]
REFERENCE_SUMMARY_DEFAULT = (
    ROOT / "results_lattice_fixed_ode_amplitudes_theta_tc_by_vw_beta0_tcmax100" / "final_summary.json"
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit all v_w together in canonical xi=(2 x / (3 t_osc))^(3/2) * f_tilde / F0 form, with shared global r, per-(v_w, theta0) t_c, and ODE-fixed F0,f_infty where f_infty=F_inf/F0."
    )
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--fixed-beta", type=float, default=0.0)
    p.add_argument("--t-osc", type=float, default=1.5)
    p.add_argument("--ode-summary", type=str, default=str(shared.ODE_SUMMARY_DEFAULT))
    p.add_argument("--reference-summary", type=str, default=str(REFERENCE_SUMMARY_DEFAULT))
    p.add_argument("--tc-max", type=float, default=100.0)
    p.add_argument(
        "--transient-kernel",
        type=str,
        choices=["power_inside", "shifted"],
        default="power_inside",
        help="Use 1 + (x/t_c)^r (`power_inside`) or (1 + x/t_c)^r (`shifted`) in the transient denominator.",
    )
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--outdir", type=str, default=OUTDIR_DEFAULT)
    return p.parse_args()


def error_exit(outdir: Path, exc: Exception):
    payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
    outdir.mkdir(parents=True, exist_ok=True)
    shared.save_json(outdir / "_error.json", payload)
    print(json.dumps(payload, sort_keys=True))


def load_reference_summary(path: Path | None):
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def build_meta(df, theta_values: np.ndarray, vw_values: np.ndarray, ode: dict, t_osc: float):
    theta_index = {float(theta): i for i, theta in enumerate(theta_values)}
    vw_index = {float(vw): i for i, vw in enumerate(vw_values)}
    theta_idx = np.array([theta_index[float(theta)] for theta in df["theta"].to_numpy(dtype=np.float64)], dtype=np.int64)
    vw_idx = np.array([vw_index[float(vw)] for vw in df["v_w"].to_numpy(dtype=np.float64)], dtype=np.int64)
    return {
        "theta": df["theta"].to_numpy(dtype=np.float64),
        "theta_idx": theta_idx,
        "theta_values": np.asarray(theta_values, dtype=np.float64),
        "v_w": df["v_w"].to_numpy(dtype=np.float64),
        "vw_idx": vw_idx,
        "vw_values": np.asarray(vw_values, dtype=np.float64),
        "tp": df["tp"].to_numpy(dtype=np.float64),
        "H": df["H"].to_numpy(dtype=np.float64),
        "beta_over_H": df["beta_over_H"].to_numpy(dtype=np.float64),
        "xi": df["xi"].to_numpy(dtype=np.float64),
        "F0": ode["F0"][theta_idx],
        "F0_sq": np.maximum(np.square(ode["F0"][theta_idx]), 1.0e-18),
        "F_inf": ode["F_inf"][theta_idx],
        "t_osc": float(t_osc),
    }


def unpack_params(params: np.ndarray, n_vw: int, n_theta: int, with_calib: bool):
    idx = 0
    r = float(params[idx])
    idx += 1
    tc_by_vw_theta = np.asarray(params[idx : idx + n_vw * n_theta], dtype=np.float64).reshape(n_vw, n_theta)
    idx += n_vw * n_theta
    c_vw = np.ones(n_vw, dtype=np.float64)
    if with_calib:
        c_vw = np.asarray(params[idx : idx + n_vw], dtype=np.float64)
        idx += n_vw
    return r, tc_by_vw_theta, c_vw


def kernel_label(kernel: str) -> str:
    if kernel == "power_inside":
        return r"$1 + (t_p/t_c)^r$"
    if kernel == "shifted":
        return r"$(1 + t_p/t_c)^r$"
    raise ValueError(f"Unknown transient kernel: {kernel}")


def kernel_denom_text(kernel: str) -> str:
    if kernel == "power_inside":
        return "D = 1 + (x/t_c)^r"
    if kernel == "shifted":
        return "D = (1 + x/t_c)^r"
    raise ValueError(f"Unknown transient kernel: {kernel}")


def xi_scale_from_x(x: np.ndarray, t_osc: float) -> np.ndarray:
    return np.power(
        np.maximum((2.0 * np.asarray(x, dtype=np.float64)) / max(3.0 * float(t_osc), 1.0e-18), 1.0e-18),
        1.5,
    )


def x_and_xi_scale(meta, fixed_beta: float):
    x = meta["tp"] * np.power(meta["H"], float(fixed_beta))
    xi_scale = xi_scale_from_x(x, meta["t_osc"])
    return x, xi_scale


def transition_denom(x: np.ndarray, tc: np.ndarray, r: float, kernel: str) -> np.ndarray:
    x_over_tc = np.maximum(x / np.maximum(tc, 1.0e-18), 1.0e-18)
    if kernel == "power_inside":
        return 1.0 + np.power(x_over_tc, r)
    if kernel == "shifted":
        return np.power(1.0 + x_over_tc, r)
    raise ValueError(f"Unknown transient kernel: {kernel}")


def transient_term(x: np.ndarray, tc: np.ndarray, r: float, kernel: str) -> np.ndarray:
    return 1.0 / transition_denom(x, tc, r, kernel)


def xi_from_f_tilde(xi_scale: np.ndarray, f_tilde: np.ndarray, f0: np.ndarray) -> np.ndarray:
    return xi_scale * f_tilde / np.maximum(f0, 1.0e-18)


def f_tilde_eval_ode_fixed(meta, fixed_beta: float, tc_by_vw_theta: np.ndarray, r: float, kernel: str, c_vw: np.ndarray):
    x, xi_scale = x_and_xi_scale(meta, fixed_beta)
    tc = tc_by_vw_theta[meta["vw_idx"], meta["theta_idx"]]
    denom = transition_denom(x, tc, r, kernel)
    f_tilde = meta["F_inf"] / np.maximum(meta["F0"], 1.0e-18) + meta["F0"] / np.maximum(xi_scale * denom, 1.0e-18)
    return x, xi_scale, c_vw[meta["vw_idx"]] * f_tilde


def model_eval(meta, fixed_beta: float, params: np.ndarray, with_calib: bool, kernel: str):
    n_vw = len(meta["vw_values"])
    n_theta = len(meta["theta_values"])
    r, tc_by_vw_theta, c_vw = unpack_params(params, n_vw, n_theta, with_calib)
    _, xi_scale, f_tilde = f_tilde_eval_ode_fixed(meta, fixed_beta, tc_by_vw_theta, r, kernel, c_vw)
    return xi_from_f_tilde(xi_scale, f_tilde, meta["F0"])


def rel_rmse(y, y_fit):
    y = np.asarray(y, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square((y_fit - y) / np.maximum(y, 1.0e-12)))))


def fit_case(
    meta,
    fixed_beta: float,
    init_r: float,
    init_tc_by_vw_theta: np.ndarray,
    init_c_vw: np.ndarray | None,
    tc_max: float,
    with_calib: bool,
    kernel: str,
):
    n_vw = len(meta["vw_values"])
    n_theta = len(meta["theta_values"])
    if np.asarray(init_tc_by_vw_theta).shape != (n_vw, n_theta):
        raise ValueError("Initial tc grid must match (n_vw, n_theta).")
    x0 = [float(init_r)]
    x0.extend(np.asarray(init_tc_by_vw_theta, dtype=np.float64).ravel().tolist())
    lower = [0.1]
    lower.extend([0.1] * (n_vw * n_theta))
    upper = [20.0]
    upper.extend([float(tc_max)] * (n_vw * n_theta))
    if with_calib:
        if init_c_vw is None or np.asarray(init_c_vw).shape != (n_vw,):
            raise ValueError("Initial c_vw vector must match n_vw for calibrated fit.")
        x0.extend(np.asarray(init_c_vw, dtype=np.float64).tolist())
        lower.extend([0.1] * n_vw)
        upper.extend([10.0] * n_vw)
    x0 = np.asarray(x0, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    def resid(par: np.ndarray) -> np.ndarray:
        y_fit = model_eval(meta, fixed_beta, par, with_calib, kernel)
        return (y_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)

    huber = least_squares(resid, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=50000)
    final = least_squares(resid, huber.x, bounds=(lower, upper), loss="linear", max_nfev=50000)
    y_fit = model_eval(meta, fixed_beta, final.x, with_calib, kernel)
    frac_resid = (y_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    aic, bic = shared.aic_bic(frac_resid, len(final.x))
    r, tc_by_vw_theta, c_vw = unpack_params(final.x, n_vw, n_theta, with_calib)
    payload = {
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "beta": float(fixed_beta),
        "transient_kernel": str(kernel),
        "canonical_xi_form": "xi = (2 x / (3 t_osc))^(3/2) * f_tilde / F0",
        "f_tilde_form": "f_tilde = c(v_w) * [f_infty(theta0) + F0(theta0) / ((2 x / (3 t_osc))^(3/2) * D)]",
        "transition_denominator": kernel_denom_text(kernel),
        "r": float(r),
        "tc_by_vw_theta": tc_by_vw_theta,
        "rel_rmse": rel_rmse(meta["xi"], y_fit),
        "AIC": float(aic),
        "BIC": float(bic),
        "n_points": int(meta["xi"].size),
        "n_params": int(len(final.x)),
        "result_x": np.asarray(final.x, dtype=np.float64),
        "y_fit": y_fit,
        "frac_resid": frac_resid,
        "per_vw_rel_rmse": {},
    }
    for i, vw in enumerate(meta["vw_values"]):
        mask = np.isclose(meta["v_w"], float(vw), atol=1.0e-12)
        payload["per_vw_rel_rmse"][f"{float(vw):.1f}"] = rel_rmse(meta["xi"][mask], y_fit[mask])
    if with_calib:
        payload["c_vw"] = c_vw
    return payload


def tc_grid_from_reference(ref_summary: dict | None, vw_values: np.ndarray, theta_values: np.ndarray, fit_key: str, default_tc: float) -> np.ndarray:
    out = np.full((len(vw_values), len(theta_values)), float(default_tc), dtype=np.float64)
    if ref_summary is None:
        return out
    fit_by_vw = ref_summary.get("fit_by_vw", {})
    for i, vw in enumerate(vw_values):
        block = fit_by_vw.get(f"{float(vw):.1f}", {})
        payload = block.get("calib" if fit_key == "calib" else "plain", {})
        table = payload.get("tc_theta", {})
        for j, theta in enumerate(theta_values):
            key = f"{float(theta):.10f}"
            if key in table:
                out[i, j] = float(table[key])
    return out


def c_vw_from_reference(ref_summary: dict | None, vw_values: np.ndarray) -> np.ndarray:
    out = np.ones(len(vw_values), dtype=np.float64)
    if ref_summary is None:
        return out
    fit_by_vw = ref_summary.get("fit_by_vw", {})
    for i, vw in enumerate(vw_values):
        block = fit_by_vw.get(f"{float(vw):.1f}", {})
        payload = block.get("calib", {})
        if "c_calib" in payload:
            out[i] = float(payload["c_calib"])
    return out


def r_from_reference(ref_summary: dict | None, fit_key: str, default_r: float) -> float:
    if ref_summary is None:
        return float(default_r)
    fit_by_vw = ref_summary.get("fit_by_vw", {})
    vals = []
    subkey = "calib" if fit_key == "calib" else "plain"
    for vw_key, block in fit_by_vw.items():
        payload = block.get(subkey, {})
        if "r" in payload:
            vals.append(float(payload["r"]))
    if not vals:
        return float(default_r)
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def summarize_payload(payload: dict, theta_values: np.ndarray, vw_values: np.ndarray, ode: dict):
    out = {k: v for k, v in payload.items() if k not in {"result_x", "y_fit", "frac_resid", "tc_by_vw_theta", "c_vw"}}
    out["tc_by_vw_theta"] = {}
    for i, vw in enumerate(vw_values):
        out["tc_by_vw_theta"][f"{float(vw):.1f}"] = {
            f"{float(theta):.10f}": float(payload["tc_by_vw_theta"][i, j])
            for j, theta in enumerate(theta_values)
        }
    if "c_vw" in payload:
        out["c_calib_by_vw"] = {f"{float(vw):.1f}": float(payload["c_vw"][i]) for i, vw in enumerate(vw_values)}
    out["F0_ode"] = {f"{float(theta):.10f}": float(val) for theta, val in zip(theta_values, ode["F0"])}
    out["F_infty_raw_ode"] = {f"{float(theta):.10f}": float(val) for theta, val in zip(theta_values, ode["F_inf"])}
    out["f_infty_ode"] = {
        f"{float(theta):.10f}": float(ode["F_inf"][i] / max(float(ode["F0"][i]), 1.0e-18))
        for i, theta in enumerate(theta_values)
    }
    return out


def build_prediction_frame(df, meta, fixed_beta: float, fit_payload: dict, with_calib: bool, kernel: str):
    n_vw = len(meta["vw_values"])
    n_theta = len(meta["theta_values"])
    r, tc_by_vw_theta, c_vw = unpack_params(fit_payload["result_x"], n_vw, n_theta, with_calib)
    x, xi_scale, f_tilde = f_tilde_eval_ode_fixed(meta, fixed_beta, tc_by_vw_theta, r, kernel, c_vw)
    y_fit = xi_from_f_tilde(xi_scale, f_tilde, meta["F0"])
    out = df.copy()
    out["x"] = x
    out["xi_fit"] = y_fit
    out["f_tilde_fit"] = f_tilde
    return out


def plot_collapse_overlay(df_pred, theta_values: np.ndarray, vw_values: np.ndarray, fit_payload: dict, outpath: Path, dpi: int, fixed_beta: float, title: str):
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
        ax.set_xlabel(r"$t_p$" if np.isclose(fixed_beta, 0.0, atol=1.0e-12) else r"$x=t_p H^\beta$")
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


def plot_raw(df_pred, theta_values: np.ndarray, vw_values: np.ndarray, outdir: Path, stem: str, dpi: int):
    cmap = plt.get_cmap("viridis")
    colors = {float(vw): cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    rows = []
    for h_value in sorted(df_pred["H"].unique()):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
        axes = axes.ravel()
        sub_h = df_pred[np.isclose(df_pred["H"], float(h_value), atol=1.0e-8)].copy()
        for ax, theta in zip(axes, theta_values):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=1.0e-8)].copy()
            for vw in vw_values:
                cur = sub[np.isclose(sub["v_w"], float(vw), atol=1.0e-8)].sort_values("beta_over_H")
                if cur.empty:
                    continue
                ax.scatter(cur["beta_over_H"], cur["xi"], s=22, color=colors[float(vw)], alpha=0.85)
                ax.plot(cur["beta_over_H"], cur["xi_fit"], color=colors[float(vw)], lw=1.8)
                rows.append(
                    {
                        "H": float(h_value),
                        "theta": float(theta),
                        "v_w": float(vw),
                        "rel_rmse": shared.rel_rmse(cur["xi"], cur["xi_fit"]),
                    }
                )
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$\beta/H_*$")
            ax.set_ylabel(r"$\xi$")
        for ax in axes[len(theta_values):]:
            ax.axis("off")
        handles = [plt.Line2D([0], [0], color=colors[float(vw)], lw=2.0) for vw in vw_values]
        labels = [rf"$v_w={float(vw):.1f}$" for vw in vw_values]
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
        fig.suptitle(rf"{stem}, $H_*={float(h_value):.1f}$", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        tag = str(float(h_value)).replace(".", "p")
        fig.savefig(outdir / f"{stem}_H{tag}.png", dpi=dpi)
        plt.close(fig)
    return rows


def plot_tc_by_vw(fit_payload: dict, theta_values: np.ndarray, vw_values: np.ndarray, outpath: Path, dpi: int):
    tc = np.asarray(fit_payload["tc_by_vw_theta"], dtype=np.float64)
    cmap = plt.get_cmap("viridis")
    colors = {float(vw): cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), constrained_layout=True)
    hvals = theta_tc.h_alt(theta_values)
    for i, vw in enumerate(vw_values):
        axes[0].plot(theta_values, tc[i], "o-", ms=4.0, lw=1.4, color=colors[float(vw)], label=rf"$v_w={float(vw):.1f}$")
        axes[1].plot(hvals, tc[i], "o-", ms=4.0, lw=1.4, color=colors[float(vw)], label=rf"$v_w={float(vw):.1f}$")
    axes[0].set_xlabel(r"$\theta_0$")
    axes[0].set_ylabel(r"$t_c(\theta_0; v_w)$")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel(r"$h(\theta_0)=\log\!\left(e/[1-(\theta/\pi)^2]\right)$")
    axes[1].set_ylabel(r"$t_c(\theta_0; v_w)$")
    axes[1].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_calib_by_vw(fit_payload: dict, vw_values: np.ndarray, outpath: Path, dpi: int):
    c_vw = np.asarray(fit_payload["c_vw"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(vw_values, c_vw, "o-", lw=1.6, color="tab:green")
    ax.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$c_{calib}(v_w)$")
    ax.grid(alpha=0.25)
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
        meta = build_meta(df, theta_values, vw_values, ode, args.t_osc)
        ref_summary = load_reference_summary(Path(args.reference_summary).resolve() if args.reference_summary else None)

        init_tc_plain = tc_grid_from_reference(ref_summary, vw_values, theta_values, "plain", default_tc=2.0)
        init_tc_calib = tc_grid_from_reference(ref_summary, vw_values, theta_values, "calib", default_tc=2.0)
        init_r_plain = r_from_reference(ref_summary, "plain", default_r=2.0)
        init_r_calib = r_from_reference(ref_summary, "calib", default_r=init_r_plain)
        init_c_vw = c_vw_from_reference(ref_summary, vw_values)

        fit_plain = fit_case(
            meta,
            args.fixed_beta,
            init_r_plain,
            init_tc_plain,
            None,
            args.tc_max,
            with_calib=False,
            kernel=args.transient_kernel,
        )
        fit_calib = fit_case(
            meta,
            args.fixed_beta,
            init_r_calib,
            init_tc_calib if ref_summary is not None else np.asarray(fit_plain["tc_by_vw_theta"], dtype=np.float64),
            init_c_vw,
            args.tc_max,
            with_calib=True,
            kernel=args.transient_kernel,
        )

        pred_plain = build_prediction_frame(df, meta, args.fixed_beta, fit_plain, with_calib=False, kernel=args.transient_kernel)
        pred_calib = build_prediction_frame(df, meta, args.fixed_beta, fit_calib, with_calib=True, kernel=args.transient_kernel)
        pred_plain.to_csv(outdir / "predictions_plain.csv", index=False)
        pred_calib.to_csv(outdir / "predictions_calib.csv", index=False)

        plot_collapse_overlay(
            pred_plain,
            theta_values,
            vw_values,
            fit_plain,
            outdir / "collapse_overlay_sharedr_plain.png",
            args.dpi,
            args.fixed_beta,
            rf"Shared-$r$ all-$v_w$ fit with ODE-fixed $F_0,f_\infty(\theta_0)$, $t_c(\theta_0; v_w)$, kernel {kernel_label(args.transient_kernel)}, $r={fit_plain['r']:.3f}$",
        )
        plot_collapse_overlay(
            pred_calib,
            theta_values,
            vw_values,
            fit_calib,
            outdir / "collapse_overlay_sharedr_calib.png",
            args.dpi,
            args.fixed_beta,
            rf"Shared-$r$ all-$v_w$ fit with ODE-fixed $F_0,f_\infty(\theta_0)$, $t_c(\theta_0; v_w)$, $c_{{calib}}(v_w)$, kernel {kernel_label(args.transient_kernel)}, $r={fit_calib['r']:.3f}$",
        )
        raw_plain = plot_raw(pred_plain, theta_values, vw_values, outdir, "xi_vs_betaH_sharedr_plain", args.dpi)
        raw_calib = plot_raw(pred_calib, theta_values, vw_values, outdir, "xi_vs_betaH_sharedr_calib", args.dpi)
        plot_tc_by_vw(fit_plain, theta_values, vw_values, outdir / "tc_by_vw_sharedr_plain.png", args.dpi)
        plot_tc_by_vw(fit_calib, theta_values, vw_values, outdir / "tc_by_vw_sharedr_calib.png", args.dpi)
        plot_calib_by_vw(fit_calib, vw_values, outdir / "c_calib_by_vw_sharedr.png", args.dpi)

        plain_payload = summarize_payload(fit_plain, theta_values, vw_values, ode)
        calib_payload = summarize_payload(fit_calib, theta_values, vw_values, ode)
        summary = {
            "status": "ok",
            "vw_values": [float(v) for v in vw_values],
            "theta_values": [float(v) for v in theta_values],
            "fixed_beta": float(args.fixed_beta),
            "tc_max": float(args.tc_max),
            "transient_kernel": str(args.transient_kernel),
            "n_points": int(len(df)),
            "ode_amplitude_source": ode["source"],
            "ode_fit_summary": ode["ode_fit_summary"],
            "shared_r_plain": plain_payload,
            "shared_r_calib": calib_payload,
            "raw_plot_rmse_plain": raw_plain,
            "raw_plot_rmse_calib": raw_calib,
            "outputs": {
                "collapse_plain": str(outdir / "collapse_overlay_sharedr_plain.png"),
                "collapse_calib": str(outdir / "collapse_overlay_sharedr_calib.png"),
                "tc_plain": str(outdir / "tc_by_vw_sharedr_plain.png"),
                "tc_calib": str(outdir / "tc_by_vw_sharedr_calib.png"),
                "c_calib": str(outdir / "c_calib_by_vw_sharedr.png"),
                "predictions_plain": str(outdir / "predictions_plain.csv"),
                "predictions_calib": str(outdir / "predictions_calib.csv"),
            },
        }
        shared.save_json(outdir / "fit_shared_r_plain.json", plain_payload)
        shared.save_json(outdir / "fit_shared_r_calib.json", calib_payload)
        shared.save_json(outdir / "final_summary.json", summary)
        print(json.dumps(shared.to_native(summary), indent=2, sort_keys=True))
    except Exception as exc:
        error_exit(outdir, exc)
        raise


if __name__ == "__main__":
    main()
