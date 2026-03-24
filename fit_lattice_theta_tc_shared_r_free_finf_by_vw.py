#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

import fit_lattice_fixed_ode_amplitudes as shared
import fit_lattice_fixed_ode_amplitudes_theta_tc as theta_tc
import fit_lattice_fixed_ode_amplitudes_theta_tc_shared_r_by_vw as shared_rvw
import fit_lattice_quadwarp_universal as quad


ROOT = Path(__file__).resolve().parent
OUTDIR_DEFAULT = "results_lattice_theta_tc_sharedr_free_finf_by_vw_beta0_tcmax300"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]
REFERENCE_SUMMARY_DEFAULT = (
    ROOT / "results_lattice_fixed_ode_amplitudes_theta_tc_by_vw_beta0_tcmax100" / "final_summary.json"
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit all v_w together in canonical xi=(2 x / (3 t_osc))^(3/2) * f_tilde / F0 form, with shared global r, per-(v_w,theta0) t_c, ODE-fixed F0(theta), and lattice-fit f_infty(theta)=F_inf(theta)/F0(theta)."
    )
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--fixed-beta", type=float, default=0.0)
    p.add_argument("--t-osc", type=float, default=1.5)
    p.add_argument("--ode-summary", type=str, default=str(shared.ODE_SUMMARY_DEFAULT))
    p.add_argument("--reference-summary", type=str, default=str(REFERENCE_SUMMARY_DEFAULT))
    p.add_argument("--tc-max", type=float, default=300.0)
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


def unpack_params(params: np.ndarray, n_vw: int, n_theta: int):
    idx = 0
    r = float(params[idx])
    idx += 1
    tc_by_vw_theta = np.asarray(params[idx : idx + n_vw * n_theta], dtype=np.float64).reshape(n_vw, n_theta)
    idx += n_vw * n_theta
    f_infty_theta = np.asarray(params[idx : idx + n_theta], dtype=np.float64)
    return r, tc_by_vw_theta, f_infty_theta


def f_tilde_eval_free_finf(meta, fixed_beta: float, tc_by_vw_theta: np.ndarray, r: float, f_infty_theta: np.ndarray, kernel: str):
    x, xi_scale = shared_rvw.x_and_xi_scale(meta, fixed_beta)
    tc = tc_by_vw_theta[meta["vw_idx"], meta["theta_idx"]]
    denom = shared_rvw.transition_denom(x, tc, r, kernel)
    f_tilde = f_infty_theta[meta["theta_idx"]] + meta["F0"] / np.maximum(xi_scale * denom, 1.0e-18)
    return x, xi_scale, f_tilde


def model_eval(meta, fixed_beta: float, params: np.ndarray, kernel: str):
    n_vw = len(meta["vw_values"])
    n_theta = len(meta["theta_values"])
    r, tc_by_vw_theta, f_infty_theta = unpack_params(params, n_vw, n_theta)
    _, xi_scale, f_tilde = f_tilde_eval_free_finf(meta, fixed_beta, tc_by_vw_theta, r, f_infty_theta, kernel)
    return shared_rvw.xi_from_f_tilde(xi_scale, f_tilde, meta["F0"])


def fit_case(meta, fixed_beta: float, init_r: float, init_tc_by_vw_theta: np.ndarray, init_f_infty: np.ndarray, tc_max: float, kernel: str):
    n_vw = len(meta["vw_values"])
    n_theta = len(meta["theta_values"])
    if np.asarray(init_tc_by_vw_theta).shape != (n_vw, n_theta):
        raise ValueError("Initial tc grid must match (n_vw, n_theta).")
    if np.asarray(init_f_infty).shape != (n_theta,):
        raise ValueError("Initial f_infty vector must match n_theta.")

    x0 = np.concatenate(
        [
            np.array([float(init_r)], dtype=np.float64),
            np.asarray(init_tc_by_vw_theta, dtype=np.float64).ravel(),
            np.asarray(init_f_infty, dtype=np.float64),
        ]
    )
    lower = np.concatenate(
        [
            np.array([0.1], dtype=np.float64),
            np.full(n_vw * n_theta, 0.1, dtype=np.float64),
            np.full(n_theta, 1.0e-8, dtype=np.float64),
        ]
    )
    upper = np.concatenate(
        [
            np.array([20.0], dtype=np.float64),
            np.full(n_vw * n_theta, float(tc_max), dtype=np.float64),
            np.full(n_theta, 1.0e4, dtype=np.float64),
        ]
    )

    def resid(par: np.ndarray) -> np.ndarray:
        y_fit = model_eval(meta, fixed_beta, par, kernel)
        return (y_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)

    huber = least_squares(resid, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=50000)
    final = least_squares(resid, huber.x, bounds=(lower, upper), loss="linear", max_nfev=50000)
    y_fit = model_eval(meta, fixed_beta, final.x, kernel)
    frac_resid = (y_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    aic, bic = shared.aic_bic(frac_resid, len(final.x))
    r, tc_by_vw_theta, f_infty_theta = unpack_params(final.x, n_vw, n_theta)
    payload = {
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "beta": float(fixed_beta),
        "transient_kernel": str(kernel),
        "canonical_xi_form": "xi = (2 x / (3 t_osc))^(3/2) * f_tilde / F0",
        "f_tilde_form": "f_tilde = f_infty(theta0) + F0(theta0) / ((2 x / (3 t_osc))^(3/2) * D)",
        "transition_denominator": shared_rvw.kernel_denom_text(kernel),
        "r": float(r),
        "tc_by_vw_theta": tc_by_vw_theta,
        "f_infty": f_infty_theta,
        "rel_rmse": shared_rvw.rel_rmse(meta["xi"], y_fit),
        "AIC": float(aic),
        "BIC": float(bic),
        "n_points": int(meta["xi"].size),
        "n_params": int(len(final.x)),
        "result_x": np.asarray(final.x, dtype=np.float64),
        "y_fit": y_fit,
        "frac_resid": frac_resid,
        "per_vw_rel_rmse": {},
    }
    for vw in meta["vw_values"]:
        mask = np.isclose(meta["v_w"], float(vw), atol=1.0e-12)
        payload["per_vw_rel_rmse"][f"{float(vw):.1f}"] = shared_rvw.rel_rmse(meta["xi"][mask], y_fit[mask])
    return payload


def summarize_payload(payload: dict, theta_values: np.ndarray, vw_values: np.ndarray, ode: dict):
    out = {k: v for k, v in payload.items() if k not in {"result_x", "y_fit", "frac_resid", "tc_by_vw_theta", "f_infty"}}
    out["tc_by_vw_theta"] = {}
    for i, vw in enumerate(vw_values):
        out["tc_by_vw_theta"][f"{float(vw):.1f}"] = {
            f"{float(theta):.10f}": float(payload["tc_by_vw_theta"][i, j])
            for j, theta in enumerate(theta_values)
        }
    out["f_infty_lattice"] = {
        f"{float(theta):.10f}": float(payload["f_infty"][i]) for i, theta in enumerate(theta_values)
    }
    out["f_infty_ode"] = {
        f"{float(theta):.10f}": float(ode["F_inf"][i] / max(float(ode["F0"][i]), 1.0e-18))
        for i, theta in enumerate(theta_values)
    }
    out["F_infty_raw_lattice"] = {
        f"{float(theta):.10f}": float(payload["f_infty"][i] * float(ode["F0"][i]))
        for i, theta in enumerate(theta_values)
    }
    out["F_infty_raw_ode"] = {f"{float(theta):.10f}": float(val) for theta, val in zip(theta_values, ode["F_inf"])}
    out["f_infty_ratio_lattice_over_ode"] = {
        f"{float(theta):.10f}": float(payload["f_infty"][i] / max(float(ode["F_inf"][i] / max(float(ode["F0"][i]), 1.0e-18)), 1.0e-18))
        for i, theta in enumerate(theta_values)
    }
    out["F0_ode"] = {f"{float(theta):.10f}": float(val) for theta, val in zip(theta_values, ode["F0"])}
    return out


def build_prediction_frame(df, meta, fixed_beta: float, fit_payload: dict, kernel: str):
    n_vw = len(meta["vw_values"])
    n_theta = len(meta["theta_values"])
    r, tc_by_vw_theta, f_infty_theta = unpack_params(fit_payload["result_x"], n_vw, n_theta)
    x, xi_scale, f_tilde = f_tilde_eval_free_finf(meta, fixed_beta, tc_by_vw_theta, r, f_infty_theta, kernel)
    y_fit = shared_rvw.xi_from_f_tilde(xi_scale, f_tilde, meta["F0"])
    out = df.copy()
    out["x"] = x
    out["xi_fit"] = y_fit
    out["f_tilde_fit"] = f_tilde
    out["f_infty_fit"] = f_infty_theta[meta["theta_idx"]]
    return out


def plot_f_infty_theta(fit_payload: dict, theta_values: np.ndarray, ode: dict, outpath: Path, dpi: int):
    f_infty = np.asarray(fit_payload["f_infty"], dtype=np.float64)
    f_infty_ode = np.asarray(ode["F_inf"], dtype=np.float64) / np.maximum(np.asarray(ode["F0"], dtype=np.float64), 1.0e-18)
    hvals = theta_tc.h_alt(theta_values)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), constrained_layout=True)
    axes[0].plot(theta_values, f_infty_ode, "k--", lw=1.5, label="ODE")
    axes[0].plot(theta_values, f_infty, "o-", ms=4.2, lw=1.5, color="tab:blue", label="lattice fit")
    axes[0].set_xlabel(r"$\theta_0$")
    axes[0].set_ylabel(r"$f_\infty(\theta_0)$")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    ratio = f_infty / np.maximum(f_infty_ode, 1.0e-18)
    axes[1].plot(hvals, ratio, "o-", ms=4.2, lw=1.5, color="tab:red")
    axes[1].axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.7)
    axes[1].set_xlabel(r"$h(\theta_0)=\log\!\left(e/[1-(\theta/\pi)^2]\right)$")
    axes[1].set_ylabel(r"$f_\infty^{lattice}/f_\infty^{ODE}$")
    axes[1].grid(alpha=0.25)
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
        ref_summary = shared_rvw.load_reference_summary(Path(args.reference_summary).resolve() if args.reference_summary else None)

        init_tc = shared_rvw.tc_grid_from_reference(ref_summary, vw_values, theta_values, "plain", default_tc=2.0)
        init_r = shared_rvw.r_from_reference(ref_summary, "plain", default_r=2.0)
        init_f_infty = np.asarray(ode["F_inf"], dtype=np.float64) / np.maximum(np.asarray(ode["F0"], dtype=np.float64), 1.0e-18)

        fit_payload = fit_case(meta, args.fixed_beta, init_r, init_tc, init_f_infty, args.tc_max, args.transient_kernel)
        pred = build_prediction_frame(df, meta, args.fixed_beta, fit_payload, args.transient_kernel)
        pred.to_csv(outdir / "predictions.csv", index=False)

        shared_rvw.plot_collapse_overlay(
            pred,
            theta_values,
            vw_values,
            fit_payload,
            outdir / "collapse_overlay_sharedr_free_finf.png",
            args.dpi,
            args.fixed_beta,
            rf"Shared-$r$ all-$v_w$ fit with ODE-fixed $F_0$, lattice-fit $f_\infty(\theta_0)$, $t_c(\theta_0; v_w)$, kernel {shared_rvw.kernel_label(args.transient_kernel)}, $r={fit_payload['r']:.3f}$",
        )
        raw_rmse = shared_rvw.plot_raw(pred, theta_values, vw_values, outdir, "xi_vs_betaH_sharedr_free_finf", args.dpi)
        shared_rvw.plot_tc_by_vw(fit_payload, theta_values, vw_values, outdir / "tc_by_vw_sharedr_free_finf.png", args.dpi)
        plot_f_infty_theta(fit_payload, theta_values, ode, outdir / "f_infty_theta_sharedr_free_finf.png", args.dpi)

        fit_summary = summarize_payload(fit_payload, theta_values, vw_values, ode)
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
            "shared_r_free_finf": fit_summary,
            "raw_plot_rmse": raw_rmse,
            "outputs": {
                "collapse": str(outdir / "collapse_overlay_sharedr_free_finf.png"),
                "tc_by_vw": str(outdir / "tc_by_vw_sharedr_free_finf.png"),
                "f_infty_theta": str(outdir / "f_infty_theta_sharedr_free_finf.png"),
                "predictions": str(outdir / "predictions.csv"),
            },
        }
        shared.save_json(outdir / "fit_shared_r_free_finf.json", fit_summary)
        shared.save_json(outdir / "final_summary.json", summary)
        print(json.dumps(shared.to_native(summary), indent=2, sort_keys=True))
    except Exception as exc:
        error_exit(outdir, exc)
        raise


if __name__ == "__main__":
    main()
