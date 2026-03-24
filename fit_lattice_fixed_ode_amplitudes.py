#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import least_squares

import collapse_and_fit_fanh_tosc as base


ROOT = Path(__file__).resolve().parent
OUTDIR_DEFAULT = "results_lattice_fixed_ode_amplitudes_vw0p9"
ODE_SUMMARY_DEFAULT = ROOT / "results_ode_direct_transition_ansatz/final_summary.json"
REFERENCE_SUMMARY_DEFAULT = ROOT / "results_tosc_lattice_vw0p9_H1p0H1p5H2p0/collapse_and_fit_fanh/final_summary.json"


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit lattice collapse with F0(theta) and F_inf(theta) fixed to ODE values."
    )
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--h-values", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    p.add_argument("--fixed-beta", type=float, default=None)
    p.add_argument("--t-osc", type=float, default=1.5)
    p.add_argument("--ode-summary", type=str, default=str(ODE_SUMMARY_DEFAULT))
    p.add_argument("--reference-summary", type=str, default=str(REFERENCE_SUMMARY_DEFAULT))
    p.add_argument("--bootstrap", type=int, default=120)
    p.add_argument("--n-jobs", type=int, default=min(8, max(1, (os_cpu_count() or 1))))
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--outdir", type=str, default=OUTDIR_DEFAULT)
    return p.parse_args()


def os_cpu_count():
    try:
        import os

        return os.cpu_count()
    except Exception:
        return 1


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


def error_exit(outdir: Path, exc: Exception):
    payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "_error.json", payload)
    print(json.dumps(payload, sort_keys=True))


def rel_rmse(y, y_fit):
    y = np.asarray(y, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square((y_fit - y) / np.maximum(y, 1.0e-12)))))


def aic_bic(frac_resid, n_params):
    frac_resid = np.asarray(frac_resid, dtype=np.float64)
    n = max(int(frac_resid.size), 1)
    rss = max(float(np.sum(np.square(frac_resid))), 1.0e-18)
    aic = float(n * math.log(rss / n) + 2.0 * n_params)
    bic = float(n * math.log(rss / n) + n_params * math.log(n))
    return aic, bic


def load_lattice_dataset(fixed_vw: float, h_values):
    raw_path = base.resolve_first_existing(base.LATTICE_RAW_CANDIDATES, "")
    if raw_path is None:
        raise FileNotFoundError("Could not resolve raw lattice data file for v9 lattice run.")
    df = base.load_lattice_data(raw_path, fixed_vw, h_values)
    return df.copy()


def load_ode_amplitudes(path: Path, theta_values):
    if not path.exists():
        raise FileNotFoundError(f"Missing ODE summary: {path}")
    payload = json.loads(path.read_text())
    fit = payload.get("fit", payload)
    f0_map = {float(k): float(v) for k, v in fit["F0"].items()}
    finf_map = {float(k): float(v) for k, v in fit["F_inf"].items()}
    theta_ref = np.array(sorted(f0_map), dtype=np.float64)
    f0_ref = np.array([f0_map[t] for t in theta_ref], dtype=np.float64)
    finf_ref = np.array([finf_map[t] for t in theta_ref], dtype=np.float64)
    idx = [base.nearest_theta(theta_ref, float(theta)) for theta in theta_values]
    f0 = f0_ref[idx]
    finf = finf_ref[idx]
    return {
        "theta_ref": theta_ref,
        "F0": f0,
        "F_inf": finf,
        "source": str(path),
        "ode_fit_summary": {
            "t_c": float(fit.get("t_c", np.nan)),
            "r": float(fit.get("r", np.nan)),
            "note": payload.get("note", ""),
        },
    }


def make_meta(df: pd.DataFrame, theta_values, ode_f0, ode_finf, t_osc: float):
    theta_index = {float(theta): i for i, theta in enumerate(theta_values)}
    theta_idx = np.array([theta_index[float(theta)] for theta in df["theta"].to_numpy(dtype=np.float64)], dtype=np.int64)
    return {
        "theta": df["theta"].to_numpy(dtype=np.float64),
        "theta_idx": theta_idx,
        "theta_values": np.asarray(theta_values, dtype=np.float64),
        "tp": df["tp"].to_numpy(dtype=np.float64),
        "H": df["H"].to_numpy(dtype=np.float64),
        "xi": df["xi"].to_numpy(dtype=np.float64),
        "beta_over_H": df["beta_over_H"].to_numpy(dtype=np.float64),
        "F0": ode_f0[theta_idx],
        "F0_sq": np.maximum(np.square(ode_f0[theta_idx]), 1.0e-18),
        "F_inf": ode_finf[theta_idx],
        "t_osc": float(t_osc),
    }


def model_eval(meta, beta, t_c, r, c_calib=1.0):
    x = meta["tp"] * np.power(meta["H"], beta)
    plateau = np.power(np.maximum(x / meta["t_osc"], 1.0e-18), 1.5) * meta["F_inf"] / meta["F0_sq"]
    transient = 1.0 / (1.0 + np.power(np.maximum(x / max(t_c, 1.0e-18), 1.0e-18), r))
    return float(c_calib) * (plateau + transient)


def residual_free(params, meta, fixed_beta=None):
    if fixed_beta is None:
        beta, t_c, r = [float(v) for v in params]
    else:
        beta = float(fixed_beta)
        t_c, r = [float(v) for v in params]
    y_fit = model_eval(meta, beta, t_c, r)
    return (y_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)


def residual_fixed_tc(params, meta, tc_fixed, fixed_beta=None):
    if fixed_beta is None:
        beta, r = [float(v) for v in params]
    else:
        beta = float(fixed_beta)
        r = float(params[0])
    y_fit = model_eval(meta, beta, tc_fixed, r)
    return (y_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)


def residual_free_calib(params, meta, fixed_beta=None):
    if fixed_beta is None:
        beta, t_c, r, c_calib = [float(v) for v in params]
    else:
        beta = float(fixed_beta)
        t_c, r, c_calib = [float(v) for v in params]
    y_fit = model_eval(meta, beta, t_c, r, c_calib=c_calib)
    return (y_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)


def residual_fixed_tc_calib(params, meta, tc_fixed, fixed_beta=None):
    if fixed_beta is None:
        beta, r, c_calib = [float(v) for v in params]
    else:
        beta = float(fixed_beta)
        r, c_calib = [float(v) for v in params]
    y_fit = model_eval(meta, beta, tc_fixed, r, c_calib=c_calib)
    return (y_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)


def fit_case(meta, fixed_tc=None, fixed_beta=None):
    if fixed_tc is None:
        if fixed_beta is None:
            x0 = np.array([-0.05, 2.2, 2.2], dtype=np.float64)
            lower = np.array([-0.5, 0.2, 0.1], dtype=np.float64)
            upper = np.array([0.5, 10.0, 20.0], dtype=np.float64)
        else:
            x0 = np.array([2.2, 2.2], dtype=np.float64)
            lower = np.array([0.2, 0.1], dtype=np.float64)
            upper = np.array([10.0, 20.0], dtype=np.float64)
        fun = lambda p: residual_free(p, meta, fixed_beta=fixed_beta)
    else:
        if fixed_beta is None:
            x0 = np.array([-0.05, 2.2], dtype=np.float64)
            lower = np.array([-0.5, 0.1], dtype=np.float64)
            upper = np.array([0.5, 20.0], dtype=np.float64)
        else:
            x0 = np.array([2.2], dtype=np.float64)
            lower = np.array([0.1], dtype=np.float64)
            upper = np.array([20.0], dtype=np.float64)
        fun = lambda p: residual_fixed_tc(p, meta, fixed_tc, fixed_beta=fixed_beta)

    huber = least_squares(fun, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=30000)
    final = least_squares(fun, huber.x, bounds=(lower, upper), loss="linear", max_nfev=30000)
    if fixed_tc is None:
        if fixed_beta is None:
            beta, t_c, r = [float(v) for v in final.x]
        else:
            beta = float(fixed_beta)
            t_c, r = [float(v) for v in final.x]
    else:
        if fixed_beta is None:
            beta, r = [float(v) for v in final.x]
        else:
            beta = float(fixed_beta)
            r = float(final.x[0])
        t_c = float(fixed_tc)
    y_fit = model_eval(meta, beta, t_c, r)
    frac_resid = (y_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    aic, bic = aic_bic(frac_resid, len(final.x))
    return {
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "beta": float(beta),
        "t_c": float(t_c),
        "r": float(r),
        "rel_rmse": rel_rmse(meta["xi"], y_fit),
        "AIC": float(aic),
        "BIC": float(bic),
        "n_points": int(meta["xi"].size),
        "n_params": int(len(final.x)),
        "result_x": final.x.copy(),
        "y_fit": y_fit,
        "frac_resid": frac_resid,
    }


def fit_case_calib(meta, fixed_tc=None, fixed_beta=None):
    if fixed_tc is None:
        if fixed_beta is None:
            x0 = np.array([-0.05, 2.2, 2.2, 1.0], dtype=np.float64)
            lower = np.array([-0.5, 0.2, 0.1, 0.1], dtype=np.float64)
            upper = np.array([0.5, 10.0, 20.0, 10.0], dtype=np.float64)
        else:
            x0 = np.array([2.2, 2.2, 1.0], dtype=np.float64)
            lower = np.array([0.2, 0.1, 0.1], dtype=np.float64)
            upper = np.array([10.0, 20.0, 10.0], dtype=np.float64)
        fun = lambda p: residual_free_calib(p, meta, fixed_beta=fixed_beta)
    else:
        if fixed_beta is None:
            x0 = np.array([-0.05, 2.2, 1.0], dtype=np.float64)
            lower = np.array([-0.5, 0.1, 0.1], dtype=np.float64)
            upper = np.array([0.5, 20.0, 10.0], dtype=np.float64)
        else:
            x0 = np.array([2.2, 1.0], dtype=np.float64)
            lower = np.array([0.1, 0.1], dtype=np.float64)
            upper = np.array([20.0, 10.0], dtype=np.float64)
        fun = lambda p: residual_fixed_tc_calib(p, meta, fixed_tc, fixed_beta=fixed_beta)

    huber = least_squares(fun, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=30000)
    final = least_squares(fun, huber.x, bounds=(lower, upper), loss="linear", max_nfev=30000)
    if fixed_tc is None:
        if fixed_beta is None:
            beta, t_c, r, c_calib = [float(v) for v in final.x]
        else:
            beta = float(fixed_beta)
            t_c, r, c_calib = [float(v) for v in final.x]
    else:
        if fixed_beta is None:
            beta, r, c_calib = [float(v) for v in final.x]
        else:
            beta = float(fixed_beta)
            r, c_calib = [float(v) for v in final.x]
        t_c = float(fixed_tc)
    y_fit = model_eval(meta, beta, t_c, r, c_calib=c_calib)
    frac_resid = (y_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    aic, bic = aic_bic(frac_resid, len(final.x))
    return {
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "beta": float(beta),
        "t_c": float(t_c),
        "r": float(r),
        "c_calib": float(c_calib),
        "rel_rmse": rel_rmse(meta["xi"], y_fit),
        "AIC": float(aic),
        "BIC": float(bic),
        "n_points": int(meta["xi"].size),
        "n_params": int(len(final.x)),
        "result_x": final.x.copy(),
        "y_fit": y_fit,
        "frac_resid": frac_resid,
    }


def bootstrap_case(meta, fit_payload, nboot, n_jobs, fixed_tc=None, fixed_beta=None):
    if nboot <= 0:
        return {"status": "skipped", "n_samples": 0}
    y_fit = fit_payload["y_fit"]
    resid = meta["xi"] - y_fit

    def one(seed):
        rng = np.random.default_rng(seed)
        boot_y = np.maximum(y_fit + resid[rng.integers(0, resid.size, size=resid.size)], 1.0e-12)
        boot_meta = dict(meta)
        boot_meta["xi"] = boot_y
        payload = fit_case(boot_meta, fixed_tc=fixed_tc, fixed_beta=fixed_beta)
        if not payload["success"]:
            return None
        out = {"r": payload["r"]}
        if fixed_beta is None:
            out["beta"] = payload["beta"]
        if fixed_tc is None:
            out["t_c"] = payload["t_c"]
        return out

    rows = Parallel(n_jobs=n_jobs)(delayed(one)(14321 + i) for i in range(nboot))
    rows = [row for row in rows if row is not None]
    if not rows:
        return {"status": "no_successful_bootstrap_samples", "n_samples": 0}
    out = {"status": "ok", "n_samples": int(len(rows))}
    for key in rows[0]:
        vals = np.array([row[key] for row in rows], dtype=np.float64)
        out[key] = {
            "p16": float(np.percentile(vals, 16)),
            "p50": float(np.percentile(vals, 50)),
            "p84": float(np.percentile(vals, 84)),
        }
    if fixed_beta is not None:
        out["beta_fixed"] = float(fixed_beta)
    return out


def bootstrap_case_calib(meta, fit_payload, nboot, n_jobs, fixed_tc=None, fixed_beta=None):
    if nboot <= 0:
        return {"status": "skipped", "n_samples": 0}
    y_fit = fit_payload["y_fit"]
    resid = meta["xi"] - y_fit

    def one(seed):
        rng = np.random.default_rng(seed)
        boot_y = np.maximum(y_fit + resid[rng.integers(0, resid.size, size=resid.size)], 1.0e-12)
        boot_meta = dict(meta)
        boot_meta["xi"] = boot_y
        payload = fit_case_calib(boot_meta, fixed_tc=fixed_tc, fixed_beta=fixed_beta)
        if not payload["success"]:
            return None
        out = {"r": payload["r"], "c_calib": payload["c_calib"]}
        if fixed_beta is None:
            out["beta"] = payload["beta"]
        if fixed_tc is None:
            out["t_c"] = payload["t_c"]
        return out

    rows = Parallel(n_jobs=n_jobs)(delayed(one)(24531 + i) for i in range(nboot))
    rows = [row for row in rows if row is not None]
    if not rows:
        return {"status": "no_successful_bootstrap_samples", "n_samples": 0}
    out = {"status": "ok", "n_samples": int(len(rows))}
    for key in rows[0]:
        vals = np.array([row[key] for row in rows], dtype=np.float64)
        out[key] = {
            "p16": float(np.percentile(vals, 16)),
            "p50": float(np.percentile(vals, 50)),
            "p84": float(np.percentile(vals, 84)),
        }
    if fixed_beta is not None:
        out["beta_fixed"] = float(fixed_beta)
    return out


def plot_overlay(df, meta, fit_payload, outpath: Path, title: str, dpi: int):
    theta_values = meta["theta_values"]
    beta = fit_payload["beta"]
    t_c = fit_payload["t_c"]
    r = fit_payload["r"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    colors = {float(h): c for h, c in zip(sorted(df["H"].unique()), ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"])}
    for ax, theta in zip(axes.ravel(), theta_values):
        sub = df[np.isclose(df["theta"], float(theta), atol=1.0e-8)].copy().sort_values("tp")
        x_data = sub["tp"].to_numpy(dtype=np.float64) * np.power(sub["H"].to_numpy(dtype=np.float64), beta)
        order = np.argsort(x_data)
        x_data = x_data[order]
        y_data = sub["xi"].to_numpy(dtype=np.float64)[order]
        h_data = sub["H"].to_numpy(dtype=np.float64)[order]
        for h in sorted(sub["H"].unique()):
            mask = np.isclose(h_data, float(h), atol=1.0e-8)
            ax.scatter(x_data[mask], y_data[mask], s=22, color=colors[float(h)], alpha=0.9, label=f"$H_*={h:g}$")
        x_grid = np.geomspace(max(x_data.min() * 0.95, 1.0e-4), x_data.max() * 1.05, 300)
        idx = base.nearest_theta(theta_values, float(theta))
        plateau = np.power(np.maximum(x_grid / meta["t_osc"], 1.0e-18), 1.5) * meta["F_inf"][meta["theta_idx"] == idx][0] / max(
            meta["F0_sq"][meta["theta_idx"] == idx][0],
            1.0e-18,
        )
        transient = 1.0 / (1.0 + np.power(np.maximum(x_grid / max(t_c, 1.0e-18), 1.0e-18), r))
        ax.plot(x_grid, plateau + transient, color="black", lw=2.0)
        ax.set_xscale("log")
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$t_p$" if np.isclose(beta, 0.0, atol=1.0e-12) else r"$x=t_p H^\beta$")
        ax.set_ylabel(r"$\xi$")
        ax.grid(True, alpha=0.2)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.suptitle(title, fontsize=14)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_raw_betaH(df, meta, fit_payload, outdir: Path, stem: str, dpi: int):
    beta = fit_payload["beta"]
    t_c = fit_payload["t_c"]
    r = fit_payload["r"]
    if "v_w" in df.columns:
        vw_values = sorted(df["v_w"].unique())
    else:
        vw_values = [0.9]
    colors = {float(vw): c for vw, c in zip(vw_values, ["#440154", "#31688e", "#35b779", "#fde725"])}
    for h in sorted(df["H"].unique()):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        sub_h = df[np.isclose(df["H"], float(h), atol=1.0e-8)].copy()
        for ax, theta in zip(axes.ravel(), meta["theta_values"]):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=1.0e-8)].copy().sort_values("beta_over_H")
            idx = base.nearest_theta(meta["theta_values"], float(theta))
            if "v_w" in sub.columns:
                groups = [
                    (float(vw), sub[np.isclose(sub["v_w"], float(vw), atol=1.0e-8)].copy())
                    for vw in sorted(sub["v_w"].unique())
                ]
            else:
                groups = [(0.9, sub.copy())]
            for vw, hh in groups:
                tp = hh["tp"].to_numpy(dtype=np.float64)
                x = tp * np.power(h, beta)
                y_fit = np.power(np.maximum(x / meta["t_osc"], 1.0e-18), 1.5) * meta["F_inf"][meta["theta_idx"] == idx][0] / max(
                    meta["F0_sq"][meta["theta_idx"] == idx][0],
                    1.0e-18,
                ) + 1.0 / (1.0 + np.power(np.maximum(x / max(t_c, 1.0e-18), 1.0e-18), r))
                ax.scatter(hh["beta_over_H"], hh["xi"], s=22, color=colors[float(vw)], alpha=0.85)
                ax.plot(hh["beta_over_H"], y_fit, color=colors[float(vw)], lw=1.8)
            ax.set_xscale("log")
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$\beta/H_*$")
            ax.set_ylabel(r"$\xi$")
            ax.grid(True, alpha=0.2)
        fig.suptitle(f"{stem}, H*={h:g}", fontsize=14)
        fig.savefig(outdir / f"{stem}_H{str(h).replace('.', 'p')}.png", dpi=dpi)
        plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        df = load_lattice_dataset(args.fixed_vw, args.h_values)
        theta_values = np.sort(df["theta"].unique())
        ode = load_ode_amplitudes(Path(args.ode_summary).resolve(), theta_values)
        meta = make_meta(df, theta_values, ode["F0"], ode["F_inf"], args.t_osc)
        fixed_beta = None if args.fixed_beta is None else float(args.fixed_beta)

        free_fit = fit_case(meta, fixed_tc=None, fixed_beta=fixed_beta)
        fixed_fit = fit_case(meta, fixed_tc=1.5, fixed_beta=fixed_beta)
        free_calib_fit = fit_case_calib(meta, fixed_tc=None, fixed_beta=fixed_beta)
        fixed_calib_fit = fit_case_calib(meta, fixed_tc=1.5, fixed_beta=fixed_beta)
        free_boot = bootstrap_case(meta, free_fit, args.bootstrap, args.n_jobs, fixed_tc=None, fixed_beta=fixed_beta)
        fixed_boot = bootstrap_case(meta, fixed_fit, args.bootstrap, args.n_jobs, fixed_tc=1.5, fixed_beta=fixed_beta)
        free_calib_boot = bootstrap_case_calib(meta, free_calib_fit, args.bootstrap, args.n_jobs, fixed_tc=None, fixed_beta=fixed_beta)
        fixed_calib_boot = bootstrap_case_calib(meta, fixed_calib_fit, args.bootstrap, args.n_jobs, fixed_tc=1.5, fixed_beta=fixed_beta)

        reference = None
        ref_path = Path(args.reference_summary).resolve()
        if ref_path.exists():
            reference = json.loads(ref_path.read_text())

        free_payload = {
            k: v
            for k, v in free_fit.items()
            if k not in {"result_x", "y_fit", "frac_resid"}
        }
        fixed_payload = {
            k: v
            for k, v in fixed_fit.items()
            if k not in {"result_x", "y_fit", "frac_resid"}
        }
        free_calib_payload = {
            k: v
            for k, v in free_calib_fit.items()
            if k not in {"result_x", "y_fit", "frac_resid"}
        }
        fixed_calib_payload = {
            k: v
            for k, v in fixed_calib_fit.items()
            if k not in {"result_x", "y_fit", "frac_resid"}
        }
        free_payload["bootstrap"] = free_boot
        fixed_payload["bootstrap"] = fixed_boot
        free_calib_payload["bootstrap"] = free_calib_boot
        fixed_calib_payload["bootstrap"] = fixed_calib_boot
        free_payload["F0_ode"] = {f"{theta:.10f}": float(val) for theta, val in zip(theta_values, ode["F0"])}
        free_payload["F_inf_ode"] = {f"{theta:.10f}": float(val) for theta, val in zip(theta_values, ode["F_inf"])}
        fixed_payload["F0_ode"] = free_payload["F0_ode"]
        fixed_payload["F_inf_ode"] = free_payload["F_inf_ode"]
        free_calib_payload["F0_ode"] = free_payload["F0_ode"]
        free_calib_payload["F_inf_ode"] = free_payload["F_inf_ode"]
        fixed_calib_payload["F0_ode"] = free_payload["F0_ode"]
        fixed_calib_payload["F_inf_ode"] = free_payload["F_inf_ode"]

        title_beta_note = "" if fixed_beta is None else rf", $\beta={fixed_beta:.1f}$ fixed"
        plot_overlay(
            df,
            meta,
            free_fit,
            outdir / "collapse_overlay_free_tc.png",
            rf"Lattice v_w={args.fixed_vw:.1f} with ODE-fixed $F_0,F_\infty$ (free $t_c$){title_beta_note}",
            args.dpi,
        )
        plot_overlay(
            df,
            meta,
            fixed_fit,
            outdir / "collapse_overlay_fixed_tc1p5.png",
            rf"Lattice v_w={args.fixed_vw:.1f} with ODE-fixed $F_0,F_\infty$ ($t_c=1.5$ fixed){title_beta_note}",
            args.dpi,
        )
        plot_overlay(
            df,
            meta,
            free_calib_fit,
            outdir / "collapse_overlay_free_tc_calib.png",
            rf"Lattice v_w={args.fixed_vw:.1f} with ODE-fixed $F_0,F_\infty$ and $c_{{calib}}$ (free $t_c$){title_beta_note}",
            args.dpi,
        )
        plot_overlay(
            df,
            meta,
            fixed_calib_fit,
            outdir / "collapse_overlay_fixed_tc1p5_calib.png",
            rf"Lattice v_w={args.fixed_vw:.1f} with ODE-fixed $F_0,F_\infty$ and $c_{{calib}}$ ($t_c=1.5$ fixed){title_beta_note}",
            args.dpi,
        )
        plot_raw_betaH(df, meta, free_fit, outdir, "xi_vs_betaH_odefixed_free_tc", args.dpi)
        plot_raw_betaH(df, meta, fixed_fit, outdir, "xi_vs_betaH_odefixed_fixed_tc1p5", args.dpi)
        plot_raw_betaH(df, meta, free_calib_fit, outdir, "xi_vs_betaH_odefixed_free_tc_calib", args.dpi)
        plot_raw_betaH(df, meta, fixed_calib_fit, outdir, "xi_vs_betaH_odefixed_fixed_tc1p5_calib", args.dpi)

        comparison = {
            "free_tc": {
                "rel_rmse": free_fit["rel_rmse"],
                "AIC": free_fit["AIC"],
                "BIC": free_fit["BIC"],
                "beta": free_fit["beta"],
                "t_c": free_fit["t_c"],
                "r": free_fit["r"],
            },
            "fixed_tc_1p5": {
                "rel_rmse": fixed_fit["rel_rmse"],
                "AIC": fixed_fit["AIC"],
                "BIC": fixed_fit["BIC"],
                "beta": fixed_fit["beta"],
                "t_c": fixed_fit["t_c"],
                "r": fixed_fit["r"],
            },
            "free_tc_calib": {
                "rel_rmse": free_calib_fit["rel_rmse"],
                "AIC": free_calib_fit["AIC"],
                "BIC": free_calib_fit["BIC"],
                "beta": free_calib_fit["beta"],
                "t_c": free_calib_fit["t_c"],
                "r": free_calib_fit["r"],
                "c_calib": free_calib_fit["c_calib"],
            },
            "fixed_tc_1p5_calib": {
                "rel_rmse": fixed_calib_fit["rel_rmse"],
                "AIC": fixed_calib_fit["AIC"],
                "BIC": fixed_calib_fit["BIC"],
                "beta": fixed_calib_fit["beta"],
                "t_c": fixed_calib_fit["t_c"],
                "r": fixed_calib_fit["r"],
                "c_calib": fixed_calib_fit["c_calib"],
            },
        }
        if reference is not None:
            comparison["reference_free_Finf"] = {
                "rel_rmse": float(reference["global_fit"]["rel_rmse"]),
                "AIC": float(reference["global_fit"]["AIC"]),
                "BIC": float(reference["global_fit"]["BIC"]),
                "beta": float(reference["beta"]),
                "t_c": float(reference["global_fit"]["t_c"]),
                "r": float(reference["global_fit"]["r"]),
            }
        save_json(outdir / "fit_free_tc.json", free_payload)
        save_json(outdir / "fit_fixed_tc1p5.json", fixed_payload)
        save_json(outdir / "fit_free_tc_calib.json", free_calib_payload)
        save_json(outdir / "fit_fixed_tc1p5_calib.json", fixed_calib_payload)
        save_json(outdir / "model_comparison.json", comparison)

        final_summary = {
            "status": "ok",
            "fixed_vw": float(args.fixed_vw),
            "fixed_beta": fixed_beta,
            "H_values": [float(v) for v in sorted(df["H"].unique())],
            "theta_values": [float(v) for v in theta_values],
            "n_points": int(len(df)),
            "ode_amplitude_source": ode["source"],
            "ode_fit_summary": ode["ode_fit_summary"],
            "free_tc": {
                "beta": free_fit["beta"],
                "t_c": free_fit["t_c"],
                "r": free_fit["r"],
                "rel_rmse": free_fit["rel_rmse"],
                "bootstrap_68": free_boot,
            },
            "fixed_tc_1p5": {
                "beta": fixed_fit["beta"],
                "t_c": fixed_fit["t_c"],
                "r": fixed_fit["r"],
                "rel_rmse": fixed_fit["rel_rmse"],
                "bootstrap_68": fixed_boot,
            },
            "free_tc_calib": {
                "beta": free_calib_fit["beta"],
                "t_c": free_calib_fit["t_c"],
                "r": free_calib_fit["r"],
                "c_calib": free_calib_fit["c_calib"],
                "rel_rmse": free_calib_fit["rel_rmse"],
                "bootstrap_68": free_calib_boot,
            },
            "fixed_tc_1p5_calib": {
                "beta": fixed_calib_fit["beta"],
                "t_c": fixed_calib_fit["t_c"],
                "r": fixed_calib_fit["r"],
                "c_calib": fixed_calib_fit["c_calib"],
                "rel_rmse": fixed_calib_fit["rel_rmse"],
                "bootstrap_68": fixed_calib_boot,
            },
            "reference_free_Finf": comparison.get("reference_free_Finf"),
        }
        save_json(outdir / "final_summary.json", final_summary)
        print(json.dumps(to_native(final_summary), indent=2, sort_keys=True))
    except Exception as exc:
        error_exit(outdir, exc)
        raise


if __name__ == "__main__":
    main()
