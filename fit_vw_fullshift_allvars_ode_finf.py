#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import least_squares

import fit_vw_timewarp as tw
import fit_vw_fullshift_allvars as fs


ROOT = Path(__file__).resolve().parent
OUTDIR_DEFAULT = "results_vw_fullshift_allvars_ode_finf"
FIT_TABLE = ROOT / "results_hf/fit_table.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Test direct t_eff fit with F_inf fixed to ODE analytic law.")
    p.add_argument("--rho", type=str, default="")
    p.add_argument("--vw-folders", nargs="*", default=None)
    p.add_argument("--h-values", type=float, nargs="+", default=[1.5, 2.0])
    p.add_argument("--t-osc", type=float, default=1.5)
    p.add_argument("--tc0", type=float, default=1.5)
    p.add_argument("--fix-tc", dest="fix_tc", action="store_true")
    p.add_argument("--free-tc", dest="fix_tc", action="store_false")
    p.add_argument("--tp-min", type=float, default=None)
    p.add_argument("--tp-max", type=float, default=None)
    p.add_argument("--nboot", type=int, default=120)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--outdir", type=str, default=OUTDIR_DEFAULT)
    p.add_argument("--use-analytic-F0", dest="use_analytic_f0", action="store_true")
    p.add_argument("--file-F0", dest="use_analytic_f0", action="store_false")
    p.set_defaults(fix_tc=True, use_analytic_f0=False)
    return p.parse_args()


def save_json(path: Path, payload):
    path.write_text(json.dumps(tw.to_native(payload), indent=2, sort_keys=True))


def error_exit(outdir: Path, exc: Exception):
    payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "_error.json", payload)
    print(json.dumps(payload, sort_keys=True))


def load_fit_table():
    if not FIT_TABLE.exists():
        raise FileNotFoundError(f"Missing fit table: {FIT_TABLE}")
    rows = list(csv.DictReader(FIT_TABLE.open()))
    finf = None
    for row in rows:
        if row["dataset"] == "Finf":
            finf = {"A": float(row["A"]), "gamma": float(row["gamma"])}
            break
    if finf is None:
        raise RuntimeError("Could not find Finf row in results_hf/fit_table.csv")
    return finf


def h_theta(theta):
    theta = np.asarray(theta, dtype=np.float64)
    return np.log(np.e / np.maximum(np.cos(theta / 2.0) ** 2, 1.0e-300))


def prepare_data(args, outdir):
    base_args = SimpleNamespace(
        rho=args.rho,
        vw_folders=args.vw_folders,
        h_values=args.h_values,
        option="B",
        fix_tc=args.fix_tc,
        t_osc=args.t_osc,
        tc0=args.tc0,
        tp_min=args.tp_min,
        tp_max=args.tp_max,
        nboot=0,
        n_jobs=1,
        outdir=str(outdir),
        plot=False,
        use_analytic_f0=args.use_analytic_f0,
        reg_Finf=0.0,
    )
    return tw.prepare_dataframe(base_args, outdir)


def ode_finf_values(theta_values, t_osc):
    rec = load_fit_table()
    A_corr = rec["A"] * (float(t_osc) ** 1.5)
    g = rec["gamma"]
    vals = A_corr * np.power(h_theta(np.asarray(theta_values, dtype=np.float64)), g)
    return np.asarray(vals, dtype=np.float64), {"A_raw": rec["A"], "A_corr": A_corr, "gamma": g}


def build_meta(df, theta_values, t_osc):
    meta = fs.make_meta(df, theta_values, t_osc)
    meta["theta"] = df["theta"].to_numpy(dtype=np.float64)
    return meta


def build_param_vector(fix_tc, tc0, with_cinf):
    parts = []
    lower = []
    upper = []
    if not fix_tc:
        parts.append(float(tc0))
        lower.append(0.2)
        upper.append(10.0)
    parts.extend([0.0, 0.0, 0.0, 0.0])  # log_s0, a_vw, b_H, c_beta
    lower.extend([-2.0, -4.0, -4.0, -4.0])
    upper.extend([2.0, 4.0, 4.0, 4.0])
    parts.append(3.0)  # r
    lower.append(0.1)
    upper.append(50.0)
    if with_cinf:
        parts.append(1.0)
        lower.append(0.2)
        upper.append(5.0)
    return np.asarray(parts, dtype=np.float64), np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def unpack_params(params, fix_tc, tc_fixed, with_cinf):
    idx = 0
    if fix_tc:
        tc = float(tc_fixed)
    else:
        tc = float(params[idx]); idx += 1
    log_s0 = float(params[idx]); idx += 1
    a_vw = float(params[idx]); idx += 1
    b_h = float(params[idx]); idx += 1
    c_beta = float(params[idx]); idx += 1
    r = float(params[idx]); idx += 1
    c_inf = 1.0
    if with_cinf:
        c_inf = float(params[idx]); idx += 1
    return tc, log_s0, a_vw, b_h, c_beta, r, c_inf


def model_eval(params, meta, fix_tc, tc_fixed, finf_ode, with_cinf):
    tc, log_s0, a_vw, b_h, c_beta, r, c_inf = unpack_params(params, fix_tc, tc_fixed, with_cinf)
    scale = fs.scale_factor(meta, log_s0, a_vw, b_h, c_beta)
    tp_scaled = meta["tp"] * scale
    finf = c_inf * finf_ode
    plateau = np.power(np.maximum(scale, 1.0e-18), 1.5) * np.power(
        meta["tp"] / meta["t_osc"], 1.5
    ) * finf[meta["theta_idx"]] / meta["F0_sq"]
    transient = 1.0 / (1.0 + np.power(np.maximum(tp_scaled, 1.0e-18) / max(tc, 1.0e-18), r))
    return plateau + transient


def residual_vector(params, meta, fix_tc, tc_fixed, finf_ode, with_cinf):
    y_model = model_eval(params, meta, fix_tc, tc_fixed, finf_ode, with_cinf)
    return (y_model - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)


def rel_rmse(y, y_fit):
    return float(np.sqrt(np.mean(np.square((y_fit - y) / np.maximum(y, 1.0e-12)))))


def aic_bic(resid, k):
    resid = np.asarray(resid, dtype=np.float64)
    n = max(int(resid.size), 1)
    rss = max(float(np.sum(np.square(resid))), 1.0e-18)
    return (
        float(n * math.log(rss / n) + 2.0 * k),
        float(n * math.log(rss / n) + k * math.log(n)),
    )


def fit_case(df, theta_values, args, finf_ode, with_cinf):
    meta = build_meta(df, theta_values, args.t_osc)
    x0, lower, upper = build_param_vector(args.fix_tc, args.tc0, with_cinf)
    fun = lambda p: residual_vector(p, meta, args.fix_tc, args.tc0, finf_ode, with_cinf)
    huber = least_squares(fun, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=30000)
    final = least_squares(fun, huber.x, bounds=(lower, upper), loss="linear", max_nfev=30000)
    y_fit = model_eval(final.x, meta, args.fix_tc, args.tc0, finf_ode, with_cinf)
    frac_resid = (y_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    tc, log_s0, a_vw, b_h, c_beta, r, c_inf = unpack_params(final.x, args.fix_tc, args.tc0, with_cinf)
    aic, bic = aic_bic(frac_resid, len(final.x))
    payload = {
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "n_points": int(len(df)),
        "n_params": int(len(final.x)),
        "dof": int(len(df) - len(final.x)),
        "t_c": float(tc),
        "s0": float(np.exp(log_s0)),
        "log_s0": float(log_s0),
        "a_vw": float(a_vw),
        "b_H": float(b_h),
        "c_beta_over_H": float(c_beta),
        "r": float(r),
        "c_inf": float(c_inf),
        "rel_rmse": rel_rmse(meta["xi"], y_fit),
        "AIC": float(aic),
        "BIC": float(bic),
        "theta_values": [float(v) for v in theta_values],
        "F_inf_ode": {f"{theta:.10f}": float(val) for theta, val in zip(theta_values, finf_ode)},
        "F_inf_used": {f"{theta:.10f}": float(c_inf * val) for theta, val in zip(theta_values, finf_ode)},
        "refs": {"vw_ref": meta["vw_ref"], "H_ref": meta["H_ref"], "beta_ref": meta["beta_ref"]},
        "per_vw_rel_rmse": {
            f"{vw:.1f}": rel_rmse(
                meta["xi"][np.isclose(df["v_w"].to_numpy(dtype=np.float64), float(vw), atol=1.0e-8)],
                y_fit[np.isclose(df["v_w"].to_numpy(dtype=np.float64), float(vw), atol=1.0e-8)],
            )
            for vw in sorted(df["v_w"].unique())
        },
    }
    return {"payload": payload, "result": final, "meta": meta, "y_fit": y_fit, "frac_resid": frac_resid}


def bootstrap_fit(fit_bundle, args, finf_ode, with_cinf):
    if args.nboot <= 0:
        return {"status": "skipped", "n_samples": 0}
    meta = fit_bundle["meta"]
    y_fit = fit_bundle["y_fit"]
    resid = meta["xi"] - y_fit
    x_best = fit_bundle["result"].x
    x0, lower, upper = build_param_vector(args.fix_tc, fit_bundle["payload"]["t_c"], with_cinf)

    def one(seed):
        rng = np.random.default_rng(seed)
        boot_y = np.maximum(y_fit + resid[rng.integers(0, resid.size, size=resid.size)], 1.0e-12)
        boot_meta = dict(meta)
        boot_meta["xi"] = boot_y
        fun = lambda p: residual_vector(p, boot_meta, args.fix_tc, args.tc0, finf_ode, with_cinf)
        try:
            huber = least_squares(fun, x_best, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=12000)
            final = least_squares(fun, huber.x, bounds=(lower, upper), loss="linear", max_nfev=12000)
            if not final.success:
                return None
            tc, log_s0, a_vw, b_h, c_beta, r, c_inf = unpack_params(final.x, args.fix_tc, args.tc0, with_cinf)
            row = {
                "s0": float(np.exp(log_s0)),
                "a_vw": float(a_vw),
                "b_H": float(b_h),
                "c_beta_over_H": float(c_beta),
                "r": float(r),
            }
            if with_cinf:
                row["c_inf"] = float(c_inf)
            if not args.fix_tc:
                row["t_c"] = float(tc)
            return row
        except Exception:
            return None

    seeds = np.arange(args.nboot, dtype=np.int64) + 7654
    rows = Parallel(n_jobs=args.n_jobs)(delayed(one)(int(seed)) for seed in seeds)
    rows = [row for row in rows if row is not None]
    if not rows:
        return {"status": "failed", "n_samples": 0}
    summary = {"status": "ok", "n_samples": len(rows)}
    for key in sorted(rows[0].keys()):
        vals = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {"p16": float(np.percentile(vals, 16)), "p50": float(np.percentile(vals, 50)), "p84": float(np.percentile(vals, 84))}
    return summary


def scale_for_triplet(vw, h_value, beta_over_h, payload):
    refs = payload["refs"]
    log_scale = (
        payload["log_s0"]
        + payload["a_vw"] * math.log(max(float(vw) / refs["vw_ref"], 1.0e-18))
        + payload["b_H"] * math.log(max(float(h_value) / refs["H_ref"], 1.0e-18))
        + payload["c_beta_over_H"] * math.log(max(float(beta_over_h) / refs["beta_ref"], 1.0e-18))
    )
    return math.exp(log_scale)


def xi_curve(beta_grid, theta, h_value, vw, f0, payload, t_osc):
    perc = tw.base_fit.PercolationCache()
    tp_grid = np.asarray([perc.get(float(h_value), float(beta), float(vw)) for beta in beta_grid], dtype=np.float64)
    tp_grid = np.maximum(tp_grid, 1.0e-18)
    scale = np.asarray([scale_for_triplet(vw, h_value, beta, payload) for beta in beta_grid], dtype=np.float64)
    tp_scaled = tp_grid * scale
    finf = float(payload["F_inf_used"][f"{theta:.10f}"])
    plateau = np.power(np.maximum(scale, 1.0e-18), 1.5) * np.power(tp_grid / float(t_osc), 1.5) * finf / max(float(f0) ** 2, 1.0e-18)
    transient = 1.0 / (1.0 + np.power(tp_scaled / max(float(payload["t_c"]), 1.0e-18), float(payload["r"])))
    return plateau + transient


def plot_grid(df, theta_values, payload, t_osc, outpath, h_value, title):
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex=True, sharey=False)
    axes = axes.flatten()
    vw_values = np.sort(df["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    sub_h = df[np.isclose(df["H"], float(h_value))].copy()
    for ax, theta in zip(axes, np.sort(theta_values)):
        sub = sub_h[np.isclose(sub_h["theta"], float(theta))].copy()
        if sub.empty:
            ax.axis("off")
            continue
        f0 = float(sub["F0"].iloc[0])
        for vw in vw_values:
            sub_vw = sub[np.isclose(sub["v_w"], float(vw))].sort_values("beta_over_H")
            if sub_vw.empty:
                continue
            ax.scatter(sub_vw["beta_over_H"], sub_vw["xi"], s=28, color=colors[vw], alpha=0.9)
            beta_grid = np.geomspace(sub_vw["beta_over_H"].min(), sub_vw["beta_over_H"].max(), 250)
            ax.plot(beta_grid, xi_curve(beta_grid, float(theta), float(h_value), float(vw), f0, payload, t_osc), color=colors[vw], lw=2.0)
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$\beta/H_*$")
        ax.set_ylabel(r"$\xi$")
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = (ROOT / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        df, _, theta_values, _ = prepare_data(args, outdir)
        finf_ode, ode_meta = ode_finf_values(theta_values, args.t_osc)
        print("[fit] case 1: F_inf(theta) fixed to ODE analytic law")
        case_fixed = fit_case(df, theta_values, args, finf_ode, with_cinf=False)
        print("[fit] case 2: F_inf(theta) = c_inf * F_inf_ODE(theta)")
        case_c = fit_case(df, theta_values, args, finf_ode, with_cinf=True)
        boot_fixed = bootstrap_fit(case_fixed, args, finf_ode, with_cinf=False)
        boot_c = bootstrap_fit(case_c, args, finf_ode, with_cinf=True)
        save_json(outdir / "fit_fixed_ode_finf.json", case_fixed["payload"])
        save_json(outdir / "fit_scaled_ode_finf.json", case_c["payload"])
        save_json(outdir / "bootstrap_fixed_ode_finf.json", boot_fixed)
        save_json(outdir / "bootstrap_scaled_ode_finf.json", boot_c)

        plot_grid(df, theta_values, case_fixed["payload"], args.t_osc, outdir / "xi_vs_betaH_fixed_ode_finf_H2p0.png", 2.0, "Fixed ODE F_inf, H*=2.0")
        plot_grid(df, theta_values, case_c["payload"], args.t_osc, outdir / "xi_vs_betaH_scaled_ode_finf_H2p0.png", 2.0, "Scaled ODE F_inf, H*=2.0")

        summary = {
            "status": "ok",
            "ode_finf": ode_meta,
            "fixed_ode_finf": case_fixed["payload"],
            "scaled_ode_finf": case_c["payload"],
            "bootstrap_fixed": boot_fixed,
            "bootstrap_scaled": boot_c,
            "baseline_allvars_rel_rmse": json.loads((ROOT / "results_vw_fullshift_allvars" / "final_summary.json").read_text())["fit"]["rel_rmse"],
        }
        save_json(outdir / "final_summary.json", summary)
        print(json.dumps(tw.to_native(summary), sort_keys=True))
    except Exception as exc:
        error_exit(outdir, exc)


if __name__ == "__main__":
    main()
