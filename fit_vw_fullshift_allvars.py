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

import fit_vw_timewarp as tw


ROOT = Path(__file__).resolve().parent
OUTDIR_DEFAULT = "results_vw_fullshift_allvars"


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit a full tp-rescaling model with s depending on vw, H, and beta/H."
    )
    p.add_argument("--rho", type=str, default="")
    p.add_argument("--vw-folders", nargs="*", default=None)
    p.add_argument("--h-values", type=float, nargs="+", default=[1.5, 2.0])
    p.add_argument("--fix-tc", dest="fix_tc", action="store_true")
    p.add_argument("--free-tc", dest="fix_tc", action="store_false")
    p.add_argument("--t-osc", type=float, default=1.5)
    p.add_argument("--tc0", type=float, default=1.5)
    p.add_argument("--tp-min", type=float, default=None)
    p.add_argument("--tp-max", type=float, default=None)
    p.add_argument("--nboot", type=int, default=120)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--outdir", type=str, default=OUTDIR_DEFAULT)
    p.add_argument("--reg-Finf", type=float, default=1.0e-3)
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


def make_meta(df, theta_values, t_osc):
    vw_ref = float(np.median(np.sort(df["v_w"].unique())))
    h_ref = float(np.median(np.sort(df["H"].unique())))
    beta_ref = float(np.median(df["beta_over_H"].to_numpy(dtype=np.float64)))
    return {
        "xi": df["xi"].to_numpy(dtype=np.float64),
        "tp": df["tp"].to_numpy(dtype=np.float64),
        "H": df["H"].to_numpy(dtype=np.float64),
        "beta_over_H": df["beta_over_H"].to_numpy(dtype=np.float64),
        "v_w": df["v_w"].to_numpy(dtype=np.float64),
        "theta_idx": df["theta_idx"].to_numpy(dtype=np.int64),
        "F0_sq": np.square(df["F0"].to_numpy(dtype=np.float64)),
        "n_theta": len(theta_values),
        "t_osc": float(t_osc),
        "vw_ref": vw_ref,
        "H_ref": h_ref,
        "beta_ref": beta_ref,
        "vw_values": [float(v) for v in np.sort(df["v_w"].unique())],
        "H_values": [float(v) for v in np.sort(df["H"].unique())],
        "theta_values": [float(v) for v in theta_values],
    }


def build_param_vector(fix_tc, tc0, finf0):
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
    parts.extend(np.asarray(finf0, dtype=np.float64).tolist())
    lower.extend([1.0e-8] * len(finf0))
    upper.extend([1.0e3] * len(finf0))
    return np.asarray(parts, dtype=np.float64), np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def unpack_params(params, n_theta, fix_tc, tc_fixed):
    idx = 0
    if fix_tc:
        tc = float(tc_fixed)
    else:
        tc = float(params[idx])
        idx += 1
    log_s0 = float(params[idx]); idx += 1
    a_vw = float(params[idx]); idx += 1
    b_h = float(params[idx]); idx += 1
    c_beta = float(params[idx]); idx += 1
    r = float(params[idx]); idx += 1
    finf = np.asarray(params[idx:idx + n_theta], dtype=np.float64)
    return tc, log_s0, a_vw, b_h, c_beta, r, finf


def scale_factor(meta, log_s0, a_vw, b_h, c_beta):
    log_scale = (
        log_s0
        + a_vw * np.log(np.maximum(meta["v_w"] / meta["vw_ref"], 1.0e-18))
        + b_h * np.log(np.maximum(meta["H"] / meta["H_ref"], 1.0e-18))
        + c_beta * np.log(np.maximum(meta["beta_over_H"] / meta["beta_ref"], 1.0e-18))
    )
    return np.exp(log_scale)


def model_eval(params, meta, fix_tc, tc_fixed):
    tc, log_s0, a_vw, b_h, c_beta, r, finf = unpack_params(params, meta["n_theta"], fix_tc, tc_fixed)
    scale = scale_factor(meta, log_s0, a_vw, b_h, c_beta)
    tp_scaled = meta["tp"] * scale
    plateau = np.power(np.maximum(scale, 1.0e-18), 1.5) * np.power(
        meta["tp"] / meta["t_osc"], 1.5
    ) * finf[meta["theta_idx"]] / meta["F0_sq"]
    transient = 1.0 / (
        1.0 + np.power(np.maximum(tp_scaled, 1.0e-18) / max(tc, 1.0e-18), r)
    )
    return plateau + transient


def residual_vector(params, meta, fix_tc, tc_fixed, reg_finf, finf_ref):
    y_model = model_eval(params, meta, fix_tc, tc_fixed)
    resid = (y_model - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    if reg_finf > 0.0:
        *_, finf = unpack_params(params, meta["n_theta"], fix_tc, tc_fixed)
        scale = np.maximum(finf_ref, 1.0e-6)
        reg = math.sqrt(reg_finf) * (finf - finf_ref) / scale
        resid = np.concatenate([resid, reg])
    return resid


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


def fit_model(df, theta_values, args, finf_init):
    meta = make_meta(df, theta_values, args.t_osc)
    x0, lower, upper = build_param_vector(args.fix_tc, args.tc0, finf_init)
    fun = lambda p: residual_vector(p, meta, args.fix_tc, args.tc0, args.reg_Finf, finf_init)
    huber = least_squares(fun, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=30000)
    final = least_squares(fun, huber.x, bounds=(lower, upper), loss="linear", max_nfev=30000)
    y_fit = model_eval(final.x, meta, args.fix_tc, args.tc0)
    frac_resid = (y_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    tc, log_s0, a_vw, b_h, c_beta, r, finf = unpack_params(final.x, len(theta_values), args.fix_tc, args.tc0)
    aic, bic = aic_bic(frac_resid, len(final.x))
    payload = {
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "n_points": int(len(df)),
        "n_params": int(len(final.x)),
        "dof": int(len(df) - len(final.x)),
        "t_c": float(tc),
        "log_s0": float(log_s0),
        "s0": float(np.exp(log_s0)),
        "a_vw": float(a_vw),
        "b_H": float(b_h),
        "c_beta_over_H": float(c_beta),
        "r": float(r),
        "rel_rmse": rel_rmse(meta["xi"], y_fit),
        "AIC": float(aic),
        "BIC": float(bic),
        "theta_values": [float(v) for v in theta_values],
        "F_inf": {f"{theta:.10f}": float(val) for theta, val in zip(theta_values, finf)},
        "vw_values": meta["vw_values"],
        "H_values": meta["H_values"],
        "refs": {
            "vw_ref": meta["vw_ref"],
            "H_ref": meta["H_ref"],
            "beta_ref": meta["beta_ref"],
        },
    }
    per_vw = {}
    for vw in meta["vw_values"]:
        mask = np.isclose(df["v_w"].to_numpy(dtype=np.float64), float(vw), atol=1.0e-8)
        per_vw[f"{vw:.1f}"] = rel_rmse(meta["xi"][mask], y_fit[mask])
    payload["per_vw_rel_rmse"] = per_vw
    return {"payload": payload, "result": final, "meta": meta, "y_fit": y_fit}


def bootstrap_fit(df, theta_values, fit_bundle, args):
    if args.nboot <= 0:
        return {"status": "skipped", "n_samples": 0}
    meta = fit_bundle["meta"]
    y_fit = fit_bundle["y_fit"]
    resid = meta["xi"] - y_fit
    payload = fit_bundle["payload"]
    finf0 = np.asarray([payload["F_inf"][f"{theta:.10f}"] for theta in theta_values], dtype=np.float64)
    x0, lower, upper = build_param_vector(args.fix_tc, payload["t_c"], finf0)
    x_best = fit_bundle["result"].x

    def one(seed):
        rng = np.random.default_rng(seed)
        boot_y = np.maximum(y_fit + resid[rng.integers(0, resid.size, size=resid.size)], 1.0e-12)
        boot_meta = dict(meta)
        boot_meta["xi"] = boot_y
        fun = lambda p: residual_vector(p, boot_meta, args.fix_tc, args.tc0, args.reg_Finf, finf0)
        try:
            huber = least_squares(fun, x_best, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=12000)
            final = least_squares(fun, huber.x, bounds=(lower, upper), loss="linear", max_nfev=12000)
            if not final.success:
                return None
            tc, log_s0, a_vw, b_h, c_beta, r, _ = unpack_params(final.x, len(theta_values), args.fix_tc, args.tc0)
            row = {
                "s0": float(np.exp(log_s0)),
                "log_s0": float(log_s0),
                "a_vw": float(a_vw),
                "b_H": float(b_h),
                "c_beta_over_H": float(c_beta),
                "r": float(r),
            }
            if not args.fix_tc:
                row["t_c"] = float(tc)
            return row
        except Exception:
            return None

    seeds = np.arange(args.nboot, dtype=np.int64) + 4321
    rows = Parallel(n_jobs=args.n_jobs)(delayed(one)(int(seed)) for seed in seeds)
    rows = [row for row in rows if row is not None]
    if not rows:
        return {"status": "failed", "n_samples": 0}
    summary = {"status": "ok", "n_samples": len(rows)}
    for key in sorted(rows[0].keys()):
        vals = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {
            "p16": float(np.percentile(vals, 16)),
            "p50": float(np.percentile(vals, 50)),
            "p84": float(np.percentile(vals, 84)),
        }
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
    finf = float(payload["F_inf"][f"{theta:.10f}"])
    plateau = np.power(np.maximum(scale, 1.0e-18), 1.5) * np.power(tp_grid / float(t_osc), 1.5) * finf / max(float(f0) ** 2, 1.0e-18)
    transient = 1.0 / (1.0 + np.power(tp_scaled / max(float(payload["t_c"]), 1.0e-18), float(payload["r"])))
    return plateau + transient


def plot_grid(df, theta_values, payload, h_value, t_osc, outpath):
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
            sub_vw = sub[np.isclose(sub["v_w"], float(vw))].sort_values("beta_over_H").copy()
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

    fig.suptitle(
        rf"General full-shift fit, $H_*={h_value:.1f}$"
        + "\n"
        + rf"$s=s_0 v_w^a H_*^b (\beta/H_*)^c$, $r={payload['r']:.3f}$, $t_c={payload['t_c']:.3f}$",
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_separate(df, theta_values, payload, t_osc, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    vw_values = np.sort(df["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    rows = []
    for h_value in np.sort(df["H"].unique()):
        sub_h = df[np.isclose(df["H"], float(h_value))]
        for theta in np.sort(theta_values):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta))].copy()
            if sub.empty:
                continue
            f0 = float(sub["F0"].iloc[0])
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            for vw in vw_values:
                sub_vw = sub[np.isclose(sub["v_w"], float(vw))].sort_values("beta_over_H").copy()
                if sub_vw.empty:
                    continue
                ax.scatter(sub_vw["beta_over_H"], sub_vw["xi"], s=30, color=colors[vw], alpha=0.9, label=rf"data $v_w={vw:.1f}$")
                beta_grid = np.geomspace(sub_vw["beta_over_H"].min(), sub_vw["beta_over_H"].max(), 250)
                ax.plot(beta_grid, xi_curve(beta_grid, float(theta), float(h_value), float(vw), f0, payload, t_osc), color=colors[vw], lw=2.0, label=rf"fit $v_w={vw:.1f}$")
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_xlabel(r"$\beta/H_*$")
            ax.set_ylabel(r"$\xi$")
            ax.set_title(rf"General shift: $H_*={h_value:.1f}$, $\theta={theta:.3f}$")
            ax.legend(frameon=False, fontsize=7, ncol=2)
            fig.tight_layout()
            h_tag = f"H{h_value:.1f}".replace(".", "p")
            theta_tag = f"theta_{theta:.10f}".replace(".", "p")
            filepath = outdir / f"xi_vs_betaH_allvars_{h_tag}_{theta_tag}.png"
            fig.savefig(filepath, dpi=220)
            plt.close(fig)
            rows.append({"H": float(h_value), "theta": float(theta), "file": str(filepath)})
    return pd.DataFrame(rows)


def compare_baselines(payload):
    rows = []
    for path in [
        ROOT / "results_vw_amp/final_summary.json",
        ROOT / "results_vw_timewarp/final_summary.json",
        ROOT / "results_vw_fullshift/final_summary.json",
    ]:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        model = path.parent.name
        if "optionB" in data:
            rel = data["optionB"]["rel_rmse"]
            aic = data["optionB"]["AIC"]
            bic = data["optionB"]["BIC"]
        elif "model_comparison" in data and "rows" in data["model_comparison"]:
            continue
        else:
            continue
        rows.append({"model": model, "rel_rmse": rel, "AIC": aic, "BIC": bic})
    rows.append({"model": "results_vw_fullshift_allvars", "rel_rmse": payload["rel_rmse"], "AIC": payload["AIC"], "BIC": payload["BIC"]})
    return rows


def main():
    args = parse_args()
    outdir = (ROOT / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        df, f0_table, theta_values, _ = tw.prepare_dataframe(args, outdir)
        f0_table.to_csv(outdir / "f0_table.csv", index=False)
        finf_init, finf_source = tw.load_finf_table(theta_values)
        if finf_init is None:
            finf_init = tw.estimate_finf_init(df, theta_values, args.t_osc)
            finf_source = "tail-estimate-from-data"
        print("[fit] general full-shift: s = s0 * v_w^a * H^b * (beta/H)^c")
        bundle = fit_model(df, theta_values, args, finf_init)
        payload = bundle["payload"]
        boot = bootstrap_fit(df, theta_values, bundle, args)
        save_json(outdir / "global_fit.json", payload)
        save_json(outdir / "bootstrap.json", boot)

        plot_files = []
        for h_value in sorted(df["H"].unique()):
            outfile = outdir / f"xi_vs_betaH_allvars_H{str(float(h_value)).replace('.', 'p')}.png"
            plot_grid(df, theta_values, payload, float(h_value), args.t_osc, outfile)
            plot_files.append(str(outfile))
        index_df = plot_separate(df, theta_values, payload, args.t_osc, outdir / "xi_vs_betaH_allvars_separate")
        index_df.to_csv(outdir / "xi_vs_betaH_allvars_index.csv", index=False)

        summary = {
            "status": "ok",
            "data": {
                "n_points": int(len(df)),
                "vw_values": [float(v) for v in sorted(df["v_w"].unique())],
                "H_values": [float(v) for v in sorted(df["H"].unique())],
                "theta_values": [float(v) for v in theta_values],
                "Finf_init_source": finf_source,
            },
            "fit": payload,
            "bootstrap": boot,
            "baseline_compare": compare_baselines(payload),
            "plot_files": plot_files,
            "separate_dir": str(outdir / "xi_vs_betaH_allvars_separate"),
            "index_csv": str(outdir / "xi_vs_betaH_allvars_index.csv"),
        }
        save_json(outdir / "final_summary.json", summary)
        print(json.dumps(tw.to_native(summary), sort_keys=True))
    except Exception as exc:
        error_exit(outdir, exc)


if __name__ == "__main__":
    main()
