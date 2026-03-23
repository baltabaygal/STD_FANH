#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import least_squares

import fit_vw_timewarp as tw


ROOT = Path(__file__).resolve().parent
OUTDIR_DEFAULT = "results_vw_fullshift"


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit a vw-dependent full tp rescaling model, tp -> s(v_w) tp everywhere."
    )
    p.add_argument("--rho", type=str, default="")
    p.add_argument("--vw-folders", nargs="*", default=None)
    p.add_argument("--h-values", type=float, nargs="+", default=[1.5, 2.0])
    p.add_argument("--option", choices=["A", "B", "both"], default="both")
    p.add_argument("--fix-tc", dest="fix_tc", action="store_true")
    p.add_argument("--free-tc", dest="fix_tc", action="store_false")
    p.add_argument("--t-osc", type=float, default=1.5)
    p.add_argument("--tc0", type=float, default=1.5)
    p.add_argument("--tp-min", type=float, default=None)
    p.add_argument("--tp-max", type=float, default=None)
    p.add_argument("--nboot", type=int, default=200)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--outdir", type=str, default=OUTDIR_DEFAULT)
    p.add_argument("--reg-Finf", type=float, default=1.0e-3)
    p.add_argument("--use-analytic-F0", dest="use_analytic_f0", action="store_true")
    p.add_argument("--file-F0", dest="use_analytic_f0", action="store_false")
    p.set_defaults(fix_tc=True, use_analytic_f0=False)
    return p.parse_args()


def error_exit(outdir: Path, exc: Exception):
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
    tw.save_json(outdir / "_error.json", payload)
    print(json.dumps(payload, sort_keys=True))


def model_eval_fullshift(params, meta, option, fix_tc, tc_fixed):
    tc, warp_params, r, finf = tw.unpack_params(
        params, option, meta["n_theta"], fix_tc, tc_fixed, meta["vw_values"]
    )
    scale = tw.warp_scale(meta["v_w"], option, warp_params, meta["vw_values"])
    tp_scaled = meta["tp"] * scale
    plateau = np.power(np.maximum(scale, 1.0e-18), 1.5) * np.power(
        meta["tp"] / meta["t_osc"], 1.5
    ) * finf[meta["theta_idx"]] / meta["F0_sq"]
    transient = 1.0 / (
        1.0 + np.power(np.maximum(tp_scaled, 1.0e-18) / max(tc, 1.0e-18), r)
    )
    return plateau + transient


def residual_vector(params, meta, option, fix_tc, tc_fixed, reg_finf, finf_ref):
    y_model = model_eval_fullshift(params, meta, option, fix_tc, tc_fixed)
    resid = (y_model - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    if reg_finf > 0.0:
        _, _, _, finf = tw.unpack_params(
            params, option, meta["n_theta"], fix_tc, tc_fixed, meta["vw_values"]
        )
        scale = np.maximum(finf_ref, 1.0e-6)
        reg = math.sqrt(reg_finf) * (finf - finf_ref) / scale
        resid = np.concatenate([resid, reg])
    return resid


def fit_option(df, theta_values, option, args, finf_init=None):
    vw_values = np.sort(df["v_w"].unique())
    meta = {
        "xi": df["xi"].to_numpy(dtype=np.float64),
        "tp": df["tp"].to_numpy(dtype=np.float64),
        "v_w": df["v_w"].to_numpy(dtype=np.float64),
        "theta_idx": df["theta_idx"].to_numpy(dtype=np.int64),
        "F0_sq": np.square(df["F0"].to_numpy(dtype=np.float64)),
        "n_theta": len(theta_values),
        "vw_values": vw_values,
        "t_osc": float(args.t_osc),
    }
    if finf_init is None:
        finf_init = tw.estimate_finf_init(df, theta_values, args.t_osc)
    tc0 = float(args.tc0)
    r0 = 5.0
    warp0 = {"eta": 0.0} if option == "A" else {"scales": np.ones(len(vw_values), dtype=np.float64)}
    x0, lower, upper = tw.build_param_vector(
        option, theta_values, args.fix_tc, tc0, r0, finf_init, warp0, vw_values
    )
    fun = lambda p: residual_vector(p, meta, option, args.fix_tc, args.tc0, args.reg_Finf, finf_init)
    huber = least_squares(fun, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=20000)
    final = least_squares(fun, huber.x, bounds=(lower, upper), loss="linear", max_nfev=20000)
    y_fit = model_eval_fullshift(final.x, meta, option, args.fix_tc, args.tc0)
    frac_resid = (y_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    tc, warp_params, r, finf = tw.unpack_params(final.x, option, len(theta_values), args.fix_tc, args.tc0, vw_values)
    aic, bic = tw.aic_bic(frac_resid, len(final.x))
    payload = {
        "option": option,
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "n_points": int(len(df)),
        "n_params": int(len(final.x)),
        "dof": int(len(df) - len(final.x)),
        "t_c": float(tc),
        "r": float(r),
        "rel_rmse": tw.rel_rmse(meta["xi"], y_fit),
        "AIC": float(aic),
        "BIC": float(bic),
        "theta_values": [float(v) for v in theta_values],
        "F_inf": {f"{theta:.10f}": float(val) for theta, val in zip(theta_values, finf)},
        "vw_values": [float(v) for v in vw_values],
        "per_vw_rel_rmse": {
            f"{vw:.1f}": tw.rel_rmse(
                df.loc[np.isclose(df["v_w"], float(vw), atol=1.0e-8), "xi"],
                y_fit[np.isclose(df["v_w"].to_numpy(dtype=np.float64), float(vw), atol=1.0e-8)],
            )
            for vw in vw_values
        },
    }
    if option == "A":
        payload["eta"] = float(warp_params["eta"])
    else:
        payload["scales"] = {f"{vw:.1f}": float(scale) for vw, scale in zip(vw_values, warp_params["scales"])}
    return {
        "payload": payload,
        "result": final,
        "meta": meta,
        "y_fit": y_fit,
        "theta_values": theta_values,
    }


def bootstrap_fit(df, theta_values, fit_bundle, option, args):
    if args.nboot <= 0:
        return {"status": "skipped", "n_samples": 0}
    x_best = fit_bundle["result"].x
    meta = fit_bundle["meta"]
    y_fit = fit_bundle["y_fit"]
    resid = meta["xi"] - y_fit
    lower = fit_bundle["result"].x * 0.0
    # reuse exact bounds from builder
    init_payload = fit_bundle["payload"]
    if option == "A":
        warp0 = {"eta": init_payload["eta"]}
    else:
        warp0 = {"scales": np.asarray([init_payload["scales"][f"{vw:.1f}"] for vw in init_payload["vw_values"]])}
    finf0 = np.asarray([init_payload["F_inf"][f"{theta:.10f}"] for theta in theta_values])
    x0, lower, upper = tw.build_param_vector(
        option,
        theta_values,
        args.fix_tc,
        init_payload["t_c"],
        init_payload["r"],
        finf0,
        warp0,
        np.asarray(init_payload["vw_values"], dtype=np.float64),
    )

    def one(seed):
        rng = np.random.default_rng(seed)
        boot_y = np.maximum(y_fit + resid[rng.integers(0, resid.size, size=resid.size)], 1.0e-12)
        boot_meta = dict(meta)
        boot_meta["xi"] = boot_y
        fun = lambda p: residual_vector(
            p, boot_meta, option, args.fix_tc, args.tc0, args.reg_Finf, finf0
        )
        try:
            huber = least_squares(fun, x_best, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=8000)
            final = least_squares(fun, huber.x, bounds=(lower, upper), loss="linear", max_nfev=8000)
            if not final.success:
                return None
            tc, warp_params, r, _ = tw.unpack_params(
                final.x, option, len(theta_values), args.fix_tc, args.tc0, np.asarray(init_payload["vw_values"])
            )
            row = {"r": float(r)}
            if not args.fix_tc:
                row["t_c"] = float(tc)
            if option == "A":
                row["eta"] = float(warp_params["eta"])
            else:
                for vw, scale in zip(init_payload["vw_values"], warp_params["scales"]):
                    row[f"s_{vw:.1f}"] = float(scale)
            return row
        except Exception:
            return None

    seeds = np.arange(args.nboot, dtype=np.int64) + 1234
    rows = Parallel(n_jobs=args.n_jobs)(delayed(one)(int(seed)) for seed in seeds)
    rows = [row for row in rows if row is not None]
    if not rows:
        return {"status": "failed", "n_samples": 0}
    summary = {"status": "ok", "n_samples": len(rows)}
    keys = sorted(rows[0].keys())
    for key in keys:
        vals = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {"p16": float(np.percentile(vals, 16)), "p50": float(np.percentile(vals, 50)), "p84": float(np.percentile(vals, 84))}
    return summary


def predicted_plateau_factors(payload):
    if payload["option"] == "A":
        vw = np.asarray(payload["vw_values"], dtype=np.float64)
        scale = np.power(vw, payload["eta"])
        return {f"{v:.1f}": float(s ** 1.5) for v, s in zip(vw, scale)}
    return {key: float(val ** 1.5) for key, val in payload["scales"].items()}


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

        summary = {
            "status": "ok",
            "data": {
                "n_points": int(len(df)),
                "vw_values": [float(v) for v in sorted(df["v_w"].unique())],
                "H_values": [float(v) for v in sorted(df["H"].unique())],
                "theta_values": [float(v) for v in theta_values],
                "Finf_init_source": finf_source,
            },
        }

        bundles = {}
        if args.option in {"A", "both"}:
            print("[fit] option A: replace every tp by tp * v_w^eta")
            bundles["A"] = fit_option(df, theta_values, "A", args, finf_init=finf_init)
            tw.save_json(outdir / "params_optionA.json", bundles["A"]["payload"])
            summary["optionA"] = bundles["A"]["payload"]
            summary["bootstrap_optionA"] = bootstrap_fit(df, theta_values, bundles["A"], "A", args)
        if args.option in {"B", "both"}:
            print("[fit] option B: replace every tp by tp * s(v_w)")
            bundles["B"] = fit_option(df, theta_values, "B", args, finf_init=finf_init)
            tw.save_json(outdir / "params_optionB.json", bundles["B"]["payload"])
            summary["optionB"] = bundles["B"]["payload"]
            summary["bootstrap_optionB"] = bootstrap_fit(df, theta_values, bundles["B"], "B", args)

        rows = []
        for key in ["A", "B"]:
            if key not in bundles:
                continue
            p = bundles[key]["payload"]
            rows.append(
                {
                    "model": f"fullshift_option{key}",
                    "rel_rmse": p["rel_rmse"],
                    "AIC": p["AIC"],
                    "BIC": p["BIC"],
                }
            )
        summary["model_comparison"] = {"rows": rows}
        if "A" in bundles:
            summary["optionA"]["plateau_factor_by_vw"] = predicted_plateau_factors(summary["optionA"])
        if "B" in bundles:
            summary["optionB"]["plateau_factor_by_vw"] = predicted_plateau_factors(summary["optionB"])

        tw.save_json(outdir / "final_summary.json", summary)
        print(json.dumps(tw.to_native(summary), sort_keys=True))
    except Exception as exc:
        error_exit(outdir, exc)


if __name__ == "__main__":
    main()
