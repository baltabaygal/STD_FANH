#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fit_vw_amplitude as base_fit


OUTDIR_DEFAULT = "results_vw_timewarp"
RESULTS_HF_CANDIDATES = [ROOT / "results_hf/final_summary.json"]
FINF_CANDIDATES = [
    ROOT / "results_collapse/Finf_tail.csv",
    ROOT / "results_tosc_lattice/collapse_and_fit_fanh/Finf_tail.csv",
]


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


def parse_args():
    p = argparse.ArgumentParser(description="Fit a vw-dependent time-warp model for lattice xi data.")
    p.add_argument("--rho", type=str, default="")
    p.add_argument("--vw-folders", nargs="*", default=None)
    p.add_argument("--h-values", type=float, nargs="+", default=base_fit.DEFAULT_H_VALUES)
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
    p.add_argument("--plot", dest="plot", action="store_true")
    p.add_argument("--no-plot", dest="plot", action="store_false")
    p.add_argument("--use-analytic-F0", dest="use_analytic_f0", action="store_true")
    p.add_argument("--file-F0", dest="use_analytic_f0", action="store_false")
    p.add_argument("--reg-Finf", type=float, default=1.0e-3)
    p.set_defaults(fix_tc=True, plot=True, use_analytic_f0=False)
    return p.parse_args()


def h_theta(theta):
    theta = np.asarray(theta, dtype=np.float64)
    cos_half = np.cos(theta / 2.0)
    good = np.abs(cos_half) > 1.0e-12
    out = np.full_like(theta, np.nan, dtype=np.float64)
    out[good] = np.log(np.e / np.maximum(cos_half[good] ** 2, 1.0e-300))
    return out


def load_analytic_defaults():
    for path in RESULTS_HF_CANDIDATES:
        if path.exists():
            payload = json.loads(path.read_text())
            return {
                "A0": float(payload["noPT"]["best_fit"]["A"]),
                "gamma0": float(payload["noPT"]["best_fit"]["gamma"]),
                "Ainf": float(payload["Finf"]["best_fit"]["A"]),
                "gamma_inf": float(payload["Finf"]["best_fit"]["gamma"]),
                "source": str(path),
            }
    raise FileNotFoundError("Missing results_hf/final_summary.json for analytic F0 defaults.")


def load_finf_table(theta_values):
    for path in FINF_CANDIDATES:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        theta_col = cols.get("theta")
        finf_col = cols.get("f_inf_tail", cols.get("finf_tail", cols.get("finf")))
        if theta_col is None or finf_col is None:
            continue
        use = pd.DataFrame(
            {
                "theta": df[theta_col].astype(float),
                "Finf": df[finf_col].astype(float),
            }
        )
        use = use[np.isfinite(use["theta"]) & np.isfinite(use["Finf"]) & (use["Finf"] > 0.0)].copy()
        if use.empty:
            continue
        out = []
        theta_ref = use["theta"].to_numpy(dtype=np.float64)
        finf_ref = use["Finf"].to_numpy(dtype=np.float64)
        for theta in theta_values:
            idx = base_fit.nearest_theta(theta_ref, theta)
            out.append(float(finf_ref[idx]))
        return np.asarray(out, dtype=np.float64), str(path)
    return None, None


def prepare_dataframe(args, outdir: Path):
    load_args = SimpleNamespace(
        rho=args.rho,
        vw_folders=args.vw_folders,
        h_values=args.h_values,
        tp_min=args.tp_min,
        tp_max=args.tp_max,
        bootstrap=0,
        n_jobs=1,
        reg_Finf=0.0,
        tc0=args.tc0,
        fix_tc=args.fix_tc,
        dpi=220,
        outdir=str(outdir),
    )
    df, f0_table, theta_values = base_fit.prepare_dataframe(load_args, outdir)
    analytic_meta = None
    if args.use_analytic_f0:
        analytic_meta = load_analytic_defaults()
        hvals = h_theta(df["theta"].to_numpy(dtype=np.float64))
        if np.any(~np.isfinite(hvals)) or np.any(hvals <= 0.0):
            raise RuntimeError("Analytic F0 requested, but some theta values are too close to pi to compute h(theta).")
        df["F0"] = analytic_meta["A0"] * np.power(hvals, analytic_meta["gamma0"])
    df["h"] = h_theta(df["theta"].to_numpy(dtype=np.float64))
    bad_h = ~np.isfinite(df["h"]) | (df["h"] <= 0.0)
    dropped = int(np.count_nonzero(bad_h))
    if dropped:
        print(f"[warn] dropping {dropped} rows with invalid h(theta)")
        df = df.loc[~bad_h].copy()
    if df.empty:
        raise RuntimeError("No valid rows remained after h(theta) filtering.")
    df["fanh"] = df["xi"] * df["F0"] / np.maximum(np.power(df["tp"] / args.t_osc, 1.5), 1.0e-18)
    theta_values = np.sort(df["theta"].unique())
    theta_index = {float(th): i for i, th in enumerate(theta_values)}
    df["theta_idx"] = [theta_index[float(th)] for th in df["theta"]]
    return df.sort_values(["v_w", "H", "theta", "tp"]).reset_index(drop=True), f0_table, theta_values, analytic_meta


def estimate_finf_init(df, theta_values, t_osc):
    out = np.zeros(len(theta_values), dtype=np.float64)
    for i, theta in enumerate(theta_values):
        sub = df[df["theta"] == float(theta)].sort_values("tp").copy()
        n_tail = max(5, int(math.ceil(0.10 * len(sub))))
        tail = sub.tail(n_tail).copy()
        coeff = tail["xi"].to_numpy(dtype=np.float64) / np.maximum(
            np.power(tail["tp"].to_numpy(dtype=np.float64) / t_osc, 1.5),
            1.0e-18,
        )
        out[i] = max(float(np.median(coeff) * (sub["F0"].iloc[0] ** 2)), 1.0e-8)
    return out


def warp_scale(vw, option, warp_params, vw_values=None):
    vw = np.asarray(vw, dtype=np.float64)
    if option == "A":
        eta = float(warp_params["eta"])
        return np.power(vw, eta)
    if option == "B":
        if vw_values is None:
            raise ValueError("vw_values required for option B.")
        scales = np.asarray(warp_params["scales"], dtype=np.float64)
        out = np.ones_like(vw, dtype=np.float64)
        for i, v in enumerate(vw_values):
            out[np.isclose(vw, float(v), atol=1.0e-8)] = float(scales[i])
        return out
    raise ValueError(f"Unsupported option {option}")


def build_param_vector(option, theta_values, fix_tc, tc0, r0, finf0, warp0, vw_values):
    parts = []
    lower = []
    upper = []
    if not fix_tc:
        parts.append(float(tc0))
        lower.append(0.5)
        upper.append(10.0)
    if option == "A":
        parts.append(float(warp0.get("eta", 0.0)))
        lower.append(-3.0)
        upper.append(3.0)
    elif option == "B":
        s0 = warp0.get("scales")
        if s0 is None:
            s0 = np.ones(len(vw_values), dtype=np.float64)
        parts.extend(np.asarray(s0, dtype=np.float64).tolist())
        lower.extend([0.2] * len(vw_values))
        upper.extend([5.0] * len(vw_values))
    else:
        raise ValueError(f"Unsupported option {option}")
    parts.append(float(r0))
    lower.append(0.1)
    upper.append(50.0)
    parts.extend(np.asarray(finf0, dtype=np.float64).tolist())
    lower.extend([1.0e-8] * len(theta_values))
    upper.extend([1.0e3] * len(theta_values))
    return np.asarray(parts, dtype=np.float64), np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def unpack_params(params, option, n_theta, fix_tc, tc_fixed, vw_values):
    idx = 0
    if fix_tc:
        tc = float(tc_fixed)
    else:
        tc = float(params[idx])
        idx += 1
    if option == "A":
        warp = {"eta": float(params[idx])}
        idx += 1
    else:
        n_vw = len(vw_values)
        warp = {"scales": np.asarray(params[idx : idx + n_vw], dtype=np.float64)}
        idx += n_vw
    r = float(params[idx])
    idx += 1
    finf = np.asarray(params[idx : idx + n_theta], dtype=np.float64)
    return tc, warp, r, finf


def model_eval(params, meta, option, fix_tc, tc_fixed):
    tc, warp_params, r, finf = unpack_params(params, option, meta["n_theta"], fix_tc, tc_fixed, meta["vw_values"])
    scale = warp_scale(meta["v_w"], option, warp_params, meta["vw_values"])
    tp_scaled = meta["tp"] * scale
    plateau = np.power(meta["tp"] / meta["t_osc"], 1.5) * finf[meta["theta_idx"]] / meta["F0_sq"]
    transient = np.power(np.maximum(scale, 1.0e-18), -1.5) / (
        1.0 + np.power(np.maximum(tp_scaled, 1.0e-18) / max(tc, 1.0e-18), r)
    )
    return plateau + transient


def fanh_univ(tp_scaled, f0, finf_theta, t_osc, tc, r):
    return finf_theta / f0 + f0 / (
        np.power(np.maximum(tp_scaled / t_osc, 1.0e-18), 1.5)
        * (1.0 + np.power(np.maximum(tp_scaled, 1.0e-18) / max(tc, 1.0e-18), r))
    )


def residual_vector(params, meta, option, fix_tc, tc_fixed, reg_finf, finf_ref):
    y_model = model_eval(params, meta, option, fix_tc, tc_fixed)
    resid = (y_model - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    if reg_finf > 0.0:
        _, _, _, finf = unpack_params(params, option, meta["n_theta"], fix_tc, tc_fixed, meta["vw_values"])
        scale = np.maximum(finf_ref, 1.0e-6)
        reg = math.sqrt(reg_finf) * (finf - finf_ref) / scale
        resid = np.concatenate([resid, reg])
    return resid


def approx_covariance(result):
    try:
        jac = np.asarray(result.jac, dtype=np.float64)
        if jac.ndim != 2 or jac.shape[0] <= jac.shape[1]:
            return None
        jt_j = jac.T @ jac
        dof = max(jac.shape[0] - jac.shape[1], 1)
        rss = float(np.sum(np.square(result.fun)))
        sigma2 = rss / dof
        return np.linalg.pinv(jt_j) * sigma2
    except Exception:
        return None


def aic_bic(resid, k):
    resid = np.asarray(resid, dtype=np.float64)
    n = max(int(resid.size), 1)
    rss = max(float(np.sum(np.square(resid))), 1.0e-18)
    return (
        float(n * math.log(rss / n) + 2.0 * k),
        float(n * math.log(rss / n) + k * math.log(n)),
    )


def rel_rmse(y, y_fit):
    y = np.asarray(y, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square((y_fit - y) / np.maximum(y, 1.0e-12)))))


def fit_option(df, theta_values, option, args, finf_init=None, init_from=None):
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
        finf_init = estimate_finf_init(df, theta_values, args.t_osc)
    if init_from is None:
        tc0 = args.tc0
        r0 = 5.0
        warp0 = {"eta": 0.0} if option == "A" else {"scales": np.ones(len(vw_values), dtype=np.float64)}
    else:
        tc0 = float(init_from["t_c"])
        r0 = float(init_from["r"])
        warp0 = {"eta": float(init_from["eta"])} if option == "A" else {"scales": np.asarray(init_from["scales"], dtype=np.float64)}
        finf_init = np.asarray(init_from["F_inf"], dtype=np.float64)

    x0, lower, upper = build_param_vector(option, theta_values, args.fix_tc, tc0, r0, finf_init, warp0, vw_values)
    fun = lambda p: residual_vector(p, meta, option, args.fix_tc, args.tc0, args.reg_Finf, finf_init)

    huber = least_squares(fun, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=20000)
    final = least_squares(fun, huber.x, bounds=(lower, upper), loss="linear", max_nfev=20000)
    y_fit = model_eval(final.x, meta, option, args.fix_tc, args.tc0)
    frac_resid = (y_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    tc, warp_params, r, finf = unpack_params(final.x, option, len(theta_values), args.fix_tc, args.tc0, vw_values)
    cov = approx_covariance(final)
    errs = None if cov is None else np.sqrt(np.maximum(np.diag(cov), 0.0))
    aic, bic = aic_bic(frac_resid, len(final.x))

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
        "rel_rmse": rel_rmse(meta["xi"], y_fit),
        "AIC": float(aic),
        "BIC": float(bic),
        "theta_values": [float(v) for v in theta_values],
        "F_inf": {f"{theta:.10f}": float(val) for theta, val in zip(theta_values, finf)},
        "vw_values": [float(v) for v in vw_values],
        "per_vw_rel_rmse": {
            f"{vw:.1f}": rel_rmse(
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
    if errs is not None:
        idx = 0
        err_payload = {}
        if not args.fix_tc:
            err_payload["t_c_err"] = float(errs[idx])
            idx += 1
        if option == "A":
            err_payload["eta_err"] = float(errs[idx])
            idx += 1
        else:
            for vw in vw_values:
                err_payload[f"s_{vw:.1f}_err"] = float(errs[idx])
                idx += 1
        err_payload["r_err"] = float(errs[idx])
        payload["approx_errors"] = err_payload
        payload["covariance"] = cov.tolist()
    return {
        "payload": payload,
        "result": final,
        "meta": meta,
        "y_fit": y_fit,
        "frac_resid": frac_resid,
        "theta_values": theta_values,
    }


def build_init_from_fit(payload):
    init = {
        "t_c": payload["t_c"],
        "r": payload["r"],
        "F_inf": [payload["F_inf"][f"{theta:.10f}"] for theta in payload["theta_values"]],
    }
    if payload["option"] == "A":
        init["eta"] = payload["eta"]
    else:
        init["scales"] = [payload["scales"][f"{vw:.1f}"] for vw in payload["vw_values"]]
    return init


def bootstrap_fit(df, theta_values, fit_bundle, option, args):
    if args.nboot <= 0:
        return {"status": "skipped", "n_samples": 0}

    x_best = fit_bundle["result"].x
    meta = fit_bundle["meta"]
    y_fit = fit_bundle["y_fit"]
    frac_resid = fit_bundle["frac_resid"]
    finf_ref = np.asarray([fit_bundle["payload"]["F_inf"][f"{theta:.10f}"] for theta in theta_values], dtype=np.float64)
    if option == "A":
        warp0 = {"eta": fit_bundle["payload"].get("eta", 0.0)}
    else:
        warp0 = {
            "scales": np.asarray(
                [fit_bundle["payload"]["scales"][f"{vw:.1f}"] for vw in fit_bundle["payload"]["vw_values"]],
                dtype=np.float64,
            )
        }
    x0, lower, upper = build_param_vector(
        option,
        theta_values,
        args.fix_tc,
        fit_bundle["payload"]["t_c"],
        fit_bundle["payload"]["r"],
        finf_ref,
        warp0,
        meta["vw_values"],
    )

    def one_boot(seed):
        rng = np.random.default_rng(seed)
        sampled = rng.choice(frac_resid, size=len(frac_resid), replace=True)
        xi_boot = y_fit / np.maximum(1.0 + sampled, 0.05)
        boot_meta = dict(meta)
        boot_meta["xi"] = xi_boot
        fun = lambda p: residual_vector(p, boot_meta, option, args.fix_tc, args.tc0, args.reg_Finf, finf_ref)
        try:
            huber = least_squares(fun, x_best, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=15000)
            final = least_squares(fun, huber.x, bounds=(lower, upper), loss="linear", max_nfev=15000)
            if not final.success:
                return None
            return final.x
        except Exception:
            return None

    seeds = np.arange(args.nboot, dtype=np.int64) + 9876
    samples = Parallel(n_jobs=args.n_jobs, prefer="threads")(delayed(one_boot)(int(seed)) for seed in seeds)
    samples = [s for s in samples if s is not None]
    if not samples:
        return {"status": "no_successful_samples", "n_samples": 0}

    arr = np.asarray(samples, dtype=np.float64)
    out = {"status": "ok", "n_samples": int(len(arr))}
    idx = 0
    if not args.fix_tc:
        vals = arr[:, idx]
        out["t_c"] = {"p16": float(np.nanpercentile(vals, 16)), "p50": float(np.nanpercentile(vals, 50)), "p84": float(np.nanpercentile(vals, 84))}
        idx += 1
    else:
        out["t_c"] = {"fixed": float(args.tc0)}
    if option == "A":
        vals = arr[:, idx]
        out["eta"] = {"p16": float(np.nanpercentile(vals, 16)), "p50": float(np.nanpercentile(vals, 50)), "p84": float(np.nanpercentile(vals, 84))}
        idx += 1
    else:
        scale_payload = {}
        for vw in fit_bundle["payload"]["vw_values"]:
            vals = arr[:, idx]
            scale_payload[f"{vw:.1f}"] = {
                "p16": float(np.nanpercentile(vals, 16)),
                "p50": float(np.nanpercentile(vals, 50)),
                "p84": float(np.nanpercentile(vals, 84)),
            }
            idx += 1
        out["scales"] = scale_payload
    vals = arr[:, idx]
    out["r"] = {"p16": float(np.nanpercentile(vals, 16)), "p50": float(np.nanpercentile(vals, 50)), "p84": float(np.nanpercentile(vals, 84))}
    idx += 1
    finf_payload = {}
    for theta in theta_values:
        vals = arr[:, idx]
        finf_payload[f"{theta:.10f}"] = {
            "p16": float(np.nanpercentile(vals, 16)),
            "p50": float(np.nanpercentile(vals, 50)),
            "p84": float(np.nanpercentile(vals, 84)),
        }
        idx += 1
    out["F_inf"] = finf_payload
    return out


def fit_postwarp_amplitude(df, fit_bundle, option, args):
    y_base = fit_bundle["y_fit"]
    xi = df["xi"].to_numpy(dtype=np.float64)
    vw = df["v_w"].to_numpy(dtype=np.float64)

    def resid(alpha_arr):
        alpha = float(alpha_arr[0])
        amp = np.power(vw, alpha)
        return (amp * y_base - xi) / np.maximum(xi, 1.0e-12)

    huber = least_squares(resid, np.asarray([0.0]), bounds=([-3.0], [3.0]), loss="huber", f_scale=0.05, max_nfev=5000)
    final = least_squares(resid, huber.x, bounds=([-3.0], [3.0]), loss="linear", max_nfev=5000)
    y_fit = np.power(vw, float(final.x[0])) * y_base
    frac_resid = (y_fit - xi) / np.maximum(xi, 1.0e-12)
    aic, bic = aic_bic(frac_resid, 1)
    return {
        "alpha": float(final.x[0]),
        "rel_rmse": rel_rmse(xi, y_fit),
        "AIC": float(aic),
        "BIC": float(bic),
    }


def representative_thetas(theta_values, n=6):
    theta_values = np.asarray(theta_values, dtype=np.float64)
    if len(theta_values) <= n:
        return theta_values
    idx = np.linspace(0, len(theta_values) - 1, n).round().astype(int)
    return theta_values[np.unique(idx)]


def plot_timewarp_vs_vw(option_a, boot_a, option_b, boot_b, outpath):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    vw_grid = np.linspace(0.3, 0.9, 300)
    if option_a is not None:
        eta = option_a["payload"]["eta"]
        ax.plot(vw_grid, np.power(vw_grid, eta), color="tab:blue", lw=2.0, label=rf"Option A: $v_w^{{\eta}}$, $\eta={eta:.3f}$")
        if boot_a.get("status") == "ok" and "eta" in boot_a:
            eta_lo = boot_a["eta"]["p16"]
            eta_hi = boot_a["eta"]["p84"]
            ax.fill_between(vw_grid, np.power(vw_grid, eta_lo), np.power(vw_grid, eta_hi), color="tab:blue", alpha=0.18)
    if option_b is not None:
        scales = option_b["payload"]["scales"]
        xs = np.asarray([float(v) for v in option_b["payload"]["vw_values"]], dtype=np.float64)
        ys = np.asarray([float(scales[f"{v:.1f}"]) for v in xs], dtype=np.float64)
        ax.scatter(xs, ys, color="tab:orange", s=55, label="Option B: fitted $s(v_w)$")
        ax.plot(xs, ys, color="tab:orange", lw=1.6)
        if boot_b.get("status") == "ok" and "scales" in boot_b:
            ylo = np.asarray([boot_b["scales"][f"{v:.1f}"]["p16"] for v in xs], dtype=np.float64)
            yhi = np.asarray([boot_b["scales"][f"{v:.1f}"]["p84"] for v in xs], dtype=np.float64)
            ax.errorbar(xs, ys, yerr=[ys - ylo, yhi - ys], fmt="none", color="tab:orange", capsize=3)
    ax.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.6)
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"time warp $s(v_w)$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_fanh_panels(df, fit_bundle, option, args, outpath):
    theta_values = np.asarray(fit_bundle["theta_values"], dtype=np.float64)
    reps = representative_thetas(theta_values)
    fig, axes = plt.subplots(2, len(reps), figsize=(3.4 * len(reps), 7.0), sharex="col")
    if len(reps) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    cmap = plt.get_cmap("viridis")
    vw_values = np.sort(df["v_w"].unique())
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    marker_map = {1.5: "o", 2.0: "s"}

    payload = fit_bundle["payload"]
    theta_to_idx = {float(th): i for i, th in enumerate(theta_values)}
    for col, theta in enumerate(reps):
        sub = df[df["theta"] == float(theta)].copy()
        f0 = float(sub["F0"].iloc[0])
        finf_theta = payload["F_inf"][f"{theta:.10f}"]
        theta_idx = theta_to_idx[float(theta)]

        ax_raw = axes[0, col]
        ax_warp = axes[1, col]
        for vw in vw_values:
            sub_vw = sub[sub["v_w"] == float(vw)].copy()
            if sub_vw.empty:
                continue
            scale = warp_scale(np.asarray([vw]), option, {"eta": payload.get("eta", 0.0), "scales": np.asarray([payload.get("scales", {}).get(f"{vv:.1f}", 1.0) for vv in payload["vw_values"]], dtype=np.float64)}, payload["vw_values"])[0]
            for hval, hh in sub_vw.groupby("H"):
                hh = hh.sort_values("tp")
                ax_raw.scatter(hh["tp"], hh["fanh"], s=18, color=colors[vw], marker=marker_map.get(float(hval), "o"), alpha=0.8)
                ax_warp.scatter(hh["tp"] * scale, hh["fanh"], s=18, color=colors[vw], marker=marker_map.get(float(hval), "o"), alpha=0.8)

            tp_grid = np.geomspace(sub_vw["tp"].min() * 0.95, sub_vw["tp"].max() * 1.05, 250)
            tp_scaled = tp_grid * scale
            curve = fanh_univ(tp_scaled, f0, finf_theta, args.t_osc, payload["t_c"], payload["r"])
            ax_raw.plot(tp_grid, curve, color=colors[vw], lw=1.8)
            ax_warp.plot(tp_scaled, curve, color=colors[vw], lw=1.8)

        xuni = np.geomspace((sub["tp"] * 0.2).min(), (sub["tp"] * 2.0).max(), 250)
        univ = fanh_univ(xuni, f0, finf_theta, args.t_osc, payload["t_c"], payload["r"])
        ax_warp.plot(xuni, univ, color="black", lw=2.0)

        for ax in [ax_raw, ax_warp]:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
        ax_raw.set_ylabel(r"$f_{\rm anh}$")
        ax_warp.set_ylabel(r"$f_{\rm anh}$")
        ax_warp.set_xlabel(r"$t_{p,\rm scaled}$")
        ax_raw.set_xlabel(r"$t_p$")

    handles = []
    labels = []
    for vw in vw_values:
        handles.append(plt.Line2D([0], [0], color=colors[vw], marker="o", linestyle="-"))
        labels.append(rf"$v_w={vw:.1f}$")
    handles.append(plt.Line2D([0], [0], color="black", linestyle="-"))
    labels.append("universal")
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_collapse_overlay(df, fit_bundle, option, args, outpath):
    theta_values = np.asarray(fit_bundle["theta_values"], dtype=np.float64)
    reps = representative_thetas(theta_values)
    fig, axes = plt.subplots(2, len(reps), figsize=(3.4 * len(reps), 6.8), sharex="col")
    if len(reps) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    cmap = plt.get_cmap("viridis")
    vw_values = np.sort(df["v_w"].unique())
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    marker_map = {1.5: "o", 2.0: "s"}
    payload = fit_bundle["payload"]
    warp_params = {"eta": payload.get("eta", 0.0), "scales": np.asarray([payload.get("scales", {}).get(f"{vv:.1f}", 1.0) for vv in payload["vw_values"]], dtype=np.float64)}

    for col, theta in enumerate(reps):
        sub = df[df["theta"] == float(theta)].copy()
        f0 = float(sub["F0"].iloc[0])
        finf_theta = payload["F_inf"][f"{theta:.10f}"]
        ax_fanh = axes[0, col]
        ax_xi = axes[1, col]
        for vw in vw_values:
            sub_vw = sub[sub["v_w"] == float(vw)].copy()
            if sub_vw.empty:
                continue
            scale = warp_scale(np.asarray([vw]), option, warp_params, payload["vw_values"])[0]
            for hval, hh in sub_vw.groupby("H"):
                xscaled = hh["tp"] * scale
                ax_fanh.scatter(xscaled, hh["fanh"], s=18, color=colors[vw], marker=marker_map.get(float(hval), "o"), alpha=0.8)
                ax_xi.scatter(xscaled, hh["xi"], s=18, color=colors[vw], marker=marker_map.get(float(hval), "o"), alpha=0.8)
            xgrid = np.geomspace((sub_vw["tp"] * scale).min() * 0.95, (sub_vw["tp"] * scale).max() * 1.05, 250)
            fanh_curve = fanh_univ(xgrid, f0, finf_theta, args.t_osc, payload["t_c"], payload["r"])
            xi_curve = np.power(np.maximum(xgrid / scale / args.t_osc, 1.0e-18), 1.5) * fanh_curve / f0
            ax_fanh.plot(xgrid, fanh_curve, color=colors[vw], lw=1.8)
            ax_xi.plot(xgrid, xi_curve, color=colors[vw], lw=1.8)
        for ax in [ax_fanh, ax_xi]:
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
        ax_fanh.set_ylabel(r"$f_{\rm anh}$")
        ax_xi.set_ylabel(r"$\xi$")
        ax_xi.set_xlabel(r"$t_{p,\rm scaled}$")

    handles = []
    labels = []
    for vw in vw_values:
        handles.append(plt.Line2D([0], [0], color=colors[vw], marker="o", linestyle="-"))
        labels.append(rf"$v_w={vw:.1f}$")
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_residual_heatmap(df, fit_bundle, outpath):
    resid = fit_bundle["frac_resid"]
    work = df.copy()
    work["resid"] = resid
    vw_values = np.sort(work["v_w"].unique())
    fig, axes = plt.subplots(1, len(vw_values), figsize=(4.2 * len(vw_values), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, vw in zip(axes, vw_values):
        sub = work[work["v_w"] == float(vw)].copy()
        sc = ax.scatter(sub["tp"], sub["theta"], c=sub["resid"], cmap="coolwarm", vmin=-0.15, vmax=0.15, s=28)
        ax.set_xscale("log")
        ax.set_title(rf"$v_w={vw:.1f}$")
        ax.set_xlabel(r"$t_p$")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(r"$\theta$")
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label(r"$(\xi_{\rm model}-\xi)/\xi$")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_amplitude_test(df, fit_bundle, amp_test, outpath):
    y_fit = fit_bundle["y_fit"]
    work = df.copy()
    work["ratio"] = work["xi"] / np.maximum(y_fit, 1.0e-18)
    obs = work.groupby("v_w", as_index=False)["ratio"].median().sort_values("v_w")
    vw_grid = np.linspace(work["v_w"].min(), work["v_w"].max(), 300)
    amp_curve = np.power(vw_grid, amp_test["alpha"])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(obs["v_w"], obs["ratio"], color="tab:purple", s=50, label="median data/model after warp")
    ax.plot(vw_grid, amp_curve, color="tab:red", lw=2.0, label=rf"best residual amp $v_w^{{{amp_test['alpha']:.3f}}}$")
    ax.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.6)
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel("residual amplitude")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def compare_with_baseline(option_a, option_b):
    baseline_path = ROOT / "results_vw_amp/final_summary.json"
    out = {"baseline_source": None}
    if baseline_path.exists():
        payload = json.loads(baseline_path.read_text())
        out["baseline_source"] = str(baseline_path)
        out["baseline_best"] = payload["best_fit"]
    rows = []
    if option_a is not None:
        rows.append({"model": "timewarp_optionA", "rel_rmse": option_a["payload"]["rel_rmse"], "AIC": option_a["payload"]["AIC"], "BIC": option_a["payload"]["BIC"]})
    if option_b is not None:
        rows.append({"model": "timewarp_optionB", "rel_rmse": option_b["payload"]["rel_rmse"], "AIC": option_b["payload"]["AIC"], "BIC": option_b["payload"]["BIC"]})
    if "baseline_best" in out:
        base = out["baseline_best"]
        rows.append({"model": "amplitude_optionB", "rel_rmse": base["rel_rmse"], "AIC": base["AIC"], "BIC": base["BIC"]})
    out["rows"] = rows
    return out


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df, _, theta_values, analytic_meta = prepare_dataframe(args, outdir)
    finf_init, finf_source = load_finf_table(theta_values)
    if finf_init is None:
        finf_init = estimate_finf_init(df, theta_values, args.t_osc)
        finf_source = "estimated_from_tail"

    option_a = None
    option_b = None
    boot_a = {"status": "skipped", "n_samples": 0}
    boot_b = {"status": "skipped", "n_samples": 0}
    amp_test_a = None
    amp_test_b = None

    if args.option in {"A", "both"}:
        print("[fit] option A: tp_scaled = tp * v_w^eta")
        option_a = fit_option(df, theta_values, "A", args, finf_init=finf_init)
        save_json(outdir / "params_optionA.json", option_a["payload"])
        amp_test_a = fit_postwarp_amplitude(df, option_a, "A", args)
        boot_a = bootstrap_fit(df, theta_values, option_a, "A", args)
        save_json(outdir / "bootstrap_optionA.json", boot_a)

    if args.option in {"B", "both"}:
        print("[fit] option B: tp_scaled = tp * s(v_w)")
        init_b = None if option_a is None else None
        option_b = fit_option(df, theta_values, "B", args, finf_init=finf_init, init_from=init_b)
        save_json(outdir / "params_optionB.json", option_b["payload"])
        amp_test_b = fit_postwarp_amplitude(df, option_b, "B", args)
        boot_b = bootstrap_fit(df, theta_values, option_b, "B", args)
        save_json(outdir / "bootstrap_optionB.json", boot_b)

    model_comparison = compare_with_baseline(option_a, option_b)
    if option_a is not None:
        model_comparison["optionA_postwarp_amplitude_test"] = amp_test_a
    if option_b is not None:
        model_comparison["optionB_postwarp_amplitude_test"] = amp_test_b
    save_json(outdir / "model_comparison_timewarp.json", model_comparison)

    if args.plot:
        plot_timewarp_vs_vw(option_a, boot_a, option_b, boot_b, outdir / "timewarp_vs_vw.png")
        best_bundle = option_b if option_b is not None else option_a
        best_amp_test = amp_test_b if option_b is not None else amp_test_a
        if best_bundle is not None:
            best_label = best_bundle["payload"]["option"]
            plot_fanh_panels(df, best_bundle, best_label, args, outdir / "fanh_vs_tp_by_theta_timewarp.png")
            plot_collapse_overlay(df, best_bundle, best_label, args, outdir / "collapse_overlay_timewarp.png")
            plot_residual_heatmap(df, best_bundle, outdir / "residual_heatmap_timewarp.png")
            plot_amplitude_test(df, best_bundle, best_amp_test, outdir / "amplitude_test_after_warp.png")

    compact_option_a = None if option_a is None else {
        "option": option_a["payload"]["option"],
        "eta": option_a["payload"].get("eta"),
        "r": option_a["payload"]["r"],
        "t_c": option_a["payload"]["t_c"],
        "rel_rmse": option_a["payload"]["rel_rmse"],
        "AIC": option_a["payload"]["AIC"],
        "BIC": option_a["payload"]["BIC"],
    }
    compact_option_b = None if option_b is None else {
        "option": option_b["payload"]["option"],
        "scales": option_b["payload"].get("scales"),
        "r": option_b["payload"]["r"],
        "t_c": option_b["payload"]["t_c"],
        "rel_rmse": option_b["payload"]["rel_rmse"],
        "AIC": option_b["payload"]["AIC"],
        "BIC": option_b["payload"]["BIC"],
    }

    summary = {
        "status": "ok",
        "data": {
            "n_points": int(len(df)),
            "theta_values": [float(v) for v in theta_values],
            "vw_values": [float(v) for v in sorted(df["v_w"].unique())],
            "H_values": [float(v) for v in sorted(df["H"].unique())],
            "F0_mode": "analytic" if args.use_analytic_f0 else "table",
            "analytic_F0_source": None if analytic_meta is None else analytic_meta["source"],
            "Finf_init_source": finf_source,
        },
        "optionA": compact_option_a,
        "optionB": compact_option_b,
        "bootstrap_optionA": None if boot_a.get("status") != "ok" else {
            "eta": boot_a.get("eta"),
            "r": boot_a.get("r"),
            "t_c": boot_a.get("t_c"),
        },
        "bootstrap_optionB": None if boot_b.get("status") != "ok" else {
            "scales": boot_b.get("scales"),
            "r": boot_b.get("r"),
            "t_c": boot_b.get("t_c"),
        },
        "postwarp_amplitude_optionA": amp_test_a,
        "postwarp_amplitude_optionB": amp_test_b,
        "model_comparison": model_comparison,
    }
    save_json(outdir / "final_summary.json", summary)
    print(json.dumps(to_native(summary), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error_exit(Path(OUTDIR_DEFAULT), exc)
        raise
