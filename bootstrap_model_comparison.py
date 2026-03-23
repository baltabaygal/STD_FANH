#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import least_squares
from statsmodels.tools.eval_measures import aic as sm_aic
from statsmodels.tools.eval_measures import bic as sm_bic

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fit_vw_amplitude as base_fit

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback if tqdm is absent
    tqdm = None


OUTDIR_DEFAULT = "results_bootstrap_model_comparison"
XI_ALL_CSV = ROOT / "data/xi_all.csv"
FIT_TABLE = ROOT / "results_hf/fit_table.csv"
AMP_SUMMARY = ROOT / "results_vw_amp/final_summary.json"
AMP_PARAMS = ROOT / "results_vw_amp/global_fit_optionB.json"
WARP_PARAMS = ROOT / "results_vw_timewarp/params_optionB.json"
RHO_CANDIDATES = [
    ROOT / "lattice_data/data/rho_noPT_data.txt",
    ROOT / "lattice_data/rho_noPT_data.txt",
    ROOT / "rho_noPT_data.txt",
]
DEFAULT_MODELS = ["amplitude", "warp", "combined"]
T_OSC = 1.5
T_C_FIXED = 1.5


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


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args():
    p = argparse.ArgumentParser(description="Bootstrap AIC/BIC comparison for amplitude, warp, and combined xi models.")
    p.add_argument("--nboot", type=int, default=500)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--outdir", type=str, default=OUTDIR_DEFAULT)
    p.add_argument("--models", type=str, default="amplitude,warp,combined")
    p.add_argument("--tp-min", type=float, default=None)
    p.add_argument("--tp-max", type=float, default=None)
    p.add_argument("--use-analytic-h", type=parse_bool, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fast", type=parse_bool, default=True)
    p.add_argument("--vw-folders", nargs="*", default=None)
    return p.parse_args()


def h_theta(theta):
    theta = np.asarray(theta, dtype=np.float64)
    cos_half = np.cos(theta / 2.0)
    mask = np.abs(cos_half) > 1.0e-12
    out = np.full_like(theta, np.nan, dtype=np.float64)
    out[mask] = np.log(np.e / np.maximum(cos_half[mask] ** 2, 1.0e-300))
    return out


def rel_rmse(y, yfit):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square((yfit - y) / np.maximum(y, 1.0e-12)))))


def llf_from_rss(rss: float, nobs: int) -> float:
    rss = max(float(rss), 1.0e-18)
    nobs = max(int(nobs), 1)
    return -0.5 * nobs * (math.log(2.0 * math.pi) + math.log(rss / nobs) + 1.0)


def aic_bic_from_resid(resid: np.ndarray, k_params: int):
    resid = np.asarray(resid, dtype=np.float64)
    nobs = max(int(resid.size), 1)
    rss = float(np.sum(np.square(resid)))
    llf = llf_from_rss(rss, nobs)
    return {
        "rss": rss,
        "llf": llf,
        "AIC": float(sm_aic(llf, nobs, k_params)),
        "BIC": float(sm_bic(llf, nobs, k_params)),
    }


def load_fit_table():
    if not FIT_TABLE.exists():
        raise FileNotFoundError(f"Missing analytic fit table: {FIT_TABLE}")
    df = pd.read_csv(FIT_TABLE)
    out = {}
    for _, row in df.iterrows():
        out[str(row["dataset"]).strip()] = row
    if "noPT" not in out or "Finf" not in out:
        raise RuntimeError("results_hf/fit_table.csv must contain rows for datasets noPT and Finf.")
    return out


def load_unified_csv():
    df = pd.read_csv(XI_ALL_CSV)
    cols = {c.lower(): c for c in df.columns}
    required = {"theta", "tp", "xi", "h", "v_w"}
    if not required.issubset(set(cols)):
        raise RuntimeError(f"{XI_ALL_CSV} must contain columns {sorted(required)}; found {list(df.columns)}")
    out = pd.DataFrame(
        {
            "theta": df[cols["theta"]].astype(float),
            "tp": df[cols["tp"]].astype(float),
            "xi": df[cols["xi"]].astype(float),
            "H": df[cols["h"]].astype(float),
            "v_w": df[cols["v_w"]].astype(float),
        }
    )
    if "source" in cols:
        out["source"] = df[cols["source"]].astype(str)
    else:
        out["source"] = "xi_all.csv"
    return out


def load_autodiscovered(vw_folders):
    args = SimpleNamespace(
        rho="",
        vw_folders=vw_folders,
        h_values=[1.5, 2.0],
        tp_min=None,
        tp_max=None,
        bootstrap=0,
        n_jobs=1,
        reg_Finf=0.0,
        tc0=T_C_FIXED,
        fix_tc=True,
        dpi=120,
        outdir=str(ROOT / OUTDIR_DEFAULT),
    )
    outdir = ROOT / OUTDIR_DEFAULT
    outdir.mkdir(parents=True, exist_ok=True)
    df, f0_table, theta_values = base_fit.prepare_dataframe(args, outdir)
    return df[["theta", "tp", "xi", "H", "v_w"]].copy(), f0_table


def load_dataset(args, outdir: Path):
    if XI_ALL_CSV.exists():
        print(f"[load] using unified table {XI_ALL_CSV.relative_to(ROOT)}")
        df = load_unified_csv()
        f0_table = None
    else:
        print("[load] data/xi_all.csv not found, autodiscovering lattice raw files")
        df, f0_table = load_autodiscovered(args.vw_folders)

    df = df[np.isfinite(df["theta"]) & np.isfinite(df["tp"]) & np.isfinite(df["xi"]) & np.isfinite(df["H"]) & np.isfinite(df["v_w"])].copy()
    df = df[(df["tp"] > 0.0) & (df["xi"] > 0.0)].copy()
    if args.tp_min is not None:
        df = df[df["tp"] >= float(args.tp_min)].copy()
    if args.tp_max is not None:
        df = df[df["tp"] <= float(args.tp_max)].copy()
    if df.empty:
        raise RuntimeError("No valid xi points remained after filtering.")

    fit_rows = load_fit_table()
    df["h"] = h_theta(df["theta"].to_numpy(dtype=np.float64))
    bad_h = ~np.isfinite(df["h"]) | (df["h"] <= 0.0)
    if np.any(bad_h):
        print(f"[warn] dropping {int(np.count_nonzero(bad_h))} rows with invalid h(theta)")
        df = df.loc[~bad_h].copy()
    if df.empty:
        raise RuntimeError("No valid rows remain after h(theta) filtering.")

    if args.use_analytic_h:
        a0 = float(fit_rows["noPT"]["A"])
        g0 = float(fit_rows["noPT"]["gamma"])
        df["F0"] = a0 * np.power(df["h"].to_numpy(dtype=np.float64), g0)
        f0_source = str(FIT_TABLE)
    else:
        rho_path = None
        for cand in RHO_CANDIDATES:
            if cand.exists():
                rho_path = cand
                break
        if rho_path is None:
            raise FileNotFoundError("Missing rho_noPT_data.txt and --use-analytic-h=False requested.")
        table = base_fit.load_f0_table(rho_path, [1.5, 2.0])
        theta_ref = table["theta"].to_numpy(dtype=np.float64)
        f0_ref = table["F0"].to_numpy(dtype=np.float64)
        idx = [base_fit.nearest_theta(theta_ref, th) for th in df["theta"].to_numpy(dtype=np.float64)]
        df["F0"] = [float(f0_ref[i]) for i in idx]
        f0_source = str(rho_path)

    df = df[np.isfinite(df["F0"]) & (df["F0"] > 0.0)].copy()
    theta_values = np.sort(df["theta"].unique())
    theta_idx = {float(th): i for i, th in enumerate(theta_values)}
    df["theta_idx"] = [theta_idx[float(th)] for th in df["theta"]]
    df["fanh"] = df["xi"] * df["F0"] / np.maximum(np.power(df["tp"] / T_OSC, 1.5), 1.0e-18)
    df.to_csv(outdir / "xi_loaded_for_bootstrap.csv", index=False)
    return df.sort_values(["v_w", "H", "theta", "tp"]).reset_index(drop=True), theta_values, f0_source


def load_fixed_finf(theta_values):
    candidates = [WARP_PARAMS, AMP_PARAMS]
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        finf_map = payload.get("F_inf")
        if not finf_map:
            continue
        vals = []
        for theta in theta_values:
            key = f"{theta:.10f}"
            if key not in finf_map:
                break
            vals.append(float(finf_map[key]))
        else:
            return np.asarray(vals, dtype=np.float64), str(path)
    return None, None


def estimate_finf_once(df, theta_values):
    vals = np.zeros(len(theta_values), dtype=np.float64)
    for i, theta in enumerate(theta_values):
        sub = df[df["theta"] == float(theta)].sort_values("tp").copy()
        n_tail = max(5, int(math.ceil(0.10 * len(sub))))
        tail = sub.tail(n_tail)
        coeff = tail["xi"].to_numpy(dtype=np.float64) / np.maximum(np.power(tail["tp"].to_numpy(dtype=np.float64) / T_OSC, 1.5), 1.0e-18)
        vals[i] = max(float(np.median(coeff) * (sub["F0"].iloc[0] ** 2)), 1.0e-8)
    return vals


def make_meta(df, theta_values, finf_fixed):
    return {
        "xi": df["xi"].to_numpy(dtype=np.float64),
        "tp": df["tp"].to_numpy(dtype=np.float64),
        "v_w": df["v_w"].to_numpy(dtype=np.float64),
        "theta_idx": df["theta_idx"].to_numpy(dtype=np.int64),
        "F0_sq": np.square(df["F0"].to_numpy(dtype=np.float64)),
        "n_theta": len(theta_values),
        "vw_values": np.sort(df["v_w"].unique()),
        "Finf": np.asarray(finf_fixed, dtype=np.float64),
    }


def unpack_params(params, model_name, meta):
    idx = 0
    if model_name == "amplitude":
        a1 = float(params[idx]); idx += 1
        alpha = float(params[idx]); idx += 1
        r = float(params[idx]); idx += 1
        return {"A1": a1, "alpha": alpha, "r": r}
    if model_name == "warp":
        n_vw = len(meta["vw_values"])
        scales = np.asarray(params[idx : idx + n_vw], dtype=np.float64); idx += n_vw
        r = float(params[idx]); idx += 1
        return {"scales": scales, "r": r}
    if model_name == "combined":
        n_vw = len(meta["vw_values"])
        scales = np.asarray(params[idx : idx + n_vw], dtype=np.float64); idx += n_vw
        a1 = float(params[idx]); idx += 1
        alpha = float(params[idx]); idx += 1
        r = float(params[idx]); idx += 1
        return {"scales": scales, "A1": a1, "alpha": alpha, "r": r}
    raise ValueError(f"Unknown model {model_name}")


def build_x0_bounds(model_name, meta, initial):
    if model_name == "amplitude":
        x0 = np.asarray([initial["A1"], initial["alpha"], initial["r"]], dtype=np.float64)
        lb = np.asarray([0.5, -1.0, 0.1], dtype=np.float64)
        ub = np.asarray([1.5, 1.0, 50.0], dtype=np.float64)
    elif model_name == "warp":
        n_vw = len(meta["vw_values"])
        x0 = np.concatenate([np.asarray(initial["scales"], dtype=np.float64), [initial["r"]]])
        lb = np.concatenate([np.full(n_vw, 0.5, dtype=np.float64), [0.1]])
        ub = np.concatenate([np.full(n_vw, 1.5, dtype=np.float64), [50.0]])
    elif model_name == "combined":
        n_vw = len(meta["vw_values"])
        x0 = np.concatenate([np.asarray(initial["scales"], dtype=np.float64), [initial["A1"], initial["alpha"], initial["r"]]])
        lb = np.concatenate([np.full(n_vw, 0.5, dtype=np.float64), [0.5, -1.0, 0.1]])
        ub = np.concatenate([np.full(n_vw, 1.5, dtype=np.float64), [1.5, 1.0, 50.0]])
    else:
        raise ValueError(f"Unknown model {model_name}")
    return x0, lb, ub


def scale_from_params(v_w, params, model_name, meta):
    v_w = np.asarray(v_w, dtype=np.float64)
    if model_name == "amplitude":
        return np.ones_like(v_w, dtype=np.float64)
    scales = np.ones_like(v_w, dtype=np.float64)
    for i, vw in enumerate(meta["vw_values"]):
        scales[np.isclose(v_w, float(vw), atol=1.0e-8)] = float(params["scales"][i])
    return scales


def amplitude_from_params(v_w, params, model_name):
    v_w = np.asarray(v_w, dtype=np.float64)
    if model_name == "warp":
        return np.ones_like(v_w, dtype=np.float64)
    return float(params["A1"]) * np.power(v_w, float(params["alpha"]))


def model_eval(params_vec, meta, model_name):
    params = unpack_params(params_vec, model_name, meta)
    scale = scale_from_params(meta["v_w"], params, model_name, meta)
    amp = amplitude_from_params(meta["v_w"], params, model_name)
    tp_scaled = meta["tp"] * scale
    plateau = np.power(meta["tp"] / T_OSC, 1.5) * meta["Finf"][meta["theta_idx"]] / meta["F0_sq"]
    transient = np.power(np.maximum(scale, 1.0e-18), -1.5) / (
        1.0 + np.power(np.maximum(tp_scaled, 1.0e-18) / T_C_FIXED, float(params["r"]))
    )
    return amp * (plateau + transient)


def residual_vector(params_vec, meta, model_name):
    yfit = model_eval(params_vec, meta, model_name)
    return (yfit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)


def fit_model(df, theta_values, finf_fixed, model_name, initial):
    meta = make_meta(df, theta_values, finf_fixed)
    x0, lb, ub = build_x0_bounds(model_name, meta, initial)
    fun = lambda p: residual_vector(p, meta, model_name)
    huber = least_squares(fun, x0, bounds=(lb, ub), loss="huber", f_scale=0.05, max_nfev=15000)
    final = least_squares(fun, huber.x, bounds=(lb, ub), loss="linear", max_nfev=15000)
    yfit = model_eval(final.x, meta, model_name)
    resid = (yfit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    crit = aic_bic_from_resid(resid, len(final.x))
    payload = {
        "model": model_name,
        "success": bool(final.success),
        "message": str(final.message),
        "params": to_native(unpack_params(final.x, model_name, meta)),
        "rel_rmse": rel_rmse(meta["xi"], yfit),
        "n_points": int(len(df)),
        "n_params": int(len(final.x)),
        "AIC": crit["AIC"],
        "BIC": crit["BIC"],
        "llf": crit["llf"],
        "rss": crit["rss"],
    }
    return {"result": final, "payload": payload, "meta": meta, "resid": resid, "yfit": yfit}


def load_initial_guesses(meta):
    amp_guess = {"A1": 1.0, "alpha": -0.1, "r": 2.5}
    if AMP_SUMMARY.exists():
        payload = json.loads(AMP_SUMMARY.read_text())
        best = payload.get("best_fit", {})
        amp_guess["A1"] = float(best.get("A1", 1.0))
        amp_guess["alpha"] = float(best.get("alpha", -0.1))
        amp_guess["r"] = float(best.get("r", 2.5))

    warp_guess = {"scales": np.ones(len(meta["vw_values"]), dtype=np.float64), "r": 2.5}
    if WARP_PARAMS.exists():
        payload = json.loads(WARP_PARAMS.read_text())
        scales = []
        for vw in meta["vw_values"]:
            scales.append(float(payload["scales"].get(f"{vw:.1f}", 1.0)))
        warp_guess["scales"] = np.asarray(scales, dtype=np.float64)
        warp_guess["r"] = float(payload.get("r", 2.5))

    combined_guess = {
        "scales": warp_guess["scales"],
        "A1": amp_guess["A1"],
        "alpha": amp_guess["alpha"],
        "r": warp_guess["r"],
    }
    return {"amplitude": amp_guess, "warp": warp_guess, "combined": combined_guess}


def bootstrap_one(seed, df, theta_values, finf_fixed, models, initials):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(df), size=len(df))
    boot_df = df.iloc[idx].reset_index(drop=True)
    out = {"seed": int(seed)}
    for model_name in models:
        try:
            fit = fit_model(boot_df, theta_values, finf_fixed, model_name, initials[model_name])
            out[model_name] = fit["payload"]
        except Exception:
            out[model_name] = None
    return out


def compute_pairwise(results, models):
    summary = {}
    for i, m1 in enumerate(models):
        for m2 in models[i + 1 :]:
            deltas_aic = []
            deltas_bic = []
            for row in results:
                p1 = row.get(m1)
                p2 = row.get(m2)
                if p1 is None or p2 is None:
                    continue
                deltas_aic.append(float(p1["AIC"] - p2["AIC"]))
                deltas_bic.append(float(p1["BIC"] - p2["BIC"]))
            da = np.asarray(deltas_aic, dtype=np.float64)
            db = np.asarray(deltas_bic, dtype=np.float64)
            if da.size == 0:
                continue
            key = f"{m1}_minus_{m2}"
            summary[key] = {
                "n": int(da.size),
                "delta_AIC": {
                    "median": float(np.nanmedian(da)),
                    "p16": float(np.nanpercentile(da, 16)),
                    "p84": float(np.nanpercentile(da, 84)),
                    "p2p5": float(np.nanpercentile(da, 2.5)),
                    "p97p5": float(np.nanpercentile(da, 97.5)),
                    "frac_lt_zero": float(np.mean(da < 0.0)),
                },
                "delta_BIC": {
                    "median": float(np.nanmedian(db)),
                    "p16": float(np.nanpercentile(db, 16)),
                    "p84": float(np.nanpercentile(db, 84)),
                    "p2p5": float(np.nanpercentile(db, 2.5)),
                    "p97p5": float(np.nanpercentile(db, 97.5)),
                    "frac_lt_zero": float(np.mean(db < 0.0)),
                },
            }
    return summary


def collect_pairwise_deltas(raw_results, models):
    out = {}
    for i, m1 in enumerate(models):
        for m2 in models[i + 1 :]:
            vals = []
            for row in raw_results:
                p1 = row.get(m1)
                p2 = row.get(m2)
                if p1 is None or p2 is None:
                    continue
                vals.append(float(p1["AIC"] - p2["AIC"]))
            if vals:
                out[f"{m1}_minus_{m2}"] = np.asarray(vals, dtype=np.float64)
    return out


def plot_delta_histograms(raw_deltas, pairwise, outpath):
    keys = list(raw_deltas.keys())
    if not keys:
        return
    fig, axes = plt.subplots(len(keys), 1, figsize=(7.2, 3.2 * len(keys)))
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        stats = pairwise[key]["delta_AIC"]
        vals = raw_deltas[key]
        ax.axvline(0.0, color="black", lw=1.0, ls="--")
        ax.hist(vals, bins=30, color="tab:blue", alpha=0.70)
        ax.set_title(f"{key} ΔAIC summary")
        ax.text(
            0.02,
            0.95,
            f"median={stats['median']:.2f}\n16/84=[{stats['p16']:.2f}, {stats['p84']:.2f}]\nP(<0)={stats['frac_lt_zero']:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        ax.set_xlabel("ΔAIC")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_delta_cdfs(raw_results, models, outpath):
    pairs = []
    for i, m1 in enumerate(models):
        for m2 in models[i + 1 :]:
            vals = []
            for row in raw_results:
                p1 = row.get(m1)
                p2 = row.get(m2)
                if p1 is None or p2 is None:
                    continue
                vals.append(float(p1["AIC"] - p2["AIC"]))
            if vals:
                pairs.append((f"{m1}-{m2}", np.sort(np.asarray(vals, dtype=np.float64))))
    if not pairs:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for label, vals in pairs:
        y = np.arange(1, len(vals) + 1, dtype=np.float64) / len(vals)
        ax.plot(vals, y, lw=2.0, label=label)
    ax.axvline(0.0, color="black", lw=1.0, ls="--")
    ax.set_xlabel("ΔAIC")
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_param_scatter(raw_results, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    alpha_vals = []
    r_amp = []
    for row in raw_results:
        amp = row.get("amplitude")
        if amp is not None:
            alpha_vals.append(float(amp["params"]["alpha"]))
            r_amp.append(float(amp["params"]["r"]))
    if alpha_vals:
        axes[0].scatter(alpha_vals, r_amp, s=12, alpha=0.45, color="tab:blue")
        axes[0].set_xlabel(r"$\alpha$")
        axes[0].set_ylabel(r"$r$")
        axes[0].set_title("Amplitude bootstrap")
        axes[0].grid(alpha=0.2)

    r_warp = []
    s03 = []
    for row in raw_results:
        warp = row.get("warp")
        if warp is not None:
            r_warp.append(float(warp["params"]["r"]))
            s03.append(float(warp["params"]["scales"][0]))
    if s03:
        axes[1].scatter(s03, r_warp, s=12, alpha=0.45, color="tab:orange")
        axes[1].set_xlabel(r"$s(v_w=0.3)$")
        axes[1].set_ylabel(r"$r$")
        axes[1].set_title("Warp bootstrap")
        axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for model_name in models:
        if model_name not in DEFAULT_MODELS:
            raise ValueError(f"Unsupported model '{model_name}'. Choose from {DEFAULT_MODELS}.")
    print("[mode] fast bootstrap enabled: per-theta F_inf will be fixed during refits")

    df, theta_values, f0_source = load_dataset(args, outdir)
    finf_fixed, finf_source = load_fixed_finf(theta_values)
    if finf_fixed is None:
        finf_fixed = estimate_finf_once(df, theta_values)
        finf_source = "estimated_from_full_data"
    meta = make_meta(df, theta_values, finf_fixed)
    initials = load_initial_guesses(meta)

    full_fit_payloads = {}
    for model_name in models:
        print(f"[fit] full-data {model_name}")
        fit = fit_model(df, theta_values, finf_fixed, model_name, initials[model_name])
        full_fit_payloads[model_name] = fit["payload"]

    save_json(outdir / "full_fit_models.json", full_fit_payloads)

    seeds = np.arange(args.nboot, dtype=np.int64) + int(args.seed)
    if tqdm is not None and args.n_jobs == 1:
        iterator = tqdm(seeds, desc="bootstrap", total=len(seeds))
        raw_results = [bootstrap_one(int(seed), df, theta_values, finf_fixed, models, initials) for seed in iterator]
    else:
        if tqdm is not None:
            print(f"[bootstrap] running {args.nboot} resamples in parallel (progress bar disabled in parallel mode)")
        raw_results = Parallel(n_jobs=args.n_jobs, prefer="threads")(
            delayed(bootstrap_one)(int(seed), df, theta_values, finf_fixed, models, initials) for seed in seeds
        )

    with open(outdir / "bootstrap_results.pkl", "wb") as f:
        pickle.dump(raw_results, f)

    pairwise = compute_pairwise(raw_results, models)
    raw_deltas = collect_pairwise_deltas(raw_results, models)
    comparison = {
        "status": "ok",
        "nboot": int(args.nboot),
        "models": models,
        "fast_mode": bool(args.fast),
        "f0_source": f0_source,
        "finf_source": finf_source,
        "full_fit": full_fit_payloads,
        "pairwise": pairwise,
    }
    save_json(outdir / "model_comparison_timewarp_bootstrap.json", comparison)

    plot_delta_histograms(raw_deltas, pairwise, outdir / "delta_aic_histograms.png")
    plot_delta_cdfs(raw_results, models, outdir / "delta_aic_cdf.png")
    plot_param_scatter(raw_results, outdir / "bootstrap_param_scatter.png")

    for key, stats in pairwise.items():
        m1, _, m2 = key.partition("_minus_")
        aic_stats = stats["delta_AIC"]
        print(
            f"{m1} beats {m2} in {100.0 * aic_stats['frac_lt_zero']:.1f}% of bootstraps "
            f"(median ΔAIC={aic_stats['median']:.2f}, 16/84=[{aic_stats['p16']:.2f}, {aic_stats['p84']:.2f}])"
        )

    final_summary = {
        "status": "ok",
        "nboot": int(args.nboot),
        "models": models,
        "fast_mode": bool(args.fast),
        "f0_source": f0_source,
        "finf_source": finf_source,
        "full_fit_rel_rmse": {name: full_fit_payloads[name]["rel_rmse"] for name in models},
        "pairwise": pairwise,
    }
    save_json(outdir / "final_summary.json", final_summary)
    print(json.dumps(to_native(final_summary), sort_keys=True))


if __name__ == "__main__":
    outdir = ROOT / OUTDIR_DEFAULT
    try:
        main()
    except Exception as exc:
        outdir.mkdir(parents=True, exist_ok=True)
        payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
        save_json(outdir / "_error.json", payload)
        print(json.dumps(payload, sort_keys=True))
        raise
