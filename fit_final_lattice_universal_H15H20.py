#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ode.hom_ODE.percolation import PercolationCache


DEFAULT_OUTDIR = ROOT / "results_final_lattice_H15H20"
DEFAULT_H_TRAIN = [1.5, 2.0]
DEFAULT_H_VALIDATION = [1.0]
DEFAULT_VW_TAGS = ["v3", "v5", "v7", "v9"]
DEFAULT_THETA_PANELS = [
    0.2617993878,
    0.7853981634,
    1.3089969390,
    1.8325957146,
    2.3561944902,
    2.8797932658,
]
RHO_CANDIDATES = [
    ROOT / "rho_noPT_data.txt",
    ROOT / "lattice_data/data/rho_noPT_data.txt",
    ROOT / "results_collapse/F0_table.csv",
]
FIT_TABLE_CANDIDATES = [
    ROOT / "results_hf/fit_table.csv",
]
BEST_BETA_CANDIDATES = [
    ROOT / "results_collapse/best_beta.json",
]
RAW_VW_GLOBS = [
    ROOT / "lattice_data/data/energy_ratio_by_theta_data_v*.txt",
    ROOT / "energy_ratio_by_theta_data_v*.txt",
]


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Could not parse boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit the final lattice-only universal model on H*=1.5 and 2.0."
    )
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument("--nboot", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fix-beta", type=str2bool, default=True)
    parser.add_argument("--t-osc", type=float, default=1.5)
    parser.add_argument("--use-analytic-h", type=str2bool, default=True)
    parser.add_argument("--use-stage2-calibration", type=str2bool, default=True)
    parser.add_argument("--plot", type=str2bool, default=True)
    parser.add_argument("--validation-H1p0", type=str2bool, default=False)
    parser.add_argument("--tp-min", type=float, default=None)
    parser.add_argument("--tp-max", type=float, default=None)
    parser.add_argument("--vw-folders", nargs="*", default=None)
    parser.add_argument("--rho-file", type=str, default="")
    parser.add_argument("--fit-table", type=str, default="")
    parser.add_argument("--best-beta", type=str, default="")
    return parser.parse_args()


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
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_native(payload), indent=2, sort_keys=True))


def error_exit(outdir: Path, message: str):
    payload = {"status": "error", "message": message}
    save_json(outdir / "_error.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 1


def h_theta(theta):
    theta = np.asarray(theta, dtype=np.float64)
    delta = np.abs(theta - np.pi)
    mask = delta < 1.0e-6
    half = 0.5 * theta
    cos_half = np.cos(half)
    denom = np.maximum(cos_half * cos_half, 1.0e-300)
    h = np.log(np.e / denom)
    h[mask] = np.nan
    return h


def rel_rmse(y_true, y_fit):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square((y_fit - y_true) / np.maximum(np.abs(y_true), 1.0e-12)))))


def aic_bic_from_residuals(resid, k):
    resid = np.asarray(resid, dtype=np.float64)
    n = max(int(resid.size), 1)
    rss = max(float(np.sum(np.square(resid))), 1.0e-18)
    aic = n * math.log(rss / n) + 2.0 * k
    bic = n * math.log(rss / n) + k * math.log(n)
    return float(aic), float(bic)


def approx_covariance(result):
    jac = np.asarray(result.jac, dtype=np.float64)
    if jac.ndim != 2 or jac.shape[0] <= jac.shape[1]:
        return None
    try:
        jt_j = jac.T @ jac
        dof = max(jac.shape[0] - jac.shape[1], 1)
        rss = float(np.sum(np.square(result.fun)))
        sigma2 = rss / dof
        cov = np.linalg.pinv(jt_j) * sigma2
        return cov
    except Exception:
        return None


def resolve_existing_file(candidates, explicit=""):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing requested file: {path}")
        return path
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing required file. Tried: {[str(c) for c in candidates]}")


def parse_vw_from_name(name: str) -> float:
    match = re.search(r"v(\d+(?:p\d+)?)", name)
    if not match:
        raise ValueError(f"Could not infer v_w from name: {name}")
    token = match.group(1).replace("p", ".")
    if "." in token:
        return float(token)
    return float(f"0.{token}")


def autodiscover_vw_sources():
    found = {}
    for pattern in RAW_VW_GLOBS:
        rel = Path(pattern).relative_to(ROOT)
        for path in sorted(ROOT.glob(str(rel))):
            try:
                vw = parse_vw_from_name(path.stem)
            except ValueError:
                continue
            found[f"v{int(round(vw * 10))}"] = path.resolve()
    return found


def resolve_vw_sources(tokens):
    discovered = autodiscover_vw_sources()
    if not tokens:
        tokens = [tag for tag in DEFAULT_VW_TAGS if tag in discovered]
    if not tokens:
        raise FileNotFoundError("No raw lattice v_w files found. Supply --vw-folders or add energy_ratio_by_theta_data_v*.txt.")

    resolved = []
    for token in tokens:
        path = Path(token)
        if path.exists():
            if path.is_dir():
                raw = sorted(path.glob("energy_ratio_by_theta_data_v*.txt"))
                if not raw:
                    raise FileNotFoundError(f"Folder {path} does not contain a supported raw lattice file.")
                source = raw[0].resolve()
                tag = source.stem
                resolved.append((parse_vw_from_name(tag), source, tag))
                continue
            source = path.resolve()
            resolved.append((parse_vw_from_name(source.stem), source, source.stem))
            continue

        if token in discovered:
            source = discovered[token]
            resolved.append((parse_vw_from_name(token), source, token))
            continue

        candidate = ROOT / "lattice_data/data" / f"energy_ratio_by_theta_data_{token}.txt"
        if candidate.exists():
            resolved.append((parse_vw_from_name(token), candidate.resolve(), token))
            continue

        raise FileNotFoundError(f"Could not resolve lattice source for {token}.")

    resolved = sorted(resolved, key=lambda item: item[0])
    return resolved


def load_best_beta(path: Path):
    payload = json.loads(path.read_text())
    if "beta" not in payload:
        raise RuntimeError(f"{path} does not contain a 'beta' entry.")
    return float(payload["beta"]), payload


def load_analytic_h_fits(path: Path, t_osc: float):
    df = pd.read_csv(path)
    if not {"dataset", "A", "gamma"} <= set(df.columns):
        raise RuntimeError(f"{path} must contain columns dataset,A,gamma.")
    rows = {str(row["dataset"]).strip(): row for _, row in df.iterrows()}
    if "noPT" not in rows or "Finf" not in rows:
        raise RuntimeError(f"{path} must contain 'noPT' and 'Finf' rows.")
    a0 = float(rows["noPT"]["A"])
    gamma0 = float(rows["noPT"]["gamma"])
    ainf_old = float(rows["Finf"]["A"])
    gammainf = float(rows["Finf"]["gamma"])
    return {
        "A0": a0,
        "gamma0": gamma0,
        "Ainf_raw": ainf_old,
        "Ainf": ainf_old * (float(t_osc) ** 1.5),
        "gammainf": gammainf,
        "source": path,
        "scaled_for_tosc": True,
    }


def load_f0_table(path: Path):
    df = pd.read_csv(path, sep=r"\s+|,", engine="python", comment="#")
    lower = {str(col).lower(): col for col in df.columns}
    if {"theta", "f0"} <= set(lower):
        out = df.rename(columns={lower["theta"]: "theta", lower["f0"]: "F0"})[["theta", "F0"]].copy()
        out = out[np.isfinite(out["theta"]) & np.isfinite(out["F0"]) & (out["F0"] > 0.0)].copy()
        return out.sort_values("theta").reset_index(drop=True)

    theta_col = lower.get("theta0", lower.get("theta"))
    h_col = lower.get("h_pt", lower.get("hstar", lower.get("h")))
    rho_col = lower.get("rho")
    if theta_col is None or h_col is None or rho_col is None:
        raise RuntimeError(
            f"{path} must contain either [theta,F0] or [theta0/theta,H_PT/Hstar/H,rho]. Found {list(df.columns)}"
        )

    work = pd.DataFrame(
        {
            "theta": pd.to_numeric(df[theta_col], errors="coerce"),
            "H": pd.to_numeric(df[h_col], errors="coerce"),
            "rho": pd.to_numeric(df[rho_col], errors="coerce"),
        }
    )
    work = work[np.isfinite(work["theta"]) & np.isfinite(work["H"]) & np.isfinite(work["rho"])].copy()
    work["F0"] = work["rho"] / np.maximum((1.0 - np.cos(work["theta"])) * np.power(work["H"], 1.5), 1.0e-18)
    out = work.groupby("theta", as_index=False)["F0"].median()
    out = out[np.isfinite(out["theta"]) & np.isfinite(out["F0"]) & (out["F0"] > 0.0)].copy()
    return out.sort_values("theta").reset_index(drop=True)


def nearest_values(theta, ref_theta, ref_values):
    theta = np.asarray(theta, dtype=np.float64)
    ref_theta = np.asarray(ref_theta, dtype=np.float64)
    ref_values = np.asarray(ref_values, dtype=np.float64)
    idx = np.abs(theta[:, None] - ref_theta[None, :]).argmin(axis=1)
    return ref_values[idx]


def load_one_raw_scan(path: Path, vw: float, perc: PercolationCache):
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    if df.shape[1] < 5:
        raise RuntimeError(f"Raw lattice file {path} does not have enough columns.")
    names = ["theta", "H", "beta_over_H", "beta", "xi", "std_ratio", "N_samples"]
    df = df.iloc[:, : len(names)].copy()
    df.columns = names[: df.shape[1]]
    work = pd.DataFrame(
        {
            "theta": pd.to_numeric(df["theta"], errors="coerce"),
            "H": pd.to_numeric(df["H"], errors="coerce"),
            "beta_over_H": pd.to_numeric(df["beta_over_H"], errors="coerce"),
            "xi": pd.to_numeric(df["xi"], errors="coerce"),
        }
    )
    if "std_ratio" in df.columns and "N_samples" in df.columns:
        std = pd.to_numeric(df["std_ratio"], errors="coerce")
        n_samp = pd.to_numeric(df["N_samples"], errors="coerce")
        work["xi_sem"] = std / np.sqrt(np.maximum(n_samp, 1.0))
    else:
        work["xi_sem"] = np.nan
    work = work[
        np.isfinite(work["theta"])
        & np.isfinite(work["H"])
        & np.isfinite(work["beta_over_H"])
        & np.isfinite(work["xi"])
    ].copy()
    work["tp"] = [
        perc.get(float(h_val), float(beta_over_h), float(vw))
        for h_val, beta_over_h in zip(work["H"], work["beta_over_H"])
    ]
    work["v_w"] = float(vw)
    work["source_file"] = path.name
    work = work[np.isfinite(work["tp"]) & (work["tp"] > 0.0) & (work["xi"] > 0.0)].copy()
    return work.sort_values(["H", "theta", "tp"]).reset_index(drop=True)


def build_dataframe(args):
    perc = PercolationCache()
    sources = resolve_vw_sources(args.vw_folders)
    frames = []
    for vw, path, tag in sources:
        print(f"[load] {tag} -> v_w={vw:.3f} from {path.relative_to(ROOT)}")
        frames.append(load_one_raw_scan(path, vw, perc))
    if not frames:
        raise RuntimeError("No lattice data frames loaded.")
    df = pd.concat(frames, ignore_index=True)
    total_rows = int(len(df))
    if args.tp_min is not None:
        df = df[df["tp"] >= float(args.tp_min)].copy()
    if args.tp_max is not None:
        df = df[df["tp"] <= float(args.tp_max)].copy()
    df = df[
        np.isfinite(df["theta"])
        & np.isfinite(df["H"])
        & np.isfinite(df["tp"])
        & np.isfinite(df["xi"])
        & np.isfinite(df["v_w"])
    ].copy()
    df = df[(df["tp"] > 0.0) & (df["xi"] > 0.0)].copy()
    return df.sort_values(["v_w", "H", "theta", "tp"]).reset_index(drop=True), total_rows


def estimate_finf_from_data(df, f0_values, t_osc):
    records = []
    for theta, sub in df.groupby("theta"):
        work = sub.sort_values("tp").copy()
        n_tail = max(5, int(math.ceil(0.10 * len(work))))
        tail = work.tail(n_tail)
        f0 = float(f0_values[np.isclose(np.asarray(list(f0_values.keys())), theta)][0]) if False else None
        values = tail["xi"].to_numpy(dtype=np.float64) * (
            np.square(tail["F0_eval"].to_numpy(dtype=np.float64))
        ) / np.maximum(np.power(tail["tp"].to_numpy(dtype=np.float64) / t_osc, 1.5), 1.0e-18)
        records.append({"theta": float(theta), "Finf": max(float(np.median(values)), 1.0e-12)})
    return pd.DataFrame.from_records(records).sort_values("theta").reset_index(drop=True)


def attach_amplitudes(df, analytic_h, f0_table, finf_table):
    out = df.copy()
    theta_arr = out["theta"].to_numpy(dtype=np.float64)
    h_arr = h_theta(theta_arr)
    out["h"] = h_arr
    if analytic_h is not None:
        out["F0_eval"] = analytic_h["A0"] * np.power(h_arr, analytic_h["gamma0"])
        out["Finf_eval"] = analytic_h["Ainf"] * np.power(h_arr, analytic_h["gammainf"])
    else:
        theta_ref = f0_table["theta"].to_numpy(dtype=np.float64)
        f0_ref = f0_table["F0"].to_numpy(dtype=np.float64)
        finf_ref = finf_table["Finf"].to_numpy(dtype=np.float64)
        out["F0_eval"] = nearest_values(theta_arr, theta_ref, f0_ref)
        out["Finf_eval"] = nearest_values(theta_arr, theta_ref, finf_ref)
    out = out[
        np.isfinite(out["h"])
        & np.isfinite(out["F0_eval"])
        & np.isfinite(out["Finf_eval"])
        & (out["F0_eval"] > 0.0)
        & (out["Finf_eval"] > 0.0)
    ].copy()
    return out


def split_datasets(df, validation_h1):
    train = df[df["H"].isin(DEFAULT_H_TRAIN)].copy()
    validation = pd.DataFrame(columns=df.columns)
    if validation_h1:
        validation = df[df["H"].isin(DEFAULT_H_VALIDATION)].copy()
    dropped = df[~df["H"].isin(DEFAULT_H_TRAIN + (DEFAULT_H_VALIDATION if validation_h1 else []))].copy()
    return train, validation, dropped


def build_meta(df, analytic_h):
    theta_values = np.sort(df["theta"].unique())
    theta_index = {float(theta): idx for idx, theta in enumerate(theta_values)}
    work = df.copy()
    work["theta_idx"] = [theta_index[float(theta)] for theta in work["theta"]]
    meta = {
        "theta": work["theta"].to_numpy(dtype=np.float64),
        "theta_idx": work["theta_idx"].to_numpy(dtype=np.int64),
        "tp": work["tp"].to_numpy(dtype=np.float64),
        "xi": work["xi"].to_numpy(dtype=np.float64),
        "H": work["H"].to_numpy(dtype=np.float64),
        "v_w": work["v_w"].to_numpy(dtype=np.float64),
        "h": work["h"].to_numpy(dtype=np.float64),
        "F0": work["F0_eval"].to_numpy(dtype=np.float64),
        "Finf": work["Finf_eval"].to_numpy(dtype=np.float64),
        "theta_values": theta_values,
        "analytic_h": analytic_h is not None,
    }
    return work, meta


def unpack_params(params, beta_fixed, fix_beta, use_calib):
    idx = 0
    if fix_beta:
        beta = float(beta_fixed)
    else:
        beta = float(params[idx])
        idx += 1
    tc = float(params[idx])
    r = float(params[idx + 1])
    idx += 2
    c_calib = 1.0
    if use_calib:
        c_calib = float(params[idx])
        idx += 1
    return beta, tc, r, c_calib


def model_xi(params, meta, beta_fixed, fix_beta, t_osc, use_calib):
    beta, tc, r, c_calib = unpack_params(params, beta_fixed, fix_beta, use_calib)
    tp = meta["tp"]
    h = meta["h"]
    f0 = meta["F0"]
    finf = meta["Finf"]
    x = tp * np.power(meta["H"], beta)
    x = np.maximum(x, 1.0e-18)
    transient = f0 / (np.power(x, 1.5) * (1.0 + np.power(x / max(tc, 1.0e-12), r)))
    f_univ = finf / f0 + transient
    xi = np.power(tp / t_osc, 1.5) / f0 * f_univ
    return c_calib * xi


def residuals(params, meta, beta_fixed, fix_beta, t_osc, use_calib):
    model = model_xi(params, meta, beta_fixed, fix_beta, t_osc, use_calib)
    return (model - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)


def initial_params(beta0, fix_beta, use_calib):
    parts = []
    lower = []
    upper = []
    if not fix_beta:
        delta = max(0.02, 0.10 * abs(beta0))
        lo = min(beta0 - delta, beta0 + delta)
        hi = max(beta0 - delta, beta0 + delta)
        parts.append(beta0)
        lower.append(lo)
        upper.append(hi)
    parts.extend([1.5, 2.5])
    lower.extend([0.1, 0.1])
    upper.extend([20.0, 20.0])
    if use_calib:
        parts.append(1.0)
        lower.append(0.5)
        upper.append(1.5)
    return np.asarray(parts, dtype=np.float64), np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def fit_stage(meta, beta0, fix_beta, t_osc, use_calib, x0_override=None):
    x0, lower, upper = initial_params(beta0, fix_beta, use_calib)
    if x0_override is not None:
        x0 = np.asarray(x0_override, dtype=np.float64)
    fun = lambda p: residuals(p, meta, beta0, fix_beta, t_osc, use_calib)
    first = least_squares(fun, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=20000)
    final = least_squares(fun, first.x, bounds=(lower, upper), loss="linear", max_nfev=20000)
    y_fit = model_xi(final.x, meta, beta0, fix_beta, t_osc, use_calib)
    resid = (y_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    beta_fit, tc, r, c_calib = unpack_params(final.x, beta0, fix_beta, use_calib)
    cov = approx_covariance(final)
    errs = None if cov is None else np.sqrt(np.maximum(np.diag(cov), 0.0))
    aic, bic = aic_bic_from_residuals(resid, len(final.x))
    payload = {
        "beta": float(beta_fit),
        "t_c": float(tc),
        "r": float(r),
        "c_calib": float(c_calib),
        "success": bool(final.success),
        "message": str(final.message),
        "rel_rmse": rel_rmse(meta["xi"], y_fit),
        "AIC": float(aic),
        "BIC": float(bic),
        "n_params": int(len(final.x)),
        "n_points": int(meta["xi"].size),
        "dof": int(meta["xi"].size - len(final.x)),
        "residual_mean": float(np.mean(resid)),
        "residual_std": float(np.std(resid)),
        "params": final.x.tolist(),
    }
    if errs is not None:
        idx = 0
        if not fix_beta:
            payload["beta_err"] = float(errs[idx])
            idx += 1
        payload["t_c_err"] = float(errs[idx])
        payload["r_err"] = float(errs[idx + 1])
        idx += 2
        if use_calib:
            payload["c_calib_err"] = float(errs[idx])
    return payload, final.x, y_fit, resid


def bootstrap_one(seed, y_fit, frac_resid, meta, beta0, fix_beta, t_osc, use_calib, x0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, frac_resid.size, size=frac_resid.size)
    eps = frac_resid[idx]
    xi_boot = np.maximum(y_fit * (1.0 + eps), 1.0e-10)
    boot_meta = dict(meta)
    boot_meta["xi"] = xi_boot
    try:
        payload, params, _, _ = fit_stage(boot_meta, beta0, fix_beta, t_osc, use_calib, x0_override=x0)
        return payload
    except Exception:
        return None


def bootstrap_fit(meta, fit_payload, params, y_fit, t_osc, nboot, n_jobs, seed, use_calib, fix_beta):
    frac_resid = (meta["xi"] - y_fit) / np.maximum(y_fit, 1.0e-12)
    seeds = [seed + 1000 + i for i in range(nboot)]
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(bootstrap_one)(
            s,
            y_fit,
            frac_resid,
            meta,
            fit_payload["beta"],
            fix_beta,
            t_osc,
            use_calib,
            params,
        )
        for s in seeds
    )
    results = [res for res in results if res is not None]
    summary = {"n_requested": int(nboot), "n_success": int(len(results))}
    if not results:
        return summary, results

    keys = ["beta", "t_c", "r"] + (["c_calib"] if use_calib else [])
    for key in keys:
        vals = np.asarray([res[key] for res in results], dtype=np.float64)
        if vals.size == 0:
            continue
        summary[key] = {
            "p2p5": float(np.percentile(vals, 2.5)),
            "p16": float(np.percentile(vals, 16.0)),
            "p50": float(np.percentile(vals, 50.0)),
            "p84": float(np.percentile(vals, 84.0)),
            "p97p5": float(np.percentile(vals, 97.5)),
        }
    return summary, results


def curve_xi(theta, h_val, f0, finf, h_grid_x, H_val, beta, t_c, r, t_osc, c_calib):
    x = np.maximum(h_grid_x, 1.0e-18)
    tp = x / np.power(H_val, beta)
    univ = finf / f0 + f0 / (np.power(x, 1.5) * (1.0 + np.power(x / max(t_c, 1.0e-12), r)))
    return c_calib * np.power(tp / t_osc, 1.5) / f0 * univ


def make_overlay_plot(df, fit_payload, outpath: Path, t_osc, title_suffix="Stage 1"):
    theta_targets = DEFAULT_THETA_PANELS
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex=False, sharey=False)
    axes = axes.flatten()
    color_map = {1.5: "tab:blue", 2.0: "tab:orange"}
    marker_map = {0.3: "o", 0.5: "s", 0.7: "^", 0.9: "D"}
    beta = fit_payload["beta"]
    for ax, theta_target in zip(axes, theta_targets):
        theta = min(df["theta"].unique(), key=lambda val: abs(val - theta_target))
        sub = df[np.isclose(df["theta"], theta)].copy()
        x = sub["tp"].to_numpy(dtype=np.float64) * np.power(sub["H"].to_numpy(dtype=np.float64), beta)
        sub["x"] = x
        for (H_val, v_w), grp in sub.groupby(["H", "v_w"]):
            ax.scatter(
                grp["x"],
                grp["xi"],
                s=18,
                alpha=0.85,
                color=color_map[float(H_val)],
                marker=marker_map[float(v_w)],
                label=f"H={H_val:g}, v_w={v_w:.1f}",
            )
        f0 = float(sub["F0_eval"].iloc[0])
        finf = float(sub["Finf_eval"].iloc[0])
        h_val = float(sub["h"].iloc[0])
        x_grid = np.geomspace(np.min(sub["x"]) * 0.95, np.max(sub["x"]) * 1.05, 300)
        for H_val in sorted(sub["H"].unique()):
            y_curve = curve_xi(theta, h_val, f0, finf, x_grid, float(H_val), beta, fit_payload["t_c"], fit_payload["r"], t_osc, fit_payload["c_calib"])
            ax.plot(x_grid, y_curve, lw=2.0, color=color_map[float(H_val)])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$x=t_p H^\beta$")
        ax.set_ylabel(r"$\xi$")
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=4, fontsize=9)
    fig.suptitle(f"Lattice-only universal fit on H*=1.5, 2.0 ({title_suffix})", y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def heatmap_matrix(df, resid, beta, nbins=28):
    work = df.copy()
    work["x"] = work["tp"].to_numpy(dtype=np.float64) * np.power(work["H"].to_numpy(dtype=np.float64), beta)
    work["resid"] = resid
    theta_values = np.sort(work["theta"].unique())
    x_min = max(float(work["x"].min()), 1.0e-8)
    x_max = max(float(work["x"].max()), x_min * 1.01)
    x_bins = np.geomspace(x_min, x_max, nbins + 1)
    mat = np.full((len(theta_values), nbins), np.nan, dtype=np.float64)
    for i, theta in enumerate(theta_values):
        sub = work[np.isclose(work["theta"], theta)]
        inds = np.digitize(sub["x"], x_bins) - 1
        for j in range(nbins):
            vals = sub["resid"].to_numpy(dtype=np.float64)[inds == j]
            if vals.size:
                mat[i, j] = float(np.mean(vals))
    return theta_values, x_bins, mat


def plot_heatmap(df, resid, beta, outpath, title):
    theta_values, x_bins, mat = heatmap_matrix(df, resid, beta)
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    extent = [math.log10(x_bins[0]), math.log10(x_bins[-1]), -0.5, len(theta_values) - 0.5]
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="coolwarm", vmin=-0.08, vmax=0.08, extent=extent)
    ax.set_yticks(range(len(theta_values)))
    ax.set_yticklabels([f"{theta:.3f}" for theta in theta_values])
    ax.set_xlabel(r"$\log_{10} x$")
    ax.set_ylabel(r"$\theta$")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$(\xi_{\rm model}-\xi_{\rm data})/\xi_{\rm data}$")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_profile(meta, beta0, fix_beta, t_osc, outpath):
    tc_grid = np.linspace(0.5, 6.0, 60)
    rel_values = []
    r_values = []
    for tc in tc_grid:
        if fix_beta:
            x0 = np.asarray([tc, 2.5], dtype=np.float64)
            lower = np.asarray([tc, 0.1], dtype=np.float64)
            upper = np.asarray([tc, 20.0], dtype=np.float64)
            fun = lambda p: residuals(np.asarray([tc, p[0]]), meta, beta0, True, t_osc, False)
            res = least_squares(fun, np.asarray([2.5]), bounds=(np.asarray([0.1]), np.asarray([20.0])), loss="linear", max_nfev=10000)
            params = np.asarray([tc, float(res.x[0])], dtype=np.float64)
            y_fit = model_xi(params, meta, beta0, True, t_osc, False)
            rel_values.append(rel_rmse(meta["xi"], y_fit))
            r_values.append(float(res.x[0]))
        else:
            fun = lambda p: residuals(np.asarray([p[0], tc, p[1]]), meta, beta0, False, t_osc, False)
            x0 = np.asarray([beta0, 2.5], dtype=np.float64)
            delta = max(0.02, 0.10 * abs(beta0))
            bounds = (
                np.asarray([min(beta0 - delta, beta0 + delta), 0.1]),
                np.asarray([max(beta0 - delta, beta0 + delta), 20.0]),
            )
            res = least_squares(fun, x0, bounds=bounds, loss="linear", max_nfev=10000)
            params = np.asarray([float(res.x[0]), tc, float(res.x[1])], dtype=np.float64)
            y_fit = model_xi(params, meta, beta0, False, t_osc, False)
            rel_values.append(rel_rmse(meta["xi"], y_fit))
            r_values.append(float(res.x[1]))

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
    axes[0].plot(tc_grid, rel_values, color="tab:blue", lw=2.0)
    axes[0].set_ylabel("rel-RMSE")
    axes[0].grid(alpha=0.25)
    axes[1].plot(tc_grid, r_values, color="tab:orange", lw=2.0)
    axes[1].set_xlabel(r"$t_c$")
    axes[1].set_ylabel(r"best $r$")
    axes[1].grid(alpha=0.25)
    fig.suptitle(r"Profile in $t_c$ with $r$ re-optimized")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_bootstrap_hist(stage1_boot, stage2_boot, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    if stage1_boot:
        axes[0].hist([res["t_c"] for res in stage1_boot], bins=25, alpha=0.6, label="Stage 1")
        axes[1].hist([res["r"] for res in stage1_boot], bins=25, alpha=0.6, label="Stage 1")
    if stage2_boot:
        axes[0].hist([res["t_c"] for res in stage2_boot], bins=25, alpha=0.6, label="Stage 2")
        axes[1].hist([res["r"] for res in stage2_boot], bins=25, alpha=0.6, label="Stage 2")
        axes[2].hist([res["c_calib"] for res in stage2_boot], bins=25, alpha=0.8, color="tab:green")
    axes[0].set_xlabel(r"$t_c$")
    axes[1].set_xlabel(r"$r$")
    axes[2].set_xlabel(r"$c_{\rm calib}$")
    axes[0].set_ylabel("count")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def stage_summary_payload(stage_name, fit_payload, bootstrap_summary):
    out = {
        "stage": stage_name,
        "beta": fit_payload["beta"],
        "t_c": fit_payload["t_c"],
        "r": fit_payload["r"],
        "rel_rmse": fit_payload["rel_rmse"],
        "AIC": fit_payload["AIC"],
        "BIC": fit_payload["BIC"],
        "bootstrap": bootstrap_summary,
    }
    if stage_name == "stage2":
        out["c_calib"] = fit_payload["c_calib"]
    return out


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        beta_path = resolve_existing_file(BEST_BETA_CANDIDATES, args.best_beta)
        beta0, beta_payload = load_best_beta(beta_path)

        rho_path = resolve_existing_file(RHO_CANDIDATES, args.rho_file)
        f0_table = load_f0_table(rho_path)
        f0_table.to_csv(outdir / "F0_table.csv", index=False)

        analytic_h = None
        fit_table_source = None
        if args.use_analytic_h:
            try:
                fit_table_source = resolve_existing_file(FIT_TABLE_CANDIDATES, args.fit_table)
                analytic_h = load_analytic_h_fits(fit_table_source, args.t_osc)
            except Exception as exc:
                print(f"[warn] Falling back from analytic h fit: {exc}")
                analytic_h = None

        raw_df, total_rows = build_dataframe(args)
        train_df_raw, validation_df_raw, excluded_df = split_datasets(raw_df, args.validation_H1p0)
        if train_df_raw.empty:
            raise RuntimeError("Training set is empty after restricting to H*=1.5 and 2.0.")

        if analytic_h is None:
            tmp = train_df_raw.copy()
            tmp["h"] = h_theta(tmp["theta"].to_numpy(dtype=np.float64))
            theta_ref = f0_table["theta"].to_numpy(dtype=np.float64)
            f0_ref = f0_table["F0"].to_numpy(dtype=np.float64)
            tmp["F0_eval"] = nearest_values(tmp["theta"].to_numpy(dtype=np.float64), theta_ref, f0_ref)
            finf_table = estimate_finf_from_data(tmp, None, args.t_osc)
        else:
            finf_table = None

        train_df = attach_amplitudes(train_df_raw, analytic_h, f0_table, finf_table)
        validation_df = attach_amplitudes(validation_df_raw, analytic_h, f0_table, finf_table) if not validation_df_raw.empty else validation_df_raw.copy()

        train_df, train_meta = build_meta(train_df, analytic_h)
        validation_meta = None
        if not validation_df.empty:
            validation_df, validation_meta = build_meta(validation_df, analytic_h)

        print(f"[fit] Stage 1 on {len(train_df)} lattice points with H*=1.5,2.0")
        stage1_payload, stage1_params, stage1_fit, stage1_resid = fit_stage(
            train_meta, beta0, args.fix_beta, args.t_osc, use_calib=False
        )
        stage1_boot_summary, stage1_boot = bootstrap_fit(
            train_meta,
            stage1_payload,
            stage1_params,
            stage1_fit,
            args.t_osc,
            args.nboot,
            args.n_jobs,
            args.seed,
            use_calib=False,
            fix_beta=args.fix_beta,
        )

        stage2_payload = None
        stage2_boot_summary = None
        stage2_boot = []
        stage2_fit = None
        stage2_resid = None
        if args.use_stage2_calibration:
            print(f"[fit] Stage 2 with global calibration constant on {len(train_df)} training points")
            stage2_init = list(stage1_params) + [1.0]
            stage2_payload, stage2_params, stage2_fit, stage2_resid = fit_stage(
                train_meta,
                stage1_payload["beta"],
                args.fix_beta,
                args.t_osc,
                use_calib=True,
                x0_override=stage2_init,
            )
            stage2_boot_summary, stage2_boot = bootstrap_fit(
                train_meta,
                stage2_payload,
                stage2_params,
                stage2_fit,
                args.t_osc,
                args.nboot,
                args.n_jobs,
                args.seed + 50000,
                use_calib=True,
                fix_beta=args.fix_beta,
            )

        validation_summary = None
        if validation_meta is not None and validation_meta["xi"].size:
            stage1_val_fit = model_xi(
                np.asarray(stage1_payload["params"], dtype=np.float64),
                validation_meta,
                stage1_payload["beta"],
                args.fix_beta,
                args.t_osc,
                False,
            )
            validation_summary = {
                "H_values": [1.0],
                "n_points": int(validation_meta["xi"].size),
                "stage1_rel_rmse": rel_rmse(validation_meta["xi"], stage1_val_fit),
            }
            if stage2_payload is not None:
                stage2_val_fit = model_xi(
                    np.asarray(stage2_payload["params"], dtype=np.float64),
                    validation_meta,
                    stage2_payload["beta"],
                    args.fix_beta,
                    args.t_osc,
                    True,
                )
                validation_summary["stage2_rel_rmse"] = rel_rmse(validation_meta["xi"], stage2_val_fit)

        model_comparison = {
            "stage1": {
                "rel_rmse": stage1_payload["rel_rmse"],
                "AIC": stage1_payload["AIC"],
                "BIC": stage1_payload["BIC"],
            }
        }
        preferred = "stage1"
        calibration_interpretation = "stage2_disabled"
        if stage2_payload is not None:
            delta_aic = stage2_payload["AIC"] - stage1_payload["AIC"]
            delta_bic = stage2_payload["BIC"] - stage1_payload["BIC"]
            preferred = "stage2" if (delta_aic < 0.0 and delta_bic < 0.0) else "stage1"
            model_comparison["stage2"] = {
                "rel_rmse": stage2_payload["rel_rmse"],
                "AIC": stage2_payload["AIC"],
                "BIC": stage2_payload["BIC"],
                "delta_AIC_vs_stage1": float(delta_aic),
                "delta_BIC_vs_stage1": float(delta_bic),
                "preferred": preferred,
            }
            if stage2_boot_summary and "c_calib" in stage2_boot_summary:
                ci = stage2_boot_summary["c_calib"]
                consistent = ci["p2p5"] <= 1.0 <= ci["p97p5"]
                calibration_interpretation = (
                    "c_calib_consistent_with_1" if consistent else "small_global_calibration_offset_detected"
                )
        residual_verdict = (
            "residuals_not_structureless_enough_for_strict_vw-independent_acceptance"
            if (stage2_payload is not None and stage2_payload["rel_rmse"] > 0.05) or stage1_payload["rel_rmse"] > 0.05
            else "residuals_small_enough_to_support_the_universal_model"
        )

        save_json(outdir / "model_comparison.json", model_comparison)

        if args.plot:
            make_overlay_plot(train_df, stage1_payload, outdir / "overlay_universal.png", args.t_osc, title_suffix="Stage 1")
            plot_heatmap(train_df, stage1_resid, stage1_payload["beta"], outdir / "residual_heatmap_stage1.png", "Stage 1 fractional residuals")
            if stage2_payload is not None:
                plot_heatmap(train_df, stage2_resid, stage2_payload["beta"], outdir / "residual_heatmap_stage2.png", "Stage 2 fractional residuals")
            plot_profile(train_meta, stage1_payload["beta"], args.fix_beta, args.t_osc, outdir / "fit_t_c_r_profile.png")
            plot_bootstrap_hist(stage1_boot, stage2_boot, outdir / "bootstrap_hist.png")

        summary = {
            "status": "ok",
            "beta_used": float(stage1_payload["beta"]),
            "beta_source": beta_path,
            "beta_fixed": bool(args.fix_beta),
            "t_osc": float(args.t_osc),
            "training_set_H": [1.5, 2.0],
            "excluded_note": "H*=0.5 and lower excluded from the main fit because of bubble misalignment / staggered-entry contamination.",
            "data_source": {
                "rho_file": rho_path,
                "fit_table": fit_table_source,
                "vw_tags": [tag for _, _, tag in resolve_vw_sources(args.vw_folders)],
                "tp_mapping": "t_p = t_perc_RD(H_*, beta/H_*, v_w) from ode/hom_ODE/percolation.py",
            },
            "analytic_amplitudes": {
                "mode": "analytic_h" if analytic_h is not None else "table_fallback",
                "A0": None if analytic_h is None else analytic_h["A0"],
                "gamma0": None if analytic_h is None else analytic_h["gamma0"],
                "Ainf_raw_from_fit_table": None if analytic_h is None else analytic_h["Ainf_raw"],
                "Ainf_used": None if analytic_h is None else analytic_h["Ainf"],
                "gammainf": None if analytic_h is None else analytic_h["gammainf"],
                "Ainf_scaled_by_tosc_3half": None if analytic_h is None else True,
            },
            "n_points": {
                "total_loaded": int(total_rows),
                "after_basic_filtering": int(len(raw_df)),
                "training_used": int(len(train_df)),
                "validation_used": int(len(validation_df)) if not validation_df.empty else 0,
                "excluded_other_H": int(len(excluded_df)),
            },
            "stage1": stage_summary_payload("stage1", stage1_payload, stage1_boot_summary),
            "stage2": None if stage2_payload is None else stage_summary_payload("stage2", stage2_payload, stage2_boot_summary),
            "validation": validation_summary,
            "model_comparison": {
                "preferred": preferred,
                "calibration_interpretation": calibration_interpretation,
                **model_comparison,
            },
            "residuals_acceptance": residual_verdict,
        }
        save_json(outdir / "final_summary.json", summary)
        print(json.dumps(to_native(summary), sort_keys=True))
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        save_json(outdir / "_error.json", {"status": "error", "message": str(exc), "traceback": trace})
        print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
