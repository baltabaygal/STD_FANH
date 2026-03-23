#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.signal import fftconvolve, savgol_filter
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fit_vw_amplitude as base_fit


OUTDIR_DEFAULT = "results_vw_smearing_tests"
ODE_CANDIDATES = [
    ROOT / "xi_DM_ODE_results.txt",
    ROOT / "ode/xi_DM_ODE_results.txt",
    ROOT / "xi_ode_scan.txt",
]
PERCOLATION_DIR_CANDIDATES = [
    ROOT / "percolation",
    ROOT / "lattice_data/percolation",
]
RESULTS_HF_CANDIDATES = [
    ROOT / "results_hf/final_summary.json",
    ROOT / "results_hf/fit_table.csv",
]
AMPLITUDE_RESULT_CANDIDATES = [
    ROOT / "results_vw_amp/global_fit_optionB.json",
    ROOT / "results_tosc_lattice/fit_vw_amplitude/global_fit_optionB.json",
]
TIMEWARP_RESULT_CANDIDATES = [
    ROOT / "results_vw_timewarp/params_optionB.json",
]
THETA_TARGETS = np.array([0.262, 0.785, 1.309, 1.833, 2.356, 2.880], dtype=np.float64)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_native(payload), indent=2, sort_keys=True))


def error_exit(outdir: Path, exc: Exception):
    payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "_error.json", payload)
    print(json.dumps(payload, sort_keys=True))


def parse_args():
    p = argparse.ArgumentParser(description="Run v_w smearing diagnostics and convolution tests.")
    p.add_argument("--rho", type=str, default="")
    p.add_argument("--vw-folders", nargs="*", default=None)
    p.add_argument("--h-values", type=float, nargs="+", default=[1.5, 2.0])
    p.add_argument("--tp-min", type=float, default=None)
    p.add_argument("--tp-max", type=float, default=None)
    p.add_argument("--t-osc", type=float, default=1.5)
    p.add_argument("--use-analytic-h", dest="use_analytic_h", action="store_true")
    p.add_argument("--no-analytic-h", dest="use_analytic_h", action="store_false")
    p.add_argument("--nboot", type=int, default=200)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--outdir", type=str, default=OUTDIR_DEFAULT)
    p.add_argument("--ode-file", type=str, default="")
    p.add_argument("--percolation-dir", type=str, default="")
    p.add_argument("--dpi", type=int, default=220)
    p.set_defaults(use_analytic_h=True)
    return p.parse_args()


def h_theta(theta):
    theta = np.asarray(theta, dtype=np.float64)
    cos_half = np.cos(theta / 2.0)
    out = np.full_like(theta, np.nan, dtype=np.float64)
    good = np.abs(cos_half) > 1.0e-12
    out[good] = np.log(np.e / np.maximum(cos_half[good] ** 2, 1.0e-300))
    return out


def resolve_ode_file(user_value: str) -> Path | None:
    if user_value:
        path = Path(user_value).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing requested ODE file: {path}")
        return path
    for candidate in ODE_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_percolation_dir(user_value: str) -> Path | None:
    if user_value:
        path = Path(user_value).resolve()
        if not path.exists():
            print(f"[warn] percolation directory not found: {path}")
            return None
        return path
    for candidate in PERCOLATION_DIR_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()
    return None


def load_analytic_h_defaults():
    for path in RESULTS_HF_CANDIDATES:
        if not path.exists():
            continue
        if path.suffix == ".json":
            payload = json.loads(path.read_text())
            return {
                "A0": float(payload["noPT"]["best_fit"]["A"]),
                "gamma0": float(payload["noPT"]["best_fit"]["gamma"]),
                "Ainf": float(payload["Finf"]["best_fit"]["A"]),
                "gamma_inf": float(payload["Finf"]["best_fit"]["gamma"]),
                "source": str(path),
            }
        df = pd.read_csv(path)
        if "dataset" not in df.columns:
            continue
        df = df.copy()
        df["dataset"] = df["dataset"].astype(str)
        row0 = df[df["dataset"].str.lower() == "nopt"]
        rowi = df[df["dataset"].str.lower() == "finf"]
        if row0.empty or rowi.empty:
            continue
        return {
            "A0": float(row0["A"].iloc[0]),
            "gamma0": float(row0["gamma"].iloc[0]),
            "Ainf": float(rowi["A"].iloc[0]),
            "gamma_inf": float(rowi["gamma"].iloc[0]),
            "source": str(path),
        }
    return None


def load_dataframe(args, outdir: Path):
    load_args = argparse.Namespace(
        rho=args.rho,
        vw_folders=args.vw_folders,
        h_values=args.h_values,
        tp_min=args.tp_min,
        tp_max=args.tp_max,
        bootstrap=0,
        n_jobs=1,
        reg_Finf=0.0,
        tc0=1.5,
        fix_tc=True,
        dpi=args.dpi,
        outdir=str(outdir),
    )
    df, f0_table, theta_values = base_fit.prepare_dataframe(load_args, outdir)
    analytic = load_analytic_h_defaults() if args.use_analytic_h else None
    if analytic is not None:
        hvals = h_theta(df["theta"].to_numpy(dtype=np.float64))
        good = np.isfinite(hvals) & (hvals > 0.0)
        bad = int(np.count_nonzero(~good))
        if bad:
            print(f"[warn] dropping {bad} rows with invalid h(theta) before analytic F0 use")
            df = df.loc[good].copy()
            hvals = hvals[good]
        df["F0"] = analytic["A0"] * np.power(hvals, analytic["gamma0"])
    else:
        analytic = {"source": None}
    df["h"] = h_theta(df["theta"].to_numpy(dtype=np.float64))
    df = df[
        np.isfinite(df["theta"])
        & np.isfinite(df["tp"])
        & np.isfinite(df["xi"])
        & np.isfinite(df["F0"])
        & np.isfinite(df["v_w"])
        & (df["tp"] > 0.0)
        & (df["xi"] > 0.0)
        & (df["F0"] > 0.0)
    ].copy()
    if df.empty:
        raise RuntimeError("No valid lattice rows remained after filtering.")
    df["fanh_obs"] = df["xi"] * df["F0"] / np.maximum(np.power(df["tp"] / args.t_osc, 1.5), 1.0e-18)
    theta_values = np.sort(df["theta"].unique())
    theta_index = {float(theta): i for i, theta in enumerate(theta_values)}
    df["theta_idx"] = [theta_index[float(theta)] for theta in df["theta"]]
    return df.sort_values(["v_w", "H", "theta", "tp"]).reset_index(drop=True), f0_table, theta_values, analytic


def load_optional_ode(path: Path | None, h_values):
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path, sep=r"\s+|,", engine="python", comment="#")
    cols = {str(c).lower(): c for c in df.columns}

    if {"theta", "tp", "xi", "h"}.issubset(cols):
        out = pd.DataFrame(
            {
                "theta": df[cols["theta"]].astype(float),
                "tp": df[cols["tp"]].astype(float),
                "xi": df[cols["xi"]].astype(float),
                "H": df[cols["h"]].astype(float),
                "v_w": df[cols.get("vw", cols.get("v_w"))].astype(float)
                if ("vw" in cols or "v_w" in cols)
                else 0.9,
            }
        )
    elif {"vw", "theta0", "h_star", "t_p", "xi_dm"}.issubset(cols):
        out = pd.DataFrame(
            {
                "v_w": df[cols["vw"]].astype(float),
                "theta": df[cols["theta0"]].astype(float),
                "H": df[cols["h_star"]].astype(float),
                "tp": df[cols["t_p"]].astype(float),
                "xi": df[cols["xi_dm"]].astype(float),
            }
        )
    else:
        raw = pd.read_csv(path, sep=r"\s+|,", engine="python", comment="#", header=None)
        if raw.shape[1] < 6:
            raise RuntimeError(f"Could not parse ODE file {path}.")
        raw = raw.iloc[:, :6].copy()
        raw.columns = ["v_w", "theta", "H", "beta_over_H", "tp", "xi"]
        out = raw.astype(float)

    out = out[
        np.isfinite(out["theta"])
        & np.isfinite(out["tp"])
        & np.isfinite(out["xi"])
        & np.isfinite(out["H"])
        & np.isfinite(out["v_w"])
    ].copy()
    out = out[(out["tp"] > 0.0) & (out["xi"] > 0.0) & out["H"].isin(h_values)].copy()
    return out.sort_values(["v_w", "H", "theta", "tp"]).reset_index(drop=True) if not out.empty else None


def choose_theta_subset(theta_values):
    out = []
    theta_values = np.asarray(theta_values, dtype=np.float64)
    for target in THETA_TARGETS:
        idx = int(np.argmin(np.abs(theta_values - target)))
        val = float(theta_values[idx])
        if val not in out:
            out.append(val)
    return np.asarray(out, dtype=np.float64)


def log_interp(x, y, xq):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if np.count_nonzero(mask) < 2:
        return np.full_like(xq, np.nan, dtype=np.float64)
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x, idx = np.unique(x, return_index=True)
    y = y[idx]
    if x.size < 2:
        return np.full_like(xq, np.nan, dtype=np.float64)
    f = interp1d(np.log(x), np.log(y), bounds_error=False, fill_value=np.nan, assume_sorted=True)
    out = np.exp(f(np.log(np.maximum(xq, 1.0e-300))))
    out[(xq < x[0]) | (xq > x[-1])] = np.nan
    return out


def linear_interp(x, y, xq):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return np.full_like(xq, np.nan, dtype=np.float64)
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x, idx = np.unique(x, return_index=True)
    y = y[idx]
    if x.size < 2:
        return np.full_like(xq, np.nan, dtype=np.float64)
    f = interp1d(x, y, bounds_error=False, fill_value=np.nan, assume_sorted=True)
    out = f(xq)
    out[(xq < x[0]) | (xq > x[-1])] = np.nan
    return out


def make_series_dict(df: pd.DataFrame):
    out = {}
    for key, sub in df.groupby(["v_w", "H", "theta"], sort=True):
        out[(float(key[0]), float(key[1]), float(key[2]))] = sub.sort_values("tp").copy()
    return out


def make_reference_dict(df: pd.DataFrame):
    out = {}
    for key, sub in df.groupby(["H", "theta"], sort=True):
        out[(float(key[0]), float(key[1]))] = sub.sort_values("tp").copy()
    return out


def pooled_rel_rmse(curves):
    arr = np.asarray(curves, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square(arr))))


def plot_overlay_by_theta(df, theta_subset, ode_df, outpath: Path, dpi: int):
    h_values = np.sort(df["H"].unique())
    vw_values = np.sort(df["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    linestyles = {1.5: "-", 2.0: "--"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
    axes = axes.ravel()
    for ax, theta in zip(axes, theta_subset):
        sub = df[np.isclose(df["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
        for h in h_values:
            for vw in vw_values:
                cur = sub[(np.isclose(sub["H"], float(h))) & (np.isclose(sub["v_w"], float(vw)))].sort_values("tp")
                if cur.empty:
                    continue
                ax.plot(
                    cur["tp"],
                    cur["xi"],
                    marker="o",
                    ms=3.0,
                    lw=1.4,
                    color=colors[vw],
                    linestyle=linestyles.get(float(h), "-"),
                    alpha=0.9,
                )
        if ode_df is not None and not ode_df.empty:
            od = ode_df[np.isclose(ode_df["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
            for h in h_values:
                for vw in vw_values:
                    cur = od[(np.isclose(od["H"], float(h))) & (np.isclose(od["v_w"], float(vw)))].sort_values("tp")
                    if cur.empty:
                        continue
                    ax.plot(
                        cur["tp"],
                        cur["xi"],
                        color=colors[vw],
                        linestyle=":",
                        lw=1.2,
                        alpha=0.8,
                    )
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$\xi$")

    for ax in axes[len(theta_subset) :]:
        ax.axis("off")

    legend_handles = []
    legend_labels = []
    for vw in vw_values:
        legend_handles.append(plt.Line2D([0], [0], color=colors[vw], lw=2.0))
        legend_labels.append(rf"$v_w={vw:.1f}$")
    for h in h_values:
        legend_handles.append(plt.Line2D([0], [0], color="black", lw=2.0, linestyle=linestyles.get(float(h), "-")))
        legend_labels.append(rf"$H_*={h:.1f}$")
    if ode_df is not None and not ode_df.empty:
        legend_handles.append(plt.Line2D([0], [0], color="black", lw=1.5, linestyle=":"))
        legend_labels.append("ODE")
    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=min(len(legend_labels), 8), frameon=False)
    fig.suptitle(r"Lattice $\xi(t_p)$ overlays across $v_w$ and $H_*$", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def compute_vw_variance(df, grid_n=120):
    h_values = np.sort(df["H"].unique())
    summary = {}
    for h in h_values:
        sub_h = df[np.isclose(df["H"], float(h), atol=1.0e-12, rtol=0.0)].copy()
        tp_lo = []
        tp_hi = []
        for _, sub in sub_h.groupby(["theta", "v_w"], sort=True):
            if len(sub) < 2:
                continue
            tp = sub["tp"].to_numpy(dtype=np.float64)
            tp_lo.append(np.min(tp))
            tp_hi.append(np.max(tp))
        if not tp_lo:
            continue
        lo = max(tp_lo)
        hi = min(tp_hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        grid = np.geomspace(lo, hi, grid_n)
        rows = []
        for theta, sub_theta in sub_h.groupby("theta", sort=True):
            curves = []
            for _, sub in sub_theta.groupby("v_w", sort=True):
                yi = log_interp(sub["tp"].to_numpy(dtype=np.float64), sub["xi"].to_numpy(dtype=np.float64), grid)
                curves.append(yi)
            arr = np.vstack(curves)
            valid = np.sum(np.isfinite(arr), axis=0) >= 2
            if not np.any(valid):
                continue
            mean = np.nanmean(arr[:, valid], axis=0)
            std = np.nanstd(arr[:, valid], axis=0, ddof=1)
            rel = std / np.maximum(np.abs(mean), 1.0e-18)
            full = np.full(grid.shape, np.nan, dtype=np.float64)
            full[valid] = rel
            rows.append(full)
        if rows:
            summary[float(h)] = {"tp": grid, "rel_std": np.nanmedian(np.vstack(rows), axis=0)}
    return summary


def plot_vw_variance(summary, outpath: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    for h, payload in sorted(summary.items()):
        ax.plot(payload["tp"], payload["rel_std"], lw=2.0, label=rf"$H_*={h:.1f}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.set_xlabel(r"$t_p$")
    ax.set_ylabel(r"$\mathrm{std}_{v_w}(\xi) / \langle \xi \rangle$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def build_alignment_pairs(df, ref_vw, theta_filter=None):
    pairs = {}
    for vw in np.sort(df["v_w"].unique()):
        if np.isclose(vw, ref_vw, atol=1.0e-12, rtol=0.0):
            continue
        pairs[float(vw)] = []
    group_cols = ["boot_group"] if "boot_group" in df.columns else ["H", "theta"]
    for _, sub in df.groupby(group_cols, sort=True):
        h = float(sub["H"].iloc[0])
        theta = float(sub["theta"].iloc[0])
        if theta_filter is not None and not np.isclose(float(theta), float(theta_filter), atol=5.0e-4, rtol=0.0):
            continue
        ref = sub[np.isclose(sub["v_w"], ref_vw, atol=1.0e-12, rtol=0.0)].sort_values("tp").copy()
        if len(ref) < 2:
            continue
        for vw, cur in sub.groupby("v_w", sort=True):
            if np.isclose(vw, ref_vw, atol=1.0e-12, rtol=0.0):
                continue
            cur = cur.sort_values("tp").copy()
            if len(cur) < 2:
                continue
            pairs[float(vw)].append(
                {
                    "H": float(h),
                    "theta": float(theta),
                    "tp": cur["tp"].to_numpy(dtype=np.float64),
                    "xi": cur["xi"].to_numpy(dtype=np.float64),
                    "ref_tp": ref["tp"].to_numpy(dtype=np.float64),
                    "ref_xi": ref["xi"].to_numpy(dtype=np.float64),
                }
            )
    return pairs


def shift_residuals(scale, pairs):
    residuals = []
    scale = float(scale)
    for pair in pairs:
        pred = log_interp(pair["ref_tp"], pair["ref_xi"], scale * pair["tp"])
        mask = np.isfinite(pred) & np.isfinite(pair["xi"]) & (pair["xi"] > 0.0)
        if np.count_nonzero(mask) < 3:
            continue
        residuals.append((pred[mask] - pair["xi"][mask]) / np.maximum(pair["xi"][mask], 1.0e-12))
    if not residuals:
        return np.array([], dtype=np.float64)
    return np.concatenate(residuals)


def optimize_scale_for_pairs(pairs, bounds=(0.2, 5.0)):
    if not pairs:
        return {"scale": np.nan, "rel_rmse": np.nan, "success": False, "n_points": 0}

    def objective(x):
        resid = shift_residuals(float(x[0]), pairs)
        if resid.size == 0:
            return 1.0e9
        return float(np.mean(np.square(resid)))

    res = minimize(
        objective,
        x0=np.array([1.0], dtype=np.float64),
        method="L-BFGS-B",
        bounds=[bounds],
    )
    scale = float(res.x[0])
    resid = shift_residuals(scale, pairs)
    return {
        "scale": scale,
        "rel_rmse": pooled_rel_rmse(resid),
        "success": bool(res.success),
        "n_points": int(resid.size),
        "message": str(res.message),
    }


def fit_global_eta(pair_map, ref_vw):
    vws = np.array(sorted(pair_map.keys()), dtype=np.float64)

    def objective(x):
        eta = float(x[0])
        residuals = []
        for vw in vws:
            scale = np.power(float(vw) / float(ref_vw), eta)
            resid = shift_residuals(scale, pair_map[float(vw)])
            if resid.size:
                residuals.append(resid)
        if not residuals:
            return 1.0e9
        return float(np.mean(np.square(np.concatenate(residuals))))

    res = minimize(objective, x0=np.array([0.0], dtype=np.float64), method="L-BFGS-B", bounds=[(-3.0, 3.0)])
    eta = float(res.x[0])
    scales = {f"{vw:.1f}": float(np.power(vw / ref_vw, eta)) for vw in vws}
    residuals = []
    for vw in vws:
        residuals.append(shift_residuals(scales[f"{vw:.1f}"], pair_map[float(vw)]))
    residuals = [r for r in residuals if r.size]
    resid = np.concatenate(residuals) if residuals else np.array([], dtype=np.float64)
    return {
        "eta": eta,
        "scales_from_eta": scales,
        "rel_rmse": pooled_rel_rmse(resid),
        "success": bool(res.success),
        "message": str(res.message),
    }


def compute_heatmap(df, ref_vw, scales, h_values, theta_values, grid_n=90):
    payload = {}
    pooled_before = []
    pooled_after = []
    for h in h_values:
        sub_h = df[np.isclose(df["H"], float(h), atol=1.0e-12, rtol=0.0)].copy()
        lo = []
        hi = []
        for _, sub in sub_h.groupby(["theta", "v_w"], sort=True):
            if len(sub) < 2:
                continue
            lo.append(np.min(sub["tp"]))
            hi.append(np.max(sub["tp"]))
        if not lo:
            continue
        tmin = max(lo)
        tmax = min(hi)
        if tmax <= tmin:
            continue
        grid = np.geomspace(tmin, tmax, grid_n)
        pre_rows = []
        post_rows = []
        for theta in theta_values:
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
            ref = sub[np.isclose(sub["v_w"], ref_vw, atol=1.0e-12, rtol=0.0)].sort_values("tp").copy()
            if len(ref) < 2:
                pre_rows.append(np.full(grid.shape, np.nan))
                post_rows.append(np.full(grid.shape, np.nan))
                continue
            pre_curves = []
            post_curves = []
            for vw, cur in sub.groupby("v_w", sort=True):
                if np.isclose(vw, ref_vw, atol=1.0e-12, rtol=0.0):
                    continue
                cur = cur.sort_values("tp").copy()
                yi = log_interp(cur["tp"].to_numpy(dtype=np.float64), cur["xi"].to_numpy(dtype=np.float64), grid)
                ref_pre = log_interp(ref["tp"].to_numpy(dtype=np.float64), ref["xi"].to_numpy(dtype=np.float64), grid)
                s = float(scales.get(f"{float(vw):.1f}", 1.0))
                ref_post = log_interp(ref["tp"].to_numpy(dtype=np.float64), ref["xi"].to_numpy(dtype=np.float64), s * grid)
                pre = (ref_pre - yi) / np.maximum(yi, 1.0e-12)
                post = (ref_post - yi) / np.maximum(yi, 1.0e-12)
                pre_curves.append(pre)
                post_curves.append(post)
                pooled_before.append(pre[np.isfinite(pre)])
                pooled_after.append(post[np.isfinite(post)])
            if pre_curves:
                pre_stack = np.vstack(pre_curves)
                pre_rows.append(np.nanmean(pre_stack, axis=0) if np.any(np.isfinite(pre_stack)) else np.full(grid.shape, np.nan))
            else:
                pre_rows.append(np.full(grid.shape, np.nan))
            if post_curves:
                post_stack = np.vstack(post_curves)
                post_rows.append(np.nanmean(post_stack, axis=0) if np.any(np.isfinite(post_stack)) else np.full(grid.shape, np.nan))
            else:
                post_rows.append(np.full(grid.shape, np.nan))
        payload[float(h)] = {
            "tp": grid,
            "theta": np.asarray(theta_values, dtype=np.float64),
            "pre": np.vstack(pre_rows),
            "post": np.vstack(post_rows),
        }
    return payload, pooled_rel_rmse(np.concatenate(pooled_before) if pooled_before else np.array([])), pooled_rel_rmse(np.concatenate(pooled_after) if pooled_after else np.array([]))


def plot_heatmap(payload, which, outpath: Path, dpi: int):
    h_values = sorted(payload.keys())
    fig, axes = plt.subplots(1, len(h_values), figsize=(6.2 * max(len(h_values), 1), 4.6), squeeze=False)
    axes = axes.ravel()
    for ax, h in zip(axes, h_values):
        item = payload[h]
        x = np.log10(item["tp"])
        y = item["theta"]
        z = item[which]
        im = ax.imshow(
            z,
            aspect="auto",
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            cmap="coolwarm",
            vmin=-0.15,
            vmax=0.15,
        )
        ax.set_title(rf"$H_*={h:.1f}$")
        ax.set_xlabel(r"$\log_{10} t_p$")
        ax.set_ylabel(r"$\theta$")
    for ax in axes[len(h_values) :]:
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes[: len(h_values)], shrink=0.9)
    cbar.set_label(r"$(\xi_{\rm ref} - \xi)/\xi$")
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def bootstrap_group_keys(df):
    keys = sorted({(float(h), float(theta)) for h, theta in zip(df["H"], df["theta"])})
    return keys


def resample_group_dataframe(df, rng, keys):
    samples = []
    key_to_rows = {key: sub.copy() for key, sub in df.groupby(["H", "theta"], sort=True)}
    chosen = [keys[idx] for idx in rng.integers(0, len(keys), size=len(keys))]
    for h, theta in chosen:
        sub = key_to_rows[(h, theta)].copy()
        sub["boot_group"] = f"{h:.6f}_{theta:.6f}_{len(samples)}"
        samples.append(sub)
    out = pd.concat(samples, ignore_index=True)
    return out


def bootstrap_alignment(df, ref_vw, nboot, n_jobs):
    keys = bootstrap_group_keys(df)

    def one(seed):
        rng = np.random.default_rng(seed)
        boot_df = resample_group_dataframe(df, rng, keys)
        pair_map = build_alignment_pairs(boot_df, ref_vw)
        svals = {}
        for vw, pairs in pair_map.items():
            fit = optimize_scale_for_pairs(pairs)
            svals[str(vw)] = fit["scale"]
        eta_fit = fit_global_eta(pair_map, ref_vw)
        return {"scales": svals, "eta": eta_fit["eta"]}

    seeds = np.arange(nboot, dtype=np.int64) + 20260318
    jobs = (delayed(one)(int(seed)) for seed in seeds)
    results = Parallel(n_jobs=n_jobs, prefer="threads")(jobs)
    scale_summary = {}
    for vw in sorted({float(k) for r in results for k in r["scales"].keys()}):
        arr = np.array([r["scales"].get(str(vw), np.nan) for r in results], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            scale_summary[f"{vw:.1f}"] = {
                "p16": float(np.percentile(arr, 16)),
                "p50": float(np.percentile(arr, 50)),
                "p84": float(np.percentile(arr, 84)),
            }
    eta_arr = np.array([r["eta"] for r in results], dtype=np.float64)
    eta_arr = eta_arr[np.isfinite(eta_arr)]
    return {
        "scale_bootstrap": scale_summary,
        "eta_bootstrap": {
            "p16": float(np.percentile(eta_arr, 16)),
            "p50": float(np.percentile(eta_arr, 50)),
            "p84": float(np.percentile(eta_arr, 84)),
        }
        if eta_arr.size
        else None,
    }


def plot_timewarp_vs_vw(vw_values, pooled_s, boot, eta_fit, ref_vw, outpath: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    xs = np.asarray(vw_values, dtype=np.float64)
    ys = np.array([pooled_s[f"{vw:.1f}"] for vw in xs], dtype=np.float64)
    yerr_lo = []
    yerr_hi = []
    for vw in xs:
        key = f"{vw:.1f}"
        if boot and key in boot["scale_bootstrap"]:
            yerr_lo.append(ys[len(yerr_lo)] - boot["scale_bootstrap"][key]["p16"])
            yerr_hi.append(boot["scale_bootstrap"][key]["p84"] - ys[len(yerr_hi)])
        else:
            yerr_lo.append(0.0)
            yerr_hi.append(0.0)
    ax.errorbar(xs, ys, yerr=np.vstack([yerr_lo, yerr_hi]), fmt="o", ms=5, lw=1.5, label="pooled $s(v_w)$")
    grid = np.linspace(float(np.min(xs)), float(np.max(xs)), 200)
    curve = np.power(grid / float(ref_vw), float(eta_fit["eta"]))
    ax.plot(grid, curve, lw=2.0, label=rf"$(v_w/{ref_vw:.1f})^\eta$, $\eta={eta_fit['eta']:.3f}$")
    ax.axhline(1.0, color="black", lw=1.0, alpha=0.4)
    ax.grid(alpha=0.25)
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$s(v_w)$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def select_reference_source(df, ode_df):
    if ode_df is not None and not ode_df.empty:
        print("[info] using ODE as reference source for convolution tests")
        return "ode", ode_df
    ref_vw = float(np.max(df["v_w"]))
    print(f"[info] ODE unavailable, using lattice v_w={ref_vw:.1f} as convolution reference")
    ref = df[np.isclose(df["v_w"], ref_vw, atol=1.0e-12, rtol=0.0)].copy()
    return "lattice_ref", ref


def build_ref_cache(ref_df, theta_values, include_vw=False):
    cache = {}
    theta_ref = np.sort(ref_df["theta"].unique())
    vw_values = np.sort(ref_df["v_w"].unique()) if include_vw else [None]
    for vw in vw_values:
        for h, theta in [(float(h), float(theta)) for h in np.sort(ref_df["H"].unique()) for theta in theta_values]:
            mask = np.isclose(ref_df["H"], h)
            if include_vw:
                mask &= np.isclose(ref_df["v_w"], float(vw), atol=1.0e-12, rtol=0.0)
            sub = ref_df[mask & np.isclose(ref_df["theta"], theta, atol=5.0e-4, rtol=0.0)].copy()
            if sub.empty and theta_ref.size:
                nearest = float(theta_ref[np.argmin(np.abs(theta_ref - theta))])
                sub = ref_df[mask & np.isclose(ref_df["theta"], nearest, atol=5.0e-4, rtol=0.0)].copy()
            if len(sub) < 2:
                continue
            sub = sub.sort_values("tp").copy()
            tmin = float(sub["tp"].min())
            tmax = float(sub["tp"].max())
            if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
                continue
            ngrid = max(512, min(2048, 8 * len(sub)))
            grid = np.linspace(tmin, tmax, ngrid)
            xi_grid = log_interp(sub["tp"].to_numpy(dtype=np.float64), sub["xi"].to_numpy(dtype=np.float64), grid)
            if np.count_nonzero(np.isfinite(xi_grid)) < 4:
                continue
            xi_grid = pd.Series(xi_grid).interpolate(limit_direction="both").to_numpy(dtype=np.float64)
            key = (float(vw), h, theta) if include_vw else (h, theta)
            cache[key] = {
                "grid": grid,
                "dt": float(grid[1] - grid[0]),
                "xi": xi_grid,
            }
    return cache


def kernel_vector(kind, param, dt, n):
    if param <= 1.0e-10:
        out = np.zeros(n, dtype=np.float64)
        out[0] = 1.0 / max(dt, 1.0e-18)
        return out
    tau = np.arange(n, dtype=np.float64) * dt
    if kind == "gaussian":
        kernel = np.exp(-0.5 * np.square(tau / param))
    elif kind == "exponential":
        kernel = np.exp(-tau / param)
    else:
        raise ValueError(f"Unsupported kernel {kind}")
    norm = np.sum(kernel) * dt
    if norm <= 0.0 or not np.isfinite(norm):
        out = np.zeros(n, dtype=np.float64)
        out[0] = 1.0 / max(dt, 1.0e-18)
        return out
    return kernel / norm


def convolve_reference(cache_entry, kind, param):
    grid = cache_entry["grid"]
    dt = cache_entry["dt"]
    signal = cache_entry["xi"]
    kernel = kernel_vector(kind, float(param), dt, len(grid))
    pad = len(kernel) - 1
    padded = np.pad(signal, (pad, 0), mode="edge")
    conv = fftconvolve(padded, kernel * dt, mode="full")
    pred = conv[pad : pad + len(signal)]
    return grid, pred


def convolution_residuals(df_target, ref_cache, kind, param, include_vw_ref=False):
    residuals = []
    predictions = {}
    group_cols = ["boot_group"] if "boot_group" in df_target.columns else ["H", "theta"]
    for _, sub in df_target.groupby(group_cols, sort=True):
        h = float(sub["H"].iloc[0])
        theta = float(sub["theta"].iloc[0])
        vw = float(sub["v_w"].iloc[0])
        key = (vw, h, theta) if include_vw_ref else (h, theta)
        if key not in ref_cache:
            continue
        grid, pred_grid = convolve_reference(ref_cache[key], kind, param)
        pred = linear_interp(grid, pred_grid, sub["tp"].to_numpy(dtype=np.float64))
        xi = sub["xi"].to_numpy(dtype=np.float64)
        mask = np.isfinite(pred) & np.isfinite(xi) & (xi > 0.0)
        if np.count_nonzero(mask) < 3:
            continue
        resid = (pred[mask] - xi[mask]) / np.maximum(xi[mask], 1.0e-12)
        residuals.append(resid)
        predictions[(h, theta)] = {
            "tp": sub["tp"].to_numpy(dtype=np.float64)[mask],
            "xi": xi[mask],
            "pred": pred[mask],
        }
    if not residuals:
        return np.array([], dtype=np.float64), predictions
    return np.concatenate(residuals), predictions


def fit_kernel_for_vw(df_vw, ref_cache, kind, include_vw_ref=False):
    def objective(x):
        resid, _ = convolution_residuals(df_vw, ref_cache, kind, float(x[0]), include_vw_ref=include_vw_ref)
        if resid.size == 0:
            return 1.0e9
        return float(np.mean(np.square(resid)))

    res = minimize(
        objective,
        x0=np.array([0.2], dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(0.0, 5.0)],
    )
    param = float(res.x[0])
    resid, predictions = convolution_residuals(df_vw, ref_cache, kind, param, include_vw_ref=include_vw_ref)
    return {
        "param": param,
        "rel_rmse": pooled_rel_rmse(resid),
        "AIC": aic_bic_from_resid(resid, 1)[0] if resid.size else np.nan,
        "BIC": aic_bic_from_resid(resid, 1)[1] if resid.size else np.nan,
        "success": bool(res.success),
        "message": str(res.message),
        "n_points": int(resid.size),
        "predictions": predictions,
    }


def aic_bic_from_resid(resid, k):
    resid = np.asarray(resid, dtype=np.float64)
    resid = resid[np.isfinite(resid)]
    n = max(int(resid.size), 1)
    rss = max(float(np.sum(np.square(resid))), 1.0e-18)
    ln_l = -0.5 * n * (math.log(2.0 * math.pi) + math.log(rss / n) + 1.0)
    aic = 2.0 * k - 2.0 * ln_l
    bic = math.log(n) * k - 2.0 * ln_l
    return float(aic), float(bic)


def bootstrap_kernel_params(df, ref_cache, kind, nboot, n_jobs, include_vw_ref=False):
    keys = bootstrap_group_keys(df)

    def one(seed):
        rng = np.random.default_rng(seed)
        boot = resample_group_dataframe(df, rng, keys)
        result = {}
        for vw, sub in boot.groupby("v_w", sort=True):
            fit = fit_kernel_for_vw(sub.copy(), ref_cache, kind, include_vw_ref=include_vw_ref)
            result[f"{float(vw):.1f}"] = fit["param"]
        return result

    seeds = np.arange(nboot, dtype=np.int64) + (3100 if kind == "gaussian" else 4200)
    results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(one)(int(seed)) for seed in seeds)
    summary = {}
    all_keys = sorted({k for item in results for k in item})
    for key in all_keys:
        arr = np.array([item.get(key, np.nan) for item in results], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            summary[key] = {
                "p16": float(np.percentile(arr, 16)),
                "p50": float(np.percentile(arr, 50)),
                "p84": float(np.percentile(arr, 84)),
            }
    return summary


def plot_convolution_fit(vw, df_vw, results_by_kind, outpath: Path, dpi: int):
    theta_subset = choose_theta_subset(np.sort(df_vw["theta"].unique()))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
    axes = axes.ravel()
    colors = {"gaussian": "tab:blue", "exponential": "tab:orange"}
    for ax, theta in zip(axes, theta_subset):
        sub = df_vw[np.isclose(df_vw["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
        for h, cur in sub.groupby("H", sort=True):
            cur = cur.sort_values("tp")
            ax.scatter(cur["tp"], cur["xi"], s=15, alpha=0.75, color="black")
            for kind, payload in results_by_kind.items():
                pred = payload["predictions"].get((float(h), float(theta)))
                if pred is None:
                    continue
                order = np.argsort(pred["tp"])
                ax.plot(
                    pred["tp"][order],
                    pred["pred"][order],
                    lw=1.8,
                    color=colors[kind],
                    alpha=0.9,
                    label=f"{kind}, H={h:g}" if ax is axes[0] else None,
                )
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$\xi$")
    for ax in axes[len(theta_subset) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
    fig.suptitle(rf"Parametric convolution fits for $v_w={vw:.1f}$", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def load_empirical_kernels(percolation_dir: Path | None, vw_values):
    if percolation_dir is None:
        return {}, "missing_percolation_dir"
    out = {}
    for vw in vw_values:
        token = f"v{int(round(vw * 10))}"
        path = percolation_dir / f"F_of_t_{token}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        t_col = cols.get("t")
        f_col = None
        for key in ("f", "f(t)", "fraction", "f_false", "f_false_vacuum"):
            if key in cols:
                f_col = cols[key]
                break
        if t_col is None or f_col is None:
            continue
        tt = df[t_col].to_numpy(dtype=np.float64)
        ff = df[f_col].to_numpy(dtype=np.float64)
        mask = np.isfinite(tt) & np.isfinite(ff)
        tt = tt[mask]
        ff = ff[mask]
        order = np.argsort(tt)
        tt = tt[order]
        ff = ff[order]
        if tt.size < 5:
            continue
        pdf = -np.gradient(ff, tt)
        pdf = np.clip(pdf, 0.0, None)
        norm = np.trapz(pdf, tt)
        if norm <= 0.0:
            continue
        out[f"{vw:.1f}"] = {"tau": tt - tt[0], "pdf": pdf / norm, "path": str(path)}
    return out, None if out else "no_matching_F_of_t_files"


def scaled_empirical_kernel(kernel, scale, dt, n):
    tau_base = np.asarray(kernel["tau"], dtype=np.float64)
    pdf_base = np.asarray(kernel["pdf"], dtype=np.float64)
    tau_grid = np.arange(n, dtype=np.float64) * dt
    scaled_tau = tau_base * float(scale)
    pdf = linear_interp(scaled_tau, pdf_base / max(float(scale), 1.0e-12), tau_grid)
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.sum(pdf) * dt
    if norm <= 0.0 or not np.isfinite(norm):
        out = np.zeros(n, dtype=np.float64)
        out[0] = 1.0 / max(dt, 1.0e-18)
        return out
    return pdf / norm


def empirical_residuals(df_vw, ref_cache, kernel, scale, include_vw_ref=False):
    residuals = []
    predictions = {}
    group_cols = ["boot_group"] if "boot_group" in df_vw.columns else ["H", "theta"]
    for _, sub in df_vw.groupby(group_cols, sort=True):
        h = float(sub["H"].iloc[0])
        theta = float(sub["theta"].iloc[0])
        vw = float(sub["v_w"].iloc[0])
        key = (vw, h, theta) if include_vw_ref else (h, theta)
        if key not in ref_cache:
            continue
        cache_entry = ref_cache[key]
        dt = cache_entry["dt"]
        signal = cache_entry["xi"]
        grid = cache_entry["grid"]
        kvals = scaled_empirical_kernel(kernel, float(scale), dt, len(grid))
        pad = len(kvals) - 1
        padded = np.pad(signal, (pad, 0), mode="edge")
        conv = fftconvolve(padded, kvals * dt, mode="full")
        pred_grid = conv[pad : pad + len(signal)]
        pred = linear_interp(grid, pred_grid, sub["tp"].to_numpy(dtype=np.float64))
        xi = sub["xi"].to_numpy(dtype=np.float64)
        mask = np.isfinite(pred) & np.isfinite(xi) & (xi > 0.0)
        if np.count_nonzero(mask) < 3:
            continue
        resid = (pred[mask] - xi[mask]) / np.maximum(xi[mask], 1.0e-12)
        residuals.append(resid)
        predictions[(h, theta)] = {"tp": sub["tp"].to_numpy(dtype=np.float64)[mask], "pred": pred[mask], "xi": xi[mask]}
    if not residuals:
        return np.array([], dtype=np.float64), predictions
    return np.concatenate(residuals), predictions


def fit_empirical_scale(df_vw, ref_cache, kernel, include_vw_ref=False):
    def objective(x):
        resid, _ = empirical_residuals(df_vw, ref_cache, kernel, float(x[0]), include_vw_ref=include_vw_ref)
        if resid.size == 0:
            return 1.0e9
        return float(np.mean(np.square(resid)))

    res = minimize(objective, x0=np.array([1.0], dtype=np.float64), method="L-BFGS-B", bounds=[(0.2, 5.0)])
    scale = float(res.x[0])
    resid, predictions = empirical_residuals(df_vw, ref_cache, kernel, scale, include_vw_ref=include_vw_ref)
    return {
        "scale": scale,
        "rel_rmse": pooled_rel_rmse(resid),
        "AIC": aic_bic_from_resid(resid, 1)[0] if resid.size else np.nan,
        "BIC": aic_bic_from_resid(resid, 1)[1] if resid.size else np.nan,
        "success": bool(res.success),
        "message": str(res.message),
        "n_points": int(resid.size),
        "predictions": predictions,
    }


def plot_empirical_fit(vw, df_vw, result, outpath: Path, dpi: int):
    theta_subset = choose_theta_subset(np.sort(df_vw["theta"].unique()))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
    axes = axes.ravel()
    for ax, theta in zip(axes, theta_subset):
        sub = df_vw[np.isclose(df_vw["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
        for h, cur in sub.groupby("H", sort=True):
            cur = cur.sort_values("tp")
            ax.scatter(cur["tp"], cur["xi"], s=15, color="black", alpha=0.8)
            pred = result["predictions"].get((float(h), float(theta)))
            if pred is None:
                continue
            order = np.argsort(pred["tp"])
            ax.plot(pred["tp"][order], pred["pred"][order], color="tab:green", lw=1.8)
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$\xi$")
    for ax in axes[len(theta_subset) :]:
        ax.axis("off")
    fig.suptitle(rf"Empirical percolation-kernel fit for $v_w={vw:.1f}$", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_fanh_by_theta(df, ref_vw, scales, t_osc, outpath: Path, dpi: int):
    theta_subset = choose_theta_subset(np.sort(df["theta"].unique()))
    vw_values = np.sort(df["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
    axes = axes.ravel()
    ref_map = make_series_dict(df[np.isclose(df["v_w"], ref_vw, atol=1.0e-12, rtol=0.0)])
    for ax, theta in zip(axes, theta_subset):
        sub = df[np.isclose(df["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
        for (vw, h), cur in sub.groupby(["v_w", "H"], sort=True):
            cur = cur.sort_values("tp")
            ax.scatter(cur["tp"], cur["fanh_obs"], s=12, alpha=0.35, color=colors[float(vw)])
            if np.isclose(vw, ref_vw, atol=1.0e-12, rtol=0.0):
                continue
            ref = ref_map.get((float(ref_vw), float(h), float(theta)))
            if ref is None:
                continue
            s = float(scales.get(f"{float(vw):.1f}", 1.0))
            xi_pred = log_interp(ref["tp"].to_numpy(dtype=np.float64), ref["xi"].to_numpy(dtype=np.float64), s * cur["tp"].to_numpy(dtype=np.float64))
            fanh_pred = xi_pred * cur["F0"].to_numpy(dtype=np.float64) / np.maximum(
                np.power(cur["tp"].to_numpy(dtype=np.float64) / float(t_osc), 1.5),
                1.0e-18,
            )
            mask = np.isfinite(fanh_pred) & (fanh_pred > 0.0)
            if np.count_nonzero(mask) < 2:
                continue
            ax.plot(cur["tp"].to_numpy(dtype=np.float64)[mask], fanh_pred[mask], color=colors[float(vw)], lw=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$f_{\rm anh}$")
    for ax in axes[len(theta_subset) :]:
        ax.axis("off")
    handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
    labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
    fig.suptitle(r"Observed $f_{\rm anh}$ and warped-reference overlays", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_collapse_overlay(df, scales, outpath: Path, dpi: int):
    theta_subset = choose_theta_subset(np.sort(df["theta"].unique()))
    vw_values = np.sort(df["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    linestyles = {1.5: "-", 2.0: "--"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
    axes = axes.ravel()
    for ax, theta in zip(axes, theta_subset):
        sub = df[np.isclose(df["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
        for (vw, h), cur in sub.groupby(["v_w", "H"], sort=True):
            tp_scaled = cur["tp"].to_numpy(dtype=np.float64) * float(scales.get(f"{float(vw):.1f}", 1.0))
            ax.plot(tp_scaled, cur["xi"], marker="o", ms=3.0, lw=1.2, color=colors[float(vw)], linestyle=linestyles.get(float(h), "-"))
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$t_p \, s(v_w)$")
        ax.set_ylabel(r"$\xi$")
    for ax in axes[len(theta_subset) :]:
        ax.axis("off")
    handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
    labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
    fig.suptitle(r"Collapse overlay after per-$v_w$ time warp", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_amplitude_after_warp(df, ref_vw, scales, outpath: Path, dpi: int):
    rows = []
    pair_map = build_alignment_pairs(df, ref_vw)
    for vw, pairs in pair_map.items():
        s = float(scales.get(f"{float(vw):.1f}", 1.0))
        ratios = []
        for pair in pairs:
            pred = log_interp(pair["ref_tp"], pair["ref_xi"], s * pair["tp"])
            mask = np.isfinite(pred) & np.isfinite(pair["xi"]) & (pred > 0.0)
            if np.count_nonzero(mask) < 3:
                continue
            ratios.append(pair["xi"][mask] / pred[mask])
        if ratios:
            arr = np.concatenate(ratios)
            rows.append({"v_w": float(vw), "ratio_median": float(np.median(arr)), "ratio_p16": float(np.percentile(arr, 16)), "ratio_p84": float(np.percentile(arr, 84))})
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    if rows:
        temp = pd.DataFrame(rows).sort_values("v_w")
        y = temp["ratio_median"].to_numpy(dtype=np.float64)
        yerr = np.vstack([y - temp["ratio_p16"].to_numpy(dtype=np.float64), temp["ratio_p84"].to_numpy(dtype=np.float64) - y])
        ax.errorbar(temp["v_w"], y, yerr=yerr, fmt="o", ms=5, lw=1.5)
    ax.axhline(1.0, color="black", lw=1.0, alpha=0.4)
    ax.grid(alpha=0.25)
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"median $\xi_{\rm obs}/\xi_{\rm warped}$")
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return rows


def compute_slopes_and_curvature(df, scales, outdir: Path, dpi: int):
    theta_subset = choose_theta_subset(np.sort(df["theta"].unique()))
    summaries = {}
    for theta in theta_subset:
        sub = df[np.isclose(df["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        for (vw, h), cur in sub.groupby(["v_w", "H"], sort=True):
            cur = cur.sort_values("tp").copy()
            x = cur["tp"].to_numpy(dtype=np.float64) * float(scales.get(f"{float(vw):.1f}", 1.0))
            y = cur["xi"].to_numpy(dtype=np.float64)
            mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
            if np.count_nonzero(mask) < 7:
                continue
            logx = np.log(x[mask])
            logy = np.log(y[mask])
            win = min(len(logx) if len(logx) % 2 == 1 else len(logx) - 1, 7)
            if win < 5:
                continue
            smooth = savgol_filter(logy, window_length=win, polyorder=2, mode="interp")
            d1 = np.gradient(smooth, logx)
            d2 = np.gradient(d1, logx)
            axes[0].plot(x[mask], d1, lw=1.4, label=rf"$v_w={vw:.1f},H={h:.1f}$")
            axes[1].plot(x[mask], d2, lw=1.4, label=rf"$v_w={vw:.1f},H={h:.1f}$")
            idx = int(np.nanargmax(np.abs(d2)))
            summaries.setdefault(f"{float(theta):.10f}", []).append({"v_w": float(vw), "H": float(h), "tp_curv_max": float(x[mask][idx]), "curvature_max": float(d2[idx])})
        axes[0].set_xscale("log")
        axes[1].set_xscale("log")
        axes[0].set_xlabel(r"$t_p s(v_w)$")
        axes[1].set_xlabel(r"$t_p s(v_w)$")
        axes[0].set_ylabel(r"$d\ln\xi / d\ln t$")
        axes[1].set_ylabel(r"$d^2\ln\xi / d(\ln t)^2$")
        axes[0].grid(alpha=0.25)
        axes[1].grid(alpha=0.25)
        axes[0].set_title(rf"$\theta={theta:.3f}$ slope")
        axes[1].set_title(rf"$\theta={theta:.3f}$ curvature")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        tag = f"{float(theta):.10f}".replace(".", "p")
        fig.savefig(outdir / f"slope_curvature_theta_{tag}.png", dpi=dpi)
        plt.close(fig)
    return summaries


def load_baseline_metrics():
    baselines = {}
    for path in AMPLITUDE_RESULT_CANDIDATES:
        if path.exists():
            data = json.loads(path.read_text())
            baselines["amplitude_only"] = {
                "rel_rmse": float(data.get("rel_rmse", np.nan)),
                "AIC": float(data.get("AIC", np.nan)),
                "BIC": float(data.get("BIC", np.nan)),
                "source": str(path),
            }
            break
    for path in TIMEWARP_RESULT_CANDIDATES:
        if path.exists():
            data = json.loads(path.read_text())
            baselines["warp_only_previous"] = {
                "rel_rmse": float(data.get("rel_rmse", np.nan)),
                "AIC": float(data.get("AIC", np.nan)),
                "BIC": float(data.get("BIC", np.nan)),
                "source": str(path),
            }
            break
    return baselines


def main():
    args = parse_args()
    outdir = (ROOT / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/8] loading lattice data")
    df, f0_table, theta_values, analytic = load_dataframe(args, outdir)
    ode_path = resolve_ode_file(args.ode_file)
    ode_df = load_optional_ode(ode_path, args.h_values)
    f0_table.to_csv(outdir / "f0_table.csv", index=False)

    theta_subset = choose_theta_subset(theta_values)
    ref_vw = float(np.max(df["v_w"]))
    vw_values = np.sort(df["v_w"].unique())
    h_values = np.sort(df["H"].unique())

    print("[2/8] plotting direct overlays")
    plot_overlay_by_theta(df, theta_subset, ode_df, outdir / "overlay_by_theta.png", args.dpi)
    variance_summary = compute_vw_variance(df)
    plot_vw_variance(variance_summary, outdir / "vw_variance_vs_tp.png", args.dpi)

    print("[3/8] fitting horizontal alignment")
    pooled_pairs = build_alignment_pairs(df, ref_vw)
    pooled_s = {f"{ref_vw:.1f}": 1.0}
    per_theta_s = {}
    for vw, pairs in pooled_pairs.items():
        fit = optimize_scale_for_pairs(pairs)
        pooled_s[f"{vw:.1f}"] = fit["scale"]
    for theta in theta_subset:
        theta_pairs = build_alignment_pairs(df, ref_vw, theta_filter=float(theta))
        theta_key = f"{float(theta):.10f}"
        per_theta_s[theta_key] = {}
        for vw, pairs in theta_pairs.items():
            fit = optimize_scale_for_pairs(pairs)
            per_theta_s[theta_key][f"{vw:.1f}"] = fit["scale"]
        per_theta_s[theta_key][f"{ref_vw:.1f}"] = 1.0
    eta_fit = fit_global_eta(pooled_pairs, ref_vw)
    boot_align = bootstrap_alignment(df, ref_vw, args.nboot, args.n_jobs)
    timewarp_payload = {
        "reference_vw": ref_vw,
        "pooled_scales": pooled_s,
        "per_theta_scales": per_theta_s,
        "eta_fit": eta_fit,
        "bootstrap": boot_align,
    }
    save_json(outdir / "timewarp_s_values.json", timewarp_payload)
    plot_timewarp_vs_vw(vw_values, pooled_s, boot_align, eta_fit, ref_vw, outdir / "timewarp_vs_vw.png", args.dpi)

    print("[4/8] building residual heatmaps")
    heatmap_payload, rmse_pre, rmse_post = compute_heatmap(df, ref_vw, pooled_s, h_values, theta_values)
    plot_heatmap(heatmap_payload, "pre", outdir / "residual_heatmap_prewarp.png", args.dpi)
    plot_heatmap(heatmap_payload, "post", outdir / "residual_heatmap_postwarp.png", args.dpi)
    plot_fanh_by_theta(df, ref_vw, pooled_s, args.t_osc, outdir / "fanh_vs_tp_by_theta_timewarp.png", args.dpi)
    plot_collapse_overlay(df, pooled_s, outdir / "collapse_overlay_timewarp.png", args.dpi)
    amp_rows = plot_amplitude_after_warp(df, ref_vw, pooled_s, outdir / "amplitude_after_warp.png", args.dpi)
    slope_summary = compute_slopes_and_curvature(df, pooled_s, outdir, args.dpi)

    print("[5/8] fitting parametric convolutions")
    ref_source, ref_df = select_reference_source(df, ode_df)
    include_vw_ref = ref_source == "ode"
    ref_cache = build_ref_cache(ref_df, theta_values, include_vw=include_vw_ref)
    if not ref_cache:
        raise RuntimeError("Could not build any reference curves for convolution tests.")
    conv_results = {
        "reference_source": ref_source,
        "gaussian": {"per_vw": {}},
        "exponential": {"per_vw": {}},
    }
    overall_resid = {"gaussian": [], "exponential": []}
    for kind in ("gaussian", "exponential"):
        for vw in tqdm(vw_values, desc=f"[conv {kind}]", leave=False):
            sub = df[np.isclose(df["v_w"], float(vw), atol=1.0e-12, rtol=0.0)].copy()
            fit = fit_kernel_for_vw(sub, ref_cache, kind, include_vw_ref=include_vw_ref)
            conv_results[kind]["per_vw"][f"{float(vw):.1f}"] = {
                k: v for k, v in fit.items() if k != "predictions"
            }
            resid, _ = convolution_residuals(sub, ref_cache, kind, fit["param"], include_vw_ref=include_vw_ref)
            if resid.size:
                overall_resid[kind].append(resid)
            if kind == "exponential":
                other_kind_fit = fit_kernel_for_vw(sub, ref_cache, "gaussian", include_vw_ref=include_vw_ref)
                plot_convolution_fit(
                    float(vw),
                    sub,
                    {
                        "gaussian": {"predictions": other_kind_fit["predictions"]},
                        "exponential": {"predictions": fit["predictions"]},
                    },
                    outdir / f"convolution_fit_vw{int(round(vw * 10)) / 10:.1f}.png",
                    args.dpi,
                )
        boot = bootstrap_kernel_params(df, ref_cache, kind, args.nboot, args.n_jobs, include_vw_ref=include_vw_ref)
        conv_results[kind]["bootstrap"] = boot
        resid = np.concatenate(overall_resid[kind]) if overall_resid[kind] else np.array([], dtype=np.float64)
        conv_results[kind]["overall"] = {
            "rel_rmse": pooled_rel_rmse(resid),
            "AIC": aic_bic_from_resid(resid, len(conv_results[kind]["per_vw"]))[0] if resid.size else np.nan,
            "BIC": aic_bic_from_resid(resid, len(conv_results[kind]["per_vw"]))[1] if resid.size else np.nan,
            "n_points": int(resid.size),
        }
    save_json(outdir / "convolution_parametric_results.json", conv_results)

    print("[6/8] empirical percolation-kernel test")
    percolation_dir = resolve_percolation_dir(args.percolation_dir)
    kernels, kernel_status = load_empirical_kernels(percolation_dir, vw_values)
    empirical = {
        "status": "skipped" if not kernels else "ok",
        "skip_reason": kernel_status,
        "reference_source": ref_source,
        "per_vw": {},
    }
    empirical_resids = []
    if kernels:
        for vw in vw_values:
            key = f"{float(vw):.1f}"
            if key not in kernels:
                empirical["per_vw"][key] = {"status": "missing_kernel"}
                continue
            sub = df[np.isclose(df["v_w"], float(vw), atol=1.0e-12, rtol=0.0)].copy()
            fit = fit_empirical_scale(sub, ref_cache, kernels[key], include_vw_ref=include_vw_ref)
            empirical["per_vw"][key] = {
                "kernel_source": kernels[key]["path"],
                "scale": fit["scale"],
                "rel_rmse": fit["rel_rmse"],
                "AIC": fit["AIC"],
                "BIC": fit["BIC"],
                "success": fit["success"],
                "message": fit["message"],
                "n_points": fit["n_points"],
            }
            resid, _ = empirical_residuals(sub, ref_cache, kernels[key], fit["scale"], include_vw_ref=include_vw_ref)
            if resid.size:
                empirical_resids.append(resid)
            plot_empirical_fit(float(vw), sub, fit, outdir / f"convolution_empirical_vw{int(round(vw * 10)) / 10:.1f}.png", args.dpi)
        resid = np.concatenate(empirical_resids) if empirical_resids else np.array([], dtype=np.float64)
        empirical["overall"] = {
            "rel_rmse": pooled_rel_rmse(resid),
            "AIC": aic_bic_from_resid(resid, len(empirical["per_vw"]))[0] if resid.size else np.nan,
            "BIC": aic_bic_from_resid(resid, len(empirical["per_vw"]))[1] if resid.size else np.nan,
            "n_points": int(resid.size),
        }
    save_json(outdir / "convolution_empirical_results.json", empirical)

    print("[7/8] loading baseline metrics")
    baselines = load_baseline_metrics()
    comparison = {
        "baseline_amplitude": baselines.get("amplitude_only"),
        "baseline_warp_previous": baselines.get("warp_only_previous"),
        "alignment_rmse_pre": rmse_pre,
        "alignment_rmse_post": rmse_post,
        "gaussian": conv_results["gaussian"]["overall"],
        "exponential": conv_results["exponential"]["overall"],
        "empirical": empirical.get("overall"),
    }
    if baselines.get("amplitude_only"):
        for name in ("gaussian", "exponential"):
            comparison[name]["delta_AIC_vs_amplitude"] = float(comparison[name]["AIC"] - baselines["amplitude_only"]["AIC"]) if np.isfinite(comparison[name]["AIC"]) and np.isfinite(baselines["amplitude_only"]["AIC"]) else np.nan
            comparison[name]["delta_BIC_vs_amplitude"] = float(comparison[name]["BIC"] - baselines["amplitude_only"]["BIC"]) if np.isfinite(comparison[name]["BIC"]) and np.isfinite(baselines["amplitude_only"]["BIC"]) else np.nan
    if baselines.get("warp_only_previous"):
        for name in ("gaussian", "exponential"):
            comparison[name]["delta_AIC_vs_warp"] = float(comparison[name]["AIC"] - baselines["warp_only_previous"]["AIC"]) if np.isfinite(comparison[name]["AIC"]) and np.isfinite(baselines["warp_only_previous"]["AIC"]) else np.nan
            comparison[name]["delta_BIC_vs_warp"] = float(comparison[name]["BIC"] - baselines["warp_only_previous"]["BIC"]) if np.isfinite(comparison[name]["BIC"]) and np.isfinite(baselines["warp_only_previous"]["BIC"]) else np.nan
    save_json(outdir / "summary.json", comparison)

    print("[8/8] writing final summary")
    best_parametric = min(
        [
            ("gaussian", conv_results["gaussian"]["overall"]["rel_rmse"]),
            ("exponential", conv_results["exponential"]["overall"]["rel_rmse"]),
        ],
        key=lambda item: np.inf if not np.isfinite(item[1]) else item[1],
    )[0]
    final_summary = {
        "status": "ok",
        "data_rows": int(len(df)),
        "theta_values": [float(v) for v in theta_values],
        "theta_subset": [float(v) for v in theta_subset],
        "analytic_h_source": analytic.get("source"),
        "ode_source": str(ode_path) if ode_path else None,
        "percolation_dir": str(percolation_dir) if percolation_dir else None,
        "reference_vw": ref_vw,
        "pooled_scales": pooled_s,
        "eta_fit": eta_fit,
        "rmse_prewarp": rmse_pre,
        "rmse_postwarp": rmse_post,
        "amplitude_after_warp": amp_rows,
        "best_parametric_kernel": best_parametric,
        "parametric_overall": {
            "gaussian": conv_results["gaussian"]["overall"],
            "exponential": conv_results["exponential"]["overall"],
        },
        "empirical_status": empirical["status"],
        "empirical_overall": empirical.get("overall"),
        "baselines": baselines,
        "curvature_summary": slope_summary,
    }
    save_json(outdir / "final_summary.json", final_summary)
    print(json.dumps(to_native(final_summary), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error_exit((ROOT / OUTDIR_DEFAULT).resolve(), exc)
        sys.exit(1)
