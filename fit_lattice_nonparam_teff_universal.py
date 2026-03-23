#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import collapse_and_fit_fanh_tosc as collapse
import fit_vw_amplitude as base_fit
import fit_vw_smearing_tests as smear
import fit_lattice_quadwarp_universal as uq


OUTDIR = ROOT / "results_nonparam_teff_universal_beta0"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit monotone nonparametric t_eff(tp) maps shared over theta for each (v_w,H), then fit one universal lattice curve."
    )
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--reference-vw", type=float, default=0.9)
    p.add_argument("--n-knots", type=int, default=5)
    p.add_argument("--smooth-lambda", type=float, default=1.0e-3)
    p.add_argument("--bootstrap", type=int, default=0)
    p.add_argument("--bootstrap-jobs", type=int, default=1)
    p.add_argument("--bootstrap-seed", type=int, default=12345)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--outdir", type=str, default=str(OUTDIR))
    return p.parse_args()


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


def load_lattice_dataframe(outdir: Path, vw_tags, h_values):
    args = SimpleNamespace(
        rho="",
        vw_folders=vw_tags,
        h_values=h_values,
        tp_min=None,
        tp_max=None,
        bootstrap=0,
        n_jobs=1,
        reg_Finf=0.0,
        tc0=1.5,
        fix_tc=True,
        dpi=220,
        outdir=str(outdir),
    )
    outdir.mkdir(parents=True, exist_ok=True)
    df, _, _ = base_fit.prepare_dataframe(args, outdir)
    return df.sort_values(["v_w", "H", "theta", "tp"]).reset_index(drop=True)


def softplus(x):
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def inv_softplus(y):
    y = np.asarray(y, dtype=np.float64)
    y = np.maximum(y, 1.0e-12)
    return y + np.log(-np.expm1(-y))


def build_pairs_by_map(df: pd.DataFrame, ref_vw: float):
    pairs = {}
    for (h, theta), sub in df.groupby(["H", "theta"], sort=True):
        ref = sub[np.isclose(sub["v_w"], ref_vw, atol=1.0e-12, rtol=0.0)].sort_values("tp").copy()
        if len(ref) < 2:
            continue
        for vw, cur in sub.groupby("v_w", sort=True):
            if np.isclose(float(vw), float(ref_vw), atol=1.0e-12, rtol=0.0):
                continue
            cur = cur.sort_values("tp").copy()
            if len(cur) < 2:
                continue
            key = (float(vw), float(h))
            pairs.setdefault(key, []).append(
                {
                    "theta": float(theta),
                    "tp": cur["tp"].to_numpy(dtype=np.float64),
                    "xi": cur["xi"].to_numpy(dtype=np.float64),
                    "ref_tp": ref["tp"].to_numpy(dtype=np.float64),
                    "ref_xi": ref["xi"].to_numpy(dtype=np.float64),
                }
            )
    return pairs


def make_knots(df: pd.DataFrame, h_value: float, n_knots: int):
    sub = df[np.isclose(df["H"], float(h_value), atol=1.0e-12, rtol=0.0)].copy()
    lmin = float(np.log(np.min(sub["tp"])))
    lmax = float(np.log(np.max(sub["tp"])))
    return np.linspace(lmin, lmax, int(n_knots), dtype=np.float64)


def params_to_y(knots_x: np.ndarray, params: np.ndarray, min_slope: float = 0.05):
    params = np.asarray(params, dtype=np.float64)
    y = np.zeros_like(knots_x, dtype=np.float64)
    y[0] = float(params[0])
    dx = np.diff(knots_x)
    increments = min_slope * dx + softplus(params[1:])
    y[1:] = y[0] + np.cumsum(increments)
    return y


def identity_params(knots_x: np.ndarray, min_slope: float = 0.05):
    dx = np.diff(knots_x)
    inc_target = np.maximum(dx * (1.0 - min_slope), 1.0e-6)
    return np.concatenate([np.array([knots_x[0]], dtype=np.float64), inv_softplus(inc_target)])


def piecewise_linear_extrap(xq: np.ndarray, knots_x: np.ndarray, knots_y: np.ndarray):
    xq = np.asarray(xq, dtype=np.float64)
    out = np.interp(xq, knots_x, knots_y)
    left_slope = (knots_y[1] - knots_y[0]) / max(knots_x[1] - knots_x[0], 1.0e-12)
    right_slope = (knots_y[-1] - knots_y[-2]) / max(knots_x[-1] - knots_x[-2], 1.0e-12)
    mask_left = xq < knots_x[0]
    mask_right = xq > knots_x[-1]
    if np.any(mask_left):
        out[mask_left] = knots_y[0] + left_slope * (xq[mask_left] - knots_x[0])
    if np.any(mask_right):
        out[mask_right] = knots_y[-1] + right_slope * (xq[mask_right] - knots_x[-1])
    return out


def map_tp(tp: np.ndarray, knots_x: np.ndarray, knots_y: np.ndarray):
    ltp = np.log(np.maximum(np.asarray(tp, dtype=np.float64), 1.0e-18))
    return np.exp(piecewise_linear_extrap(ltp, knots_x, knots_y))


def fit_one_map(pair_list, knots_x: np.ndarray, smooth_lambda: float):
    x0 = identity_params(knots_x)

    def objective(params):
        knots_y = params_to_y(knots_x, params)
        residuals = []
        for pair in pair_list:
            teff = map_tp(pair["tp"], knots_x, knots_y)
            pred = smear.log_interp(pair["ref_tp"], pair["ref_xi"], teff)
            mask = np.isfinite(pred) & np.isfinite(pair["xi"]) & (pair["xi"] > 0.0)
            if np.count_nonzero(mask) >= 3:
                residuals.append((pred[mask] - pair["xi"][mask]) / np.maximum(pair["xi"][mask], 1.0e-12))
        if not residuals:
            return 1.0e9
        resid = np.concatenate(residuals)
        rough = np.sum(np.square(np.diff(knots_y - knots_x, n=2))) if len(knots_x) >= 3 else 0.0
        return float(np.mean(np.square(resid)) + float(smooth_lambda) * rough)

    res = minimize(objective, x0=x0, method="L-BFGS-B")
    knots_y = params_to_y(knots_x, res.x)
    residuals = []
    for pair in pair_list:
        teff = map_tp(pair["tp"], knots_x, knots_y)
        pred = smear.log_interp(pair["ref_tp"], pair["ref_xi"], teff)
        mask = np.isfinite(pred) & np.isfinite(pair["xi"]) & (pair["xi"] > 0.0)
        if np.count_nonzero(mask) >= 3:
            residuals.append((pred[mask] - pair["xi"][mask]) / np.maximum(pair["xi"][mask], 1.0e-12))
    resid = np.concatenate(residuals) if residuals else np.array([], dtype=np.float64)
    slopes = np.diff(knots_y) / np.maximum(np.diff(knots_x), 1.0e-12)
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "params": np.asarray(res.x, dtype=np.float64),
        "knots_x": np.asarray(knots_x, dtype=np.float64),
        "knots_y": np.asarray(knots_y, dtype=np.float64),
        "rel_rmse": float(smear.pooled_rel_rmse(resid)),
        "n_points": int(resid.size),
        "min_slope": float(np.min(slopes)) if len(slopes) else np.nan,
    }


def fit_all_maps(df: pd.DataFrame, ref_vw: float, n_knots: int, smooth_lambda: float):
    pairs = build_pairs_by_map(df, ref_vw)
    results = {}
    for (vw, h), pair_list in sorted(pairs.items()):
        knots_x = make_knots(df, h, n_knots)
        results[(vw, h)] = fit_one_map(pair_list, knots_x, smooth_lambda)
    return results


def apply_all_maps(df: pd.DataFrame, ref_vw: float, map_results: dict, beta: float):
    out = df.copy()
    t_eff = np.zeros(len(out), dtype=np.float64)
    x = np.zeros(len(out), dtype=np.float64)
    for i, row in enumerate(out.itertuples(index=False)):
        if np.isclose(float(row.v_w), float(ref_vw), atol=1.0e-12, rtol=0.0):
            teff = float(row.tp)
        else:
            rec = map_results[(float(row.v_w), float(row.H))]
            teff = float(map_tp(np.array([float(row.tp)], dtype=np.float64), rec["knots_x"], rec["knots_y"])[0])
        t_eff[i] = teff
        x[i] = teff * np.power(float(row.H), float(beta))
    out["t_eff"] = t_eff
    out["x"] = x
    return out[np.isfinite(out["x"]) & (out["x"] > 0.0)].copy()


def plot_maps(df: pd.DataFrame, ref_vw: float, map_results: dict, outdir: Path, dpi: int):
    h_values = np.sort(df["H"].unique())
    vw_values = [vw for vw in np.sort(df["v_w"].unique()) if not np.isclose(vw, ref_vw, atol=1.0e-12, rtol=0.0)]
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}

    fig, axes = plt.subplots(len(h_values), 2, figsize=(10.5, 3.6 * len(h_values)), squeeze=False)
    for row_idx, h in enumerate(h_values):
        ax0, ax1 = axes[row_idx]
        sub = df[np.isclose(df["H"], float(h), atol=1.0e-12, rtol=0.0)].copy()
        tp_grid = np.geomspace(float(np.min(sub["tp"])), float(np.max(sub["tp"])), 300)
        ax0.plot(tp_grid, tp_grid, color="black", ls="--", lw=1.2)
        ax1.axhline(1.0, color="black", ls="--", lw=1.2)
        for vw in vw_values:
            rec = map_results[(float(vw), float(h))]
            teff = map_tp(tp_grid, rec["knots_x"], rec["knots_y"])
            ax0.plot(tp_grid, teff, color=colors[float(vw)], lw=2.0, label=rf"$v_w={vw:.1f}$")
            ax0.plot(np.exp(rec["knots_x"]), np.exp(rec["knots_y"]), "o", color=colors[float(vw)], ms=3.2)
            ax1.plot(tp_grid, teff / tp_grid, color=colors[float(vw)], lw=2.0)
        ax0.set_xscale("log")
        ax0.set_yscale("log")
        ax0.set_xlabel(r"$t_p$")
        ax0.set_ylabel(r"$t_{\rm eff}$")
        ax0.set_title(rf"Monotone map, $H_*={h:.1f}$")
        ax0.grid(alpha=0.25)
        ax1.set_xscale("log")
        ax1.set_xlabel(r"$t_p$")
        ax1.set_ylabel(r"$t_{\rm eff}/t_p$")
        ax1.set_title(rf"Relative map, $H_*={h:.1f}$")
        ax1.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "nonparam_teff_maps.png", dpi=dpi)
    plt.close(fig)


def plot_residual_by_theta(warped_df: pd.DataFrame, fit_result: dict, outdir: Path, dpi: int):
    xi_fit, _ = collapse.xi_model_from_params(warped_df, fit_result["theta_values"], fit_result["params"])
    work = warped_df.copy()
    work["rel_resid"] = (xi_fit - work["xi"].to_numpy(dtype=np.float64)) / np.maximum(work["xi"].to_numpy(dtype=np.float64), 1.0e-12)
    summary = (
        work.groupby("theta", as_index=False)["rel_resid"]
        .agg(mean_abs=lambda s: float(np.mean(np.abs(s))), mean=lambda s: float(np.mean(s)), std=lambda s: float(np.std(s, ddof=1) if len(s) > 1 else 0.0))
        .sort_values("theta")
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.errorbar(summary["theta"], summary["mean"], yerr=summary["std"], fmt="o-", capsize=3)
    ax.axhline(0.0, color="black", ls="--", lw=1.0)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"mean residual $(\xi_{\rm fit}-\xi)/\xi$")
    ax.set_title("Residuals vs theta after nonparametric $t_{eff}$")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "residual_vs_theta.png", dpi=dpi)
    plt.close(fig)
    summary.to_csv(outdir / "residual_by_theta.csv", index=False)
    return summary


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("[load] reading lattice dataframe")
    df = load_lattice_dataframe(outdir, args.vw_folders, args.h_values)

    print("[warp] fitting monotone nonparametric maps per (v_w,H)")
    map_results = fit_all_maps(df, float(args.reference_vw), int(args.n_knots), float(args.smooth_lambda))
    map_payload = {
        f"vw{vw:.1f}_H{h:.1f}": {
            "success": bool(rec["success"]),
            "message": rec["message"],
            "knots_tp": [float(v) for v in np.exp(rec["knots_x"])],
            "knots_teff": [float(v) for v in np.exp(rec["knots_y"])],
            "rel_rmse": float(rec["rel_rmse"]),
            "n_points": int(rec["n_points"]),
            "min_slope": float(rec["min_slope"]),
        }
        for (vw, h), rec in map_results.items()
    }
    save_json(outdir / "nonparam_teff_maps.json", map_payload)

    print("[warp] applying monotone maps")
    warped_df = apply_all_maps(df, float(args.reference_vw), map_results, float(args.beta))

    print("[fit] estimating tail amplitudes")
    finf_tail_df = collapse.fit_tail(warped_df, outdir, args.dpi)
    print("[fit] fitting universal curve")
    fit_result = collapse.fit_global(warped_df, finf_tail_df)
    global_payload = collapse.save_global_fit(fit_result, float(args.beta), outdir)

    bootstrap_payload = None
    if int(args.bootstrap) > 0:
        bootstrap_payload = collapse.bootstrap_global_fit(
            warped_df,
            fit_result,
            bootstrap_n=int(args.bootstrap),
            bootstrap_jobs=int(args.bootstrap_jobs),
            bootstrap_seed=int(args.bootstrap_seed),
        )
        save_json(outdir / "bootstrap_global_fit.json", bootstrap_payload)

    print("[plot] writing diagnostics")
    plot_maps(df, float(args.reference_vw), map_results, outdir, args.dpi)
    uq.plot_collapse_overlay(warped_df, fit_result, outdir, args.dpi, float(args.beta))
    rmse_rows = uq.plot_raw_xi_vs_betaH(warped_df, fit_result, outdir, args.dpi, float(args.beta))
    separate_rows = uq.plot_raw_xi_vs_betaH_separate(
        warped_df, fit_result, outdir / "xi_vs_betaH_nonparam_teff_separate", args.dpi
    )
    collapse.plot_residual_heatmap(warped_df, fit_result, outdir, args.dpi)
    theta_summary = plot_residual_by_theta(warped_df, fit_result, outdir, args.dpi)

    index_path = outdir / "xi_vs_betaH_nonparam_teff_index.csv"
    pd.DataFrame(separate_rows).to_csv(index_path, index=False)

    summary = {
        "status": "ok",
        "beta": float(args.beta),
        "reference_vw": float(args.reference_vw),
        "n_knots": int(args.n_knots),
        "smooth_lambda": float(args.smooth_lambda),
        "global_fit": {
            "t_c": float(global_payload["t_c"]),
            "r": float(global_payload["r"]),
            "rel_rmse": float(global_payload["rel_rmse"]),
            "AIC": float(global_payload["AIC"]),
            "BIC": float(global_payload["BIC"]),
        },
        "map_rel_rmse": map_payload,
        "rmse_by_vw": pd.DataFrame(rmse_rows).groupby("v_w", as_index=False)["rel_rmse"].mean().to_dict(orient="records"),
        "rmse_by_H": pd.DataFrame(rmse_rows).groupby("H", as_index=False)["rel_rmse"].mean().to_dict(orient="records"),
        "residual_by_theta": theta_summary.to_dict(orient="records"),
        "outputs": {
            "maps_plot": str(outdir / "nonparam_teff_maps.png"),
            "collapse_overlay": str(outdir / "collapse_overlay_quadwarp_universal.png"),
            "raw_H1p0": str(outdir / "xi_vs_betaH_quadwarp_universal_H1p0.png"),
            "raw_H1p5": str(outdir / "xi_vs_betaH_quadwarp_universal_H1p5.png"),
            "raw_H2p0": str(outdir / "xi_vs_betaH_quadwarp_universal_H2p0.png"),
            "residual_heatmap": str(outdir / "residual_heatmap.png"),
            "residual_by_theta": str(outdir / "residual_by_theta.csv"),
            "index_csv": str(index_path),
        },
    }
    if bootstrap_payload is not None:
        summary["bootstrap"] = bootstrap_payload
    save_json(outdir / "final_summary.json", summary)
    print(json.dumps(to_native(summary), sort_keys=True))


if __name__ == "__main__":
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        main()
    except Exception as exc:
        payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
        save_json(outdir / "_error.json", payload)
        print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True))
        raise
