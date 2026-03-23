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
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import collapse_and_fit_fanh_tosc as collapse
import fit_vw_amplitude as base_fit
import fit_vw_smearing_tests as smear
import fit_lattice_quadwarp_universal as uq


OUTDIR = ROOT / "results_teff_clean_universal_beta0"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit a clean analytic t_eff(v_w,t_p) law, then fit one universal lattice curve in the warped variable."
    )
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--reference-vw", type=float, default=0.9)
    p.add_argument("--model", choices=["scalar", "linear", "quadratic", "auto"], default="auto")
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


def g_vw(vw: float, ref_vw: float) -> float:
    return float(np.log(float(ref_vw) / float(vw)))


def warp_tp(tp, vw: float, ref_vw: float, model: str, params):
    tp = np.asarray(tp, dtype=np.float64)
    ltp = np.log(np.maximum(tp, 1.0e-18))
    g = g_vw(vw, ref_vw)
    if model == "scalar":
        a = float(params[0])
        delta = g * a
    elif model == "linear":
        a, b = [float(v) for v in params[:2]]
        delta = g * (a + b * ltp)
    elif model == "quadratic":
        a, b, c = [float(v) for v in params[:3]]
        delta = g * (a + b * ltp + c * ltp * ltp)
    else:
        raise ValueError(model)
    return np.exp(ltp + delta)


def derivative_margin(tp_all, vw_values, ref_vw: float, model: str, params):
    ltp = np.log(np.maximum(np.asarray(tp_all, dtype=np.float64), 1.0e-18))
    mins = []
    for vw in vw_values:
        if np.isclose(float(vw), float(ref_vw), atol=1.0e-12, rtol=0.0):
            continue
        g = g_vw(float(vw), ref_vw)
        if model == "scalar":
            deriv = np.ones_like(ltp)
        elif model == "linear":
            _, b = [float(v) for v in params[:2]]
            deriv = np.ones_like(ltp) * (1.0 + g * b)
        elif model == "quadratic":
            _, b, c = [float(v) for v in params[:3]]
            deriv = 1.0 + g * (b + 2.0 * c * ltp)
        mins.append(float(np.min(deriv)))
    if not mins:
        return 1.0
    return float(np.min(mins))


def alignment_residuals(pairs_map, ref_vw: float, model: str, params):
    residuals = []
    for vw, pair_list in pairs_map.items():
        for pair in pair_list:
            x_ref = pair["ref_tp"]
            x_cur = warp_tp(pair["tp"], float(vw), ref_vw, model, params)
            pred = smear.log_interp(x_ref, pair["ref_xi"], x_cur)
            mask = np.isfinite(pred) & np.isfinite(pair["xi"]) & (pair["xi"] > 0.0)
            if np.count_nonzero(mask) < 3:
                continue
            residuals.append((pred[mask] - pair["xi"][mask]) / np.maximum(pair["xi"][mask], 1.0e-12))
    if not residuals:
        return np.array([], dtype=np.float64)
    return np.concatenate(residuals)


def fit_clean_teff_model(df, ref_vw: float, model: str):
    pairs_map = smear.build_alignment_pairs(df, ref_vw)
    tp_all = df["tp"].to_numpy(dtype=np.float64)
    vw_values = np.sort(df["v_w"].unique())

    if model == "scalar":
        x0 = np.array([0.20], dtype=np.float64)
        bounds = [(-2.0, 2.0)]
    elif model == "linear":
        x0 = np.array([0.18, -0.25], dtype=np.float64)
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
    elif model == "quadratic":
        x0 = np.array([0.16, -0.12, 0.35], dtype=np.float64)
        bounds = [(-2.0, 2.0), (-2.0, 2.0), (-1.0, 1.0)]
    else:
        raise ValueError(model)

    def objective(x):
        margin = derivative_margin(tp_all, vw_values, ref_vw, model, x)
        if (not np.isfinite(margin)) or margin <= 0.05:
            return 1.0e9 + 1.0e6 * (0.05 - margin if np.isfinite(margin) else 1.0)
        resid = alignment_residuals(pairs_map, ref_vw, model, x)
        if resid.size == 0:
            return 1.0e9
        return float(np.mean(np.square(resid)))

    res = minimize(objective, x0=x0, method="L-BFGS-B", bounds=bounds)
    resid = alignment_residuals(pairs_map, ref_vw, model, res.x)
    return {
        "model": model,
        "success": bool(res.success),
        "message": str(res.message),
        "params": np.asarray(res.x, dtype=np.float64),
        "rel_rmse": float(smear.pooled_rel_rmse(resid)),
        "n_points": int(resid.size),
        "monotonic_margin": derivative_margin(tp_all, vw_values, ref_vw, model, res.x),
    }


def fit_all_clean_models(df, ref_vw: float):
    out = {}
    for model in ["scalar", "linear", "quadratic"]:
        out[model] = fit_clean_teff_model(df, ref_vw, model)
    best = min(out.values(), key=lambda rec: rec["rel_rmse"])
    return out, best["model"]


def apply_clean_warp(df, ref_vw: float, model: str, params, beta: float):
    out = df.copy()
    t_eff = np.zeros(len(out), dtype=np.float64)
    x = np.zeros(len(out), dtype=np.float64)
    for i, row in enumerate(out.itertuples(index=False)):
        teff = float(warp_tp(float(row.tp), float(row.v_w), ref_vw, model, params))
        t_eff[i] = teff
        x[i] = teff * np.power(float(row.H), float(beta))
    out["t_eff"] = t_eff
    out["x"] = x
    return out[np.isfinite(out["x"]) & (out["x"] > 0.0)].copy()


def plot_teff_function(df, ref_vw: float, model: str, params, outdir: Path, dpi: int):
    vw_values = np.sort(df["v_w"].unique())
    tp_grid = np.geomspace(float(np.min(df["tp"])), float(np.max(df["tp"])), 250)
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax0, ax1 = axes
    for vw in vw_values:
        teff = warp_tp(tp_grid, float(vw), ref_vw, model, params)
        ax0.plot(tp_grid, teff, color=colors[float(vw)], lw=2.0, label=rf"$v_w={vw:.1f}$")
        ax1.plot(tp_grid, teff / tp_grid, color=colors[float(vw)], lw=2.0)
    ax0.plot(tp_grid, tp_grid, color="black", ls="--", lw=1.2)
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel(r"$t_p$")
    ax0.set_ylabel(r"$t_{\rm eff}$")
    ax0.set_title("Clean time remapping")
    ax0.grid(alpha=0.25)
    ax0.legend(frameon=False)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$t_p$")
    ax1.set_ylabel(r"$t_{\rm eff}/t_p$")
    ax1.set_title("Relative warp")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "teff_clean_form.png", dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("[load] reading lattice dataframe")
    df = load_lattice_dataframe(outdir, args.vw_folders, args.h_values)

    print("[warp] fitting clean t_eff(v_w,t_p) families")
    all_models, auto_best = fit_all_clean_models(df, float(args.reference_vw))
    selected = auto_best if args.model == "auto" else args.model
    selected_fit = all_models[selected]
    save_json(outdir / "teff_model_search.json", all_models)

    print(f"[warp] selected model: {selected}")
    warped_df = apply_clean_warp(df, float(args.reference_vw), selected, selected_fit["params"], float(args.beta))

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

    print("[plot] writing overlays")
    plot_teff_function(df, float(args.reference_vw), selected, selected_fit["params"], outdir, args.dpi)
    uq.plot_collapse_overlay(warped_df, fit_result, outdir, args.dpi, float(args.beta))
    rmse_rows = uq.plot_raw_xi_vs_betaH(warped_df, fit_result, outdir, args.dpi, float(args.beta))
    separate_rows = uq.plot_raw_xi_vs_betaH_separate(
        warped_df, fit_result, outdir / "xi_vs_betaH_clean_teff_separate", args.dpi
    )
    collapse.plot_residual_heatmap(warped_df, fit_result, outdir, args.dpi)

    import pandas as pd

    index_path = outdir / "xi_vs_betaH_clean_teff_index.csv"
    pd.DataFrame(separate_rows).to_csv(index_path, index=False)

    formula = None
    if selected == "scalar":
        a = float(selected_fit["params"][0])
        formula = rf"\log t_{{eff}} = \log t_p + {a:.4f}\,\log({args.reference_vw:.1f}/v_w)"
    elif selected == "linear":
        a, b = [float(v) for v in selected_fit["params"][:2]]
        formula = rf"\log t_{{eff}} = \log t_p + \log({args.reference_vw:.1f}/v_w)\,({a:.4f} + {b:.4f}\log t_p)"
    else:
        a, b, c = [float(v) for v in selected_fit["params"][:3]]
        formula = rf"\log t_{{eff}} = \log t_p + \log({args.reference_vw:.1f}/v_w)\,({a:.4f} + {b:.4f}\log t_p + {c:.4f}(\log t_p)^2)"

    summary = {
        "status": "ok",
        "beta": float(args.beta),
        "reference_vw": float(args.reference_vw),
        "selected_model": selected,
        "selected_formula": formula,
        "selected_fit": {
            "model": selected_fit["model"],
            "params": [float(v) for v in selected_fit["params"]],
            "alignment_rel_rmse": float(selected_fit["rel_rmse"]),
            "monotonic_margin": float(selected_fit["monotonic_margin"]),
        },
        "all_teff_models": {
            key: {
                "params": [float(v) for v in val["params"]],
                "alignment_rel_rmse": float(val["rel_rmse"]),
                "success": bool(val["success"]),
                "monotonic_margin": float(val["monotonic_margin"]),
            }
            for key, val in all_models.items()
        },
        "global_fit": {
            "t_c": float(global_payload["t_c"]),
            "r": float(global_payload["r"]),
            "rel_rmse": float(global_payload["rel_rmse"]),
            "AIC": float(global_payload["AIC"]),
            "BIC": float(global_payload["BIC"]),
        },
        "rmse_by_vw": pd.DataFrame(rmse_rows).groupby("v_w", as_index=False)["rel_rmse"].mean().to_dict(orient="records"),
        "rmse_by_H": pd.DataFrame(rmse_rows).groupby("H", as_index=False)["rel_rmse"].mean().to_dict(orient="records"),
        "outputs": {
            "teff_plot": str(outdir / "teff_clean_form.png"),
            "collapse_overlay": str(outdir / "collapse_overlay_quadwarp_universal.png"),
            "raw_H1p0": str(outdir / "xi_vs_betaH_quadwarp_universal_H1p0.png"),
            "raw_H1p5": str(outdir / "xi_vs_betaH_quadwarp_universal_H1p5.png"),
            "raw_H2p0": str(outdir / "xi_vs_betaH_quadwarp_universal_H2p0.png"),
            "residual_heatmap": str(outdir / "residual_heatmap.png"),
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
