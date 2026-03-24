#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import apply_pointwise_shift_powerlaw as shiftmod
import infer_pointwise_shift_from_vw0p9_baseline as inv
import fit_lattice_quadwarp_universal as uq


ROOT = Path(__file__).resolve().parent
MODEL_JSON = ROOT / "results_tosc_lattice_vw0p9_H1p0H1p5H2p0" / "collapse_and_fit_fanh" / "global_fit.json"
SHIFT_JSON = ROOT / "results_pointwise_shift_powerlaw" / "final_summary.json"
OUTDIR = ROOT / "results_shift_width_correction"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]
REF_VW = 0.9


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


def rel_rmse(y, yfit):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(yfit) & (y > 0.0)
    if np.count_nonzero(mask) == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square((yfit[mask] - y[mask]) / np.maximum(y[mask], 1.0e-12)))))


def width_sigma2(tp: np.ndarray, vw: np.ndarray, B: float, m: float, p: float):
    tp = np.asarray(tp, dtype=np.float64)
    vw = np.asarray(vw, dtype=np.float64)
    return B * (np.power(REF_VW / vw, m) - 1.0) * np.power(tp, p)


def curvature_terms(model: dict, theta: np.ndarray, f0: np.ndarray, x: np.ndarray):
    theta = np.asarray(theta, dtype=np.float64)
    f0 = np.asarray(f0, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    h = np.maximum(1.0e-3 * x, 1.0e-5)
    h = np.minimum(h, 0.49 * x)
    xp = x + h
    xm = np.maximum(x - h, 1.0e-8)
    f0v = f0

    y0 = inv.baseline_predict(model["theta_values"], model["finf"], theta, f0v, x, model["t_c"], model["r"])
    yp = inv.baseline_predict(model["theta_values"], model["finf"], theta, f0v, xp, model["t_c"], model["r"])
    ym = inv.baseline_predict(model["theta_values"], model["finf"], theta, f0v, xm, model["t_c"], model["r"])
    curv_x = 0.5 * (yp - 2.0 * y0 + ym) / np.maximum(h * h, 1.0e-18)

    dlog = 1.0e-3
    xp_log = x * np.exp(dlog)
    xm_log = x * np.exp(-dlog)
    yp_log = inv.baseline_predict(model["theta_values"], model["finf"], theta, f0v, xp_log, model["t_c"], model["r"])
    ym_log = inv.baseline_predict(model["theta_values"], model["finf"], theta, f0v, xm_log, model["t_c"], model["r"])
    curv_log = 0.5 * (yp_log - 2.0 * y0 + ym_log) / (dlog * dlog)
    return curv_x, curv_log


def build_base_table():
    model = inv.load_model(MODEL_JSON.resolve())
    A, m, p, shift_payload = shiftmod.load_shift_params(SHIFT_JSON.resolve())
    df = uq.load_lattice_dataframe(OUTDIR.resolve(), VW_TAGS, H_VALUES)
    pred = shiftmod.build_predictions(df, model, A, m, p)
    curv_x, curv_log = curvature_terms(
        model,
        pred["theta"].to_numpy(dtype=np.float64),
        pred["F0"].to_numpy(dtype=np.float64),
        pred["x_eff"].to_numpy(dtype=np.float64),
    )
    pred["curv_x_half"] = curv_x
    pred["curv_log_half"] = curv_log
    pred["shift_payload_A"] = A
    pred["shift_payload_m"] = m
    pred["shift_payload_p"] = p
    return pred, model, shift_payload


def fit_width(df: pd.DataFrame, curvature_col: str):
    tp = df["tp"].to_numpy(dtype=np.float64)
    vw = df["v_w"].to_numpy(dtype=np.float64)
    xi = df["xi"].to_numpy(dtype=np.float64)
    xi_shift = df["xi_pred"].to_numpy(dtype=np.float64)
    K = df[curvature_col].to_numpy(dtype=np.float64)

    def model(par):
        B, m, p = par
        sig2 = width_sigma2(tp, vw, B, m, p)
        return xi_shift + sig2 * K

    def resid(par):
        return (model(par) - xi) / np.maximum(xi, 1.0e-12)

    res = least_squares(
        resid,
        x0=np.array([0.01, 0.5, -1.0], dtype=np.float64),
        bounds=([0.0, 0.0, -5.0], [100.0, 5.0, 5.0]),
        loss="linear",
        max_nfev=8000,
    )
    pred = model(res.x)
    return {
        "params": res.x,
        "success": bool(res.success),
        "message": res.message,
        "pred": pred,
        "rel_rmse": rel_rmse(xi, pred),
    }


def plot_tp_residual(df: pd.DataFrame, outdir: Path):
    vw_values = np.sort(df["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for vw in vw_values:
        cur = df[np.isclose(df["v_w"], float(vw), atol=1.0e-12, rtol=0.0)].copy()
        cur = cur.sort_values("tp")
        axes[0].scatter(cur["tp"], cur["frac_resid_shift"], s=12, color=colors[float(vw)], alpha=0.35)
        axes[1].scatter(cur["tp"], cur["frac_resid_widthlog"], s=12, color=colors[float(vw)], alpha=0.35)
    for ax, title in zip(axes, ["Shift only", "Shift + width"]):
        ax.axhline(0.0, color="black", lw=1.0, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel(r"$t_p$")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("fractional residual")
    fig.tight_layout()
    fig.savefig(outdir / "residual_vs_tp_shift_vs_width.png", dpi=220)
    plt.close(fig)


def main():
    outdir = OUTDIR.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    df, model, shift_payload = build_base_table()
    df.to_csv(outdir / "base_predictions_with_curvature.csv", index=False)

    fit_x = fit_width(df, "curv_x_half")
    fit_log = fit_width(df, "curv_log_half")

    df["xi_pred_widthx"] = fit_x["pred"]
    df["xi_pred_widthlog"] = fit_log["pred"]
    df["frac_resid_shift"] = (df["xi_pred"] - df["xi"]) / np.maximum(df["xi"], 1.0e-12)
    df["frac_resid_widthx"] = (df["xi_pred_widthx"] - df["xi"]) / np.maximum(df["xi"], 1.0e-12)
    df["frac_resid_widthlog"] = (df["xi_pred_widthlog"] - df["xi"]) / np.maximum(df["xi"], 1.0e-12)
    df.to_csv(outdir / "predictions_with_width.csv", index=False)

    plot_tp_residual(df, outdir)

    summary = {
        "status": "ok",
        "shift_only_rel_rmse": rel_rmse(df["xi"], df["xi_pred"]),
        "width_x_model": {
            "formula": r"xi = xi_shift + sigma_x^2 * (1/2) d^2 xi/dx^2",
            "sigma2_formula": r"sigma_x^2 = B[(0.9/v_w)^m - 1] t_p^p",
            "params": {"B": float(fit_x["params"][0]), "m": float(fit_x["params"][1]), "p": float(fit_x["params"][2])},
            "rel_rmse": float(fit_x["rel_rmse"]),
        },
        "width_log_model": {
            "formula": r"xi = xi_shift + sigma_log^2 * (1/2) d^2 xi/d(log x)^2",
            "sigma2_formula": r"sigma_log^2 = B[(0.9/v_w)^m - 1] t_p^p",
            "params": {"B": float(fit_log["params"][0]), "m": float(fit_log["params"][1]), "p": float(fit_log["params"][2])},
            "rel_rmse": float(fit_log["rel_rmse"]),
        },
        "correlations": {
            "shift_resid_vs_abs_curv_x": float(np.corrcoef(np.abs(df["frac_resid_shift"]), np.abs(df["curv_x_half"]))[0, 1]),
            "shift_resid_vs_abs_curv_log": float(np.corrcoef(np.abs(df["frac_resid_shift"]), np.abs(df["curv_log_half"]))[0, 1]),
        },
        "by_vw": [
            {
                "v_w": float(vw),
                "rel_rmse_shift": rel_rmse(sub["xi"], sub["xi_pred"]),
                "rel_rmse_widthx": rel_rmse(sub["xi"], sub["xi_pred_widthx"]),
                "rel_rmse_widthlog": rel_rmse(sub["xi"], sub["xi_pred_widthlog"]),
                "median_abs_frac_shift": float(np.median(np.abs(sub["frac_resid_shift"]))),
                "median_abs_frac_widthlog": float(np.median(np.abs(sub["frac_resid_widthlog"]))),
            }
            for vw, sub in df.groupby("v_w", sort=True)
        ],
        "low_tp_vs_high_tp": [
            {
                "v_w": float(vw),
                "shift_lowtp_mean_abs": float(np.mean(np.abs(sub.loc[sub["tp"] <= sub["tp"].median(), "frac_resid_shift"]))),
                "shift_hightp_mean_abs": float(np.mean(np.abs(sub.loc[sub["tp"] > sub["tp"].median(), "frac_resid_shift"]))),
                "widthlog_lowtp_mean_abs": float(np.mean(np.abs(sub.loc[sub["tp"] <= sub["tp"].median(), "frac_resid_widthlog"]))),
                "widthlog_hightp_mean_abs": float(np.mean(np.abs(sub.loc[sub["tp"] > sub["tp"].median(), "frac_resid_widthlog"]))),
            }
            for vw, sub in df.groupby("v_w", sort=True)
        ],
        "outputs": {
            "base_table": str((outdir / "base_predictions_with_curvature.csv").resolve()),
            "predictions": str((outdir / "predictions_with_width.csv").resolve()),
            "residual_plot": str((outdir / "residual_vs_tp_shift_vs_width.png").resolve()),
        },
        "shift_model_summary": shift_payload,
    }
    save_json(outdir / "final_summary.json", summary)
    print(json.dumps(to_native(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
