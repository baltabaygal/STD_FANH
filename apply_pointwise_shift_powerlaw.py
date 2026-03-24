#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fit_lattice_quadwarp_universal as uq
import infer_pointwise_shift_from_vw0p9_baseline as inv


ROOT = Path(__file__).resolve().parent
MODEL_JSON = ROOT / "results_tosc_lattice_vw0p9_H1p0H1p5H2p0" / "collapse_and_fit_fanh" / "global_fit.json"
SHIFT_JSON = ROOT / "results_pointwise_shift_powerlaw" / "final_summary.json"
OUTDIR = ROOT / "results_pointwise_shift_powerlaw_applied"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]
REF_VW = 0.9


def parse_args():
    p = argparse.ArgumentParser(description="Apply a fitted shift power law to the frozen vw=0.9 baseline model.")
    p.add_argument("--model-json", type=str, default=str(MODEL_JSON))
    p.add_argument("--shift-json", type=str, default=str(SHIFT_JSON))
    p.add_argument("--outdir", type=str, default=str(OUTDIR))
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
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


def rel_rmse(y, yfit):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(yfit) & (y > 0.0)
    if np.count_nonzero(mask) == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square((yfit[mask] - y[mask]) / np.maximum(y[mask], 1.0e-12)))))


def shift_model(tp: np.ndarray, vw: np.ndarray, A: float, m: float, p: float):
    tp = np.asarray(tp, dtype=np.float64)
    vw = np.asarray(vw, dtype=np.float64)
    return 1.0 + A * (np.power(REF_VW / vw, m) - 1.0) * np.power(tp, p)


def load_shift_params(path: Path):
    payload = json.loads(path.read_text())
    rec = payload["best_params_named"]
    return float(rec["A"]), float(rec["m"]), float(rec["p"]), payload


def build_predictions(df: pd.DataFrame, model: dict, A: float, m: float, p: float):
    out = df.copy()
    out["x_base"] = out["tp"].to_numpy(dtype=np.float64) * np.power(out["H"].to_numpy(dtype=np.float64), float(model["beta"]))
    out["s_fit"] = shift_model(out["tp"].to_numpy(dtype=np.float64), out["v_w"].to_numpy(dtype=np.float64), A, m, p)
    out["x_eff"] = out["x_base"] * out["s_fit"]
    out["xi_pred"] = inv.baseline_predict(
        model["theta_values"],
        model["finf"],
        out["theta"].to_numpy(dtype=np.float64),
        out["F0"].to_numpy(dtype=np.float64),
        out["x_eff"].to_numpy(dtype=np.float64),
        model["t_c"],
        model["r"],
    )
    out["xi_pred_baseline"] = inv.baseline_predict(
        model["theta_values"],
        model["finf"],
        out["theta"].to_numpy(dtype=np.float64),
        out["F0"].to_numpy(dtype=np.float64),
        out["x_base"].to_numpy(dtype=np.float64),
        model["t_c"],
        model["r"],
    )
    out["frac_resid_shift"] = (out["xi_pred"] - out["xi"]) / np.maximum(out["xi"], 1.0e-12)
    out["frac_resid_baseline"] = (out["xi_pred_baseline"] - out["xi"]) / np.maximum(out["xi"], 1.0e-12)
    return out


def plot_collapse(df: pd.DataFrame, outdir: Path):
    theta_values = inv.choose_theta_subset(df["theta"].unique())
    vw_values = np.sort(df["v_w"].unique())
    h_values = np.sort(df["H"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    marker_map = {1.0: "s", 1.5: "^", 2.0: "D", 0.5: "o"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, theta in zip(axes, theta_values):
        sub = df[np.isclose(df["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
        for vw in vw_values:
            for h in h_values:
                cur = sub[
                    np.isclose(sub["v_w"], float(vw), atol=1.0e-12, rtol=0.0)
                    & np.isclose(sub["H"], float(h), atol=1.0e-12, rtol=0.0)
                ].sort_values("x_eff")
                if cur.empty:
                    continue
                ax.scatter(cur["x_eff"], cur["xi"], s=18, color=colors[float(vw)], marker=marker_map.get(float(h), "o"), alpha=0.85)
                ax.plot(cur["x_eff"], cur["xi_pred"], color=colors[float(vw)], lw=1.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$x_{\rm eff} = s\, t_p H_*^\beta$")
        ax.set_ylabel(r"$\xi$")
    for ax in axes[len(theta_values):]:
        ax.axis("off")
    vw_handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
    vw_labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
    h_handles = [plt.Line2D([0], [0], color="black", marker=marker_map[h], linestyle="None") for h in h_values]
    h_labels = [rf"$H_*={h:g}$" for h in h_values]
    fig.legend(vw_handles + h_handles, vw_labels + h_labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(r"Frozen $v_w=0.9$ baseline with fitted pointwise-shift power law", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outdir / "collapse_overlay_shift_powerlaw.png", dpi=220)
    plt.close(fig)


def plot_raw(df: pd.DataFrame, outdir: Path):
    theta_values = inv.choose_theta_subset(df["theta"].unique())
    vw_values = np.sort(df["v_w"].unique())
    h_values = np.sort(df["H"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    rows = []
    for h_value in h_values:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
        axes = axes.ravel()
        sub_h = df[np.isclose(df["H"], float(h_value), atol=1.0e-12, rtol=0.0)].copy()
        for ax, theta in zip(axes, theta_values):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
            for vw in vw_values:
                cur = sub[np.isclose(sub["v_w"], float(vw), atol=1.0e-12, rtol=0.0)].sort_values("beta_over_H")
                if cur.empty:
                    continue
                ax.scatter(cur["beta_over_H"], cur["xi"], s=22, color=colors[float(vw)], alpha=0.85)
                ax.plot(cur["beta_over_H"], cur["xi_pred"], color=colors[float(vw)], lw=1.8)
                rows.append(
                    {
                        "H": float(h_value),
                        "theta": float(theta),
                        "v_w": float(vw),
                        "rel_rmse_shift": rel_rmse(cur["xi"], cur["xi_pred"]),
                        "rel_rmse_baseline": rel_rmse(cur["xi"], cur["xi_pred_baseline"]),
                    }
                )
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$\beta/H_*$")
            ax.set_ylabel(r"$\xi$")
        for ax in axes[len(theta_values):]:
            ax.axis("off")
        handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
        labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
        fig.suptitle(rf"Shift-powerlaw prediction in raw $\xi(\beta/H_*)$, $H_*={h_value:.1f}$", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        tag = str(float(h_value)).replace(".", "p")
        fig.savefig(outdir / f"xi_vs_betaH_shift_powerlaw_H{tag}.png", dpi=220)
        plt.close(fig)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    model_json = Path(args.model_json).resolve()
    shift_json = Path(args.shift_json).resolve()
    model = inv.load_model(model_json)
    A, m, p, shift_payload = load_shift_params(shift_json)
    df = uq.load_lattice_dataframe(outdir, args.vw_folders, args.h_values)
    pred = build_predictions(df, model, A, m, p)
    pred.to_csv(outdir / "predictions.csv", index=False)

    plot_collapse(pred, outdir)
    fit_rows = plot_raw(pred, outdir)
    fit_rows.to_csv(outdir / "fit_rows.csv", index=False)

    summary = {
        "status": "ok",
        "frozen_model_source": str(model_json),
        "shift_source": str(shift_json),
        "shift_formula": r"s = 1 + A[(0.9/v_w)^m - 1] t_p^p",
        "shift_params": {"A": A, "m": m, "p": p},
        "global_rel_rmse_shift": rel_rmse(pred["xi"], pred["xi_pred"]),
        "global_rel_rmse_baseline": rel_rmse(pred["xi"], pred["xi_pred_baseline"]),
        "by_vw": [
            {
                "v_w": float(vw),
                "rel_rmse_shift": rel_rmse(sub["xi"], sub["xi_pred"]),
                "rel_rmse_baseline": rel_rmse(sub["xi"], sub["xi_pred_baseline"]),
                "median_abs_frac_resid_shift": float(np.median(np.abs(sub["frac_resid_shift"]))),
                "median_abs_frac_resid_baseline": float(np.median(np.abs(sub["frac_resid_baseline"]))),
            }
            for vw, sub in pred.groupby("v_w", sort=True)
        ],
        "by_H": [
            {
                "H": float(h),
                "rel_rmse_shift": rel_rmse(sub["xi"], sub["xi_pred"]),
                "rel_rmse_baseline": rel_rmse(sub["xi"], sub["xi_pred_baseline"]),
            }
            for h, sub in pred.groupby("H", sort=True)
        ],
        "outputs": {
            "collapse_overlay": str((outdir / "collapse_overlay_shift_powerlaw.png").resolve()),
            "raw_fit_by_H": [str((outdir / f"xi_vs_betaH_shift_powerlaw_H{str(float(h)).replace('.', 'p')}.png").resolve()) for h in sorted(pred["H"].unique())],
            "predictions": str((outdir / "predictions.csv").resolve()),
            "fit_rows": str((outdir / "fit_rows.csv").resolve()),
        },
        "fit_model_summary": shift_payload,
    }
    save_json(outdir / "final_summary.json", summary)
    print(json.dumps(to_native(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
