#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import collapse_and_fit_fanh_tosc as collapse
import fit_lattice_quadwarp_universal as uq


OUTDIR = ROOT / "results_vw0p9_kernel_amp_fullshift"
MODEL_JSON = ROOT / "results_tosc_lattice_vw0p9_H1p0H1p5H2p0" / "collapse_and_fit_fanh" / "global_fit.json"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(description="Fit amplitude A(v_w) and full time scale s(v_w) on top of the frozen vw=0.9 lattice kernel.")
    p.add_argument("--model-json", type=str, default=str(MODEL_JSON))
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--outdir", type=str, default=str(OUTDIR))
    p.add_argument("--a-min", type=float, default=0.4)
    p.add_argument("--a-max", type=float, default=1.8)
    p.add_argument("--s-min", type=float, default=0.2)
    p.add_argument("--s-max", type=float, default=3.0)
    return p.parse_args()


def to_native(obj):
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_native(v) for v in obj.tolist()]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
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


def aic_bic_frac(y, yfit, k):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(yfit) & (y > 0.0)
    n = int(np.count_nonzero(mask))
    if n == 0:
        return np.nan, np.nan
    resid = (yfit[mask] - y[mask]) / np.maximum(y[mask], 1.0e-12)
    rss = float(np.sum(resid * resid))
    scale = max(rss / max(n, 1), 1.0e-18)
    aic = float(n * math.log(scale) + 2.0 * k)
    bic = float(n * math.log(scale) + k * math.log(n))
    return aic, bic


def choose_theta_subset(theta_values):
    targets = np.array([0.262, 0.785, 1.309, 1.833, 2.356, 2.880], dtype=np.float64)
    theta_values = np.asarray(sorted(theta_values), dtype=np.float64)
    out = []
    for target in targets:
        idx = int(np.argmin(np.abs(theta_values - target)))
        val = float(theta_values[idx])
        if val not in out:
            out.append(val)
    return np.asarray(out, dtype=np.float64)


def load_model(path: Path):
    payload = json.loads(path.read_text())
    theta_values = np.array(sorted(float(k) for k in payload["F_inf"].keys()), dtype=np.float64)
    params = [float(payload["t_c"]), float(payload["r"])]
    for theta in theta_values:
        params.append(float(payload["F_inf"][f"{theta:.10f}"]["value"]))
    return {
        "beta": float(payload["beta"]),
        "t_c": float(payload["t_c"]),
        "r": float(payload["r"]),
        "theta_values": theta_values,
        "params": np.asarray(params, dtype=np.float64),
        "source": str(path.resolve()),
    }


def predict_with_amp_shift(sub_df, model, A, s):
    work = sub_df.copy()
    tp_scaled = float(s) * work["tp"].to_numpy(dtype=np.float64)
    x = tp_scaled * np.power(work["H"].to_numpy(dtype=np.float64), float(model["beta"]))
    work["x"] = x
    ybase, _ = collapse.xi_model_from_params(work, model["theta_values"], model["params"])
    return float(A) * ybase, x


def fit_amp_shift_for_vw(sub_df, model, a_min, a_max, s_min, s_max, fixed=False):
    if fixed:
        yfit, _ = predict_with_amp_shift(sub_df, model, 1.0, 1.0)
        return {"A": 1.0, "s": 1.0, "success": True, "message": "fixed", "rel_rmse": rel_rmse(sub_df["xi"], yfit)}

    y = sub_df["xi"].to_numpy(dtype=np.float64)

    def objective(vec):
        A, s = vec
        yfit, _ = predict_with_amp_shift(sub_df, model, A, s)
        return rel_rmse(y, yfit)

    x0 = np.array([1.0, 1.0], dtype=np.float64)
    bounds = [(float(a_min), float(a_max)), (float(s_min), float(s_max))]
    res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
    if res.success:
        A, s = [float(v) for v in res.x]
    else:
        A, s = 1.0, 1.0
    return {"A": A, "s": s, "success": bool(res.success), "message": str(getattr(res, "message", "")), "rel_rmse": float(objective(np.array([A, s], dtype=np.float64)))}


def apply_fit(df, model, per_vw):
    out = df.copy()
    x_vals = np.zeros(len(out), dtype=np.float64)
    yfit_vals = np.zeros(len(out), dtype=np.float64)
    A_vals = np.zeros(len(out), dtype=np.float64)
    s_vals = np.zeros(len(out), dtype=np.float64)
    for vw, sub_idx in out.groupby("v_w", sort=False).groups.items():
        rec = per_vw[f"{float(vw):.1f}"]
        sub = out.loc[sub_idx].copy()
        yfit, x = predict_with_amp_shift(sub, model, rec["A"], rec["s"])
        idx = np.asarray(sub_idx, dtype=np.int64)
        x_vals[idx] = x
        yfit_vals[idx] = yfit
        A_vals[idx] = float(rec["A"])
        s_vals[idx] = float(rec["s"])
    out["x"] = x_vals
    out["A_fit"] = A_vals
    out["s_fit"] = s_vals
    out["xi_fit_amp_fullshift"] = yfit_vals
    return out


def plot_collapse_overlay(df, outdir: Path, dpi: int, beta: float):
    theta_values = choose_theta_subset(df["theta"].unique())
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
                ].sort_values("x")
                if cur.empty:
                    continue
                ax.scatter(cur["x"], cur["xi"], s=20, color=colors[float(vw)], marker=marker_map.get(float(h), "o"), alpha=0.85)
                ax.plot(cur["x"], cur["xi_fit_amp_fullshift"], color=colors[float(vw)], lw=1.6, alpha=0.95)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$x=s(v_w)\,t_p H_*^\beta$")
        ax.set_ylabel(r"$\xi$")
    for ax in axes[len(theta_values):]:
        ax.axis("off")
    handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
    labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
    fig.suptitle(rf"Frozen $v_w=0.9$ kernel with fitted $A(v_w)$ and $s(v_w)$, $\beta={beta:.4f}$", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outdir / "collapse_overlay_amp_fullshift.png", dpi=dpi)
    plt.close(fig)


def plot_raw(df, outdir: Path, dpi: int, beta: float):
    theta_values = choose_theta_subset(df["theta"].unique())
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
                cur = sub[np.isclose(sub["v_w"], float(vw), atol=1.0e-12, rtol=0.0)].sort_values("beta_over_H").copy()
                if cur.empty:
                    continue
                ax.scatter(cur["beta_over_H"], cur["xi"], s=22, color=colors[float(vw)], alpha=0.85)
                ax.plot(cur["beta_over_H"], cur["xi_fit_amp_fullshift"], color=colors[float(vw)], lw=1.8)
                rows.append({"H": float(h_value), "theta": float(theta), "v_w": float(vw), "A_fit": float(cur["A_fit"].iloc[0]), "s_fit": float(cur["s_fit"].iloc[0]), "rel_rmse": rel_rmse(cur["xi"], cur["xi_fit_amp_fullshift"])})
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
        fig.suptitle(rf"Frozen $v_w=0.9$ kernel with $A(v_w)$ and $s(v_w)$ in raw $\xi(\beta/H_*)$, $H_*={h_value:.1f}$, $\beta={beta:.4f}$", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        htag = str(float(h_value)).replace(".", "p")
        fig.savefig(outdir / f"xi_vs_betaH_amp_fullshift_H{htag}.png", dpi=dpi)
        plt.close(fig)
    return rows


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    model = load_model(Path(args.model_json).resolve())
    print("[load] reading lattice dataframe")
    df = uq.load_lattice_dataframe(outdir, args.vw_folders, args.h_values)

    print("[fit] fitting A(v_w) and s(v_w)")
    per_vw = {}
    for vw, sub in df.groupby("v_w", sort=True):
        per_vw[f"{float(vw):.1f}"] = fit_amp_shift_for_vw(
            sub.copy(),
            model,
            args.a_min,
            args.a_max,
            args.s_min,
            args.s_max,
            fixed=np.isclose(float(vw), 0.9, atol=1.0e-12, rtol=0.0),
        )

    out = apply_fit(df, model, per_vw)
    out.to_csv(outdir / "predictions.csv", index=False)
    print("[plot] writing overlays")
    plot_collapse_overlay(out, outdir, args.dpi, model["beta"])
    raw_rows = plot_raw(out, outdir, args.dpi, model["beta"])

    summary = {
        "status": "ok",
        "frozen_model_source": model["source"],
        "beta": float(model["beta"]),
        "t_c_frozen": float(model["t_c"]),
        "r_frozen": float(model["r"]),
        "fit_by_vw": per_vw,
        "global_rel_rmse": rel_rmse(out["xi"], out["xi_fit_amp_fullshift"]),
        "AIC": aic_bic_frac(out["xi"], out["xi_fit_amp_fullshift"], k=6)[0],
        "BIC": aic_bic_frac(out["xi"], out["xi_fit_amp_fullshift"], k=6)[1],
        "rmse_by_vw": [{"v_w": float(vw), "rel_rmse": rel_rmse(sub["xi"], sub["xi_fit_amp_fullshift"])} for vw, sub in out.groupby("v_w", sort=True)],
        "rmse_by_H": [{"H": float(h), "rel_rmse": rel_rmse(sub["xi"], sub["xi_fit_amp_fullshift"])} for h, sub in out.groupby("H", sort=True)],
        "rmse_by_theta": [{"theta": float(theta), "rel_rmse": rel_rmse(sub["xi"], sub["xi_fit_amp_fullshift"])} for theta, sub in out.groupby("theta", sort=True)],
        "outputs": {
            "predictions": str(outdir / "predictions.csv"),
            "collapse_overlay": str(outdir / "collapse_overlay_amp_fullshift.png"),
            "raw_H1p0": str(outdir / "xi_vs_betaH_amp_fullshift_H1p0.png"),
            "raw_H1p5": str(outdir / "xi_vs_betaH_amp_fullshift_H1p5.png"),
            "raw_H2p0": str(outdir / "xi_vs_betaH_amp_fullshift_H2p0.png"),
        },
        "raw_plot_rmse": raw_rows,
    }
    save_json(outdir / "final_summary.json", summary)
    print(json.dumps(to_native(summary), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        outdir = OUTDIR.resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
        save_json(outdir / "_error.json", payload)
        print(json.dumps(payload, sort_keys=True))
        raise
