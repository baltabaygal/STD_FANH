#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import collapse_and_fit_fanh_tosc as collapse
import fit_lattice_quadwarp_universal as uq


OUTDIR = ROOT / "results_vw0p9_kernel_vwmods"
MODEL_JSON = ROOT / "results_tosc_lattice_vw0p9_H1p0H1p5H2p0" / "collapse_and_fit_fanh" / "global_fit.json"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(description="Test vw-only modifiers on top of the frozen vw=0.9 lattice kernel without any time shift.")
    p.add_argument("--model-json", type=str, default=str(MODEL_JSON))
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--outdir", type=str, default=str(OUTDIR))
    p.add_argument("--a-min", type=float, default=0.4)
    p.add_argument("--a-max", type=float, default=1.8)
    p.add_argument("--r-min", type=float, default=0.2)
    p.add_argument("--r-max", type=float, default=15.0)
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


def base_prediction(sub_df, model, r_value=None):
    work = sub_df.copy()
    work["x"] = work["tp"].to_numpy(dtype=np.float64) * np.power(work["H"].to_numpy(dtype=np.float64), float(model["beta"]))
    params = np.asarray(model["params"], dtype=np.float64).copy()
    if r_value is not None:
        params[1] = float(r_value)
    yfit, _ = collapse.xi_model_from_params(work, model["theta_values"], params)
    return yfit, work["x"].to_numpy(dtype=np.float64)


def fit_none(sub_df, model):
    yfit, _ = base_prediction(sub_df, model, r_value=model["r"])
    return {"A": 1.0, "r": float(model["r"]), "success": True, "message": "fixed", "rel_rmse": rel_rmse(sub_df["xi"], yfit)}


def fit_amp(sub_df, model, a_min, a_max, fixed=False):
    ybase, _ = base_prediction(sub_df, model, r_value=model["r"])
    if fixed:
        return {"A": 1.0, "r": float(model["r"]), "success": True, "message": "fixed", "rel_rmse": rel_rmse(sub_df["xi"], ybase)}
    y = sub_df["xi"].to_numpy(dtype=np.float64)

    def objective(A):
        return rel_rmse(y, float(A) * ybase)

    res = minimize_scalar(objective, bounds=(float(a_min), float(a_max)), method="bounded", options={"xatol": 1.0e-4})
    A = float(res.x) if res.success else 1.0
    return {"A": A, "r": float(model["r"]), "success": bool(res.success), "message": str(getattr(res, "message", "")), "rel_rmse": float(objective(A))}


def fit_rvw(sub_df, model, r_min, r_max, fixed=False):
    if fixed:
        return fit_none(sub_df, model)
    y = sub_df["xi"].to_numpy(dtype=np.float64)

    def objective(r_value):
        yfit, _ = base_prediction(sub_df, model, r_value=float(r_value))
        return rel_rmse(y, yfit)

    res = minimize_scalar(objective, bounds=(float(r_min), float(r_max)), method="bounded", options={"xatol": 1.0e-4})
    r_value = float(res.x) if res.success else float(model["r"])
    return {"A": 1.0, "r": r_value, "success": bool(res.success), "message": str(getattr(res, "message", "")), "rel_rmse": float(objective(r_value))}


def fit_amp_rvw(sub_df, model, a_min, a_max, r_min, r_max, fixed=False):
    if fixed:
        return fit_none(sub_df, model)
    y = sub_df["xi"].to_numpy(dtype=np.float64)

    def objective(vec):
        A, r_value = vec
        ybase, _ = base_prediction(sub_df, model, r_value=float(r_value))
        return rel_rmse(y, float(A) * ybase)

    x0 = np.array([1.0, float(model["r"])], dtype=np.float64)
    bounds = [(float(a_min), float(a_max)), (float(r_min), float(r_max))]
    res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
    if res.success:
        A, r_value = [float(v) for v in res.x]
    else:
        A, r_value = 1.0, float(model["r"])
    return {"A": A, "r": r_value, "success": bool(res.success), "message": str(getattr(res, "message", "")), "rel_rmse": float(objective(np.array([A, r_value], dtype=np.float64)))}


def fit_model(df, model, kind, args):
    per_vw = {}
    for vw, sub in df.groupby("v_w", sort=True):
        fixed = np.isclose(float(vw), 0.9, atol=1.0e-12, rtol=0.0)
        if kind == "baseline":
            per_vw[f"{float(vw):.1f}"] = fit_none(sub, model)
        elif kind == "amp":
            per_vw[f"{float(vw):.1f}"] = fit_amp(sub, model, args.a_min, args.a_max, fixed=fixed)
        elif kind == "rvw":
            per_vw[f"{float(vw):.1f}"] = fit_rvw(sub, model, args.r_min, args.r_max, fixed=fixed)
        elif kind == "amp_rvw":
            per_vw[f"{float(vw):.1f}"] = fit_amp_rvw(sub, model, args.a_min, args.a_max, args.r_min, args.r_max, fixed=fixed)
        else:
            raise ValueError(f"Unknown kind: {kind}")
    return per_vw


def apply_model(df, model, per_vw, suffix):
    out = df.copy()
    x_vals = np.zeros(len(out), dtype=np.float64)
    yfit_vals = np.zeros(len(out), dtype=np.float64)
    A_vals = np.zeros(len(out), dtype=np.float64)
    r_vals = np.zeros(len(out), dtype=np.float64)
    for vw, sub_idx in out.groupby("v_w", sort=False).groups.items():
        rec = per_vw[f"{float(vw):.1f}"]
        sub = out.loc[sub_idx].copy()
        ybase, x = base_prediction(sub, model, r_value=rec["r"])
        yfit = float(rec["A"]) * ybase
        idx = np.asarray(sub_idx, dtype=np.int64)
        x_vals[idx] = x
        yfit_vals[idx] = yfit
        A_vals[idx] = float(rec["A"])
        r_vals[idx] = float(rec["r"])
    out["x"] = x_vals
    out[f"A_fit_{suffix}"] = A_vals
    out[f"r_fit_{suffix}"] = r_vals
    out[f"xi_fit_{suffix}"] = yfit_vals
    return out


def plot_collapse_overlay(df, fit_col, outpath: Path, title: str, dpi: int):
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
                ax.plot(cur["x"], cur[fit_col], color=colors[float(vw)], lw=1.6, alpha=0.95)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$x=t_p H_*^\beta$")
        ax.set_ylabel(r"$\xi$")
    for ax in axes[len(theta_values):]:
        ax.axis("off")
    handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
    labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_metric_scan(summary_rows, outpath: Path, dpi: int):
    models = [row["model"] for row in summary_rows]
    rmse = [row["global_rel_rmse"] for row in summary_rows]
    aic = [row["AIC"] for row in summary_rows]
    bic = [row["BIC"] for row in summary_rows]
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(x, rmse, color="tab:blue")
    axes[0].set_xticks(x, models, rotation=20)
    axes[0].set_title("Global rel-RMSE")
    axes[1].bar(x, aic, color="tab:orange")
    axes[1].set_xticks(x, models, rotation=20)
    axes[1].set_title("AIC")
    axes[2].bar(x, bic, color="tab:green")
    axes[2].set_xticks(x, models, rotation=20)
    axes[2].set_title("BIC")
    for ax in axes:
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    model = load_model(Path(args.model_json).resolve())

    print("[load] reading lattice dataframe")
    df = uq.load_lattice_dataframe(outdir, args.vw_folders, args.h_values)

    results = []
    model_k = {"baseline": 0, "amp": 3, "rvw": 3, "amp_rvw": 6}
    best_payload = None
    best_rmse = np.inf
    for kind in ["baseline", "amp", "rvw", "amp_rvw"]:
        print(f"[fit] {kind}")
        per_vw = fit_model(df, model, kind, args)
        out = apply_model(df, model, per_vw, kind)
        out.to_csv(outdir / f"predictions_{kind}.csv", index=False)
        fit_col = f"xi_fit_{kind}"
        rmse_global = rel_rmse(out["xi"], out[fit_col])
        aic, bic = aic_bic_frac(out["xi"], out[fit_col], model_k[kind])
        payload = {
            "model": kind,
            "global_rel_rmse": rmse_global,
            "AIC": aic,
            "BIC": bic,
            "fit_by_vw": per_vw,
            "rmse_by_vw": [{"v_w": float(vw), "rel_rmse": rel_rmse(sub["xi"], sub[fit_col])} for vw, sub in out.groupby("v_w", sort=True)],
            "rmse_by_H": [{"H": float(h), "rel_rmse": rel_rmse(sub["xi"], sub[fit_col])} for h, sub in out.groupby("H", sort=True)],
            "rmse_by_theta": [{"theta": float(theta), "rel_rmse": rel_rmse(sub["xi"], sub[fit_col])} for theta, sub in out.groupby("theta", sort=True)],
            "outputs": {
                "predictions": str(outdir / f"predictions_{kind}.csv"),
                "collapse_overlay": str(outdir / f"collapse_overlay_{kind}.png"),
            },
        }
        save_json(outdir / f"summary_{kind}.json", payload)
        title = rf"Frozen $v_w=0.9$ kernel with vw-only modifiers: {kind}, $\beta={model['beta']:.4f}$"
        plot_collapse_overlay(out, fit_col, outdir / f"collapse_overlay_{kind}.png", title, args.dpi)
        results.append(payload)
        if rmse_global < best_rmse:
            best_rmse = rmse_global
            best_payload = payload

    summary_rows = [{"model": row["model"], "global_rel_rmse": row["global_rel_rmse"], "AIC": row["AIC"], "BIC": row["BIC"]} for row in results]
    plot_metric_scan(summary_rows, outdir / "model_comparison.png", args.dpi)
    final_summary = {
        "status": "ok",
        "frozen_model_source": model["source"],
        "beta_frozen": float(model["beta"]),
        "t_c_frozen": float(model["t_c"]),
        "r_frozen": float(model["r"]),
        "models": results,
        "best_model": best_payload["model"] if best_payload else None,
        "outputs": {"comparison_plot": str(outdir / "model_comparison.png")},
    }
    save_json(outdir / "final_summary.json", final_summary)
    print(json.dumps(to_native(final_summary), sort_keys=True))


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
