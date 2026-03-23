#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import collapse_and_fit_fanh_tosc as collapse
import fit_lattice_quadwarp_universal as uq


OUTDIR = ROOT / "results_vw0p9_model_applied_all_vw"
MODEL_JSON = ROOT / "results_tosc_lattice_vw0p9_H1p0H1p5H2p0" / "collapse_and_fit_fanh" / "global_fit.json"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(
        description="Apply the frozen vw=0.9 lattice collapse fit to all vw values without any extra vw-dependent correction."
    )
    p.add_argument("--model-json", type=str, default=str(MODEL_JSON))
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
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


def rel_rmse(y, yfit):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(yfit) & (y > 0.0)
    if np.count_nonzero(mask) == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square((yfit[mask] - y[mask]) / np.maximum(y[mask], 1.0e-12)))))


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


def compute_predictions(df: pd.DataFrame, model):
    out = df.copy()
    out["x"] = out["tp"].to_numpy(dtype=np.float64) * np.power(out["H"].to_numpy(dtype=np.float64), float(model["beta"]))
    xi_fit, _ = collapse.xi_model_from_params(out, model["theta_values"], model["params"])
    out["xi_fit_vw0p9_model"] = xi_fit
    return out


def plot_collapse_overlay(df: pd.DataFrame, outdir: Path, dpi: int, beta: float):
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
                ax.plot(cur["x"], cur["xi_fit_vw0p9_model"], color=colors[float(vw)], lw=1.6, alpha=0.95)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$x=t_p H_*^\beta$")
        ax.set_ylabel(r"$\xi$")
    for ax in axes[len(theta_values):]:
        ax.axis("off")
    vw_handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
    vw_labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
    h_handles = [plt.Line2D([0], [0], color="black", marker=marker_map[h], linestyle="None") for h in h_values]
    h_labels = [rf"$H_*={h:g}$" for h in h_values]
    fig.legend(vw_handles + h_handles, vw_labels + h_labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(rf"Frozen $v_w=0.9$ collapse model applied to all $v_w$, $\beta={beta:.4f}$", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outdir / "collapse_overlay_vw0p9_model_allvw.png", dpi=dpi)
    plt.close(fig)


def plot_raw(df: pd.DataFrame, outdir: Path, dpi: int, beta: float):
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
                ax.plot(cur["beta_over_H"], cur["xi_fit_vw0p9_model"], color=colors[float(vw)], lw=1.8)
                rows.append({"H": float(h_value), "theta": float(theta), "v_w": float(vw), "rel_rmse": rel_rmse(cur["xi"], cur["xi_fit_vw0p9_model"])})
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
        fig.suptitle(rf"Frozen $v_w=0.9$ model in raw $\xi(\beta/H_*)$, $H_*={h_value:.1f}$, $\beta={beta:.4f}$", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        htag = str(float(h_value)).replace(".", "p")
        fig.savefig(outdir / f"xi_vs_betaH_vw0p9model_H{htag}.png", dpi=dpi)
        plt.close(fig)
    return rows


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    model = load_model(Path(args.model_json).resolve())

    print("[load] reading lattice dataframe")
    df = uq.load_lattice_dataframe(outdir, args.vw_folders, args.h_values)
    df = compute_predictions(df, model)
    df.to_csv(outdir / "predictions.csv", index=False)

    print("[plot] writing collapse and raw overlays")
    plot_collapse_overlay(df, outdir, args.dpi, model["beta"])
    raw_rows = plot_raw(df, outdir, args.dpi, model["beta"])

    rmse_by_vw = [{"v_w": float(vw), "rel_rmse": rel_rmse(sub["xi"], sub["xi_fit_vw0p9_model"])} for vw, sub in df.groupby("v_w", sort=True)]
    rmse_by_h = [{"H": float(h), "rel_rmse": rel_rmse(sub["xi"], sub["xi_fit_vw0p9_model"])} for h, sub in df.groupby("H", sort=True)]
    rmse_by_theta = [{"theta": float(theta), "rel_rmse": rel_rmse(sub["xi"], sub["xi_fit_vw0p9_model"])} for theta, sub in df.groupby("theta", sort=True)]

    summary = {
        "status": "ok",
        "frozen_model_source": model["source"],
        "beta": float(model["beta"]),
        "t_c": float(model["t_c"]),
        "r": float(model["r"]),
        "global_rel_rmse": rel_rmse(df["xi"], df["xi_fit_vw0p9_model"]),
        "rmse_by_vw": rmse_by_vw,
        "rmse_by_H": rmse_by_h,
        "rmse_by_theta": rmse_by_theta,
        "outputs": {
            "collapse_overlay": str(outdir / "collapse_overlay_vw0p9_model_allvw.png"),
            "raw_H1p0": str(outdir / "xi_vs_betaH_vw0p9model_H1p0.png"),
            "raw_H1p5": str(outdir / "xi_vs_betaH_vw0p9model_H1p5.png"),
            "raw_H2p0": str(outdir / "xi_vs_betaH_vw0p9model_H2p0.png"),
            "predictions": str(outdir / "predictions.csv"),
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
