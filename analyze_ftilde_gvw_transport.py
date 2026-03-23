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
DEFAULT_PRED = ROOT / "results_vw0p9_model_with_gvw" / "predictions_with_gvw.csv"
DEFAULT_OUTDIR = ROOT / "results_ftilde_gvw_transport"
T_OSC = 1.5


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract corrected ftilde from the g(v_w) transport fit and compare to the frozen vw=0.9 baseline."
    )
    p.add_argument("--predictions", type=str, default=str(DEFAULT_PRED))
    p.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    p.add_argument("--dpi", type=int, default=220)
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


def marker_map_for_h():
    return {1.0: "o", 1.5: "s", 2.0: "^"}


def build_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    x = np.maximum(df["x"].to_numpy(dtype=np.float64), 1.0e-18)
    pref = np.power(x / T_OSC, 1.5)
    df["ftilde_data"] = df["xi"].to_numpy(dtype=np.float64) * df["F0"].to_numpy(dtype=np.float64) / np.maximum(pref, 1.0e-18)
    df["ftilde_baseline"] = (
        df["xi_fit_vw0p9_model"].to_numpy(dtype=np.float64) * df["F0"].to_numpy(dtype=np.float64) / np.maximum(pref, 1.0e-18)
    )
    df["ftilde_corr"] = df["ftilde_data"].to_numpy(dtype=np.float64) / np.maximum(df["g_vw"].to_numpy(dtype=np.float64), 1.0e-18)
    df["ftilde_ratio"] = df["ftilde_corr"].to_numpy(dtype=np.float64) / np.maximum(df["ftilde_baseline"].to_numpy(dtype=np.float64), 1.0e-18)
    df["ftilde_frac_resid"] = df["ftilde_ratio"] - 1.0
    return df


def plot_by_theta(df: pd.DataFrame, outdir: Path, dpi: int):
    theta_values = np.sort(df["theta"].unique())
    vw_values = np.sort(df["v_w"].unique())
    colors = {float(vw): plt.get_cmap("viridis")(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    markers = marker_map_for_h()
    rows = []

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, theta in zip(axes, theta_values):
        sub = df[np.isclose(df["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy().sort_values("x")
        for (vw, h), cur in sub.groupby(["v_w", "H"], sort=True):
            ax.scatter(
                cur["x"],
                cur["ftilde_corr"],
                color=colors[float(vw)],
                marker=markers.get(float(h), "o"),
                s=18,
                alpha=0.9,
            )
        base = sub[sub["v_w"] == 0.9].sort_values("x")
        if not base.empty:
            ax.plot(base["x"], base["ftilde_baseline"], color="black", lw=1.8, label="baseline")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$x=t_p H^\beta$")
        ax.set_ylabel(r"$\tilde f_{\rm corr}$")
        rows.append(
            {
                "theta": float(theta),
                "mean_abs_frac_resid": float(np.mean(np.abs(sub["ftilde_frac_resid"].to_numpy(dtype=np.float64)))),
                "median_abs_frac_resid": float(np.median(np.abs(sub["ftilde_frac_resid"].to_numpy(dtype=np.float64)))),
            }
        )
    for ax in axes[len(theta_values):]:
        ax.axis("off")
    color_handles = [plt.Line2D([0], [0], color=colors[float(vw)], lw=2.0) for vw in vw_values]
    color_labels = [rf"$v_w={float(vw):.1f}$" for vw in vw_values]
    marker_handles = [plt.Line2D([0], [0], color="black", marker=markers[h], lw=0, ms=6) for h in sorted(markers)]
    marker_labels = [rf"$H={h:.1f}$" for h in sorted(markers)]
    fig.legend(color_handles + marker_handles, color_labels + marker_labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outdir / "ftilde_corrected_vs_x_by_theta.png", dpi=dpi)
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_ratio_heatmaps(df: pd.DataFrame, outdir: Path, dpi: int):
    for col_col, stem in [("beta_over_H", "theta_betaH"), ("v_w", "theta_vw")]:
        pivot = (
            df.groupby(["theta", col_col])["ftilde_frac_resid"]
            .mean()
            .reset_index()
            .pivot(index="theta", columns=col_col, values="ftilde_frac_resid")
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        vmax = float(np.nanmax(np.abs(pivot.to_numpy(dtype=np.float64))))
        im = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{float(v):.3g}" for v in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{float(v):.3f}" for v in pivot.index])
        ax.set_xlabel(col_col)
        ax.set_ylabel("theta")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"$\tilde f_{\rm corr}/\tilde f_{v_w=0.9} - 1$")
        fig.tight_layout()
        fig.savefig(outdir / f"ftilde_resid_heatmap_{stem}.png", dpi=dpi)
        plt.close(fig)


def plot_ratio_vs_x(df: pd.DataFrame, outdir: Path, dpi: int):
    bins = np.linspace(np.log10(df["x"].min()), np.log10(df["x"].max()), 16)
    cats = pd.cut(np.log10(df["x"]), bins=bins, include_lowest=True)
    rows = []
    for interval, g in df.groupby(cats, observed=False):
        if len(g) == 0:
            continue
        rows.append(
            {
                "bin_center": float(0.5 * (interval.left + interval.right)),
                "mean_frac_resid": float(np.mean(g["ftilde_frac_resid"])),
                "mean_abs_frac_resid": float(np.mean(np.abs(g["ftilde_frac_resid"]))),
                "n_points": int(len(g)),
            }
        )
    agg = pd.DataFrame(rows).sort_values("bin_center")
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(agg["bin_center"], agg["mean_abs_frac_resid"], marker="o", lw=1.8, label="mean |frac resid|")
    ax.plot(agg["bin_center"], np.abs(agg["mean_frac_resid"]), marker="s", lw=1.5, label="|mean frac resid|")
    ax.grid(alpha=0.25)
    ax.set_xlabel(r"$\log_{10} x$")
    ax.set_ylabel(r"$\tilde f$ residual level")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "ftilde_resid_vs_logx.png", dpi=dpi)
    plt.close(fig)
    return agg


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = build_df(Path(args.predictions).resolve())
    df.to_csv(outdir / "ftilde_table.csv", index=False)

    by_theta = plot_by_theta(df, outdir, args.dpi)
    by_theta.to_csv(outdir / "ftilde_resid_by_theta.csv", index=False)
    by_x = plot_ratio_vs_x(df, outdir, args.dpi)
    by_x.to_csv(outdir / "ftilde_resid_by_logx.csv", index=False)
    plot_ratio_heatmaps(df, outdir, args.dpi)

    summary = {
        "status": "ok",
        "source_predictions": str(Path(args.predictions).resolve()),
        "aggregate": {
            "mean_abs_frac_resid": float(np.mean(np.abs(df["ftilde_frac_resid"].to_numpy(dtype=np.float64)))),
            "median_abs_frac_resid": float(np.median(np.abs(df["ftilde_frac_resid"].to_numpy(dtype=np.float64)))),
            "std_frac_resid": float(np.std(df["ftilde_frac_resid"].to_numpy(dtype=np.float64))),
        },
        "worst_theta": by_theta.sort_values("mean_abs_frac_resid", ascending=False).head(3).to_dict(orient="records"),
        "outputs": {
            "ftilde_table": str(outdir / "ftilde_table.csv"),
            "ftilde_by_theta_plot": str(outdir / "ftilde_corrected_vs_x_by_theta.png"),
            "ftilde_resid_by_theta": str(outdir / "ftilde_resid_by_theta.csv"),
            "ftilde_resid_by_logx": str(outdir / "ftilde_resid_by_logx.csv"),
            "ftilde_resid_vs_logx": str(outdir / "ftilde_resid_vs_logx.png"),
            "ftilde_resid_heatmap_theta_betaH": str(outdir / "ftilde_resid_heatmap_theta_betaH.png"),
            "ftilde_resid_heatmap_theta_vw": str(outdir / "ftilde_resid_heatmap_theta_vw.png"),
        },
    }
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
