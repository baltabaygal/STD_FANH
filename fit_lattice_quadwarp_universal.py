#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import collapse_and_fit_fanh_tosc as collapse
import fit_vw_amplitude as base_fit
import plot_lattice_xi_vs_x_all_vw_quadwarped as quadwarp


OUTDIR = ROOT / "results_vw_quadwarp_universal_beta0"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit a true universal lattice curve after applying the fitted per-vw quadratic log-time warp."
    )
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--reference-vw", type=float, default=0.9)
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


def apply_quadwarp(df, warp_params, beta):
    out = df.copy()
    x_vals = []
    teff_vals = []
    for row in out.itertuples(index=False):
        rec = warp_params.get(f"{float(row.v_w):.1f}")
        if rec is None:
            raise RuntimeError(f"Missing warp parameters for v_w={float(row.v_w):.3f}")
        teff = quadwarp.warp_tp(float(row.tp), rec["log_s"], rec["b"], rec["c"])
        teff_vals.append(float(teff))
        x_vals.append(float(teff * np.power(float(row.H), float(beta))))
    out["t_eff"] = np.asarray(teff_vals, dtype=np.float64)
    out["x"] = np.asarray(x_vals, dtype=np.float64)
    out = out[np.isfinite(out["x"]) & (out["x"] > 0.0)].copy()
    return out.sort_values(["v_w", "H", "theta", "beta_over_H"]).reset_index(drop=True)


def xi_model(sub_df, fit_result):
    xi_fit, _ = collapse.xi_model_from_params(sub_df, fit_result["theta_values"], fit_result["params"])
    return xi_fit


def rel_rmse(y, yfit):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(yfit) & (y > 0.0)
    if np.count_nonzero(mask) == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square((yfit[mask] - y[mask]) / np.maximum(y[mask], 1.0e-12)))))


def plot_collapse_overlay(df, fit_result, outdir: Path, dpi: int, beta: float):
    theta_values = np.sort(df["theta"].unique())
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
                pred = xi_model(cur, fit_result)
                ax.scatter(cur["x"], cur["xi"], s=20, color=colors[float(vw)], marker=marker_map.get(float(h), "o"), alpha=0.85)
                ax.plot(cur["x"], pred, color=colors[float(vw)], lw=1.6, alpha=0.95)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$x = t_{\rm eff} H_*^\beta$")
        ax.set_ylabel(r"$\xi$")
    for ax in axes[len(theta_values):]:
        ax.axis("off")
    vw_handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
    vw_labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
    h_handles = [plt.Line2D([0], [0], color="black", marker=marker_map[h], linestyle="None") for h in h_values]
    h_labels = [rf"$H_*={h:g}$" for h in h_values]
    fig.legend(vw_handles + h_handles, vw_labels + h_labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(rf"Universal fit after quadratic $v_w$ warp, $\beta={beta:.4f}$", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outdir / "collapse_overlay_quadwarp_universal.png", dpi=dpi)
    plt.close(fig)


def plot_raw_xi_vs_betaH(df, fit_result, outdir: Path, dpi: int, beta: float):
    vw_values = np.sort(df["v_w"].unique())
    theta_values = np.sort(df["theta"].unique())
    h_values = np.sort(df["H"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    rmse_rows = []

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
                pred = xi_model(cur, fit_result)
                ax.scatter(cur["beta_over_H"], cur["xi"], s=22, color=colors[float(vw)], alpha=0.85)
                ax.plot(cur["beta_over_H"], pred, color=colors[float(vw)], lw=1.8)
                rmse_rows.append(
                    {
                        "H": float(h_value),
                        "theta": float(theta),
                        "v_w": float(vw),
                        "rel_rmse": rel_rmse(cur["xi"], pred),
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
        fig.suptitle(
            rf"Universal quadratic-warp fit in raw $\xi(\beta/H_*)$, $H_*={h_value:.1f}$, $\beta={beta:.4f}$",
            y=0.995,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        tag = str(float(h_value)).replace(".", "p")
        fig.savefig(outdir / f"xi_vs_betaH_quadwarp_universal_H{tag}.png", dpi=dpi)
        plt.close(fig)

    return rmse_rows


def plot_raw_xi_vs_betaH_separate(df, fit_result, outdir: Path, dpi: int):
    outdir.mkdir(parents=True, exist_ok=True)
    vw_values = np.sort(df["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    rows = []
    for h_value in np.sort(df["H"].unique()):
        sub_h = df[np.isclose(df["H"], float(h_value), atol=1.0e-12, rtol=0.0)].copy()
        for theta in np.sort(sub_h["theta"].unique()):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            for vw in vw_values:
                cur = sub[np.isclose(sub["v_w"], float(vw), atol=1.0e-12, rtol=0.0)].sort_values("beta_over_H").copy()
                if cur.empty:
                    continue
                pred = xi_model(cur, fit_result)
                ax.scatter(cur["beta_over_H"], cur["xi"], s=24, color=colors[float(vw)], alpha=0.9, label=rf"data $v_w={vw:.1f}$")
                ax.plot(cur["beta_over_H"], pred, color=colors[float(vw)], lw=2.0, label=rf"fit $v_w={vw:.1f}$")
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_xlabel(r"$\beta/H_*$")
            ax.set_ylabel(r"$\xi$")
            ax.set_title(rf"$H_*={h_value:.1f}$, $\theta={theta:.3f}$")
            ax.legend(frameon=False, fontsize=7, ncol=2)
            fig.tight_layout()
            h_tag = f"H{h_value:.1f}".replace(".", "p")
            theta_tag = f"theta_{theta:.10f}".replace(".", "p")
            path = outdir / f"xi_vs_betaH_quadwarp_universal_{h_tag}_{theta_tag}.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            rows.append({"H": float(h_value), "theta": float(theta), "file": str(path)})
    return rows


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("[load] reading lattice dataframe")
    df = load_lattice_dataframe(outdir, args.vw_folders, args.h_values)

    print("[warp] fitting per-vw quadratic log-time remapping")
    warp_params, warp_rows = quadwarp.fit_quadwarps(df, args.reference_vw, args.beta)

    print("[warp] applying warp and building collapsed coordinate")
    warped_df = apply_quadwarp(df, warp_params, args.beta)

    print("[fit] estimating tail amplitudes")
    finf_tail_df = collapse.fit_tail(warped_df, outdir, args.dpi)

    print("[fit] fitting universal curve")
    fit_result = collapse.fit_global(warped_df, finf_tail_df)
    global_payload = collapse.save_global_fit(fit_result, args.beta, outdir)

    bootstrap_payload = None
    if int(args.bootstrap) > 0:
        print(f"[bootstrap] running {int(args.bootstrap)} resamples")
        bootstrap_payload = collapse.bootstrap_global_fit(
            warped_df,
            fit_result,
            bootstrap_n=int(args.bootstrap),
            bootstrap_jobs=int(args.bootstrap_jobs),
            bootstrap_seed=int(args.bootstrap_seed),
        )
        save_json(outdir / "bootstrap_global_fit.json", bootstrap_payload)

    print("[plot] writing warped collapse and raw xi(beta/H*) plots")
    plot_collapse_overlay(warped_df, fit_result, outdir, args.dpi, args.beta)
    rmse_rows = plot_raw_xi_vs_betaH(warped_df, fit_result, outdir, args.dpi, args.beta)
    separate_rows = plot_raw_xi_vs_betaH_separate(
        warped_df, fit_result, outdir / "xi_vs_betaH_quadwarp_universal_separate", args.dpi
    )
    collapse.plot_residual_heatmap(warped_df, fit_result, outdir, args.dpi)

    index_path = outdir / "xi_vs_betaH_quadwarp_universal_index.csv"
    __import__("pandas").DataFrame(separate_rows).to_csv(index_path, index=False)

    summary = {
        "status": "ok",
        "beta": float(args.beta),
        "reference_vw": float(args.reference_vw),
        "h_values": [float(v) for v in args.h_values],
        "warp_rows": warp_rows,
        "global_fit": {
            "t_c": float(global_payload["t_c"]),
            "r": float(global_payload["r"]),
            "rel_rmse": float(global_payload["rel_rmse"]),
            "AIC": float(global_payload["AIC"]),
            "BIC": float(global_payload["BIC"]),
        },
        "rmse_by_vw": (
            __import__("pandas")
            .DataFrame(rmse_rows)
            .groupby("v_w", as_index=False)["rel_rmse"]
            .mean()
            .to_dict(orient="records")
        ),
        "rmse_by_H": (
            __import__("pandas")
            .DataFrame(rmse_rows)
            .groupby("H", as_index=False)["rel_rmse"]
            .mean()
            .to_dict(orient="records")
        ),
        "outputs": {
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
        payload = {
            "status": "error",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_json(outdir / "_error.json", payload)
        print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True))
        raise
