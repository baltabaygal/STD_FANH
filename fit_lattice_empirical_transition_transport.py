#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import collapse_and_fit_fanh_tosc as collapse
import fit_lattice_quadwarp_universal as uq
import plot_lattice_xi_vs_x_all_vw_quadwarped as quadwarp


OUTDIR = ROOT / "results_empirical_transition_transport_beta0"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(
        description="Learn an empirical universal transition function G(log x) from lattice v_w=0.9, then transport it to other v_w runs."
    )
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--reference-vw", type=float, default=0.9)
    p.add_argument("--grid-n", type=int, default=200)
    p.add_argument("--max-iter", type=int, default=12)
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


def theta_index_map(theta_values):
    return {float(th): i for i, th in enumerate(np.asarray(theta_values, dtype=np.float64))}


def xp_from_x(x):
    x = np.asarray(x, dtype=np.float64)
    return np.power(np.maximum(x / collapse.T_OSC, 1.0e-18), 1.5)


def initial_c_theta(train_df, theta_values):
    rows = []
    for theta in theta_values:
        sub = train_df[np.isclose(train_df["theta"], float(theta), atol=5.0e-4, rtol=0.0)].sort_values("x").copy()
        ntail = max(4, int(math.ceil(0.20 * len(sub))))
        tail = sub.tail(ntail).copy()
        xp = xp_from_x(tail["x"].to_numpy(dtype=np.float64))
        ratio = tail["xi"].to_numpy(dtype=np.float64) / np.maximum(xp, 1.0e-18)
        ratio = ratio[np.isfinite(ratio) & (ratio > 0.0)]
        if ratio.size == 0:
            rows.append(1.0e-6)
        else:
            rows.append(float(np.median(ratio)))
    return np.asarray(rows, dtype=np.float64)


def fit_empirical_transition(train_df: pd.DataFrame, grid_n: int, max_iter: int):
    theta_values = np.array(sorted(train_df["theta"].unique()), dtype=np.float64)
    idx_map = theta_index_map(theta_values)
    theta_index = np.array([idx_map[float(th)] for th in train_df["theta"].to_numpy(dtype=np.float64)], dtype=np.int64)
    x = train_df["x"].to_numpy(dtype=np.float64)
    xp = xp_from_x(x)
    y = train_df["xi"].to_numpy(dtype=np.float64)
    z = np.log(np.maximum(x, 1.0e-18))
    z_grid = np.linspace(float(np.min(z)), float(np.max(z)), int(grid_n))
    x_grid = np.exp(z_grid)
    c_theta = initial_c_theta(train_df, theta_values)
    iso = None
    history = []

    for _ in range(int(max_iter)):
        plateau = xp * c_theta[theta_index]
        resid = y - plateau
        y_max = float(max(np.nanmax(resid), np.nanmax(y), 1.0))
        iso = IsotonicRegression(increasing=False, y_min=0.0, y_max=y_max, out_of_bounds="clip")
        g_fit = iso.fit_transform(z, resid)

        new_c = np.zeros_like(c_theta)
        for i, theta in enumerate(theta_values):
            mask = theta_index == i
            y_minus_g = y[mask] - g_fit[mask]
            numer = float(np.sum(xp[mask] * y_minus_g))
            denom = float(np.sum(np.square(xp[mask])))
            c_val = numer / max(denom, 1.0e-18)
            new_c[i] = max(c_val, 1.0e-10)

        delta = float(np.max(np.abs(new_c - c_theta) / np.maximum(c_theta, 1.0e-18)))
        c_theta = new_c
        history.append(delta)
        if delta < 1.0e-5:
            break

    if iso is None:
        raise RuntimeError("Failed to fit isotonic transition.")

    g_grid = iso.predict(z_grid)
    g_grid = np.maximum(g_grid, 0.0)
    g_train = iso.predict(z)
    yfit = xp * c_theta[theta_index] + g_train
    f0_map = {float(th): float(train_df[np.isclose(train_df["theta"], float(th), atol=5.0e-4, rtol=0.0)]["F0"].iloc[0]) for th in theta_values}
    finf_map = {
        f"{float(th):.10f}": float(c_theta[i] * f0_map[float(th)] * f0_map[float(th)])
        for i, th in enumerate(theta_values)
    }
    return {
        "theta_values": theta_values,
        "c_theta": c_theta,
        "F_inf": finf_map,
        "x_grid": x_grid,
        "z_grid": z_grid,
        "g_grid": g_grid,
        "history": history,
        "train_pred": yfit,
        "train_rmse": rel_rmse(y, yfit),
    }


def g_interp(x, model):
    x = np.asarray(x, dtype=np.float64)
    z = np.log(np.maximum(x, 1.0e-18))
    return np.interp(z, model["z_grid"], model["g_grid"], left=model["g_grid"][0], right=model["g_grid"][-1])


def xi_empirical_model(df: pd.DataFrame, model, alpha: float = 0.0, ref_vw: float = 0.9):
    theta_values = np.asarray(model["theta_values"], dtype=np.float64)
    idx_map = theta_index_map(theta_values)
    theta_index = np.array([idx_map[float(th)] for th in df["theta"].to_numpy(dtype=np.float64)], dtype=np.int64)
    x = df["x"].to_numpy(dtype=np.float64)
    base = xp_from_x(x) * model["c_theta"][theta_index] + g_interp(x, model)
    factor = np.power(np.maximum(df["v_w"].to_numpy(dtype=np.float64) / float(ref_vw), 1.0e-18), float(alpha))
    return base * factor


def fit_alpha(df: pd.DataFrame, model, ref_vw: float):
    def objective(alpha):
        pred = xi_empirical_model(df, model, float(alpha), ref_vw)
        return rel_rmse(df["xi"].to_numpy(dtype=np.float64), pred)

    res = minimize_scalar(objective, bounds=(-1.0, 1.0), method="bounded", options={"xatol": 1.0e-4})
    alpha = float(res.x) if res.success else 0.0
    return {
        "alpha": alpha,
        "rel_rmse": float(objective(alpha)),
        "success": bool(res.success),
        "message": str(getattr(res, "message", "")),
    }


def rmse_tables(df: pd.DataFrame, pred_col: str):
    rows_theta = []
    rows_vw = []
    rows_h = []
    for theta, sub in df.groupby("theta", sort=True):
        rows_theta.append({"theta": float(theta), "rel_rmse": rel_rmse(sub["xi"], sub[pred_col])})
    for vw, sub in df.groupby("v_w", sort=True):
        rows_vw.append({"v_w": float(vw), "rel_rmse": rel_rmse(sub["xi"], sub[pred_col])})
    for h, sub in df.groupby("H", sort=True):
        rows_h.append({"H": float(h), "rel_rmse": rel_rmse(sub["xi"], sub[pred_col])})
    return rows_theta, rows_vw, rows_h


def plot_transition(model, train_df: pd.DataFrame, outdir: Path, dpi: int):
    theta_values = np.asarray(model["theta_values"], dtype=np.float64)
    idx_map = theta_index_map(theta_values)
    theta_index = np.array([idx_map[float(th)] for th in train_df["theta"].to_numpy(dtype=np.float64)], dtype=np.int64)
    plateau = xp_from_x(train_df["x"].to_numpy(dtype=np.float64)) * model["c_theta"][theta_index]
    transition = train_df["xi"].to_numpy(dtype=np.float64) - plateau
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.scatter(train_df["x"], transition, s=12, alpha=0.35, color="tab:green", label=r"$v_w=0.9$ residual transition")
    ax.plot(model["x_grid"], model["g_grid"], color="black", lw=2.2, label=r"learned $G(x)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$G(x)$")
    ax.set_title(r"Empirical universal transition learned from $v_w=0.9$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "empirical_transition_G.png", dpi=dpi)
    plt.close(fig)


def plot_collapse_overlay(df: pd.DataFrame, pred_col: str, outdir: Path, dpi: int, beta: float, title: str, filename: str):
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
                ax.plot(cur["x"], cur[pred_col], color=colors[float(vw)], lw=1.6, alpha=0.95)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$x=t_{\rm eff} H_*^\beta$")
        ax.set_ylabel(r"$\xi$")
    for ax in axes[len(theta_values):]:
        ax.axis("off")
    vw_handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
    vw_labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
    h_handles = [plt.Line2D([0], [0], color="black", marker=marker_map[h], linestyle="None") for h in h_values]
    h_labels = [rf"$H_*={h:g}$" for h in h_values]
    fig.legend(vw_handles + h_handles, vw_labels + h_labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(title + rf", $\beta={beta:.4f}$", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outdir / filename, dpi=dpi)
    plt.close(fig)


def plot_raw_xi_vs_betaH(df: pd.DataFrame, pred_col: str, outdir: Path, dpi: int, beta: float, tag: str):
    vw_values = np.sort(df["v_w"].unique())
    theta_values = choose_theta_subset(df["theta"].unique())
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
                ax.scatter(cur["beta_over_H"], cur["xi"], s=22, color=colors[float(vw)], alpha=0.85)
                ax.plot(cur["beta_over_H"], cur[pred_col], color=colors[float(vw)], lw=1.8)
                rmse_rows.append({"H": float(h_value), "theta": float(theta), "v_w": float(vw), "rel_rmse": rel_rmse(cur["xi"], cur[pred_col])})
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
        fig.suptitle(rf"Empirical $G(x)$ transport in raw $\xi(\beta/H_*)$, $H_*={h_value:.1f}$, $\beta={beta:.4f}$, {tag}", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        htag = str(float(h_value)).replace(".", "p")
        fig.savefig(outdir / f"xi_vs_betaH_empirical_{tag}_H{htag}.png", dpi=dpi)
        plt.close(fig)
    return rmse_rows


def plot_residual_vs_theta(df: pd.DataFrame, pred_col: str, outdir: Path, dpi: int, filename: str, title: str):
    stats = []
    for theta, sub in df.groupby("theta", sort=True):
        resid = (sub[pred_col].to_numpy(dtype=np.float64) - sub["xi"].to_numpy(dtype=np.float64)) / np.maximum(
            sub["xi"].to_numpy(dtype=np.float64), 1.0e-12
        )
        stats.append({"theta": float(theta), "mean_abs_resid": float(np.mean(np.abs(resid))), "mean_signed_resid": float(np.mean(resid))})
    stat_df = pd.DataFrame(stats).sort_values("theta").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(stat_df["theta"], stat_df["mean_abs_resid"], "o-", color="tab:blue", label="mean |frac resid|")
    ax.plot(stat_df["theta"], stat_df["mean_signed_resid"], "s--", color="tab:red", label="mean frac resid")
    ax.axhline(0.0, color="black", lw=1.0, alpha=0.7)
    ax.grid(alpha=0.25)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("residual")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=dpi)
    plt.close(fig)
    return stat_df


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("[load] reading lattice dataframe")
    df = uq.load_lattice_dataframe(outdir, args.vw_folders, args.h_values)

    print("[warp] fitting per-vw quadratic time warp")
    warp_params, warp_rows = quadwarp.fit_quadwarps(df, args.reference_vw, args.beta)

    print("[warp] applying quadratic warp")
    warped_df = uq.apply_quadwarp(df, warp_params, args.beta)

    ref_mask = np.isclose(warped_df["v_w"], float(args.reference_vw), atol=1.0e-12, rtol=0.0)
    train_df = warped_df[ref_mask].copy().sort_values(["H", "theta", "x"]).reset_index(drop=True)
    if train_df.empty:
        raise RuntimeError(f"No training rows found for reference v_w={args.reference_vw:.3f}")

    print("[fit] learning empirical universal transition G(log x) from reference subset")
    model = fit_empirical_transition(train_df, args.grid_n, args.max_iter)

    print("[eval] transporting learned G(x) to all v_w without extra amplitude")
    warped_df["xi_empirical"] = xi_empirical_model(warped_df, model, alpha=0.0, ref_vw=args.reference_vw)

    print("[eval] fitting residual global amplitude v_w^alpha on top of empirical G(x)")
    alpha_fit = fit_alpha(warped_df, model, args.reference_vw)
    warped_df["xi_empirical_alpha"] = xi_empirical_model(
        warped_df, model, alpha=alpha_fit["alpha"], ref_vw=args.reference_vw
    )

    print("[plot] writing diagnostics")
    plot_transition(model, train_df, outdir, args.dpi)
    plot_collapse_overlay(
        warped_df,
        "xi_empirical",
        outdir,
        args.dpi,
        float(args.beta),
        r"Empirical universal transition $G(x)$, no extra amplitude",
        "collapse_overlay_empirical_noalpha.png",
    )
    plot_collapse_overlay(
        warped_df,
        "xi_empirical_alpha",
        outdir,
        args.dpi,
        float(args.beta),
        rf"Empirical universal transition $G(x)$ with $(v_w/{args.reference_vw:.1f})^\alpha$",
        "collapse_overlay_empirical_alpha.png",
    )
    rmse_raw_noalpha = plot_raw_xi_vs_betaH(warped_df, "xi_empirical", outdir, args.dpi, float(args.beta), "noalpha")
    rmse_raw_alpha = plot_raw_xi_vs_betaH(
        warped_df, "xi_empirical_alpha", outdir, args.dpi, float(args.beta), "alpha"
    )
    theta_stats_noalpha = plot_residual_vs_theta(
        warped_df,
        "xi_empirical",
        outdir,
        args.dpi,
        "residual_vs_theta_noalpha.png",
        r"Empirical $G(x)$ transport residuals vs $\theta$, no extra amplitude",
    )
    theta_stats_alpha = plot_residual_vs_theta(
        warped_df,
        "xi_empirical_alpha",
        outdir,
        args.dpi,
        "residual_vs_theta_alpha.png",
        r"Empirical $G(x)$ transport residuals vs $\theta$, with extra amplitude",
    )

    theta_rmse_noalpha, vw_rmse_noalpha, h_rmse_noalpha = rmse_tables(warped_df, "xi_empirical")
    theta_rmse_alpha, vw_rmse_alpha, h_rmse_alpha = rmse_tables(warped_df, "xi_empirical_alpha")

    predictions_path = outdir / "predictions.csv"
    warped_df.to_csv(predictions_path, index=False)
    theta_stats_noalpha.to_csv(outdir / "theta_stats_noalpha.csv", index=False)
    theta_stats_alpha.to_csv(outdir / "theta_stats_alpha.csv", index=False)

    summary = {
        "status": "ok",
        "beta": float(args.beta),
        "reference_vw": float(args.reference_vw),
        "t_osc": float(collapse.T_OSC),
        "warp_params": warp_params,
        "warp_rows": warp_rows,
        "empirical_transition": {
            "train_rel_rmse": float(model["train_rmse"]),
            "theta_values": [float(v) for v in model["theta_values"]],
            "F_inf": model["F_inf"],
            "c_theta": [float(v) for v in model["c_theta"]],
            "history": [float(v) for v in model["history"]],
        },
        "transport_noalpha": {
            "global_rel_rmse": rel_rmse(warped_df["xi"], warped_df["xi_empirical"]),
            "rmse_by_theta": theta_rmse_noalpha,
            "rmse_by_vw": vw_rmse_noalpha,
            "rmse_by_H": h_rmse_noalpha,
        },
        "transport_alpha": {
            "alpha": float(alpha_fit["alpha"]),
            "alpha_rel_rmse": float(alpha_fit["rel_rmse"]),
            "global_rel_rmse": rel_rmse(warped_df["xi"], warped_df["xi_empirical_alpha"]),
            "rmse_by_theta": theta_rmse_alpha,
            "rmse_by_vw": vw_rmse_alpha,
            "rmse_by_H": h_rmse_alpha,
            "fit_success": bool(alpha_fit["success"]),
            "fit_message": alpha_fit["message"],
        },
        "raw_plot_rmse_noalpha": rmse_raw_noalpha,
        "raw_plot_rmse_alpha": rmse_raw_alpha,
        "outputs": {
            "transition_plot": str(outdir / "empirical_transition_G.png"),
            "collapse_noalpha": str(outdir / "collapse_overlay_empirical_noalpha.png"),
            "collapse_alpha": str(outdir / "collapse_overlay_empirical_alpha.png"),
            "raw_H1p0_noalpha": str(outdir / "xi_vs_betaH_empirical_noalpha_H1p0.png"),
            "raw_H1p5_noalpha": str(outdir / "xi_vs_betaH_empirical_noalpha_H1p5.png"),
            "raw_H2p0_noalpha": str(outdir / "xi_vs_betaH_empirical_noalpha_H2p0.png"),
            "predictions": str(predictions_path),
        },
    }
    save_json(outdir / "final_summary.json", summary)
    print(json.dumps(to_native(summary), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        outdir = OUTDIR.resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "error",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_json(outdir / "_error.json", payload)
        print(json.dumps(payload, sort_keys=True))
        raise
