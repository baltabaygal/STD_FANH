#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fit_lattice_quadwarp_universal as uq


ROOT = Path(__file__).resolve().parent
MODEL_JSON = ROOT / "results_tosc_lattice_vw0p9_H1p0H1p5H2p0" / "collapse_and_fit_fanh" / "global_fit.json"
OUTDIR = ROOT / "results_pointwise_shift_from_vw0p9_baseline"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]
T_OSC = 1.5


def parse_args():
    p = argparse.ArgumentParser(
        description="Infer the pointwise multiplicative time shift s = x_eff/x_base required for each lattice point under the frozen vw=0.9 baseline model."
    )
    p.add_argument("--model-json", type=str, default=str(MODEL_JSON))
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--outdir", type=str, default=str(OUTDIR))
    p.add_argument("--x-min", type=float, default=1.0e-4)
    p.add_argument("--x-max", type=float, default=1.0e3)
    p.add_argument("--x-grid", type=int, default=6000)
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
    finf = np.array([float(payload["F_inf"][f"{theta:.10f}"]["value"]) for theta in theta_values], dtype=np.float64)
    return {
        "beta": float(payload["beta"]),
        "t_c": float(payload["t_c"]),
        "r": float(payload["r"]),
        "theta_values": theta_values,
        "finf": finf,
        "source": str(path.resolve()),
    }


def baseline_curve(theta_values: np.ndarray, finf: np.ndarray, theta: float, f0: float, t_c: float, r: float, x_min: float, x_max: float, x_grid: int):
    idx = int(np.argmin(np.abs(theta_values - float(theta))))
    x = np.geomspace(float(x_min), float(x_max), int(x_grid))
    y = np.power(x / T_OSC, 1.5) * float(finf[idx]) / max(float(f0) * float(f0), 1.0e-18) + 1.0 / (
        1.0 + np.power(x / max(float(t_c), 1.0e-12), float(r))
    )
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    x = x[keep]
    y = y[keep]
    order = np.argsort(y)
    y = y[order]
    x = x[order]
    yuniq, idxuniq = np.unique(y, return_index=True)
    return yuniq, np.log(x[idxuniq])


def baseline_predict(theta_values: np.ndarray, finf: np.ndarray, theta: np.ndarray, f0: np.ndarray, x: np.ndarray, t_c: float, r: float):
    theta = np.asarray(theta, dtype=np.float64)
    f0 = np.asarray(f0, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    theta_index = np.array([int(np.argmin(np.abs(theta_values - float(th)))) for th in theta], dtype=np.int64)
    transient = 1.0 / (1.0 + np.power(x / max(float(t_c), 1.0e-12), float(r)))
    return np.power(x / T_OSC, 1.5) * finf[theta_index] / np.maximum(f0 * f0, 1.0e-18) + transient


def infer_pointwise_shift(df: pd.DataFrame, model: dict, x_min: float, x_max: float, x_grid: int) -> pd.DataFrame:
    out = df.copy()
    out["x_base"] = out["tp"].to_numpy(dtype=np.float64) * np.power(out["H"].to_numpy(dtype=np.float64), float(model["beta"]))
    cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
    x_eff = np.full(len(out), np.nan, dtype=np.float64)
    in_range = np.zeros(len(out), dtype=bool)

    for i, row in enumerate(out.itertuples(index=False)):
        theta_key = float(row.theta)
        f0_key = float(row.F0)
        key = (theta_key, f0_key)
        if key not in cache:
            cache[key] = baseline_curve(
                model["theta_values"], model["finf"], theta_key, f0_key, model["t_c"], model["r"], x_min, x_max, x_grid
            )
        ygrid, logxgrid = cache[key]
        target = float(row.xi)
        if target < float(ygrid[0]) or target > float(ygrid[-1]):
            continue
        x_eff[i] = float(np.exp(np.interp(target, ygrid, logxgrid)))
        in_range[i] = True

    out["x_eff_pointwise"] = x_eff
    out["s_pointwise"] = out["x_eff_pointwise"] / np.maximum(out["x_base"], 1.0e-18)
    out["tp_eff_pointwise"] = out["s_pointwise"] * out["tp"]
    out["in_range"] = in_range
    out["xi_model_pointwise"] = np.nan
    valid = out["in_range"].to_numpy(dtype=bool)
    if np.any(valid):
        out.loc[valid, "xi_model_pointwise"] = baseline_predict(
            model["theta_values"],
            model["finf"],
            out.loc[valid, "theta"].to_numpy(dtype=np.float64),
            out.loc[valid, "F0"].to_numpy(dtype=np.float64),
            out.loc[valid, "x_eff_pointwise"].to_numpy(dtype=np.float64),
            model["t_c"],
            model["r"],
        )
        out.loc[valid, "interp_frac_resid"] = (
            out.loc[valid, "xi_model_pointwise"].to_numpy(dtype=np.float64) - out.loc[valid, "xi"].to_numpy(dtype=np.float64)
        ) / np.maximum(out.loc[valid, "xi"].to_numpy(dtype=np.float64), 1.0e-12)
    return out


def plot_shift_vs_tp(df: pd.DataFrame, outdir: Path, dpi: int):
    theta_values = choose_theta_subset(df["theta"].unique())
    vw_values = np.sort(df["v_w"].unique())
    h_values = np.sort(df["H"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}

    rows = []
    for h_value in h_values:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
        axes = axes.ravel()
        sub_h = df[np.isclose(df["H"], float(h_value), atol=1.0e-12, rtol=0.0) & df["in_range"]].copy()
        for ax, theta in zip(axes, theta_values):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
            for vw in vw_values:
                cur = sub[np.isclose(sub["v_w"], float(vw), atol=1.0e-12, rtol=0.0)].sort_values("tp").copy()
                if cur.empty:
                    continue
                ax.plot(cur["tp"], cur["s_pointwise"], "o-", ms=3.6, lw=1.4, color=colors[float(vw)], alpha=0.95)
                rows.append(
                    {
                        "H": float(h_value),
                        "theta": float(theta),
                        "v_w": float(vw),
                        "s_min": float(cur["s_pointwise"].min()),
                        "s_max": float(cur["s_pointwise"].max()),
                        "s_median": float(cur["s_pointwise"].median()),
                        "tp_min": float(cur["tp"].min()),
                        "tp_max": float(cur["tp"].max()),
                    }
                )
            ax.axhline(1.0, color="black", lw=1.0, ls="--")
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$t_p$")
            ax.set_ylabel(r"$s = x_{\rm eff}/x$")
        for ax in axes[len(theta_values):]:
            ax.axis("off")
        handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
        labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
        fig.suptitle(rf"Pointwise shift from frozen $v_w=0.9$ baseline, $H_*={h_value:.1f}$", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        tag = str(float(h_value)).replace(".", "p")
        fig.savefig(outdir / f"s_vs_tp_H{tag}.png", dpi=dpi)
        plt.close(fig)
    return rows


def plot_summary(df: pd.DataFrame, outdir: Path, dpi: int):
    sub = df[df["in_range"]].copy()
    vw_values = np.sort(sub["v_w"].unique())
    h_values = np.sort(sub["H"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    marker_map = {1.0: "s", 1.5: "^", 2.0: "D", 0.5: "o"}

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    for h in h_values:
        for vw in vw_values:
            cur = sub[np.isclose(sub["H"], float(h), atol=1.0e-12, rtol=0.0) & np.isclose(sub["v_w"], float(vw), atol=1.0e-12, rtol=0.0)]
            if cur.empty:
                continue
            ax.scatter(cur["tp"], cur["s_pointwise"], s=26, color=colors[float(vw)], marker=marker_map.get(float(h), "o"), alpha=0.7)
    ax.axhline(1.0, color="black", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$t_p$")
    ax.set_ylabel(r"$s = x_{\rm eff}/x$")
    ax.set_title(r"Pointwise inferred shift from frozen $v_w=0.9$ baseline")
    ax.grid(alpha=0.25)
    vw_handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
    vw_labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
    h_handles = [plt.Line2D([0], [0], color="black", marker=marker_map[h], linestyle="None") for h in h_values]
    h_labels = [rf"$H_*={h:g}$" for h in h_values]
    ax.legend(vw_handles + h_handles, vw_labels + h_labels, frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "s_vs_tp_all.png", dpi=dpi)
    plt.close(fig)

    # Aggregate by tp quartiles
    tp_bins = np.quantile(sub["tp"], [0.0, 0.25, 0.5, 0.75, 1.0])
    tp_bins = np.unique(tp_bins)
    if len(tp_bins) >= 3:
        labels = []
        rows = []
        for lo, hi in zip(tp_bins[:-1], tp_bins[1:]):
            mask = (sub["tp"] >= lo) & (sub["tp"] <= hi if hi == tp_bins[-1] else sub["tp"] < hi)
            labels.append((lo, hi))
            curbin = sub.loc[mask].copy()
            for vw in vw_values:
                cur = curbin[np.isclose(curbin["v_w"], float(vw), atol=1.0e-12, rtol=0.0)]
                if cur.empty:
                    continue
                rows.append({"tp_bin_lo": float(lo), "tp_bin_hi": float(hi), "v_w": float(vw), "s_mean": float(cur["s_pointwise"].mean())})
        bdf = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(7.8, 5.4))
        for lo, hi in labels:
            cur = bdf[np.isclose(bdf["tp_bin_lo"], lo) & np.isclose(bdf["tp_bin_hi"], hi)].sort_values("v_w")
            if cur.empty:
                continue
            ax.plot(cur["v_w"], cur["s_mean"], "o-", lw=1.7, ms=4.5, label=rf"$t_p\in[{lo:.3g},{hi:.3g}]$")
        ax.axhline(1.0, color="black", lw=1.0, ls="--")
        ax.set_xlabel(r"$v_w$")
        ax.set_ylabel(r"mean pointwise $s$")
        ax.set_title(r"Pointwise shift averaged in $t_p$ bins")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "s_vs_vw_tp_bins.png", dpi=dpi)
        plt.close(fig)
        return bdf
    return pd.DataFrame()


def plot_collapse_overlay_pointwise(df: pd.DataFrame, model: dict, outdir: Path, dpi: int):
    sub = df[df["in_range"]].copy()
    theta_values = choose_theta_subset(sub["theta"].unique())
    vw_values = np.sort(sub["v_w"].unique())
    h_values = np.sort(sub["H"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    marker_map = {1.0: "s", 1.5: "^", 2.0: "D", 0.5: "o"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, theta in zip(axes, theta_values):
        cur_theta = sub[np.isclose(sub["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
        if cur_theta.empty:
            ax.axis("off")
            continue
        xlo = float(np.min(cur_theta["x_eff_pointwise"]))
        xhi = float(np.max(cur_theta["x_eff_pointwise"]))
        xfit = np.geomspace(xlo, xhi, 300)
        f0_ref = float(cur_theta.sort_values("H").iloc[0]["F0"])
        yfit = baseline_predict(
            model["theta_values"],
            model["finf"],
            np.full_like(xfit, float(theta), dtype=np.float64),
            np.full_like(xfit, f0_ref, dtype=np.float64),
            xfit,
            model["t_c"],
            model["r"],
        )
        ax.plot(xfit, yfit, color="black", lw=1.8, label="frozen baseline")
        for vw in vw_values:
            for h in h_values:
                cur = cur_theta[
                    np.isclose(cur_theta["v_w"], float(vw), atol=1.0e-12, rtol=0.0)
                    & np.isclose(cur_theta["H"], float(h), atol=1.0e-12, rtol=0.0)
                ].sort_values("x_eff_pointwise")
                if cur.empty:
                    continue
                ax.scatter(
                    cur["x_eff_pointwise"],
                    cur["xi"],
                    s=22,
                    color=colors[float(vw)],
                    marker=marker_map.get(float(h), "o"),
                    alpha=0.9,
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$x_{\rm eff}$")
        ax.set_ylabel(r"$\xi$")
    for ax in axes[len(theta_values):]:
        ax.axis("off")
    vw_handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
    vw_labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
    h_handles = [plt.Line2D([0], [0], color="black", marker=marker_map[h], linestyle="None") for h in h_values]
    h_labels = [rf"$H_*={h:g}$" for h in h_values]
    base_handle = plt.Line2D([0], [0], color="black", lw=2.0)
    fig.legend([base_handle] + vw_handles + h_handles, ["frozen baseline"] + vw_labels + h_labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(r"Pointwise-shift collapse onto frozen $v_w=0.9$ baseline", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outdir / "collapse_overlay_pointwise_shift.png", dpi=dpi)
    plt.close(fig)


def plot_raw_xi_vs_betaH_pointwise(df: pd.DataFrame, outdir: Path, dpi: int):
    sub = df[df["in_range"]].copy()
    theta_values = choose_theta_subset(sub["theta"].unique())
    vw_values = np.sort(sub["v_w"].unique())
    h_values = np.sort(sub["H"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}

    rows = []
    for h_value in h_values:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
        axes = axes.ravel()
        sub_h = sub[np.isclose(sub["H"], float(h_value), atol=1.0e-12, rtol=0.0)].copy()
        for ax, theta in zip(axes, theta_values):
            cur_theta = sub_h[np.isclose(sub_h["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
            for vw in vw_values:
                cur = cur_theta[np.isclose(cur_theta["v_w"], float(vw), atol=1.0e-12, rtol=0.0)].sort_values("beta_over_H")
                if cur.empty:
                    continue
                ax.scatter(cur["beta_over_H"], cur["xi"], s=24, color=colors[float(vw)], alpha=0.9)
                ax.plot(cur["beta_over_H"], cur["xi_model_pointwise"], color=colors[float(vw)], lw=1.8)
                rows.append(
                    {
                        "H": float(h_value),
                        "theta": float(theta),
                        "v_w": float(vw),
                        "rel_rmse_interp": rel_rmse(cur["xi"], cur["xi_model_pointwise"]),
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
        fig.suptitle(rf"Pointwise-shift fit in raw $\xi(\beta/H_*)$, $H_*={h_value:.1f}$", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        tag = str(float(h_value)).replace(".", "p")
        fig.savefig(outdir / f"xi_vs_betaH_pointwise_shift_H{tag}.png", dpi=dpi)
        plt.close(fig)
    return rows


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    model = load_model(Path(args.model_json).resolve())
    print("[load] reading lattice dataframe")
    df = uq.load_lattice_dataframe(outdir, args.vw_folders, args.h_values)
    print("[invert] inferring pointwise shift")
    pred = infer_pointwise_shift(df, model, args.x_min, args.x_max, args.x_grid)
    pred.to_csv(outdir / "pointwise_shift_table.csv", index=False)

    print("[plot] writing shift plots")
    by_panel_rows = plot_shift_vs_tp(pred, outdir, args.dpi)
    binned_df = plot_summary(pred, outdir, args.dpi)
    print("[plot] writing fit-visualization overlays")
    plot_collapse_overlay_pointwise(pred, model, outdir, args.dpi)
    fit_rows = plot_raw_xi_vs_betaH_pointwise(pred, outdir, args.dpi)
    if not binned_df.empty:
        binned_df.to_csv(outdir / "s_vs_vw_tp_bins.csv", index=False)

    valid = pred[pred["in_range"]].copy()
    summary = {
        "status": "ok",
        "frozen_model_source": model["source"],
        "beta": float(model["beta"]),
        "t_c": float(model["t_c"]),
        "r": float(model["r"]),
        "n_points_total": int(len(pred)),
        "n_points_inverted": int(valid.shape[0]),
        "inversion_fraction": float(valid.shape[0] / max(len(pred), 1)),
        "s_range": [float(valid["s_pointwise"].min()), float(valid["s_pointwise"].max())] if not valid.empty else None,
        "interp_rel_rmse": rel_rmse(valid["xi"], valid["xi_model_pointwise"]) if not valid.empty else None,
        "interp_median_abs_frac_resid": float(np.median(np.abs(valid["interp_frac_resid"]))) if not valid.empty else None,
        "s_by_vw": [
            {
                "v_w": float(vw),
                "s_min": float(sub["s_pointwise"].min()),
                "s_median": float(sub["s_pointwise"].median()),
                "s_max": float(sub["s_pointwise"].max()),
            }
            for vw, sub in valid.groupby("v_w", sort=True)
        ],
        "s_by_H": [
            {
                "H": float(h),
                "s_min": float(sub["s_pointwise"].min()),
                "s_median": float(sub["s_pointwise"].median()),
                "s_max": float(sub["s_pointwise"].max()),
            }
            for h, sub in valid.groupby("H", sort=True)
        ],
        "outputs": {
            "table": str(outdir / "pointwise_shift_table.csv"),
            "all_scatter": str(outdir / "s_vs_tp_all.png"),
            "by_H": [str(outdir / f"s_vs_tp_H{str(float(h)).replace('.', 'p')}.png") for h in sorted(valid["H"].unique())],
            "tpbin_summary": str(outdir / "s_vs_vw_tp_bins.png"),
            "collapse_overlay": str(outdir / "collapse_overlay_pointwise_shift.png"),
            "raw_fit_by_H": [str(outdir / f"xi_vs_betaH_pointwise_shift_H{str(float(h)).replace('.', 'p')}.png") for h in sorted(valid["H"].unique())],
        },
        "panel_rows": by_panel_rows,
        "fit_rows": fit_rows,
    }
    save_json(outdir / "final_summary.json", summary)
    print(json.dumps(to_native(summary), sort_keys=True))


if __name__ == "__main__":
    main()
