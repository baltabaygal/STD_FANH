#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

import fit_lattice_fixed_ode_amplitudes as shared
import fit_lattice_quadwarp_universal as uq
from fit_lattice_broken_powerlaw_knob_h0 import h_theta, xi_broken_power


ROOT = Path(__file__).resolve().parent
BASELINE_SUMMARY = ROOT / "results_lattice_broken_powerlaw_knob_h0_betafree_H1p0H1p5H2p0_vw0p9" / "final_summary.json"
OUTDIR = ROOT / "results_broken_power_shift_from_vw0p9_H1p0H1p5H2p0"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]
REF_VW = 0.9


def parse_args():
    p = argparse.ArgumentParser(
        description="Freeze a vw=0.9 broken-power baseline and fit an anchored pointwise time shift s(tp,vw)=1+A[(0.9/vw)^m-1]tp^p across all vw."
    )
    p.add_argument("--baseline-summary", type=str, default=str(BASELINE_SUMMARY))
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--ode-summary", type=str, default=str(shared.ODE_SUMMARY_DEFAULT))
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--outdir", type=str, default=str(OUTDIR))
    return p.parse_args()


def save_json(path: Path, payload):
    path.write_text(json.dumps(shared.to_native(payload), indent=2, sort_keys=True))


def error_exit(outdir: Path, exc: Exception):
    payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "_error.json", payload)
    print(json.dumps(payload, sort_keys=True))


def rel_rmse(y, yfit):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(yfit) & (y > 0.0)
    if np.count_nonzero(mask) == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square((yfit[mask] - y[mask]) / np.maximum(y[mask], 1.0e-12)))))


def load_baseline(path: Path):
    payload = json.loads(path.read_text())
    case = payload["cases"]["allvw_c1_rvw_h0gammavw"]
    vw_key = "0.9"
    return {
        "source": str(path.resolve()),
        "beta": float(case["beta"]),
        "t_osc": 1.5,
        "h0": float(case["h0"]),
        "r": float(case["r_by_vw"][vw_key]),
        "tc0": float(case["tc0_by_vw"][vw_key]),
        "gamma": float(case["gamma_by_vw"][vw_key]),
        "theta_values": np.asarray(payload["theta_values"], dtype=np.float64),
    }


def shift_model(tp: np.ndarray, vw: np.ndarray, A: float, m: float, p: float):
    tp = np.asarray(tp, dtype=np.float64)
    vw = np.asarray(vw, dtype=np.float64)
    return 1.0 + A * (np.power(REF_VW / np.maximum(vw, 1.0e-18), m) - 1.0) * np.power(np.maximum(tp, 1.0e-18), p)


def build_meta(df, theta_values: np.ndarray, ode: dict):
    theta_index = {float(theta): i for i, theta in enumerate(theta_values)}
    theta_idx = np.array([theta_index[float(theta)] for theta in df["theta"].to_numpy(dtype=np.float64)], dtype=np.int64)
    return {
        "theta_values": np.asarray(theta_values, dtype=np.float64),
        "theta_idx": theta_idx,
        "theta": df["theta"].to_numpy(dtype=np.float64),
        "v_w": df["v_w"].to_numpy(dtype=np.float64),
        "tp": df["tp"].to_numpy(dtype=np.float64),
        "H": df["H"].to_numpy(dtype=np.float64),
        "beta_over_H": df["beta_over_H"].to_numpy(dtype=np.float64),
        "xi": df["xi"].to_numpy(dtype=np.float64),
        "F0": ode["F0"][theta_idx],
        "F0_sq": np.maximum(np.square(ode["F0"][theta_idx]), 1.0e-18),
        "F_inf": ode["F_inf"][theta_idx],
        "h_theta": h_theta(df["theta"].to_numpy(dtype=np.float64)),
    }


def baseline_predict(meta, baseline: dict, A: float, m: float, p: float):
    s = shift_model(meta["tp"], meta["v_w"], A, m, p)
    x_base = meta["tp"] * np.power(meta["H"], float(baseline["beta"]))
    x_eff = x_base * s
    tc_theta = baseline["tc0"] * np.power(np.maximum(meta["h_theta"] + float(baseline["h0"]), 1.0e-18), baseline["gamma"])
    xi_scale = np.power(
        np.maximum((2.0 * x_eff) / max(3.0 * float(baseline["t_osc"]), 1.0e-18), 1.0e-18),
        1.5,
    )
    u = meta["F_inf"] * xi_scale / np.maximum(meta["F0_sq"] * np.power(tc_theta, 1.5), 1.0e-18)
    xi_fit = xi_broken_power(u, np.full_like(u, baseline["r"]))
    return x_base, x_eff, s, tc_theta, u, xi_fit


def fit_shift(meta, baseline: dict):
    x0 = np.asarray([0.6, 0.3, -1.3], dtype=np.float64)
    lower = np.asarray([-10.0, -5.0, -5.0], dtype=np.float64)
    upper = np.asarray([10.0, 5.0, 5.0], dtype=np.float64)

    def resid(par: np.ndarray) -> np.ndarray:
        A, m, p = map(float, par)
        _, _, s, _, _, xi_fit = baseline_predict(meta, baseline, A, m, p)
        if np.any(s <= 0.0):
            return np.full_like(meta["xi"], 1.0e6, dtype=np.float64)
        return (xi_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)

    huber = least_squares(resid, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=60000)
    final = least_squares(resid, huber.x, bounds=(lower, upper), loss="linear", max_nfev=60000)
    A, m, p = map(float, final.x)
    x_base, x_eff, s, tc_theta, u, xi_fit = baseline_predict(meta, baseline, A, m, p)
    frac_resid = (xi_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    aic, bic = shared.aic_bic(frac_resid, len(final.x))
    return {
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "A": A,
        "m": m,
        "p": p,
        "AIC": float(aic),
        "BIC": float(bic),
        "rel_rmse": rel_rmse(meta["xi"], xi_fit),
        "n_points": int(meta["xi"].size),
        "n_params": int(len(final.x)),
        "x_base": x_base,
        "x_eff": x_eff,
        "s_fit": s,
        "tc_theta": tc_theta,
        "u_fit": u,
        "xi_fit": xi_fit,
        "frac_resid": frac_resid,
        "baseline_rel_rmse": rel_rmse(meta["xi"], baseline_predict(meta, baseline, 0.0, 0.0, 0.0)[5]),
    }


def rmse_tables(df_pred):
    rows = []
    by_h = {}
    by_vw = {}
    for h in sorted(df_pred["H"].unique()):
        sub_h = df_pred[np.isclose(df_pred["H"], float(h), atol=1.0e-12)].copy()
        by_h[f"{float(h):.1f}"] = rel_rmse(sub_h["xi"], sub_h["xi_fit"])
    for vw in sorted(df_pred["v_w"].unique()):
        sub = df_pred[np.isclose(df_pred["v_w"], float(vw), atol=1.0e-12)].copy()
        by_vw[f"{float(vw):.1f}"] = rel_rmse(sub["xi"], sub["xi_fit"])
    for h in sorted(df_pred["H"].unique()):
        sub_h = df_pred[np.isclose(df_pred["H"], float(h), atol=1.0e-12)].copy()
        for theta in sorted(sub_h["theta"].unique()):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=1.0e-8)].copy()
            for vw in sorted(sub["v_w"].unique()):
                cur = sub[np.isclose(sub["v_w"], float(vw), atol=1.0e-12)]
                rows.append({"H": float(h), "theta": float(theta), "v_w": float(vw), "rel_rmse": rel_rmse(cur["xi"], cur["xi_fit"])})
    return rows, by_h, by_vw


def plot_collapse(df_pred, outpath: Path, dpi: int):
    theta_values = np.sort(df_pred["theta"].unique())
    vw_values = np.sort(df_pred["v_w"].unique())
    h_values = np.sort(df_pred["H"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {float(vw): cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    marker_map = {1.0: "s", 1.5: "^", 2.0: "D", 0.5: "o"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, theta in zip(axes, theta_values):
        sub = df_pred[np.isclose(df_pred["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
        for vw in vw_values:
            for h in h_values:
                cur = sub[
                    np.isclose(sub["v_w"], float(vw), atol=1.0e-12, rtol=0.0)
                    & np.isclose(sub["H"], float(h), atol=1.0e-12, rtol=0.0)
                ].sort_values("x_eff")
                if cur.empty:
                    continue
                ax.scatter(cur["x_eff"], cur["xi"], s=20, color=colors[float(vw)], marker=marker_map.get(float(h), "o"), alpha=0.85)
                ax.plot(cur["x_eff"], cur["xi_fit"], color=colors[float(vw)], lw=1.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$x_{\rm eff}=s(t_p,v_w)\,t_pH_*^\beta$")
        ax.set_ylabel(r"$\xi$")
    for ax in axes[len(theta_values):]:
        ax.axis("off")
    vw_handles = [plt.Line2D([0], [0], color=colors[float(vw)], lw=2.0) for vw in vw_values]
    vw_labels = [rf"$v_w={float(vw):.1f}$" for vw in vw_values]
    h_handles = [plt.Line2D([0], [0], color="black", marker=marker_map[float(h)], linestyle="None") for h in h_values]
    h_labels = [rf"$H_*={float(h):g}$" for h in h_values]
    fig.legend(vw_handles + h_handles, vw_labels + h_labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(r"Broken-power vw=0.9 baseline with fitted pointwise time shift", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_raw(df_pred, outdir: Path, dpi: int):
    theta_values = np.sort(df_pred["theta"].unique())
    vw_values = np.sort(df_pred["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {float(vw): cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    for h in sorted(df_pred["H"].unique()):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
        axes = axes.ravel()
        sub_h = df_pred[np.isclose(df_pred["H"], float(h), atol=1.0e-12)].copy()
        for ax, theta in zip(axes, theta_values):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
            for vw in vw_values:
                cur = sub[np.isclose(sub["v_w"], float(vw), atol=1.0e-12)].sort_values("beta_over_H")
                if cur.empty:
                    continue
                ax.scatter(cur["beta_over_H"], cur["xi"], s=22, color=colors[float(vw)], alpha=0.85)
                ax.plot(cur["beta_over_H"], cur["xi_fit"], color=colors[float(vw)], lw=1.8)
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$\beta/H_*$")
            ax.set_ylabel(r"$\xi$")
        for ax in axes[len(theta_values):]:
            ax.axis("off")
        handles = [plt.Line2D([0], [0], color=colors[float(vw)], lw=2.0) for vw in vw_values]
        labels = [rf"$v_w={float(vw):.1f}$" for vw in vw_values]
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
        fig.suptitle(rf"Broken-power shift prediction in raw $\xi(\beta/H_*)$, $H_*={float(h):.1f}$", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        tag = str(float(h)).replace(".", "p")
        fig.savefig(outdir / f"xi_vs_betaH_shift_H{tag}.png", dpi=dpi)
        plt.close(fig)


def plot_shift(df_pred, outpath: Path, dpi: int):
    vw_values = np.sort(df_pred["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {float(vw): cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    for vw in vw_values:
        cur = df_pred[np.isclose(df_pred["v_w"], float(vw), atol=1.0e-12)].copy().sort_values("tp")
        ax.scatter(cur["tp"], cur["s_fit"], s=18, color=colors[float(vw)], alpha=0.35)
        xfit = np.geomspace(float(cur["tp"].min()), float(cur["tp"].max()), 300)
        sfit = shift_model(xfit, np.full_like(xfit, float(vw)), df_pred["A"].iloc[0], df_pred["m"].iloc[0], df_pred["p"].iloc[0])
        ax.plot(xfit, sfit, color=colors[float(vw)], lw=2.0, label=rf"$v_w={float(vw):.1f}$")
    ax.axhline(1.0, color="black", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$t_p$")
    ax.set_ylabel(r"$s(t_p,v_w)$")
    ax.set_title("Fitted broken-power time shift")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        baseline = load_baseline(Path(args.baseline_summary).resolve())
        df = uq.load_lattice_dataframe(outdir, args.vw_folders, args.h_values)
        theta_values = np.sort(df["theta"].unique())
        ode = shared.load_ode_amplitudes(Path(args.ode_summary).resolve(), theta_values)
        meta = build_meta(df, theta_values, ode)

        fit = fit_shift(meta, baseline)

        x_base0, x_eff0, s0, tc_theta0, u0, xi0 = baseline_predict(meta, baseline, 0.0, 0.0, 0.0)
        df_pred = df.copy()
        df_pred["x_base"] = fit["x_base"]
        df_pred["x_eff"] = fit["x_eff"]
        df_pred["s_fit"] = fit["s_fit"]
        df_pred["tc_theta"] = fit["tc_theta"]
        df_pred["u_fit"] = fit["u_fit"]
        df_pred["xi_fit"] = fit["xi_fit"]
        df_pred["xi_baseline"] = xi0
        df_pred["frac_resid_shift"] = fit["frac_resid"]
        df_pred["frac_resid_baseline"] = (xi0 - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
        df_pred["A"] = fit["A"]
        df_pred["m"] = fit["m"]
        df_pred["p"] = fit["p"]
        df_pred.to_csv(outdir / "predictions.csv", index=False)

        plot_collapse(df_pred, outdir / "collapse_overlay_shift.png", args.dpi)
        plot_raw(df_pred, outdir, args.dpi)
        plot_shift(df_pred, outdir / "s_vs_tp_fit.png", args.dpi)

        raw_rows, by_h, by_vw = rmse_tables(df_pred)
        summary = {
            "status": fit["status"],
            "baseline_source": baseline["source"],
            "ode_amplitude_source": str(Path(args.ode_summary).resolve()),
            "model": {
                "family": "broken_power_shifted_time",
                "baseline_formula": "xi = [1 + u^r]^(1/r), u = F_inf(theta0) * (2 x / (3 t_osc tc(theta0)))^(3/2) / F0(theta0)^2",
                "time_shift_formula": "s(tp,v_w) = 1 + A[(0.9/v_w)^m - 1] tp^p",
                "x_eff_formula": "x_eff = s(tp,v_w) * tp * H_*^beta",
            },
            "baseline_params": {
                "beta": baseline["beta"],
                "r": baseline["r"],
                "tc0_vw0p9": baseline["tc0"],
                "gamma_vw0p9": baseline["gamma"],
                "h0": baseline["h0"],
                "t_osc": baseline["t_osc"],
            },
            "shift_params": {"A": fit["A"], "m": fit["m"], "p": fit["p"]},
            "rel_rmse_shift": fit["rel_rmse"],
            "rel_rmse_baseline": fit["baseline_rel_rmse"],
            "AIC": fit["AIC"],
            "BIC": fit["BIC"],
            "n_points": fit["n_points"],
            "n_params": fit["n_params"],
            "mean_raw_rmse_by_h": by_h,
            "per_vw_rel_rmse": by_vw,
            "raw_plot_rmse": raw_rows,
            "outputs": {
                "collapse_overlay": str((outdir / "collapse_overlay_shift.png").resolve()),
                "shift_plot": str((outdir / "s_vs_tp_fit.png").resolve()),
                "raw_fit_by_H": [str((outdir / f"xi_vs_betaH_shift_H{str(float(h)).replace('.', 'p')}.png").resolve()) for h in sorted(df_pred["H"].unique())],
                "predictions": str((outdir / "predictions.csv").resolve()),
            },
        }
        save_json(outdir / "final_summary.json", summary)
        print(json.dumps(shared.to_native(summary), indent=2, sort_keys=True))
    except Exception as exc:
        error_exit(outdir, exc)
        raise


if __name__ == "__main__":
    main()
