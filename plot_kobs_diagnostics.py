#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fit_lattice_fixed_ode_amplitudes as shared
import fit_lattice_quadwarp_universal as quad


ROOT = Path(__file__).resolve().parent
OUTDIR_DEFAULT = ROOT / "results_kobs_observed_plainR3_weibull_rvw_hgrid"
FIT_SUMMARY_DEFAULT = ROOT / "results_lattice_theta_tc_rvw_tests_beta0_tcmax300_weibull" / "final_summary.json"
FIT_CASE_DEFAULT = "rvw_hgrid"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(description="Plot the observed kernel K_obs = xi - (f_inf/F0) * R^3 directly from lattice xi.")
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--t-osc", type=float, default=1.5)
    p.add_argument("--ode-summary", type=str, default=str(shared.ODE_SUMMARY_DEFAULT))
    p.add_argument("--fit-summary", type=str, default=str(FIT_SUMMARY_DEFAULT))
    p.add_argument("--fit-case", type=str, default=FIT_CASE_DEFAULT)
    p.add_argument("--r3-kind", type=str, choices=["plain", "scale_factor"], default="plain")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--outdir", type=str, default=str(OUTDIR_DEFAULT))
    return p.parse_args()


def save_json(path: Path, payload):
    path.write_text(json.dumps(shared.to_native(payload), indent=2, sort_keys=True))


def error_exit(outdir: Path, exc: Exception):
    payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "_error.json", payload)
    print(json.dumps(payload, sort_keys=True))


def r3_factor(tp: np.ndarray, t_osc: float, kind: str):
    tp = np.asarray(tp, dtype=np.float64)
    if kind == "plain":
        return np.power(np.maximum(tp / max(float(t_osc), 1.0e-18), 1.0e-18), 1.5)
    if kind == "scale_factor":
        return np.power(np.maximum((2.0 * tp) / max(3.0 * float(t_osc), 1.0e-18), 1.0e-18), 1.5)
    raise ValueError(f"Unknown r3 kind: {kind}")


def tc_lookup(case_summary: dict, theta: float, vw: float, h_value: float):
    if "tc_by_vw_theta_h" in case_summary:
        return float(case_summary["tc_by_vw_theta_h"][f"{float(vw):.1f}"][f"{float(theta):.10f}"][f"{float(h_value):.1f}"])
    if "tc_by_vw_theta" in case_summary:
        return float(case_summary["tc_by_vw_theta"][f"{float(vw):.1f}"][f"{float(theta):.10f}"])
    if "t_c_shared" in case_summary:
        return float(case_summary["t_c_shared"])
    raise KeyError("Could not find a usable tc table in the selected fit summary.")


def build_frame(df: pd.DataFrame, ode: dict, t_osc: float, r3_kind: str, fit_case_summary: dict):
    theta_values = np.sort(df["theta"].unique())
    theta_index = {float(theta): i for i, theta in enumerate(theta_values)}
    f0 = np.asarray(ode["F0"], dtype=np.float64)
    finf = np.asarray(ode["F_inf"], dtype=np.float64)
    out = df.copy()
    out["theta_idx"] = np.array([theta_index[float(t)] for t in out["theta"].to_numpy(dtype=np.float64)], dtype=np.int64)
    out["f_infty_over_F0_ode"] = finf[out["theta_idx"]] / np.maximum(f0[out["theta_idx"]], 1.0e-18)
    out["R3"] = r3_factor(out["tp"].to_numpy(dtype=np.float64), t_osc, r3_kind)
    out["K_obs"] = out["xi"] - out["f_infty_over_F0_ode"] * out["R3"]
    out["tc_fit"] = np.array(
        [tc_lookup(fit_case_summary, row.theta, row.v_w, row.H) for row in out.itertuples(index=False)],
        dtype=np.float64,
    )
    out["tp_over_tc"] = out["tp"].to_numpy(dtype=np.float64) / np.maximum(out["tc_fit"], 1.0e-18)
    return out


def compute_ylim(values: np.ndarray):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-0.2, 1.2)
    lo = min(float(np.percentile(vals, 1.0)), -0.05)
    hi = max(float(np.percentile(vals, 99.0)), 1.05)
    pad = 0.08 * max(hi - lo, 1.0)
    return lo - pad, hi + pad


def plot_by_vw(df: pd.DataFrame, outpath: Path, dpi: int):
    theta_values = np.sort(df["theta"].unique())
    vw_values = np.sort(df["v_w"].unique())
    h_values = np.sort(df["H"].unique())
    colors = {float(theta): plt.get_cmap("plasma")(i / max(len(theta_values) - 1, 1)) for i, theta in enumerate(theta_values)}
    marker_map = {1.0: "s", 1.5: "^", 2.0: "D", 0.5: "o"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True, constrained_layout=True)
    ylim = compute_ylim(df["K_obs"])
    for ax, vw in zip(axes.ravel(), vw_values):
        sub_vw = df[np.isclose(df["v_w"], float(vw), atol=1.0e-8)].copy()
        for theta in theta_values:
            sub_theta = sub_vw[np.isclose(sub_vw["theta"], float(theta), atol=1.0e-8)].copy()
            for h in h_values:
                cur = sub_theta[np.isclose(sub_theta["H"], float(h), atol=1.0e-8)].sort_values("tp")
                ax.plot(cur["tp"], cur["K_obs"], color=colors[float(theta)], marker=marker_map.get(float(h), "o"), lw=1.2, ms=4.0, alpha=0.9)
        ax.axhline(1.0, color="gray", ls="--", lw=1.0)
        ax.axhline(0.0, color="black", ls=":", lw=1.0)
        ax.set_xscale("log")
        ax.set_ylim(*ylim)
        ax.set_title(rf"$v_w={float(vw):.1f}$")
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$K_{\rm obs}$")
    theta_handles = [plt.Line2D([0], [0], color=colors[float(theta)], lw=2.0) for theta in theta_values]
    theta_labels = [rf"$\theta_0={float(theta):.3f}$" for theta in theta_values]
    h_handles = [plt.Line2D([0], [0], color="black", marker=marker_map[float(h)], linestyle="None") for h in h_values]
    h_labels = [rf"$H_*={float(h):.1f}$" for h in h_values]
    fig.legend(theta_handles + h_handles, theta_labels + h_labels, loc="upper center", ncol=5, frameon=False)
    fig.suptitle(r"$K_{\rm obs}(t_p,\theta_0,v_w)$ at fixed $v_w$")
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_by_theta(df: pd.DataFrame, outpath: Path, dpi: int):
    theta_values = np.sort(df["theta"].unique())
    vw_values = np.sort(df["v_w"].unique())
    h_values = np.sort(df["H"].unique())
    colors = {float(vw): plt.get_cmap("viridis")(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    marker_map = {1.0: "s", 1.5: "^", 2.0: "D", 0.5: "o"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True, constrained_layout=True)
    ylim = compute_ylim(df["K_obs"])
    for ax, theta in zip(axes.ravel(), theta_values):
        sub_theta = df[np.isclose(df["theta"], float(theta), atol=1.0e-8)].copy()
        for vw in vw_values:
            sub_vw = sub_theta[np.isclose(sub_theta["v_w"], float(vw), atol=1.0e-8)].copy()
            for h in h_values:
                cur = sub_vw[np.isclose(sub_vw["H"], float(h), atol=1.0e-8)].sort_values("tp")
                ax.plot(cur["tp"], cur["K_obs"], color=colors[float(vw)], marker=marker_map.get(float(h), "o"), lw=1.2, ms=4.0, alpha=0.9)
        ax.axhline(1.0, color="gray", ls="--", lw=1.0)
        ax.axhline(0.0, color="black", ls=":", lw=1.0)
        ax.set_xscale("log")
        ax.set_ylim(*ylim)
        ax.set_title(rf"$\theta_0={float(theta):.3f}$")
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$K_{\rm obs}$")
    vw_handles = [plt.Line2D([0], [0], color=colors[float(vw)], lw=2.0) for vw in vw_values]
    vw_labels = [rf"$v_w={float(vw):.1f}$" for vw in vw_values]
    h_handles = [plt.Line2D([0], [0], color="black", marker=marker_map[float(h)], linestyle="None") for h in h_values]
    h_labels = [rf"$H_*={float(h):.1f}$" for h in h_values]
    fig.legend(vw_handles + h_handles, vw_labels + h_labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(r"$K_{\rm obs}(t_p,\theta_0,v_w)$ at fixed $\theta_0$")
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_scaled_by_vw(df: pd.DataFrame, outpath: Path, dpi: int):
    theta_values = np.sort(df["theta"].unique())
    vw_values = np.sort(df["v_w"].unique())
    h_values = np.sort(df["H"].unique())
    colors = {float(theta): plt.get_cmap("plasma")(i / max(len(theta_values) - 1, 1)) for i, theta in enumerate(theta_values)}
    marker_map = {1.0: "s", 1.5: "^", 2.0: "D", 0.5: "o"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True, constrained_layout=True)
    ylim = compute_ylim(df["K_obs"])
    for ax, vw in zip(axes.ravel(), vw_values):
        sub_vw = df[np.isclose(df["v_w"], float(vw), atol=1.0e-8)].copy()
        for theta in theta_values:
            sub_theta = sub_vw[np.isclose(sub_vw["theta"], float(theta), atol=1.0e-8)].copy()
            for h in h_values:
                cur = sub_theta[np.isclose(sub_theta["H"], float(h), atol=1.0e-8)].sort_values("tp_over_tc")
                ax.plot(cur["tp_over_tc"], cur["K_obs"], color=colors[float(theta)], marker=marker_map.get(float(h), "o"), lw=1.2, ms=4.0, alpha=0.9)
        ax.axhline(1.0, color="gray", ls="--", lw=1.0)
        ax.axhline(0.0, color="black", ls=":", lw=1.0)
        ax.set_xscale("log")
        ax.set_ylim(*ylim)
        ax.set_title(rf"$v_w={float(vw):.1f}$")
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"$t_p/t_c^{\rm fit}$")
        ax.set_ylabel(r"$K_{\rm obs}$")
    theta_handles = [plt.Line2D([0], [0], color=colors[float(theta)], lw=2.0) for theta in theta_values]
    theta_labels = [rf"$\theta_0={float(theta):.3f}$" for theta in theta_values]
    h_handles = [plt.Line2D([0], [0], color="black", marker=marker_map[float(h)], linestyle="None") for h in h_values]
    h_labels = [rf"$H_*={float(h):.1f}$" for h in h_values]
    fig.legend(theta_handles + h_handles, theta_labels + h_labels, loc="upper center", ncol=5, frameon=False)
    fig.suptitle(r"$K_{\rm obs}$ versus $t_p/t_c^{\rm fit}$")
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        df = quad.load_lattice_dataframe(outdir, args.vw_folders, args.h_values)
        theta_values = np.sort(df["theta"].unique())
        ode = shared.load_ode_amplitudes(Path(args.ode_summary).resolve(), theta_values)
        fit_summary = json.loads(Path(args.fit_summary).resolve().read_text())
        fit_case_summary = fit_summary["cases"][args.fit_case]
        frame = build_frame(df, ode, args.t_osc, args.r3_kind, fit_case_summary)
        frame.to_csv(outdir / "kobs_table.csv", index=False)
        plot_by_vw(frame, outdir / "kobs_vs_tp_by_vw.png", args.dpi)
        plot_by_theta(frame, outdir / "kobs_vs_tp_by_theta.png", args.dpi)
        plot_scaled_by_vw(frame, outdir / "kobs_vs_tp_over_tc_by_vw.png", args.dpi)
        summary = {
            "status": "ok",
            "fit_summary": str(Path(args.fit_summary).resolve()),
            "fit_case": str(args.fit_case),
            "ode_summary": str(Path(args.ode_summary).resolve()),
            "t_osc": float(args.t_osc),
            "r3_kind": str(args.r3_kind),
            "r3_form": "(tp/t_osc)^(3/2)" if args.r3_kind == "plain" else "(2 tp / (3 t_osc))^(3/2)",
            "k_obs_form": "K_obs = xi - (f_infty^ODE / F0^ODE) * R^3",
            "n_points": int(len(frame)),
            "kobs_range": {
                "min": float(np.nanmin(frame["K_obs"])),
                "max": float(np.nanmax(frame["K_obs"])),
            },
            "outputs": {
                "table": str(outdir / "kobs_table.csv"),
                "kobs_vs_tp_by_vw": str(outdir / "kobs_vs_tp_by_vw.png"),
                "kobs_vs_tp_by_theta": str(outdir / "kobs_vs_tp_by_theta.png"),
                "kobs_vs_tp_over_tc_by_vw": str(outdir / "kobs_vs_tp_over_tc_by_vw.png"),
            },
        }
        save_json(outdir / "final_summary.json", summary)
        print(json.dumps(shared.to_native(summary), indent=2, sort_keys=True))
    except Exception as exc:
        error_exit(outdir, exc)
        raise


if __name__ == "__main__":
    main()
