#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_SUMMARY = ROOT / "knob_tc_weibull" / "rvw_theta" / "final_summary.json"
DEFAULT_OUTDIR = ROOT / "results_log_tcfit_vs_log_htheta_weibull_rvw_theta"


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot log t_c^fit(theta0,v_w) vs log h(theta0) for each v_w from a fitted tc_by_vw_theta summary."
    )
    p.add_argument("--summary", type=str, default=str(DEFAULT_SUMMARY))
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    return p.parse_args()


def h_theta(theta):
    theta = np.asarray(theta, dtype=np.float64)
    cos_half = np.cos(theta / 2.0)
    return np.log(np.e / np.maximum(cos_half * cos_half, 1.0e-300))


def load_summary(path: Path):
    obj = json.loads(path.read_text())
    if "tc_by_vw_theta" not in obj:
        raise KeyError(f"{path} does not contain `tc_by_vw_theta`.")
    return obj


def parse_grid(summary: dict):
    tc_table = summary["tc_by_vw_theta"]
    vw_values = np.array(sorted(float(k) for k in tc_table.keys()), dtype=np.float64)
    theta_values = np.array(sorted(float(k) for k in next(iter(tc_table.values())).keys()), dtype=np.float64)
    tc_grid = np.empty((len(vw_values), len(theta_values)), dtype=np.float64)
    for i, vw in enumerate(vw_values):
        block = tc_table[f"{float(vw):.1f}"]
        for j, theta in enumerate(theta_values):
            tc_grid[i, j] = float(block[f"{float(theta):.10f}"])
    return vw_values, theta_values, tc_grid


def fit_lines(log_h: np.ndarray, log_tc_grid: np.ndarray, vw_values: np.ndarray):
    rows = []
    slopes = {}
    intercepts = {}
    for i, vw in enumerate(vw_values):
        coeff = np.polyfit(log_h, log_tc_grid[i], deg=1)
        slope = float(coeff[0])
        intercept = float(coeff[1])
        slopes[f"{float(vw):.1f}"] = slope
        intercepts[f"{float(vw):.1f}"] = intercept
        for x, y in zip(log_h, log_tc_grid[i]):
            rows.append(
                {
                    "v_w": float(vw),
                    "log_h_theta": float(x),
                    "log_tc_fit": float(y),
                    "theta": None,
                }
            )
    return slopes, intercepts


def plot_all(log_h: np.ndarray, log_tc_grid: np.ndarray, theta_values: np.ndarray, vw_values: np.ndarray, slopes: dict, intercepts: dict, outpath: Path, dpi: int):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True, constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    theta_colors = {float(theta): cmap(i / max(len(theta_values) - 1, 1)) for i, theta in enumerate(theta_values)}
    x_line = np.linspace(float(log_h.min()) - 0.05, float(log_h.max()) + 0.05, 200)
    for ax, vw in zip(axes.ravel(), vw_values):
        i = int(np.argmin(np.abs(vw_values - float(vw))))
        y = log_tc_grid[i]
        ax.plot(log_h, y, "o", color="black", ms=4)
        for x0, y0, theta in zip(log_h, y, theta_values):
            ax.scatter([x0], [y0], color=theta_colors[float(theta)], s=40)
            ax.annotate(rf"{theta:.3f}", (x0, y0), textcoords="offset points", xytext=(4, 4), fontsize=8)
        m = slopes[f"{float(vw):.1f}"]
        b = intercepts[f"{float(vw):.1f}"]
        ax.plot(x_line, m * x_line + b, color="tab:red", lw=1.8, label=rf"$\gamma_{{fit}}={m:.3f}$")
        ax.set_title(rf"$v_w={float(vw):.1f}$")
        ax.set_xlabel(r"$\log h(\theta_0)$")
        ax.set_ylabel(r"$\log t_c^{\rm fit}(\theta_0,v_w)$")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(r"Log-Log Check of $t_c^{\rm fit}(\theta_0,v_w)$ vs $h(\theta_0)$", fontsize=16)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def save_csv(theta_values: np.ndarray, vw_values: np.ndarray, log_h: np.ndarray, tc_grid: np.ndarray, outpath: Path):
    lines = ["v_w,theta,h_theta,log_h_theta,tc_fit,log_tc_fit"]
    for i, vw in enumerate(vw_values):
        for theta, h_val, tc in zip(theta_values, np.exp(log_h), tc_grid[i]):
            lines.append(
                f"{float(vw):.6f},{float(theta):.10f},{float(h_val):.10f},{float(np.log(h_val)):.10f},{float(tc):.10f},{float(np.log(tc)):.10f}"
            )
    outpath.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    summary_path = Path(args.summary).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_path)
    vw_values, theta_values, tc_grid = parse_grid(summary)
    h_vals = h_theta(theta_values)
    log_h = np.log(np.maximum(h_vals, 1.0e-300))
    log_tc_grid = np.log(np.maximum(tc_grid, 1.0e-300))
    slopes, intercepts = fit_lines(log_h, log_tc_grid, vw_values)

    plot_all(log_h, log_tc_grid, theta_values, vw_values, slopes, intercepts, outdir / "log_tcfit_vs_log_htheta.png", args.dpi)
    save_csv(theta_values, vw_values, log_h, tc_grid, outdir / "log_tcfit_vs_log_htheta.csv")

    payload = {
        "status": "ok",
        "summary_source": str(summary_path),
        "h_theta_form": "h(theta0) = log(e / cos(theta0/2)^2)",
        "plot_form": "log tc_fit(theta0,v_w) vs log h(theta0)",
        "vw_values": [float(v) for v in vw_values],
        "theta_values": [float(t) for t in theta_values],
        "gamma_fit_by_vw": slopes,
        "intercept_by_vw": intercepts,
        "outputs": {
            "plot": str(outdir / "log_tcfit_vs_log_htheta.png"),
            "csv": str(outdir / "log_tcfit_vs_log_htheta.csv"),
        },
    }
    (outdir / "fit_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
