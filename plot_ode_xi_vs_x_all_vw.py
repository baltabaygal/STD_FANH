#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "results_vw_overlay"
ODE_PATH = ROOT / "ode/xi_DM_ODE_results.txt"
BETA_PATH = ROOT / "results_collapse/best_beta.json"
THETA_TARGETS = np.array([0.262, 0.785, 1.309, 1.833, 2.356, 2.880], dtype=np.float64)


def load_beta():
    if BETA_PATH.exists():
        data = json.loads(BETA_PATH.read_text())
        return float(data["beta"])
    return -0.0976177386


def load_ode():
    if not ODE_PATH.exists():
        raise FileNotFoundError(f"Missing ODE file: {ODE_PATH}")
    df = pd.read_csv(ODE_PATH, sep=r"\s+|,", engine="python", comment="#", header=None)
    if df.shape[1] < 6:
        raise RuntimeError("ODE table must have at least 6 columns: vw theta H beta_over_H tp xi")
    df = df.iloc[:, :6].copy()
    df.columns = ["v_w", "theta", "H", "beta_over_H", "tp", "xi"]
    df = df.astype(float)
    df = df[np.isfinite(df["v_w"]) & np.isfinite(df["theta"]) & np.isfinite(df["H"]) & np.isfinite(df["tp"]) & np.isfinite(df["xi"])].copy()
    df = df[(df["tp"] > 0.0) & (df["xi"] > 0.0)].copy()
    return df.sort_values(["v_w", "H", "theta", "tp"]).reset_index(drop=True)


def choose_theta_subset(theta_values):
    theta_values = np.asarray(sorted(theta_values), dtype=np.float64)
    out = []
    for target in THETA_TARGETS:
        idx = int(np.argmin(np.abs(theta_values - target)))
        val = float(theta_values[idx])
        if val not in out:
            out.append(val)
    return np.asarray(out, dtype=np.float64)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    beta = load_beta()
    df = load_ode()
    df["x"] = df["tp"] * np.power(df["H"], beta)

    theta_subset = choose_theta_subset(df["theta"].unique())
    vw_values = np.sort(df["v_w"].unique())
    h_values = np.sort(df["H"].unique())

    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    marker_map = {0.5: "o", 1.0: "s", 1.5: "^", 2.0: "D"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
    axes = axes.ravel()

    for ax, theta in zip(axes, theta_subset):
        sub = df[np.isclose(df["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
        for vw in vw_values:
            for h in h_values:
                cur = sub[np.isclose(sub["v_w"], float(vw), atol=1.0e-12, rtol=0.0) & np.isclose(sub["H"], float(h), atol=1.0e-12, rtol=0.0)].sort_values("x")
                if cur.empty:
                    continue
                ax.plot(
                    cur["x"],
                    cur["xi"],
                    color=colors[float(vw)],
                    marker=marker_map.get(float(h), "o"),
                    lw=1.4,
                    ms=3.2,
                    alpha=0.9,
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$x = t_p H_*^\beta$")
        ax.set_ylabel(r"$\xi$")

    for ax in axes[len(theta_subset) :]:
        ax.axis("off")

    vw_handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
    vw_labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
    h_handles = [plt.Line2D([0], [0], color="black", marker=marker_map[h], linestyle="None") for h in h_values]
    h_labels = [rf"$H_*={h:g}$" for h in h_values]
    fig.legend(vw_handles + h_handles, vw_labels + h_labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(rf"ODE $\xi$ vs $x=t_p H_*^\beta$ with all $v_w$ overlaid, $\beta={beta:.4f}$", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUTDIR / "ode_xi_vs_x_all_vw.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)

    summary = {
        "status": "ok",
        "beta": beta,
        "output": str(out),
        "vw_values": [float(v) for v in vw_values],
        "H_values": [float(v) for v in h_values],
        "theta_subset": [float(v) for v in theta_subset],
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        payload = {"status": "error", "message": str(exc)}
        OUTDIR.mkdir(parents=True, exist_ok=True)
        (OUTDIR / "_error.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(json.dumps(payload, sort_keys=True))
        sys.exit(1)
