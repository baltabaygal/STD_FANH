#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fit_vw_amplitude as base_fit


OUTDIR = ROOT / "results_vw_overlay"
BETA_PATH = ROOT / "results_collapse/best_beta.json"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [0.5, 1.0, 1.5, 2.0]
THETA_TARGETS = np.array([0.262, 0.785, 1.309, 1.833, 2.356, 2.880], dtype=np.float64)


def parse_args():
    p = argparse.ArgumentParser(description="Overlay lattice xi vs x = tp H^beta for all v_w values.")
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--beta-from", type=str, default=str(BETA_PATH))
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--outdir", type=str, default=str(OUTDIR))
    p.add_argument("--title-tag", type=str, default="")
    return p.parse_args()


def load_beta(path: Path):
    if path.exists():
        data = json.loads(path.read_text())
        return float(data["beta"])
    return -0.0976177386


def choose_theta_subset(theta_values):
    theta_values = np.asarray(sorted(theta_values), dtype=np.float64)
    out = []
    for target in THETA_TARGETS:
        idx = int(np.argmin(np.abs(theta_values - target)))
        val = float(theta_values[idx])
        if val not in out:
            out.append(val)
    return np.asarray(out, dtype=np.float64)


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


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    beta = float(args.beta) if args.beta is not None else load_beta(Path(args.beta_from).resolve())
    df = load_lattice_dataframe(outdir, args.vw_folders, args.h_values)
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
                cur = sub[
                    np.isclose(sub["v_w"], float(vw), atol=1.0e-12, rtol=0.0)
                    & np.isclose(sub["H"], float(h), atol=1.0e-12, rtol=0.0)
                ].sort_values("x")
                if cur.empty:
                    continue
                ax.plot(
                    cur["x"],
                    cur["xi"],
                    color=colors[float(vw)],
                    marker=marker_map.get(float(h), "o"),
                    lw=1.4,
                    ms=3.0,
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
    title_tag = f", {args.title_tag}" if args.title_tag else ""
    fig.suptitle(rf"Lattice $\xi$ vs $x=t_p H_*^\beta$ with all $v_w$ overlaid, $\beta={beta:.4f}{title_tag}$", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = outdir / "lattice_xi_vs_x_all_vw.png"
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
    (outdir / "lattice_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        payload = {"status": "error", "message": str(exc)}
        outdir = Path(parse_args().outdir).resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "lattice_error.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(json.dumps(payload, sort_keys=True))
        sys.exit(1)
