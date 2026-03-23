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
import fit_vw_smearing_tests as smear


OUTDIR = ROOT / "results_vw_overlay_timewarped"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]
THETA_TARGETS = np.array([0.262, 0.785, 1.309, 1.833, 2.356, 2.880], dtype=np.float64)


def parse_args():
    p = argparse.ArgumentParser(description="Overlay lattice xi after fitting per-vw time-shift factors.")
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--reference-vw", type=float, default=None)
    p.add_argument("--outdir", type=str, default=str(OUTDIR))
    p.add_argument("--title-tag", type=str, default="")
    return p.parse_args()


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


def fit_scales(df: np.ndarray, ref_vw: float):
    pair_map = smear.build_alignment_pairs(df, ref_vw)
    scales = {f"{ref_vw:.1f}": 1.0}
    fit_rows = []
    for vw, pairs in pair_map.items():
        fit = smear.optimize_scale_for_pairs(pairs)
        scales[f"{vw:.1f}"] = float(fit["scale"])
        fit_rows.append(
            {
                "v_w": float(vw),
                "scale": float(fit["scale"]),
                "rel_rmse": float(fit["rel_rmse"]),
                "n_points": int(fit["n_points"]),
                "success": bool(fit["success"]),
                "message": str(fit["message"]),
            }
        )
    fit_rows.append({"v_w": float(ref_vw), "scale": 1.0, "rel_rmse": 0.0, "n_points": 0, "success": True, "message": "reference"})
    fit_rows = sorted(fit_rows, key=lambda row: row["v_w"])
    return scales, fit_rows


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_lattice_dataframe(outdir, args.vw_folders, args.h_values)
    ref_vw = float(args.reference_vw) if args.reference_vw is not None else float(np.max(df["v_w"]))
    scales, fit_rows = fit_scales(df, ref_vw)

    theta_subset = choose_theta_subset(df["theta"].unique())
    vw_values = np.sort(df["v_w"].unique())
    h_values = np.sort(df["H"].unique())

    cmap = plt.get_cmap("viridis")
    colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    marker_map = {1.0: "s", 1.5: "^", 2.0: "D", 0.5: "o"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
    axes = axes.ravel()

    for ax, theta in zip(axes, theta_subset):
        sub = df[np.isclose(df["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
        for vw in vw_values:
            scale = float(scales.get(f"{float(vw):.1f}", 1.0))
            for h in h_values:
                cur = sub[
                    np.isclose(sub["v_w"], float(vw), atol=1.0e-12, rtol=0.0)
                    & np.isclose(sub["H"], float(h), atol=1.0e-12, rtol=0.0)
                ].sort_values("tp")
                if cur.empty:
                    continue
                x = scale * cur["tp"].to_numpy(dtype=np.float64) * np.power(cur["H"].to_numpy(dtype=np.float64), float(args.beta))
                ax.plot(
                    x,
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
        if abs(float(args.beta)) < 1.0e-12:
            ax.set_xlabel(r"$x = s(v_w)\, t_p$")
        else:
            ax.set_xlabel(r"$x = s(v_w)\, t_p H_*^\beta$")
        ax.set_ylabel(r"$\xi$")

    for ax in axes[len(theta_subset) :]:
        ax.axis("off")

    vw_handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
    vw_labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
    h_handles = [plt.Line2D([0], [0], color="black", marker=marker_map[h], linestyle="None") for h in h_values]
    h_labels = [rf"$H_*={h:g}$" for h in h_values]
    fig.legend(vw_handles + h_handles, vw_labels + h_labels, loc="upper center", ncol=4, frameon=False)
    title_tag = f", {args.title_tag}" if args.title_tag else ""
    if abs(float(args.beta)) < 1.0e-12:
        fig.suptitle(rf"Lattice $\xi$ after fitted per-$v_w$ time warp{title_tag}", y=0.995)
    else:
        fig.suptitle(rf"Lattice $\xi$ after fitted per-$v_w$ time warp and $H^\beta$ rescaling, $\beta={args.beta:.4f}{title_tag}$", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = outdir / "lattice_xi_vs_x_all_vw_timewarped.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)

    summary = {
        "status": "ok",
        "beta": float(args.beta),
        "reference_vw": float(ref_vw),
        "scales": scales,
        "scale_fit_rows": fit_rows,
        "output": str(out),
        "vw_values": [float(v) for v in vw_values],
        "H_values": [float(v) for v in h_values],
        "theta_subset": [float(v) for v in theta_subset],
    }
    (outdir / "lattice_timewarp_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        payload = {"status": "error", "message": str(exc)}
        outdir = Path(parse_args().outdir).resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "lattice_timewarp_error.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(json.dumps(payload, sort_keys=True))
        sys.exit(1)
