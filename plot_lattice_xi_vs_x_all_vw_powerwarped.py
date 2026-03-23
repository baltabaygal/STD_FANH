#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fit_vw_amplitude as base_fit
import fit_vw_smearing_tests as smear


OUTDIR = ROOT / "results_vw_overlay_powerwarped"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]
THETA_TARGETS = np.array([0.262, 0.785, 1.309, 1.833, 2.356, 2.880], dtype=np.float64)


def parse_args():
    p = argparse.ArgumentParser(description="Overlay lattice xi after fitting per-vw power-law time warps.")
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


def warp_tp(tp, log_s, b):
    tp = np.asarray(tp, dtype=np.float64)
    return np.exp(float(log_s)) * np.power(np.maximum(tp, 1.0e-18), float(b))


def powerwarp_residuals(log_s, b, pairs, beta):
    residuals = []
    beta = float(beta)
    for pair in pairs:
        h = float(pair["H"])
        x_ref = pair["ref_tp"] * (h ** beta)
        x_cur = warp_tp(pair["tp"], log_s, b) * (h ** beta)
        pred = smear.log_interp(x_ref, pair["ref_xi"], x_cur)
        mask = np.isfinite(pred) & np.isfinite(pair["xi"]) & (pair["xi"] > 0.0)
        if np.count_nonzero(mask) < 3:
            continue
        residuals.append((pred[mask] - pair["xi"][mask]) / np.maximum(pair["xi"][mask], 1.0e-12))
    if not residuals:
        return np.array([], dtype=np.float64)
    return np.concatenate(residuals)


def fit_powerwarp_for_pairs(pairs, beta):
    if not pairs:
        return {
            "log_s": np.nan,
            "s": np.nan,
            "b": np.nan,
            "rel_rmse": np.nan,
            "success": False,
            "n_points": 0,
            "message": "no_pairs",
        }

    def objective(x):
        resid = powerwarp_residuals(float(x[0]), float(x[1]), pairs, beta)
        if resid.size == 0:
            return 1.0e9
        return float(np.mean(np.square(resid)))

    res = minimize(
        objective,
        x0=np.array([0.0, 1.0], dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(-2.0, 2.0), (0.5, 1.5)],
    )
    log_s = float(res.x[0])
    b = float(res.x[1])
    resid = powerwarp_residuals(log_s, b, pairs, beta)
    return {
        "log_s": log_s,
        "s": float(math.exp(log_s)),
        "b": b,
        "rel_rmse": smear.pooled_rel_rmse(resid),
        "success": bool(res.success),
        "n_points": int(resid.size),
        "message": str(res.message),
    }


def fit_powerwarps(df, ref_vw, beta):
    pair_map = smear.build_alignment_pairs(df, ref_vw)
    params = {f"{ref_vw:.1f}": {"log_s": 0.0, "s": 1.0, "b": 1.0}}
    rows = [{"v_w": float(ref_vw), "s": 1.0, "log_s": 0.0, "b": 1.0, "rel_rmse": 0.0, "n_points": 0, "success": True, "message": "reference"}]
    for vw, pairs in pair_map.items():
        fit = fit_powerwarp_for_pairs(pairs, beta)
        params[f"{float(vw):.1f}"] = {"log_s": fit["log_s"], "s": fit["s"], "b": fit["b"]}
        rows.append({"v_w": float(vw), **fit})
    rows = sorted(rows, key=lambda row: row["v_w"])
    return params, rows


def plot_overlay(df, params, beta, outpath: Path, title_tag: str):
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
            rec = params.get(f"{float(vw):.1f}", {"log_s": 0.0, "b": 1.0})
            for h in h_values:
                cur = sub[
                    np.isclose(sub["v_w"], float(vw), atol=1.0e-12, rtol=0.0)
                    & np.isclose(sub["H"], float(h), atol=1.0e-12, rtol=0.0)
                ].sort_values("tp")
                if cur.empty:
                    continue
                x = warp_tp(cur["tp"].to_numpy(dtype=np.float64), rec["log_s"], rec["b"]) * np.power(
                    cur["H"].to_numpy(dtype=np.float64), float(beta)
                )
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
        if abs(float(beta)) < 1.0e-12:
            ax.set_xlabel(r"$x = s(v_w)\, t_p^{\,b(v_w)}$")
        else:
            ax.set_xlabel(r"$x = s(v_w)\, t_p^{\,b(v_w)} H_*^\beta$")
        ax.set_ylabel(r"$\xi$")

    for ax in axes[len(theta_subset) :]:
        ax.axis("off")

    vw_handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
    vw_labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
    h_handles = [plt.Line2D([0], [0], color="black", marker=marker_map[h], linestyle="None") for h in h_values]
    h_labels = [rf"$H_*={h:g}$" for h in h_values]
    fig.legend(vw_handles + h_handles, vw_labels + h_labels, loc="upper center", ncol=4, frameon=False)
    suffix = f", {title_tag}" if title_tag else ""
    if abs(float(beta)) < 1.0e-12:
        fig.suptitle(rf"Lattice $\xi$ after fitted per-$v_w$ power warp{suffix}", y=0.995)
    else:
        fig.suptitle(rf"Lattice $\xi$ after fitted per-$v_w$ power warp and $H^\beta$ rescaling, $\beta={beta:.4f}{suffix}$", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_lattice_dataframe(outdir, args.vw_folders, args.h_values)
    ref_vw = float(args.reference_vw) if args.reference_vw is not None else float(np.max(df["v_w"]))
    params, rows = fit_powerwarps(df, ref_vw, args.beta)

    outpath = outdir / "lattice_xi_vs_x_all_vw_powerwarped.png"
    plot_overlay(df, params, args.beta, outpath, args.title_tag)

    summary = {
        "status": "ok",
        "beta": float(args.beta),
        "reference_vw": float(ref_vw),
        "params": params,
        "fit_rows": rows,
        "output": str(outpath),
        "vw_values": [float(v) for v in np.sort(df["v_w"].unique())],
        "H_values": [float(v) for v in np.sort(df["H"].unique())],
        "theta_subset": [float(v) for v in choose_theta_subset(df["theta"].unique())],
    }
    (outdir / "lattice_powerwarp_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        payload = {"status": "error", "message": str(exc)}
        outdir = Path(parse_args().outdir).resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "lattice_powerwarp_error.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(json.dumps(payload, sort_keys=True))
        sys.exit(1)
