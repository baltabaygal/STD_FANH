#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import fit_vw_amplitude as base_fit


OUTDIR = ROOT / "results_vw_xi_times_F0_ampcollapse"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]
DEFAULT_ALPHA = -0.12114099707506708


def parse_args():
    p = argparse.ArgumentParser(description=r"Plot lattice $\xi F_0 / v_w^\alpha$ versus inferred $t_p$.")
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--outdir", type=str, default=str(OUTDIR))
    p.add_argument("--dpi", type=int, default=220)
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
    df = df.sort_values(["H", "theta", "v_w", "tp"]).reset_index(drop=True)
    return df


def build_quantity(df: pd.DataFrame, alpha: float):
    out = df.copy()
    out["xi_times_F0_over_vwalpha"] = (
        out["xi"].to_numpy(dtype=np.float64)
        * out["F0"].to_numpy(dtype=np.float64)
        / np.power(out["v_w"].to_numpy(dtype=np.float64), float(alpha))
    )
    return out


def plot_main(df: pd.DataFrame, alpha: float, outdir: Path, dpi: int):
    theta_values = np.sort(df["theta"].unique())
    h_values = np.sort(df["H"].unique())
    vw_values = np.sort(df["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {float(vw): cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    marker_map = {4.0: "o", 5.0: "s", 6.0: "^", 8.0: "D", 10.0: "v", 12.0: "P", 16.0: "X", 20.0: "<", 25.0: ">", 32.0: "h", 40.0: "*"}
    summary_rows = []

    for h in h_values:
        sub_h = df[np.isclose(df["H"], float(h), atol=1.0e-12, rtol=0.0)].copy()
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
        axes = axes.ravel()
        for ax, theta in zip(axes, theta_values):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
            for vw, cur in sub.groupby("v_w", sort=True):
                cur = cur.sort_values("tp")
                ax.plot(
                    cur["tp"],
                    cur["xi_times_F0_over_vwalpha"],
                    color=colors[float(vw)],
                    marker="o",
                    lw=1.5,
                    ms=3.2,
                    alpha=0.9,
                    label=rf"$v_w={float(vw):.1f}$",
                )
                for row in cur.itertuples(index=False):
                    ax.scatter(
                        [float(row.tp)],
                        [float(row.xi_times_F0_over_vwalpha)],
                        color=colors[float(vw)],
                        marker=marker_map.get(float(row.beta_over_H), "o"),
                        s=28,
                        alpha=0.95,
                    )
                if len(cur) >= 2:
                    y = cur["xi_times_F0_over_vwalpha"].to_numpy(dtype=np.float64)
                    summary_rows.append(
                        {
                            "H": float(h),
                            "theta": float(theta),
                            "v_w": float(vw),
                            "tp_min": float(cur["tp"].min()),
                            "tp_max": float(cur["tp"].max()),
                            "ratio_endpoints": float(y[-1] / np.maximum(y[0], 1.0e-18)),
                        }
                    )
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$t_p$")
            ax.set_ylabel(r"$\xi F_0 / v_w^\alpha$")
        for ax in axes[len(theta_values):]:
            ax.axis("off")
        handles = [plt.Line2D([0], [0], color=colors[float(vw)], lw=2.0) for vw in vw_values]
        labels = [rf"$v_w={float(vw):.1f}$" for vw in vw_values]
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
        fig.suptitle(rf"Lattice $\xi F_0 / v_w^\alpha$ vs inferred $t_p$, $H_*={float(h):.1f}$, $\alpha={float(alpha):.4f}$", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        h_tag = str(float(h)).replace(".", "p")
        fig.savefig(outdir / f"xi_times_F0_over_vwalpha_vs_tp_H{h_tag}.png", dpi=dpi)
        plt.close(fig)

    return summary_rows


def plot_separate(df: pd.DataFrame, alpha: float, outdir: Path, dpi: int):
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    vw_values = np.sort(df["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {float(vw): cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    marker_map = {4.0: "o", 5.0: "s", 6.0: "^", 8.0: "D", 10.0: "v", 12.0: "P", 16.0: "X", 20.0: "<", 25.0: ">", 32.0: "h", 40.0: "*"}

    for (h, theta), sub in df.groupby(["H", "theta"], sort=True):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for vw, cur in sub.groupby("v_w", sort=True):
            cur = cur.sort_values("tp")
            ax.plot(
                cur["tp"],
                cur["xi_times_F0_over_vwalpha"],
                color=colors[float(vw)],
                marker="o",
                lw=1.6,
                ms=4.0,
                alpha=0.9,
                label=rf"$v_w={float(vw):.1f}$",
            )
            for row in cur.itertuples(index=False):
                ax.scatter(
                    [float(row.tp)],
                    [float(row.xi_times_F0_over_vwalpha)],
                    color=colors[float(vw)],
                    marker=marker_map.get(float(row.beta_over_H), "o"),
                    s=32,
                    alpha=0.95,
                )
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$\xi F_0 / v_w^\alpha$")
        ax.set_title(rf"$H_*={float(h):.1f}$, $\theta={float(theta):.3f}$, $\alpha={float(alpha):.4f}$")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        h_tag = f"H{float(h):.1f}".replace(".", "p")
        theta_tag = f"theta_{float(theta):.10f}".replace(".", "p")
        path = outdir / f"xi_times_F0_over_vwalpha_vs_tp_{h_tag}_{theta_tag}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        rows.append({"H": float(h), "theta": float(theta), "file": str(path)})
    return rows


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("[load] reading lattice dataframe")
    df = load_lattice_dataframe(outdir, args.vw_folders, args.h_values)
    df = build_quantity(df, float(args.alpha))

    print("[plot] writing xi*F0/vw^alpha vs tp diagnostics")
    summary_rows = plot_main(df, float(args.alpha), outdir, args.dpi)
    separate_rows = plot_separate(df, float(args.alpha), outdir / "xi_times_F0_over_vwalpha_vs_tp_separate", args.dpi)
    pd.DataFrame(separate_rows).to_csv(outdir / "xi_times_F0_over_vwalpha_vs_tp_index.csv", index=False)

    summary = {
        "status": "ok",
        "alpha": float(args.alpha),
        "h_values": [float(v) for v in np.sort(df["H"].unique())],
        "vw_values": [float(v) for v in np.sort(df["v_w"].unique())],
        "theta_values": [float(v) for v in np.sort(df["theta"].unique())],
        "outputs": {
            **{f"H{str(float(h)).replace('.', 'p')}": str(outdir / f"xi_times_F0_over_vwalpha_vs_tp_H{str(float(h)).replace('.', 'p')}.png") for h in np.sort(df["H"].unique())},
            "separate_dir": str(outdir / "xi_times_F0_over_vwalpha_vs_tp_separate"),
            "index_csv": str(outdir / "xi_times_F0_over_vwalpha_vs_tp_index.csv"),
        },
        "summary_rows": summary_rows,
    }
    save_json(outdir / "final_summary.json", summary)
    print(json.dumps(to_native(summary), sort_keys=True))


if __name__ == "__main__":
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        main()
    except Exception as exc:
        payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
        save_json(outdir / "_error.json", payload)
        print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True))
        raise
