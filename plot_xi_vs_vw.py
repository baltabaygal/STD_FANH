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


OUTDIR = ROOT / "results_vw_xi_vs_vw"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(description="Plot lattice xi versus v_w, grouped by H* and theta.")
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
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
    return df.sort_values(["H", "theta", "beta_over_H", "v_w"]).reset_index(drop=True)


def plot_main(df: pd.DataFrame, outdir: Path, dpi: int):
    theta_values = np.sort(df["theta"].unique())
    h_values = np.sort(df["H"].unique())
    summary_rows = []
    for h in h_values:
        sub_h = df[np.isclose(df["H"], float(h), atol=1.0e-12, rtol=0.0)].copy()
        beta_vals = np.sort(sub_h["beta_over_H"].unique())
        cmap = plt.get_cmap("viridis")
        colors = {float(b): cmap(i / max(len(beta_vals) - 1, 1)) for i, b in enumerate(beta_vals)}

        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
        axes = axes.ravel()
        for ax, theta in zip(axes, theta_values):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
            for beta_val, cur in sub.groupby("beta_over_H", sort=True):
                cur = cur.sort_values("v_w")
                ax.plot(
                    cur["v_w"],
                    cur["xi"],
                    color=colors[float(beta_val)],
                    marker="o",
                    lw=1.4,
                    ms=3.0,
                    alpha=0.8,
                )
                summary_rows.append(
                    {
                        "H": float(h),
                        "theta": float(theta),
                        "beta_over_H": float(beta_val),
                        "vw_min": float(cur["v_w"].min()),
                        "vw_max": float(cur["v_w"].max()),
                        "ratio_max_over_min": float(
                            np.max(cur["xi"].to_numpy(dtype=np.float64))
                            / np.maximum(np.min(cur["xi"].to_numpy(dtype=np.float64)), 1.0e-18)
                        ),
                    }
                )
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$v_w$")
            ax.set_ylabel(r"$\xi$")
        for ax in axes[len(theta_values):]:
            ax.axis("off")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=float(np.min(beta_vals)), vmax=float(np.max(beta_vals))))
        cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.9)
        cbar.set_label(r"$\beta/H_*$")
        fig.suptitle(rf"Lattice $\xi$ vs $v_w$, $H_*={h:.1f}$", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        tag = str(float(h)).replace(".", "p")
        fig.savefig(outdir / f"xi_vs_vw_H{tag}.png", dpi=dpi)
        plt.close(fig)
    return summary_rows


def plot_separate(df: pd.DataFrame, outdir: Path, dpi: int):
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for (h, theta), sub in df.groupby(["H", "theta"], sort=True):
        beta_vals = np.sort(sub["beta_over_H"].unique())
        cmap = plt.get_cmap("viridis")
        colors = {float(b): cmap(i / max(len(beta_vals) - 1, 1)) for i, b in enumerate(beta_vals)}
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for beta_val, cur in sub.groupby("beta_over_H", sort=True):
            cur = cur.sort_values("v_w")
            ax.plot(
                cur["v_w"],
                cur["xi"],
                color=colors[float(beta_val)],
                marker="o",
                lw=1.6,
                ms=4.0,
                alpha=0.9,
                label=rf"$\beta/H_*={float(beta_val):.3g}$",
            )
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"$v_w$")
        ax.set_ylabel(r"$\xi$")
        ax.set_title(rf"$H_*={float(h):.1f}$, $\theta={float(theta):.3f}$")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        h_tag = f"H{float(h):.1f}".replace(".", "p")
        theta_tag = f"theta_{float(theta):.10f}".replace(".", "p")
        path = outdir / f"xi_vs_vw_{h_tag}_{theta_tag}.png"
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

    print("[plot] writing xi vs vw diagnostics")
    summary_rows = plot_main(df, outdir, args.dpi)
    separate_rows = plot_separate(df, outdir / "xi_vs_vw_separate", args.dpi)
    pd.DataFrame(separate_rows).to_csv(outdir / "xi_vs_vw_index.csv", index=False)

    summary = {
        "status": "ok",
        "h_values": [float(v) for v in np.sort(df["H"].unique())],
        "vw_values": [float(v) for v in np.sort(df["v_w"].unique())],
        "theta_values": [float(v) for v in np.sort(df["theta"].unique())],
        "outputs": {
            **{f"H{str(float(h)).replace('.', 'p')}": str(outdir / f"xi_vs_vw_H{str(float(h)).replace('.', 'p')}.png") for h in np.sort(df["H"].unique())},
            "separate_dir": str(outdir / "xi_vs_vw_separate"),
            "index_csv": str(outdir / "xi_vs_vw_index.csv"),
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
