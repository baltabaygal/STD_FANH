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


OUTDIR = ROOT / "results_vw_factorization_tests"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]
VW_REF = 0.9


def parse_args():
    p = argparse.ArgumentParser(
        description="Test simple v_w factorization diagnostics on lattice xi."
    )
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
    df = df.sort_values(["H", "theta", "beta_over_H", "v_w"]).reset_index(drop=True)
    return df


def prepare_diagnostics(df: pd.DataFrame):
    ref = (
        df[np.isclose(df["v_w"], VW_REF, atol=1.0e-12, rtol=0.0)][
            ["H", "theta", "beta_over_H", "xi"]
        ]
        .rename(columns={"xi": "xi_ref_vw09"})
        .copy()
    )
    merged = df.merge(ref, on=["H", "theta", "beta_over_H"], how="left", validate="many_to_one")
    merged["xi_ratio_to_vw09"] = merged["xi"] / np.maximum(merged["xi_ref_vw09"], 1.0e-18)
    merged["xi_over_tp32"] = merged["xi"] / np.maximum(
        np.power(merged["tp"].to_numpy(dtype=np.float64), 1.5),
        1.0e-18,
    )
    ref_tp = (
        merged[np.isclose(merged["v_w"], VW_REF, atol=1.0e-12, rtol=0.0)][
            ["H", "theta", "beta_over_H", "xi_over_tp32"]
        ]
        .rename(columns={"xi_over_tp32": "xi_over_tp32_ref_vw09"})
        .copy()
    )
    merged = merged.merge(
        ref_tp,
        on=["H", "theta", "beta_over_H"],
        how="left",
        validate="many_to_one",
    )
    merged["xi_over_tp32_ratio_to_vw09"] = merged["xi_over_tp32"] / np.maximum(
        merged["xi_over_tp32_ref_vw09"],
        1.0e-18,
    )
    return merged


def plot_grid(df: pd.DataFrame, ycol: str, ylabel: str, title_prefix: str, stem: str, outdir: Path, dpi: int):
    theta_values = np.sort(df["theta"].unique())
    h_values = np.sort(df["H"].unique())
    beta_vals_all = np.sort(df["beta_over_H"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {
        float(b): cmap(i / max(len(beta_vals_all) - 1, 1))
        for i, b in enumerate(beta_vals_all)
    }

    summary_rows = []
    for h in h_values:
        sub_h = df[np.isclose(df["H"], float(h), atol=1.0e-12, rtol=0.0)].copy()
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
        axes = axes.ravel()
        for ax, theta in zip(axes, theta_values):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
            for beta_val, cur in sub.groupby("beta_over_H", sort=True):
                cur = cur.sort_values("v_w")
                ax.plot(
                    cur["v_w"],
                    cur[ycol],
                    color=colors[float(beta_val)],
                    marker="o",
                    lw=1.5,
                    ms=3.2,
                    alpha=0.9,
                )
                vals = cur[ycol].to_numpy(dtype=np.float64)
                summary_rows.append(
                    {
                        "plot": stem,
                        "H": float(h),
                        "theta": float(theta),
                        "beta_over_H": float(beta_val),
                        "spread_max_over_min": float(
                            np.max(vals) / np.maximum(np.min(vals), 1.0e-18)
                        ),
                        "spread_std_over_mean": float(
                            np.std(vals) / np.maximum(np.mean(vals), 1.0e-18)
                        ),
                    }
                )
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$v_w$")
            ax.set_ylabel(ylabel)
        for ax in axes[len(theta_values):]:
            ax.axis("off")
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=float(np.min(beta_vals_all)), vmax=float(np.max(beta_vals_all))),
        )
        cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.9)
        cbar.set_label(r"$\beta/H_*$")
        fig.suptitle(rf"{title_prefix}, $H_*={float(h):.1f}$", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        tag = str(float(h)).replace(".", "p")
        fig.savefig(outdir / f"{stem}_H{tag}.png", dpi=dpi)
        plt.close(fig)
    return summary_rows


def build_summary(df: pd.DataFrame):
    ratio = df["xi_ratio_to_vw09"].to_numpy(dtype=np.float64)
    ratio_tp = df["xi_over_tp32_ratio_to_vw09"].to_numpy(dtype=np.float64)
    nonref = ~np.isclose(df["v_w"].to_numpy(dtype=np.float64), VW_REF, atol=1.0e-12, rtol=0.0)
    return {
        "vw_reference": VW_REF,
        "xi_ratio_nonref_mean_abs_deviation_from_1": float(np.mean(np.abs(ratio[nonref] - 1.0))),
        "xi_ratio_nonref_std": float(np.std(ratio[nonref])),
        "xi_over_tp32_ratio_nonref_mean_abs_deviation_from_1": float(
            np.mean(np.abs(ratio_tp[nonref] - 1.0))
        ),
        "xi_over_tp32_ratio_nonref_std": float(np.std(ratio_tp[nonref])),
    }


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("[load] reading lattice dataframe")
    df = load_lattice_dataframe(outdir, args.vw_folders, args.h_values)
    print("[prep] building v_w normalization diagnostics")
    df = prepare_diagnostics(df)
    df.to_csv(outdir / "diagnostic_table.csv", index=False)

    print("[plot] writing xi/xi(vw=0.9) vs vw")
    rows_ratio = plot_grid(
        df,
        "xi_ratio_to_vw09",
        r"$\xi/\xi(v_w=0.9)$",
        r"Lattice normalized ratio $\xi/\xi(v_w=0.9)$ vs $v_w$",
        "xi_ratio_to_vw09_vs_vw",
        outdir,
        args.dpi,
    )
    print("[plot] writing (xi/t_p^{3/2})/(vw=0.9) vs vw")
    rows_tp = plot_grid(
        df,
        "xi_over_tp32_ratio_to_vw09",
        r"$[\xi/t_p^{3/2}] / [\xi/t_p^{3/2}]_{v_w=0.9}$",
        r"Lattice normalized ratio $[\xi/t_p^{3/2}] / [\xi/t_p^{3/2}]_{v_w=0.9}$ vs $v_w$",
        "xi_over_tp32_ratio_to_vw09_vs_vw",
        outdir,
        args.dpi,
    )

    summary = {
        "status": "ok",
        "h_values": [float(v) for v in np.sort(df["H"].unique())],
        "vw_values": [float(v) for v in np.sort(df["v_w"].unique())],
        "theta_values": [float(v) for v in np.sort(df["theta"].unique())],
        "aggregate": build_summary(df),
        "outputs": {
            **{
                f"xi_ratio_H{str(float(h)).replace('.', 'p')}": str(
                    outdir / f"xi_ratio_to_vw09_vs_vw_H{str(float(h)).replace('.', 'p')}.png"
                )
                for h in np.sort(df["H"].unique())
            },
            **{
                f"xi_over_tp32_ratio_H{str(float(h)).replace('.', 'p')}": str(
                    outdir / f"xi_over_tp32_ratio_to_vw09_vs_vw_H{str(float(h)).replace('.', 'p')}.png"
                )
                for h in np.sort(df["H"].unique())
            },
            "diagnostic_table": str(outdir / "diagnostic_table.csv"),
        },
        "summary_rows": rows_ratio + rows_tp,
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
