#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_PRED = ROOT / "results_vw0p9_model_applied_all_vw" / "predictions.csv"
DEFAULT_SUMMARY = ROOT / "results_vw0p9_model_applied_all_vw" / "final_summary.json"
DEFAULT_OUTDIR = ROOT / "results_vw0p9_model_ratio_vs_vw"
T_OSC = 1.5


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot xi/model ratios vs v_w for the frozen v_w=0.9 baseline model."
    )
    p.add_argument("--predictions", type=str, default=str(DEFAULT_PRED))
    p.add_argument("--summary", type=str, default=str(DEFAULT_SUMMARY))
    p.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
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


def build_diagnostics(df: pd.DataFrame):
    df = df.copy()
    df["ratio_exact_model"] = df["xi"] / np.maximum(df["xi_fit_vw0p9_model"], 1.0e-18)
    # Reconstruct the implied \tilde f of the frozen v_w=0.9 model.
    x_over_tosc_32 = np.power(np.maximum(df["x"].to_numpy(dtype=np.float64) / T_OSC, 1.0e-18), 1.5)
    tp_over_tosc_32 = np.power(np.maximum(df["tp"].to_numpy(dtype=np.float64) / T_OSC, 1.0e-18), 1.5)
    df["ftilde_vw0p9"] = df["F0"] * df["xi_fit_vw0p9_model"] / np.maximum(x_over_tosc_32, 1.0e-18)
    denom_tp = tp_over_tosc_32 * df["ftilde_vw0p9"] / np.maximum(df["F0"], 1.0e-18)
    df["ratio_tp_ftilde"] = df["xi"] / np.maximum(denom_tp, 1.0e-18)
    return df


def plot_grid(df: pd.DataFrame, ycol: str, ylabel: str, title_prefix: str, stem: str, outdir: Path, dpi: int):
    theta_values = np.sort(df["theta"].unique())
    h_values = np.sort(df["H"].unique())
    beta_vals = np.sort(df["beta_over_H"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {float(b): cmap(i / max(len(beta_vals) - 1, 1)) for i, b in enumerate(beta_vals)}

    rows = []
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
                    ms=3.0,
                    alpha=0.85,
                )
                vals = cur[ycol].to_numpy(dtype=np.float64)
                rows.append(
                    {
                        "plot": stem,
                        "H": float(h),
                        "theta": float(theta),
                        "beta_over_H": float(beta_val),
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "max_abs_dev_from_1": float(np.max(np.abs(vals - 1.0))),
                    }
                )
            ax.axhline(1.0, color="black", lw=0.8, alpha=0.5)
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$v_w$")
            ax.set_ylabel(ylabel)
        for ax in axes[len(theta_values):]:
            ax.axis("off")
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=float(np.min(beta_vals)), vmax=float(np.max(beta_vals))),
        )
        cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.9)
        cbar.set_label(r"$\beta/H_*$")
        fig.suptitle(rf"{title_prefix}, $H_*={float(h):.1f}$", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        tag = str(float(h)).replace(".", "p")
        fig.savefig(outdir / f"{stem}_H{tag}.png", dpi=dpi)
        plt.close(fig)
    return rows


def summarize(df: pd.DataFrame):
    rows = []
    for col in ["ratio_exact_model", "ratio_tp_ftilde"]:
        vals = df[col].to_numpy(dtype=np.float64)
        rows.append(
            {
                "quantity": col,
                "mean_abs_dev_from_1": float(np.mean(np.abs(vals - 1.0))),
                "std": float(np.std(vals)),
                "max_abs_dev_from_1": float(np.max(np.abs(vals - 1.0))),
            }
        )
    return rows


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(args.predictions).resolve())
    summary_src = json.loads(Path(args.summary).resolve().read_text())
    df = build_diagnostics(df)
    df.to_csv(outdir / "ratio_diagnostic_table.csv", index=False)

    rows_exact = plot_grid(
        df,
        "ratio_exact_model",
        r"$\xi / \xi_{\rm model}^{(v_w=0.9)}$",
        r"Frozen $v_w=0.9$ baseline ratio $\xi / \xi_{\rm model}$ vs $v_w$",
        "ratio_exact_model_vs_vw",
        outdir,
        args.dpi,
    )
    rows_tp = plot_grid(
        df,
        "ratio_tp_ftilde",
        r"$\xi \,/\, [ (t_p/t_{\rm osc})^{3/2}\,\tilde f_{v_w=0.9}/F_0 ]$",
        r"Frozen $v_w=0.9$ baseline ratio using $t_p^{3/2}\tilde f/F_0$ vs $v_w$",
        "ratio_tp_ftilde_vs_vw",
        outdir,
        args.dpi,
    )

    summary = {
        "status": "ok",
        "source_predictions": str(Path(args.predictions).resolve()),
        "source_summary": str(Path(args.summary).resolve()),
        "source_beta": float(summary_src["beta"]),
        "aggregate": summarize(df),
        "outputs": {
            **{
                f"exact_H{str(float(h)).replace('.', 'p')}": str(
                    outdir / f"ratio_exact_model_vs_vw_H{str(float(h)).replace('.', 'p')}.png"
                )
                for h in np.sort(df["H"].unique())
            },
            **{
                f"tpftilde_H{str(float(h)).replace('.', 'p')}": str(
                    outdir / f"ratio_tp_ftilde_vs_vw_H{str(float(h)).replace('.', 'p')}.png"
                )
                for h in np.sort(df["H"].unique())
            },
            "diagnostic_table": str(outdir / "ratio_diagnostic_table.csv"),
        },
        "summary_rows": rows_exact + rows_tp,
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
