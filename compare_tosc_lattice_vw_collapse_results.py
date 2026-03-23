#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
RUNS = {
    0.3: ROOT / "results_tosc_lattice_vw0p3_H1p0H1p5H2p0" / "collapse_and_fit_fanh",
    0.5: ROOT / "results_tosc_lattice_vw0p5_H1p0H1p5H2p0" / "collapse_and_fit_fanh",
    0.7: ROOT / "results_tosc_lattice_vw0p7_H1p0H1p5H2p0" / "collapse_and_fit_fanh",
    0.9: ROOT / "results_tosc_lattice_vw0p9_H1p0H1p5H2p0" / "collapse_and_fit_fanh",
}
OUTDIR = ROOT / "results_tosc_lattice_vw_compare_H1p0H1p5H2p0"


def load_summary(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing summary: {path}")
    return json.loads(path.read_text())


def build_table():
    rows = []
    for vw, run_dir in RUNS.items():
        payload = load_summary(run_dir / "final_summary.json")
        rows.append(
            {
                "vw": vw,
                "beta": payload["beta"],
                "collapse_score": payload["collapse_score"],
                "t_c": payload["global_fit"]["t_c"],
                "r": payload["global_fit"]["r"],
                "rel_rmse": payload["global_fit"]["rel_rmse"],
                "AIC": payload["global_fit"]["AIC"],
                "BIC": payload["global_fit"]["BIC"],
                "t_c_p16": payload["bootstrap_68"]["t_c"][0],
                "t_c_p84": payload["bootstrap_68"]["t_c"][1],
                "r_p16": payload["bootstrap_68"]["r"][0],
                "r_p84": payload["bootstrap_68"]["r"][1],
                "n_points": payload["n_lattice_points"],
            }
        )
    return pd.DataFrame(rows).sort_values("vw").reset_index(drop=True)


def plot_metric_summary(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    axes = axes.flatten()

    axes[0].plot(df["vw"], df["beta"], "o-", color="tab:blue")
    axes[0].set_xlabel(r"$v_w$")
    axes[0].set_ylabel(r"best $\beta$")
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(
        df["vw"],
        df["t_c"],
        yerr=[(df["t_c"] - df["t_c_p16"]).abs(), (df["t_c_p84"] - df["t_c"]).abs()],
        fmt="o-",
        color="tab:red",
    )
    axes[1].set_xlabel(r"$v_w$")
    axes[1].set_ylabel(r"$t_c$")
    axes[1].grid(alpha=0.25)

    axes[2].errorbar(
        df["vw"],
        df["r"],
        yerr=[(df["r"] - df["r_p16"]).abs(), (df["r_p84"] - df["r"]).abs()],
        fmt="o-",
        color="tab:green",
    )
    axes[2].set_xlabel(r"$v_w$")
    axes[2].set_ylabel(r"$r$")
    axes[2].grid(alpha=0.25)

    axes[3].plot(df["vw"], df["rel_rmse"], "o-", color="tab:purple", label="rel-RMSE")
    ax2 = axes[3].twinx()
    ax2.plot(df["vw"], df["collapse_score"], "s--", color="black", label="collapse score")
    axes[3].set_xlabel(r"$v_w$")
    axes[3].set_ylabel("rel-RMSE", color="tab:purple")
    ax2.set_ylabel("collapse score", color="black")
    axes[3].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTDIR / "vw_collapse_metrics.png", dpi=220)
    plt.close(fig)


def plot_overlay_grid():
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    axes = axes.flatten()
    for ax, vw in zip(axes, sorted(RUNS)):
        img_path = RUNS[vw] / "collapse_overlay.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing overlay image: {img_path}")
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(rf"$v_w={vw:.1f}$", fontsize=14)
        ax.axis("off")
    fig.savefig(OUTDIR / "collapse_overlay_by_vw.png", dpi=220)
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = build_table()
    df.to_csv(OUTDIR / "vw_collapse_summary.csv", index=False)
    (OUTDIR / "vw_collapse_summary.json").write_text(df.to_json(orient="records", indent=2))
    plot_metric_summary(df)
    plot_overlay_grid()


if __name__ == "__main__":
    main()
