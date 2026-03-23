#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RUNS = {
    0.3: ROOT / "results_gamma_v3_H15H20_scalar" / "final_summary.json",
    0.5: ROOT / "results_gamma_v5_H15H20_scalar" / "final_summary.json",
    0.7: ROOT / "results_gamma_v7_H15H20_scalar" / "final_summary.json",
    0.9: ROOT / "results_gamma_v9_H15H20_scalar" / "final_summary.json",
}
BOOTSTRAPS = {
    0.3: ROOT / "results_gamma_v3_H15H20_scalar" / "bootstrap_twoexp_fit.json",
    0.5: ROOT / "results_gamma_v5_H15H20_scalar" / "bootstrap_twoexp_fit.json",
    0.7: ROOT / "results_gamma_v7_H15H20_scalar" / "bootstrap_twoexp_fit.json",
    0.9: ROOT / "results_gamma_v9_H15H20_scalar" / "bootstrap_twoexp_fit.json",
}
OUTDIR = ROOT / "results_gamma_vw_compare"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for vw, path in RUNS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing summary for v_w={vw}: {path}")
        payload = json.loads(path.read_text())
        boot = json.loads(BOOTSTRAPS[vw].read_text())
        rows.append(
            {
                "vw": vw,
                "gamma_pooled": payload["gamma_pooled"],
                "gamma_fit": payload["twoexp_fit"]["gamma"],
                "gamma_fit_p16": boot["gamma"]["p16"],
                "gamma_fit_p84": boot["gamma"]["p84"],
                "t_c": payload["twoexp_fit"]["t_c"],
                "t_c_p16": boot["t_c"]["p16"],
                "t_c_p84": boot["t_c"]["p84"],
                "r": payload["twoexp_fit"]["r"],
                "r_p16": boot["r"]["p16"],
                "r_p84": boot["r"]["p84"],
                "rel_rmse": payload["twoexp_fit"]["rel_rmse"],
                "delta_AIC": payload["model_comparison"]["delta_AIC_scalar_minus_original"],
                "delta_BIC": payload["model_comparison"]["delta_BIC_scalar_minus_original"],
            }
        )
    df = pd.DataFrame(rows).sort_values("vw").reset_index(drop=True)
    df.to_csv(OUTDIR / "vw_gamma_summary.csv", index=False)
    (OUTDIR / "vw_gamma_summary.json").write_text(df.to_json(orient="records", indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8))
    axes = axes.flatten()

    axes[0].errorbar(
        df["vw"],
        df["gamma_fit"],
        yerr=[
            np.maximum(df["gamma_fit"] - df["gamma_fit_p16"], 0.0),
            np.maximum(df["gamma_fit_p84"] - df["gamma_fit"], 0.0),
        ],
        fmt="o-",
        color="tab:blue",
        label=r"fit $\gamma$",
    )
    axes[0].plot(df["vw"], df["gamma_pooled"], "s--", color="black", label=r"pooled $\gamma$")
    axes[0].set_xlabel(r"$v_w$")
    axes[0].set_ylabel(r"$\gamma$")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].errorbar(
        df["vw"],
        df["t_c"],
        yerr=[
            np.maximum(df["t_c"] - df["t_c_p16"], 0.0),
            np.maximum(df["t_c_p84"] - df["t_c"], 0.0),
        ],
        fmt="o-",
        color="tab:red",
    )
    axes[1].set_xlabel(r"$v_w$")
    axes[1].set_ylabel(r"$t_c$")
    axes[1].grid(alpha=0.25)

    axes[2].errorbar(
        df["vw"],
        df["r"],
        yerr=[
            np.maximum(df["r"] - df["r_p16"], 0.0),
            np.maximum(df["r_p84"] - df["r"], 0.0),
        ],
        fmt="o-",
        color="tab:green",
    )
    axes[2].set_xlabel(r"$v_w$")
    axes[2].set_ylabel(r"$r$")
    axes[2].grid(alpha=0.25)

    axes[3].plot(df["vw"], df["rel_rmse"], "o-", color="tab:purple", label="rel-RMSE")
    ax2 = axes[3].twinx()
    ax2.plot(df["vw"], df["delta_AIC"], "s--", color="black", label=r"$\Delta$AIC")
    axes[3].set_xlabel(r"$v_w$")
    axes[3].set_ylabel("rel-RMSE", color="tab:purple")
    ax2.set_ylabel(r"$\Delta$AIC", color="black")
    axes[3].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTDIR / "vw_gamma_comparison.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
