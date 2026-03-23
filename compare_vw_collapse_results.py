#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
RUNS = {
    0.3: ROOT / "results_collapse_v3_H15H20" / "final_summary.json",
    0.5: ROOT / "results_collapse_v5_H15H20" / "final_summary.json",
    0.7: ROOT / "results_collapse_v7_H15H20" / "final_summary.json",
    0.9: ROOT / "results_collapse_v9_H15H20" / "final_summary.json",
}
OUTDIR = ROOT / "results_collapse_vw_compare"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for vw, path in RUNS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing summary for v_w={vw}: {path}")
        payload = json.loads(path.read_text())
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
                "inversion_r_global": payload["inversion_global"]["r_global"],
                "delta_AIC": payload["model_comparison"]["delta_AIC"],
                "delta_BIC": payload["model_comparison"]["delta_BIC"],
            }
        )

    df = pd.DataFrame(rows).sort_values("vw").reset_index(drop=True)
    df.to_csv(OUTDIR / "vw_collapse_summary.csv", index=False)
    (OUTDIR / "vw_collapse_summary.json").write_text(df.to_json(orient="records", indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    axes = axes.flatten()

    axes[0].plot(df["vw"], df["beta"], "o-", color="tab:blue")
    axes[0].set_xlabel(r"$v_w$")
    axes[0].set_ylabel(r"best $\beta$")
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(
        df["vw"],
        df["t_c"],
        yerr=[df["t_c"] - df["t_c_p16"], df["t_c_p84"] - df["t_c"]],
        fmt="o-",
        color="tab:red",
    )
    axes[1].set_xlabel(r"$v_w$")
    axes[1].set_ylabel(r"$t_c$")
    axes[1].grid(alpha=0.25)

    axes[2].errorbar(
        df["vw"],
        df["r"],
        yerr=[df["r"] - df["r_p16"], df["r_p84"] - df["r"]],
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
    fig.savefig(OUTDIR / "vw_collapse_comparison.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
