import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results_transport_diagnostics_suite" / "transport_diagnostics_table.csv"
OUTDIR = ROOT / "results_sweeptime_residual_collapse"


def log_interp(x_ref, y_ref, x_new):
    x_ref = np.asarray(x_ref, dtype=np.float64)
    y_ref = np.asarray(y_ref, dtype=np.float64)
    x_new = np.asarray(x_new, dtype=np.float64)
    good = np.isfinite(x_ref) & np.isfinite(y_ref) & (x_ref > 0.0) & (y_ref > 0.0)
    if np.count_nonzero(good) < 3:
        return np.full_like(x_new, np.nan, dtype=np.float64)
    x_ref = x_ref[good]
    y_ref = y_ref[good]
    order = np.argsort(x_ref)
    x_ref = x_ref[order]
    y_ref = y_ref[order]
    lx = np.log(x_ref)
    ly = np.log(y_ref)
    out = np.full_like(x_new, np.nan, dtype=np.float64)
    mask = np.isfinite(x_new) & (x_new >= x_ref.min()) & (x_new <= x_ref.max())
    if np.any(mask):
        out[mask] = np.exp(np.interp(np.log(x_new[mask]), lx, ly))
    return out


def collapse_score(df, x_col, ref_vw=0.9):
    rows = []
    all_resid = []
    for (H, theta), sub in df.groupby(["H", "theta"]):
        ref = sub[np.isclose(sub["v_w"], float(ref_vw), atol=1.0e-12)].sort_values(x_col)
        if len(ref) < 3:
            continue
        x_ref = ref[x_col].to_numpy(dtype=np.float64)
        y_ref = ref["ratio_teff"].to_numpy(dtype=np.float64)
        for vw, cur in sub.groupby("v_w"):
            cur = cur.sort_values(x_col)
            x_cur = cur[x_col].to_numpy(dtype=np.float64)
            y_cur = cur["ratio_teff"].to_numpy(dtype=np.float64)
            pred = log_interp(x_ref, y_ref, x_cur)
            mask = np.isfinite(pred) & np.isfinite(y_cur)
            overlap = int(np.count_nonzero(mask))
            frac = float(overlap / max(len(cur), 1))
            if overlap >= 3:
                resid = (pred[mask] - y_cur[mask]) / np.maximum(y_cur[mask], 1.0e-12)
                rmse = float(np.sqrt(np.mean(resid**2)))
                mean_abs = float(np.mean(np.abs(resid)))
                all_resid.append(resid)
            else:
                rmse = np.nan
                mean_abs = np.nan
            rows.append(
                {
                    "H": float(H),
                    "theta": float(theta),
                    "v_w": float(vw),
                    "n_total": int(len(cur)),
                    "n_overlap": overlap,
                    "overlap_frac": frac,
                    "rel_rmse": rmse,
                    "mean_abs_frac_resid": mean_abs,
                }
            )
    detail = pd.DataFrame(rows)
    pooled = np.concatenate(all_resid) if all_resid else np.array([], dtype=np.float64)
    summary = {
        "pooled_rel_rmse": float(np.sqrt(np.mean(pooled**2))) if pooled.size else np.nan,
        "pooled_mean_abs_frac_resid": float(np.mean(np.abs(pooled))) if pooled.size else np.nan,
        "mean_overlap_frac": float(detail["overlap_frac"].mean()) if len(detail) else np.nan,
        "median_overlap_frac": float(detail["overlap_frac"].median()) if len(detail) else np.nan,
    }
    return detail, summary


def plot_by_H(df, x_col, xlabel, stem):
    thetas = sorted(df["theta"].unique())
    for H in sorted(df["H"].unique()):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
        axes = axes.ravel()
        for ax, theta in zip(axes, thetas):
            sub = df[(np.isclose(df["H"], H)) & (np.isclose(df["theta"], theta))].sort_values(x_col)
            for vw, cur in sub.groupby("v_w", sort=True):
                cur = cur.sort_values(x_col)
                ax.plot(cur[x_col], cur["ratio_teff"], marker="o", ms=3.0, lw=1.5, label=fr"$v_w={vw:.1f}$")
            ax.axhline(1.0, color="black", lw=1.0, alpha=0.5)
            ax.set_xscale("log")
            ax.set_title(fr"$\theta={theta:.3f}$")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$\xi/\xi_{v9}(t_{\rm eff})$")
            ax.grid(alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles[:4], labels[:4], loc="upper center", ncol=4, frameon=False)
        fig.suptitle(f"Timing-corrected residual vs {xlabel}, $H_*={H:.1f}$", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(OUTDIR / f"{stem}_H{str(H).replace('.', 'p')}.png", dpi=180)
        plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)
    df["u_tp_vw"] = df["tp"] * df["v_w"]
    df["u_tp_vw_H"] = df["tp"] * df["v_w"] * df["H"]

    variables = [
        ("tp", r"$t_p$", "ratio_vs_tp"),
        ("u_tp_vw", r"$u=t_p\,v_w$", "ratio_vs_u_tpvw"),
        ("u_tp_vw_H", r"$u_H=t_p\,v_w\,H_*$", "ratio_vs_u_tpvwH"),
    ]

    summaries = {}
    model_rows = []
    for x_col, xlabel, stem in variables:
        detail, summary = collapse_score(df, x_col)
        detail.to_csv(OUTDIR / f"{stem}_collapse_detail.csv", index=False)
        summaries[x_col] = summary
        model_rows.append({"x_variable": x_col, **summary})
        plot_by_H(df, x_col, xlabel, stem)

    comp = pd.DataFrame(model_rows).sort_values("pooled_rel_rmse")
    comp.to_csv(OUTDIR / "collapse_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(comp["x_variable"], comp["pooled_rel_rmse"])
    ax.set_ylabel("pooled relative RMSE")
    ax.set_title("Residual collapse score by x-variable")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "collapse_score_comparison.png", dpi=180)
    plt.close(fig)

    with open(OUTDIR / "final_summary.json", "w") as f:
        json.dump(
            {
                "input": str(INPUT),
                "summaries": summaries,
                "outputs": {
                    "comparison": str(OUTDIR / "collapse_comparison.csv"),
                    "score_plot": str(OUTDIR / "collapse_score_comparison.png"),
                },
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
