#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import collapse_and_fit_fanh_tosc as cf


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "results_fanh_theta_behavior"
VW_MAP = {"v3": 0.3, "v5": 0.5, "v7": 0.7, "v9": 0.9}
TARGET_H = [1.5, 2.0]


def compute_x(df: pd.DataFrame, beta_mode: str, beta_value: float):
    out = df.copy()
    if beta_mode == "tp":
        out["x"] = out["tp"].to_numpy(dtype=np.float64)
    else:
        out["x"] = out["tp"].to_numpy(dtype=np.float64) * np.power(out["H"].to_numpy(dtype=np.float64), beta_value)
    return out


def load_dataset(tag: str, beta_mode: str):
    raw = ROOT / "lattice_data" / "data" / f"energy_ratio_by_theta_data_{tag}.txt"
    rho = cf.resolve_first_existing(cf.RHO_CANDIDATES, "")
    f0 = cf.load_f0_table(rho, TARGET_H)
    df = cf.load_lattice_data(raw, VW_MAP[tag], TARGET_H)
    df = cf.merge_f0(df, f0)
    fit_dir = ROOT / (
        f"results_collapse_fixed_tc_tp_{tag}_H15H20_tosc"
        if beta_mode == "tp"
        else f"results_collapse_fixed_tc_{tag}_H15H20_tosc"
    )
    summary = json.loads((fit_dir / "final_summary.json").read_text())
    fit = json.loads((fit_dir / "global_fit.json").read_text())
    beta = float(summary["beta"])
    df = compute_x(df, beta_mode, beta)
    df["fanh"] = df["xi"].to_numpy(dtype=np.float64) * df["F0"].to_numpy(dtype=np.float64) / np.power(
        df["tp"].to_numpy(dtype=np.float64) / cf.T_OSC, 1.5
    )
    theta_values = np.array(sorted(float(k) for k in fit["F_inf"].keys()), dtype=np.float64)
    finf = np.array([fit["F_inf"][f"{th:.10f}"]["value"] for th in theta_values], dtype=np.float64)
    r = float(fit["r"])
    t_c = float(fit["t_c"])
    return df, theta_values, finf, r, t_c, beta


def interpolate_rel_split(sub: pd.DataFrame):
    hs = sorted(float(h) for h in sub["H"].unique())
    if len(hs) != 2:
        return np.nan, np.nan
    a = sub[np.isclose(sub["H"], hs[0], atol=1e-12)].sort_values("x")
    b = sub[np.isclose(sub["H"], hs[1], atol=1e-12)].sort_values("x")
    xa = a["x"].to_numpy(dtype=np.float64)
    xb = b["x"].to_numpy(dtype=np.float64)
    ya = a["fanh"].to_numpy(dtype=np.float64)
    yb = b["fanh"].to_numpy(dtype=np.float64)
    lo = max(np.min(xa), np.min(xb))
    hi = min(np.max(xa), np.max(xb))
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        return np.nan, np.nan
    xg = np.geomspace(lo, hi, 100)
    fa = interp1d(np.log(xa), np.log(ya), bounds_error=False, fill_value=np.nan)
    fb = interp1d(np.log(xb), np.log(yb), bounds_error=False, fill_value=np.nan)
    ya_i = np.exp(fa(np.log(xg)))
    yb_i = np.exp(fb(np.log(xg)))
    mask = np.isfinite(ya_i) & np.isfinite(yb_i) & (ya_i > 0) & (yb_i > 0)
    if np.sum(mask) < 10:
        return np.nan, np.nan
    rel = np.abs(ya_i[mask] - yb_i[mask]) / np.maximum(0.5 * (ya_i[mask] + yb_i[mask]), 1e-18)
    return float(np.mean(rel)), float(np.max(rel))


def analyze_mode(beta_mode: str):
    rows = []
    for tag, vw in VW_MAP.items():
        df, theta_values, finf, r, t_c, beta = load_dataset(tag, beta_mode)
        for theta, sub in df.groupby("theta", sort=True):
            idx = cf.nearest_theta(theta_values, theta)
            x = sub["x"].to_numpy(dtype=np.float64)
            f0 = sub["F0"].to_numpy(dtype=np.float64)
            fanh = sub["fanh"].to_numpy(dtype=np.float64)
            fanh_fit = finf[idx] / np.maximum(f0, 1e-18) + f0 / (
                np.power(x / cf.T_OSC, 1.5) * (1.0 + np.power(x / max(t_c, 1e-12), r))
            )
            rel = np.abs(fanh_fit - fanh) / np.maximum(fanh, 1e-18)
            hsplit_mean, hsplit_max = interpolate_rel_split(sub)
            rec = {
                "mode": beta_mode,
                "tag": tag,
                "vw": vw,
                "beta": beta,
                "theta": float(theta),
                "r": r,
                "t_c": t_c,
                "model_rel_mean": float(np.mean(rel)),
                "model_rel_max": float(np.max(rel)),
                "hsplit_mean": hsplit_mean,
                "hsplit_max": hsplit_max,
            }
            for H in TARGET_H:
                hsub = sub[np.isclose(sub["H"], H, atol=1e-12)]
                xh = hsub["x"].to_numpy(dtype=np.float64)
                f0h = hsub["F0"].to_numpy(dtype=np.float64)
                yh = hsub["fanh"].to_numpy(dtype=np.float64)
                yfit = finf[idx] / np.maximum(f0h, 1e-18) + f0h / (
                    np.power(xh / cf.T_OSC, 1.5) * (1.0 + np.power(xh / max(t_c, 1e-12), r))
                )
                signed = (yfit - yh) / np.maximum(yh, 1e-18)
                rec[f"signed_mean_H{str(H).replace('.', 'p')}"] = float(np.mean(signed))
            rows.append(rec)
    return pd.DataFrame(rows).sort_values(["vw", "theta"]).reset_index(drop=True)


def plot_heatmap(df: pd.DataFrame, value_col: str, path: Path, title: str):
    pivot = df.pivot(index="theta", columns="vw", values=value_col).sort_index()
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    pcm = ax.pcolormesh(
        np.arange(len(pivot.columns) + 1),
        np.arange(len(pivot.index) + 1),
        pivot.to_numpy(dtype=np.float64),
        shading="auto",
        cmap="coolwarm",
    )
    ax.set_xticks(np.arange(len(pivot.columns)) + 0.5)
    ax.set_xticklabels([f"{c:.1f}" for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)) + 0.5)
    ax.set_yticklabels([f"{y:.3f}" for y in pivot.index])
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$\theta$")
    ax.set_title(title)
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tp_df = analyze_mode("tp")
    beta_df = analyze_mode("beta")
    tp_df.to_csv(OUTDIR / "summary_tp.csv", index=False)
    beta_df.to_csv(OUTDIR / "summary_beta.csv", index=False)

    plot_heatmap(tp_df, "model_rel_mean", OUTDIR / "model_rel_mean_tp.png", r"Mean model rel. error, $x=t_p$")
    plot_heatmap(tp_df, "hsplit_mean", OUTDIR / "hsplit_mean_tp.png", r"Mean $H_*$ split in fanh, $x=t_p$")
    plot_heatmap(beta_df, "model_rel_mean", OUTDIR / "model_rel_mean_beta.png", r"Mean model rel. error, $x=t_p H^\beta$")
    plot_heatmap(beta_df, "hsplit_mean", OUTDIR / "hsplit_mean_beta.png", r"Mean $H_*$ split in fanh, $x=t_p H^\beta$")

    summary = {
        "status": "ok",
        "main_findings": {
            "tp": {
                "worst_model_slice": tp_df.sort_values("model_rel_mean", ascending=False).iloc[0][["vw", "theta", "model_rel_mean"]].to_dict(),
                "worst_hsplit_slice": tp_df.sort_values("hsplit_mean", ascending=False).iloc[0][["vw", "theta", "hsplit_mean"]].to_dict(),
            },
            "beta": {
                "worst_model_slice": beta_df.sort_values("model_rel_mean", ascending=False).iloc[0][["vw", "theta", "model_rel_mean"]].to_dict(),
                "worst_hsplit_slice": beta_df.sort_values("hsplit_mean", ascending=False).iloc[0][["vw", "theta", "hsplit_mean"]].to_dict(),
            },
        },
    }
    cf.save_json(OUTDIR / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
