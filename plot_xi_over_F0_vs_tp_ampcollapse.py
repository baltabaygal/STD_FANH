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
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import fit_vw_amplitude as base_fit


OUTDIR = ROOT / "results_vw_xi_over_F0_ampcollapse"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]
COMPARE_ALPHAS = [0.0, -0.1318]


def parse_args():
    p = argparse.ArgumentParser(description="Collapse xi/F0 vs inferred tp with an amplitude factor v_w^alpha.")
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--alpha-min", type=float, default=-1.0)
    p.add_argument("--alpha-max", type=float, default=1.0)
    p.add_argument("--alpha-grid", type=int, default=121)
    p.add_argument("--grid-n", type=int, default=80)
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
    df["xi_over_F0"] = df["xi"] / np.maximum(df["F0"], 1.0e-18)
    return df


def loglog_interp(x, y, xq):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if np.count_nonzero(mask) < 2:
        return np.full_like(xq, np.nan, dtype=np.float64)
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    order = np.argsort(lx)
    fn = interp1d(lx[order], ly[order], kind="linear", bounds_error=False, fill_value=np.nan, assume_sorted=True)
    return np.exp(fn(np.log(np.maximum(xq, 1.0e-18))))


def scaled_y(cur: pd.DataFrame, alpha: float):
    return cur["xi_over_F0"].to_numpy(dtype=np.float64) / np.power(cur["v_w"].to_numpy(dtype=np.float64), float(alpha))


def collapse_score(df: pd.DataFrame, alpha: float, grid_n: int):
    scores = []
    per_group = []
    for (h, theta), sub in df.groupby(["H", "theta"], sort=True):
        groups = []
        for vw, cur in sub.groupby("v_w", sort=True):
            cur = cur.sort_values("tp")
            y = scaled_y(cur, alpha)
            mask = np.isfinite(cur["tp"]) & np.isfinite(y) & (cur["tp"] > 0.0) & (y > 0.0)
            if np.count_nonzero(mask) < 3:
                continue
            groups.append((float(vw), cur["tp"].to_numpy(dtype=np.float64)[mask], y[mask]))
        if len(groups) < 2:
            continue
        tmin = max(np.min(tp) for _, tp, _ in groups)
        tmax = min(np.max(tp) for _, tp, _ in groups)
        if (not np.isfinite(tmin)) or (not np.isfinite(tmax)) or tmax <= tmin:
            continue
        grid = np.geomspace(float(tmin), float(tmax), int(grid_n))
        arrs = []
        for _, tp, y in groups:
            vals = loglog_interp(tp, y, grid)
            arrs.append(vals)
        arr = np.vstack(arrs)
        valid = np.sum(np.isfinite(arr), axis=0) >= 2
        if not np.any(valid):
            continue
        mean = np.nanmean(arr[:, valid], axis=0)
        var = np.nanvar(arr[:, valid], axis=0, ddof=1)
        rel_var = var / np.maximum(mean * mean, 1.0e-18)
        score = float(np.nanmean(rel_var))
        if np.isfinite(score):
            scores.append(score)
            per_group.append({"H": float(h), "theta": float(theta), "score": score})
    return (float(np.mean(scores)) if scores else np.inf), per_group


def fit_best_alpha(df: pd.DataFrame, alpha_min: float, alpha_max: float, alpha_grid: int, grid_n: int):
    scan = np.linspace(float(alpha_min), float(alpha_max), int(alpha_grid), dtype=np.float64)
    scores = np.array([collapse_score(df, float(a), grid_n)[0] for a in scan], dtype=np.float64)
    best_idx = int(np.nanargmin(scores))
    bracket_lo = scan[max(best_idx - 1, 0)]
    bracket_hi = scan[min(best_idx + 1, len(scan) - 1)]

    def objective(alpha):
        return collapse_score(df, float(alpha), grid_n)[0]

    opt = minimize_scalar(objective, bounds=(float(bracket_lo), float(bracket_hi)), method="bounded")
    best_alpha = float(opt.x if opt.success else scan[best_idx])
    best_score, best_groups = collapse_score(df, best_alpha, grid_n)
    return {
        "alpha_scan": scan,
        "score_scan": scores,
        "best_alpha": best_alpha,
        "best_score": float(best_score),
        "group_scores": best_groups,
        "optimizer_success": bool(opt.success),
    }


def plot_alpha_scan(best_payload: dict, outdir: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.plot(best_payload["alpha_scan"], best_payload["score_scan"], lw=1.8, color="tab:blue")
    ax.axvline(best_payload["best_alpha"], color="tab:red", ls="--", lw=1.4, label=rf"best $\alpha={best_payload['best_alpha']:.4f}$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"collapse score")
    ax.set_title(r"$\xi/F_0 \,/\, v_w^\alpha$ collapse vs inferred $t_p$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "alpha_scan.png", dpi=dpi)
    plt.close(fig)


def plot_overlay_for_alpha(df: pd.DataFrame, alpha: float, outdir: Path, dpi: int, tag: str):
    theta_values = np.sort(df["theta"].unique())
    h_values = np.sort(df["H"].unique())
    vw_values = np.sort(df["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {float(vw): cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}

    for h in h_values:
        sub_h = df[np.isclose(df["H"], float(h), atol=1.0e-12, rtol=0.0)].copy()
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
        axes = axes.ravel()
        for ax, theta in zip(axes, theta_values):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
            for vw, cur in sub.groupby("v_w", sort=True):
                cur = cur.sort_values("tp")
                y = scaled_y(cur, alpha)
                ax.plot(cur["tp"], y, color=colors[float(vw)], marker="o", lw=1.5, ms=3.2, alpha=0.9, label=rf"$v_w={float(vw):.1f}$")
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$t_p$")
            ax.set_ylabel(r"$\xi/(F_0\,v_w^\alpha)$")
        for ax in axes[len(theta_values):]:
            ax.axis("off")
        handles = [plt.Line2D([0], [0], color=colors[float(vw)], lw=2.0) for vw in vw_values]
        labels = [rf"$v_w={float(vw):.1f}$" for vw in vw_values]
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
        fig.suptitle(rf"$\xi/(F_0 v_w^\alpha)$ vs inferred $t_p$, $H_*={float(h):.1f}$, $\alpha={float(alpha):.4f}$", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        h_tag = str(float(h)).replace(".", "p")
        fig.savefig(outdir / f"xi_over_F0_vwalpha_vs_tp_{tag}_H{h_tag}.png", dpi=dpi)
        plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("[load] reading lattice dataframe")
    df = load_lattice_dataframe(outdir, args.vw_folders, args.h_values)

    print("[fit] scanning alpha for best amplitude collapse")
    best = fit_best_alpha(df, args.alpha_min, args.alpha_max, args.alpha_grid, args.grid_n)
    plot_alpha_scan(best, outdir, args.dpi)

    compare = []
    seen = set()
    for alpha in [0.0, float(best["best_alpha"]), *COMPARE_ALPHAS]:
        key = round(float(alpha), 8)
        if key in seen:
            continue
        seen.add(key)
        compare.append(float(alpha))

    print("[plot] writing overlays for alpha values")
    for alpha in compare:
        if np.isclose(alpha, best["best_alpha"], atol=1.0e-8, rtol=0.0):
            tag = "best"
        elif np.isclose(alpha, 0.0, atol=1.0e-8, rtol=0.0):
            tag = "alpha0"
        else:
            tag = f"alpha_{alpha:+.4f}".replace(".", "p").replace("+", "plus").replace("-", "minus")
        plot_overlay_for_alpha(df, alpha, outdir, args.dpi, tag)

    summary = {
        "status": "ok",
        "best_alpha": float(best["best_alpha"]),
        "best_score": float(best["best_score"]),
        "optimizer_success": bool(best["optimizer_success"]),
        "compare_alphas": compare,
        "outputs": {
            "alpha_scan": str(outdir / "alpha_scan.png"),
            **{
                f"H{str(float(h)).replace('.', 'p')}_best": str(outdir / f"xi_over_F0_vwalpha_vs_tp_best_H{str(float(h)).replace('.', 'p')}.png")
                for h in np.sort(df["H"].unique())
            },
            **{
                f"H{str(float(h)).replace('.', 'p')}_alpha0": str(outdir / f"xi_over_F0_vwalpha_vs_tp_alpha0_H{str(float(h)).replace('.', 'p')}.png")
                for h in np.sort(df["H"].unique())
            },
        },
        "group_scores": best["group_scores"],
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
