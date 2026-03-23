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

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import fit_vw_amplitude as base_fit
import plot_lattice_xi_vs_x_all_vw_quadwarped as quadwarp


OUTDIR = ROOT / "results_fanh_raw_vs_teff_quadwarp"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare raw fanh-like diagnostic xi*F0/tp^(3/2) to the same quantity built from t_eff."
    )
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--reference-vw", type=float, default=0.9)
    p.add_argument("--beta", type=float, default=0.0)
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
    return df.sort_values(["H", "theta", "v_w", "tp"]).reset_index(drop=True)


def compute_warped_df(df: pd.DataFrame, ref_vw: float, beta: float):
    params, fit_rows = quadwarp.fit_quadwarps(df, ref_vw, beta)
    out = df.copy()
    t_eff = np.zeros(len(out), dtype=np.float64)
    for i, row in enumerate(out.itertuples(index=False)):
        rec = params.get(f"{float(row.v_w):.1f}", {"log_s": 0.0, "b": 1.0, "c": 0.0})
        t_eff[i] = float(quadwarp.warp_tp(float(row.tp), rec["log_s"], rec["b"], rec["c"]))
    out["t_eff"] = t_eff
    out["y_raw"] = out["xi"].to_numpy(dtype=np.float64) * out["F0"].to_numpy(dtype=np.float64) / np.maximum(
        np.power(out["tp"].to_numpy(dtype=np.float64), 1.5), 1.0e-18
    )
    out["y_eff"] = out["xi"].to_numpy(dtype=np.float64) * out["F0"].to_numpy(dtype=np.float64) / np.maximum(
        np.power(out["t_eff"].to_numpy(dtype=np.float64), 1.5), 1.0e-18
    )
    return out, params, fit_rows


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


def collapse_score_one_panel(sub: pd.DataFrame, x_col: str, y_col: str, grid_n: int):
    curves = []
    for vw, cur in sub.groupby("v_w", sort=True):
        cur = cur.sort_values(x_col)
        x = cur[x_col].to_numpy(dtype=np.float64)
        y = cur[y_col].to_numpy(dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        if np.count_nonzero(mask) < 3:
            continue
        curves.append((float(vw), x[mask], y[mask]))
    if len(curves) < 2:
        return np.nan
    xmin = max(np.min(x) for _, x, _ in curves)
    xmax = min(np.max(x) for _, x, _ in curves)
    if (not np.isfinite(xmin)) or (not np.isfinite(xmax)) or xmax <= xmin:
        return np.nan
    grid = np.geomspace(float(xmin), float(xmax), int(grid_n))
    arr = np.vstack([loglog_interp(x, y, grid) for _, x, y in curves])
    valid = np.sum(np.isfinite(arr), axis=0) >= 2
    if not np.any(valid):
        return np.nan
    mean = np.nanmean(arr[:, valid], axis=0)
    var = np.nanvar(arr[:, valid], axis=0, ddof=1)
    return float(np.nanmean(var / np.maximum(mean * mean, 1.0e-18)))


def plot_h_figure(sub_h: pd.DataFrame, h_value: float, outdir: Path, dpi: int, grid_n: int):
    theta_values = np.sort(sub_h["theta"].unique())
    vw_values = np.sort(sub_h["v_w"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {float(vw): cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    fig, axes = plt.subplots(len(theta_values), 2, figsize=(11.5, 3.0 * len(theta_values)), sharex=False, sharey=False)
    if len(theta_values) == 1:
        axes = np.asarray([axes])
    summary_rows = []
    for row_idx, theta in enumerate(theta_values):
        sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=5.0e-4, rtol=0.0)].copy()
        ax_raw = axes[row_idx, 0]
        ax_eff = axes[row_idx, 1]
        for vw in vw_values:
            cur = sub[np.isclose(sub["v_w"], float(vw), atol=1.0e-12, rtol=0.0)].sort_values("tp")
            if cur.empty:
                continue
            ax_raw.plot(cur["tp"], cur["y_raw"], color=colors[float(vw)], marker="o", lw=1.5, ms=3.0, alpha=0.9)
            cur_eff = cur.sort_values("t_eff")
            ax_eff.plot(cur_eff["t_eff"], cur_eff["y_eff"], color=colors[float(vw)], marker="o", lw=1.5, ms=3.0, alpha=0.9)
        ax_raw.set_xscale("log")
        ax_eff.set_xscale("log")
        ax_raw.set_yscale("log")
        ax_eff.set_yscale("log")
        ax_raw.grid(alpha=0.25)
        ax_eff.grid(alpha=0.25)
        ax_raw.set_ylabel(rf"$\xi F_0 / t_p^{{3/2}}$" + "\n" + rf"$\theta={float(theta):.3f}$")
        ax_eff.set_ylabel(rf"$\xi F_0 / t_{{eff}}^{{3/2}}$" + "\n" + rf"$\theta={float(theta):.3f}$")
        ax_raw.set_xlabel(r"$t_p$")
        ax_eff.set_xlabel(r"$t_{\rm eff}$")
        raw_score = collapse_score_one_panel(sub, "tp", "y_raw", grid_n)
        eff_score = collapse_score_one_panel(sub, "t_eff", "y_eff", grid_n)
        ax_raw.set_title(rf"raw, $S={raw_score:.2e}$" if np.isfinite(raw_score) else "raw")
        ax_eff.set_title(rf"effective, $S={eff_score:.2e}$" if np.isfinite(eff_score) else "effective")
        summary_rows.append(
            {
                "H": float(h_value),
                "theta": float(theta),
                "raw_score": float(raw_score) if np.isfinite(raw_score) else np.nan,
                "eff_score": float(eff_score) if np.isfinite(eff_score) else np.nan,
                "improvement_factor": float(raw_score / eff_score) if np.isfinite(raw_score) and np.isfinite(eff_score) and eff_score > 0.0 else np.nan,
            }
        )
    handles = [plt.Line2D([0], [0], color=colors[float(vw)], lw=2.0) for vw in vw_values]
    labels = [rf"$v_w={float(vw):.1f}$" for vw in vw_values]
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
    fig.suptitle(rf"Raw vs effective-clock fanh-like diagnostic, $H_*={float(h_value):.1f}$", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    tag = str(float(h_value)).replace(".", "p")
    fig.savefig(outdir / f"fanh_raw_vs_teff_H{tag}.png", dpi=dpi)
    plt.close(fig)
    return summary_rows


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("[load] reading lattice dataframe")
    df = load_lattice_dataframe(outdir, args.vw_folders, args.h_values)

    print("[warp] fitting best current quadratic v_w time remapping")
    warped_df, warp_params, warp_rows = compute_warped_df(df, float(args.reference_vw), float(args.beta))

    print("[plot] writing raw-vs-teff comparison figures")
    summary_rows = []
    for h in np.sort(warped_df["H"].unique()):
        sub_h = warped_df[np.isclose(warped_df["H"], float(h), atol=1.0e-12, rtol=0.0)].copy()
        summary_rows.extend(plot_h_figure(sub_h, float(h), outdir, args.dpi, int(args.grid_n)))

    summary_df = pd.DataFrame(summary_rows).sort_values(["H", "theta"]).reset_index(drop=True)
    summary_df.to_csv(outdir / "collapse_score_comparison.csv", index=False)

    summary = {
        "status": "ok",
        "reference_vw": float(args.reference_vw),
        "beta": float(args.beta),
        "warp_rows": warp_rows,
        "outputs": {
            **{f"H{str(float(h)).replace('.', 'p')}": str(outdir / f"fanh_raw_vs_teff_H{str(float(h)).replace('.', 'p')}.png") for h in np.sort(warped_df["H"].unique())},
            "score_csv": str(outdir / "collapse_score_comparison.csv"),
        },
        "score_summary": summary_rows,
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
