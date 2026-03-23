#!/usr/bin/env python3
"""
study_vw_teff_correction.py

Test whether a vw-dependent rescaling or additive shift of tp can
absorb the FANH-model degradation at lower wall velocities.

Two correction types are scanned for each vw:
  A) multiplicative:  tp_eff = tp * s_vw
  B) additive:        tp_eff = tp + delta_vw

In both cases the model parameters (beta, t_c, r, F_inf) are fixed to a
reference fit obtained from the vw=0.9 data alone.  The best per-vw
correction is found by minimising relative RMSE over a 1-D grid, then
refined with a scalar minimiser.

A global formula for s_vw(vw) is also tested:
  s_vw = (vw / vw_ref)^eta              [power-law]
  delta_vw = A * (1/vw - 1/vw_ref)     [1/vw additive]

Outputs (in --outdir):
  reference_fit.json
  correction_scan_vw*.png   per-vw RMSE vs correction value
  correction_summary.png    s and delta vs vw, with global formula
  summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, least_squares

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ode.hom_ODE.percolation import PercolationCache
import fit_vw_amplitude as base  # reuse data-loading helpers


# ── helpers ──────────────────────────────────────────────────────────────────

def to_native(obj):
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_native(v) for v in obj.tolist()]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def save_json(path: Path, payload):
    path.write_text(json.dumps(to_native(payload), indent=2, sort_keys=True))


def rel_rmse(y, yfit):
    return float(np.sqrt(np.mean(np.square((yfit - y) / np.maximum(y, 1e-12)))))


def nearest_theta(theta_values, theta0, atol=5e-4):
    theta_values = np.asarray(theta_values)
    idx = int(np.argmin(np.abs(theta_values - float(theta0))))
    if abs(theta_values[idx] - float(theta0)) > atol:
        raise RuntimeError(f"No theta match for {theta0:.8f}")
    return idx


# ── FANH model ────────────────────────────────────────────────────────────────

def xi_model(tp_eff, H, beta, theta_idx, F0, F_inf, t_c, r):
    """Plateau + transient FANH model."""
    x = tp_eff * np.power(H, beta)
    xi = np.power(x, 1.5) * F_inf[theta_idx] / np.maximum(F0 ** 2, 1e-18)
    xi += 1.0 / (1.0 + np.power(x / max(t_c, 1e-12), r))
    return xi


def fit_fanh(df, theta_values, beta, tc0=1.5):
    """Fit (t_c, r, F_inf[theta]) to collapsed data at given beta."""
    x = df["tp"].to_numpy() * np.power(df["H"].to_numpy(), beta)
    xi = df["xi"].to_numpy()
    F0 = df["F0"].to_numpy()
    theta_idx = np.array([nearest_theta(theta_values, th) for th in df["theta"]], dtype=np.int64)

    # initial F_inf from tail amplitude
    finf0 = np.array([
        float(np.median(xi[theta_idx == i] / np.maximum(x[theta_idx == i] ** 1.5 / F0[theta_idx == i] ** 2, 1e-18)))
        if np.any(theta_idx == i) else 1.0
        for i in range(len(theta_values))
    ], dtype=np.float64)
    finf0 = np.maximum(finf0, 1e-6)

    x0 = np.concatenate([[tc0, 3.0], finf0])
    lo = np.concatenate([[1e-3, 0.1], np.full(len(theta_values), 1e-8)])
    hi = np.concatenate([[20.0, 50.0], np.full(len(theta_values), 1e4)])

    def resid(p):
        tc, r = p[0], p[1]
        finf = p[2:]
        yfit = xi_model(x / np.power(df["H"].to_numpy(), beta),  # tp
                        df["H"].to_numpy(), beta, theta_idx, F0, finf, tc, r)
        # Note: x = tp * H^beta, so tp = x / H^beta — pass tp directly
        xc = x  # this IS tp * H^beta
        yfit2 = np.power(xc, 1.5) * finf[theta_idx] / np.maximum(F0 ** 2, 1e-18)
        yfit2 += 1.0 / (1.0 + np.power(xc / max(tc, 1e-12), r))
        return (yfit2 - xi) / np.maximum(xi, 1e-12)

    r0 = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.05, max_nfev=8000)
    r1 = least_squares(resid, r0.x, bounds=(lo, hi), loss="linear", max_nfev=8000)
    tc, r_val = r1.x[0], r1.x[1]
    finf = r1.x[2:]
    yfit = np.power(x, 1.5) * finf[theta_idx] / np.maximum(F0 ** 2, 1e-18)
    yfit += 1.0 / (1.0 + np.power(x / max(tc, 1e-12), r_val))
    return {"t_c": float(tc), "r": float(r_val), "F_inf": finf,
            "rel_rmse": rel_rmse(xi, yfit), "beta": float(beta)}


def find_best_beta(df, grid_n=80):
    """Minimize collapse variance over beta."""
    from scipy.optimize import minimize_scalar as ms
    theta_values = np.array(sorted(df["theta"].unique()))

    def score(beta):
        x = df["tp"].to_numpy() * np.power(df["H"].to_numpy(), beta)
        total = 0.0; count = 0
        for th in theta_values:
            mask = np.abs(df["theta"].to_numpy() - th) < 5e-4
            if np.sum(mask) < 2:
                continue
            sub_x = x[mask]; sub_xi = df["xi"].to_numpy()[mask]; sub_H = df["H"].to_numpy()[mask]
            xgrid = np.geomspace(sub_x.min(), sub_x.max(), grid_n)
            curves = []
            for h in np.unique(sub_H):
                m2 = np.abs(sub_H - h) < 1e-6
                if np.sum(m2) < 2:
                    continue
                sx = sub_x[m2]; sy = sub_xi[m2]
                order = np.argsort(sx)
                f = interp1d(np.log(sx[order]), np.log(sy[order]), kind="linear",
                             bounds_error=False, fill_value=np.nan)
                curves.append(np.exp(f(np.log(xgrid))))
            if len(curves) < 2:
                continue
            arr = np.vstack(curves)
            valid = np.sum(np.isfinite(arr), axis=0) >= 2
            if not np.any(valid):
                continue
            arr = arr[:, valid]
            mean = np.nanmean(arr, axis=0)
            var = np.nanvar(arr, axis=0, ddof=1)
            frac = var / np.maximum(mean ** 2, 1e-18)
            total += float(np.nanmean(frac)); count += 1
        return total / count if count > 0 else np.inf

    coarse = np.linspace(-2.0, 2.0, 81)
    sc = np.array([score(b) for b in coarse])
    best_idx = int(np.nanargmin(sc))
    lo, hi = max(-3.0, coarse[max(best_idx - 1, 0)]), min(3.0, coarse[min(best_idx + 1, len(coarse) - 1)])
    res = ms(score, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3})
    return float(res.x if res.success else coarse[best_idx]), float(res.fun if res.success else sc[best_idx])


# ── per-vw correction scan ────────────────────────────────────────────────────

def rmse_with_correction(correction, mode, tp, H, xi, F0, theta_idx, ref):
    """Evaluate RMSE for a multiplicative (mode='mult') or additive (mode='add') tp correction."""
    if mode == "mult":
        tp_eff = tp * float(correction)
    else:
        tp_eff = tp + float(correction)
    tp_eff = np.maximum(tp_eff, 1e-12)
    x = tp_eff * np.power(H, ref["beta"])
    yfit = np.power(x, 1.5) * ref["F_inf"][theta_idx] / np.maximum(F0 ** 2, 1e-18)
    yfit += 1.0 / (1.0 + np.power(x / max(ref["t_c"], 1e-12), ref["r"]))
    return rel_rmse(xi, yfit)


def find_best_correction(mode, tp, H, xi, F0, theta_idx, ref):
    if mode == "mult":
        grid = np.geomspace(0.1, 10.0, 120)
        lo, hi = 1e-3, 100.0
    else:
        tp_scale = float(np.median(tp))
        grid = np.linspace(-tp_scale, tp_scale * 5, 120)
        lo, hi = -tp_scale * 2, tp_scale * 20

    scores = np.array([rmse_with_correction(c, mode, tp, H, xi, F0, theta_idx, ref) for c in grid])
    best_idx = int(np.argmin(scores))
    lo2, hi2 = grid[max(best_idx - 1, 0)], grid[min(best_idx + 1, len(grid) - 1)]
    if lo2 >= hi2:
        return float(grid[best_idx]), float(scores[best_idx]), grid.tolist(), scores.tolist()
    try:
        res = minimize_scalar(lambda c: rmse_with_correction(c, mode, tp, H, xi, F0, theta_idx, ref),
                              bounds=(lo2, hi2), method="bounded")
        best_c = float(res.x)
        best_s = float(rmse_with_correction(best_c, mode, tp, H, xi, F0, theta_idx, ref))
    except Exception:
        best_c, best_s = float(grid[best_idx]), float(scores[best_idx])
    return best_c, best_s, grid.tolist(), scores.tolist()


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_scan(mode, vw, grid, scores, best_c, rmse_ref, rmse_corr, outdir, dpi):
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.plot(grid, scores, "b-", lw=1.5)
    ax.axvline(best_c, color="red", lw=1.5, ls="--", label=rf"best = {best_c:.4g}")
    ax.axhline(rmse_ref, color="gray", lw=1.0, ls=":", label=rf"no correction = {rmse_ref:.4f}")
    ax.axhline(rmse_corr, color="green", lw=1.0, ls="--", label=rf"corrected = {rmse_corr:.4f}")
    label = r"$s_{v_w}$" if mode == "mult" else r"$\delta_{v_w}$"
    ax.set_xlabel(label)
    ax.set_ylabel("rel RMSE")
    ax.set_title(rf"$v_w = {vw:.2g}$, mode={mode}")
    if mode == "mult":
        ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f"correction_scan_vw{vw:.2g}_{mode}.png", dpi=dpi)
    plt.close(fig)


def plot_summary(vw_list, s_best, delta_best, rmse_ref, rmse_s, rmse_d, outdir, dpi):
    vw_arr = np.array(vw_list)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # panel 1: multiplicative correction vs vw
    ax = axes[0]
    ax.plot(vw_arr, s_best, "o-", color="tab:blue", lw=1.5, ms=6)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    # fit power law s = (vw/vw_ref)^eta
    vw_ref = vw_arr[-1]
    log_vw = np.log(vw_arr / vw_ref)
    log_s = np.log(np.array(s_best))
    if np.any(np.isfinite(log_vw)) and len(log_vw) >= 2:
        eta = float(np.polyfit(log_vw[np.isfinite(log_vw)], log_s[np.isfinite(log_vw)], 1)[0])
        vw_fine = np.linspace(vw_arr.min(), vw_arr.max(), 80)
        ax.plot(vw_fine, (vw_fine / vw_ref) ** eta, "r--", lw=1.3,
                label=rf"$(v_w/v_{{ref}})^{{\eta}}$, $\eta={eta:.3f}$")
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel(r"$v_w$"); ax.set_ylabel(r"$s_{v_w}$ (mult)"); ax.set_title("Multiplicative correction")
    ax.grid(alpha=0.25)

    # panel 2: additive correction vs vw
    ax = axes[1]
    ax.plot(vw_arr, delta_best, "s-", color="tab:orange", lw=1.5, ms=6)
    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    # fit delta = A * (1/vw - 1/vw_ref)
    inv_vw = 1.0 / vw_arr - 1.0 / vw_ref
    try:
        A_fit = float(np.polyfit(inv_vw, np.array(delta_best), 1)[0])
        vw_fine = np.linspace(max(vw_arr.min(), 0.05), vw_arr.max(), 80)
        ax.plot(vw_fine, A_fit * (1.0 / vw_fine - 1.0 / vw_ref), "r--", lw=1.3,
                label=rf"$A(1/v_w - 1/v_{{ref}})$, $A={A_fit:.3g}$")
        ax.legend(frameon=False, fontsize=8)
    except Exception:
        pass
    ax.set_xlabel(r"$v_w$"); ax.set_ylabel(r"$\delta_{v_w}$ (add)"); ax.set_title("Additive correction")
    ax.grid(alpha=0.25)

    # panel 3: RMSE comparison
    ax = axes[2]
    ax.plot(vw_arr, rmse_ref, "k^-", lw=1.4, ms=7, label="no correction")
    ax.plot(vw_arr, rmse_s, "bo-", lw=1.4, ms=6, label="mult")
    ax.plot(vw_arr, rmse_d, "rs-", lw=1.4, ms=6, label="add")
    ax.set_xlabel(r"$v_w$"); ax.set_ylabel("rel RMSE"); ax.set_title("RMSE comparison")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outdir / "correction_summary.png", dpi=dpi)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rho", type=str, default="")
    p.add_argument("--vw-tags", nargs="*", default=None)
    p.add_argument("--h-values", type=float, nargs="+", default=[1.5, 2.0])
    p.add_argument("--ref-vw", type=float, default=0.9,
                   help="Wall velocity used as reference (default 0.9)")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--outdir", type=str, default="results_vw_teff_correction")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # load F0 table
    rho_path = base.resolve_rho_file(args.rho)
    f0_table = base.load_f0_table(rho_path, args.h_values)
    f0_map = dict(zip(f0_table["theta"].to_numpy(), f0_table["F0"].to_numpy()))

    # discover and load all vw data sets
    perc = PercolationCache()
    vw_sources = base.resolve_vw_sources(args.vw_tags)

    f0_keys_arr = np.array(sorted(f0_map.keys()))
    f0_vals_arr = np.array([f0_map[k] for k in sorted(f0_map.keys())])

    def lookup_f0(theta):
        idx = int(np.argmin(np.abs(f0_keys_arr - float(theta))))
        return float(f0_vals_arr[idx])

    frames = {}
    for vw, path, tag in vw_sources:
        df = base.load_one_raw_scan(path, vw, args.h_values, perc)
        df = df[df["H"].isin(args.h_values)].copy()
        df["F0"] = [lookup_f0(th) for th in df["theta"]]
        df = df[np.isfinite(df["F0"]) & np.isfinite(df["tp"]) & np.isfinite(df["xi"])].copy()
        frames[float(vw)] = df
        print(f"  vw={vw:.2g}: {len(df)} points")

    if args.ref_vw not in frames:
        sys.exit(f"Reference vw={args.ref_vw} not found in data. Available: {list(frames.keys())}")

    # ── fit reference model on vw_ref data ────────────────────────────────────
    df_ref = frames[args.ref_vw]
    theta_values = np.array(sorted(df_ref["theta"].unique()))
    print(f"\nFitting reference collapse model on vw={args.ref_vw} ...")
    beta_ref, _ = find_best_beta(df_ref)
    ref = fit_fanh(df_ref, theta_values, beta_ref)
    ref_rmse = ref["rel_rmse"]
    print(f"  beta={beta_ref:.4f}  t_c={ref['t_c']:.4f}  r={ref['r']:.4f}  rel_rmse={ref_rmse:.5f}")
    save_json(outdir / "reference_fit.json", {
        "vw_ref": args.ref_vw, "beta": beta_ref,
        "t_c": ref["t_c"], "r": ref["r"], "rel_rmse": ref_rmse,
        "F_inf": {f"{th:.10f}": float(v) for th, v in zip(theta_values, ref["F_inf"])},
    })

    # ── per-vw correction scan ─────────────────────────────────────────────────
    vw_list, s_best, delta_best = [], [], []
    rmse_ref_list, rmse_s_list, rmse_d_list = [], [], []
    results = {}

    for vw in sorted(frames.keys()):
        df = frames[vw]
        # align theta indices to reference theta_values
        theta_ok = df["theta"].apply(lambda th: any(abs(th - tv) < 5e-4 for tv in theta_values))
        df = df[theta_ok].copy()
        if df.empty:
            print(f"  vw={vw:.2g}: no theta overlap with reference — skip")
            continue

        tp = df["tp"].to_numpy()
        H = df["H"].to_numpy()
        xi = df["xi"].to_numpy()
        F0 = df["F0"].to_numpy()
        theta_idx = np.array([nearest_theta(theta_values, th) for th in df["theta"]], dtype=np.int64)

        rmse_no = rmse_with_correction(1.0 if True else 0.0, "mult", tp, H, xi, F0, theta_idx, ref)

        sc, rmse_sc, grid_m, scores_m = find_best_correction("mult", tp, H, xi, F0, theta_idx, ref)
        dc, rmse_dc, grid_a, scores_a = find_best_correction("add",  tp, H, xi, F0, theta_idx, ref)

        print(f"  vw={vw:.2g}: no_corr={rmse_no:.5f}  mult(s={sc:.4g})={rmse_sc:.5f}  add(d={dc:.4g})={rmse_dc:.5f}")

        plot_scan("mult", vw, grid_m, scores_m, sc, rmse_no, rmse_sc, outdir, args.dpi)
        plot_scan("add",  vw, grid_a, scores_a, dc, rmse_no, rmse_dc, outdir, args.dpi)

        vw_list.append(vw)
        s_best.append(sc); delta_best.append(dc)
        rmse_ref_list.append(rmse_no); rmse_s_list.append(rmse_sc); rmse_d_list.append(rmse_dc)
        results[f"{vw:.2g}"] = {
            "rmse_no_correction": rmse_no,
            "mult_s": sc, "mult_rmse": rmse_sc,
            "add_delta": dc, "add_rmse": rmse_dc,
        }

    plot_summary(vw_list, s_best, delta_best, rmse_ref_list, rmse_s_list, rmse_d_list, outdir, args.dpi)

    save_json(outdir / "summary.json", {
        "ref_vw": args.ref_vw, "ref_beta": beta_ref,
        "ref_t_c": ref["t_c"], "ref_r": ref["r"],
        "per_vw": results,
    })
    print(f"\nResults saved to {outdir}/")


if __name__ == "__main__":
    main()
