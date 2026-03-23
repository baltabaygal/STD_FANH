#!/usr/bin/env python3
"""
study_vw_param_evolution.py

Run an independent collapse fit for each wall velocity and study how
the FANH model parameters (beta, t_c, r) evolve with vw.

For each vw:
  1. Find the best collapse exponent beta(vw) by minimising the
     inter-H* variance of xi curves on the collapsed coordinate x=tp*H^beta.
  2. Fit the FANH model  xi = x^{3/2} * F_inf(theta)/F0^2 + 1/(1+(x/t_c)^r)
     to obtain (t_c, r, F_inf).

Summary outputs:
  - per_vw_fits.json           all per-vw parameter values
  - param_evolution.png        beta, t_c, r vs vw with power-law fits
  - loo_quality.png            LOO-interpolation RMSE (leave one vw out)
  - overlay_<param>.png        beta/tc/r scan curves overlaid across vw
  - summary.json

Leave-one-out (LOO) evaluation:
  For each held-out vw, interpolate (beta, t_c, r) from the remaining
  three vw values and evaluate rel_rmse on the held-out data.
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
import fit_vw_amplitude as base


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


# ── collapse score and beta search ───────────────────────────────────────────

def collapse_score(df, beta, grid_n=80):
    tp = df["tp"].to_numpy(); H = df["H"].to_numpy(); xi = df["xi"].to_numpy()
    x = tp * np.power(H, beta)
    theta_arr = df["theta"].to_numpy()
    total = 0.0; count = 0
    for th in np.unique(theta_arr):
        mask = np.abs(theta_arr - th) < 5e-4
        sx = x[mask]; sxi = xi[mask]; sH = H[mask]
        xgrid = np.geomspace(sx.min(), sx.max(), grid_n)
        curves = []
        for h in np.unique(sH):
            m2 = np.abs(sH - h) < 1e-6
            if np.sum(m2) < 2:
                continue
            idx = np.argsort(sx[m2])
            f = interp1d(np.log(sx[m2][idx]), np.log(sxi[m2][idx]),
                         kind="linear", bounds_error=False, fill_value=np.nan)
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
        total += float(np.nanmean(frac[np.isfinite(frac)])); count += 1
    return total / count if count > 0 else np.inf


def find_best_beta(df, grid_n=80):
    coarse = np.linspace(-2.0, 2.0, 81)
    sc = np.array([collapse_score(df, b, grid_n) for b in coarse])
    best_idx = int(np.nanargmin(sc))
    lo = max(-3.0, coarse[max(best_idx - 1, 0)])
    hi = min(3.0, coarse[min(best_idx + 1, len(coarse) - 1)])
    res = minimize_scalar(lambda b: collapse_score(df, float(b), grid_n),
                          bounds=(lo, hi), method="bounded", options={"xatol": 1e-3})
    beta = float(res.x if res.success else coarse[best_idx])
    score = float(res.fun if res.success else sc[best_idx])
    return beta, score, coarse.tolist(), sc.tolist()


# ── FANH model fit ────────────────────────────────────────────────────────────

def fit_fanh(df, theta_values, beta, tc0=1.5):
    tp = df["tp"].to_numpy(); H = df["H"].to_numpy()
    xi = df["xi"].to_numpy(); F0 = df["F0"].to_numpy()
    x = tp * np.power(H, beta)
    theta_idx = np.array([nearest_theta(theta_values, th) for th in df["theta"]], dtype=np.int64)

    # naive initial F_inf from tail
    finf0 = np.zeros(len(theta_values))
    for i in range(len(theta_values)):
        mask = theta_idx == i
        if not np.any(mask):
            finf0[i] = 1.0; continue
        xi_i = xi[mask]; x_i = x[mask]; F0_i = F0[mask]
        tail = x_i >= np.percentile(x_i, 70)
        num = xi_i[tail] * F0_i[tail] ** 2
        den = x_i[tail] ** 1.5
        finf0[i] = max(float(np.median(num / np.maximum(den, 1e-18))), 1e-6)

    x0 = np.concatenate([[tc0, 3.0], finf0])
    lo = np.concatenate([[1e-3, 0.1], np.full(len(theta_values), 1e-8)])
    hi = np.concatenate([[30.0, 50.0], np.full(len(theta_values), 1e4)])

    def resid(p):
        tc, r = p[0], p[1]; finf = p[2:]
        yfit = np.power(x, 1.5) * finf[theta_idx] / np.maximum(F0 ** 2, 1e-18)
        yfit += 1.0 / (1.0 + np.power(x / max(tc, 1e-12), r))
        return (yfit - xi) / np.maximum(xi, 1e-12)

    r0 = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.05, max_nfev=8000)
    r1 = least_squares(resid, r0.x, bounds=(lo, hi), loss="linear", max_nfev=8000)
    tc, r_val = r1.x[0], r1.x[1]; finf = r1.x[2:]
    yfit = np.power(x, 1.5) * finf[theta_idx] / np.maximum(F0 ** 2, 1e-18)
    yfit += 1.0 / (1.0 + np.power(x / max(tc, 1e-12), r_val))
    return {"t_c": float(tc), "r": float(r_val), "F_inf": finf,
            "rel_rmse": rel_rmse(xi, yfit), "beta": float(beta)}


def eval_fanh_rmse(df, theta_values, beta, tc, r, F_inf):
    tp = df["tp"].to_numpy(); H = df["H"].to_numpy()
    xi = df["xi"].to_numpy(); F0 = df["F0"].to_numpy()
    x = tp * np.power(H, beta)
    theta_idx = np.array([nearest_theta(theta_values, th) for th in df["theta"]], dtype=np.int64)
    yfit = np.power(x, 1.5) * F_inf[theta_idx] / np.maximum(F0 ** 2, 1e-18)
    yfit += 1.0 / (1.0 + np.power(x / max(tc, 1e-12), r))
    return rel_rmse(xi, yfit)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_param_evolution(vw_arr, beta_arr, tc_arr, r_arr, outdir, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    params = [
        (beta_arr, r"$\beta$ (collapse exponent)", "tab:blue"),
        (tc_arr,   r"$t_c$ (crossover time)",      "tab:orange"),
        (r_arr,    r"$r$ (transient exponent)",     "tab:green"),
    ]
    for ax, (vals, label, color) in zip(axes, params):
        ax.plot(vw_arr, vals, "o-", color=color, lw=1.6, ms=7)
        # power-law fit
        if len(vw_arr) >= 2 and np.all(np.isfinite(vals)) and np.all(np.array(vals) > 0):
            log_vw = np.log(vw_arr); log_v = np.log(np.array(vals))
            p = np.polyfit(log_vw, log_v, 1)
            vw_fine = np.linspace(vw_arr.min(), vw_arr.max(), 80)
            ax.plot(vw_fine, np.exp(p[1]) * vw_fine ** p[0], "r--", lw=1.3,
                    label=rf"$\propto v_w^{{{p[0]:.3f}}}$")
            ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel(r"$v_w$"); ax.set_ylabel(label); ax.grid(alpha=0.25)
    fig.suptitle("FANH parameter evolution with wall velocity", y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "param_evolution.png", dpi=dpi)
    plt.close(fig)


def plot_beta_scans(vw_list, beta_coarse_list, score_coarse_list, best_betas, outdir, dpi):
    """Overlay beta score curves for all vw."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    cmap = plt.get_cmap("viridis")
    for i, (vw, betas, scores, best) in enumerate(zip(vw_list, beta_coarse_list, score_coarse_list, best_betas)):
        color = cmap(i / max(len(vw_list) - 1, 1))
        scores_arr = np.array(scores)
        ax.semilogy(betas, scores_arr, color=color, lw=1.4, label=rf"$v_w={vw:.2g}$")
        ax.axvline(best, color=color, lw=0.8, ls="--")
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel("collapse score (log)")
    ax.set_title("Collapse score vs beta for each vw")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "beta_scan_overlay.png", dpi=dpi)
    plt.close(fig)


def plot_loo(vw_arr, loo_rmse, fit_rmse, outdir, dpi):
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(vw_arr, fit_rmse, "o-", color="tab:blue", lw=1.5, ms=6, label="in-sample (indep fit)")
    ax.plot(vw_arr, loo_rmse, "s--", color="tab:red",  lw=1.5, ms=6, label="LOO (interp params)")
    ax.set_xlabel(r"$v_w$"); ax.set_ylabel("rel RMSE")
    ax.set_title("In-sample vs LOO RMSE")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "loo_quality.png", dpi=dpi)
    plt.close(fig)


def plot_xi_overlay(frames, theta_values, fit_params, outdir, dpi):
    """For a few representative theta values, overlay xi vs x for all vw."""
    n_show = min(4, len(theta_values))
    show_idx = np.linspace(0, len(theta_values) - 1, n_show, dtype=int)
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
    if n_show == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis")
    vw_list = sorted(frames.keys())
    for col, ti in enumerate(show_idx):
        theta = theta_values[ti]
        ax = axes[col]
        for i, vw in enumerate(vw_list):
            color = cmap(i / max(len(vw_list) - 1, 1))
            df = frames[vw]
            mask = np.abs(df["theta"].to_numpy() - theta) < 5e-4
            if not np.any(mask):
                continue
            sub = df[mask].sort_values("tp")
            fp = fit_params[vw]
            x = sub["tp"].to_numpy() * np.power(sub["H"].to_numpy(), fp["beta"])
            xi = sub["xi"].to_numpy()
            ax.scatter(x, xi, s=14, color=color, alpha=0.7, label=rf"$v_w={vw:.2g}$")
            # model curve
            xfine = np.geomspace(x.min(), x.max(), 120)
            F0_val = float(sub["F0"].iloc[0])
            finf_val = float(fp["F_inf"][ti])
            yfine = xfine ** 1.5 * finf_val / F0_val ** 2
            yfine += 1.0 / (1.0 + (xfine / max(fp["t_c"], 1e-12)) ** fp["r"])
            ax.plot(xfine, yfine, "-", color=color, lw=1.2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$x = t_p H^\beta$"); ax.set_ylabel(r"$\xi$")
        ax.set_title(rf"$\theta_0 = {theta:.3f}$")
        ax.legend(frameon=False, fontsize=7); ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "xi_overlay_collapsed.png", dpi=dpi)
    plt.close(fig)


# ── LOO evaluation ────────────────────────────────────────────────────────────

def loo_eval(vw_arr, fit_params_list, frames, theta_values_list):
    """For each vw, interpolate parameters from remaining three and evaluate."""
    loo_rmse = []
    for i, vw in enumerate(vw_arr):
        others = [j for j in range(len(vw_arr)) if j != i]
        if len(others) < 2:
            loo_rmse.append(np.nan); continue
        vw_others = np.array([vw_arr[j] for j in others])
        beta_others = np.array([fit_params_list[j]["beta"] for j in others])
        tc_others = np.array([fit_params_list[j]["t_c"] for j in others])
        r_others = np.array([fit_params_list[j]["r"] for j in others])
        finf_others = np.array([fit_params_list[j]["F_inf"] for j in others])

        # linear interpolation in log(vw)
        log_vw_o = np.log(vw_others); log_vw_i = math.log(vw)
        def interp1(arr):
            if len(arr) == 1:
                return float(arr[0])
            f = interp1d(log_vw_o, arr, kind="linear", fill_value="extrapolate")
            return float(f(log_vw_i))

        beta_i = interp1(beta_others)
        tc_i = interp1(tc_others)
        r_i = interp1(r_others)
        finf_i = np.array([interp1(finf_others[:, k]) for k in range(finf_others.shape[1])])

        df = frames[vw]
        # theta intersection
        theta_ref = theta_values_list[others[0]]
        theta_ok = df["theta"].apply(lambda th: any(abs(th - tv) < 5e-4 for tv in theta_ref))
        df = df[theta_ok].copy()
        if df.empty:
            loo_rmse.append(np.nan); continue
        try:
            rmse = eval_fanh_rmse(df, theta_ref, beta_i, tc_i, r_i, finf_i)
        except Exception:
            rmse = np.nan
        loo_rmse.append(rmse)
    return loo_rmse


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rho", type=str, default="")
    p.add_argument("--vw-tags", nargs="*", default=None)
    p.add_argument("--h-values", type=float, nargs="+", default=[1.5, 2.0])
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--outdir", type=str, default="results_vw_param_evolution")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    rho_path = base.resolve_rho_file(args.rho)
    f0_table = base.load_f0_table(rho_path, args.h_values)
    f0_keys_arr = f0_table["theta"].to_numpy()
    f0_vals_arr = f0_table["F0"].to_numpy()

    def lookup_f0(theta):
        idx = int(np.argmin(np.abs(f0_keys_arr - float(theta))))
        return float(f0_vals_arr[idx])

    perc = PercolationCache()
    vw_sources = base.resolve_vw_sources(args.vw_tags)

    frames = {}
    for vw, path, tag in vw_sources:
        df = base.load_one_raw_scan(path, vw, args.h_values, perc)
        df = df[df["H"].isin(args.h_values)].copy()
        df["F0"] = [lookup_f0(th) for th in df["theta"]]
        df = df[np.isfinite(df["F0"]) & np.isfinite(df["tp"]) & np.isfinite(df["xi"])].copy()
        frames[float(vw)] = df
        print(f"  vw={vw:.2g}: {len(df)} points")

    # ── per-vw independent fits ────────────────────────────────────────────────
    fit_params = {}
    beta_scan_data = {}
    per_vw_json = {}
    vw_sorted = sorted(frames.keys())

    for vw in vw_sorted:
        df = frames[vw]
        theta_values = np.array(sorted(df["theta"].unique()))
        print(f"\nFitting vw={vw:.2g} ...")
        beta, cscore, beta_coarse, score_coarse = find_best_beta(df)
        result = fit_fanh(df, theta_values, beta)
        fit_params[vw] = {**result, "theta_values": theta_values, "collapse_score": cscore}
        beta_scan_data[vw] = (beta_coarse, score_coarse, beta)
        print(f"  beta={beta:.4f}  t_c={result['t_c']:.4f}  r={result['r']:.4f}  "
              f"rel_rmse={result['rel_rmse']:.5f}  collapse_score={cscore:.3e}")
        per_vw_json[f"{vw:.2g}"] = {
            "beta": beta, "collapse_score": cscore,
            "t_c": result["t_c"], "r": result["r"],
            "rel_rmse": result["rel_rmse"],
            "n_points": int(len(df)),
            "F_inf": {f"{th:.10f}": float(v) for th, v in zip(theta_values, result["F_inf"])},
        }

    save_json(outdir / "per_vw_fits.json", per_vw_json)

    # arrays for plotting
    vw_arr = np.array(vw_sorted)
    beta_arr = np.array([fit_params[v]["beta"] for v in vw_sorted])
    tc_arr   = np.array([fit_params[v]["t_c"]  for v in vw_sorted])
    r_arr    = np.array([fit_params[v]["r"]     for v in vw_sorted])
    fit_rmse = np.array([fit_params[v]["rel_rmse"] for v in vw_sorted])

    # ── LOO evaluation ─────────────────────────────────────────────────────────
    fit_params_list = [fit_params[v] for v in vw_sorted]
    theta_values_list = [fit_params[v]["theta_values"] for v in vw_sorted]
    loo_rmse = loo_eval(vw_arr, fit_params_list, frames, theta_values_list)

    # ── plots ──────────────────────────────────────────────────────────────────
    plot_param_evolution(vw_arr, beta_arr, tc_arr, r_arr, outdir, args.dpi)
    plot_beta_scans(
        vw_sorted,
        [beta_scan_data[v][0] for v in vw_sorted],
        [beta_scan_data[v][1] for v in vw_sorted],
        [beta_scan_data[v][2] for v in vw_sorted],
        outdir, args.dpi,
    )
    plot_loo(vw_arr, loo_rmse, fit_rmse, outdir, args.dpi)
    plot_xi_overlay(frames, fit_params[vw_sorted[-1]]["theta_values"],
                    {v: fit_params[v] for v in vw_sorted}, outdir, args.dpi)

    # power-law fits for each parameter
    plaw = {}
    for name, arr in [("beta", beta_arr), ("t_c", tc_arr), ("r", r_arr)]:
        if np.all(np.isfinite(arr)) and np.all(arr > 0) and len(vw_arr) >= 2:
            p = np.polyfit(np.log(vw_arr), np.log(arr), 1)
            plaw[name] = {"exponent": float(p[0]), "amplitude_at_vw1": float(np.exp(p[1]))}
        else:
            plaw[name] = {}

    save_json(outdir / "summary.json", {
        "vw_values": vw_arr.tolist(),
        "beta": beta_arr.tolist(),
        "t_c": tc_arr.tolist(),
        "r": r_arr.tolist(),
        "rel_rmse_insample": fit_rmse.tolist(),
        "rel_rmse_loo": [float(v) if v is not None else None for v in loo_rmse],
        "power_law_fits": plaw,
    })

    print(f"\nResults saved to {outdir}/")
    print("\nParameter summary:")
    for vw, b, tc, r, rmse, loo in zip(vw_sorted, beta_arr, tc_arr, r_arr, fit_rmse, loo_rmse):
        print(f"  vw={vw:.2g}  beta={b:+.4f}  t_c={tc:.4f}  r={r:.4f}  "
              f"rmse={rmse:.5f}  loo={loo:.5f}" if loo is not None else
              f"  vw={vw:.2g}  beta={b:+.4f}  t_c={tc:.4f}  r={r:.4f}  rmse={rmse:.5f}")


if __name__ == "__main__":
    main()
