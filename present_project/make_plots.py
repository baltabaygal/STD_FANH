#!/usr/bin/env python3
"""
make_plots.py  —  Supervisor-meeting figures for the axion PT project.

Figures produced (all saved to present_project/ as 220-DPI PNGs):

  fig1_xi_vs_betaH.png       xi vs beta/H*  (4 vw × 4 H* panels, coloured by theta0)
  fig2_xi_vs_tp.png          xi vs t_p      (same layout)
  fig3_ftilde_vs_tp.png      xi*F0/tp^{3/2} vs t_p   (shows collapse quality per vw)
  fig4_ftilde_vs_betaH.png   xi*F0/tp^{3/2} vs beta/H*
  fig5_vw09_best_fit.png     vw=0.9 collapse with fitted FANH model overlaid
  fig6_all_vw_model.png      vw=0.9 model applied to all vw (shows degradation)
  fig7_s_tp_vw.png           Inferred pointwise shift s(tp,vw) + fitted formula
  fig8_shift_collapse.png    All-vw collapse after applying the shift (xi vs x_eff)
  fig9_residual_comparison.png  Fractional residuals before vs after shift, per vw

Run from repo root:
  conda run -n julia_env python present_project/make_plots.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ode.hom_ODE.percolation import PercolationCache
import fit_vw_amplitude as base

# ── constants ─────────────────────────────────────────────────────────────────
VW_TAGS   = ["v3", "v5", "v7", "v9"]
VW_VALUES = [0.3, 0.5, 0.7, 0.9]
H_VALUES  = [0.5, 1.0, 1.5, 2.0]
DPI       = 220

VW_COLORS  = {0.3: "#e41a1c", 0.5: "#ff7f00", 0.7: "#4daf4a", 0.9: "#377eb8"}
H_MARKERS  = {0.5: "o", 1.0: "s", 1.5: "^", 2.0: "D"}
H_COLORS   = {0.5: "#1f78b4", 1.0: "#33a02c", 1.5: "#e31a1c", 2.0: "#ff7f00"}

FITS_PATH       = ROOT / "results_vw_param_evolution" / "per_vw_fits.json"
SHIFT_TABLE     = ROOT / "results_pointwise_shift_from_vw0p9_baseline" / "pointwise_shift_table.csv"
SHIFT_POWERLAW  = ROOT / "results_pointwise_shift_powerlaw" / "final_summary.json"
SHIFT_APPLIED   = ROOT / "results_pointwise_shift_powerlaw_applied" / "predictions.csv"

# ── helpers ───────────────────────────────────────────────────────────────────

def potential(theta):
    return 1.0 - np.cos(np.asarray(theta, dtype=np.float64))


def load_data():
    """Load all vw datasets and attach F0 and tp columns."""
    rho_path = base.resolve_rho_file("")
    f0_table = base.load_f0_table(rho_path, H_VALUES)
    f0_keys  = f0_table["theta"].to_numpy()
    f0_vals  = f0_table["F0"].to_numpy()

    def lookup_f0(theta):
        idx = int(np.argmin(np.abs(f0_keys - float(theta))))
        return float(f0_vals[idx])

    perc = PercolationCache()
    frames = {}
    for tag, vw in zip(VW_TAGS, VW_VALUES):
        sources = base.resolve_vw_sources([tag])
        _, path, _ = sources[0]
        df = base.load_one_raw_scan(path, vw, H_VALUES, perc)
        df = df[df["H"].isin(H_VALUES)].copy()
        df["F0"]    = [lookup_f0(th) for th in df["theta"]]
        df["ftilde"] = (df["xi"] * df["F0"]
                        / np.power(np.maximum(df["tp"], 1e-18), 1.5))
        df = df[np.isfinite(df["F0"]) & np.isfinite(df["tp"]) & np.isfinite(df["xi"])].copy()
        frames[vw] = df
        print(f"  loaded vw={vw}: {len(df)} rows")
    return frames


def load_fits():
    if not FITS_PATH.exists():
        print(f"  WARNING: {FITS_PATH} not found — model overlays skipped")
        return None
    return json.loads(FITS_PATH.read_text())


def theta_cmap(theta_values):
    """Map sorted theta values to viridis colours."""
    th = np.array(sorted(theta_values))
    cmap = mpl.colormaps["viridis"]
    return {t: cmap(i / max(len(th) - 1, 1)) for i, t in enumerate(th)}


def nearest_theta(theta_values, theta0, atol=5e-4):
    idx = int(np.argmin(np.abs(np.asarray(theta_values) - float(theta0))))
    return idx


# ── figure helpers ────────────────────────────────────────────────────────────

def make_4vw_4H_figure(frames, x_col, y_col, xlabel, ylabel, title, fname,
                        logx=True, logy=True, connect=True):
    """
    4-row (vw) × 4-col (H*) grid.
    Each panel: scatter/line of y_col vs x_col coloured by theta0.
    """
    fig, axes = plt.subplots(4, 4, figsize=(16, 13), sharey=False)
    fig.suptitle(title, fontsize=13, y=1.01)

    for row, vw in enumerate(VW_VALUES):
        df = frames[vw]
        theta_vals = sorted(df["theta"].unique())
        cmap_th = theta_cmap(theta_vals)

        for col, H in enumerate(H_VALUES):
            ax = axes[row, col]
            sub = df[np.isclose(df["H"], H, atol=1e-6)].copy()
            if sub.empty:
                ax.set_visible(False); continue

            for th in theta_vals:
                mask = np.abs(sub["theta"].to_numpy() - th) < 5e-4
                if not np.any(mask): continue
                pts = sub[mask].sort_values(x_col)
                xv = pts[x_col].to_numpy()
                yv = pts[y_col].to_numpy()
                good = np.isfinite(xv) & np.isfinite(yv) & (xv > 0) & (yv > 0)
                if not np.any(good): continue
                color = cmap_th[th]
                label = rf"$\theta_0={th:.2f}$" if col == 0 and row == 0 else None
                if connect:
                    ax.plot(xv[good], yv[good], "-o", color=color,
                            ms=3, lw=1.2, alpha=0.85, label=label)
                else:
                    ax.scatter(xv[good], yv[good], color=color,
                               s=12, alpha=0.75, label=label)

            if logx: ax.set_xscale("log")
            if logy: ax.set_yscale("log")
            ax.grid(alpha=0.2)
            if row == 0: ax.set_title(rf"$H_*={H}$", fontsize=10)
            if col == 0: ax.set_ylabel(rf"$v_w={vw}$  |  {ylabel}", fontsize=8)
            if row == 3: ax.set_xlabel(xlabel, fontsize=8)
            ax.tick_params(labelsize=7)

    # shared colorbar for theta0
    all_theta = sorted(frames[VW_VALUES[0]]["theta"].unique())
    sm = mpl.cm.ScalarMappable(
        cmap="viridis",
        norm=mpl.colors.Normalize(vmin=all_theta[0], vmax=all_theta[-1])
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.02)
    cbar.set_label(r"$\theta_0$", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUTDIR / fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


def make_ftilde_collapsed_figure(frames, x_col, xlabel, fname):
    """
    4-panel figure (one per vw).  Each panel overlays all H* as different
    colours so we can see whether f_tilde collapses across H*.
    Curves are coloured by H*, line style by theta0 group.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=False)
    fig.suptitle(
        r"$\tilde{f} \equiv \xi \cdot F_0 / t_p^{3/2}$ — collapse quality across $H_*$",
        fontsize=12
    )

    ls_cycle = ["-", "--", "-.", ":", (0, (3,1,1,1))]

    for col, vw in enumerate(VW_VALUES):
        ax = axes[col]
        df = frames[vw]
        theta_vals = sorted(df["theta"].unique())

        for i_th, th in enumerate(theta_vals):
            ls = ls_cycle[i_th % len(ls_cycle)]
            for H in H_VALUES:
                mask = (np.abs(df["theta"].to_numpy() - th) < 5e-4) & \
                       np.isclose(df["H"].to_numpy(), H, atol=1e-6)
                if not np.any(mask): continue
                pts = df[mask].sort_values(x_col)
                xv = pts[x_col].to_numpy()
                yv = pts["ftilde"].to_numpy()
                good = np.isfinite(xv) & np.isfinite(yv) & (xv > 0) & (yv > 0)
                if not np.any(good): continue
                label = rf"$H_*={H}$" if i_th == 0 else None
                ax.plot(xv[good], yv[good], ls=ls,
                        color=H_COLORS[H], lw=1.3, alpha=0.8,
                        label=label)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(r"$\tilde{f}$", fontsize=9)
        ax.set_title(rf"$v_w = {vw}$", fontsize=10)
        ax.grid(alpha=0.2)
        if col == 0:
            ax.legend(frameon=False, fontsize=7, loc="best")
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(OUTDIR / fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


def make_vw09_best_fit(frames, fits):
    """
    vw=0.9 collapse: xi vs x = tp * H^beta with FANH model overlaid.
    One panel per theta0, H* as different markers/colours.
    """
    vw = 0.9
    df  = frames[vw]
    key = f"{vw}"
    fit = fits[key]
    beta = fit["beta"]; tc = fit["t_c"]; r = fit["r"]
    finf_map = {float(k): float(v) for k, v in fit["F_inf"].items()}
    theta_vals = sorted(finf_map.keys())
    n = len(theta_vals)

    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
    axes = np.array(axes).flatten()
    fig.suptitle(
        rf"$v_w = 0.9$ — best collapse fit  "
        rf"($\beta={beta:.3f},\ t_c={tc:.3f},\ r={r:.3f}$, rel RMSE = {fit['rel_rmse']*100:.2f}\%)",
        fontsize=11
    )

    for i, th in enumerate(theta_vals):
        ax = axes[i]
        finf = finf_map[th]
        for H in H_VALUES:
            mask = (np.abs(df["theta"].to_numpy() - th) < 5e-4) & \
                   np.isclose(df["H"].to_numpy(), H, atol=1e-6)
            if not np.any(mask): continue
            pts = df[mask].sort_values("tp")
            x = pts["tp"].to_numpy() * pts["H"].to_numpy() ** beta
            xi = pts["xi"].to_numpy()
            good = np.isfinite(x) & np.isfinite(xi) & (x > 0) & (xi > 0)
            ax.scatter(x[good], xi[good], s=18, marker=H_MARKERS[H],
                       color=H_COLORS[H], alpha=0.85,
                       label=rf"$H_*={H}$", zorder=3)

        # model curve
        sub = df[np.abs(df["theta"].to_numpy() - th) < 5e-4]
        if not sub.empty:
            f0 = float(sub["F0"].iloc[0])
            x_all = sub["tp"].to_numpy() * sub["H"].to_numpy() ** beta
            xfine = np.geomspace(x_all.min(), x_all.max(), 200)
            xi_model = xfine**1.5 * finf / f0**2 + 1.0 / (1.0 + (xfine / tc)**r)
            ax.plot(xfine, xi_model, "k-", lw=1.8, zorder=4, label="FANH model")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(rf"$x = t_p H_*^\beta$", fontsize=9)
        ax.set_ylabel(r"$\xi$", fontsize=9)
        ax.set_title(rf"$\theta_0 = {th:.3f}$", fontsize=9)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig5_vw09_best_fit.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig5_vw09_best_fit.png")


def make_all_vw_model_comparison(frames, fits):
    """
    Apply the vw=0.9 model to all vw datasets.
    2-row layout: top row = data + model overlay, bottom row = relative residuals.
    One column per vw.  Data coloured by H*.
    Show a representative theta0 (the middle one).
    """
    vw_ref = 0.9
    fit_ref = fits[f"{vw_ref}"]
    beta = fit_ref["beta"]; tc = fit_ref["t_c"]; r = fit_ref["r"]
    finf_map = {float(k): float(v) for k, v in fit_ref["F_inf"].items()}
    theta_vals = sorted(finf_map.keys())
    th_show = theta_vals[len(theta_vals) // 2]  # middle theta0

    fig, axes = plt.subplots(2, 4, figsize=(16, 8),
                              gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(
        rf"vw=0.9 model applied to all wall velocities  "
        rf"($\theta_0 = {th_show:.3f}$)",
        fontsize=12
    )

    finf_th = finf_map[th_show]

    for col, vw in enumerate(VW_VALUES):
        df = frames[vw]
        fit_vw = fits[f"{vw}"]
        rmse = fit_vw["rel_rmse"]

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        all_x, all_xi, all_res = [], [], []
        for H in H_VALUES:
            mask = (np.abs(df["theta"].to_numpy() - th_show) < 5e-4) & \
                   np.isclose(df["H"].to_numpy(), H, atol=1e-6)
            if not np.any(mask): continue
            pts = df[mask].sort_values("tp")
            x   = pts["tp"].to_numpy() * pts["H"].to_numpy() ** beta
            xi  = pts["xi"].to_numpy()
            xi_err = pts["xi_sem"].to_numpy() if "xi_sem" in pts.columns else np.zeros_like(xi)
            xi_err = np.where(np.isfinite(xi_err), xi_err, 0.0)
            f0  = float(pts["F0"].iloc[0])
            good = np.isfinite(x) & np.isfinite(xi) & (x > 0) & (xi > 0)

            xi_pred  = x[good]**1.5 * finf_th / f0**2 + 1.0 / (1.0 + (x[good] / tc)**r)
            residual = (xi_pred - xi[good]) / np.maximum(xi[good], 1e-12)
            res_err  = xi_err[good] / np.maximum(xi[good], 1e-12)

            ax_top.errorbar(x[good], xi[good], yerr=xi_err[good],
                            fmt=H_MARKERS[H], color=H_COLORS[H],
                            ms=4, lw=0, elinewidth=1.2, capsize=2, alpha=0.85,
                            label=rf"$H_*={H}$", zorder=3)
            ax_bot.errorbar(x[good], residual, yerr=res_err,
                            fmt=H_MARKERS[H], color=H_COLORS[H],
                            ms=4, lw=0, elinewidth=1.2, capsize=2, alpha=0.85)

            all_x.extend(x[good].tolist())
            all_xi.extend(xi[good].tolist())
            all_res.extend(residual.tolist())

        # model curve
        if all_x:
            f0_ref = float(frames[vw][
                np.abs(frames[vw]["theta"].to_numpy() - th_show) < 5e-4
            ]["F0"].iloc[0])
            xfine = np.geomspace(min(all_x), max(all_x), 200)
            xi_model = xfine**1.5 * finf_th / f0_ref**2 + 1.0 / (1.0 + (xfine / tc)**r)
            ax_top.plot(xfine, xi_model, "k-", lw=1.8, label="vw=0.9 model", zorder=5)

        rmse_shown = float(np.sqrt(np.mean(np.square(all_res)))) if all_res else np.nan
        ax_top.set_xscale("log"); ax_top.set_yscale("log")
        ax_top.set_ylabel(r"$\xi$", fontsize=9)
        ax_top.set_title(rf"$v_w = {vw}$" + "\n" +
                         rf"rel RMSE = {rmse_shown*100:.1f}\%", fontsize=9)
        ax_top.grid(alpha=0.2)
        ax_top.legend(frameon=False, fontsize=6.5)
        ax_top.tick_params(labelsize=7)

        ax_bot.axhline(0, color="k", lw=1.0, ls="--")
        ax_bot.set_xscale("log")
        ax_bot.set_xlabel(rf"$x = t_p H_*^{{\beta}}$", fontsize=8)
        ax_bot.set_ylabel("rel residual", fontsize=8)
        ax_bot.grid(alpha=0.2)
        ax_bot.tick_params(labelsize=7)
        ylim = max(abs(r) for r in all_res) * 1.15 if all_res else 1.0
        ax_bot.set_ylim(-ylim, ylim)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig6_all_vw_model.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig6_all_vw_model.png")


# ── shift-pipeline figures ────────────────────────────────────────────────────

def _s_formula(tp, vw, A, m, p, vw_ref=0.9):
    return 1.0 + A * (np.power(vw_ref / np.asarray(vw), m) - 1.0) * np.power(np.asarray(tp), p)


def make_s_tp_vw(outname="fig7_s_tp_vw.png"):
    """
    Fig 7 — inferred pointwise shift s vs tp for each vw.
    Panel A (large): scatter of all s(tp) coloured by vw, fitted formula as lines.
    Panel B (inset-style, right): s vs vw at fixed tp quantiles to show the vw dependence.
    """
    if not SHIFT_TABLE.exists() or not SHIFT_POWERLAW.exists():
        print("  skipped fig7 (shift table / powerlaw json not found)")
        return

    df = pd.read_csv(SHIFT_TABLE)
    df = df[df["in_range"]].copy()
    params = json.loads(SHIFT_POWERLAW.read_text())["best_params_named"]
    A, m, p = float(params["A"]), float(params["m"]), float(params["p"])
    vw_ref = 0.9

    vw_values = np.sort(df["v_w"].unique())
    cmap = mpl.colormaps["viridis"]
    vw_colors = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                              gridspec_kw={"width_ratios": [2, 1]})

    # ── left panel: s vs tp ──
    ax = axes[0]
    for vw in vw_values:
        sub = df[np.isclose(df["v_w"], vw, atol=1e-9)].copy()
        ax.scatter(sub["tp"], sub["s_pointwise"], s=10,
                   color=vw_colors[vw], alpha=0.30, rasterized=True)
        tp_fit = np.geomspace(float(sub["tp"].min()), float(sub["tp"].max()), 400)
        s_fit  = _s_formula(tp_fit, vw, A, m, p)
        ax.plot(tp_fit, s_fit, color=vw_colors[vw], lw=2.2,
                label=rf"$v_w = {vw:.1f}$")

    ax.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"$t_p$", fontsize=12)
    ax.set_ylabel(r"$s(t_p,\, v_w)$", fontsize=12)
    ax.set_title(
        r"Effective-time shift $s = x_{\rm eff}/x_{\rm base}$" + "\n"
        + rf"$s = 1 + {A:.3f}\left[(0.9/v_w)^{{{m:.3f}}} - 1\right] t_p^{{{p:.3f}}}$",
        fontsize=10
    )
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.25)

    # ── right panel: s vs vw at three fixed tp quantiles ──
    ax2 = axes[1]
    tp_all = df["tp"].to_numpy()
    tp_quantiles = np.quantile(tp_all, [0.20, 0.50, 0.80])
    qcolors = ["#d62728", "#1f77b4", "#2ca02c"]
    qlabels = [r"$t_p$ = 20th pct", r"$t_p$ = 50th pct", r"$t_p$ = 80th pct"]
    vw_fine = np.linspace(0.25, 0.95, 300)
    for tp_q, qc, ql in zip(tp_quantiles, qcolors, qlabels):
        s_curve = _s_formula(tp_q, vw_fine, A, m, p)
        ax2.plot(vw_fine, s_curve, color=qc, lw=2.0, label=rf"{ql} ($t_p={tp_q:.2f}$)")
        # scatter: binned median at that tp percentile ±10%
        lo, hi = tp_q * 0.90, tp_q * 1.10
        for vw in vw_values:
            sub = df[np.isclose(df["v_w"], vw, atol=1e-9) &
                     (df["tp"] >= lo) & (df["tp"] <= hi)]
            if sub.empty:
                continue
            ax2.scatter(vw, sub["s_pointwise"].median(), s=50,
                        color=qc, zorder=4, marker="D")

    ax2.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.6)
    ax2.set_xlabel(r"$v_w$", fontsize=12)
    ax2.set_ylabel(r"$s(t_p,\, v_w)$", fontsize=12)
    ax2.set_title(r"$s$ vs $v_w$ at fixed $t_p$", fontsize=10)
    ax2.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTDIR / outname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {outname}")


def make_shift_collapse(outname="fig8_shift_collapse.png"):
    """
    Fig 8 — all-vw collapse after applying the shift.
    6 panels (one per theta0 subset), each showing xi vs x_eff for all vw and H*,
    with the vw=0.9 FANH baseline curve overlaid.  Points collapse onto a single curve.
    """
    if not SHIFT_APPLIED.exists() or not SHIFT_POWERLAW.exists():
        print("  skipped fig8 (shift-applied predictions not found)")
        return

    pred = pd.read_csv(SHIFT_APPLIED)
    params_json = json.loads(SHIFT_POWERLAW.read_text())

    theta_vals = np.sort(pred["theta"].unique())
    # pick up to 6 representative theta values
    if len(theta_vals) > 6:
        idxs = np.linspace(0, len(theta_vals) - 1, 6, dtype=int)
        theta_vals = theta_vals[idxs]

    vw_values  = np.sort(pred["v_w"].unique())
    h_values   = np.sort(pred["H"].unique())
    cmap       = mpl.colormaps["viridis"]
    vw_colors_local  = {vw: cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    marker_map = {1.0: "s", 1.5: "^", 2.0: "D", 0.5: "o"}

    ncols = 3
    nrows = int(np.ceil(len(theta_vals) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()
    fig.suptitle(
        r"All-$v_w$ collapse after effective-time shift  "
        r"($x_{\rm eff} = s(t_p, v_w)\cdot t_p H_*^\beta$)",
        fontsize=12, y=1.01
    )

    # load baseline model for overlay
    model_path = ROOT / "results_tosc_lattice_vw0p9_H1p0H1p5H2p0" / "collapse_and_fit_fanh" / "global_fit.json"
    model = json.loads(model_path.read_text()) if model_path.exists() else None

    for i, theta in enumerate(theta_vals):
        ax = axes[i]
        sub = pred[np.isclose(pred["theta"], theta, atol=5e-4)].copy()

        for vw in vw_values:
            for H in h_values:
                cur = sub[
                    np.isclose(sub["v_w"], vw, atol=1e-9) &
                    np.isclose(sub["H"], H, atol=1e-9)
                ].sort_values("x_eff")
                if cur.empty:
                    continue
                ax.scatter(cur["x_eff"], cur["xi"],
                           s=18, color=vw_colors_local[vw],
                           marker=marker_map.get(H, "o"), alpha=0.80)

        # FANH baseline model curve
        if model is not None:
            finf_map_m = {float(k): (float(v["value"]) if isinstance(v, dict) else float(v))
                          for k, v in model["F_inf"].items()}
            theta_keys = np.array(sorted(finf_map_m.keys()))
            idx_th     = int(np.argmin(np.abs(theta_keys - float(theta))))
            finf_th    = finf_map_m[theta_keys[idx_th]]

            f0_sub = float(sub["F0"].iloc[0]) if not sub.empty else 1.0
            x_range = sub["x_eff"].dropna()
            if len(x_range) > 1:
                xfine = np.geomspace(float(x_range.min()), float(x_range.max()), 300)
                tc, r = float(model["t_c"]), float(model["r"])
                xi_model = xfine**1.5 * finf_th / f0_sub**2 + 1.0 / (1.0 + (xfine / tc)**r)
                ax.plot(xfine, xi_model, "k-", lw=1.8, zorder=5, label="FANH model")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(alpha=0.2)
        ax.set_title(rf"$\theta_0 = {theta:.3f}$", fontsize=9)
        ax.set_xlabel(r"$x_{\rm eff}$", fontsize=9)
        ax.set_ylabel(r"$\xi$", fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.plot([], [], "k-", lw=1.8, label="FANH model")

    # shared legend for vw colours
    vw_handles = [mpl.lines.Line2D([0], [0], color=vw_colors_local[vw], lw=2.0,
                                    label=rf"$v_w={vw:.1f}$") for vw in vw_values]
    h_handles  = [mpl.lines.Line2D([0], [0], color="grey", marker=marker_map.get(H, "o"),
                                    linestyle="None", ms=6, label=rf"$H_*={H:g}$")
                  for H in h_values]
    fig.legend(handles=vw_handles + h_handles,
               loc="lower center", ncol=len(vw_values) + len(h_values),
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.03))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTDIR / outname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {outname}")


def make_residual_comparison(outname="fig9_residual_comparison.png"):
    """
    Fig 9 — fractional residual (xi_pred - xi)/xi for each vw,
    comparing: baseline (no shift) vs shift model.
    2-row × 4-col layout.  Residuals vs x = tp H*^beta.
    Inset bar chart shows global rel-RMSE improvement.
    """
    if not SHIFT_APPLIED.exists():
        print("  skipped fig9 (shift-applied predictions not found)")
        return

    pred = pd.read_csv(SHIFT_APPLIED)
    vw_values = np.sort(pred["v_w"].unique())

    fig, axes = plt.subplots(2, 4, figsize=(16, 7),
                              gridspec_kw={"height_ratios": [1, 1]},
                              sharex=False, sharey=False)
    fig.suptitle(
        r"Fractional residual $(\xi_{\rm pred} - \xi)/\xi$ — baseline vs shift model",
        fontsize=12
    )

    rmse_baseline, rmse_shift = [], []

    for col, vw in enumerate(vw_values):
        sub = pred[np.isclose(pred["v_w"], vw, atol=1e-9)].copy()
        x   = sub["x_base"].to_numpy()
        res_base  = sub["frac_resid_baseline"].to_numpy()
        res_shift = sub["frac_resid_shift"].to_numpy()
        good = np.isfinite(x) & np.isfinite(res_base) & np.isfinite(res_shift) & (x > 0)

        rmse_b = float(np.sqrt(np.mean(np.square(res_base[good]))))
        rmse_s = float(np.sqrt(np.mean(np.square(res_shift[good]))))
        rmse_baseline.append(rmse_b)
        rmse_shift.append(rmse_s)

        h_vals = np.sort(sub["H"].unique())
        cmap_h = mpl.colormaps["plasma"]
        hcolors = {H: cmap_h(i / max(len(h_vals) - 1, 1)) for i, H in enumerate(h_vals)}
        mk = {1.0: "s", 1.5: "^", 2.0: "D", 0.5: "o"}

        for row, (res_col, label, rmse) in enumerate([
            (res_base,  "baseline", rmse_b),
            (res_shift, "shift",    rmse_s),
        ]):
            ax = axes[row, col]
            for H in h_vals:
                hmask = good & np.isclose(sub["H"].to_numpy(), H, atol=1e-9)
                ax.scatter(x[hmask], res_col[hmask],
                           s=14, color=hcolors[H], marker=mk.get(H, "o"),
                           alpha=0.50, label=rf"$H_*={H:g}$")
            ax.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.7)
            ax.set_xscale("log")
            ax.grid(alpha=0.20)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(rf"$v_w = {vw}$"
                             + f"\nRMSE = {rmse_b*100:.1f}%", fontsize=9)
            else:
                ax.set_title(f"After shift\nRMSE = {rmse_s*100:.1f}%", fontsize=9)
            if col == 0:
                ax.set_ylabel(
                    "baseline residual" if row == 0 else "shift residual",
                    fontsize=8
                )
            if row == 1:
                ax.set_xlabel(r"$x = t_p H_*^\beta$", fontsize=8)
            if col == 0 and row == 0:
                ax.legend(frameon=False, fontsize=6.5, loc="best")

            # symmetric y-limit
            ylim = min(np.nanpercentile(np.abs(res_col[good]), 98) * 1.3, 1.0)
            ax.set_ylim(-ylim, ylim)

    # ── bar chart inset: RMSE improvement ──
    ax_bar = fig.add_axes([0.91, 0.38, 0.075, 0.52])
    x_pos  = np.arange(len(vw_values))
    width  = 0.35
    ax_bar.bar(x_pos - width/2, [r * 100 for r in rmse_baseline],
               width, label="baseline", color="#d62728", alpha=0.80)
    ax_bar.bar(x_pos + width/2, [r * 100 for r in rmse_shift],
               width, label="shift",    color="#1f77b4", alpha=0.80)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([f"{vw:.1f}" for vw in vw_values], fontsize=7)
    ax_bar.set_xlabel(r"$v_w$", fontsize=7)
    ax_bar.set_ylabel("rel RMSE (%)", fontsize=7)
    ax_bar.set_title("RMSE summary", fontsize=7)
    ax_bar.legend(frameon=False, fontsize=6)
    ax_bar.tick_params(labelsize=6)
    ax_bar.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 0.90, 1.0])
    fig.savefig(OUTDIR / outname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {outname}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data ...")
    frames = load_data()
    fits   = load_fits()

    print("\nGenerating figures ...")

    # Fig 1: xi vs beta/H*
    make_4vw_4H_figure(
        frames,
        x_col="beta_over_H", y_col="xi",
        xlabel=r"$\beta / H_*$", ylabel=r"$\xi$",
        title=r"$\xi$ vs $\beta/H_*$ for all wall velocities and $H_*$ values",
        fname="fig1_xi_vs_betaH.png",
        logx=True, logy=True,
    )

    # Fig 2: xi vs tp
    make_4vw_4H_figure(
        frames,
        x_col="tp", y_col="xi",
        xlabel=r"$t_p$", ylabel=r"$\xi$",
        title=r"$\xi$ vs $t_p$ for all wall velocities and $H_*$ values",
        fname="fig2_xi_vs_tp.png",
        logx=True, logy=True,
    )

    # Fig 3: ftilde vs tp  (collapse view)
    make_ftilde_collapsed_figure(
        frames, x_col="tp",
        xlabel=r"$t_p$",
        fname="fig3_ftilde_vs_tp.png",
    )

    # Fig 4: ftilde vs beta/H*  (collapse view)
    make_ftilde_collapsed_figure(
        frames, x_col="beta_over_H",
        xlabel=r"$\beta / H_*$",
        fname="fig4_ftilde_vs_betaH.png",
    )

    # Fig 5: vw=0.9 best fit
    if fits is not None:
        make_vw09_best_fit(frames, fits)
    else:
        print("  skipped fig5 (no fit data)")

    # Fig 6: vw=0.9 model applied to all vw
    if fits is not None:
        make_all_vw_model_comparison(frames, fits)
    else:
        print("  skipped fig6 (no fit data)")

    # Fig 7: s(tp, vw) function — inferred shift + fitted formula
    make_s_tp_vw()

    # Fig 8: all-vw collapse after applying the shift
    make_shift_collapse()

    # Fig 9: residual comparison baseline vs shift
    make_residual_comparison()

    print(f"\nAll figures saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
