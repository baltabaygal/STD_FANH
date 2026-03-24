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

FITS_PATH  = ROOT / "results_vw_param_evolution" / "per_vw_fits.json"

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
            x = pts["tp"].to_numpy() * pts["H"].to_numpy() ** beta
            xi = pts["xi"].to_numpy()
            f0 = float(pts["F0"].iloc[0])
            good = np.isfinite(x) & np.isfinite(xi) & (x > 0) & (xi > 0)

            xi_pred = x[good]**1.5 * finf_th / f0**2 + 1.0 / (1.0 + (x[good] / tc)**r)
            residual = (xi_pred - xi[good]) / np.maximum(xi[good], 1e-12)

            ax_top.scatter(x[good], xi[good], s=16, marker=H_MARKERS[H],
                           color=H_COLORS[H], alpha=0.8, label=rf"$H_*={H}$")
            ax_bot.scatter(x[good], residual, s=16, marker=H_MARKERS[H],
                           color=H_COLORS[H], alpha=0.8)

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

    print(f"\nAll figures saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
