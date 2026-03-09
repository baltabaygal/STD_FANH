# run_dm_xi.py
import numpy as np
import matplotlib.pyplot as plt

from percolation import PercolationCache
from hom_ode import solve_noPT, solve_PT, Ea3_of_solution

# ---------------- CONFIG ----------------
VW_LIST     = [0.3, 0.5, 0.7, 0.9]
H_LIST      = [0.5, 1.0, 1.5, 2.0]
THETA0_LIST = [0.2618, 0.7854, 1.309, 1.833, 2.356, 2.88]

BETA_OVER_H_MIN = 4.0
BETA_OVER_H_MAX = 40.0
N_BETA          = 30

# ODE numerics
T_START_NOPT = 1e-3
T_END_MIN    = 800.0
EXTRA_AFTER  = 400.0
RTOL         = 1e-9
ATOL         = 1e-11
METHOD       = "DOP853"
LATE_FRAC    = 0.25
LATE_MODE    = "time_weighted"   # or "median"

# plots
PLOT_LOGX = True
PLOT_LOGY = True
DPI       = 200
OUTDIR    = "."
# ----------------------------------------

BETA_GRID = np.logspace(np.log10(BETA_OVER_H_MIN), np.log10(BETA_OVER_H_MAX), N_BETA)
perc = PercolationCache()

print("Precomputing no-PT reference Ea^3(theta0)...")
Ea3_noPT = {}
for th0 in THETA0_LIST:
    Ea3, _ = solve_noPT(
        th0,
        t_start=T_START_NOPT, t_end=T_END_MIN,
        method=METHOD, rtol=RTOL, atol=ATOL,
        late_frac=LATE_FRAC, late_mode=LATE_MODE,
    )
    Ea3_noPT[th0] = Ea3
    print(f"  theta0={th0:.4f}  Ea3_noPT={Ea3:.6g}")

results = {}   # results[vw][theta0][H] = xi array
tp_map  = {}   # tp_map[vw][H] = tp array over BETA_GRID (independent of theta0)

for vw in VW_LIST:
    results[vw] = {}
    tp_map[vw] = {}
    for H in H_LIST:
        tp_arr = np.array([perc.get(H, bH, vw) for bH in BETA_GRID], dtype=np.float64)
        tp_map[vw][H] = tp_arr

    for th0 in THETA0_LIST:
        results[vw][th0] = {}
        ref = Ea3_noPT.get(th0, np.nan)
        if (not np.isfinite(ref)) or (ref <= 0):
            print(f"WARNING: noPT invalid for theta0={th0}")
            continue

        for H in H_LIST:
            tp_arr = tp_map[vw][H]
            xi = np.full_like(tp_arr, np.nan, dtype=np.float64)

            for k, t_p in enumerate(tp_arr):
                if not np.isfinite(t_p):
                    continue
                Ea3PT, _ = solve_PT(
                    th0, t_p,
                    t_end_min=T_END_MIN, extra_after=EXTRA_AFTER,
                    method=METHOD, rtol=RTOL, atol=ATOL,
                    late_frac=LATE_FRAC, late_mode=LATE_MODE,
                )
                if np.isfinite(Ea3PT):
                    xi[k] = Ea3PT / ref

            results[vw][th0][H] = xi
            fin = xi[np.isfinite(xi)]
            if len(fin):
                print(f"vw={vw} theta0={th0:.4f} H={H:g}  xi=[{fin.min():.3g},{fin.max():.3g}]")
            else:
                print(f"vw={vw} theta0={th0:.4f} H={H:g}  xi=all NaN")

# ---------------- Plot 1: per vw, grid over theta0, curves over H ----------------
colorsH = plt.cm.viridis(np.linspace(0.0, 0.9, len(H_LIST)))

for vw in VW_LIST:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False)
    axf = axes.ravel()

    for i, th0 in enumerate(THETA0_LIST):
        ax = axf[i]
        for ci, H in enumerate(H_LIST):
            xi = results[vw][th0].get(H, None)
            if xi is None:
                continue
            m = np.isfinite(xi)
            if m.sum() < 2:
                continue
            ax.plot(BETA_GRID[m], xi[m], lw=2, color=colorsH[ci], label=f"H*={H:g}")
        ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5)
        ax.set_title(rf"$\theta_0={th0:.4g}$")
        ax.set_xlabel(r"$\beta/H_*$")
        ax.set_ylabel(r"$\xi_{\rm DM}$")
        ax.grid(alpha=0.25)
        if PLOT_LOGX: ax.set_xscale("log")
        if PLOT_LOGY: ax.set_yscale("log")
        if i == 0:
            ax.legend(frameon=False, fontsize=8, ncol=2)

    for j in range(len(THETA0_LIST), 6):
        axf[j].set_visible(False)

    fig.suptitle(rf"$\xi_{{\rm DM}}$ (hom. ODE) | $v_w={vw}$", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{OUTDIR}/xi_DM_ODE_vw{vw}.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print("Saved:", out)

# ---------------- Plot 2: per vw, panels over H, curves over theta0 ----------------
colorsT = plt.cm.plasma(np.linspace(0.0, 0.85, len(THETA0_LIST)))

for vw in VW_LIST:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False)
    axf = axes.ravel()

    for pi, H in enumerate(H_LIST):
        ax = axf[pi]
        for ci, th0 in enumerate(THETA0_LIST):
            xi = results[vw][th0].get(H, None)
            if xi is None:
                continue
            m = np.isfinite(xi)
            if m.sum() < 2:
                continue
            ax.plot(BETA_GRID[m], xi[m], lw=2, color=colorsT[ci], label=rf"$\theta_0={th0:.3g}$")
        ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5)
        ax.set_title(rf"$H_*/M_\phi={H:g}$")
        ax.set_xlabel(r"$\beta/H_*$")
        ax.set_ylabel(r"$\xi_{\rm DM}$")
        ax.grid(alpha=0.25)
        if PLOT_LOGX: ax.set_xscale("log")
        if PLOT_LOGY: ax.set_yscale("log")
        if pi == 0:
            ax.legend(frameon=False, fontsize=8, ncol=2)

    for j in range(len(H_LIST), 6):
        axf[j].set_visible(False)

    fig.suptitle(rf"$\xi_{{\rm DM}}$ (hom. ODE) | $v_w={vw}$ | all $\theta_0$ overlaid per $H_*$", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{OUTDIR}/xi_DM_ODE_alltheta_vw{vw}.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print("Saved:", out)

# ---------------- Plot 3: t_p vs beta/H ----------------
fig, axes = plt.subplots(1, len(VW_LIST), figsize=(5*len(VW_LIST), 4), squeeze=False)
colorsH2 = plt.cm.viridis(np.linspace(0.0, 0.9, len(H_LIST)))

for vi, vw in enumerate(VW_LIST):
    ax = axes[0, vi]
    for ci, H in enumerate(H_LIST):
        tp = tp_map[vw][H]
        m = np.isfinite(tp)
        if m.sum() < 2:
            continue
        ax.plot(BETA_GRID[m], tp[m], lw=2, color=colorsH2[ci], label=f"H*={H:g}")
    ax.set_title(rf"$v_w={vw}$")
    ax.set_xlabel(r"$\beta/H_*$")
    ax.set_ylabel(r"$t_p \; (M_\phi^{-1})$")
    ax.grid(alpha=0.25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if vi == 0:
        ax.legend(frameon=False, fontsize=8)

fig.suptitle(r"Percolation time $t_p$ vs $\beta/H_*$", y=1.02)
fig.tight_layout()
out = f"{OUTDIR}/t_perc_vs_betaH.png"
fig.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("Saved:", out)

# ---------------- Save table ----------------
out_txt = f"{OUTDIR}/xi_DM_ODE_results.txt"
with open(out_txt, "w") as f:
    f.write("# vw  theta0  H_star  beta_over_H  t_p  xi_DM\n")
    for vw in VW_LIST:
        for th0 in THETA0_LIST:
            for H in H_LIST:
                xi = results[vw][th0].get(H, None)
                if xi is None:
                    continue
                tp = tp_map[vw][H]
                for k, bH in enumerate(BETA_GRID):
                    if np.isfinite(xi[k]) and np.isfinite(tp[k]):
                        f.write(f"{vw:.6g}  {th0:.6g}  {H:.6g}  {bH:.6g}  {tp[k]:.8e}  {xi[k]:.8e}\n")

print("Saved table:", out_txt)
print("ALL DONE.")
