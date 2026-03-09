import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Plot Y2(theta0, tp) as a contour map.")
    p.add_argument("--data", type=str, default="ode/analysis/data/dm_tp_fitready_H1p000.txt")
    p.add_argument("--outdir", type=str, default="ode/analysis/results")
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def load_table(path):
    arr = np.loadtxt(path, comments="#")
    if arr.ndim != 2 or arr.shape[1] < 10:
        raise ValueError(f"Expected at least 10 columns in {path}, got shape {arr.shape}")
    return arr


def main():
    args = parse_args()
    data_path = Path(args.data).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    arr = load_table(data_path)
    theta0 = arr[:, 2]
    tp = arr[:, 3]
    fanh_no = arr[:, 8]
    xi = arr[:, 9]
    y2 = xi * fanh_no / np.power(tp, 1.5)

    th_unique = np.unique(theta0)
    tp_unique = np.unique(tp)
    grid = np.full((len(th_unique), len(tp_unique)), np.nan, dtype=np.float64)

    for i, th0 in enumerate(th_unique):
        for j, tpi in enumerate(tp_unique):
            mask = np.isclose(theta0, th0) & np.isclose(tp, tpi)
            if mask.any():
                grid[i, j] = y2[mask][0]

    levels = np.geomspace(np.nanmin(grid), np.nanmax(grid), 12)

    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    cf = ax.contourf(tp_unique, th_unique, grid, levels=levels, cmap="magma")
    cs = ax.contour(tp_unique, th_unique, grid, levels=levels, colors="white", linewidths=0.7, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    ax.set_xscale("log")
    ax.set_xlabel(r"$t_p$")
    ax.set_ylabel(r"$\theta_0$")
    ax.set_title(r"$Y_2(\theta_0,t_p)=\xi f_{\rm anh}^{\rm noPT}/t_p^{3/2}$")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"$Y_2$")
    fig.tight_layout()

    out_path = outdir / f"y2_contour_{data_path.stem}.png"
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    log_grid = np.log10(grid)
    log_levels = np.linspace(np.nanmin(log_grid), np.nanmax(log_grid), 12)

    fig2, ax2 = plt.subplots(figsize=(8.2, 5.5))
    cf2 = ax2.contourf(tp_unique, th_unique, log_grid, levels=log_levels, cmap="magma")
    cs2 = ax2.contour(tp_unique, th_unique, log_grid, levels=log_levels, colors="white", linewidths=0.7, alpha=0.7)
    ax2.clabel(cs2, inline=True, fontsize=7, fmt="%.2f")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$t_p$")
    ax2.set_ylabel(r"$\theta_0$")
    ax2.set_title(r"$\log_{10} Y_2(\theta_0,t_p)$")
    cbar2 = fig2.colorbar(cf2, ax=ax2)
    cbar2.set_label(r"$\log_{10}(Y_2)$")
    fig2.tight_layout()

    log_out_path = outdir / f"y2_log10_contour_{data_path.stem}.png"
    fig2.savefig(log_out_path, dpi=args.dpi)
    plt.close(fig2)

    print(f"Loaded: {data_path}")
    print(f"Saved: {out_path}")
    print(f"Saved: {log_out_path}")


if __name__ == "__main__":
    main()
