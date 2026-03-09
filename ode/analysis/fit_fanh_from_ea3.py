import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def parse_args():
    p = argparse.ArgumentParser(
        description="Reconstruct f_anh from Ea3 and fit f_anh(theta0,tp)=A(tp)/(1-u^2)^alpha(tp)."
    )
    p.add_argument("--data", type=str, default="data/dm_tp_fitready_H1p000.txt")
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--n-tp-plot", type=int, default=6, help="How many tp slices to plot.")
    return p.parse_args()


def potential(theta0):
    return 1.0 - np.cos(theta0)


def theta_model(theta0, a, alpha):
    u = theta0 / np.pi
    den = np.clip(1.0 - u * u, 1e-10, None)
    return a / (den**alpha)


def main():
    args = parse_args()
    data_path = Path(args.data).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    arr = np.loadtxt(data_path, comments="#")
    # cols:
    # 0 H_star, 1 t_star, 2 theta0, 3 t_p, 4 x=t_p/tosc,
    # 5 Ea3_PT, 6 Ea3_noPT, 7 f_anh_PT, 8 f_anh_noPT, 9 xi_DM ...
    theta0 = arr[:, 2]
    tp = arr[:, 3]
    ea3_pt = arr[:, 5]
    fanh_col = arr[:, 7]

    v = potential(theta0)
    fanh_raw = ea3_pt / (v * np.power(tp, 1.5))

    m = np.isfinite(theta0) & np.isfinite(tp) & np.isfinite(fanh_raw) & (fanh_raw > 0.0)
    theta0 = theta0[m]
    tp = tp[m]
    fanh_raw = fanh_raw[m]
    fanh_col = fanh_col[m]

    # Sanity check: reconstructed f_anh must match stored column.
    rel_diff = np.abs((fanh_raw - fanh_col) / np.clip(np.abs(fanh_col), 1e-30, None))
    max_rel_diff = float(np.max(rel_diff))
    med_rel_diff = float(np.median(rel_diff))

    tp_unique = np.unique(tp)
    rows = []

    for tpi in tp_unique:
        sel = tp == tpi
        th = theta0[sel]
        y = fanh_raw[sel]
        if th.size < 4:
            continue

        idx = np.argsort(th)
        th = th[idx]
        y = y[idx]
        try:
            p0 = [np.median(y), 0.3]
            bounds = ([1e-12, 0.0], [1e3, 5.0])
            popt, _ = curve_fit(theta_model, th, y, p0=p0, bounds=bounds, maxfev=100000)
            yfit = theta_model(th, *popt)
            log_rmse = float(np.sqrt(np.mean((np.log(yfit) - np.log(y)) ** 2)))
            rel_rmse = float(np.sqrt(np.mean(((yfit - y) / y) ** 2)))
            rows.append((tpi, popt[0], popt[1], log_rmse, rel_rmse))
        except Exception:
            continue

    rows = np.array(rows, dtype=np.float64)
    if rows.size == 0:
        raise RuntimeError("No successful tp-slice fits.")

    out_params = outdir / f"fanh_from_ea3_theta_fit_{data_path.stem}.txt"
    with open(out_params, "w") as f:
        f.write("# tp A_tp alpha_tp log_rmse rel_rmse\n")
        for r in rows:
            f.write(f"{r[0]:.10e} {r[1]:.10e} {r[2]:.10e} {r[3]:.10e} {r[4]:.10e}\n")

    # Plot A(tp), alpha(tp)
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.2))
    ax[0].plot(rows[:, 0], rows[:, 1], "-o", ms=3)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$t_p$")
    ax[0].set_ylabel(r"$A(t_p)$")
    ax[0].grid(alpha=0.25)

    ax[1].plot(rows[:, 0], rows[:, 2], "-o", ms=3, color="tab:red")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$t_p$")
    ax[1].set_ylabel(r"$\alpha(t_p)$")
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    out_trends = outdir / f"fanh_from_ea3_A_alpha_vs_tp_{data_path.stem}.png"
    fig.savefig(out_trends, dpi=220)
    plt.close(fig)

    # Plot f_anh(theta0) slices and fitted curves for representative t_p
    tp_sorted = np.sort(tp_unique)
    if args.n_tp_plot < 2:
        pick = np.array([tp_sorted[0]])
    else:
        idx_pick = np.linspace(0, len(tp_sorted) - 1, args.n_tp_plot).astype(int)
        pick = np.unique(tp_sorted[idx_pick])

    fig2, ax2 = plt.subplots(figsize=(7.0, 5.0))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(pick)))
    for c, tpi in zip(colors, pick):
        sel = tp == tpi
        th = theta0[sel]
        y = fanh_raw[sel]
        if th.size < 4:
            continue
        idx = np.argsort(th)
        th = th[idx]
        y = y[idx]

        row_match = rows[np.isclose(rows[:, 0], tpi, rtol=1e-12, atol=0.0)]
        if row_match.size == 0:
            continue
        a_fit, alpha_fit = row_match[0, 1], row_match[0, 2]
        th_dense = np.linspace(th.min(), th.max(), 240)
        y_dense = theta_model(th_dense, a_fit, alpha_fit)

        ax2.plot(th, y, "o", ms=4, color=c, alpha=0.8)
        ax2.plot(th_dense, y_dense, "-", lw=1.8, color=c, label=rf"$t_p={tpi:.3g}$")

    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\theta_0$")
    ax2.set_ylabel(r"$f_{\rm anh}$ (from $E_a a^3$)")
    ax2.set_title(r"Directly reconstructed $f_{\rm anh}(\theta_0,t_p)$ and fits")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False, fontsize=8, ncol=2)
    fig2.tight_layout()
    out_slices = outdir / f"fanh_from_ea3_theta_slices_{data_path.stem}.png"
    fig2.savefig(out_slices, dpi=220)
    plt.close(fig2)

    print(f"Data: {data_path}")
    print(f"max_rel_diff(reconstructed vs column f_anh_PT)={max_rel_diff:.3e}")
    print(f"med_rel_diff(reconstructed vs column f_anh_PT)={med_rel_diff:.3e}")
    print(f"Saved params: {out_params}")
    print(f"Saved trends: {out_trends}")
    print(f"Saved slices: {out_slices}")
    print(
        "mean(log_rmse)={:.4e}, mean(rel_rmse)={:.4e}".format(
            float(np.mean(rows[:, 3])), float(np.mean(rows[:, 4]))
        )
    )


if __name__ == "__main__":
    main()
