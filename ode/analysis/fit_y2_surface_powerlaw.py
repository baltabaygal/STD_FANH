import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit the full Y2(theta0, tp) surface using c0(theta0)+c1(theta0)/tp^p(theta0)."
    )
    p.add_argument("--data", type=str, default="ode/analysis/data/dm_tp_fitready_H1p000.txt")
    p.add_argument("--outdir", type=str, default="ode/analysis/results")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--n-slices", type=int, default=6)
    return p.parse_args()


def load_table(path):
    arr = np.loadtxt(path, comments="#")
    if arr.ndim != 2 or arr.shape[1] < 10:
        raise ValueError(f"Expected at least 10 columns in {path}, got shape {arr.shape}")
    return arr


def y2_powerlaw(tp, c0, c1, p):
    tp = np.asarray(tp, dtype=np.float64)
    return c0 + c1 / np.power(tp, p)


def poly2(u, a0, a1, a2):
    u = np.asarray(u, dtype=np.float64)
    return a0 + a1 * u + a2 * u * u


def exp_poly2(u, b0, b1, b2):
    return np.exp(poly2(u, b0, b1, b2))


def rel_rmse(y, yfit):
    return float(np.sqrt(np.mean(np.square((yfit - y) / y))))


def log_rmse(y, yfit):
    return float(np.sqrt(np.mean(np.square(np.log(yfit) - np.log(y)))))


def fit_per_theta(theta0, tp, y2):
    rows = []
    for th0 in np.unique(theta0):
        mask = np.isclose(theta0, th0) & np.isfinite(tp) & np.isfinite(y2) & (tp > 0) & (y2 > 0)
        x = tp[mask]
        y = y2[mask]
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]
        p0 = [y[-1], max(y[0] - y[-1], 1e-6), 1.0]
        bounds = ([0.0, 0.0, 0.05], [10.0, 1e4, 8.0])
        popt, _ = curve_fit(y2_powerlaw, x, y, p0=p0, bounds=bounds, maxfev=200000)
        yfit = y2_powerlaw(x, *popt)
        rows.append((th0, popt[0], popt[1], popt[2], log_rmse(y, yfit), rel_rmse(y, yfit)))
    return np.array(rows, dtype=np.float64)


def surface_model(xdata, *params):
    theta0, tp = xdata
    u = theta0 / np.pi
    c0 = poly2(u, *params[0:3])
    c1 = exp_poly2(u, *params[3:6])
    p = poly2(u, *params[6:9])
    return y2_powerlaw(tp, c0, c1, p)


def make_theta_param_plot(u, rows, coeffs, out_path, dpi):
    u_dense = np.linspace(u.min(), u.max(), 400)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))

    axes[0].plot(u, rows[:, 1], "o", ms=5, label="per-theta fit")
    axes[0].plot(u_dense, poly2(u_dense, *coeffs[0:3]), "-", lw=1.8, label="quadratic")
    axes[0].set_xlabel(r"$u=\theta_0/\pi$")
    axes[0].set_ylabel(r"$c_0(u)$")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(u, rows[:, 2], "o", ms=5, color="tab:red", label="per-theta fit")
    axes[1].plot(u_dense, exp_poly2(u_dense, *coeffs[3:6]), "-", lw=1.8, color="tab:red", label=r"$e^{\rm quad}$")
    axes[1].set_xlabel(r"$u=\theta_0/\pi$")
    axes[1].set_ylabel(r"$c_1(u)$")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(u, rows[:, 3], "o", ms=5, color="tab:green", label="per-theta fit")
    axes[2].plot(u_dense, poly2(u_dense, *coeffs[6:9]), "-", lw=1.8, color="tab:green", label="quadratic")
    axes[2].set_xlabel(r"$u=\theta_0/\pi$")
    axes[2].set_ylabel(r"$p(u)$")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_surface_tp_plot(theta0, tp, y2, yfit, out_path, dpi):
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(np.unique(theta0))))
    for color, th0 in zip(colors, np.unique(theta0)):
        mask = np.isclose(theta0, th0)
        idx = np.argsort(tp[mask])
        x = tp[mask][idx]
        y = y2[mask][idx]
        yf = yfit[mask][idx]
        ax.plot(x, y, "o", ms=3.5, color=color, alpha=0.75)
        ax.plot(x, yf, "-", lw=1.8, color=color, label=rf"$\theta_0={th0:.3g}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t_p$")
    ax.set_ylabel(r"$Y_2$")
    ax.set_title(r"Surface power-law fit vs data")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_surface_theta_slices(theta0, tp, y2, yfit, out_path, n_slices, dpi):
    tp_unique = np.unique(tp)
    pick = np.unique(tp_unique[np.linspace(0, len(tp_unique) - 1, n_slices).astype(int)])
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.2), squeeze=False)
    axes = axes.ravel()
    for ax, tpi in zip(axes, pick):
        mask = np.isclose(tp, tpi)
        idx = np.argsort(theta0[mask])
        th = theta0[mask][idx]
        y = y2[mask][idx]
        yf = yfit[mask][idx]
        ax.plot(th, y, "ko", ms=4, label="data")
        ax.plot(th, yf, "-", lw=1.8, color="tab:blue", label=f"fit, rel={rel_rmse(y, yf):.3f}")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\theta_0$")
        ax.set_ylabel(r"$Y_2$")
        ax.set_title(rf"$t_p={tpi:.3g}$")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_summary(out_path, coeffs, global_log, global_rel, rows):
    with open(out_path, "w") as f:
        f.write("# global_log_rmse global_rel_rmse\n")
        f.write(f"{global_log:.10e} {global_rel:.10e}\n")
        f.write("# c0 coefficients a0 a1 a2\n")
        f.write(f"{coeffs[0]:.10e} {coeffs[1]:.10e} {coeffs[2]:.10e}\n")
        f.write("# log(c1) coefficients b0 b1 b2\n")
        f.write(f"{coeffs[3]:.10e} {coeffs[4]:.10e} {coeffs[5]:.10e}\n")
        f.write("# p coefficients d0 d1 d2\n")
        f.write(f"{coeffs[6]:.10e} {coeffs[7]:.10e} {coeffs[8]:.10e}\n")
        f.write("# theta0 c0 c1 p log_rmse rel_rmse\n")
        for row in rows:
            f.write(f"{row[0]:.8g} {row[1]:.10e} {row[2]:.10e} {row[3]:.10e} {row[4]:.10e} {row[5]:.10e}\n")


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

    rows = fit_per_theta(theta0, tp, y2)
    u_rows = rows[:, 0] / np.pi
    c0_rows = rows[:, 1]
    c1_rows = rows[:, 2]
    p_rows = rows[:, 3]

    c0_coeffs, _ = curve_fit(poly2, u_rows, c0_rows, p0=[c0_rows[0], 0.0, 0.0], maxfev=200000)
    logc1_coeffs, _ = curve_fit(poly2, u_rows, np.log(c1_rows), p0=[np.log(c1_rows[0]), 0.0, 0.0], maxfev=200000)
    p_coeffs, _ = curve_fit(poly2, u_rows, p_rows, p0=[p_rows[0], 0.0, 0.0], maxfev=200000)
    coeffs = np.concatenate([c0_coeffs, logc1_coeffs, p_coeffs])

    yfit = surface_model((theta0, tp), *coeffs)
    global_log = log_rmse(y2, yfit)
    global_rel = rel_rmse(y2, yfit)

    stem = data_path.stem
    param_plot = outdir / f"fit_y2_surface_powerlaw_params_{stem}.png"
    tp_plot = outdir / f"fit_y2_surface_powerlaw_vs_tp_{stem}.png"
    theta_plot = outdir / f"fit_y2_surface_powerlaw_vs_theta_{stem}.png"
    summary_out = outdir / f"fit_y2_surface_powerlaw_{stem}.txt"

    make_theta_param_plot(u_rows, rows, coeffs, param_plot, args.dpi)
    make_surface_tp_plot(theta0, tp, y2, yfit, tp_plot, args.dpi)
    make_surface_theta_slices(theta0, tp, y2, yfit, theta_plot, args.n_slices, args.dpi)
    save_summary(summary_out, coeffs, global_log, global_rel, rows)

    print(f"Loaded: {data_path}")
    print(f"Saved: {param_plot}")
    print(f"Saved: {tp_plot}")
    print(f"Saved: {theta_plot}")
    print(f"Saved: {summary_out}")
    print(f"Surface fit: log_rmse={global_log:.4e} rel_rmse={global_rel:.4e}")


if __name__ == "__main__":
    main()
