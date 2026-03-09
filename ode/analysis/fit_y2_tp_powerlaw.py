import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit Y2(tp) per theta0 with a plateau plus power-law decay."
    )
    p.add_argument("--data", type=str, default="ode/analysis/data/dm_tp_fitready_H1p000.txt")
    p.add_argument("--outdir", type=str, default="ode/analysis/results")
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def load_table(path):
    arr = np.loadtxt(path, comments="#")
    if arr.ndim != 2 or arr.shape[1] < 10:
        raise ValueError(f"Expected at least 10 columns in {path}, got shape {arr.shape}")
    return arr


def y2_powerlaw(tp, c0, c1, p):
    tp = np.asarray(tp, dtype=np.float64)
    return c0 + c1 / np.power(tp, p)


def rel_rmse(y, yfit):
    return float(np.sqrt(np.mean(np.square((yfit - y) / y))))


def log_rmse(y, yfit):
    return float(np.sqrt(np.mean(np.square(np.log(yfit) - np.log(y)))))


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

    rows = []
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 10.8), squeeze=False)
    axes = axes.ravel()

    for ax, th0 in zip(axes, np.unique(theta0)):
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
        rr = rel_rmse(y, yfit)
        lr = log_rmse(y, yfit)
        rows.append((th0, popt[0], popt[1], popt[2], lr, rr))

        x_dense = np.logspace(np.log10(x.min()), np.log10(x.max()), 300)
        ax.plot(x, y, "ko", ms=4, label="data")
        ax.plot(x_dense, y2_powerlaw(x_dense, *popt), "-", lw=1.8, color="tab:blue", label=f"fit, rel={rr:.3f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(rf"$\theta_0={th0:.3g}$")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$Y_2$")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    plot_out = outdir / f"fit_y2_tp_powerlaw_{data_path.stem}.png"
    fig.savefig(plot_out, dpi=args.dpi)
    plt.close(fig)

    rows = np.array(rows, dtype=np.float64)
    txt_out = outdir / f"fit_y2_tp_powerlaw_{data_path.stem}.txt"
    with open(txt_out, "w") as f:
        f.write("# theta0 c0 c1 p log_rmse rel_rmse\n")
        for row in rows:
            f.write(f"{row[0]:.8g} {row[1]:.10e} {row[2]:.10e} {row[3]:.10e} {row[4]:.10e} {row[5]:.10e}\n")

    trend_out = outdir / f"fit_y2_tp_powerlaw_trends_{data_path.stem}.png"
    fig2, axes2 = plt.subplots(1, 3, figsize=(12.2, 3.8))
    axes2[0].plot(rows[:, 0], rows[:, 1], "-o", ms=4)
    axes2[0].set_xlabel(r"$\theta_0$")
    axes2[0].set_ylabel(r"$c_0$")
    axes2[0].grid(alpha=0.25)

    axes2[1].plot(rows[:, 0], rows[:, 2], "-o", ms=4, color="tab:red")
    axes2[1].set_xlabel(r"$\theta_0$")
    axes2[1].set_ylabel(r"$c_1$")
    axes2[1].grid(alpha=0.25)

    axes2[2].plot(rows[:, 0], rows[:, 3], "-o", ms=4, color="tab:green")
    axes2[2].set_xlabel(r"$\theta_0$")
    axes2[2].set_ylabel(r"$p$")
    axes2[2].grid(alpha=0.25)

    fig2.tight_layout()
    fig2.savefig(trend_out, dpi=args.dpi)
    plt.close(fig2)

    print(f"Loaded: {data_path}")
    print(f"Saved: {plot_out}")
    print(f"Saved: {txt_out}")
    print(f"Saved: {trend_out}")
    print("Mean rel RMSE: {:.4e}".format(float(np.mean(rows[:, 5]))))


if __name__ == "__main__":
    main()
