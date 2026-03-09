import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit c0(theta0) and c1(theta0) with shared anharmonic-shape families."
    )
    p.add_argument(
        "--coeff-data",
        type=str,
        default="ode/analysis/results/fit_y2_tp_powerlaw_dm_tp_fitready_H1p000.txt",
        help="Path to per-theta powerlaw coefficient table",
    )
    p.add_argument("--outdir", type=str, default="ode/analysis/results")
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def load_coeffs(path):
    arr = np.loadtxt(path, comments="#")
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"Expected at least 6 columns in {path}, got shape {arr.shape}")
    return arr


def rel_rmse(y, yfit):
    return float(np.sqrt(np.mean(np.square((yfit - y) / y))))


def log_rmse(y, yfit):
    return float(np.sqrt(np.mean(np.square(np.log(yfit) - np.log(y)))))


def power_shape(theta0, amp, alpha):
    u = theta0 / np.pi
    den = np.clip(1.0 - u * u, 1e-12, None)
    return amp / np.power(den, alpha)


def log_shape(theta0, amp, alpha):
    u = theta0 / np.pi
    base = np.log(np.e / np.clip(1.0 - u * u, 1e-12, None))
    return amp * np.power(base, alpha)


def pendulum_log_shape(theta0, amp, alpha):
    base = np.log(np.e / np.clip(np.cos(theta0 / 2.0) ** 2, 1e-12, None))
    return amp * np.power(base, alpha)


def fit_family(theta0, y, family_name, fn):
    popt, _ = curve_fit(
        fn,
        theta0,
        y,
        p0=[np.median(y), 0.3],
        bounds=([1e-12, 0.0], [1e3, 6.0]),
        maxfev=200000,
    )
    yfit = fn(theta0, *popt)
    return {
        "family": family_name,
        "amp": popt[0],
        "alpha": popt[1],
        "log_rmse": log_rmse(y, yfit),
        "rel_rmse": rel_rmse(y, yfit),
        "yfit": yfit,
    }


def save_summary(out_path, results_c0, results_c1):
    with open(out_path, "w") as f:
        f.write("# quantity family amplitude alpha log_rmse rel_rmse\n")
        for result in results_c0:
            f.write(
                f"c0 {result['family']} {result['amp']:.10e} {result['alpha']:.10e} "
                f"{result['log_rmse']:.10e} {result['rel_rmse']:.10e}\n"
            )
        for result in results_c1:
            f.write(
                f"c1 {result['family']} {result['amp']:.10e} {result['alpha']:.10e} "
                f"{result['log_rmse']:.10e} {result['rel_rmse']:.10e}\n"
            )


def make_plot(theta0, c0, c1, results_c0, results_c1, out_path, dpi):
    theta_dense = np.linspace(theta0.min(), theta0.max(), 400)
    families = {
        "power": power_shape,
        "log": log_shape,
        "pendulum_log": pendulum_log_shape,
    }
    colors = {
        "power": "tab:blue",
        "log": "tab:orange",
        "pendulum_log": "tab:green",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))

    axes[0].plot(theta0, c0, "ko", ms=5, label="data")
    for result in results_c0:
        axes[0].plot(
            theta_dense,
            families[result["family"]](theta_dense, result["amp"], result["alpha"]),
            lw=1.8,
            color=colors[result["family"]],
            label=f"{result['family']}, rel={result['rel_rmse']:.3f}",
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\theta_0$")
    axes[0].set_ylabel(r"$c_0(\theta_0)$")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(theta0, c1, "ko", ms=5, label="data")
    for result in results_c1:
        axes[1].plot(
            theta_dense,
            families[result["family"]](theta_dense, result["amp"], result["alpha"]),
            lw=1.8,
            color=colors[result["family"]],
            label=f"{result['family']}, rel={result['rel_rmse']:.3f}",
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\theta_0$")
    axes[1].set_ylabel(r"$c_1(\theta_0)$")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    coeff_path = Path(args.coeff_data).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    arr = load_coeffs(coeff_path)
    theta0 = arr[:, 0]
    c0 = arr[:, 1]
    c1 = arr[:, 2]

    families = {
        "power": power_shape,
        "log": log_shape,
        "pendulum_log": pendulum_log_shape,
    }

    results_c0 = [fit_family(theta0, c0, name, fn) for name, fn in families.items()]
    results_c1 = [fit_family(theta0, c1, name, fn) for name, fn in families.items()]

    stem = coeff_path.stem
    summary_out = outdir / f"fit_y2_coeff_shapes_{stem}.txt"
    plot_out = outdir / f"fit_y2_coeff_shapes_{stem}.png"
    save_summary(summary_out, results_c0, results_c1)
    make_plot(theta0, c0, c1, results_c0, results_c1, plot_out, args.dpi)

    print(f"Loaded: {coeff_path}")
    print(f"Saved: {summary_out}")
    print(f"Saved: {plot_out}")
    for result in results_c0:
        print(
            "c0 {} rel_rmse={:.4e} alpha={:.4f}".format(
                result["family"], result["rel_rmse"], result["alpha"]
            )
        )
    for result in results_c1:
        print(
            "c1 {} rel_rmse={:.4e} alpha={:.4f}".format(
                result["family"], result["rel_rmse"], result["alpha"]
            )
        )


if __name__ == "__main__":
    main()
