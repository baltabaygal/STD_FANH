import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


def parse_args():
    p = argparse.ArgumentParser(
        description="Refine compact physical Y2(theta0,tp) models directly on the full surface."
    )
    p.add_argument("--data", type=str, default="ode/analysis/data/dm_tp_fitready_H1p000.txt")
    p.add_argument("--outdir", type=str, default="ode/analysis/results")
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def rel_rmse(y, yfit):
    return float(np.sqrt(np.mean(np.square((yfit - y) / y))))


def log_rmse(y, yfit):
    return float(np.sqrt(np.mean(np.square(np.log(yfit) - np.log(y)))))


def features(theta0):
    u = theta0 / np.pi
    s2 = np.sin(theta0 / 2.0) ** 2
    h = np.log(np.e / np.clip(np.cos(theta0 / 2.0) ** 2, 1e-12, None))
    return u, s2, h


def model_powh_quadH_linC0(theta0, tp, params):
    a, alpha, b0, b1, b2, p0, p1 = params
    _, _, h = features(theta0)
    c0 = a * np.power(h, alpha)
    logc1 = b0 + b1 * h + b2 * h * h
    p = p0 + p1 * c0
    return c0 + np.exp(logc1) / np.power(tp, p)


def model_powh_quadC0_linC0(theta0, tp, params):
    a, alpha, b0, b1, b2, p0, p1 = params
    _, _, h = features(theta0)
    c0 = a * np.power(h, alpha)
    logc1 = b0 + b1 * c0 + b2 * c0 * c0
    p = p0 + p1 * c0
    return c0 + np.exp(logc1) / np.power(tp, p)


def model_powh_quadH_linH(theta0, tp, params):
    a, alpha, b0, b1, b2, p0, p1 = params
    _, _, h = features(theta0)
    c0 = a * np.power(h, alpha)
    logc1 = b0 + b1 * h + b2 * h * h
    p = p0 + p1 * h
    return c0 + np.exp(logc1) / np.power(tp, p)


def model_powh_quadH_linS2(theta0, tp, params):
    a, alpha, b0, b1, b2, p0, p1 = params
    _, s2, h = features(theta0)
    c0 = a * np.power(h, alpha)
    logc1 = b0 + b1 * h + b2 * h * h
    p = p0 + p1 * s2
    return c0 + np.exp(logc1) / np.power(tp, p)


MODELS = {
    "powh_quadH_linC0": {
        "fn": model_powh_quadH_linC0,
        "p0": np.array([5.4012301318e-01, 1.4906181818e-01, -2.1048806032e00, 9.5719768221e-01, -6.8247723962e-02, 2.6400849878e00, -1.7107737558e00]),
    },
    "powh_quadC0_linC0": {
        "fn": model_powh_quadC0_linC0,
        "p0": np.array([5.4012301318e-01, 1.4906181818e-01, -2.3794649927e00, -8.1013038802e00, 1.8914491709e01, 2.6400849878e00, -1.7107737558e00]),
    },
    "powh_quadH_linH": {
        "fn": model_powh_quadH_linH,
        "p0": np.array([5.4012301318e-01, 1.4906181818e-01, -2.1048806032e00, 9.5719768221e-01, -6.8247723962e-02, 1.8024227944e00, -2.1544438027e-01]),
    },
    "powh_quadH_linS2": {
        "fn": model_powh_quadH_linS2,
        "p0": np.array([5.4012301318e-01, 1.4906181818e-01, -2.1048806032e00, 9.5719768221e-01, -6.8247723962e-02, 1.7906057027e00, -1.9657266176e-01]),
    },
}


def residuals(params, theta0, tp, y, fn):
    yfit = fn(theta0, tp, params)
    return (np.log(yfit) - np.log(y))


def fit_model(theta0, tp, y, name, spec):
    lower = np.array([1e-6, 0.0, -20.0, -20.0, -20.0, 0.1, -10.0], dtype=np.float64)
    upper = np.array([10.0, 3.0, 20.0, 20.0, 20.0, 5.0, 2.0], dtype=np.float64)
    res = least_squares(
        residuals,
        spec["p0"],
        bounds=(lower, upper),
        args=(theta0, tp, y, spec["fn"]),
        max_nfev=50000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    yfit = spec["fn"](theta0, tp, res.x)
    return {
        "name": name,
        "params": res.x,
        "cost": float(res.cost),
        "success": bool(res.success),
        "message": str(res.message),
        "global_log_rmse": log_rmse(y, yfit),
        "global_rel_rmse": rel_rmse(y, yfit),
        "yfit": yfit,
    }


def save_summary(path, results):
    with open(path, "w") as f:
        f.write("# name global_log_rmse global_rel_rmse success\n")
        for r in results:
            f.write(
                f"{r['name']} {r['global_log_rmse']:.10e} {r['global_rel_rmse']:.10e} {int(r['success'])}\n"
            )
            f.write("# params " + " ".join(f"{x:.10e}" for x in r["params"]) + "\n")
            f.write(f"# message {r['message']}\n")


def plot_vs_tp(theta0, tp, y, results, out_path, dpi):
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(np.unique(theta0))))
    best = results[0]
    for color, th0 in zip(colors, np.unique(theta0)):
        mask = np.isclose(theta0, th0)
        idx = np.argsort(tp[mask])
        x = tp[mask][idx]
        yd = y[mask][idx]
        yf = best["yfit"][mask][idx]
        ax.plot(x, yd, "o", ms=3, color=color, alpha=0.75)
        ax.plot(x, yf, "-", lw=1.7, color=color, label=rf"$\theta_0={th0:.3g}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t_p$")
    ax.set_ylabel(r"$Y_2$")
    ax.set_title(best["name"] + rf", rel={best['global_rel_rmse']:.4e}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_vs_theta(theta0, tp, y, results, out_path, dpi):
    best = results[0]
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.2), squeeze=False)
    axes = axes.ravel()
    pick_tp = np.unique(tp)[np.linspace(0, len(np.unique(tp)) - 1, 6).astype(int)]
    for ax, tpi in zip(axes, pick_tp):
        mask = np.isclose(tp, tpi)
        idx = np.argsort(theta0[mask])
        th = theta0[mask][idx]
        yd = y[mask][idx]
        yf = best["yfit"][mask][idx]
        ax.plot(th, yd, "ko", ms=4, label="data")
        ax.plot(th, yf, "-", lw=1.8, label=f"fit, rel={rel_rmse(yd, yf):.3e}")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\theta_0$")
        ax.set_ylabel(r"$Y_2$")
        ax.set_title(rf"$t_p={tpi:.3g}$")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    data_path = Path(args.data).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    arr = np.loadtxt(data_path, comments="#")
    theta0 = arr[:, 2]
    tp = arr[:, 3]
    fanh_no = arr[:, 8]
    xi = arr[:, 9]
    y = xi * fanh_no / np.power(tp, 1.5)

    results = []
    for name, spec in MODELS.items():
        results.append(fit_model(theta0, tp, y, name, spec))
    results.sort(key=lambda r: (r["global_rel_rmse"], r["global_log_rmse"]))

    stem = data_path.stem
    save_summary(outdir / f"refine_y2_physical_models_{stem}.txt", results)
    plot_vs_tp(theta0, tp, y, results, outdir / f"refine_y2_physical_models_vs_tp_{stem}.png", args.dpi)
    plot_vs_theta(theta0, tp, y, results, outdir / f"refine_y2_physical_models_vs_theta_{stem}.png", args.dpi)

    print(f"Loaded: {data_path}")
    print(f"Saved: {outdir / f'refine_y2_physical_models_{stem}.txt'}")
    print(f"Saved: {outdir / f'refine_y2_physical_models_vs_tp_{stem}.png'}")
    print(f"Saved: {outdir / f'refine_y2_physical_models_vs_theta_{stem}.png'}")
    for r in results:
        print(
            f"{r['name']}: log={r['global_log_rmse']:.4e} rel={r['global_rel_rmse']:.4e} success={r['success']}"
        )


if __name__ == "__main__":
    main()
