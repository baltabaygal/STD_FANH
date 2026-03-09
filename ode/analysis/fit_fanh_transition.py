import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def model_m1(tp, a, p, c):
    # r(tp) = f_anh / f_anh_noPT
    return 1.0 + a / (1.0 + (tp / c) ** p)


def model_m2(tp, a, p, c, q):
    return 1.0 + a / (1.0 + (tp / c) ** p) ** q


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit constrained transition model for f_anh(theta0, t_p)."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dm_tp_fitready_H1p000.txt",
        help="Input fit-ready table from run_dm_tp_fitready.py",
    )
    parser.add_argument("--outdir", type=str, default=".", help="Output directory")
    parser.add_argument("--model", type=str, default="m1", choices=["m1", "m2"])
    return parser.parse_args()


def fit_one_theta(tp, r, model):
    if model == "m1":
        p0 = [3.0, 1.2, 1.0]
        bounds = ([0.0, 0.1, 1e-6], [1e3, 8.0, 1e3])
        popt, _ = curve_fit(model_m1, tp, r, p0=p0, bounds=bounds, maxfev=60000)
        r_fit = model_m1(tp, *popt)
        names = ["a", "p", "c"]
    else:
        p0 = [3.0, 1.2, 1.0, 1.0]
        bounds = ([0.0, 0.1, 1e-6, 0.2], [1e3, 8.0, 1e3, 8.0])
        popt, _ = curve_fit(model_m2, tp, r, p0=p0, bounds=bounds, maxfev=80000)
        r_fit = model_m2(tp, *popt)
        names = ["a", "p", "c", "q"]
    return popt, r_fit, names


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (Path.cwd() / data_path).resolve()

    arr = np.loadtxt(data_path, comments="#")
    # cols: H_star t_star theta0 t_p x Ea3_PT Ea3_noPT f_anh_PT f_anh_noPT xi nsteps_PT nsteps_noPT
    theta = arr[:, 2]
    tp = arr[:, 3]
    f_pt = arr[:, 7]
    f_no = arr[:, 8]

    th_unique = np.unique(theta)
    rows = []
    fit_rows = []

    for th0 in th_unique:
        m = (theta == th0) & np.isfinite(tp) & np.isfinite(f_pt) & np.isfinite(f_no) & (f_no > 0) & (f_pt > 0)
        x = tp[m]
        y = f_pt[m]
        base = f_no[m]
        if len(x) < 8:
            continue

        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]
        base = base[idx]
        r = y / base

        popt, r_fit, pnames = fit_one_theta(x, r, args.model)
        y_fit = base * r_fit
        log_rmse = float(np.sqrt(np.mean((np.log(y_fit) - np.log(y)) ** 2)))
        rel_rmse = float(np.sqrt(np.mean(((y_fit - y) / y) ** 2)))
        max_abs_rel = float(np.max(np.abs((y_fit - y) / y)))

        param_map = {k: v for k, v in zip(pnames, popt)}
        rows.append(
            {
                "theta0": float(th0),
                "n": int(len(x)),
                "log_rmse": log_rmse,
                "rel_rmse": rel_rmse,
                "max_abs_rel": max_abs_rel,
                **param_map,
            }
        )
        fit_rows.append((th0, x, y, y_fit))

    tag = data_path.stem.replace("dm_tp_fitready_", "")
    out_table = outdir / f"fanh_transition_params_{args.model}_{tag}.txt"
    with open(out_table, "w") as f:
        if args.model == "m1":
            f.write("# theta0 n a p c log_rmse rel_rmse max_abs_rel\n")
            for r in rows:
                f.write(
                    f"{r['theta0']:.8g} {r['n']} {r['a']:.10e} {r['p']:.10e} {r['c']:.10e} "
                    f"{r['log_rmse']:.10e} {r['rel_rmse']:.10e} {r['max_abs_rel']:.10e}\n"
                )
        else:
            f.write("# theta0 n a p c q log_rmse rel_rmse max_abs_rel\n")
            for r in rows:
                f.write(
                    f"{r['theta0']:.8g} {r['n']} {r['a']:.10e} {r['p']:.10e} {r['c']:.10e} {r['q']:.10e} "
                    f"{r['log_rmse']:.10e} {r['rel_rmse']:.10e} {r['max_abs_rel']:.10e}\n"
                )

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    colors = plt.cm.plasma(np.linspace(0.08, 0.95, len(fit_rows)))
    for c, (th0, x, y, y_fit) in zip(colors, fit_rows):
        ax.plot(x, y, "o", ms=3, color=c, alpha=0.6)
        ax.plot(x, y_fit, "-", lw=1.8, color=c, label=rf"$\theta_0={th0:.3g}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t_p$")
    ax.set_ylabel(r"$f_{\rm anh}$")
    ax.set_title(rf"Transition fit ({args.model}) on $f_{{\rm anh}}(\theta_0,t_p)$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    out_overlay = outdir / f"fanh_transition_overlay_{args.model}_{tag}.png"
    fig.savefig(out_overlay, dpi=240)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8.2, 5.0))
    for c, (th0, x, y, y_fit) in zip(colors, fit_rows):
        rel = (y_fit - y) / y
        ax2.plot(x, rel, "-o", lw=1.5, ms=3, color=c, label=rf"$\theta_0={th0:.3g}$")
    ax2.axhline(0.0, color="k", ls="--", lw=1, alpha=0.6)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$t_p$")
    ax2.set_ylabel(r"relative residual $(f_{\rm fit}-f)/f$")
    ax2.set_title("Fit residuals")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False, ncol=2, fontsize=8)
    fig2.tight_layout()
    out_res = outdir / f"fanh_transition_residuals_{args.model}_{tag}.png"
    fig2.savefig(out_res, dpi=240)
    plt.close(fig2)

    mean_log_rmse = np.mean([r["log_rmse"] for r in rows]) if rows else np.nan
    worst_rel = np.max([r["max_abs_rel"] for r in rows]) if rows else np.nan
    print(f"Saved params: {out_table}")
    print(f"Saved overlay: {out_overlay}")
    print(f"Saved residuals: {out_res}")
    print(f"mean_log_rmse={mean_log_rmse:.6g}  worst_max_abs_rel={worst_rel:.6g}")


if __name__ == "__main__":
    main()
