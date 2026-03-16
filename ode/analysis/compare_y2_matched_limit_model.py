import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares

from refine_y2_physical_models import MODELS, features, fit_model, log_rmse, rel_rmse


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compare asymptotic-matched Y2 models against the current compact "
            "plateau-plus-powerlaw fit."
        )
    )
    p.add_argument("--data", type=str, default="ode/analysis/data/dm_tp_fitready_H1p000.txt")
    p.add_argument("--outdir", type=str, default="ode/analysis/results/old_attempts")
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def y2_powerlaw(tp, c0, c1, p):
    tp = np.asarray(tp, dtype=np.float64)
    return c0 + c1 / np.power(tp, p)


def y2_matched(tp, yinf, q, fanh_no):
    tp = np.asarray(tp, dtype=np.float64)
    tc = np.power(fanh_no / yinf, 2.0 / 3.0)
    return yinf * np.power(1.0 + np.power(tc / tp, q), 1.5 / q)


def fanh_no_fit(theta0):
    _, _, h = features(np.asarray(theta0, dtype=np.float64))
    return 0.2660539485 + 0.1393009266 * np.power(h, 1.7576588657)


def fit_powerlaw_slices(theta0, tp, y2):
    rows = []
    yfit_all = np.full_like(y2, np.nan, dtype=np.float64)

    for th0 in np.unique(theta0):
        mask = np.isclose(theta0, th0) & np.isfinite(tp) & np.isfinite(y2) & (tp > 0.0) & (y2 > 0.0)
        x = tp[mask]
        y = y2[mask]
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

        p0 = [y[-1], max(y[0] - y[-1], 1e-6), 1.0]
        bounds = ([0.0, 0.0, 0.05], [10.0, 1e4, 8.0])
        popt, _ = curve_fit(y2_powerlaw, x, y, p0=p0, bounds=bounds, maxfev=200000)
        yfit = y2_powerlaw(x, *popt)
        yfit_all[np.where(mask)[0][idx]] = yfit
        rows.append((th0, popt[0], popt[1], popt[2], log_rmse(y, yfit), rel_rmse(y, yfit)))

    return np.array(rows, dtype=np.float64), yfit_all


def fit_matched_slices(theta0, tp, y2, fanh_no):
    rows = []
    yfit_all = np.full_like(y2, np.nan, dtype=np.float64)

    for th0 in np.unique(theta0):
        mask = np.isclose(theta0, th0) & np.isfinite(tp) & np.isfinite(y2) & (tp > 0.0) & (y2 > 0.0)
        x = tp[mask]
        y = y2[mask]
        fno = float(fanh_no[mask][0])
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

        p0 = [max(y[-1], 1e-8), 1.5]
        bounds = ([1e-12, 0.05], [10.0, 8.0])
        popt, _ = curve_fit(
            lambda tpi, yinf, q: y2_matched(tpi, yinf, q, fno),
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=200000,
        )
        yfit = y2_matched(x, *popt, fno)
        yfit_all[np.where(mask)[0][idx]] = yfit
        tc = np.power(fno / popt[0], 2.0 / 3.0)
        rows.append((th0, fno, popt[0], popt[1], tc, log_rmse(y, yfit), rel_rmse(y, yfit)))

    return np.array(rows, dtype=np.float64), yfit_all


def hpow(theta0, amp, alpha):
    _, _, h = features(np.asarray(theta0, dtype=np.float64))
    return amp * np.power(h, alpha)


def q_lin_yinf(theta0, yinf, q0, q1):
    return q0 + q1 * yinf


def q_lin_h(theta0, yinf, q0, q1):
    _, _, h = features(np.asarray(theta0, dtype=np.float64))
    return q0 + q1 * h


def q_quad_h(theta0, yinf, q0, q1, q2):
    _, _, h = features(np.asarray(theta0, dtype=np.float64))
    return q0 + q1 * h + q2 * h * h


MATCHED_MODELS = {
    "matched_powh_linYinf": {
        "q_fn": q_lin_yinf,
        "q_p0": lambda theta_u, yinf_u, h_u: np.array([2.7048990951, -1.8121025642], dtype=np.float64),
        "bounds": (
            np.array([1e-6, 0.0, 0.05, -10.0], dtype=np.float64),
            np.array([10.0, 3.0, 6.0, 4.0], dtype=np.float64),
        ),
    },
    "matched_powh_linH": {
        "q_fn": q_lin_h,
        "q_p0": lambda theta_u, yinf_u, h_u: np.array([1.80, -0.21], dtype=np.float64),
        "bounds": (
            np.array([1e-6, 0.0, 0.05, -4.0], dtype=np.float64),
            np.array([10.0, 3.0, 6.0, 2.0], dtype=np.float64),
        ),
    },
    "matched_powh_quadH": {
        "q_fn": q_quad_h,
        "q_p0": lambda theta_u, yinf_u, h_u: np.array([2.15, -0.55, 0.05], dtype=np.float64),
        "bounds": (
            np.array([1e-6, 0.0, 0.05, -6.0, -2.0], dtype=np.float64),
            np.array([10.0, 3.0, 6.0, 6.0, 2.0], dtype=np.float64),
        ),
    },
}


def matched_surface(theta0, tp, fanh_no, params, spec):
    amp = params[0]
    alpha = params[1]
    yinf = hpow(theta0, amp, alpha)
    q = spec["q_fn"](theta0, yinf, *params[2:])
    if np.any(~np.isfinite(yinf)) or np.any(~np.isfinite(q)) or np.any(yinf <= 0.0) or np.any(q <= 0.05):
        return np.full_like(tp, np.nan, dtype=np.float64)
    return y2_matched(tp, yinf, q, fanh_no)


def matched_residuals(params, theta0, tp, fanh_no, y, spec):
    yfit = matched_surface(theta0, tp, fanh_no, params, spec)
    if np.any(~np.isfinite(yfit)) or np.any(yfit <= 0.0):
        return np.full_like(y, 1e6, dtype=np.float64)
    return np.log(yfit) - np.log(y)


def initial_matched_params(theta_rows, spec):
    theta_u = theta_rows[:, 0]
    yinf_u = theta_rows[:, 2]
    _, _, h_u = features(theta_u)
    popt_yinf, _ = curve_fit(
        hpow,
        theta_u,
        yinf_u,
        p0=[max(float(np.median(yinf_u)), 1e-8), 0.2],
        bounds=([1e-8, 0.0], [10.0, 3.0]),
        maxfev=50000,
    )
    q_p0 = spec["q_p0"](theta_u, yinf_u, h_u)
    return np.concatenate([popt_yinf, q_p0])


def fit_matched_model(theta0, tp, fanh_no, y2, name, spec, theta_rows):
    p0 = initial_matched_params(theta_rows, spec)
    lower, upper = spec["bounds"]
    res = least_squares(
        matched_residuals,
        p0,
        bounds=(lower, upper),
        args=(theta0, tp, fanh_no, y2, spec),
        max_nfev=50000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    yfit = matched_surface(theta0, tp, fanh_no, res.x, spec)
    return {
        "name": name,
        "params": res.x,
        "success": bool(res.success),
        "message": str(res.message),
        "global_log_rmse": log_rmse(y2, yfit),
        "global_rel_rmse": rel_rmse(y2, yfit),
        "yfit": yfit,
    }


def make_slice_plot(theta0, tp, y2, powerlaw_fit, matched_fit, out_path, dpi):
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.2), squeeze=False)
    axes = axes.ravel()

    for ax, th0 in zip(axes, np.unique(theta0)):
        mask = np.isclose(theta0, th0)
        idx = np.argsort(tp[mask])
        x = tp[mask][idx]
        y = y2[mask][idx]
        yp = powerlaw_fit[mask][idx]
        ym = matched_fit[mask][idx]

        ax.plot(x, y, "ko", ms=4, label="data")
        ax.plot(x, yp, "-", lw=1.8, color="tab:blue", label="powerlaw slice")
        ax.plot(x, ym, "--", lw=1.8, color="tab:red", label="matched slice")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(rf"$\theta_0={th0:.3g}$")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$Y_2$")
        ax.grid(alpha=0.25)

    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_global_plot(theta0, tp, y2, compact, matched, out_path, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(np.unique(theta0))))
    panels = [
        (axes[0], compact["yfit"], f"{compact['name']}, rel={compact['global_rel_rmse']:.3e}"),
        (axes[1], matched["yfit"], f"{matched['name']}, rel={matched['global_rel_rmse']:.3e}"),
    ]

    for ax, yfit, title in panels:
        for color, th0 in zip(colors, np.unique(theta0)):
            mask = np.isclose(theta0, th0)
            idx = np.argsort(tp[mask])
            x = tp[mask][idx]
            yd = y2[mask][idx]
            yf = yfit[mask][idx]
            ax.plot(x, yd, "o", ms=3, color=color, alpha=0.75)
            ax.plot(x, yf, "-", lw=1.7, color=color, label=rf"$\theta_0={th0:.3g}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$t_p$")
        ax.set_title(title)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(r"$Y_2$")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_summary(path, powerlaw_rows, matched_rows, compact, matched_results, analytic_ref_rel):
    with open(path, "w") as f:
        f.write("# Per-theta slice comparison: unconstrained powerlaw vs asymptotic-matched form\n")
        f.write("# theta0 f_no yinf_matched q_matched tc_matched matched_rel powerlaw_rel\n")
        for prow, mrow in zip(powerlaw_rows, matched_rows):
            f.write(
                f"{mrow[0]:.8g} {mrow[1]:.10e} {mrow[2]:.10e} {mrow[3]:.10e} "
                f"{mrow[4]:.10e} {mrow[6]:.10e} {prow[5]:.10e}\n"
            )

        f.write("\n# Global compact-model comparison\n")
        f.write("# name global_log_rmse global_rel_rmse success\n")
        f.write(
            f"{compact['name']} {compact['global_log_rmse']:.10e} {compact['global_rel_rmse']:.10e} "
            f"{int(compact['success'])}\n"
        )
        for rec in matched_results:
            f.write(
                f"{rec['name']} {rec['global_log_rmse']:.10e} {rec['global_rel_rmse']:.10e} "
                f"{int(rec['success'])}\n"
            )
            f.write("# params " + " ".join(f"{x:.10e}" for x in rec["params"]) + "\n")
            f.write(f"# message {rec['message']}\n")
        f.write(
            f"# best_matched_with_analytic_fanh_noPT_rel_rmse {analytic_ref_rel:.10e}\n"
        )


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
    y2 = xi * fanh_no / np.power(tp, 1.5)

    powerlaw_rows, powerlaw_fit = fit_powerlaw_slices(theta0, tp, y2)
    matched_rows, matched_fit = fit_matched_slices(theta0, tp, y2, fanh_no)

    compact_results = [fit_model(theta0, tp, y2, name, spec) for name, spec in MODELS.items()]
    compact_results.sort(key=lambda r: (r["global_rel_rmse"], r["global_log_rmse"]))
    compact = compact_results[0]

    matched_results = [
        fit_matched_model(theta0, tp, fanh_no, y2, name, spec, matched_rows)
        for name, spec in MATCHED_MODELS.items()
    ]
    matched_results.sort(key=lambda r: (r["global_rel_rmse"], r["global_log_rmse"]))
    matched_best = matched_results[0]
    analytic_ref_fit = matched_surface(
        theta0,
        tp,
        fanh_no_fit(theta0),
        matched_best["params"],
        MATCHED_MODELS[matched_best["name"]],
    )
    analytic_ref_rel = rel_rmse(y2, analytic_ref_fit)

    stem = data_path.stem
    txt_out = outdir / f"compare_y2_matched_limit_model_{stem}.txt"
    slice_out = outdir / f"compare_y2_matched_limit_slices_{stem}.png"
    global_out = outdir / f"compare_y2_matched_limit_global_{stem}.png"

    save_summary(txt_out, powerlaw_rows, matched_rows, compact, matched_results, analytic_ref_rel)
    make_slice_plot(theta0, tp, y2, powerlaw_fit, matched_fit, slice_out, args.dpi)
    make_global_plot(theta0, tp, y2, compact, matched_best, global_out, args.dpi)

    print(f"Loaded: {data_path}")
    print(f"Saved: {txt_out}")
    print(f"Saved: {slice_out}")
    print(f"Saved: {global_out}")
    print(
        "Slice rel RMSE: powerlaw={:.4e}, matched={:.4e}".format(
            rel_rmse(y2, powerlaw_fit),
            rel_rmse(y2, matched_fit),
        )
    )
    print(
        "Best compact global: {} rel={:.4e}".format(
            compact["name"],
            compact["global_rel_rmse"],
        )
    )
    print(
        "Best matched global: {} rel={:.4e}".format(
            matched_best["name"],
            matched_best["global_rel_rmse"],
        )
    )
    print("Best matched global with analytic noPT fit rel={:.4e}".format(analytic_ref_rel))


if __name__ == "__main__":
    main()
