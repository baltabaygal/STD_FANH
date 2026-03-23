import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_TABLES = [
    "ode/analysis/data/dm_tp_fitready_H0p500.txt",
    "ode/analysis/data/dm_tp_fitready_H1p000.txt",
    "ode/analysis/data/dm_tp_fitready_H1p500.txt",
    "ode/analysis/data/dm_tp_fitready_H2p000.txt",
]

DEFAULT_THETA = [0.2618, 1.309, 2.88]
DEFAULT_TP = [1.0, 2.0, 5.0]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run H_* dependence diagnostics on direct ODE fit-ready tables: "
            "fixed-t_p Y3(H_*), xi-collapse scans, and a global transient H^gamma profile."
        )
    )
    p.add_argument("--tables", type=str, nargs="+", default=DEFAULT_TABLES)
    p.add_argument("--theta-list", type=float, nargs="+", default=DEFAULT_THETA)
    p.add_argument("--tp-list", type=float, nargs="+", default=DEFAULT_TP)
    p.add_argument("--collapse-beta-min", type=float, default=-1.0)
    p.add_argument("--collapse-beta-max", type=float, default=2.0)
    p.add_argument("--collapse-beta-step", type=float, default=0.02)
    p.add_argument("--gamma-min", type=float, default=-2.0)
    p.add_argument("--gamma-max", type=float, default=2.0)
    p.add_argument("--gamma-step", type=float, default=0.05)
    p.add_argument("--s-bounds", type=float, nargs=2, default=[0.2, 3.0])
    p.add_argument("--r-bounds", type=float, nargs=2, default=[0.5, 5.0])
    p.add_argument(
        "--outdir",
        type=str,
        default="ode/analysis/results/hstar_dependence_tests",
    )
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def nearest_theta(values, theta0, atol=5.0e-4):
    values = np.asarray(values, dtype=np.float64)
    idx = int(np.argmin(np.abs(values - float(theta0))))
    if abs(values[idx] - float(theta0)) > atol:
        raise RuntimeError(f"No theta match for theta0={theta0:.10f}")
    return idx


def rel_rmse(y, yfit):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square((yfit - y) / y))))


def load_fitready_table(path):
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    return {
        "hstar": arr[:, 0].astype(np.float64),
        "t_star": arr[:, 1].astype(np.float64),
        "theta0": arr[:, 2].astype(np.float64),
        "tp": arr[:, 3].astype(np.float64),
        "xi": arr[:, 9].astype(np.float64),
        "fanh_no": arr[:, 8].astype(np.float64),
    }


def build_dataset(path):
    data = load_fitready_table(path)
    y3 = data["xi"] * np.square(data["fanh_no"]) / np.power(data["tp"], 1.5)
    data["y3"] = y3
    return data


def filter_theta_list(data_h, theta_list):
    keep = np.zeros(len(data_h["theta0"]), dtype=bool)
    for theta in theta_list:
        keep |= np.isclose(data_h["theta0"], float(theta), rtol=0.0, atol=5.0e-4)
    return {key: np.asarray(val[keep], dtype=np.float64) for key, val in data_h.items()}


def unique_theta(data_h):
    return np.array(sorted(np.unique(data_h["theta0"])), dtype=np.float64)


def build_theta_arrays(data_h):
    theta_u = unique_theta(data_h)
    theta_index = np.array([nearest_theta(theta_u, th) for th in data_h["theta0"]], dtype=np.int64)
    f_theta = np.array(
        [float(np.median(data_h["fanh_no"][theta_index == i])) for i in range(len(theta_u))],
        dtype=np.float64,
    )
    return theta_u, theta_index, f_theta


def solve_finf_per_theta_hgroup(xi, tp, hstar, theta_index, f_theta, transient, lower=1.0e-6, upper=1.0e4):
    a = np.power(tp, 1.5) / np.square(f_theta[theta_index])
    ngroup = (int(np.max(hstar)) + 1) * 1000  # unused placeholder for deterministic order
    keys = []
    finf = {}
    unique_h = np.array(sorted(np.unique(hstar)), dtype=np.float64)
    for hv in unique_h:
        for i in range(len(f_theta)):
            keys.append((float(hv), int(i)))
    for hv, i in keys:
        mask = np.isclose(hstar, hv, rtol=0.0, atol=1.0e-12) & (theta_index == i)
        u = a[mask] / xi[mask]
        v = 1.0 - transient[mask] / xi[mask]
        denom = float(np.dot(u, u))
        finf_i = lower if denom <= 0.0 else float(np.dot(u, v) / denom)
        finf[(hv, i)] = float(np.clip(finf_i, lower, upper))
    return finf


def xi_model_hgroup(tp, hstar, theta_index, f_theta, finf, transient):
    plateau = np.zeros_like(tp, dtype=np.float64)
    a = np.power(tp, 1.5) / np.square(f_theta[theta_index])
    for idx in range(len(tp)):
        plateau[idx] = a[idx] * finf[(float(hstar[idx]), int(theta_index[idx]))]
    return plateau + transient


def transient_tc_tstar(tp, hstar, s, r, gamma):
    t_star = 1.0 / (2.0 * hstar)
    return np.power(hstar, gamma) / (1.0 + np.power((s * tp) / t_star, r))


def score_global_gamma(data_all, theta_index, f_theta, s, r, gamma):
    tp = data_all["tp"]
    hstar = data_all["hstar"]
    xi = data_all["xi"]
    transient = transient_tc_tstar(tp, hstar, float(s), float(r), float(gamma))
    finf = solve_finf_per_theta_hgroup(xi, tp, hstar, theta_index, f_theta, transient)
    xi_fit = xi_model_hgroup(tp, hstar, theta_index, f_theta, finf, transient)
    resid = (xi_fit - xi) / xi
    return {
        "s": float(s),
        "r": float(r),
        "gamma": float(gamma),
        "F_inf": finf,
        "xi_fit": xi_fit,
        "rel_resid": resid,
        "chi2_rel": float(np.sum(np.square(resid))),
        "rel_rmse": rel_rmse(xi, xi_fit),
    }


def fit_sr_for_gamma(data_all, theta_index, f_theta, gamma, s_bounds, r_bounds):
    def objective(x):
        s, r = x
        return score_global_gamma(data_all, theta_index, f_theta, s, r, gamma)["chi2_rel"]

    starts = [
        np.array([0.7, 2.2], dtype=np.float64),
        np.array([1.0, 2.2], dtype=np.float64),
        np.array([1.2, 3.0], dtype=np.float64),
        np.array([0.5, 1.5], dtype=np.float64),
    ]
    bounds = [tuple(map(float, s_bounds)), tuple(map(float, r_bounds))]
    best = None
    for x0 in starts:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        cand = score_global_gamma(data_all, theta_index, f_theta, float(res.x[0]), float(res.x[1]), gamma)
        cand["success"] = bool(res.success)
        if best is None or cand["chi2_rel"] < best["chi2_rel"]:
            best = cand
    return best


def fit_sr_per_h(data_h, s_bounds, r_bounds):
    theta_u, theta_index, f_theta = build_theta_arrays(data_h)

    def solve_finf_per_theta(xi, tp, theta_index, f_theta, transient, lower=1.0e-6, upper=1.0e4):
        a = np.power(tp, 1.5) / np.square(f_theta[theta_index])
        finf = np.zeros(len(f_theta), dtype=np.float64)
        for i in range(len(f_theta)):
            mask = theta_index == i
            u = a[mask] / xi[mask]
            v = 1.0 - transient[mask] / xi[mask]
            denom = float(np.dot(u, u))
            finf_i = lower if denom <= 0.0 else float(np.dot(u, v) / denom)
            finf[i] = np.clip(finf_i, lower, upper)
        return finf

    def score(s, r):
        t_star = np.median(data_h["t_star"])
        transient = 1.0 / (1.0 + np.power((float(s) * data_h["tp"]) / float(t_star), float(r)))
        finf = solve_finf_per_theta(data_h["xi"], data_h["tp"], theta_index, f_theta, transient)
        xi_fit = np.power(data_h["tp"], 1.5) * finf[theta_index] / np.square(f_theta[theta_index]) + transient
        resid = (xi_fit - data_h["xi"]) / data_h["xi"]
        return {
            "s": float(s),
            "r": float(r),
            "F_inf": finf,
            "xi_fit": xi_fit,
            "rel_resid": resid,
            "chi2_rel": float(np.sum(np.square(resid))),
            "rel_rmse": rel_rmse(data_h["xi"], xi_fit),
            "theta_u": theta_u,
            "theta_index": theta_index,
            "f_theta": f_theta,
        }

    def objective(x):
        return score(float(x[0]), float(x[1]))["chi2_rel"]

    starts = [
        np.array([0.7, 2.2], dtype=np.float64),
        np.array([1.0, 2.2], dtype=np.float64),
        np.array([1.2, 3.0], dtype=np.float64),
        np.array([0.5, 1.5], dtype=np.float64),
    ]
    bounds = [tuple(map(float, s_bounds)), tuple(map(float, r_bounds))]
    best = None
    for x0 in starts:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        cand = score(float(res.x[0]), float(res.x[1]))
        cand["success"] = bool(res.success)
        if best is None or cand["chi2_rel"] < best["chi2_rel"]:
            best = cand
    return best


def interp_logx(x, y, x0):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    return float(np.interp(np.log(x0), np.log(x), y))


def powerlaw_fit_loglog(x, y):
    lx = np.log(np.asarray(x, dtype=np.float64))
    ly = np.log(np.asarray(y, dtype=np.float64))
    coeff, cov = np.polyfit(lx, ly, deg=1, cov=True)
    gamma = float(coeff[0])
    intercept = float(coeff[1])
    gamma_err = float(np.sqrt(max(cov[0, 0], 0.0)))
    return {
        "gamma": gamma,
        "gamma_err": gamma_err,
        "amp": float(np.exp(intercept)),
    }


def choose_theta(data_h, theta_target):
    theta_u = unique_theta(data_h)
    idx = nearest_theta(theta_u, theta_target)
    return float(theta_u[idx])


def run_test1_fixed_tp_powerlaw(datasets, theta_list, tp_list):
    results = []
    for theta_target in theta_list:
        for tp0 in tp_list:
            hvals = []
            yvals = []
            for hstar in sorted(datasets):
                data_h = datasets[hstar]
                theta = choose_theta(data_h, theta_target)
                mask = np.isclose(data_h["theta0"], theta, rtol=0.0, atol=5.0e-4)
                tp = data_h["tp"][mask]
                if tp0 < float(np.min(tp)) or tp0 > float(np.max(tp)):
                    continue
                y3 = data_h["y3"][mask]
                hvals.append(float(hstar))
                yvals.append(interp_logx(tp, y3, tp0))
            if len(hvals) >= 3:
                fit = powerlaw_fit_loglog(hvals, yvals)
                results.append(
                    {
                        "theta0": float(theta_target),
                        "tp": float(tp0),
                        "hvals": np.array(hvals, dtype=np.float64),
                        "yvals": np.array(yvals, dtype=np.float64),
                        **fit,
                    }
                )
    return results


def collapse_score(datasets, beta, ykey="xi"):
    theta_all = unique_theta(next(iter(datasets.values())))
    scores = []
    for theta in theta_all:
        curves = []
        xmin = -np.inf
        xmax = np.inf
        for hstar, data_h in datasets.items():
            mask = np.isclose(data_h["theta0"], theta, rtol=0.0, atol=5.0e-4)
            x = data_h["tp"][mask] * np.power(float(hstar), float(beta))
            y = data_h[ykey][mask]
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            curves.append((x, y))
            xmin = max(xmin, float(np.min(x)))
            xmax = min(xmax, float(np.max(x)))
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            continue
        x_common = np.geomspace(xmin, xmax, 80)
        y_stack = np.array([np.interp(np.log(x_common), np.log(x), y) for x, y in curves], dtype=np.float64)
        ref = np.mean(y_stack, axis=0)
        score = np.sqrt(np.mean(np.square((y_stack - ref[None, :]) / ref[None, :])))
        scores.append(float(score))
    if not scores:
        return np.inf
    return float(np.mean(scores))


def make_test1_plot(results, out_path, dpi):
    theta_u = sorted({rec["theta0"] for rec in results})
    fig, axes = plt.subplots(1, len(theta_u), figsize=(4.8 * len(theta_u), 4.2), squeeze=False)
    colors = {1.0: "tab:green", 2.0: "tab:orange", 5.0: "tab:red"}
    for ax, theta in zip(axes[0], theta_u):
        subset = [rec for rec in results if np.isclose(rec["theta0"], theta, atol=5.0e-4)]
        for rec in subset:
            color = colors.get(round(rec["tp"], 2), None)
            label = rf"$t_p={rec['tp']:.2g}$, $\gamma={rec['gamma']:.2f}\pm{rec['gamma_err']:.2f}$"
            ax.plot(rec["hvals"], rec["yvals"], "o", ms=5.0, color=color, label=label)
            hfit = np.geomspace(np.min(rec["hvals"]), np.max(rec["hvals"]), 100)
            ax.plot(hfit, rec["amp"] * np.power(hfit, rec["gamma"]), color=color, lw=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$H_*$")
        ax.set_ylabel(r"$Y_3$")
        ax.set_title(rf"$\theta_0={theta:.3g}$")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_collapse_profile_plot(beta_grid, score_data, score_model, out_path, dpi):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(beta_grid, score_data, color="tab:blue", lw=2.0, label="ODE data")
    ax.plot(beta_grid, score_model, color="tab:red", lw=2.0, label="best-fit model")
    ib = int(np.argmin(score_model))
    ax.axvline(beta_grid[ib], color="tab:red", lw=1.0, ls="--", alpha=0.6)
    ax.set_xlabel(r"$\beta$ in $x=t_p H_*^\beta$")
    ax.set_ylabel("collapse score")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_collapse_overlay(datasets, beta, theta_list, out_path, dpi, ykey="xi"):
    fig, axes = plt.subplots(1, len(theta_list), figsize=(4.8 * len(theta_list), 4.2), squeeze=False)
    colors = {0.5: "tab:green", 1.0: "tab:orange", 1.5: "tab:blue", 2.0: "tab:red"}
    for ax, theta_target in zip(axes[0], theta_list):
        for hstar in sorted(datasets):
            data_h = datasets[hstar]
            theta = choose_theta(data_h, theta_target)
            mask = np.isclose(data_h["theta0"], theta, rtol=0.0, atol=5.0e-4)
            x = data_h["tp"][mask] * np.power(float(hstar), float(beta))
            y = data_h[ykey][mask]
            order = np.argsort(x)
            ax.plot(x[order], y[order], "o-", ms=3.2, lw=1.2, color=colors.get(float(hstar), None), label=rf"$H_*={float(hstar):g}$")
        ax.set_xscale("log")
        ax.set_xlabel(r"$t_p H_*^\beta$")
        ax.set_ylabel(r"$\xi$")
        ax.set_title(rf"$\theta_0={float(theta_target):.3g}$")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_gamma_profile_plot(rows, out_path, dpi):
    gamma = np.array([row["gamma"] for row in rows], dtype=np.float64)
    rel = np.array([row["rel_rmse"] for row in rows], dtype=np.float64)
    s = np.array([row["s"] for row in rows], dtype=np.float64)
    r = np.array([row["r"] for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 8.0), sharex=True)
    axes[0].plot(gamma, rel, color="black", lw=2.0)
    axes[0].set_ylabel("rel-RMSE")
    axes[0].grid(alpha=0.25)
    axes[1].plot(gamma, s, color="tab:blue", lw=1.8)
    axes[1].set_ylabel(r"best $s$")
    axes[1].grid(alpha=0.25)
    axes[2].plot(gamma, r, color="tab:red", lw=1.8)
    axes[2].set_ylabel(r"best $r$")
    axes[2].set_xlabel(r"transient exponent $\gamma$ in $H_*^\gamma$")
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_summary(path, datasets, test1, per_h_fits, beta_grid, score_data, score_model, gamma_rows, gamma_best):
    with open(path, "w") as f:
        f.write("# H_* dependence diagnostics on direct ODE fit-ready tables\n")
        f.write("# Model convention for per-H fits: xi = tp^(3/2) F_inf(theta,H)/F(theta)^2 + 1/[1 + (s tp / t_*)^r], with t_* = 1/(2H_*) and c=1.\n")
        f.write("# Global gamma profile tests the transient-only extension xi -> plateau + H_*^gamma / [1 + (s tp / t_*)^r].\n")
        f.write("# Plateau-only gamma is not identifiable if F_inf(theta,H) is left free per H and theta.\n\n")

        f.write("# Test 1: fixed-t_p power-law Y3(H_*) fits\n")
        f.write("# theta0 tp gamma gamma_err nH\n")
        for rec in test1:
            f.write(
                f"{rec['theta0']:.10f} {rec['tp']:.10e} {rec['gamma']:.10e} {rec['gamma_err']:.10e} {len(rec['hvals'])}\n"
            )
        f.write("\n")

        f.write("# Per-H best fits with tc=t_*=1/(2H_*), c=1\n")
        f.write("# H_star t_star s_best r_best rel_rmse\n")
        for hstar in sorted(per_h_fits):
            rec = per_h_fits[hstar]
            f.write(
                f"{float(hstar):.10e} {float(np.median(datasets[hstar]['t_star'])):.10e} {rec['s']:.10e} {rec['r']:.10e} {rec['rel_rmse']:.10e}\n"
            )
        f.write("\n")

        i_data = int(np.argmin(score_data))
        i_model = int(np.argmin(score_model))
        f.write("# Test 2: collapse scan with x = tp * H_*^beta\n")
        f.write(f"# best_beta_data={beta_grid[i_data]:.10e} collapse_score_data={score_data[i_data]:.10e}\n")
        f.write(f"# best_beta_model={beta_grid[i_model]:.10e} collapse_score_model={score_model[i_model]:.10e}\n\n")

        f.write("# Test 3: global transient-H^gamma profile\n")
        f.write("# gamma s_best r_best rel_rmse chi2_rel\n")
        for row in gamma_rows:
            f.write(
                f"{row['gamma']:.10e} {row['s']:.10e} {row['r']:.10e} {row['rel_rmse']:.10e} {row['chi2_rel']:.10e}\n"
            )
        f.write("\n")
        f.write(
            f"# gamma_best={gamma_best['gamma']:.10e} s_best={gamma_best['s']:.10e} "
            f"r_best={gamma_best['r']:.10e} rel_rmse={gamma_best['rel_rmse']:.10e}\n"
        )


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = {}
    for path in args.tables:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing required fit-ready table: {p}")
        data_h = build_dataset(p)
        data_h = filter_theta_list(data_h, args.theta_list)
        hstar = float(np.median(data_h["hstar"]))
        datasets[hstar] = data_h

    test1 = run_test1_fixed_tp_powerlaw(datasets, args.theta_list, args.tp_list)
    make_test1_plot(test1, outdir / "test1_y3_vs_hstar.png", args.dpi)

    per_h_fits = {}
    model_datasets = {}
    for hstar in sorted(datasets):
        fit = fit_sr_per_h(datasets[hstar], args.s_bounds, args.r_bounds)
        per_h_fits[hstar] = fit
        model_h = {key: np.array(val, copy=True) for key, val in datasets[hstar].items()}
        model_h["xi"] = fit["xi_fit"]
        model_datasets[hstar] = model_h

    beta_grid = np.arange(
        float(args.collapse_beta_min),
        float(args.collapse_beta_max) + 0.5 * float(args.collapse_beta_step),
        float(args.collapse_beta_step),
        dtype=np.float64,
    )
    score_data = np.array([collapse_score(datasets, beta, ykey="xi") for beta in beta_grid], dtype=np.float64)
    score_model = np.array([collapse_score(model_datasets, beta, ykey="xi") for beta in beta_grid], dtype=np.float64)
    make_collapse_profile_plot(beta_grid, score_data, score_model, outdir / "test2_collapse_profile.png", args.dpi)
    best_beta = float(beta_grid[int(np.argmin(score_model))])
    make_collapse_overlay(datasets, best_beta, args.theta_list, outdir / "test2_collapse_overlay_data.png", args.dpi, ykey="xi")
    make_collapse_overlay(model_datasets, best_beta, args.theta_list, outdir / "test2_collapse_overlay_model.png", args.dpi, ykey="xi")

    data_all = {key: np.concatenate([datasets[h][key] for h in sorted(datasets)], axis=0) for key in next(iter(datasets.values())).keys()}
    theta_u, theta_index, f_theta = build_theta_arrays(data_all)
    gamma_grid = np.arange(
        float(args.gamma_min),
        float(args.gamma_max) + 0.5 * float(args.gamma_step),
        float(args.gamma_step),
        dtype=np.float64,
    )
    gamma_rows = [fit_sr_for_gamma(data_all, theta_index, f_theta, float(gamma), args.s_bounds, args.r_bounds) for gamma in gamma_grid]
    gamma_best = min(gamma_rows, key=lambda row: row["rel_rmse"])
    make_gamma_profile_plot(gamma_rows, outdir / "test3_gamma_profile.png", args.dpi)

    save_summary(
        outdir / "hstar_dependence_summary.txt",
        datasets,
        test1,
        per_h_fits,
        beta_grid,
        score_data,
        score_model,
        gamma_rows,
        gamma_best,
    )
    print(outdir / "hstar_dependence_summary.txt")


if __name__ == "__main__":
    main()
