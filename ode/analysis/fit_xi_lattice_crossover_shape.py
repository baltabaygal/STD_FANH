import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ode.hom_ODE.percolation import PercolationCache


TARGET_H_DEFAULT = [1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compare lattice-only xi fits for the baseline asymptotic model and an "
            "asymptotics-preserving crossover model, with fixed v_w."
        )
    )
    p.add_argument("--lattice-data", type=str, default="lattice_data/data/energy_ratio_by_theta_data_v9.txt")
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--target-h", type=float, nargs="+", default=TARGET_H_DEFAULT)
    p.add_argument("--nopt-feature", type=str, default="pendulum_log", choices=["pendulum_log", "hilltop_log1m"])
    p.add_argument("--fixed-nopt-amp", type=float, default=None)
    p.add_argument("--fixed-nopt-alpha", type=float, default=None)
    p.add_argument("--outdir", type=str, default="ode/analysis/results/lattice_fit/crossover_shape_v9")
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def pendulum_log_of_theta(theta0):
    theta0 = np.asarray(theta0, dtype=np.float64)
    return np.log(np.e / np.clip(np.cos(theta0 / 2.0) ** 2, 1.0e-12, None))


def hilltop_log_of_theta(theta0):
    theta0 = np.asarray(theta0, dtype=np.float64)
    u = theta0 / np.pi
    return np.log(np.e / np.clip(1.0 - u * u, 1.0e-12, None))


def nopt_feature(theta0, family):
    if family == "pendulum_log":
        return pendulum_log_of_theta(theta0)
    if family == "hilltop_log1m":
        return hilltop_log_of_theta(theta0)
    raise ValueError(family)


def rel_rmse(y, yfit):
    return float(np.sqrt(np.mean(np.square((yfit - y) / y))))


def log_rmse(y, yfit):
    return float(np.sqrt(np.mean(np.square(np.log(yfit) - np.log(y)))))


def load_lattice(path, target_h):
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    mask = np.zeros(arr.shape[0], dtype=bool)
    for hstar in target_h:
        mask |= np.isclose(arr[:, 1], float(hstar), rtol=0.0, atol=1.0e-12)
    arr = arr[mask]
    sem = arr[:, 5] / np.sqrt(np.maximum(arr[:, 6], 1.0))
    sem = np.maximum(sem, 1.0e-12)
    return {
        "theta0": arr[:, 0].astype(np.float64),
        "hstar": arr[:, 1].astype(np.float64),
        "beta_over_h": arr[:, 2].astype(np.float64),
        "mean_ratio": arr[:, 4].astype(np.float64),
        "sem": sem.astype(np.float64),
    }


def select_h(data, hstar):
    mask = np.isclose(data["hstar"], float(hstar), rtol=0.0, atol=1.0e-12)
    return {key: val[mask] for key, val in data.items()}


def attach_tp(data, vw):
    perc = PercolationCache()
    tp = np.array(
        [perc.get(float(h), float(bh), float(vw)) for h, bh in zip(data["hstar"], data["beta_over_h"])],
        dtype=np.float64,
    )
    out = dict(data)
    out["tp"] = tp
    return out


def unpack_params(x, model_name, q_mode, fixed_nopt):
    pos = 0
    if fixed_nopt is None:
        af = float(np.exp(x[pos]))
        gf = float(x[pos + 1])
        pos += 2
    else:
        af = float(fixed_nopt["amp"])
        gf = float(fixed_nopt["alpha"])
    ainf = float(np.exp(x[pos]))
    ginf = float(x[pos + 1])
    c = float(np.exp(x[pos + 2]))
    pos += 3
    if q_mode == "free":
        q = float(x[pos])
        pos += 1
    else:
        q = 1.5
    if model_name == "crossover":
        tc = float(np.exp(x[pos]))
        r = float(np.exp(x[pos + 1]))
    else:
        tc = None
        r = None
    return {
        "af": af,
        "gf": gf,
        "ainf": ainf,
        "ginf": ginf,
        "c": c,
        "q": q,
        "tc": tc,
        "r": r,
        "pos": pos,
    }


def xi_model(theta0, tp, params, model_name, nopt_feature_name):
    h = nopt_feature(theta0, nopt_feature_name)
    fanh_no = params["af"] * np.power(h, params["gf"])
    fanh_inf = params["ainf"] * np.power(h, params["ginf"])
    transient = params["c"] * np.square(fanh_no) / np.power(tp, params["q"])
    if model_name == "crossover":
        transient = transient / (1.0 + np.power(tp / params["tc"], params["r"]))
    y3 = fanh_inf + transient
    return np.power(tp, 1.5) / np.square(fanh_no) * y3


def build_x0_and_bounds(model_name, q_mode, fixed_nopt):
    log_ainf = np.log(0.15)
    log_c = np.log(1.0)
    if fixed_nopt is None:
        log_af = np.log(0.36)
        lower = [np.log(1.0e-6), 0.0, np.log(1.0e-8), 0.0, np.log(1.0e-8)]
        upper = [np.log(10.0), 4.0, np.log(10.0), 4.0, np.log(10.0)]
    else:
        lower = [np.log(1.0e-8), 0.0, np.log(1.0e-8)]
        upper = [np.log(10.0), 4.0, np.log(10.0)]

    starts = []
    if q_mode == "fixed":
        if fixed_nopt is None:
            base = [log_af, 1.17, log_ainf, 1.60, log_c]
            alt = [log_af, 1.00, log_ainf, 1.35, np.log(0.8)]
        else:
            base = [log_ainf, 1.60, log_c]
            alt = [log_ainf, 1.35, np.log(0.8)]
        if model_name == "crossover":
            starts.append(np.array(base + [np.log(0.7), np.log(1.5)], dtype=np.float64))
            starts.append(np.array(alt + [np.log(0.5), np.log(2.0)], dtype=np.float64))
            if fixed_nopt is None:
                starts.append(np.array([np.log(0.25), 1.1, np.log(0.08), 1.8, np.log(1.2), np.log(1.0), np.log(1.0)], dtype=np.float64))
            else:
                starts.append(np.array([np.log(0.08), 1.8, np.log(1.2), np.log(1.0), np.log(1.0)], dtype=np.float64))
            lower += [np.log(1.0e-3), np.log(5.0e-2)]
            upper += [np.log(20.0), np.log(10.0)]
        else:
            starts.append(np.array(base, dtype=np.float64))
            starts.append(np.array(alt, dtype=np.float64))
    else:
        if fixed_nopt is None:
            base = [log_af, 1.17, log_ainf, 1.60, log_c, 1.5]
            alt = [log_af, 1.00, log_ainf, 1.35, np.log(0.8), 1.7]
        else:
            base = [log_ainf, 1.60, log_c, 1.5]
            alt = [log_ainf, 1.35, np.log(0.8), 1.7]
        if model_name == "crossover":
            starts.append(np.array(base + [np.log(0.7), np.log(1.5)], dtype=np.float64))
            starts.append(np.array(alt + [np.log(0.5), np.log(2.0)], dtype=np.float64))
            if fixed_nopt is None:
                starts.append(np.array([np.log(0.25), 1.1, np.log(0.08), 1.8, np.log(1.2), 1.8, np.log(1.0), np.log(1.0)], dtype=np.float64))
            else:
                starts.append(np.array([np.log(0.08), 1.8, np.log(1.2), 1.8, np.log(1.0), np.log(1.0)], dtype=np.float64))
            lower += [0.5, np.log(1.0e-3), np.log(5.0e-2)]
            upper += [3.0, np.log(20.0), np.log(10.0)]
        else:
            starts.append(np.array(base, dtype=np.float64))
            starts.append(np.array(alt, dtype=np.float64))
            lower += [0.5]
            upper += [3.0]
    return starts, (np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64))


def fit_model(data, model_name, q_mode, nopt_feature_name, fixed_nopt):
    starts, bounds = build_x0_and_bounds(model_name, q_mode, fixed_nopt)
    best = None

    def residuals(x):
        params = unpack_params(x, model_name, q_mode, fixed_nopt)
        xi_fit = xi_model(data["theta0"], data["tp"], params, model_name, nopt_feature_name)
        return (xi_fit - data["mean_ratio"]) / data["sem"]

    for x0 in starts:
        res = least_squares(
            residuals,
            x0,
            bounds=bounds,
            max_nfev=50000,
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.0e-12,
        )
        params = unpack_params(res.x, model_name, q_mode, fixed_nopt)
        xi_fit = xi_model(data["theta0"], data["tp"], params, model_name, nopt_feature_name)
        chi2 = float(np.sum(np.square((xi_fit - data["mean_ratio"]) / data["sem"])))
        npar = len(res.x)
        dof = max(len(xi_fit) - npar, 1)
        record = {
            "model_name": model_name,
            "q_mode": q_mode,
            "nopt_feature": nopt_feature_name,
            "fixed_nopt": fixed_nopt,
            "params": params,
            "xi_fit": xi_fit,
            "chi2": chi2,
            "chi2_red": chi2 / dof,
            "rel_rmse": rel_rmse(data["mean_ratio"], xi_fit),
            "log_rmse": log_rmse(data["mean_ratio"], xi_fit),
            "n": len(xi_fit),
            "dof": dof,
            "success": bool(res.success),
            "message": str(res.message),
        }
        if best is None or record["chi2"] < best["chi2"]:
            best = record
    return best


def make_plot(data, records, out_path, dpi, title):
    theta_u = np.array(sorted(np.unique(data["theta0"])), dtype=np.float64)
    ntheta = len(theta_u)
    ncols = 3 if ntheta > 4 else 2
    nrows = int(np.ceil(ntheta / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.9 * nrows), squeeze=False)
    axes = axes.ravel()

    styles = {
        ("baseline", "fixed"): ("tab:red", "-", r"baseline, fixed $q=1.5$"),
        ("baseline", "free"): ("tab:blue", "-", r"baseline, free $q$"),
        ("crossover", "fixed"): ("tab:red", "--", r"crossover, fixed $q=1.5$"),
        ("crossover", "free"): ("tab:blue", "--", r"crossover, free $q$"),
    }

    for ax, th0 in zip(axes, theta_u):
        mask = np.isclose(data["theta0"], th0, rtol=0.0, atol=1.0e-8)
        beta_h = data["beta_over_h"][mask]
        y = data["mean_ratio"][mask]
        sem = data["sem"][mask]
        idx = np.argsort(beta_h)
        beta_h = beta_h[idx]
        y = y[idx]
        sem = sem[idx]
        ax.errorbar(beta_h, y, yerr=sem, fmt="ko", ms=3.4, lw=1.0, capsize=2, label="lattice")

        for rec in records:
            rmask = np.isclose(data["theta0"], th0, rtol=0.0, atol=1.0e-8)
            x = data["beta_over_h"][rmask]
            yfit = rec["xi_fit"][rmask]
            order = np.argsort(x)
            color, ls, label = styles[(rec["model_name"], rec["q_mode"])]
            ax.plot(x[order], yfit[order], color=color, ls=ls, lw=1.8, label=label)

        ax.set_xscale("log")
        ax.set_xlabel(r"$\beta/H_*$")
        ax.set_ylabel(r"$\xi$")
        ax.set_title(rf"$\theta_0={th0:.3g}$")
        ax.grid(alpha=0.25)

    for ax in axes[ntheta:]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_summary(path, vw, per_h, nopt_feature_name, fixed_nopt):
    with open(path, "w") as f:
        f.write("# Lattice-only comparison of baseline and crossover xi models\n")
        f.write("# Fixed wall velocity inferred from file choice\n")
        f.write(f"# v_w={float(vw):.6f}\n")
        if fixed_nopt is None:
            f.write(f"# nopt_feature={nopt_feature_name} fit_Af_alpha_from_PT=1\n")
        else:
            f.write(
                f"# nopt_feature={nopt_feature_name} fit_Af_alpha_from_PT=0 "
                f"fixed_A_f={float(fixed_nopt['amp']):.10e} fixed_alpha_f={float(fixed_nopt['alpha']):.10e}\n"
            )
        f.write("# baseline: xi = tp^(3/2)/fanh_no^2 * [fanh_inf + c*fanh_no^2/tp^q]\n")
        f.write("# crossover: xi = tp^(3/2)/fanh_no^2 * [fanh_inf + c*fanh_no^2/(tp^q*(1+(tp/tc)^r))]\n\n")
        f.write("# hstar model q_mode A_f gamma_f A_inf gamma_inf c q tc r chi2 chi2_red rel_rmse log_rmse n dof success\n")
        for hstar, records in per_h.items():
            for rec in records:
                params = rec["params"]
                tc = params["tc"] if params["tc"] is not None else np.nan
                r = params["r"] if params["r"] is not None else np.nan
                f.write(
                    f"{float(hstar):.8g} {rec['model_name']} {rec['q_mode']} "
                    f"{params['af']:.10e} {params['gf']:.10e} "
                    f"{params['ainf']:.10e} {params['ginf']:.10e} {params['c']:.10e} {params['q']:.10e} "
                    f"{tc:.10e} {r:.10e} {rec['chi2']:.10e} {rec['chi2_red']:.10e} "
                    f"{rec['rel_rmse']:.10e} {rec['log_rmse']:.10e} {rec['n']} {rec['dof']} {int(rec['success'])}\n"
                )
                f.write(f"# message {rec['message']}\n")


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    fixed_nopt = None
    if (args.fixed_nopt_amp is None) ^ (args.fixed_nopt_alpha is None):
        raise ValueError("Provide both --fixed-nopt-amp and --fixed-nopt-alpha, or neither.")
    if args.fixed_nopt_amp is not None:
        fixed_nopt = {"amp": float(args.fixed_nopt_amp), "alpha": float(args.fixed_nopt_alpha)}

    lattice = attach_tp(load_lattice(Path(args.lattice_data).resolve(), args.target_h), args.fixed_vw)

    per_h = {}
    for hstar in args.target_h:
        subset = select_h(lattice, float(hstar))
        records = []
        for model_name in ["baseline", "crossover"]:
            for q_mode in ["fixed", "free"]:
                print(f"Fitting H*={float(hstar):g} model={model_name} q_mode={q_mode}...")
                records.append(fit_model(subset, model_name, q_mode, args.nopt_feature, fixed_nopt))
        per_h[float(hstar)] = records

        plot_out = outdir / f"fit_xi_lattice_crossover_shape_H{float(hstar):.1f}".replace(".", "p")
        plot_out = Path(str(plot_out) + ".png")
        make_plot(
            subset,
            records,
            plot_out,
            args.dpi,
            rf"Crossover-shape test at $H_*={float(hstar):g}$, $v_w={args.fixed_vw:g}$",
        )
        print(f"Saved: {plot_out}")

    summary_out = outdir / "fit_xi_lattice_crossover_shape_summary.txt"
    save_summary(summary_out, args.fixed_vw, per_h, args.nopt_feature, fixed_nopt)
    print(f"Saved: {summary_out}")

    for hstar in args.target_h:
        print(f"\nH*={float(hstar):.1f}")
        for rec in per_h[float(hstar)]:
            params = rec["params"]
            extras = ""
            if rec["model_name"] == "crossover":
                extras = f", tc={params['tc']:.4f}, r={params['r']:.4f}"
            print(
                "{} {}: q={:.6f}{} chi2_red={:.3e}, rel={:.3e}".format(
                    rec["model_name"],
                    rec["q_mode"],
                    params["q"],
                    extras,
                    rec["chi2_red"],
                    rec["rel_rmse"],
                )
            )


if __name__ == "__main__":
    main()
