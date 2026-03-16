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
Q_FIXED = 1.5


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Fit H*-shared lattice xi models with fixed q=3/2 and fixed noPT law, "
            "and compare tp-based versus x=tp/t* transition variables."
        )
    )
    p.add_argument("--lattice-data", type=str, default="lattice_data/data/energy_ratio_by_theta_data_v9.txt")
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--target-h", type=float, nargs="+", default=TARGET_H_DEFAULT)
    p.add_argument("--nopt-feature", type=str, default="hilltop_log1m", choices=["pendulum_log", "hilltop_log1m"])
    p.add_argument("--fixed-nopt-amp", type=float, required=True)
    p.add_argument("--fixed-nopt-alpha", type=float, required=True)
    p.add_argument(
        "--outdir",
        type=str,
        default="ode/analysis/results/lattice_fit/shared_transition_variable_v9",
    )
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def pendulum_log_of_theta(theta0):
    theta0 = np.asarray(theta0, dtype=np.float64)
    return np.log(np.e / np.clip(np.cos(theta0 / 2.0) ** 2, 1.0e-12, None))


def hilltop_log_of_theta(theta0):
    theta0 = np.asarray(theta0, dtype=np.float64)
    u = theta0 / np.pi
    return np.log(np.e / np.clip(1.0 - u * u, 1.0e-12, None))


def feature_of_theta(theta0, feature_name):
    if feature_name == "pendulum_log":
        return pendulum_log_of_theta(theta0)
    if feature_name == "hilltop_log1m":
        return hilltop_log_of_theta(theta0)
    raise ValueError(feature_name)


def rel_rmse(y, yfit):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square((yfit - y) / y))))


def log_rmse(y, yfit):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
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


def attach_tp(data, vw):
    perc = PercolationCache()
    tp = np.array(
        [perc.get(float(h), float(bh), float(vw)) for h, bh in zip(data["hstar"], data["beta_over_h"])],
        dtype=np.float64,
    )
    out = dict(data)
    out["tp"] = tp
    out["xvar"] = 2.0 * out["hstar"] * out["tp"]
    return out


def select_h(data, hstar):
    mask = np.isclose(data["hstar"], float(hstar), rtol=0.0, atol=1.0e-12)
    return {key: val[mask] for key, val in data.items()}


def unpack_params(x, model_name):
    ainf = float(np.exp(x[0]))
    ginf = float(x[1])
    c = float(np.exp(x[2]))
    pos = 3
    if model_name == "baseline":
        zc = None
        r = None
    else:
        zc = float(np.exp(x[pos]))
        r = float(np.exp(x[pos + 1]))
    return {
        "ainf": ainf,
        "ginf": ginf,
        "c": c,
        "zc": zc,
        "r": r,
    }


def xi_model(theta0, tp, xvar, params, model_name, feature_name, nopt_amp, nopt_alpha):
    feat = feature_of_theta(theta0, feature_name)
    fanh_no = nopt_amp * np.power(feat, nopt_alpha)
    fanh_inf = params["ainf"] * np.power(feat, params["ginf"])
    transient = params["c"] * np.square(fanh_no) / np.power(tp, Q_FIXED)
    if model_name == "crossover_tp":
        transient = transient / (1.0 + np.power(tp / params["zc"], params["r"]))
    elif model_name == "crossover_x":
        transient = transient / (1.0 + np.power(xvar / params["zc"], params["r"]))
    elif model_name != "baseline":
        raise ValueError(model_name)
    y3 = fanh_inf + transient
    return np.power(tp, 1.5) / np.square(fanh_no) * y3


def initial_guesses(model_name):
    base = [np.log(0.15), 2.20, np.log(1.02)]
    alt = [np.log(0.12), 2.00, np.log(0.95)]
    lower = [np.log(1.0e-8), 0.0, np.log(1.0e-8)]
    upper = [np.log(10.0), 5.0, np.log(10.0)]

    if model_name == "baseline":
        starts = [
            np.array(base, dtype=np.float64),
            np.array(alt, dtype=np.float64),
        ]
    else:
        starts = [
            np.array(base + [np.log(1.0), np.log(1.5)], dtype=np.float64),
            np.array(base + [np.log(2.0), np.log(3.0)], dtype=np.float64),
            np.array(alt + [np.log(0.7), np.log(1.0)], dtype=np.float64),
        ]
        lower += [np.log(1.0e-3), np.log(5.0e-2)]
        upper += [np.log(50.0), np.log(20.0)]
    return starts, (np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64))


def fit_model(data, model_name, feature_name, nopt_amp, nopt_alpha):
    starts, bounds = initial_guesses(model_name)
    best = None

    def residuals(x):
        params = unpack_params(x, model_name)
        xi_fit = xi_model(
            data["theta0"],
            data["tp"],
            data["xvar"],
            params,
            model_name,
            feature_name,
            nopt_amp,
            nopt_alpha,
        )
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
        params = unpack_params(res.x, model_name)
        xi_fit = xi_model(
            data["theta0"],
            data["tp"],
            data["xvar"],
            params,
            model_name,
            feature_name,
            nopt_amp,
            nopt_alpha,
        )
        chi2 = float(np.sum(np.square((xi_fit - data["mean_ratio"]) / data["sem"])))
        npar = len(res.x)
        n = len(xi_fit)
        dof = max(n - npar, 1)
        record = {
            "model_name": model_name,
            "params": params,
            "xi_fit": xi_fit,
            "chi2": chi2,
            "chi2_red": chi2 / dof,
            "rel_rmse": rel_rmse(data["mean_ratio"], xi_fit),
            "log_rmse": log_rmse(data["mean_ratio"], xi_fit),
            "n": n,
            "dof": dof,
            "success": bool(res.success),
            "message": str(res.message),
        }
        if best is None or record["chi2"] < best["chi2"]:
            best = record
    return best


def metrics_for_subset(data, xi_fit):
    chi2 = float(np.sum(np.square((xi_fit - data["mean_ratio"]) / data["sem"])))
    return {
        "chi2": chi2,
        "chi2_red": chi2 / max(len(xi_fit), 1),
        "rel_rmse": rel_rmse(data["mean_ratio"], xi_fit),
        "log_rmse": log_rmse(data["mean_ratio"], xi_fit),
        "n": len(xi_fit),
    }


def save_summary(path, args, records, per_h_metrics):
    with open(path, "w") as f:
        f.write("# Shared-H lattice fit with fixed q=3/2 and fixed noPT law\n")
        f.write(f"# lattice_data={Path(args.lattice_data).resolve()}\n")
        f.write(f"# fixed_vw={float(args.fixed_vw):.6f}\n")
        f.write(
            f"# nopt_feature={args.nopt_feature} "
            f"fixed_A_f={float(args.fixed_nopt_amp):.10e} "
            f"fixed_alpha_f={float(args.fixed_nopt_alpha):.10e}\n"
        )
        f.write("# baseline:     Y3 = Yinf + c fno^2 / tp^(3/2)\n")
        f.write("# crossover_tp: Y3 = Yinf + c fno^2 / [tp^(3/2) (1 + (tp/zc)^r)]\n")
        f.write("# crossover_x:  Y3 = Yinf + c fno^2 / [tp^(3/2) (1 + (x/zc)^r)], x=tp/t*=2H*tp\n\n")
        f.write("# combined model A_inf gamma_inf c zc r chi2 chi2_red rel_rmse log_rmse n dof success\n")
        for rec in records:
            p = rec["params"]
            zc = p["zc"] if p["zc"] is not None else np.nan
            r = p["r"] if p["r"] is not None else np.nan
            f.write(
                f"{rec['model_name']} "
                f"{p['ainf']:.10e} {p['ginf']:.10e} {p['c']:.10e} "
                f"{zc:.10e} {r:.10e} "
                f"{rec['chi2']:.10e} {rec['chi2_red']:.10e} "
                f"{rec['rel_rmse']:.10e} {rec['log_rmse']:.10e} "
                f"{rec['n']} {rec['dof']} {int(rec['success'])}\n"
            )
            f.write(f"# message {rec['message']}\n")
        f.write("\n")
        f.write("# per_h model hstar chi2 chi2_red rel_rmse log_rmse n\n")
        for rec in records:
            for hstar, met in per_h_metrics[rec["model_name"]].items():
                f.write(
                    f"{rec['model_name']} {float(hstar):.8g} "
                    f"{met['chi2']:.10e} {met['chi2_red']:.10e} "
                    f"{met['rel_rmse']:.10e} {met['log_rmse']:.10e} {met['n']}\n"
                )


def make_plot(data, records, out_path, dpi, title):
    theta_u = np.array(sorted(np.unique(data["theta0"])), dtype=np.float64)
    ntheta = len(theta_u)
    ncols = 3 if ntheta > 4 else 2
    nrows = int(np.ceil(ntheta / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.0 * nrows), squeeze=False)
    axes = axes.ravel()

    styles = {
        "baseline": ("tab:gray", "-", "baseline"),
        "crossover_tp": ("tab:red", "--", "crossover tp"),
        "crossover_x": ("tab:blue", "-.", "crossover x"),
    }

    for ax, th0 in zip(axes, theta_u):
        mask = np.isclose(data["theta0"], th0, rtol=0.0, atol=1.0e-8)
        beta_h = data["beta_over_h"][mask]
        y = data["mean_ratio"][mask]
        sem = data["sem"][mask]
        order = np.argsort(beta_h)
        ax.errorbar(beta_h[order], y[order], yerr=sem[order], fmt="ko", ms=3.4, lw=1.0, capsize=2, label="lattice")
        for rec in records:
            yfit = rec["plot_yfit"][mask]
            color, ls, label = styles[rec["model_name"]]
            ax.plot(beta_h[order], yfit[order], color=color, ls=ls, lw=1.8, label=label)
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


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    data = attach_tp(load_lattice(Path(args.lattice_data).resolve(), args.target_h), args.fixed_vw)
    models = ["baseline", "crossover_tp", "crossover_x"]
    records = []
    per_h_metrics = {}

    for model_name in models:
        print(f"Fitting shared model={model_name} ...")
        rec = fit_model(
            data,
            model_name,
            args.nopt_feature,
            float(args.fixed_nopt_amp),
            float(args.fixed_nopt_alpha),
        )
        records.append(rec)
        per_h_metrics[model_name] = {}
        for hstar in args.target_h:
            subset = select_h(data, float(hstar))
            mask = np.isclose(data["hstar"], float(hstar), rtol=0.0, atol=1.0e-12)
            per_h_metrics[model_name][float(hstar)] = metrics_for_subset(subset, rec["xi_fit"][mask])

    summary_out = outdir / "fit_lattice_shared_transition_variable_summary.txt"
    save_summary(summary_out, args, records, per_h_metrics)
    print(f"Saved: {summary_out}")

    for hstar in args.target_h:
        subset = select_h(data, float(hstar))
        plot_records = []
        full_mask = np.isclose(data["hstar"], float(hstar), rtol=0.0, atol=1.0e-12)
        for rec in records:
            rec_plot = dict(rec)
            rec_plot["plot_yfit"] = rec["xi_fit"][full_mask]
            plot_records.append(rec_plot)
        out_plot = outdir / f"fit_lattice_shared_transition_variable_H{float(hstar):.1f}".replace(".", "p")
        out_plot = Path(str(out_plot) + ".png")
        make_plot(
            subset,
            plot_records,
            out_plot,
            args.dpi,
            rf"Shared-H fit at $H_*={float(hstar):g}$, $v_w={args.fixed_vw:g}$",
        )
        print(f"Saved: {out_plot}")

    for rec in records:
        p = rec["params"]
        extra = ""
        if rec["model_name"] != "baseline":
            extra = f", zc={p['zc']:.4f}, r={p['r']:.4f}"
        print(
            "{}: A_inf={:.6e}, ginf={:.6f}, c={:.6f}{} chi2_red={:.3e}, rel={:.3e}".format(
                rec["model_name"],
                p["ainf"],
                p["ginf"],
                p["c"],
                extra,
                rec["chi2_red"],
                rec["rel_rmse"],
            )
        )
        for hstar in args.target_h:
            met = per_h_metrics[rec["model_name"]][float(hstar)]
            print(
                "  H*={:.1f}: chi2_red={:.3e}, rel={:.3e}".format(
                    float(hstar),
                    met["chi2_red"],
                    met["rel_rmse"],
                )
            )


if __name__ == "__main__":
    main()
