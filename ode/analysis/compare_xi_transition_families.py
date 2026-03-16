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


Q_FIXED = 1.5
TARGET_H_DEFAULT = [1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compare asymptotics-preserving transition families for xi(theta,tp) "
            "on lattice data with fixed v_w and fixed q=3/2."
        )
    )
    p.add_argument("--lattice-data", type=str, default="lattice_data/data/energy_ratio_by_theta_data_v9.txt")
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--target-h", type=float, nargs="+", default=TARGET_H_DEFAULT)
    p.add_argument("--nopt-feature", type=str, default="pendulum_log", choices=["pendulum_log", "hilltop_log1m"])
    p.add_argument("--fixed-nopt-amp", type=float, default=None)
    p.add_argument("--fixed-nopt-alpha", type=float, default=None)
    p.add_argument("--outdir", type=str, default="ode/analysis/results/lattice_fit/transition_family_compare_v9")
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


def attach_tp(data, vw):
    perc = PercolationCache()
    tp = np.array(
        [perc.get(float(h), float(bh), float(vw)) for h, bh in zip(data["hstar"], data["beta_over_h"])],
        dtype=np.float64,
    )
    out = dict(data)
    out["tp"] = tp
    return out


def select_h(data, hstar):
    mask = np.isclose(data["hstar"], float(hstar), rtol=0.0, atol=1.0e-12)
    return {key: val[mask] for key, val in data.items()}


def unpack_common(x, fixed_nopt):
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
    return {
        "af": af,
        "gf": gf,
        "ainf": ainf,
        "ginf": ginf,
        "c": c,
        "pos": pos,
    }


def y3_components(theta0, tp, params, nopt_feature_name):
    nvar = nopt_feature(theta0, nopt_feature_name)
    fanh_no = params["af"] * np.power(nvar, params["gf"])
    yinf = params["ainf"] * np.power(nvar, params["ginf"])
    b = params["c"] * np.square(fanh_no)
    transient = b / np.power(tp, Q_FIXED)
    return fanh_no, yinf, b, transient


def xi_from_y3(tp, fanh_no, y3):
    return np.power(tp, 1.5) / np.square(fanh_no) * y3


def xi_model(theta0, tp, x, family, nopt_feature_name, fixed_nopt):
    params = unpack_common(x, fixed_nopt)
    fanh_no, yinf, b, transient = y3_components(theta0, tp, params, nopt_feature_name)

    if family == "baseline":
        y3 = yinf + transient
    elif family == "gmean":
        m = float(np.exp(x[params["pos"]]))
        y3 = np.power(np.power(yinf, m) + np.power(transient, m), 1.0 / m)
    elif family == "supp_tc":
        r = float(np.exp(x[params["pos"]]))
        tc = np.power(np.maximum(b / np.maximum(yinf, 1.0e-12), 1.0e-300), 1.0 / Q_FIXED)
        y3 = yinf + transient / (1.0 + np.power(tp / np.maximum(tc, 1.0e-12), r))
    elif family == "supp_free":
        tc = float(np.exp(x[params["pos"]]))
        r = float(np.exp(x[params["pos"] + 1]))
        y3 = yinf + transient / (1.0 + np.power(tp / tc, r))
    else:
        raise ValueError(family)

    return xi_from_y3(tp, fanh_no, y3)


def initial_guesses(family, fixed_nopt):
    if fixed_nopt is None:
        base = [np.log(0.30), 1.10, np.log(0.10), 1.45, np.log(1.0)]
        alt = [np.log(0.25), 1.00, np.log(0.08), 1.70, np.log(1.0)]
        lower = [np.log(1.0e-6), 0.0, np.log(1.0e-8), 0.0, np.log(1.0e-8)]
        upper = [np.log(10.0), 4.0, np.log(10.0), 4.0, np.log(10.0)]
    else:
        base = [np.log(0.10), 1.45, np.log(1.0)]
        alt = [np.log(0.08), 1.70, np.log(1.0)]
        lower = [np.log(1.0e-8), 0.0, np.log(1.0e-8)]
        upper = [np.log(10.0), 4.0, np.log(10.0)]

    if family == "baseline":
        x0 = [np.array(base, dtype=np.float64), np.array(alt, dtype=np.float64)]
    elif family == "gmean":
        x0 = [
            np.array(base + [np.log(1.5)], dtype=np.float64),
            np.array(alt + [np.log(2.0)], dtype=np.float64),
            np.array(base + [np.log(0.7)], dtype=np.float64),
        ]
        lower += [np.log(5.0e-2)]
        upper += [np.log(20.0)]
    elif family == "supp_tc":
        x0 = [
            np.array(base + [np.log(1.5)], dtype=np.float64),
            np.array(alt + [np.log(2.0)], dtype=np.float64),
            np.array(base + [np.log(0.8)], dtype=np.float64),
        ]
        lower += [np.log(5.0e-2)]
        upper += [np.log(20.0)]
    elif family == "supp_free":
        x0 = [
            np.array(base + [np.log(1.0), np.log(1.5)], dtype=np.float64),
            np.array(alt + [np.log(0.7), np.log(2.0)], dtype=np.float64),
            np.array(base + [np.log(2.0), np.log(0.8)], dtype=np.float64),
        ]
        lower += [np.log(1.0e-3), np.log(5.0e-2)]
        upper += [np.log(20.0), np.log(20.0)]
    else:
        raise ValueError(family)
    return x0, (np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64))


def fit_family(data, family, nopt_feature_name, fixed_nopt):
    x0_list, bounds = initial_guesses(family, fixed_nopt)
    best = None

    def residuals(x):
        yfit = xi_model(data["theta0"], data["tp"], x, family, nopt_feature_name, fixed_nopt)
        return (yfit - data["mean_ratio"]) / data["sem"]

    for x0 in x0_list:
        res = least_squares(
            residuals,
            x0,
            bounds=bounds,
            max_nfev=50000,
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.0e-12,
        )
        yfit = xi_model(data["theta0"], data["tp"], res.x, family, nopt_feature_name, fixed_nopt)
        chi2 = float(np.sum(np.square((yfit - data["mean_ratio"]) / data["sem"])))
        npar = len(res.x)
        n = len(yfit)
        dof = max(n - npar, 1)
        record = {
            "family": family,
            "nopt_feature": nopt_feature_name,
            "fixed_nopt": fixed_nopt,
            "x": res.x.copy(),
            "chi2": chi2,
            "chi2_red": chi2 / dof,
            "rel_rmse": rel_rmse(data["mean_ratio"], yfit),
            "log_rmse": log_rmse(data["mean_ratio"], yfit),
            "n": n,
            "dof": dof,
            "aic": chi2 + 2.0 * npar,
            "bic": chi2 + npar * np.log(n),
            "success": bool(res.success),
            "message": str(res.message),
            "yfit": yfit,
        }
        if best is None or record["chi2"] < best["chi2"]:
            best = record
    return best


def summarize_params(record):
    common = unpack_common(record["x"], record.get("fixed_nopt"))
    extras = ""
    if record["family"] == "gmean":
        extras = f" m={np.exp(record['x'][common['pos']]):.6f}"
    elif record["family"] == "supp_tc":
        extras = f" r={np.exp(record['x'][common['pos']]):.6f}"
    elif record["family"] == "supp_free":
        extras = f" tc={np.exp(record['x'][common['pos']]):.6f} r={np.exp(record['x'][common['pos'] + 1]):.6f}"
    nopt_tag = record.get("nopt_feature", "pendulum_log")
    return (
        f"nopt={nopt_tag} A_f={common['af']:.6e} gf={common['gf']:.6f} "
        f"A_inf={common['ainf']:.6e} ginf={common['ginf']:.6f} c={common['c']:.6f}{extras}"
    )


def make_plot(data, records, out_path, dpi, title):
    theta_u = np.array(sorted(np.unique(data["theta0"])), dtype=np.float64)
    ntheta = len(theta_u)
    ncols = 3 if ntheta > 4 else 2
    nrows = int(np.ceil(ntheta / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 3.9 * nrows), squeeze=False)
    axes = axes.ravel()

    style = {
        "baseline": ("tab:gray", "-", "baseline"),
        "gmean": ("tab:red", "--", "gmean"),
        "supp_tc": ("tab:green", "-.", "supp_tc"),
        "supp_free": ("tab:blue", ":", "supp_free"),
    }

    for ax, th0 in zip(axes, theta_u):
        mask = np.isclose(data["theta0"], th0, rtol=0.0, atol=1.0e-8)
        beta_h = data["beta_over_h"][mask]
        y = data["mean_ratio"][mask]
        sem = data["sem"][mask]
        order = np.argsort(beta_h)
        beta_h = beta_h[order]
        y = y[order]
        sem = sem[order]

        ax.errorbar(beta_h, y, yerr=sem, fmt="ko", ms=3.4, lw=1.0, capsize=2, label="lattice")
        for rec in records:
            rmask = np.isclose(data["theta0"], th0, rtol=0.0, atol=1.0e-8)
            x = data["beta_over_h"][rmask]
            yfit = rec["yfit"][rmask]
            idx = np.argsort(x)
            color, ls, label = style[rec["family"]]
            ax.plot(x[idx], yfit[idx], color=color, ls=ls, lw=1.8, label=label)

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


def save_summary(path, vw, q_fixed, records_by_h, nopt_feature_name, fixed_nopt):
    with open(path, "w") as f:
        f.write("# Transition-family comparison on lattice xi data\n")
        f.write(f"# fixed_vw={float(vw):.6f} fixed_q={float(q_fixed):.6f}\n")
        if fixed_nopt is None:
            f.write(f"# nopt_feature={nopt_feature_name} fit_Af_alpha_from_PT=1\n")
        else:
            f.write(
                f"# nopt_feature={nopt_feature_name} fit_Af_alpha_from_PT=0 "
                f"fixed_A_f={float(fixed_nopt['amp']):.10e} fixed_alpha_f={float(fixed_nopt['alpha']):.10e}\n"
            )
        f.write("# baseline: Y3 = Yinf + B/tp^q\n")
        f.write("# gmean:    Y3 = (Yinf^m + (B/tp^q)^m)^(1/m)\n")
        f.write("# supp_tc:  Y3 = Yinf + [B/tp^q] / [1 + (tp/tc)^r], tc=(B/Yinf)^(1/q)\n")
        f.write("# supp_free:Y3 = Yinf + [B/tp^q] / [1 + (tp/tc)^r]\n\n")
        f.write("# hstar family chi2 chi2_red aic bic rel_rmse log_rmse n dof success params\n")
        for hstar, records in records_by_h.items():
            for rec in records:
                f.write(
                    f"{float(hstar):.8g} {rec['family']} {rec['chi2']:.10e} {rec['chi2_red']:.10e} "
                    f"{rec['aic']:.10e} {rec['bic']:.10e} {rec['rel_rmse']:.10e} {rec['log_rmse']:.10e} "
                    f"{rec['n']} {rec['dof']} {int(rec['success'])} {summarize_params(rec)}\n"
                )
                f.write(f"# message {rec['message']}\n")


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    lattice = attach_tp(load_lattice(Path(args.lattice_data).resolve(), args.target_h), args.fixed_vw)
    families = ["baseline", "gmean", "supp_tc", "supp_free"]
    fixed_nopt = None
    if (args.fixed_nopt_amp is None) ^ (args.fixed_nopt_alpha is None):
        raise ValueError("Provide both --fixed-nopt-amp and --fixed-nopt-alpha, or neither.")
    if args.fixed_nopt_amp is not None:
        fixed_nopt = {"amp": float(args.fixed_nopt_amp), "alpha": float(args.fixed_nopt_alpha)}

    records_by_h = {}
    for hstar in args.target_h:
        subset = select_h(lattice, float(hstar))
        print(f"Fitting H*={float(hstar):g} transition families...")
        records = [fit_family(subset, family, args.nopt_feature, fixed_nopt) for family in families]
        records.sort(key=lambda rec: (rec["chi2_red"], rec["rel_rmse"]))
        records_by_h[float(hstar)] = records

        plot_out = outdir / f"compare_xi_transition_families_H{float(hstar):.1f}".replace(".", "p")
        plot_out = Path(str(plot_out) + ".png")
        make_plot(
            subset,
            records,
            plot_out,
            args.dpi,
            rf"Transition-family comparison at $H_*={float(hstar):g}$, $v_w={args.fixed_vw:g}$",
        )
        print(f"Saved: {plot_out}")
        for rec in records:
            print(
                "{}: chi2_red={:.3e}, rel={:.3e}, {}".format(
                    rec["family"],
                    rec["chi2_red"],
                    rec["rel_rmse"],
                    summarize_params(rec),
                )
            )

    summary_out = outdir / "compare_xi_transition_families_summary.txt"
    save_summary(summary_out, args.fixed_vw, Q_FIXED, records_by_h, args.nopt_feature, fixed_nopt)
    print(f"Saved: {summary_out}")


if __name__ == "__main__":
    main()
