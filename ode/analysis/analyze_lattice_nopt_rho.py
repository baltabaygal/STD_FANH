import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Extract the relative no-PT anharmonic factor from lattice rho_noPT "
            "data, compare it to the ODE no-PT reference, and quantify whether "
            "the hilltop behavior is already captured before any PT fitting."
        )
    )
    p.add_argument("--lattice-data", type=str, default="lattice_data/data/rho_noPT_data.txt")
    p.add_argument(
        "--ode-nopt",
        type=str,
        default="ode/analysis/results/noPT_reference_dm_tp_fitready_H1p000.txt",
    )
    p.add_argument(
        "--analytic-summary",
        type=str,
        default="ode/analysis/results/fit_xi_asymptotic_model_H1p000.txt",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="ode/analysis/results/lattice_fit/noPT_rho_study",
    )
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def potential(theta0):
    theta0 = np.asarray(theta0, dtype=np.float64)
    return 1.0 - np.cos(theta0)


def h_of_theta(theta0):
    theta0 = np.asarray(theta0, dtype=np.float64)
    return np.log(np.e / np.clip(np.cos(theta0 / 2.0) ** 2, 1.0e-12, None))


def hilltop_log_of_theta(theta0):
    theta0 = np.asarray(theta0, dtype=np.float64)
    u = theta0 / np.pi
    return np.log(np.e / np.clip(1.0 - u * u, 1.0e-12, None))


def feature_of_theta(theta0, family):
    if family == "pendulum_log":
        return h_of_theta(theta0)
    if family == "hilltop_log1m":
        return hilltop_log_of_theta(theta0)
    raise ValueError(family)


def rel_rmse(y, yfit):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square((yfit - y) / y))))


def fit_powerlaw_feature(theta0, y, family):
    theta0 = np.asarray(theta0, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def model(th, amp, alpha):
        return amp * np.power(feature_of_theta(th, family), alpha)

    amp0 = float(np.median(y))
    popt, _ = curve_fit(model, theta0, y, p0=[amp0, 1.2], maxfev=20000)
    yfit = model(theta0, *popt)
    return {
        "family": family,
        "amp": float(popt[0]),
        "alpha": float(popt[1]),
        "yfit": yfit,
        "rel_rmse": rel_rmse(y, yfit),
        "hilltop_delta_frac": float((yfit[-1] - y[-1]) / y[-1]),
    }


def parse_analytic_nopt(path):
    text = Path(path).read_text().splitlines()
    patt = re.compile(r"A_f=([0-9eE+.-]+)\s+gamma_f=([0-9eE+.-]+)")
    for line in text:
        if not line.startswith("# fanh_noPT(theta)"):
            continue
        m = patt.search(line)
        if m is None:
            raise RuntimeError(f"Failed to parse analytic noPT line from {path}")
        return float(m.group(1)), float(m.group(2))
    raise RuntimeError(f"Could not find analytic noPT fit in {path}")


def load_lattice(path):
    arr = np.loadtxt(path, skiprows=1)
    theta = arr[:, 0]
    hstar = arr[:, 1]
    rho = arr[:, 2]
    theta_u = np.array(sorted(np.unique(theta)), dtype=np.float64)
    hstar_u = np.array(sorted(np.unique(hstar)), dtype=np.float64)
    return theta, hstar, rho, theta_u, hstar_u


def build_relative_curves(theta, hstar, rho, theta_u, hstar_u):
    rel_rows = []
    rho_over_v_rows = []
    for hval in hstar_u:
        mask = np.isclose(hstar, hval, rtol=0.0, atol=1.0e-12)
        sub_theta = theta[mask]
        sub_rho = rho[mask]
        order = np.argsort(sub_theta)
        sub_theta = sub_theta[order]
        sub_rho = sub_rho[order]
        if not np.allclose(sub_theta, theta_u, rtol=0.0, atol=1.0e-10):
            raise RuntimeError(f"H={hval:g} does not share the common theta grid.")
        rho_over_v = sub_rho / potential(sub_theta)
        rel_rows.append(rho_over_v / rho_over_v[0])
        rho_over_v_rows.append(rho_over_v)
    return np.array(rel_rows), np.array(rho_over_v_rows)


def factorization_metrics(rho_over_v_rows):
    logs = np.log(rho_over_v_rows)
    interaction = (
        logs
        - logs.mean(axis=0, keepdims=True)
        - logs.mean(axis=1, keepdims=True)
        + logs.mean()
    )
    return {
        "interaction_std": float(np.std(interaction)),
        "interaction_max": float(np.max(np.abs(interaction))),
    }


def load_ode_absolute(path, theta_u):
    arr = np.loadtxt(path)
    theta = arr[:, 0]
    fanh = arr[:, 2]
    out = []
    for th in theta_u:
        idx = int(np.argmin(np.abs(theta - th)))
        if abs(theta[idx] - th) > 5.0e-4:
            raise RuntimeError(f"Could not match theta={th:.8f} in {path}")
        out.append(float(fanh[idx]))
    return np.array(out, dtype=np.float64)


def make_plots(
    outdir,
    theta_u,
    hstar_u,
    rho_over_v_rows,
    rel_rows,
    rel_mean_all,
    rel_std_all,
    rel_mean_hi,
    rel_std_hi,
    ode_rel,
    analytic_rel,
    trial_rel,
    dpi,
):
    deg = np.degrees(theta_u)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for hval, row in zip(hstar_u, rho_over_v_rows):
        ax.plot(deg, row, marker="o", ms=3.5, lw=1.2, label=rf"$H_*={hval:g}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\theta_0$ [deg]")
    ax.set_ylabel(r"$\rho_{\rm noPT}/(1-\cos\theta_0)$")
    ax.set_title(r"Lattice noPT raw shape after removing $1-\cos\theta_0$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "lattice_nopt_rho_over_potential.png", dpi=dpi)
    plt.close(fig)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 7.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    ax = axes[0]
    for hval, row in zip(hstar_u, rel_rows):
        ax.plot(deg, row, marker="o", ms=3.4, lw=1.0, alpha=0.65, label=rf"$H_*={hval:g}$")
    ax.fill_between(
        deg,
        rel_mean_all - rel_std_all,
        rel_mean_all + rel_std_all,
        color="tab:blue",
        alpha=0.15,
        label="all-H mean +/- 1 sigma",
    )
    ax.plot(deg, rel_mean_all, color="tab:blue", lw=2.0, label="all-H mean")
    ax.fill_between(
        deg,
        rel_mean_hi - rel_std_hi,
        rel_mean_hi + rel_std_hi,
        color="tab:green",
        alpha=0.12,
        label=r"$H_*=1.5,2.0$ mean +/- 1 sigma",
    )
    ax.plot(deg, rel_mean_hi, color="tab:green", lw=1.8, ls="--", label=r"$H_*=1.5,2.0$ mean")
    ax.plot(deg, ode_rel, "ks-", ms=4.2, lw=1.1, label="ODE noPT")
    ax.plot(deg, analytic_rel, color="tab:red", lw=1.8, ls=":", label=r"current $A_f h^{\gamma_f}$")
    ax.plot(deg, trial_rel, color="tab:orange", lw=1.8, ls="-.", label=r"trial $A_f L^{\alpha_f}$")
    ax.set_ylabel(r"relative $f_{\rm anh}^{\rm noPT}$")
    ax.set_title(r"Extracted lattice noPT relative factor")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1]
    ax.axhline(0.0, color="black", lw=1.0)
    ax.plot(deg, 100.0 * (rel_mean_all / ode_rel - 1.0), "o-", color="tab:blue", lw=1.6, ms=4.0, label="all-H mean vs ODE")
    ax.plot(deg, 100.0 * (rel_mean_hi / ode_rel - 1.0), "o--", color="tab:green", lw=1.4, ms=3.8, label=r"$H_*=1.5,2.0$ vs ODE")
    ax.plot(deg, 100.0 * (analytic_rel / ode_rel - 1.0), "o:", color="tab:red", lw=1.6, ms=3.8, label=r"current $A_f h^{\gamma_f}$ vs ODE")
    ax.plot(deg, 100.0 * (trial_rel / ode_rel - 1.0), "o-.", color="tab:orange", lw=1.4, ms=3.8, label=r"trial $A_f L^{\alpha_f}$ vs ODE")
    ax.set_xlabel(r"$\theta_0$ [deg]")
    ax.set_ylabel("delta [%]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "lattice_nopt_relative_collapse.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    hvals = h_of_theta(theta_u)
    hgrid = np.linspace(hvals.min(), hvals.max(), 300)
    interp = PchipInterpolator(hvals, rel_mean_all)
    ax.plot(hvals, rel_mean_all, "o", color="tab:blue", ms=5.0, label="lattice mean")
    ax.plot(hgrid, interp(hgrid), "-", color="tab:blue", lw=1.8, alpha=0.85, label="PCHIP through lattice mean")
    ax.plot(hvals, ode_rel, "ks", ms=4.4, label="ODE noPT")
    ax.plot(hvals, analytic_rel, ":", color="tab:red", lw=1.8, label=r"current $A_f h^{\gamma_f}$")
    ax.plot(hvals, trial_rel, "-.", color="tab:orange", lw=1.8, label=r"trial $A_f L^{\alpha_f}$")
    ax.set_xlabel(r"$h(\theta_0)=\log(e/\cos^2(\theta_0/2))$")
    ax.set_ylabel(r"relative $f_{\rm anh}^{\rm noPT}$")
    ax.set_title(r"Hilltop behavior in the extracted noPT factor")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "lattice_nopt_relative_vs_h.png", dpi=dpi)
    plt.close(fig)


def save_outputs(
    outdir,
    theta_u,
    rel_mean_all,
    rel_std_all,
    rel_mean_hi,
    rel_std_hi,
    ode_rel,
    analytic_rel,
    trial_rel,
    factor_stats,
    fit_old,
    fit_new,
):
    table_path = outdir / "lattice_nopt_relative_table.txt"
    with open(table_path, "w") as f:
        f.write("# theta0 h h_hill rel_mean_allH rel_std_allH rel_mean_H15H20 rel_std_H15H20 ode_rel analytic_rel trial_rel\n")
        for th, mean_a, std_a, mean_h, std_h, ode, ana, trial in zip(
            theta_u,
            rel_mean_all,
            rel_std_all,
            rel_mean_hi,
            rel_std_hi,
            ode_rel,
            analytic_rel,
            trial_rel,
        ):
            f.write(
                f"{th:.10f} {h_of_theta(th):.10e} {hilltop_log_of_theta(th):.10e} "
                f"{mean_a:.10e} {std_a:.10e} "
                f"{mean_h:.10e} {std_h:.10e} "
                f"{ode:.10e} {ana:.10e} {trial:.10e}\n"
            )

    summary_path = outdir / "lattice_nopt_summary.txt"
    rel_err_all = rel_mean_all / ode_rel - 1.0
    rel_err_hi = rel_mean_hi / ode_rel - 1.0
    rel_err_ana = analytic_rel / ode_rel - 1.0
    rel_err_trial = trial_rel / ode_rel - 1.0
    with open(summary_path, "w") as f:
        f.write("# Lattice noPT rho study\n")
        f.write("# raw rho_noPT_data factorizes very well as C(H_*) * (1-cos(theta0)) * f_rel(theta0)\n")
        f.write(
            "# factorization_metrics "
            f"interaction_std_log={factor_stats['interaction_std']:.10e} "
            f"interaction_max_log={factor_stats['interaction_max']:.10e}\n"
        )
        f.write(
            "# all_H_vs_ODE "
            f"rel_rmse={rel_rmse(ode_rel, rel_mean_all):.10e} "
            f"hilltop_delta_frac={rel_err_all[-1]:.10e}\n"
        )
        f.write(
            "# H15_H20_vs_ODE "
            f"rel_rmse={rel_rmse(ode_rel, rel_mean_hi):.10e} "
            f"hilltop_delta_frac={rel_err_hi[-1]:.10e}\n"
        )
        f.write(
            "# current_analytic_Ahgamma_vs_ODE "
            f"rel_rmse={rel_rmse(ode_rel, analytic_rel):.10e} "
            f"hilltop_delta_frac={rel_err_ana[-1]:.10e}\n"
        )
        f.write(
            "# trial_hilltoplog_A_Lalpha_vs_ODE "
            f"rel_rmse={rel_rmse(ode_rel, trial_rel):.10e} "
            f"hilltop_delta_frac={rel_err_trial[-1]:.10e}\n"
        )
        f.write(
            "# fit_to_ODE_absolute pendulum_log "
            f"A_f={fit_old['amp']:.10e} alpha_f={fit_old['alpha']:.10e} "
            f"rel_rmse={fit_old['rel_rmse']:.10e}\n"
        )
        f.write(
            "# fit_to_ODE_absolute hilltop_log1m "
            f"A_f={fit_new['amp']:.10e} alpha_f={fit_new['alpha']:.10e} "
            f"rel_rmse={fit_new['rel_rmse']:.10e}\n"
        )
        f.write("\n")
        f.write("# theta0 deg allH_minus_ODE_pct H15H20_minus_ODE_pct analytic_minus_ODE_pct trial_minus_ODE_pct\n")
        for th, ea, eh, ec, et in zip(theta_u, rel_err_all, rel_err_hi, rel_err_ana, rel_err_trial):
            f.write(
                f"{th:.10f} {np.degrees(th):.6f} "
                f"{100.0 * ea:.10e} {100.0 * eh:.10e} {100.0 * ec:.10e} {100.0 * et:.10e}\n"
            )


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    theta, hstar, rho, theta_u, hstar_u = load_lattice(args.lattice_data)
    rel_rows, rho_over_v_rows = build_relative_curves(theta, hstar, rho, theta_u, hstar_u)
    factor_stats = factorization_metrics(rho_over_v_rows)

    rel_mean_all = rel_rows.mean(axis=0)
    rel_std_all = rel_rows.std(axis=0, ddof=0)

    hi_mask = np.isin(np.round(hstar_u, 12), np.round(np.array([1.5, 2.0]), 12))
    rel_rows_hi = rel_rows[hi_mask]
    rel_mean_hi = rel_rows_hi.mean(axis=0)
    rel_std_hi = rel_rows_hi.std(axis=0, ddof=0)

    ode_abs = load_ode_absolute(args.ode_nopt, theta_u)
    ode_rel = ode_abs / ode_abs[0]
    af, gf = parse_analytic_nopt(args.analytic_summary)
    analytic_abs = af * np.power(h_of_theta(theta_u), gf)
    analytic_rel = analytic_abs / analytic_abs[0]
    fit_old = fit_powerlaw_feature(theta_u, ode_abs, "pendulum_log")
    fit_new = fit_powerlaw_feature(theta_u, ode_abs, "hilltop_log1m")
    trial_rel = fit_new["yfit"] / fit_new["yfit"][0]

    make_plots(
        outdir,
        theta_u,
        hstar_u,
        rho_over_v_rows,
        rel_rows,
        rel_mean_all,
        rel_std_all,
        rel_mean_hi,
        rel_std_hi,
        ode_rel,
        analytic_rel,
        trial_rel,
        args.dpi,
    )
    save_outputs(
        outdir,
        theta_u,
        rel_mean_all,
        rel_std_all,
        rel_mean_hi,
        rel_std_hi,
        ode_rel,
        analytic_rel,
        trial_rel,
        factor_stats,
        fit_old,
        fit_new,
    )

    print(
        "all-H vs ODE rel_rmse={:.4e}, H*=1.5/2.0 vs ODE rel_rmse={:.4e}, analytic Ah^gamma vs ODE rel_rmse={:.4e}, trial AL^alpha vs ODE rel_rmse={:.4e}".format(
            rel_rmse(ode_rel, rel_mean_all),
            rel_rmse(ode_rel, rel_mean_hi),
            rel_rmse(ode_rel, analytic_rel),
            rel_rmse(ode_rel, trial_rel),
        )
    )
    print(
        "hilltop deltas: all-H={:+.3f}%, H15/H20={:+.3f}%, analytic={:+.3f}%, trial={:+.3f}%".format(
            100.0 * (rel_mean_all[-1] / ode_rel[-1] - 1.0),
            100.0 * (rel_mean_hi[-1] / ode_rel[-1] - 1.0),
            100.0 * (analytic_rel[-1] / ode_rel[-1] - 1.0),
            100.0 * (trial_rel[-1] / ode_rel[-1] - 1.0),
        )
    )


if __name__ == "__main__":
    main()
