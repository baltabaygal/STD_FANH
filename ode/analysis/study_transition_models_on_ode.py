import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ode.analysis.analyze_modelB_degeneracy_checks import build_theta_arrays
from ode.analysis.fit_tau_delay_lattice import (
    optimize_r_baseline,
    optimize_tau_r,
    optimize_tau_r_tc,
    profile_tau_fixed_tc,
    score_tau,
)
from ode.analysis.plot_lattice_y3_from_rho import build_ode_dataset, load_ode_nopt, load_ode_ratio, select_h
from ode.analysis.profile_tc_fixed_sc import profile_tc


TARGET_H_DEFAULT = [0.5, 1.0, 1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Study the transition-shift/additive-delay model on direct ODE data "
            "for multiple H* slices."
        )
    )
    p.add_argument("--ode-xi", type=str, default="ode/xi_DM_ODE_results.txt")
    p.add_argument("--ode-nopt", type=str, default="ode/analysis/results/noPT_reference_dm_tp_fitready_H1p000.txt")
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--target-h", type=float, nargs="+", default=TARGET_H_DEFAULT)
    p.add_argument("--fixed-s", type=float, default=1.0)
    p.add_argument("--fixed-c", type=float, default=1.0)
    p.add_argument("--fixed-tc", type=float, default=1.5)
    p.add_argument("--tc-min", type=float, default=0.8)
    p.add_argument("--tc-max", type=float, default=8.0)
    p.add_argument("--tc-step", type=float, default=0.02)
    p.add_argument("--tau-min", type=float, default=0.0)
    p.add_argument("--tau-max", type=float, default=3.0)
    p.add_argument("--tau-step", type=float, default=0.02)
    p.add_argument("--r-min", type=float, default=0.5)
    p.add_argument("--r-max", type=float, default=12.0)
    p.add_argument(
        "--outdir",
        type=str,
        default="ode/analysis/results/ode_transition_models_vw0p9",
    )
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def y3_from_record(data_h, theta_index, f_theta, rec):
    frow = f_theta[theta_index]
    return rec["xi_fit"] * np.square(frow) / np.power(data_h["tp"], 1.5)


def make_overlay_plot(data_h, theta_u, theta_index, f_theta, baseline_rec, best_rec, out_path, dpi, title):
    ntheta = len(theta_u)
    fig, axes = plt.subplots(
        2,
        ntheta,
        figsize=(3.8 * ntheta, 6.0),
        squeeze=False,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        sharex="col",
    )
    y3_base = y3_from_record(data_h, theta_index, f_theta, baseline_rec)
    y3_best = y3_from_record(data_h, theta_index, f_theta, best_rec)
    y3_data = data_h["y3"]
    resid = best_rec["rel_resid"]

    for j, th0 in enumerate(theta_u):
        mask = np.isclose(data_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
        order = np.argsort(data_h["tp"][mask])
        tp = data_h["tp"][mask][order]
        y3 = y3_data[mask][order]
        y3b = y3_base[mask][order]
        y3m = y3_best[mask][order]
        rr = resid[mask][order]

        ax = axes[0, j]
        ax.plot(tp, y3, "ko", ms=3.2, label="ODE")
        ax.plot(tp, y3b, color="tab:gray", lw=1.5, ls="--", label="baseline")
        ax.plot(tp, y3m, color="tab:red", lw=1.8, label="best")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$Y_3$")
        ax.set_title(rf"$\theta_0={th0:.3g}$")
        ax.grid(alpha=0.25)

        ax = axes[1, j]
        ax.axhline(0.0, color="black", lw=1.0)
        ax.plot(tp, rr, "o-", color="tab:red", ms=3.0, lw=1.2)
        ax.set_xscale("log")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$\Delta\xi/\xi$")
        ax.grid(alpha=0.25)

    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_relrmse_summary_plot(rows, fixed_tc, out_path, dpi):
    h = np.array([row["hstar"] for row in rows], dtype=np.float64)
    rel_base = np.array([row["baseline"]["rel_rmse"] for row in rows], dtype=np.float64)
    rel_tc = np.array([row["free_tc"]["rel_rmse"] for row in rows], dtype=np.float64)
    rel_tau = np.array([row["tau_fixed_tc"]["rel_rmse"] for row in rows], dtype=np.float64)
    rel_tau_tc = np.array([row["tau_free_tc"]["rel_rmse"] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(h, rel_base, "o-", color="tab:gray", lw=1.8, ms=5.0, label=rf"baseline ($t_c={fixed_tc:g}, \tau=0$)")
    ax.plot(h, rel_tc, "o-", color="tab:blue", lw=1.8, ms=5.0, label=r"free $t_c$")
    ax.plot(h, rel_tau, "o-", color="tab:green", lw=1.8, ms=5.0, label=rf"free $\tau_p$, fixed $t_c={fixed_tc:g}$")
    ax.plot(h, rel_tau_tc, "o-", color="tab:red", lw=1.8, ms=5.0, label=r"free $\tau_p$, free $t_c$")
    ax.set_xlabel(r"$H_*$")
    ax.set_ylabel("rel-RMSE")
    ax.set_title("Transition-model comparison on direct ODE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_param_summary_plot(rows, fixed_tc, out_path, dpi):
    h = np.array([row["hstar"] for row in rows], dtype=np.float64)
    tc_best = np.array([row["free_tc"]["tc"] for row in rows], dtype=np.float64)
    r_tc = np.array([row["free_tc"]["r"] for row in rows], dtype=np.float64)
    tau_fix = np.array([row["tau_fixed_tc"]["tau"] for row in rows], dtype=np.float64)
    r_tau = np.array([row["tau_fixed_tc"]["r"] for row in rows], dtype=np.float64)
    tau_free = np.array([row["tau_free_tc"]["tau"] for row in rows], dtype=np.float64)
    tc_tau = np.array([row["tau_free_tc"]["tc"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True)

    axes[0].plot(h, tc_best, "o-", color="tab:blue", lw=1.8, ms=5.0, label=r"free $t_c$")
    axes[0].plot(h, tc_tau, "o-", color="tab:red", lw=1.8, ms=5.0, label=r"free $\tau_p$, free $t_c$")
    axes[0].axhline(float(fixed_tc), color="black", lw=1.0, ls="--")
    axes[0].set_ylabel(r"$t_c$")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(h, tau_fix, "o-", color="tab:green", lw=1.8, ms=5.0, label=rf"fixed $t_c={fixed_tc:g}$")
    axes[1].plot(h, tau_free, "o-", color="tab:red", lw=1.8, ms=5.0, label=r"free $t_c$")
    axes[1].set_ylabel(r"$\tau_p$")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(h, r_tc, "o-", color="tab:blue", lw=1.8, ms=5.0, label=r"free $t_c$")
    axes[2].plot(h, r_tau, "o-", color="tab:green", lw=1.8, ms=5.0, label=rf"free $\tau_p$, fixed $t_c={fixed_tc:g}$")
    axes[2].plot(h, np.array([row["tau_free_tc"]["r"] for row in rows], dtype=np.float64), "o-", color="tab:red", lw=1.8, ms=5.0, label=r"free $\tau_p$, free $t_c$")
    axes[2].set_ylabel(r"$r$")
    axes[2].set_xlabel(r"$H_*$")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_tc_profile_plot(rows, out_path, dpi):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    colors = {0.5: "tab:green", 1.0: "tab:orange", 1.5: "tab:blue", 2.0: "tab:red"}
    for row in rows:
        prof = row["tc_profile"]
        tc = np.array([rec["tc"] for rec in prof["rows"]], dtype=np.float64)
        rel = np.array([rec["rel_rmse"] for rec in prof["rows"]], dtype=np.float64)
        color = colors.get(row["hstar"], None)
        ax.plot(tc, rel, lw=1.8, color=color, label=rf"$H_*={row['hstar']:g}$")
    ax.axvline(1.5, color="black", lw=1.0, ls="--")
    ax.set_xlabel(r"$t_c$")
    ax.set_ylabel("rel-RMSE")
    ax.set_title(r"ODE profile over $t_c$ with fixed $s=1$, $c=1$, $\tau_p=0$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_tau_profile_plot(rows, out_path, dpi):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    colors = {0.5: "tab:green", 1.0: "tab:orange", 1.5: "tab:blue", 2.0: "tab:red"}
    for row in rows:
        prof = row["tau_profile"]
        tau = np.array([rec["tau"] for rec in prof["rows"]], dtype=np.float64)
        rel = np.array([rec["rel_rmse"] for rec in prof["rows"]], dtype=np.float64)
        color = colors.get(row["hstar"], None)
        ax.plot(tau, rel, lw=1.8, color=color, label=rf"$H_*={row['hstar']:g}$")
    ax.set_xlabel(r"$\tau_p$")
    ax.set_ylabel("rel-RMSE")
    ax.set_title(r"ODE profile over $\tau_p$ with fixed $s=1$, $c=1$, $t_c=1.5$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_summary(path, args, rows):
    with open(path, "w") as f:
        f.write("# Transition-model study on direct ODE data\n")
        f.write(f"# ode_xi={Path(args.ode_xi).resolve()}\n")
        f.write(f"# ode_nopt={Path(args.ode_nopt).resolve()}\n")
        f.write(f"# fixed_vw={float(args.fixed_vw):.6f}\n")
        f.write(f"# fixed_s={float(args.fixed_s):.6f}\n")
        f.write(f"# fixed_c={float(args.fixed_c):.6f}\n")
        f.write(f"# fixed_tc={float(args.fixed_tc):.6f}\n")
        f.write("# baseline: tau=0, tc fixed, fit r and per-theta F_inf\n")
        f.write("# free_tc: tau=0, tc free, fit r and per-theta F_inf\n")
        f.write("# tau_fixed_tc: tc fixed, tau and r free, per-theta F_inf free\n")
        f.write("# tau_free_tc: tc, tau, r free, per-theta F_inf free\n\n")
        f.write("# hstar baseline_r baseline_rel free_tc tc_best r_best rel_best tau_fixed tau_fixed_r tau_fixed_rel tau_free tc_free r_free rel_free\n")
        for row in rows:
            f.write(
                f"{row['hstar']:.8g} "
                f"{row['baseline']['r']:.10e} {row['baseline']['rel_rmse']:.10e} "
                f"{row['free_tc']['tc']:.10e} {row['free_tc']['r']:.10e} {row['free_tc']['rel_rmse']:.10e} "
                f"{row['tau_fixed_tc']['tau']:.10e} {row['tau_fixed_tc']['r']:.10e} {row['tau_fixed_tc']['rel_rmse']:.10e} "
                f"{row['tau_free_tc']['tau']:.10e} {row['tau_free_tc']['tc']:.10e} {row['tau_free_tc']['r']:.10e} {row['tau_free_tc']['rel_rmse']:.10e}\n"
            )
        f.write("\n")
        for row in rows:
            f.write(f"# H*={row['hstar']:g} free_tc F_inf(theta)\n")
            f.write("# theta0 F_inf\n")
            for th0, finf in zip(row["theta_u"], row["free_tc"]["F_inf"]):
                f.write(f"{th0:.10f} {finf:.10e}\n")
            f.write("\n")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ode_ratio = load_ode_ratio(args.ode_xi, args.fixed_vw, args.target_h)
    ode_nopt = load_ode_nopt(args.ode_nopt)
    ode_data = build_ode_dataset(ode_ratio, ode_nopt)

    tc_grid = np.arange(args.tc_min, args.tc_max + 0.5 * args.tc_step, args.tc_step, dtype=np.float64)
    tau_grid = np.arange(args.tau_min, args.tau_max + 0.5 * args.tau_step, args.tau_step, dtype=np.float64)

    rows = []
    for hstar in args.target_h:
        data_h = select_h(ode_data, hstar)
        theta_u, theta_index, f_theta = build_theta_arrays(data_h)
        baseline = optimize_r_baseline(data_h, theta_index, f_theta, args.fixed_s, args.fixed_tc, args.fixed_c, args.r_min, args.r_max)
        tc_prof = profile_tc(data_h, args.fixed_s, args.fixed_c, tc_grid, args.r_min, args.r_max)
        tau_prof = profile_tau_fixed_tc(data_h, args.fixed_s, args.fixed_tc, args.fixed_c, tau_grid, args.r_min, args.r_max)
        tau_fit = optimize_tau_r(
            data_h,
            theta_index,
            f_theta,
            args.fixed_s,
            args.fixed_tc,
            args.fixed_c,
            (args.tau_min, args.tau_max),
            (args.r_min, args.r_max),
        )
        tau_tc_fit = optimize_tau_r_tc(
            data_h,
            theta_index,
            f_theta,
            args.fixed_s,
            args.fixed_c,
            (args.tau_min, args.tau_max),
            (args.r_min, args.r_max),
            (args.tc_min, args.tc_max),
        )

        candidates = {
            "baseline": baseline,
            "free_tc": tc_prof["best"],
            "tau_fixed_tc": tau_fit,
            "tau_free_tc": tau_tc_fit,
        }
        best_name = min(candidates, key=lambda key: candidates[key]["rel_rmse"])
        best_rec = candidates[best_name]

        rows.append(
            {
                "hstar": float(hstar),
                "theta_u": theta_u,
                "theta_index": theta_index,
                "f_theta": f_theta,
                "baseline": baseline,
                "tc_profile": tc_prof,
                "free_tc": tc_prof["best"],
                "tau_profile": tau_prof,
                "tau_fixed_tc": tau_fit,
                "tau_free_tc": tau_tc_fit,
                "best_name": best_name,
            }
        )

        htag = f"H{float(hstar):.1f}".replace(".", "p")
        make_overlay_plot(
            data_h,
            theta_u,
            theta_index,
            f_theta,
            baseline,
            best_rec,
            outdir / f"best_overlay_{htag}.png",
            args.dpi,
            rf"ODE transition fit at $H_*={float(hstar):g}$: best = {best_name}",
        )

    make_relrmse_summary_plot(rows, args.fixed_tc, outdir / "ode_transition_relrmse_vs_h.png", args.dpi)
    make_param_summary_plot(rows, args.fixed_tc, outdir / "ode_transition_params_vs_h.png", args.dpi)
    make_tc_profile_plot(rows, outdir / "ode_transition_tc_profiles.png", args.dpi)
    make_tau_profile_plot(rows, outdir / "ode_transition_tau_profiles.png", args.dpi)
    save_summary(outdir / "ode_transition_summary.txt", args, rows)
    print(outdir / "ode_transition_summary.txt")


if __name__ == "__main__":
    main()
