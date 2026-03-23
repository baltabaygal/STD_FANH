import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ode.analysis.analyze_modelB_degeneracy_checks import build_theta_arrays, rel_rmse, score_from_globals
from ode.analysis.plot_lattice_y3_from_rho import build_ode_dataset, load_ode_nopt, load_ode_ratio, select_h


TARGET_H_DEFAULT = [0.5, 1.0, 1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Test the reduced Model B ansatz on direct ODE data across H*. "
            "Uses fixed global parameters and fits only F_inf(theta) per H* slice."
        )
    )
    p.add_argument("--ode-xi", type=str, default="ode/xi_DM_ODE_results.txt")
    p.add_argument("--ode-nopt", type=str, default="ode/analysis/results/noPT_reference_dm_tp_fitready_H1p000.txt")
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--target-h", type=float, nargs="+", default=TARGET_H_DEFAULT)
    p.add_argument("--fixed-r", type=float, default=2.2)
    p.add_argument("--fixed-s", type=float, default=0.65)
    p.add_argument("--fixed-c", type=float, default=1.0)
    p.add_argument("--fixed-tc", type=float, default=1.5)
    p.add_argument("--s-min", type=float, default=0.4)
    p.add_argument("--s-max", type=float, default=1.2)
    p.add_argument(
        "--outdir",
        type=str,
        default="ode/analysis/results/ode_reduced_modelB_test_vw0p9",
    )
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def fit_best_s(data_h, theta_index, f_theta, fixed_r, fixed_c, fixed_tc, s_min, s_max):
    def objective(s):
        return score_from_globals(data_h, theta_index, f_theta, float(s), fixed_r, fixed_tc, fixed_c)["chi2_rel"]

    probe_s = np.linspace(s_min, s_max, 81, dtype=np.float64)
    vals = np.array([objective(s) for s in probe_s], dtype=np.float64)
    idx = int(np.argmin(vals))
    if idx == 0:
        a, b = probe_s[0], probe_s[1]
    elif idx == len(probe_s) - 1:
        a, b = probe_s[-2], probe_s[-1]
    else:
        a, b = probe_s[idx - 1], probe_s[idx + 1]
    res = minimize_scalar(objective, bounds=(float(a), float(b)), method="bounded")
    rec = score_from_globals(data_h, theta_index, f_theta, float(res.x), fixed_r, fixed_tc, fixed_c)
    rec["success"] = bool(res.success)
    return rec


def y3_from_record(data_h, theta_index, f_theta, rec):
    frow = f_theta[theta_index]
    return rec["xi_fit"] * np.square(frow) / np.power(data_h["tp"], 1.5)


def make_overlay_plot(data_h, theta_u, theta_index, f_theta, fixed_rec, out_path, dpi, title):
    ntheta = len(theta_u)
    fig, axes = plt.subplots(
        2,
        ntheta,
        figsize=(3.8 * ntheta, 6.2),
        squeeze=False,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        sharex="col",
    )
    y3_model = y3_from_record(data_h, theta_index, f_theta, fixed_rec)
    y3_data = data_h["y3"]
    rel_resid = fixed_rec["rel_resid"]

    for j, th0 in enumerate(theta_u):
        mask = np.isclose(data_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
        order = np.argsort(data_h["tp"][mask])
        tp = data_h["tp"][mask][order]
        y3 = y3_data[mask][order]
        y3m = y3_model[mask][order]
        resid = rel_resid[mask][order]

        ax = axes[0, j]
        ax.plot(tp, y3, "ko", ms=3.4, label="ODE")
        ax.plot(tp, y3m, color="tab:red", lw=1.8, label="model")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$Y_3$")
        ax.set_title(rf"$\theta_0={th0:.3g}$")
        ax.grid(alpha=0.25)

        ax = axes[1, j]
        ax.axhline(0.0, color="black", lw=1.0)
        ax.plot(tp, resid, "o-", color="tab:red", ms=3.5, lw=1.2)
        ax.set_xscale("log")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$\Delta\xi/\xi$")
        ax.grid(alpha=0.25)

    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_summary_plot(rows, out_path, dpi, title):
    h = np.array([row["hstar"] for row in rows], dtype=np.float64)
    rel_fixed = np.array([row["fixed"]["rel_rmse"] for row in rows], dtype=np.float64)
    rel_best = np.array([row["best_s"]["rel_rmse"] for row in rows], dtype=np.float64)
    s_best = np.array([row["best_s"]["s"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.4), sharex=True)
    axes[0].plot(h, rel_fixed, "o-", color="tab:red", lw=1.8, ms=5.0, label=rf"fixed $s={rows[0]['fixed']['s']:.2f}$")
    axes[0].plot(h, rel_best, "o--", color="tab:blue", lw=1.6, ms=4.8, label=r"best $s(H_*)$")
    axes[0].set_ylabel("rel-RMSE")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(h, s_best, "o-", color="tab:blue", lw=1.8, ms=5.0)
    axes[1].axhline(rows[0]["fixed"]["s"], color="black", lw=1.0, ls="--")
    axes[1].set_xlabel(r"$H_*$")
    axes[1].set_ylabel(r"best $s$")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_summary(path, args, rows):
    with open(path, "w") as f:
        f.write("# Reduced Model B test on ODE data\n")
        f.write(f"# ode_xi={Path(args.ode_xi).resolve()}\n")
        f.write(f"# ode_nopt={Path(args.ode_nopt).resolve()}\n")
        f.write(f"# fixed_vw={float(args.fixed_vw):.6f}\n")
        f.write(
            f"# fixed globals: s={float(args.fixed_s):.6f} r={float(args.fixed_r):.6f} "
            f"c={float(args.fixed_c):.6f} tc={float(args.fixed_tc):.6f}\n"
        )
        f.write("# only F_inf(theta) is fit per H* slice for the fixed-parameter test\n")
        f.write("# reference diagnostic: best s(H*) with r,c,tc fixed\n\n")
        f.write("# hstar rel_rmse_fixed rel_rmse_bests best_s_fixedr chi2_fixed chi2_bests\n")
        for row in rows:
            f.write(
                f"{row['hstar']:.8g} {row['fixed']['rel_rmse']:.10e} {row['best_s']['rel_rmse']:.10e} "
                f"{row['best_s']['s']:.10e} {row['fixed']['chi2_rel']:.10e} {row['best_s']['chi2_rel']:.10e}\n"
            )
        f.write("\n")
        for row in rows:
            f.write(f"# H*={row['hstar']:g} fixed F_inf(theta)\n")
            f.write("# theta0 F_inf_fixed\n")
            for th0, finf in zip(row["theta_u"], row["fixed"]["F_inf"]):
                f.write(f"{th0:.10f} {finf:.10e}\n")
            f.write("\n")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ode_ratio = load_ode_ratio(args.ode_xi, args.fixed_vw, args.target_h)
    ode_nopt = load_ode_nopt(args.ode_nopt)
    ode_data = build_ode_dataset(ode_ratio, ode_nopt)

    rows = []
    for hstar in args.target_h:
        data_h = select_h(ode_data, hstar)
        theta_u, theta_index, f_theta = build_theta_arrays(data_h)
        fixed_rec = score_from_globals(data_h, theta_index, f_theta, args.fixed_s, args.fixed_r, args.fixed_tc, args.fixed_c)
        best_s_rec = fit_best_s(data_h, theta_index, f_theta, args.fixed_r, args.fixed_c, args.fixed_tc, args.s_min, args.s_max)
        rows.append(
            {
                "hstar": float(hstar),
                "theta_u": theta_u,
                "theta_index": theta_index,
                "f_theta": f_theta,
                "fixed": fixed_rec,
                "best_s": best_s_rec,
            }
        )
        htag = f"H{float(hstar):.1f}".replace(".", "p")
        make_overlay_plot(
            data_h,
            theta_u,
            theta_index,
            f_theta,
            fixed_rec,
            outdir / f"reduced_modelB_fixed_overlay_{htag}.png",
            args.dpi,
            rf"Reduced Model B on ODE: $H_*={float(hstar):g}$, $s={float(args.fixed_s):.2f}$, $r={float(args.fixed_r):.2f}$",
        )

    make_summary_plot(
        rows,
        outdir / "reduced_modelB_ode_summary.png",
        args.dpi,
        rf"Reduced Model B on ODE, $v_w={float(args.fixed_vw):g}$",
    )
    save_summary(outdir / "reduced_modelB_ode_summary.txt", args, rows)
    print(outdir / "reduced_modelB_ode_summary.txt")


if __name__ == "__main__":
    main()
