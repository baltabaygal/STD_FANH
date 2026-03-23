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
            "Test the ODE fit quality of a reduced Model B ansatz with s_eff = a * H*, "
            "and fixed r, c, tc."
        )
    )
    p.add_argument("--ode-xi", type=str, default="ode/xi_DM_ODE_results.txt")
    p.add_argument("--ode-nopt", type=str, default="ode/analysis/results/noPT_reference_dm_tp_fitready_H1p000.txt")
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--target-h", type=float, nargs="+", default=TARGET_H_DEFAULT)
    p.add_argument("--fixed-r", type=float, default=2.2)
    p.add_argument("--fixed-c", type=float, default=1.0)
    p.add_argument("--fixed-tc", type=float, default=1.5)
    p.add_argument("--a-min", type=float, default=0.1)
    p.add_argument("--a-max", type=float, default=2.0)
    p.add_argument("--a-step", type=float, default=0.01)
    p.add_argument(
        "--outdir",
        type=str,
        default="ode/analysis/results/ode_modelB_s_times_H_vw0p9",
    )
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def evaluate_a(datasets, coeff, fixed_r, fixed_c, fixed_tc):
    all_xi = []
    all_fit = []
    per_h = {}
    total_chi2 = 0.0
    total_n = 0
    for hstar, data_h in datasets.items():
        theta_u, theta_index, f_theta = build_theta_arrays(data_h)
        rec = score_from_globals(data_h, theta_index, f_theta, coeff * float(hstar), fixed_r, fixed_tc, fixed_c)
        per_h[float(hstar)] = {
            "theta_u": theta_u,
            "theta_index": theta_index,
            "f_theta": f_theta,
            **rec,
        }
        all_xi.append(data_h["xi"])
        all_fit.append(rec["xi_fit"])
        total_chi2 += rec["chi2_rel"]
        total_n += len(data_h["xi"])
    return {
        "a": float(coeff),
        "per_h": per_h,
        "rel_rmse": rel_rmse(np.concatenate(all_xi), np.concatenate(all_fit)),
        "chi2_rel": float(total_chi2),
        "chi2_rel_red": float(total_chi2 / max(total_n - 1, 1)),
    }


def make_profile_plot(rows, out_path, dpi, title):
    a = np.array([row["a"] for row in rows], dtype=np.float64)
    combined = np.array([row["rel_rmse"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(a, combined, color="black", lw=2.2, label="combined")
    colors = {0.5: "tab:green", 1.0: "tab:orange", 1.5: "tab:blue", 2.0: "tab:red"}
    for hstar in sorted(rows[0]["per_h"]):
        rel = np.array([row["per_h"][hstar]["rel_rmse"] for row in rows], dtype=np.float64)
        ax.plot(a, rel, color=colors.get(float(hstar), None), lw=1.6, label=rf"$H_*={float(hstar):g}$")
    best = rows[int(np.argmin(combined))]
    ax.plot(best["a"], best["rel_rmse"], "ko", ms=6)
    ax.set_xlabel(r"$a$ in $s_{\rm eff}=aH_*$")
    ax.set_ylabel("rel-RMSE")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_summary(path, args, rows, best):
    with open(path, "w") as f:
        f.write("# ODE test for Model B with s_eff = a * H*\n")
        f.write(f"# ode_xi={Path(args.ode_xi).resolve()}\n")
        f.write(f"# ode_nopt={Path(args.ode_nopt).resolve()}\n")
        f.write(f"# fixed_vw={float(args.fixed_vw):.6f}\n")
        f.write(
            f"# fixed globals: r={float(args.fixed_r):.6f} c={float(args.fixed_c):.6f} "
            f"tc={float(args.fixed_tc):.6f}\n"
        )
        f.write("# only F_inf(theta) is fit per H* slice\n\n")
        f.write("# a rel_rmse_combined chi2_rel_combined chi2_rel_red rel_H0p5 rel_H1p0 rel_H1p5 rel_H2p0\n")
        for row in rows:
            f.write(
                f"{row['a']:.10e} {row['rel_rmse']:.10e} {row['chi2_rel']:.10e} {row['chi2_rel_red']:.10e} "
                f"{row['per_h'][0.5]['rel_rmse']:.10e} {row['per_h'][1.0]['rel_rmse']:.10e} "
                f"{row['per_h'][1.5]['rel_rmse']:.10e} {row['per_h'][2.0]['rel_rmse']:.10e}\n"
            )
        f.write("\n")
        f.write("# best coefficient\n")
        f.write(f"# a_best={best['a']:.10e} rel_rmse_combined={best['rel_rmse']:.10e} chi2_rel_combined={best['chi2_rel']:.10e}\n")
        for hstar in sorted(best["per_h"]):
            rec = best["per_h"][hstar]
            f.write(f"# H*={hstar:g} s_eff={best['a']*hstar:.10e} rel_rmse={rec['rel_rmse']:.10e}\n")
            f.write("# theta0 F_inf_best\n")
            for th0, finf in zip(rec["theta_u"], rec["F_inf"]):
                f.write(f"{th0:.10f} {finf:.10e}\n")
            f.write("\n")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ode_ratio = load_ode_ratio(args.ode_xi, args.fixed_vw, args.target_h)
    ode_nopt = load_ode_nopt(args.ode_nopt)
    ode_data = build_ode_dataset(ode_ratio, ode_nopt)
    datasets = {float(h): select_h(ode_data, h) for h in args.target_h}

    a_grid = np.arange(args.a_min, args.a_max + 0.5 * args.a_step, args.a_step, dtype=np.float64)
    rows = [evaluate_a(datasets, float(a), args.fixed_r, args.fixed_c, args.fixed_tc) for a in a_grid]
    best = min(rows, key=lambda row: row["rel_rmse"])

    make_profile_plot(
        rows,
        outdir / "modelB_s_times_H_profile.png",
        args.dpi,
        rf"ODE test: Model B with $s_{{\rm eff}}=aH_*$, $r={float(args.fixed_r):g}$, $c={float(args.fixed_c):g}$, $t_c={float(args.fixed_tc):g}$",
    )
    save_summary(outdir / "modelB_s_times_H_summary.txt", args, rows, best)
    print(outdir / "modelB_s_times_H_summary.txt")


if __name__ == "__main__":
    main()
