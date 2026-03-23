import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ode.analysis.analyze_modelB_degeneracy_checks import (
    build_theta_arrays,
    optimize_r_for_fixed_sc_tc,
    rel_rmse,
    score_from_globals,
)
from ode.analysis.plot_lattice_y3_from_rho import (
    TARGET_H_DEFAULT,
    build_lattice_dataset,
    build_lattice_fanh_lookup,
    build_ode_dataset,
    fit_lattice_nopt_scale,
    load_lattice_ratio,
    load_lattice_rho,
    load_ode_nopt,
    load_ode_ratio,
    select_h,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Profile a shared H*-independent r for Model B with c=1 and fixed tc, "
            "allowing separate s(H_*) and free per-theta F_inf."
        )
    )
    p.add_argument("--lattice-xi", type=str, default="lattice_data/data/energy_ratio_by_theta_data_v9.txt")
    p.add_argument("--lattice-rho", type=str, default="lattice_data/data/rho_noPT_data.txt")
    p.add_argument("--ode-xi", type=str, default="ode/xi_DM_ODE_results.txt")
    p.add_argument("--ode-nopt", type=str, default="ode/analysis/results/noPT_reference_dm_tp_fitready_H1p000.txt")
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--target-h", type=float, nargs="+", default=TARGET_H_DEFAULT)
    p.add_argument("--fixed-tc", type=float, default=1.5)
    p.add_argument("--fixed-c", type=float, default=1.0)
    p.add_argument("--r-min", type=float, default=1.0)
    p.add_argument("--r-max", type=float, default=4.0)
    p.add_argument("--r-step", type=float, default=0.02)
    p.add_argument("--s-min", type=float, default=0.4)
    p.add_argument("--s-max", type=float, default=1.2)
    p.add_argument(
        "--outdir",
        type=str,
        default="ode/analysis/results/lattice_fit/modelB_shared_r_v9",
    )
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def optimize_s_for_fixed_r(data_h, theta_index, f_theta, r, tc, c, s_min, s_max):
    def objective(s):
        rec = score_from_globals(data_h, theta_index, f_theta, float(s), float(r), float(tc), float(c))
        return rec["chi2_rel"]

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
    rec = score_from_globals(data_h, theta_index, f_theta, float(res.x), float(r), float(tc), float(c))
    rec["success"] = bool(res.success)
    return rec


def profile_shared_r(datasets, tc, c, r_grid, s_min, s_max):
    rows = []
    best = None
    for r in r_grid:
        per_h = {}
        total_chi2 = 0.0
        total_n = 0
        all_xi = []
        all_fit = []
        for hstar, data_h in datasets.items():
            theta_u, theta_index, f_theta = build_theta_arrays(data_h)
            rec = optimize_s_for_fixed_r(data_h, theta_index, f_theta, float(r), tc, c, s_min, s_max)
            per_h[float(hstar)] = {
                "theta_u": theta_u,
                "theta_index": theta_index,
                "f_theta": f_theta,
                **rec,
            }
            total_chi2 += rec["chi2_rel"]
            total_n += len(data_h["xi"])
            all_xi.append(data_h["xi"])
            all_fit.append(rec["xi_fit"])
        combined_rel = rel_rmse(np.concatenate(all_xi), np.concatenate(all_fit))
        row = {
            "r": float(r),
            "per_h": per_h,
            "chi2_rel": float(total_chi2),
            "chi2_rel_red": float(total_chi2 / max(total_n - (1 + len(datasets)), 1)),
            "rel_rmse": combined_rel,
        }
        rows.append(row)
        if best is None or row["rel_rmse"] < best["rel_rmse"]:
            best = row
    return {"rows": rows, "best": best}


def make_profile_plot(profile, out_path, dpi, title):
    rows = profile["rows"]
    r = np.array([row["r"] for row in rows], dtype=np.float64)
    rel = np.array([row["rel_rmse"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(r, rel, color="tab:blue", lw=2.0)
    ax.plot(profile["best"]["r"], profile["best"]["rel_rmse"], "o", color="tab:red", ms=6)
    ax.set_xlabel(r"shared $r$")
    ax.set_ylabel("combined rel-RMSE")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_s_plot(profile, target_h, out_path, dpi, title):
    rows = profile["rows"]
    r = np.array([row["r"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    colors = {1.5: "tab:blue", 2.0: "tab:red"}
    for hstar in target_h:
        sval = np.array([row["per_h"][float(hstar)]["s"] for row in rows], dtype=np.float64)
        ax.plot(r, sval, lw=2.0, color=colors.get(float(hstar), None), label=rf"$H_*={float(hstar):g}$")
    ax.set_xlabel(r"shared $r$")
    ax.set_ylabel(r"best $s(H_*)$")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_summary(path, args, profile):
    with open(path, "w") as f:
        f.write("# Model B shared-r profile\n")
        f.write(f"# fixed_tc={float(args.fixed_tc):.6f}\n")
        f.write(f"# fixed_c={float(args.fixed_c):.6f}\n")
        f.write("# shared r across H*=1.5 and 2.0; s(H_*) free, F_inf(theta;H_*) free\n\n")
        f.write("# r rel_rmse_combined chi2_rel_combined chi2_rel_red s_H1p5 rel_H1p5 s_H2p0 rel_H2p0\n")
        for row in profile["rows"]:
            rec15 = row["per_h"].get(1.5)
            rec20 = row["per_h"].get(2.0)
            f.write(
                f"{row['r']:.10e} {row['rel_rmse']:.10e} {row['chi2_rel']:.10e} {row['chi2_rel_red']:.10e} "
                f"{rec15['s']:.10e} {rec15['rel_rmse']:.10e} {rec20['s']:.10e} {rec20['rel_rmse']:.10e}\n"
            )
        f.write("\n")
        best = profile["best"]
        f.write("# best shared-r solution\n")
        f.write(
            f"# r_best={best['r']:.10e} rel_rmse_combined={best['rel_rmse']:.10e} "
            f"chi2_rel_combined={best['chi2_rel']:.10e}\n"
        )
        for hstar, rec in best["per_h"].items():
            f.write(f"# H*={hstar:g} s_best={rec['s']:.10e} rel_rmse={rec['rel_rmse']:.10e}\n")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ratio_lat = load_lattice_ratio(args.lattice_xi, args.target_h)
    rho_lat = load_lattice_rho(args.lattice_rho)
    ode_ratio = load_ode_ratio(args.ode_xi, args.fixed_vw, args.target_h)
    ode_nopt = load_ode_nopt(args.ode_nopt)

    scale_fit = fit_lattice_nopt_scale(rho_lat, ode_nopt)
    rho_lookup = build_lattice_fanh_lookup(rho_lat, scale_fit["scale"])
    lat_data = build_lattice_dataset(ratio_lat, rho_lookup, args.fixed_vw)
    _ = build_ode_dataset(ode_ratio, ode_nopt)

    datasets = {float(h): select_h(lat_data, h) for h in args.target_h}
    r_grid = np.arange(args.r_min, args.r_max + 0.5 * args.r_step, args.r_step, dtype=np.float64)
    profile = profile_shared_r(datasets, args.fixed_tc, args.fixed_c, r_grid, args.s_min, args.s_max)

    make_profile_plot(
        profile,
        outdir / "shared_r_relrmse.png",
        args.dpi,
        rf"Shared-$r$ profile for Model B with $c={float(args.fixed_c):g}$, $t_c={float(args.fixed_tc):g}$",
    )
    make_s_plot(
        profile,
        args.target_h,
        outdir / "shared_r_best_s.png",
        args.dpi,
        rf"Best $s(H_*)$ along the shared-$r$ profile",
    )
    save_summary(outdir / "shared_r_summary.txt", args, profile)
    print(outdir / "shared_r_summary.txt")


if __name__ == "__main__":
    main()
