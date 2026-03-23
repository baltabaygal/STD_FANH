import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ode.analysis.analyze_modelB_degeneracy_checks import (
    build_theta_arrays,
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
            "Profile one shared s across H* slices for Model B with fixed r, c, and tc, "
            "while fitting per-theta F_inf separately in each H* slice."
        )
    )
    p.add_argument("--lattice-xi", type=str, default="lattice_data/data/energy_ratio_by_theta_data_v9.txt")
    p.add_argument("--lattice-rho", type=str, default="lattice_data/data/rho_noPT_data.txt")
    p.add_argument("--ode-xi", type=str, default="ode/xi_DM_ODE_results.txt")
    p.add_argument("--ode-nopt", type=str, default="ode/analysis/results/noPT_reference_dm_tp_fitready_H1p000.txt")
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--target-h", type=float, nargs="+", default=TARGET_H_DEFAULT)
    p.add_argument("--fixed-r", type=float, default=2.2)
    p.add_argument("--fixed-c", type=float, default=1.0)
    p.add_argument("--fixed-tc", type=float, default=1.5)
    p.add_argument("--s-min", type=float, default=0.4)
    p.add_argument("--s-max", type=float, default=1.2)
    p.add_argument("--s-step", type=float, default=0.01)
    p.add_argument(
        "--outdir",
        type=str,
        default="ode/analysis/results/lattice_fit/modelB_shared_s_fixedr_v9",
    )
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def profile_shared_s(datasets, s_grid, fixed_r, fixed_c, fixed_tc):
    rows = []
    best = None
    for s in s_grid:
        per_h = {}
        all_xi = []
        all_fit = []
        total_chi2 = 0.0
        total_n = 0
        for hstar, data_h in datasets.items():
            theta_u, theta_index, f_theta = build_theta_arrays(data_h)
            rec = score_from_globals(data_h, theta_index, f_theta, float(s), fixed_r, fixed_tc, fixed_c)
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
        row = {
            "s": float(s),
            "per_h": per_h,
            "rel_rmse": rel_rmse(np.concatenate(all_xi), np.concatenate(all_fit)),
            "chi2_rel": float(total_chi2),
            "chi2_rel_red": float(total_chi2 / max(total_n - 1, 1)),
        }
        rows.append(row)
        if best is None or row["rel_rmse"] < best["rel_rmse"]:
            best = row
    return {"rows": rows, "best": best}


def make_profile_plot(profile, out_path, dpi, title):
    s = np.array([row["s"] for row in profile["rows"]], dtype=np.float64)
    rel = np.array([row["rel_rmse"] for row in profile["rows"]], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(s, rel, color="tab:blue", lw=2.0)
    ax.plot(profile["best"]["s"], profile["best"]["rel_rmse"], "o", color="tab:red", ms=6)
    ax.set_xlabel(r"shared $s$")
    ax.set_ylabel("combined rel-RMSE")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_summary(path, args, profile):
    with open(path, "w") as f:
        f.write("# Model B shared-s profile with fixed r\n")
        f.write(f"# fixed_r={float(args.fixed_r):.6f}\n")
        f.write(f"# fixed_c={float(args.fixed_c):.6f}\n")
        f.write(f"# fixed_tc={float(args.fixed_tc):.6f}\n")
        f.write("# shared s across H*=1.5 and 2.0; F_inf(theta;H_*) free per slice\n\n")
        f.write("# s rel_rmse_combined chi2_rel_combined chi2_rel_red rel_H1p5 rel_H2p0\n")
        for row in profile["rows"]:
            rec15 = row["per_h"].get(1.5)
            rec20 = row["per_h"].get(2.0)
            f.write(
                f"{row['s']:.10e} {row['rel_rmse']:.10e} {row['chi2_rel']:.10e} {row['chi2_rel_red']:.10e} "
                f"{rec15['rel_rmse']:.10e} {rec20['rel_rmse']:.10e}\n"
            )
        f.write("\n")
        best = profile["best"]
        f.write("# best shared-s solution\n")
        f.write(
            f"# s_best={best['s']:.10e} rel_rmse_combined={best['rel_rmse']:.10e} "
            f"chi2_rel_combined={best['chi2_rel']:.10e}\n"
        )
        for hstar, rec in best["per_h"].items():
            f.write(f"# H*={hstar:g} rel_rmse={rec['rel_rmse']:.10e}\n")
            f.write("# theta0 F_inf_best\n")
            for th0, finf in zip(rec["theta_u"], rec["F_inf"]):
                f.write(f"{th0:.10f} {finf:.10e}\n")
            f.write("\n")


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
    s_grid = np.arange(args.s_min, args.s_max + 0.5 * args.s_step, args.s_step, dtype=np.float64)
    profile = profile_shared_s(datasets, s_grid, args.fixed_r, args.fixed_c, args.fixed_tc)

    make_profile_plot(
        profile,
        outdir / "shared_s_profile.png",
        args.dpi,
        rf"Shared-$s$ profile for Model B with $r={float(args.fixed_r):g}$, $c={float(args.fixed_c):g}$, $t_c={float(args.fixed_tc):g}$",
    )
    save_summary(outdir / "shared_s_summary.txt", args, profile)
    print(outdir / "shared_s_summary.txt")


if __name__ == "__main__":
    main()
