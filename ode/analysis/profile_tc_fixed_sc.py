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
    optimize_r_for_fixed_sc_tc,
    score_from_globals,
)
from ode.analysis.plot_lattice_y3_from_rho import (
    TARGET_H_DEFAULT,
    build_lattice_dataset,
    build_lattice_fanh_lookup,
    fit_lattice_nopt_scale,
    load_lattice_ratio,
    load_lattice_rho,
    load_ode_nopt,
    select_h,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Profile t_c for Model B on lattice data with s=1 and c=1 fixed, "
            "fitting only r and per-theta F_inf."
        )
    )
    p.add_argument("--lattice-xi", type=str, default="lattice_data/data/energy_ratio_by_theta_data_v9.txt")
    p.add_argument("--lattice-rho", type=str, default="lattice_data/data/rho_noPT_data.txt")
    p.add_argument("--ode-nopt", type=str, default="ode/analysis/results/noPT_reference_dm_tp_fitready_H1p000.txt")
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--target-h", type=float, nargs="+", default=TARGET_H_DEFAULT)
    p.add_argument("--fixed-s", type=float, default=1.0)
    p.add_argument("--fixed-c", type=float, default=1.0)
    p.add_argument("--tosc", type=float, default=1.5)
    p.add_argument("--tc-min", type=float, default=0.5)
    p.add_argument("--tc-max", type=float, default=3.0)
    p.add_argument("--tc-step", type=float, default=0.02)
    p.add_argument("--r-min", type=float, default=0.5)
    p.add_argument("--r-max", type=float, default=4.0)
    p.add_argument("--r-step", type=float, default=0.05)
    p.add_argument(
        "--outdir",
        type=str,
        default="ode/analysis/results/lattice_fit/tc_profile_fixed_sc_v9",
    )
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def profile_tc(data_h, fixed_s, fixed_c, tc_grid, r_min, r_max):
    theta_u, theta_index, f_theta = build_theta_arrays(data_h)
    rows = [
        optimize_r_for_fixed_sc_tc(data_h, theta_index, f_theta, float(fixed_s), float(fixed_c), float(tc), float(r_min), float(r_max))
        for tc in tc_grid
    ]
    rel = np.array([row["rel_rmse"] for row in rows], dtype=np.float64)
    ibest = int(np.argmin(rel))
    return {
        "theta_u": theta_u,
        "theta_index": theta_index,
        "f_theta": f_theta,
        "rows": rows,
        "best": rows[ibest],
    }


def scan_tc_r(data_h, fixed_s, fixed_c, tc_grid, r_grid):
    theta_u, theta_index, f_theta = build_theta_arrays(data_h)
    rmse = np.zeros((len(r_grid), len(tc_grid)), dtype=np.float64)
    best = None
    for ir, r in enumerate(r_grid):
        for it, tc in enumerate(tc_grid):
            rec = score_from_globals(data_h, theta_index, f_theta, float(fixed_s), float(r), float(tc), float(fixed_c))
            rmse[ir, it] = rec["rel_rmse"]
            if best is None or rec["rel_rmse"] < best["rel_rmse"]:
                best = rec
    return {
        "theta_u": theta_u,
        "theta_index": theta_index,
        "f_theta": f_theta,
        "tc_grid": tc_grid,
        "r_grid": r_grid,
        "rmse_grid": rmse,
        "best": best,
    }


def interval_band(x, y, frac=1.01):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = y <= float(frac) * float(np.min(y))
    return float(np.min(x[mask])), float(np.max(x[mask]))


def make_tc_profile_plot(profiles, tosc, out_path, dpi):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    colors = {1.5: "tab:blue", 2.0: "tab:red"}
    for hstar, prof in profiles.items():
        tc = np.array([row["tc"] for row in prof["rows"]], dtype=np.float64)
        rel = np.array([row["rel_rmse"] for row in prof["rows"]], dtype=np.float64)
        color = colors.get(float(hstar), None)
        ax.plot(tc, rel, lw=2.0, color=color, label=rf"$H_*={float(hstar):g}$")
        ax.plot(prof["best"]["tc"], prof["best"]["rel_rmse"], "o", ms=6, color=color)
    ax.axvline(float(tosc), color="black", lw=1.0, ls="--", alpha=0.7, label=rf"$t_{{osc}}={float(tosc):.2f}$")
    ax.set_xlabel(r"$t_c$")
    ax.set_ylabel("rel-RMSE")
    ax.set_title(r"Model B with fixed $s=1$, $c=1$: profile over $t_c$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_r_profile_plot(profiles, tosc, out_path, dpi):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    colors = {1.5: "tab:blue", 2.0: "tab:red"}
    for hstar, prof in profiles.items():
        tc = np.array([row["tc"] for row in prof["rows"]], dtype=np.float64)
        r = np.array([row["r"] for row in prof["rows"]], dtype=np.float64)
        color = colors.get(float(hstar), None)
        ax.plot(tc, r, lw=2.0, color=color, label=rf"$H_*={float(hstar):g}$")
    ax.axvline(float(tosc), color="black", lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel(r"$t_c$")
    ax.set_ylabel(r"best-fit $r$")
    ax.set_title(r"Best-fit $r(t_c)$ with fixed $s=1$, $c=1$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_tc_r_contour(scan, tosc, out_path, dpi, title):
    tc_grid = scan["tc_grid"]
    r_grid = scan["r_grid"]
    rmse = scan["rmse_grid"]
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    vmax = float(np.percentile(rmse, 92.0))
    levels = np.linspace(float(np.min(rmse)), vmax, 16)
    cs = ax.contourf(tc_grid, r_grid, rmse, levels=levels, cmap="viridis")
    ax.contour(tc_grid, r_grid, rmse, levels=levels[::2], colors="white", linewidths=0.5, alpha=0.55)
    ax.plot(scan["best"]["tc"], scan["best"]["r"], "wo", ms=6, mec="black", mew=0.8, label="best")
    ax.axvline(float(tosc), color="tab:red", lw=1.2, ls="--", label=rf"$t_{{osc}}={float(tosc):.2f}$")
    ax.set_xlabel(r"$t_c$")
    ax.set_ylabel(r"$r$")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.colorbar(cs, ax=ax, pad=0.02, label="rel-RMSE")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_profile_table(path, profile):
    with open(path, "w") as f:
        f.write("# tc r rel_rmse chi2_rel\n")
        for row in profile["rows"]:
            f.write(f"{row['tc']:.10e} {row['r']:.10e} {row['rel_rmse']:.10e} {row['chi2_rel']:.10e}\n")


def save_summary(path, args, profiles):
    with open(path, "w") as f:
        f.write("# t_c profile for Model B with s=1 and c=1 fixed\n")
        f.write(f"# lattice_xi={Path(args.lattice_xi).resolve()}\n")
        f.write(f"# lattice_rho={Path(args.lattice_rho).resolve()}\n")
        f.write(f"# ode_nopt={Path(args.ode_nopt).resolve()}\n")
        f.write(f"# fixed_vw={float(args.fixed_vw):.6f}\n")
        f.write(f"# fixed_s={float(args.fixed_s):.6f}\n")
        f.write(f"# fixed_c={float(args.fixed_c):.6f}\n")
        f.write(f"# t_osc_reference={float(args.tosc):.6f}\n")
        f.write("# hstar tc_best r_best rel_rmse_best tc_1pct_lo tc_1pct_hi\n")
        for hstar in sorted(profiles):
            prof = profiles[hstar]
            tc = np.array([row["tc"] for row in prof["rows"]], dtype=np.float64)
            rel = np.array([row["rel_rmse"] for row in prof["rows"]], dtype=np.float64)
            lo, hi = interval_band(tc, rel, frac=1.01)
            best = prof["best"]
            f.write(
                f"{float(hstar):.8g} {best['tc']:.10e} {best['r']:.10e} {best['rel_rmse']:.10e} "
                f"{lo:.10e} {hi:.10e}\n"
            )


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ratio_lat = load_lattice_ratio(args.lattice_xi, args.target_h)
    rho_lat = load_lattice_rho(args.lattice_rho)
    ode_nopt = load_ode_nopt(args.ode_nopt)

    scale_fit = fit_lattice_nopt_scale(rho_lat, ode_nopt)
    rho_lookup = build_lattice_fanh_lookup(rho_lat, scale_fit["scale"])
    lat_data = build_lattice_dataset(ratio_lat, rho_lookup, args.fixed_vw)

    tc_grid = np.arange(args.tc_min, args.tc_max + 0.5 * args.tc_step, args.tc_step, dtype=np.float64)
    r_grid = np.arange(args.r_min, args.r_max + 0.5 * args.r_step, args.r_step, dtype=np.float64)

    profiles = {}
    for hstar in args.target_h:
        hval = float(hstar)
        data_h = select_h(lat_data, hval)
        prof = profile_tc(data_h, args.fixed_s, args.fixed_c, tc_grid, args.r_min, args.r_max)
        profiles[hval] = prof
        hdir = outdir / f"H{hval:.1f}".replace(".", "p")
        hdir.mkdir(parents=True, exist_ok=True)
        save_profile_table(hdir / "tc_profile_table.txt", prof)
        scan = scan_tc_r(data_h, args.fixed_s, args.fixed_c, tc_grid, r_grid)
        make_tc_r_contour(
            scan,
            args.tosc,
            hdir / "tc_r_contour.png",
            args.dpi,
            rf"Model B rel-RMSE$(t_c,r)$ at $H_*={hval:g}$ with $s=1$, $c=1$",
        )

    make_tc_profile_plot(profiles, args.tosc, outdir / "tc_profile_relrmse.png", args.dpi)
    make_r_profile_plot(profiles, args.tosc, outdir / "tc_profile_r.png", args.dpi)
    save_summary(outdir / "tc_profile_summary.txt", args, profiles)
    print(outdir / "tc_profile_summary.txt")


if __name__ == "__main__":
    main()
