import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
            "Profile Model B with c=1 and fixed t_c by scanning s, "
            "optimizing r and per-theta F_inf at each s."
        )
    )
    p.add_argument("--lattice-xi", type=str, default="lattice_data/data/energy_ratio_by_theta_data_v9.txt")
    p.add_argument("--lattice-rho", type=str, default="lattice_data/data/rho_noPT_data.txt")
    p.add_argument("--ode-xi", type=str, default="ode/xi_DM_ODE_results.txt")
    p.add_argument("--ode-nopt", type=str, default="ode/analysis/results/noPT_reference_dm_tp_fitready_H1p000.txt")
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--target-h", type=float, nargs="+", default=TARGET_H_DEFAULT)
    p.add_argument("--fixed-tc", type=float, default=1.5)
    p.add_argument("--s-min", type=float, default=0.8)
    p.add_argument("--s-max", type=float, default=2.5)
    p.add_argument("--s-step", type=float, default=0.01)
    p.add_argument("--r-min", type=float, default=0.5)
    p.add_argument("--r-max", type=float, default=4.0)
    p.add_argument(
        "--outdir",
        type=str,
        default="ode/analysis/results/lattice_fit/modelB_s_profile_v9",
    )
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def rel_rmse(y, yfit):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square((yfit - y) / y))))


def nearest_theta(values, theta0, atol=5.0e-4):
    values = np.asarray(values, dtype=np.float64)
    idx = int(np.argmin(np.abs(values - float(theta0))))
    if abs(values[idx] - float(theta0)) > atol:
        raise RuntimeError(f"No theta match for theta0={theta0:.10f}")
    return idx


def unique_theta(data_h):
    return np.array(sorted(np.unique(data_h["theta0"])), dtype=np.float64)


def build_theta_arrays(data_h):
    theta_u = unique_theta(data_h)
    theta_index = np.array([nearest_theta(theta_u, th) for th in data_h["theta0"]], dtype=np.int64)
    f_theta = np.array(
        [float(np.median(data_h["fanh_no"][theta_index == i])) for i in range(len(theta_u))],
        dtype=np.float64,
    )
    return theta_u, theta_index, f_theta


def solve_finf_per_theta(xi, tp, theta_index, f_theta, s, r, tc, lower=1.0e-6, upper=1.0e3):
    transient = 1.0 / (1.0 + np.power((s * tp) / tc, r))
    a = np.power(tp, 1.5) / np.square(f_theta[theta_index])
    finf = np.zeros(len(f_theta), dtype=np.float64)

    for i in range(len(f_theta)):
        mask = theta_index == i
        u = a[mask] / xi[mask]
        v = 1.0 - transient[mask] / xi[mask]
        denom = float(np.dot(u, u))
        if denom <= 0.0:
            finf_i = lower
        else:
            finf_i = float(np.dot(u, v) / denom)
        finf[i] = np.clip(finf_i, lower, upper)
    return finf


def xi_model(tp, theta_index, f_theta, finf, s, r, tc):
    plateau = np.power(tp, 1.5) * finf[theta_index] / np.square(f_theta[theta_index])
    transient = 1.0 / (1.0 + np.power((s * tp) / tc, r))
    return plateau + transient


def fit_r_for_s(data_h, theta_index, f_theta, s, tc, r_min, r_max):
    xi = data_h["xi"]
    tp = data_h["tp"]

    def objective(r):
        finf = solve_finf_per_theta(xi, tp, theta_index, f_theta, s, r, tc)
        xi_fit = xi_model(tp, theta_index, f_theta, finf, s, r, tc)
        return np.mean(np.square((xi_fit - xi) / xi))

    probe_r = np.linspace(r_min, r_max, 41, dtype=np.float64)
    probe_val = np.array([objective(r) for r in probe_r], dtype=np.float64)
    idx = int(np.argmin(probe_val))
    if idx == 0:
        a, b = probe_r[0], probe_r[1]
    elif idx == len(probe_r) - 1:
        a, b = probe_r[-2], probe_r[-1]
    else:
        a, b = probe_r[idx - 1], probe_r[idx + 1]
    res = minimize_scalar(objective, bounds=(float(a), float(b)), method="bounded")
    r_best = float(res.x)
    finf = solve_finf_per_theta(xi, tp, theta_index, f_theta, s, r_best, tc)
    xi_fit = xi_model(tp, theta_index, f_theta, finf, s, r_best, tc)
    rel = rel_rmse(xi, xi_fit)
    return {
        "s": float(s),
        "r": r_best,
        "F_inf": finf,
        "xi_fit": xi_fit,
        "rel_rmse": rel,
        "chi2_rel": float(np.sum(np.square((xi_fit - xi) / xi))),
        "success": bool(res.success),
    }


def profile_h(data_h, tc, s_grid, r_min, r_max):
    theta_u, theta_index, f_theta = build_theta_arrays(data_h)
    rows = [fit_r_for_s(data_h, theta_index, f_theta, float(s), tc, r_min, r_max) for s in s_grid]
    rel = np.array([row["rel_rmse"] for row in rows], dtype=np.float64)
    ibest = int(np.argmin(rel))
    return {
        "theta_u": theta_u,
        "theta_index": theta_index,
        "f_theta": f_theta,
        "rows": rows,
        "best": rows[ibest],
    }


def make_profile_plot(profiles, out_path, dpi, title):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    colors = {1.5: "tab:blue", 2.0: "tab:red"}
    for hstar, prof in profiles.items():
        s = np.array([row["s"] for row in prof["rows"]], dtype=np.float64)
        rel = np.array([row["rel_rmse"] for row in prof["rows"]], dtype=np.float64)
        best = prof["best"]
        color = colors.get(float(hstar), None)
        ax.plot(s, rel, lw=2.0, color=color, label=rf"$H_*={float(hstar):g}$")
        ax.plot(best["s"], best["rel_rmse"], "o", ms=6, color=color)
    ax.set_xlabel(r"$s$")
    ax.set_ylabel("rel-RMSE")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_r_plot(profiles, out_path, dpi, title):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    colors = {1.5: "tab:blue", 2.0: "tab:red"}
    for hstar, prof in profiles.items():
        s = np.array([row["s"] for row in prof["rows"]], dtype=np.float64)
        r = np.array([row["r"] for row in prof["rows"]], dtype=np.float64)
        color = colors.get(float(hstar), None)
        ax.plot(s, r, lw=2.0, color=color, label=rf"$H_*={float(hstar):g}$")
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"best-fit $r$")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_profile_table(path, hstar, prof):
    with open(path, "w") as f:
        f.write(f"# Model B s-profile at H*={float(hstar):g}\n")
        f.write("# c=1 fixed, t_c=1.5 fixed; for each s we optimize r and per-theta F_inf\n")
        f.write("# s r rel_rmse chi2_rel\n")
        for row in prof["rows"]:
            f.write(f"{row['s']:.10e} {row['r']:.10e} {row['rel_rmse']:.10e} {row['chi2_rel']:.10e}\n")
        f.write("\n")
        f.write("# best-fit F_inf(theta) at profile minimum\n")
        f.write("# theta0 F_theta F_inf_best\n")
        for th0, fth, finf in zip(prof["theta_u"], prof["f_theta"], prof["best"]["F_inf"]):
            f.write(f"{th0:.10f} {fth:.10e} {finf:.10e}\n")


def save_summary(path, args, profiles):
    with open(path, "w") as f:
        f.write("# Model B s-profile summary\n")
        f.write(f"# lattice_xi={Path(args.lattice_xi).resolve()}\n")
        f.write(f"# lattice_rho={Path(args.lattice_rho).resolve()}\n")
        f.write(f"# ode_xi={Path(args.ode_xi).resolve()}\n")
        f.write(f"# ode_nopt={Path(args.ode_nopt).resolve()}\n")
        f.write(f"# fixed_vw={float(args.fixed_vw):.6f}\n")
        f.write(f"# fixed_tc={float(args.fixed_tc):.6f}\n")
        f.write("# c=1 fixed; F_inf(theta) free per theta and solved analytically at each (s,r)\n\n")
        f.write("# hstar s_best r_best rel_rmse_best chi2_rel_best s_1pct_lo s_1pct_hi\n")
        for hstar, prof in profiles.items():
            s = np.array([row["s"] for row in prof["rows"]], dtype=np.float64)
            rel = np.array([row["rel_rmse"] for row in prof["rows"]], dtype=np.float64)
            mask = rel <= 1.01 * prof["best"]["rel_rmse"]
            f.write(
                f"{float(hstar):.8g} {prof['best']['s']:.10e} {prof['best']['r']:.10e} "
                f"{prof['best']['rel_rmse']:.10e} {prof['best']['chi2_rel']:.10e} "
                f"{s[mask].min():.10e} {s[mask].max():.10e}\n"
            )


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

    s_grid = np.arange(args.s_min, args.s_max + 0.5 * args.s_step, args.s_step, dtype=np.float64)
    profiles = {}
    for hstar in args.target_h:
        data_h = select_h(lat_data, hstar)
        prof = profile_h(data_h, args.fixed_tc, s_grid, args.r_min, args.r_max)
        profiles[float(hstar)] = prof

    # save per-H tables after creating dirs
    for hstar, prof in profiles.items():
        hdir = outdir / f"H{float(hstar):.1f}".replace(".", "p")
        hdir.mkdir(parents=True, exist_ok=True)
        save_profile_table(hdir / "profile_table.txt", hstar, prof)

    make_profile_plot(
        profiles,
        outdir / "modelB_relrmse_vs_s.png",
        args.dpi,
        rf"Model B profile: $c=1$, $t_c={float(args.fixed_tc):g}$ fixed",
    )
    make_r_plot(
        profiles,
        outdir / "modelB_r_vs_s.png",
        args.dpi,
        rf"Best-fit $r(s)$ for Model B: $c=1$, $t_c={float(args.fixed_tc):g}$",
    )
    save_summary(outdir / "modelB_profile_summary.txt", args, profiles)
    print(outdir / "modelB_profile_summary.txt")


if __name__ == "__main__":
    main()
