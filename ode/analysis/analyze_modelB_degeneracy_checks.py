import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, minimize_scalar

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
            "Run degeneracy checks for Model B using measured lattice fanh_noPT. "
            "Includes fixed-c s profiles, s-c contours, tc equivalence profiles, "
            "weak-prior fits, and bootstrap stability for fixed-c best s."
        )
    )
    p.add_argument("--lattice-xi", type=str, default="lattice_data/data/energy_ratio_by_theta_data_v9.txt")
    p.add_argument("--lattice-rho", type=str, default="lattice_data/data/rho_noPT_data.txt")
    p.add_argument("--ode-xi", type=str, default="ode/xi_DM_ODE_results.txt")
    p.add_argument("--ode-nopt", type=str, default="ode/analysis/results/noPT_reference_dm_tp_fitready_H1p000.txt")
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--target-h", type=float, nargs="+", default=TARGET_H_DEFAULT)
    p.add_argument("--fixed-tc", type=float, default=1.5)
    p.add_argument("--s-profile-min", type=float, default=0.4)
    p.add_argument("--s-profile-max", type=float, default=1.1)
    p.add_argument("--s-profile-step", type=float, default=0.02)
    p.add_argument("--s-vlines", type=float, nargs="+", default=[0.8, 1.2])
    p.add_argument("--s2d-min", type=float, default=0.6)
    p.add_argument("--s2d-max", type=float, default=2.0)
    p.add_argument("--s2d-step", type=float, default=0.05)
    p.add_argument("--c2d-min", type=float, default=0.7)
    p.add_argument("--c2d-max", type=float, default=1.1)
    p.add_argument("--c2d-step", type=float, default=0.05)
    p.add_argument("--tc-profile-min", type=float, default=0.5)
    p.add_argument("--tc-profile-max", type=float, default=3.0)
    p.add_argument("--tc-profile-step", type=float, default=0.05)
    p.add_argument("--r-min", type=float, default=0.5)
    p.add_argument("--r-max", type=float, default=4.0)
    p.add_argument("--c-min", type=float, default=0.5)
    p.add_argument("--c-max", type=float, default=1.5)
    p.add_argument("--s-min", type=float, default=0.5)
    p.add_argument("--s-max", type=float, default=2.5)
    p.add_argument("--prior-c-mean", type=float, default=1.0)
    p.add_argument("--prior-c-sigma", type=float, default=0.05)
    p.add_argument("--prior-s-mean", type=float, default=1.2)
    p.add_argument("--prior-s-sigma", type=float, default=0.5)
    p.add_argument("--bootstrap", type=int, default=100)
    p.add_argument("--bootstrap-seed", type=int, default=12345)
    p.add_argument("--bootstrap-jobs", type=int, default=min(6, max(os.cpu_count() or 1, 1)))
    p.add_argument(
        "--outdir",
        type=str,
        default="ode/analysis/results/lattice_fit/modelB_degeneracy_checks_v9",
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


def solve_finf_per_theta(xi, tp, theta_index, f_theta, transient, lower=1.0e-6, upper=1.0e3):
    a = np.power(tp, 1.5) / np.square(f_theta[theta_index])
    finf = np.zeros(len(f_theta), dtype=np.float64)
    for i in range(len(f_theta)):
        mask = theta_index == i
        u = a[mask] / xi[mask]
        v = 1.0 - transient[mask] / xi[mask]
        denom = float(np.dot(u, u))
        finf_i = lower if denom <= 0.0 else float(np.dot(u, v) / denom)
        finf[i] = np.clip(finf_i, lower, upper)
    return finf


def transient_model(tp, s, r, tc, c):
    return float(c) / (1.0 + np.power((float(s) * tp) / float(tc), float(r)))


def xi_model(tp, theta_index, f_theta, finf, transient):
    plateau = np.power(tp, 1.5) * finf[theta_index] / np.square(f_theta[theta_index])
    return plateau + transient


def score_from_globals(data_h, theta_index, f_theta, s, r, tc, c):
    tp = data_h["tp"]
    xi = data_h["xi"]
    transient = transient_model(tp, s, r, tc, c)
    finf = solve_finf_per_theta(xi, tp, theta_index, f_theta, transient)
    xi_fit = xi_model(tp, theta_index, f_theta, finf, transient)
    resid = (xi_fit - xi) / xi
    return {
        "s": float(s),
        "r": float(r),
        "tc": float(tc),
        "c": float(c),
        "F_inf": finf,
        "xi_fit": xi_fit,
        "rel_resid": resid,
        "chi2_rel": float(np.sum(np.square(resid))),
        "rel_rmse": rel_rmse(xi, xi_fit),
    }


def optimize_r_for_fixed_sc_tc(data_h, theta_index, f_theta, s, c, tc, r_min, r_max):
    def objective(r):
        return score_from_globals(data_h, theta_index, f_theta, s, r, tc, c)["chi2_rel"]

    probe_r = np.linspace(r_min, r_max, 41, dtype=np.float64)
    vals = np.array([objective(r) for r in probe_r], dtype=np.float64)
    idx = int(np.argmin(vals))
    if idx == 0:
        a, b = probe_r[0], probe_r[1]
    elif idx == len(probe_r) - 1:
        a, b = probe_r[-2], probe_r[-1]
    else:
        a, b = probe_r[idx - 1], probe_r[idx + 1]
    res = minimize_scalar(objective, bounds=(float(a), float(b)), method="bounded")
    out = score_from_globals(data_h, theta_index, f_theta, s, float(res.x), tc, c)
    out["success"] = bool(res.success)
    return out


def optimize_rc_for_fixed_s_tc(data_h, theta_index, f_theta, s, tc, r_bounds, c_bounds):
    def objective(x):
        r, c = x
        return score_from_globals(data_h, theta_index, f_theta, s, r, tc, c)["chi2_rel"]

    starts = [
        np.array([2.0, 1.0], dtype=np.float64),
        np.array([1.5, 0.9], dtype=np.float64),
        np.array([2.5, 0.8], dtype=np.float64),
    ]
    best = None
    for x0 in starts:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=[r_bounds, c_bounds])
        cand = score_from_globals(data_h, theta_index, f_theta, s, float(res.x[0]), tc, float(res.x[1]))
        cand["success"] = bool(res.success)
        if best is None or cand["chi2_rel"] < best["chi2_rel"]:
            best = cand
    return best


def optimize_src_for_fixed_tc(data_h, theta_index, f_theta, tc, s_bounds, r_bounds, c_bounds, prior=None):
    def objective(x):
        s, r, c = x
        score = score_from_globals(data_h, theta_index, f_theta, s, r, tc, c)["chi2_rel"]
        if prior is not None:
            score += ((c - prior["c_mean"]) / prior["c_sigma"]) ** 2
            score += ((s - prior["s_mean"]) / prior["s_sigma"]) ** 2
        return score

    starts = [
        np.array([1.0, 2.0, 1.0], dtype=np.float64),
        np.array([1.2, 2.5, 0.9], dtype=np.float64),
        np.array([0.8, 1.8, 1.0], dtype=np.float64),
        np.array([1.6, 3.0, 0.85], dtype=np.float64),
    ]
    best = None
    bounds = [s_bounds, r_bounds, c_bounds]
    for x0 in starts:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        cand = score_from_globals(data_h, theta_index, f_theta, float(res.x[0]), float(res.x[1]), tc, float(res.x[2]))
        cand["success"] = bool(res.success)
        cand["penalized_obj"] = float(res.fun)
        if best is None or cand["penalized_obj"] < best["penalized_obj"]:
            best = cand
    return best


def profile_s_fixed_c(data_h, tc, c, s_grid, r_min, r_max):
    theta_u, theta_index, f_theta = build_theta_arrays(data_h)
    rows = [optimize_r_for_fixed_sc_tc(data_h, theta_index, f_theta, float(s), c, tc, r_min, r_max) for s in s_grid]
    rel = np.array([row["rel_rmse"] for row in rows], dtype=np.float64)
    ibest = int(np.argmin(rel))
    return {
        "theta_u": theta_u,
        "theta_index": theta_index,
        "f_theta": f_theta,
        "rows": rows,
        "best": rows[ibest],
    }


def profile_tc_free_c(data_h, s_fixed, tc_grid, r_bounds, c_bounds):
    theta_u, theta_index, f_theta = build_theta_arrays(data_h)
    rows = [optimize_rc_for_fixed_s_tc(data_h, theta_index, f_theta, s_fixed, float(tc), r_bounds, c_bounds) for tc in tc_grid]
    rel = np.array([row["rel_rmse"] for row in rows], dtype=np.float64)
    ibest = int(np.argmin(rel))
    return {
        "theta_u": theta_u,
        "theta_index": theta_index,
        "f_theta": f_theta,
        "rows": rows,
        "best": rows[ibest],
    }


def scan_s_c(data_h, tc, s_grid, c_grid, r_min, r_max):
    theta_u, theta_index, f_theta = build_theta_arrays(data_h)
    rmse_grid = np.zeros((len(c_grid), len(s_grid)), dtype=np.float64)
    r_grid = np.zeros_like(rmse_grid)
    best = None
    for ic, c in enumerate(c_grid):
        for is_, s in enumerate(s_grid):
            rec = optimize_r_for_fixed_sc_tc(data_h, theta_index, f_theta, float(s), float(c), tc, r_min, r_max)
            rmse_grid[ic, is_] = rec["rel_rmse"]
            r_grid[ic, is_] = rec["r"]
            if best is None or rec["rel_rmse"] < best["rel_rmse"]:
                best = rec
    return {
        "theta_u": theta_u,
        "theta_index": theta_index,
        "f_theta": f_theta,
        "s_grid": s_grid,
        "c_grid": c_grid,
        "rmse_grid": rmse_grid,
        "r_grid": r_grid,
        "best": best,
    }


def bootstrap_profile_worker(payload):
    idx = payload["sample_idx"]
    data_h = payload["data_h"]
    theta_index = payload["theta_index"]
    f_theta = payload["f_theta"]
    s_grid = payload["s_grid"]
    tc = payload["tc"]
    c = payload["c"]
    r_min = payload["r_min"]
    r_max = payload["r_max"]

    data_boot = {key: np.asarray(val[idx], dtype=np.float64) for key, val in data_h.items()}
    theta_boot = np.asarray(theta_index[idx], dtype=np.int64)
    rows = [optimize_r_for_fixed_sc_tc(data_boot, theta_boot, f_theta, float(s), c, tc, r_min, r_max) for s in s_grid]
    rel = np.array([row["rel_rmse"] for row in rows], dtype=np.float64)
    ibest = int(np.argmin(rel))
    return {
        "s_best": float(rows[ibest]["s"]),
        "r_best": float(rows[ibest]["r"]),
        "rel_rmse_best": float(rows[ibest]["rel_rmse"]),
    }


def run_bootstrap_profile(data_h, theta_index, f_theta, s_grid, tc, c, r_min, r_max, nboot, seed, jobs):
    rng = np.random.default_rng(seed)
    payloads = []
    for _ in range(nboot):
        idx = rng.integers(0, len(data_h["theta0"]), size=len(data_h["theta0"]), endpoint=False, dtype=np.int64)
        payloads.append(
            {
                "sample_idx": idx,
                "data_h": data_h,
                "theta_index": theta_index,
                "f_theta": f_theta,
                "s_grid": s_grid,
                "tc": tc,
                "c": c,
                "r_min": r_min,
                "r_max": r_max,
            }
        )
    results = []
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for rec in ex.map(bootstrap_profile_worker, payloads):
                results.append(rec)
    else:
        for payload in payloads:
            results.append(bootstrap_profile_worker(payload))
    return results


def interval(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return float(np.nanmedian(vals)), float(np.nanpercentile(vals, 16.0)), float(np.nanpercentile(vals, 84.0))


def make_s_profile_plot(profiles, out_path, dpi, title, vlines):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    colors = {1.5: "tab:blue", 2.0: "tab:red"}
    for hstar, prof in profiles.items():
        s = np.array([row["s"] for row in prof["rows"]], dtype=np.float64)
        rel = np.array([row["rel_rmse"] for row in prof["rows"]], dtype=np.float64)
        color = colors.get(float(hstar), None)
        ax.plot(s, rel, lw=2.0, color=color, label=rf"$H_*={float(hstar):g}$")
        ax.plot(prof["best"]["s"], prof["best"]["rel_rmse"], "o", ms=6, color=color)
    for val in vlines:
        ax.axvline(float(val), color="black", lw=1.0, ls="--", alpha=0.5)
    ax.set_xlabel(r"$s$")
    ax.set_ylabel("rel-RMSE")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_r_profile_plot(profiles, out_path, dpi, title, xkey="s", xlabel=r"$s$"):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    colors = {1.5: "tab:blue", 2.0: "tab:red"}
    for hstar, prof in profiles.items():
        x = np.array([row[xkey] for row in prof["rows"]], dtype=np.float64)
        r = np.array([row["r"] for row in prof["rows"]], dtype=np.float64)
        color = colors.get(float(hstar), None)
        ax.plot(x, r, lw=2.0, color=color, label=rf"$H_*={float(hstar):g}$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"best-fit $r$")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_tc_profile_plot(profiles, out_path, dpi, title):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    colors = {1.5: "tab:blue", 2.0: "tab:red"}
    for hstar, prof in profiles.items():
        tc = np.array([row["tc"] for row in prof["rows"]], dtype=np.float64)
        rel = np.array([row["rel_rmse"] for row in prof["rows"]], dtype=np.float64)
        color = colors.get(float(hstar), None)
        ax.plot(tc, rel, lw=2.0, color=color, label=rf"$H_*={float(hstar):g}$")
        ax.plot(prof["best"]["tc"], prof["best"]["rel_rmse"], "o", ms=6, color=color)
    ax.set_xlabel(r"$t_c$")
    ax.set_ylabel("rel-RMSE")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_sc_contour(scan, free_fit, out_path, dpi, title):
    s_grid = scan["s_grid"]
    c_grid = scan["c_grid"]
    rmse = scan["rmse_grid"]
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    levels = np.linspace(float(rmse.min()), float(np.percentile(rmse, 90.0)), 14)
    cs = ax.contourf(s_grid, c_grid, rmse, levels=levels, cmap="viridis")
    ax.contour(s_grid, c_grid, rmse, levels=levels[::2], colors="white", linewidths=0.5, alpha=0.55)
    ax.plot(scan["best"]["s"], scan["best"]["c"], "wo", ms=6, mec="black", mew=0.8, label="best on grid")
    ax.plot(free_fit["s"], free_fit["c"], "r*", ms=10, label="best free fit")
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$c$")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.colorbar(cs, ax=ax, pad=0.02, label="rel-RMSE")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_bootstrap_hist(boot, out_path, dpi, title):
    s_best = np.array([row["s_best"] for row in boot], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    bins = np.arange(s_best.min() - 0.01, s_best.max() + 0.03, 0.02)
    ax.hist(s_best, bins=bins, color="tab:blue", alpha=0.8, edgecolor="black")
    med, lo, hi = interval(s_best)
    ax.axvline(med, color="tab:red", lw=1.8, label=rf"median={med:.3f}")
    ax.axvspan(lo, hi, color="tab:red", alpha=0.15, label=rf"16-84% = [{lo:.3f}, {hi:.3f}]")
    ax.set_xlabel(r"best $s$")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_profile_table(path, profile, xkey):
    with open(path, "w") as f:
        f.write(f"# x={xkey} r rel_rmse chi2_rel c tc s\n")
        for row in profile["rows"]:
            f.write(
                f"{row[xkey]:.10e} {row['r']:.10e} {row['rel_rmse']:.10e} {row['chi2_rel']:.10e} "
                f"{row['c']:.10e} {row['tc']:.10e} {row['s']:.10e}\n"
            )


def save_bootstrap_table(path, boot):
    with open(path, "w") as f:
        f.write("# s_best r_best rel_rmse_best\n")
        for row in boot:
            f.write(f"{row['s_best']:.10e} {row['r_best']:.10e} {row['rel_rmse_best']:.10e}\n")


def save_summary(path, args, s_profiles, sc_scans, tc_profiles, free_fits, prior_fits, boot_map):
    with open(path, "w") as f:
        f.write("# Model B degeneracy checks summary\n")
        f.write(f"# lattice_xi={Path(args.lattice_xi).resolve()}\n")
        f.write(f"# lattice_rho={Path(args.lattice_rho).resolve()}\n")
        f.write(f"# ode_xi={Path(args.ode_xi).resolve()}\n")
        f.write(f"# ode_nopt={Path(args.ode_nopt).resolve()}\n")
        f.write(f"# fixed_vw={float(args.fixed_vw):.6f}\n")
        f.write(f"# fixed_tc={float(args.fixed_tc):.6f}\n")
        f.write("# step1: c=1 fixed, tc fixed, profile s and optimize r,F_inf\n")
        f.write("# step2: scan s,c at fixed tc and optimize r,F_inf\n")
        f.write("# step3: s=1 fixed, profile tc and optimize r,c,F_inf\n")
        f.write("# step4: free s,r,c with weak Gaussian priors on s and c, tc fixed\n")
        f.write("# step5: bootstrap stability of fixed-c s minimum\n\n")
        f.write("# fixed-c s profile summary\n")
        f.write("# hstar s_best r_best rel_rmse_best s_1pct_lo s_1pct_hi\n")
        for hstar, prof in s_profiles.items():
            s = np.array([row["s"] for row in prof["rows"]], dtype=np.float64)
            rel = np.array([row["rel_rmse"] for row in prof["rows"]], dtype=np.float64)
            mask = rel <= 1.01 * prof["best"]["rel_rmse"]
            f.write(
                f"{float(hstar):.8g} {prof['best']['s']:.10e} {prof['best']['r']:.10e} "
                f"{prof['best']['rel_rmse']:.10e} {s[mask].min():.10e} {s[mask].max():.10e}\n"
            )
        f.write("\n")

        f.write("# s-c contour best points and free-fit overlay\n")
        f.write("# hstar grid_s_best grid_c_best grid_r_best grid_rel free_s free_c free_r free_rel\n")
        for hstar in sorted(sc_scans):
            grid_best = sc_scans[hstar]["best"]
            free_best = free_fits[hstar]
            f.write(
                f"{float(hstar):.8g} {grid_best['s']:.10e} {grid_best['c']:.10e} {grid_best['r']:.10e} {grid_best['rel_rmse']:.10e} "
                f"{free_best['s']:.10e} {free_best['c']:.10e} {free_best['r']:.10e} {free_best['rel_rmse']:.10e}\n"
            )
        f.write("\n")

        f.write("# tc profile with s=1 fixed\n")
        f.write("# hstar tc_best r_best c_best rel_rmse_best\n")
        for hstar, prof in tc_profiles.items():
            best = prof["best"]
            f.write(
                f"{float(hstar):.8g} {best['tc']:.10e} {best['r']:.10e} "
                f"{best['c']:.10e} {best['rel_rmse']:.10e}\n"
            )
        f.write("\n")

        f.write("# weak-prior fit versus unregularized free fit\n")
        f.write("# hstar free_s free_c free_r free_rel prior_s prior_c prior_r prior_rel\n")
        for hstar in sorted(free_fits):
            free_best = free_fits[hstar]
            prior_best = prior_fits[hstar]
            f.write(
                f"{float(hstar):.8g} {free_best['s']:.10e} {free_best['c']:.10e} {free_best['r']:.10e} {free_best['rel_rmse']:.10e} "
                f"{prior_best['s']:.10e} {prior_best['c']:.10e} {prior_best['r']:.10e} {prior_best['rel_rmse']:.10e}\n"
            )
        f.write("\n")

        f.write("# bootstrap stability of fixed-c best s\n")
        f.write("# hstar s_median s_p16 s_p84 r_median r_p16 r_p84\n")
        for hstar, boot in boot_map.items():
            s_vals = [row["s_best"] for row in boot]
            r_vals = [row["r_best"] for row in boot]
            smed, slo, shi = interval(s_vals)
            rmed, rlo, rhi = interval(r_vals)
            f.write(
                f"{float(hstar):.8g} {smed:.10e} {slo:.10e} {shi:.10e} "
                f"{rmed:.10e} {rlo:.10e} {rhi:.10e}\n"
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

    s_profile_grid = np.arange(args.s_profile_min, args.s_profile_max + 0.5 * args.s_profile_step, args.s_profile_step, dtype=np.float64)
    s2d_grid = np.arange(args.s2d_min, args.s2d_max + 0.5 * args.s2d_step, args.s2d_step, dtype=np.float64)
    c2d_grid = np.arange(args.c2d_min, args.c2d_max + 0.5 * args.c2d_step, args.c2d_step, dtype=np.float64)
    tc_grid = np.arange(args.tc_profile_min, args.tc_profile_max + 0.5 * args.tc_profile_step, args.tc_profile_step, dtype=np.float64)

    s_profiles = {}
    sc_scans = {}
    tc_profiles = {}
    free_fits = {}
    prior_fits = {}
    boot_map = {}

    for hstar in args.target_h:
        hval = float(hstar)
        hdir = outdir / f"H{hval:.1f}".replace(".", "p")
        hdir.mkdir(parents=True, exist_ok=True)

        data_h = select_h(lat_data, hval)
        theta_u, theta_index, f_theta = build_theta_arrays(data_h)

        s_prof = profile_s_fixed_c(data_h, args.fixed_tc, 1.0, s_profile_grid, args.r_min, args.r_max)
        s_profiles[hval] = s_prof
        save_profile_table(hdir / "s_profile_fixedc_table.txt", s_prof, "s")

        sc_scan = scan_s_c(data_h, args.fixed_tc, s2d_grid, c2d_grid, args.r_min, args.r_max)
        sc_scans[hval] = sc_scan

        free_fit = optimize_src_for_fixed_tc(
            data_h,
            theta_index,
            f_theta,
            args.fixed_tc,
            (args.s_min, args.s_max),
            (args.r_min, args.r_max),
            (args.c_min, args.c_max),
            prior=None,
        )
        free_fits[hval] = free_fit

        prior_fit = optimize_src_for_fixed_tc(
            data_h,
            theta_index,
            f_theta,
            args.fixed_tc,
            (args.s_min, args.s_max),
            (args.r_min, args.r_max),
            (args.c_min, args.c_max),
            prior={
                "c_mean": args.prior_c_mean,
                "c_sigma": args.prior_c_sigma,
                "s_mean": args.prior_s_mean,
                "s_sigma": args.prior_s_sigma,
            },
        )
        prior_fits[hval] = prior_fit

        tc_prof = profile_tc_free_c(
            data_h,
            1.0,
            tc_grid,
            (args.r_min, args.r_max),
            (args.c_min, args.c_max),
        )
        tc_profiles[hval] = tc_prof
        save_profile_table(hdir / "tc_profile_sfixed1_table.txt", tc_prof, "tc")

        boot = run_bootstrap_profile(
            data_h,
            theta_index,
            f_theta,
            s_profile_grid,
            args.fixed_tc,
            1.0,
            args.r_min,
            args.r_max,
            args.bootstrap,
            args.bootstrap_seed + int(round(100 * hval)),
            args.bootstrap_jobs,
        )
        boot_map[hval] = boot
        save_bootstrap_table(hdir / "bootstrap_s_profile_table.txt", boot)
        make_bootstrap_hist(
            boot,
            hdir / "bootstrap_s_profile_hist.png",
            args.dpi,
            rf"Bootstrap best-$s$ distribution at $H_*={hval:g}$",
        )

        make_sc_contour(
            sc_scan,
            free_fit,
            hdir / "sc_contour_relrmse.png",
            args.dpi,
            rf"Model B rel-RMSE$(s,c)$ at $H_*={hval:g}$",
        )

    make_s_profile_plot(
        s_profiles,
        outdir / "modelB_fixedc_relrmse_vs_s.png",
        args.dpi,
        rf"Model B fixed-$c$ profile: $c=1$, $t_c={float(args.fixed_tc):g}$",
        args.s_vlines,
    )
    make_r_profile_plot(
        s_profiles,
        outdir / "modelB_fixedc_r_vs_s.png",
        args.dpi,
        rf"Best-fit $r(s)$ for fixed-$c$ Model B, $t_c={float(args.fixed_tc):g}$",
        xkey="s",
        xlabel=r"$s$",
    )
    make_tc_profile_plot(
        tc_profiles,
        outdir / "modelB_sfixed1_relrmse_vs_tc.png",
        args.dpi,
        r"Model B profile with fixed $s=1$ and free $t_c,c,r$",
    )
    make_r_profile_plot(
        tc_profiles,
        outdir / "modelB_sfixed1_r_vs_tc.png",
        args.dpi,
        r"Best-fit $r(t_c)$ with fixed $s=1$",
        xkey="tc",
        xlabel=r"$t_c$",
    )
    save_summary(outdir / "modelB_degeneracy_summary.txt", args, s_profiles, sc_scans, tc_profiles, free_fits, prior_fits, boot_map)
    print(outdir / "modelB_degeneracy_summary.txt")


if __name__ == "__main__":
    main()
