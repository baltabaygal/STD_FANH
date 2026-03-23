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

from ode.analysis.analyze_modelB_degeneracy_checks import build_theta_arrays, rel_rmse
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
            "Fit additive-delay tau_p transition models to lattice xi data with measured F(theta) "
            "from rho_noPT and per-theta F_inf."
        )
    )
    p.add_argument("--lattice-xi", type=str, default="lattice_data/data/energy_ratio_by_theta_data_v9.txt")
    p.add_argument("--lattice-rho", type=str, default="lattice_data/data/rho_noPT_data.txt")
    p.add_argument("--ode-nopt", type=str, default="ode/analysis/results/noPT_reference_dm_tp_fitready_H1p000.txt")
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--target-h", type=float, nargs="+", default=TARGET_H_DEFAULT)
    p.add_argument("--fixed-s", type=float, default=1.0)
    p.add_argument("--fixed-c", type=float, default=1.0)
    p.add_argument("--fixed-tc", type=float, default=1.5)
    p.add_argument("--tau-min", type=float, default=0.0)
    p.add_argument("--tau-max", type=float, default=3.0)
    p.add_argument("--tau-step", type=float, default=0.02)
    p.add_argument("--r-min", type=float, default=0.5)
    p.add_argument("--r-max", type=float, default=12.0)
    p.add_argument("--r-step", type=float, default=0.1)
    p.add_argument("--tc-min", type=float, default=0.8)
    p.add_argument("--tc-max", type=float, default=4.0)
    p.add_argument("--tc-step", type=float, default=0.05)
    p.add_argument("--bootstrap", type=int, default=100)
    p.add_argument("--bootstrap-seed", type=int, default=12345)
    p.add_argument("--bootstrap-jobs", type=int, default=min(6, max(os.cpu_count() or 1, 1)))
    p.add_argument(
        "--outdir",
        type=str,
        default="ode/analysis/results/lattice_fit/tau_delay_v9",
    )
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def solve_finf_per_theta(xi, tp, theta_index, f_theta, transient, lower=1.0e-6, upper=1.0e4):
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


def transient_tau(tp, s, tau, tc, r, c):
    return float(c) / (1.0 + np.power((float(s) * tp + float(tau)) / float(tc), float(r)))


def xi_model(tp, theta_index, f_theta, finf, transient):
    plateau = np.power(tp, 1.5) * finf[theta_index] / np.square(f_theta[theta_index])
    return plateau + transient


def score_tau(data_h, theta_index, f_theta, s, tau, tc, r, c):
    tp = data_h["tp"]
    xi = data_h["xi"]
    transient = transient_tau(tp, s, tau, tc, r, c)
    finf = solve_finf_per_theta(xi, tp, theta_index, f_theta, transient)
    xi_fit = xi_model(tp, theta_index, f_theta, finf, transient)
    resid = (xi_fit - xi) / xi
    return {
        "s": float(s),
        "tau": float(tau),
        "tc": float(tc),
        "r": float(r),
        "c": float(c),
        "F_inf": finf,
        "xi_fit": xi_fit,
        "rel_resid": resid,
        "chi2_rel": float(np.sum(np.square(resid))),
        "rel_rmse": rel_rmse(xi, xi_fit),
    }


def optimize_r_baseline(data_h, theta_index, f_theta, fixed_s, fixed_tc, fixed_c, r_min, r_max):
    def objective(r):
        return score_tau(data_h, theta_index, f_theta, fixed_s, 0.0, fixed_tc, float(r), fixed_c)["chi2_rel"]

    probe = np.linspace(r_min, r_max, 61, dtype=np.float64)
    vals = np.array([objective(r) for r in probe], dtype=np.float64)
    idx = int(np.argmin(vals))
    a = probe[max(idx - 1, 0)]
    b = probe[min(idx + 1, len(probe) - 1)]
    res = minimize_scalar(objective, bounds=(float(a), float(b)), method="bounded")
    out = score_tau(data_h, theta_index, f_theta, fixed_s, 0.0, fixed_tc, float(res.x), fixed_c)
    out["success"] = bool(res.success)
    return out


def optimize_r_for_tau(data_h, theta_index, f_theta, fixed_s, tau, fixed_tc, fixed_c, r_min, r_max):
    def objective(r):
        return score_tau(data_h, theta_index, f_theta, fixed_s, tau, fixed_tc, float(r), fixed_c)["chi2_rel"]

    probe = np.linspace(r_min, r_max, 61, dtype=np.float64)
    vals = np.array([objective(r) for r in probe], dtype=np.float64)
    idx = int(np.argmin(vals))
    a = probe[max(idx - 1, 0)]
    b = probe[min(idx + 1, len(probe) - 1)]
    res = minimize_scalar(objective, bounds=(float(a), float(b)), method="bounded")
    out = score_tau(data_h, theta_index, f_theta, fixed_s, float(tau), fixed_tc, float(res.x), fixed_c)
    out["success"] = bool(res.success)
    return out


def optimize_tau_r(data_h, theta_index, f_theta, fixed_s, fixed_tc, fixed_c, tau_bounds, r_bounds):
    def objective(x):
        tau, r = x
        return score_tau(data_h, theta_index, f_theta, fixed_s, tau, fixed_tc, r, fixed_c)["chi2_rel"]

    starts = [
        np.array([0.2, 2.0], dtype=np.float64),
        np.array([0.5, 3.0], dtype=np.float64),
        np.array([0.1, 1.5], dtype=np.float64),
    ]
    best = None
    bounds = [tau_bounds, r_bounds]
    for x0 in starts:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        cand = score_tau(data_h, theta_index, f_theta, fixed_s, float(res.x[0]), fixed_tc, float(res.x[1]), fixed_c)
        cand["success"] = bool(res.success)
        if best is None or cand["chi2_rel"] < best["chi2_rel"]:
            best = cand
    return best


def optimize_tau_r_tc(data_h, theta_index, f_theta, fixed_s, fixed_c, tau_bounds, r_bounds, tc_bounds):
    def objective(x):
        tau, r, tc = x
        return score_tau(data_h, theta_index, f_theta, fixed_s, tau, tc, r, fixed_c)["chi2_rel"]

    starts = [
        np.array([0.2, 2.0, 1.5], dtype=np.float64),
        np.array([0.5, 3.0, 2.0], dtype=np.float64),
        np.array([0.1, 1.5, 1.8], dtype=np.float64),
    ]
    best = None
    bounds = [tau_bounds, r_bounds, tc_bounds]
    for x0 in starts:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        cand = score_tau(data_h, theta_index, f_theta, fixed_s, float(res.x[0]), float(res.x[2]), float(res.x[1]), fixed_c)
        cand["success"] = bool(res.success)
        if best is None or cand["chi2_rel"] < best["chi2_rel"]:
            best = cand
    return best


def profile_tau_fixed_tc(data_h, fixed_s, fixed_tc, fixed_c, tau_grid, r_min, r_max):
    theta_u, theta_index, f_theta = build_theta_arrays(data_h)
    rows = [
        optimize_r_for_tau(data_h, theta_index, f_theta, fixed_s, float(tau), fixed_tc, fixed_c, r_min, r_max)
        for tau in tau_grid
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


def scan_tau_r_fixed_tc(data_h, fixed_s, fixed_tc, fixed_c, tau_grid, r_grid):
    theta_u, theta_index, f_theta = build_theta_arrays(data_h)
    rmse = np.zeros((len(r_grid), len(tau_grid)), dtype=np.float64)
    best = None
    for ir, r in enumerate(r_grid):
        for it, tau in enumerate(tau_grid):
            rec = score_tau(data_h, theta_index, f_theta, fixed_s, float(tau), fixed_tc, float(r), fixed_c)
            rmse[ir, it] = rec["rel_rmse"]
            if best is None or rec["rel_rmse"] < best["rel_rmse"]:
                best = rec
    return {
        "theta_u": theta_u,
        "theta_index": theta_index,
        "f_theta": f_theta,
        "tau_grid": tau_grid,
        "r_grid": r_grid,
        "rmse_grid": rmse,
        "best": best,
    }


def bootstrap_worker(payload):
    idx = payload["sample_idx"]
    data_h = payload["data_h"]
    theta_index = payload["theta_index"]
    f_theta = payload["f_theta"]
    fixed_s = payload["fixed_s"]
    fixed_tc = payload["fixed_tc"]
    fixed_c = payload["fixed_c"]
    tau_bounds = payload["tau_bounds"]
    r_bounds = payload["r_bounds"]
    mode = payload["mode"]
    tc_bounds = payload.get("tc_bounds")

    data_boot = {key: np.asarray(val[idx], dtype=np.float64) for key, val in data_h.items()}
    theta_boot = np.asarray(theta_index[idx], dtype=np.int64)
    if mode == "fixed_tc":
        rec = optimize_tau_r(data_boot, theta_boot, f_theta, fixed_s, fixed_tc, fixed_c, tau_bounds, r_bounds)
    else:
        rec = optimize_tau_r_tc(data_boot, theta_boot, f_theta, fixed_s, fixed_c, tau_bounds, r_bounds, tc_bounds)
    return {
        "tau": rec["tau"],
        "r": rec["r"],
        "tc": rec["tc"],
        "rel_rmse": rec["rel_rmse"],
    }


def run_bootstrap(data_h, theta_index, f_theta, fixed_s, fixed_tc, fixed_c, tau_bounds, r_bounds, nboot, seed, jobs, mode, tc_bounds=None):
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
                "fixed_s": fixed_s,
                "fixed_tc": fixed_tc,
                "fixed_c": fixed_c,
                "tau_bounds": tau_bounds,
                "r_bounds": r_bounds,
                "tc_bounds": tc_bounds,
                "mode": mode,
            }
        )
    results = []
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for rec in ex.map(bootstrap_worker, payloads):
                results.append(rec)
    else:
        for payload in payloads:
            results.append(bootstrap_worker(payload))
    return results


def interval(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return float(np.nanmedian(vals)), float(np.nanpercentile(vals, 16.0)), float(np.nanpercentile(vals, 84.0))


def make_tau_profile_plot(profiles, out_path, dpi, title):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    colors = {1.5: "tab:blue", 2.0: "tab:red"}
    for hstar, prof in profiles.items():
        tau = np.array([row["tau"] for row in prof["rows"]], dtype=np.float64)
        rel = np.array([row["rel_rmse"] for row in prof["rows"]], dtype=np.float64)
        color = colors.get(float(hstar), None)
        ax.plot(tau, rel, lw=2.0, color=color, label=rf"$H_*={float(hstar):g}$")
        ax.plot(prof["best"]["tau"], prof["best"]["rel_rmse"], "o", ms=6, color=color)
    ax.set_xlabel(r"$\tau_p$")
    ax.set_ylabel("rel-RMSE")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_r_vs_tau_plot(profiles, out_path, dpi, title):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    colors = {1.5: "tab:blue", 2.0: "tab:red"}
    for hstar, prof in profiles.items():
        tau = np.array([row["tau"] for row in prof["rows"]], dtype=np.float64)
        r = np.array([row["r"] for row in prof["rows"]], dtype=np.float64)
        color = colors.get(float(hstar), None)
        ax.plot(tau, r, lw=2.0, color=color, label=rf"$H_*={float(hstar):g}$")
    ax.set_xlabel(r"$\tau_p$")
    ax.set_ylabel(r"best-fit $r$")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_tau_r_contour(scan, out_path, dpi, title):
    tau_grid = scan["tau_grid"]
    r_grid = scan["r_grid"]
    rmse = scan["rmse_grid"]
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    vmax = float(np.percentile(rmse, 92.0))
    levels = np.linspace(float(np.min(rmse)), vmax, 16)
    cs = ax.contourf(tau_grid, r_grid, rmse, levels=levels, cmap="viridis")
    ax.contour(tau_grid, r_grid, rmse, levels=levels[::2], colors="white", linewidths=0.5, alpha=0.55)
    ax.plot(scan["best"]["tau"], scan["best"]["r"], "wo", ms=6, mec="black", mew=0.8, label="best")
    ax.set_xlabel(r"$\tau_p$")
    ax.set_ylabel(r"$r$")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.colorbar(cs, ax=ax, pad=0.02, label="rel-RMSE")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_residual_heatmap(data_h, theta_u, theta_index, rec, out_path, dpi, title):
    tp_u = np.array(sorted(np.unique(data_h["tp"])), dtype=np.float64)
    grid = np.full((len(theta_u), len(tp_u)), np.nan, dtype=np.float64)
    for i, th in enumerate(theta_u):
        for j, tp in enumerate(tp_u):
            mask = np.isclose(data_h["theta0"], th, rtol=0.0, atol=5.0e-4) & np.isclose(data_h["tp"], tp, rtol=0.0, atol=1.0e-12)
            if np.any(mask):
                grid[i, j] = float(np.median(rec["rel_resid"][mask]))
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    im = ax.pcolormesh(tp_u, np.arange(len(theta_u) + 1), np.vstack([grid, grid[-1:]]), cmap="coolwarm", shading="auto", vmin=-np.nanmax(np.abs(grid)), vmax=np.nanmax(np.abs(grid)))
    ax.set_xscale("log")
    ax.set_xlabel(r"$t_p$")
    ax.set_ylabel(r"$\theta_0$ index")
    ax.set_yticks(np.arange(len(theta_u)) + 0.5)
    ax.set_yticklabels([f"{th:.3g}" for th in theta_u])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, pad=0.02, label=r"$(\xi_{\rm model}-\xi)/\xi$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_bootstrap_hist(boot, key, out_path, dpi, title):
    vals = np.array([row[key] for row in boot], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.hist(vals, bins=20, color="tab:blue", alpha=0.8, edgecolor="black")
    med, lo, hi = interval(vals)
    ax.axvline(med, color="tab:red", lw=1.8, label=rf"median={med:.3f}")
    ax.axvspan(lo, hi, color="tab:red", alpha=0.15, label=rf"16-84%=[{lo:.3f},{hi:.3f}]")
    ax.set_xlabel(key)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_profile_table(path, profile):
    with open(path, "w") as f:
        f.write("# tau r rel_rmse chi2_rel tc\n")
        for row in profile["rows"]:
            f.write(f"{row['tau']:.10e} {row['r']:.10e} {row['rel_rmse']:.10e} {row['chi2_rel']:.10e} {row['tc']:.10e}\n")


def save_bootstrap_table(path, boot):
    with open(path, "w") as f:
        f.write("# tau r tc rel_rmse\n")
        for row in boot:
            f.write(f"{row['tau']:.10e} {row['r']:.10e} {row['tc']:.10e} {row['rel_rmse']:.10e}\n")


def save_summary(path, args, baseline_map, fixed_tau_profiles, fixed_tau_best, free_tc_best, boot_fixed, boot_free):
    with open(path, "w") as f:
        f.write("# Additive-delay tau_p fits on lattice data\n")
        f.write(f"# lattice_xi={Path(args.lattice_xi).resolve()}\n")
        f.write(f"# lattice_rho={Path(args.lattice_rho).resolve()}\n")
        f.write(f"# ode_nopt={Path(args.ode_nopt).resolve()}\n")
        f.write(f"# fixed_vw={float(args.fixed_vw):.6f}\n")
        f.write(f"# fixed_s={float(args.fixed_s):.6f}\n")
        f.write(f"# fixed_c={float(args.fixed_c):.6f}\n")
        f.write(f"# fixed_tc={float(args.fixed_tc):.6f}\n\n")
        f.write("# per-H results\n")
        f.write("# hstar baseline_r baseline_rel tau_fixed_best tau_fixed_r tau_fixed_rel tc_fixed free_tau free_r free_tc free_rel\n")
        for hstar in sorted(baseline_map):
            base = baseline_map[hstar]
            fix = fixed_tau_best[hstar]
            free = free_tc_best[hstar]
            f.write(
                f"{float(hstar):.8g} {base['r']:.10e} {base['rel_rmse']:.10e} "
                f"{fix['tau']:.10e} {fix['r']:.10e} {fix['rel_rmse']:.10e} {fix['tc']:.10e} "
                f"{free['tau']:.10e} {free['r']:.10e} {free['tc']:.10e} {free['rel_rmse']:.10e}\n"
            )
        f.write("\n")
        f.write("# bootstrap summaries for fixed-tc tau fit\n")
        f.write("# hstar tau_med tau_p16 tau_p84 r_med r_p16 r_p84\n")
        for hstar in sorted(boot_fixed):
            tau_med, tau_lo, tau_hi = interval([row["tau"] for row in boot_fixed[hstar]])
            r_med, r_lo, r_hi = interval([row["r"] for row in boot_fixed[hstar]])
            f.write(f"{float(hstar):.8g} {tau_med:.10e} {tau_lo:.10e} {tau_hi:.10e} {r_med:.10e} {r_lo:.10e} {r_hi:.10e}\n")
        f.write("\n")
        f.write("# bootstrap summaries for free-tc tau fit\n")
        f.write("# hstar tau_med tau_p16 tau_p84 r_med r_p16 r_p84 tc_med tc_p16 tc_p84\n")
        for hstar in sorted(boot_free):
            tau_med, tau_lo, tau_hi = interval([row["tau"] for row in boot_free[hstar]])
            r_med, r_lo, r_hi = interval([row["r"] for row in boot_free[hstar]])
            tc_med, tc_lo, tc_hi = interval([row["tc"] for row in boot_free[hstar]])
            f.write(
                f"{float(hstar):.8g} {tau_med:.10e} {tau_lo:.10e} {tau_hi:.10e} {r_med:.10e} {r_lo:.10e} {r_hi:.10e} "
                f"{tc_med:.10e} {tc_lo:.10e} {tc_hi:.10e}\n"
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

    tau_grid = np.arange(args.tau_min, args.tau_max + 0.5 * args.tau_step, args.tau_step, dtype=np.float64)
    r_grid = np.arange(args.r_min, args.r_max + 0.5 * args.r_step, args.r_step, dtype=np.float64)

    baseline_map = {}
    fixed_tau_profiles = {}
    fixed_tau_best = {}
    free_tc_best = {}
    boot_fixed = {}
    boot_free = {}

    tau_bounds = (args.tau_min, args.tau_max)
    r_bounds = (args.r_min, args.r_max)
    tc_bounds = (args.tc_min, args.tc_max)

    for hstar in args.target_h:
        hval = float(hstar)
        data_h = select_h(lat_data, hval)
        theta_u, theta_index, f_theta = build_theta_arrays(data_h)
        hdir = outdir / f"H{hval:.1f}".replace(".", "p")
        hdir.mkdir(parents=True, exist_ok=True)

        baseline = optimize_r_baseline(data_h, theta_index, f_theta, args.fixed_s, args.fixed_tc, args.fixed_c, args.r_min, args.r_max)
        baseline_map[hval] = baseline

        prof = profile_tau_fixed_tc(data_h, args.fixed_s, args.fixed_tc, args.fixed_c, tau_grid, args.r_min, args.r_max)
        fixed_tau_profiles[hval] = prof
        fixed_tau_best[hval] = optimize_tau_r(
            data_h,
            theta_index,
            f_theta,
            args.fixed_s,
            args.fixed_tc,
            args.fixed_c,
            tau_bounds,
            r_bounds,
        )
        save_profile_table(hdir / "tau_profile_table.txt", prof)

        scan = scan_tau_r_fixed_tc(data_h, args.fixed_s, args.fixed_tc, args.fixed_c, tau_grid, r_grid)
        make_tau_r_contour(
            scan,
            hdir / "tau_vs_r_contour.png",
            args.dpi,
            rf"Tau-delay rel-RMSE$(\tau_p,r)$ at $H_*={hval:g}$ with fixed $t_c={float(args.fixed_tc):.2f}$",
        )

        free = optimize_tau_r_tc(data_h, theta_index, f_theta, args.fixed_s, args.fixed_c, tau_bounds, r_bounds, tc_bounds)
        free_tc_best[hval] = free

        make_residual_heatmap(
            data_h,
            theta_u,
            theta_index,
            baseline,
            hdir / "residual_heatmap_baseline.png",
            args.dpi,
            rf"Baseline residuals at $H_*={hval:g}$ ($\tau_p=0$, $t_c={float(args.fixed_tc):.2f}$)",
        )
        make_residual_heatmap(
            data_h,
            theta_u,
            theta_index,
            fixed_tau_best[hval],
            hdir / "residual_heatmap_tau_fixedtc.png",
            args.dpi,
            rf"Tau-delay residuals at $H_*={hval:g}$ (fixed $t_c={float(args.fixed_tc):.2f}$)",
        )

        boot_fix = run_bootstrap(
            data_h,
            theta_index,
            f_theta,
            args.fixed_s,
            args.fixed_tc,
            args.fixed_c,
            tau_bounds,
            r_bounds,
            args.bootstrap,
            args.bootstrap_seed + int(round(100 * hval)),
            args.bootstrap_jobs,
            mode="fixed_tc",
        )
        boot_fixed[hval] = boot_fix
        save_bootstrap_table(hdir / "bootstrap_fixedtc_table.txt", boot_fix)
        make_bootstrap_hist(boot_fix, "tau", hdir / "bootstrap_tau_fixedtc.png", args.dpi, rf"Bootstrap $\tau_p$ at $H_*={hval:g}$, fixed $t_c$")
        make_bootstrap_hist(boot_fix, "r", hdir / "bootstrap_r_fixedtc.png", args.dpi, rf"Bootstrap $r$ at $H_*={hval:g}$, fixed $t_c$")

        boot_fr = run_bootstrap(
            data_h,
            theta_index,
            f_theta,
            args.fixed_s,
            args.fixed_tc,
            args.fixed_c,
            tau_bounds,
            r_bounds,
            args.bootstrap,
            args.bootstrap_seed + int(round(1000 * hval)),
            args.bootstrap_jobs,
            mode="free_tc",
            tc_bounds=tc_bounds,
        )
        boot_free[hval] = boot_fr
        save_bootstrap_table(hdir / "bootstrap_freetc_table.txt", boot_fr)

    make_tau_profile_plot(
        fixed_tau_profiles,
        outdir / "tau_profile_relrmse.png",
        args.dpi,
        rf"Tau-delay profile with fixed $t_c={float(args.fixed_tc):g}$, $s={float(args.fixed_s):g}$, $c={float(args.fixed_c):g}$",
    )
    make_r_vs_tau_plot(
        fixed_tau_profiles,
        outdir / "tau_fit_relrmse.png",
        args.dpi,
        rf"Best-fit $r(\tau_p)$ with fixed $t_c={float(args.fixed_tc):g}$",
    )
    save_summary(
        outdir / "tau_summary.txt",
        args,
        baseline_map,
        fixed_tau_profiles,
        fixed_tau_best,
        free_tc_best,
        boot_fixed,
        boot_free,
    )
    print(outdir / "tau_summary.txt")


if __name__ == "__main__":
    main()
