import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
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
            "Compare local log-slopes of Y3(tp) between lattice and direct ODE curves. "
            "Also tests whether a pure horizontal shift in log(tp) can align the slope profiles."
        )
    )
    p.add_argument("--lattice-xi", type=str, default="lattice_data/data/energy_ratio_by_theta_data_v9.txt")
    p.add_argument("--lattice-rho", type=str, default="lattice_data/data/rho_noPT_data.txt")
    p.add_argument("--ode-xi", type=str, default="ode/xi_DM_ODE_results.txt")
    p.add_argument("--ode-nopt", type=str, default="ode/analysis/results/noPT_reference_dm_tp_fitready_H1p000.txt")
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--target-h", type=float, nargs="+", default=TARGET_H_DEFAULT)
    p.add_argument("--shift-min", type=float, default=-0.6)
    p.add_argument("--shift-max", type=float, default=0.6)
    p.add_argument(
        "--outdir",
        type=str,
        default="ode/analysis/results/lattice_fit/lattice_y3_slope_compare_v9",
    )
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def rel_rmse(y, yfit):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square((yfit - y) / y))))


def rmse(y, yfit):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(yfit - y))))


def log_slope(tp, y):
    ltp = np.log(np.asarray(tp, dtype=np.float64))
    ly = np.log(np.asarray(y, dtype=np.float64))
    mids = np.exp(0.5 * (ltp[1:] + ltp[:-1]))
    slopes = np.diff(ly) / np.diff(ltp)
    return mids, slopes


def nearest_theta(values, theta0, atol=5.0e-4):
    values = np.asarray(values, dtype=np.float64)
    idx = int(np.argmin(np.abs(values - float(theta0))))
    if abs(values[idx] - float(theta0)) > atol:
        raise RuntimeError(f"No theta match for theta0={theta0:.10f}")
    return float(values[idx])


def build_slope_interpolators(ode_h):
    theta_u = np.array(sorted(np.unique(ode_h["theta0"])), dtype=np.float64)
    interps = {}
    ranges = {}
    for th0 in theta_u:
        mask = np.isclose(ode_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
        tp = ode_h["tp"][mask]
        y3 = ode_h["y3"][mask]
        order = np.argsort(tp)
        mids, slopes = log_slope(tp[order], y3[order])
        interps[float(th0)] = interp1d(
            np.log(mids),
            slopes,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        ranges[float(th0)] = (float(np.min(np.log(mids))), float(np.max(np.log(mids))))
    return interps, ranges


def evaluate_theta_slope_shift(lat_h, slope_interps, shift, theta0):
    th_key = nearest_theta(np.array(sorted(slope_interps.keys()), dtype=np.float64), theta0)
    mask = np.isclose(lat_h["theta0"], float(theta0), rtol=0.0, atol=5.0e-4)
    tp = lat_h["tp"][mask]
    y3 = lat_h["y3"][mask]
    order = np.argsort(tp)
    mids, slopes_lat = log_slope(tp[order], y3[order])
    slopes_ode = slope_interps[th_key](np.log(mids) + float(shift))
    return {
        "theta0": float(theta0),
        "mid_tp": mids,
        "slope_lat": slopes_lat,
        "slope_ode": np.asarray(slopes_ode, dtype=np.float64),
        "slope_rmse": rmse(slopes_lat, slopes_ode),
        "slope_rel_rmse": rel_rmse(np.abs(slopes_lat), np.abs(slopes_ode)),
    }


def fit_theta_slope_shift(lat_h, slope_interps, theta0, shift_min, shift_max):
    def target(shift):
        rec = evaluate_theta_slope_shift(lat_h, slope_interps, shift, theta0)
        return rec["slope_rmse"]

    res = minimize_scalar(target, bounds=(shift_min, shift_max), method="bounded")
    rec = evaluate_theta_slope_shift(lat_h, slope_interps, res.x, theta0)
    rec["delta_logtp"] = float(res.x)
    rec["scale_tp"] = float(np.exp(res.x))
    rec["success"] = bool(res.success)
    return rec


def summarize_h(lat_h, ode_h, shift_min, shift_max):
    theta_u = np.array(sorted(np.unique(lat_h["theta0"])), dtype=np.float64)
    interps, _ = build_slope_interpolators(ode_h)
    rows = []
    all_lat = []
    all_ode_noshift = []
    all_ode_best = []
    for th0 in theta_u:
        no = evaluate_theta_slope_shift(lat_h, interps, 0.0, th0)
        best = fit_theta_slope_shift(lat_h, interps, th0, shift_min, shift_max)
        best["slope_rmse_no_shift"] = no["slope_rmse"]
        best["slope_rel_rmse_no_shift"] = no["slope_rel_rmse"]
        rows.append(best)
        all_lat.append(no["slope_lat"])
        all_ode_noshift.append(no["slope_ode"])
        all_ode_best.append(best["slope_ode"])

    all_lat = np.concatenate(all_lat)
    all_ode_noshift = np.concatenate(all_ode_noshift)
    all_ode_best = np.concatenate(all_ode_best)
    return {
        "rows": rows,
        "global_slope_rmse_no_shift": rmse(all_lat, all_ode_noshift),
        "global_slope_rmse_best_theta_shift": rmse(all_lat, all_ode_best),
        "global_slope_rel_rmse_no_shift": rel_rmse(np.abs(all_lat), np.abs(all_ode_noshift)),
        "global_slope_rel_rmse_best_theta_shift": rel_rmse(np.abs(all_lat), np.abs(all_ode_best)),
    }


def make_panel_plot(lat_h, ode_h, summary, out_path, dpi, title):
    theta_u = np.array(sorted(np.unique(lat_h["theta0"])), dtype=np.float64)
    ntheta = len(theta_u)
    ncols = 3 if ntheta > 4 else 2
    nrows = int(np.ceil(ntheta / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.0 * nrows), squeeze=False)
    axes = axes.ravel()

    interps, _ = build_slope_interpolators(ode_h)
    row_lookup = {float(row["theta0"]): row for row in summary["rows"]}

    for ax, th0 in zip(axes, theta_u):
        th_key = nearest_theta(np.array(sorted(interps.keys()), dtype=np.float64), th0)
        mask_lat = np.isclose(lat_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
        mask_ode = np.isclose(ode_h["theta0"], th0, rtol=0.0, atol=5.0e-4)

        tp_lat = lat_h["tp"][mask_lat]
        y3_lat = lat_h["y3"][mask_lat]
        order_lat = np.argsort(tp_lat)
        mids_lat, slopes_lat = log_slope(tp_lat[order_lat], y3_lat[order_lat])

        tp_ode = ode_h["tp"][mask_ode]
        y3_ode = ode_h["y3"][mask_ode]
        order_ode = np.argsort(tp_ode)
        mids_ode, slopes_ode = log_slope(tp_ode[order_ode], y3_ode[order_ode])

        best = row_lookup[float(th0)]
        shifted_ode = interps[th_key](np.log(mids_lat) + best["delta_logtp"])

        ax.plot(mids_ode, slopes_ode, color="black", lw=1.0, alpha=0.5, label="ODE")
        ax.plot(mids_lat, slopes_lat, "ko", ms=3.6, label="lattice")
        ax.plot(mids_lat, interps[th_key](np.log(mids_lat)), color="tab:blue", ls="--", lw=1.6, label="ODE @ same $t_p$")
        ax.plot(mids_lat, shifted_ode, color="tab:red", ls="-.", lw=1.6, label="ODE best shift")
        ax.set_xscale("log")
        ax.set_xlabel(r"$t_p$ midpoint")
        ax.set_ylabel(r"$d\log Y_3 / d\log t_p$")
        ax.set_title(
            rf"$\theta_0={th0:.3g}$, $s_{{t_p}}={best['scale_tp']:.3f}$" "\n"
            rf"RMSE: {best['slope_rmse_no_shift']:.3f} $\to$ {best['slope_rmse']:.3f}"
        )
        ax.grid(alpha=0.25)

    for ax in axes[ntheta:]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_shift_plot(summary, out_path, dpi, title):
    rows = summary["rows"]
    theta_deg = np.degrees(np.array([row["theta0"] for row in rows], dtype=np.float64))
    scale_tp = np.array([row["scale_tp"] for row in rows], dtype=np.float64)
    rmse_no = np.array([row["slope_rmse_no_shift"] for row in rows], dtype=np.float64)
    rmse_best = np.array([row["slope_rmse"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True)
    axes[0].plot(theta_deg, scale_tp, "o-", color="tab:blue", lw=1.8, ms=4.8)
    axes[0].axhline(1.0, color="black", lw=1.0, ls="--")
    axes[0].set_ylabel(r"best $s_{t_p}$")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)

    axes[1].plot(theta_deg, rmse_no, "o--", color="tab:gray", lw=1.4, ms=4.2, label="no shift")
    axes[1].plot(theta_deg, rmse_best, "o-", color="tab:red", lw=1.8, ms=4.8, label="best shift")
    axes[1].set_xlabel(r"$\theta_0$ [deg]")
    axes[1].set_ylabel("slope RMSE")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_summary(path, args, scale_fit, summaries):
    with open(path, "w") as f:
        f.write("# Local slope comparison for lattice and ODE Y3(tp)\n")
        f.write(f"# lattice_xi={Path(args.lattice_xi).resolve()}\n")
        f.write(f"# lattice_rho={Path(args.lattice_rho).resolve()}\n")
        f.write(f"# ode_xi={Path(args.ode_xi).resolve()}\n")
        f.write(f"# ode_nopt={Path(args.ode_nopt).resolve()}\n")
        f.write(f"# fixed_vw={float(args.fixed_vw):.6f}\n")
        f.write(f"# fitted_K={scale_fit['scale']:.10e}\n")
        f.write("# slope definition: secant slope d log(Y3) / d log(tp) on adjacent tp bins\n")
        f.write("# best shift: compare lattice slopes to ODE slopes evaluated at shifted log(tp)\n\n")
        f.write(
            "# hstar global_slope_rmse_no_shift global_slope_rmse_best_theta_shift "
            "global_slope_rel_rmse_no_shift global_slope_rel_rmse_best_theta_shift\n"
        )
        for hstar, summary in summaries.items():
            f.write(
                f"{float(hstar):.8g} "
                f"{summary['global_slope_rmse_no_shift']:.10e} "
                f"{summary['global_slope_rmse_best_theta_shift']:.10e} "
                f"{summary['global_slope_rel_rmse_no_shift']:.10e} "
                f"{summary['global_slope_rel_rmse_best_theta_shift']:.10e}\n"
            )
        f.write("\n")
        f.write(
            "# per_theta hstar theta0 best_delta_logtp best_scale_tp "
            "slope_rmse_no_shift slope_rmse_best_shift "
            "slope_rel_rmse_no_shift slope_rel_rmse_best_shift\n"
        )
        for hstar, summary in summaries.items():
            for row in summary["rows"]:
                f.write(
                    f"{float(hstar):.8g} {row['theta0']:.10f} "
                    f"{row['delta_logtp']:.10e} {row['scale_tp']:.10e} "
                    f"{row['slope_rmse_no_shift']:.10e} {row['slope_rmse']:.10e} "
                    f"{row['slope_rel_rmse_no_shift']:.10e} {row['slope_rel_rmse']:.10e}\n"
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
    ode_data = build_ode_dataset(ode_ratio, ode_nopt)

    summaries = {}
    for hstar in args.target_h:
        lat_h = select_h(lat_data, hstar)
        ode_h = select_h(ode_data, hstar)
        summary = summarize_h(lat_h, ode_h, args.shift_min, args.shift_max)
        summaries[float(hstar)] = summary

        panel_out = outdir / f"compare_lattice_y3_slopes_H{float(hstar):.1f}".replace(".", "p")
        make_panel_plot(
            lat_h,
            ode_h,
            summary,
            panel_out.with_suffix(".png"),
            args.dpi,
            rf"Local $Y_3$ slopes vs $t_p$ at $H_*={float(hstar):g}$",
        )

        shift_out = outdir / f"compare_lattice_y3_slope_shifts_H{float(hstar):.1f}".replace(".", "p")
        make_shift_plot(
            summary,
            shift_out.with_suffix(".png"),
            args.dpi,
            rf"Best slope-alignment shifts at $H_*={float(hstar):g}$",
        )

    summary_out = outdir / "compare_lattice_y3_slopes_summary.txt"
    save_summary(summary_out, args, scale_fit, summaries)
    print(summary_out)


if __name__ == "__main__":
    main()
