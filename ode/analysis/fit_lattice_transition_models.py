import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

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


@dataclass(frozen=True)
class ModelSpec:
    name: str
    tc_mode: str
    loss: str


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Fit lattice transition models using measured lattice fanh_noPT from rho_noPT_data. "
            "Implements the user-specified xi-model variants A/B/C and derives Y3 from xi."
        )
    )
    p.add_argument("--lattice-xi", type=str, default="lattice_data/data/energy_ratio_by_theta_data_v9.txt")
    p.add_argument("--lattice-rho", type=str, default="lattice_data/data/rho_noPT_data.txt")
    p.add_argument("--ode-xi", type=str, default="ode/xi_DM_ODE_results.txt")
    p.add_argument("--ode-nopt", type=str, default="ode/analysis/results/noPT_reference_dm_tp_fitready_H1p000.txt")
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--target-h", type=float, nargs="+", default=TARGET_H_DEFAULT)
    p.add_argument("--fixed-tc", type=float, default=1.5)
    p.add_argument("--s-min", type=float, default=0.5)
    p.add_argument("--s-max", type=float, default=1.2)
    p.add_argument("--huber-scale", type=float, default=2.0e-2)
    p.add_argument("--f-inf-reg", type=float, default=1.0e-2)
    p.add_argument("--alpha-reg", type=float, default=1.0e-2)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--bootstrap-seed", type=int, default=12345)
    p.add_argument("--bootstrap-jobs", type=int, default=min(6, max(os.cpu_count() or 1, 1)))
    p.add_argument("--grid-init-step", type=float, default=0.02)
    p.add_argument("--simplicity-tol", type=float, default=0.05)
    p.add_argument(
        "--outdir",
        type=str,
        default="ode/analysis/results/lattice_fit/transition_model_scan_v9",
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


def log_slope(x, y):
    lx = np.log(np.asarray(x, dtype=np.float64))
    ly = np.log(np.asarray(y, dtype=np.float64))
    mids = np.exp(0.5 * (lx[1:] + lx[:-1]))
    slopes = np.diff(ly) / np.diff(lx)
    return mids, slopes


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


def estimate_finf_guess(data_h, theta_u, source_key="y3", top_n=3):
    guesses = []
    for th0 in theta_u:
        mask = np.isclose(data_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
        tp = data_h["tp"][mask]
        vals = data_h[source_key][mask]
        order = np.argsort(tp)
        take = order[-top_n:]
        guesses.append(float(np.median(vals[take])))
    return np.array(guesses, dtype=np.float64)


def interpolate_ode_on_lattice(lattice_h, ode_h, key):
    theta_u = unique_theta(lattice_h)
    target_key = key if key in lattice_h else "xi"
    yfit = np.zeros_like(lattice_h[target_key], dtype=np.float64)
    for th0 in theta_u:
        lmask = np.isclose(lattice_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
        omask = np.isclose(ode_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
        obeta = ode_h["beta_over_h"][omask]
        oy = ode_h[key][omask]
        order = np.argsort(obeta)
        interp = np.interp(np.log(lattice_h["beta_over_h"][lmask]), np.log(obeta[order]), oy[order])
        yfit[lmask] = interp
    return yfit


def build_ode_shifted_slope_interps(ode_h):
    theta_u = unique_theta(ode_h)
    out = {}
    for th0 in theta_u:
        mask = np.isclose(ode_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
        tp = ode_h["tp"][mask]
        y3 = ode_h["y3"][mask]
        order = np.argsort(tp)
        mids, slopes = log_slope(tp[order], y3[order])
        out[float(th0)] = interp1d(
            np.log(mids),
            slopes,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
    return out


def unpack_params(x, spec, ntheta, fixed_tc):
    pos = 0
    s = float(x[pos])
    r = float(x[pos + 1])
    pos += 2
    if spec.tc_mode == "free":
        tc = float(x[pos])
        pos += 1
    else:
        tc = float(fixed_tc)

    c = None
    amp = None
    alpha = None
    if spec.name == "B":
        c = float(x[pos])
        pos += 1
    elif spec.name == "C":
        amp = float(x[pos])
        alpha = float(x[pos + 1])
        pos += 2

    finf = np.array(x[pos : pos + ntheta], dtype=np.float64)
    pos += ntheta
    if pos != len(x):
        raise RuntimeError("Parameter packing mismatch.")
    return {
        "s": s,
        "r": r,
        "tc": tc,
        "c": c,
        "A": amp,
        "alpha": alpha,
        "F_inf": finf,
    }


def xi_model_from_params(data_h, theta_index, f_theta, params):
    tp = data_h["tp"]
    frow = f_theta[theta_index]
    finf_row = params["F_inf"][theta_index]
    plateau = np.power(tp, 1.5) * finf_row / np.square(frow)
    crossover = 1.0 + np.power((params["s"] * tp) / params["tc"], params["r"])

    if params["c"] is not None:
        transient = params["c"] / crossover
    elif params["A"] is not None:
        transient = params["A"] * np.power(frow, params["alpha"] - 1.0) / crossover
    else:
        transient = 1.0 / crossover
    return plateau + transient


def y3_from_xi(data_h, theta_index, f_theta, xi):
    frow = f_theta[theta_index]
    return xi * np.square(frow) / np.power(data_h["tp"], 1.5)


def residual_vector(x, spec, data_h, theta_index, f_theta, finf_ode, fixed_tc, f_inf_reg, alpha_reg):
    params = unpack_params(x, spec, len(f_theta), fixed_tc)
    xi_fit = xi_model_from_params(data_h, theta_index, f_theta, params)
    resid = (xi_fit - data_h["xi"]) / data_h["xi"]
    extras = []
    if f_inf_reg > 0.0:
        extras.append(math.sqrt(f_inf_reg) * (params["F_inf"] - finf_ode) / np.maximum(finf_ode, 1.0e-8))
    if spec.name == "C" and alpha_reg > 0.0:
        extras.append(np.array([math.sqrt(alpha_reg) * (params["alpha"] - 1.0)], dtype=np.float64))
    if extras:
        resid = np.concatenate([resid] + extras)
    return resid


def bounds_for_spec(spec, ntheta, fixed_tc, s_bounds):
    lower = [float(s_bounds[0]), 0.5]
    upper = [float(s_bounds[1]), 4.0]
    if spec.tc_mode == "free":
        lower += [0.5]
        upper += [3.0]
    if spec.name == "B":
        lower += [0.5]
        upper += [1.5]
    elif spec.name == "C":
        lower += [0.1, 0.5]
        upper += [3.0, 2.5]
    lower += [1.0e-6] * ntheta
    upper += [1.0e3] * ntheta
    return np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64)


def initial_guesses(spec, finf_guess, fixed_tc, grid_step, s_bounds):
    starts = []
    s_min, s_max = float(s_bounds[0]), float(s_bounds[1])
    s_seed = np.array([0.7, 0.8, 0.9, 1.0, 1.1, s_min, 0.5 * (s_min + s_max), s_max], dtype=np.float64)
    s_vals = np.unique(np.clip(s_seed, s_min, s_max))
    r_vals = [1.0, 1.5, 0.8]
    tc_vals = [1.5, 1.0, 2.0]

    for s in s_vals:
        for r in r_vals[:2]:
            vec = [s, r]
            if spec.tc_mode == "free":
                vec.append(1.5)
            if spec.name == "B":
                vec.append(1.0)
            elif spec.name == "C":
                vec += [1.0, 1.0]
            vec += list(finf_guess)
            starts.append(np.array(vec, dtype=np.float64))

    if spec.tc_mode == "free":
        extra = []
        for tc in tc_vals:
            vec = [0.9, 1.0, tc]
            if spec.name == "B":
                vec.append(1.0)
            elif spec.name == "C":
                vec += [1.0, 1.0]
            vec += list(finf_guess)
            extra.append(np.array(vec, dtype=np.float64))
        starts.extend(extra)

    if grid_step > 0.0:
        s_grid = np.arange(s_min, s_max + 0.5 * grid_step, grid_step, dtype=np.float64)
        for s in s_grid[:: max(1, int(round(0.08 / grid_step)))]:
            vec = [float(s), 1.0]
            if spec.tc_mode == "free":
                vec.append(fixed_tc)
            if spec.name == "B":
                vec.append(1.0)
            elif spec.name == "C":
                vec += [1.0, 1.0]
            vec += list(finf_guess)
            starts.append(np.array(vec, dtype=np.float64))
    return starts


def fit_one_spec(
    spec,
    data_h,
    theta_index,
    f_theta,
    finf_guess,
    finf_ode,
    fixed_tc,
    f_inf_reg,
    alpha_reg,
    huber_scale,
    grid_step,
    s_bounds,
    starts_override=None,
):
    ntheta = len(f_theta)
    lower, upper = bounds_for_spec(spec, ntheta, fixed_tc, s_bounds)
    starts = starts_override if starts_override is not None else initial_guesses(spec, finf_guess, fixed_tc, grid_step, s_bounds)
    best = None
    loss = "linear" if spec.loss == "linear" else "huber"

    for x0 in starts:
        x0 = np.clip(np.array(x0, dtype=np.float64), lower + 1.0e-10, upper - 1.0e-10)
        try:
            res = least_squares(
                residual_vector,
                x0,
                bounds=(lower, upper),
                args=(spec, data_h, theta_index, f_theta, finf_ode, fixed_tc, f_inf_reg, alpha_reg),
                max_nfev=60000,
                ftol=1.0e-12,
                xtol=1.0e-12,
                gtol=1.0e-12,
                loss=loss,
                f_scale=huber_scale,
            )
        except Exception:
            continue

        params = unpack_params(res.x, spec, ntheta, fixed_tc)
        xi_fit = xi_model_from_params(data_h, theta_index, f_theta, params)
        y3_fit = y3_from_xi(data_h, theta_index, f_theta, xi_fit)
        rel_resid = (xi_fit - data_h["xi"]) / data_h["xi"]
        chi2_rel = float(np.sum(np.square(rel_resid)))
        npar = len(res.x)
        dof = max(len(data_h["xi"]) - npar, 1)
        record = {
            "spec": spec,
            "x": np.array(res.x, dtype=np.float64),
            "params": params,
            "xi_fit": xi_fit,
            "y3_fit": y3_fit,
            "rel_resid": rel_resid,
            "chi2_rel": chi2_rel,
            "chi2_rel_red": chi2_rel / dof,
            "rel_rmse": rel_rmse(data_h["xi"], xi_fit),
            "success": bool(res.success),
            "message": str(res.message),
            "cost": float(res.cost),
            "n": len(data_h["xi"]),
            "dof": dof,
        }
        if best is None or record["chi2_rel"] < best["chi2_rel"]:
            best = record

    if best is None:
        raise RuntimeError(f"All fits failed for spec {spec}")
    return best


def choose_simple_model(records_linear, simplicity_tol):
    by_model = {}
    for model_name in ["A", "B", "C"]:
        recs = [rec for rec in records_linear if rec["spec"].name == model_name]
        best = min(recs, key=lambda rec: rec["rel_rmse"])
        fixed = [rec for rec in recs if rec["spec"].tc_mode == "fixed"][0]
        free = [rec for rec in recs if rec["spec"].tc_mode == "free"][0]
        if fixed["rel_rmse"] <= (1.0 + simplicity_tol) * min(fixed["rel_rmse"], free["rel_rmse"]):
            by_model[model_name] = fixed
        else:
            by_model[model_name] = free

    best_rel = min(rec["rel_rmse"] for rec in by_model.values())
    for model_name in ["A", "B", "C"]:
        if by_model[model_name]["rel_rmse"] <= (1.0 + simplicity_tol) * best_rel:
            return by_model[model_name], by_model
    return by_model["C"], by_model


def make_model_panel_plot(data_h, ode_h, theta_u, theta_index, f_theta, record, out_path, dpi, title):
    ntheta = len(theta_u)
    fig, axes = plt.subplots(
        2,
        ntheta,
        figsize=(3.8 * ntheta, 6.2),
        squeeze=False,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        sharex="col",
    )

    y3_sem = data_h["sem"] * np.square(f_theta[theta_index]) / np.power(data_h["tp"], 1.5)
    for j, th0 in enumerate(theta_u):
        lmask = np.isclose(data_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
        omask = np.isclose(ode_h["theta0"], th0, rtol=0.0, atol=5.0e-4)

        order_l = np.argsort(data_h["tp"][lmask])
        tp_l = data_h["tp"][lmask][order_l]
        y3_l = data_h["y3"][lmask][order_l]
        y3_e = y3_sem[lmask][order_l]
        resid = record["rel_resid"][lmask][order_l]

        order_o = np.argsort(ode_h["tp"][omask])
        tp_o = ode_h["tp"][omask][order_o]
        y3_o = ode_h["y3"][omask][order_o]

        fval = float(f_theta[j])
        finf = float(record["params"]["F_inf"][j])
        tp_grid = np.logspace(np.log10(tp_l.min() * 0.95), np.log10(tp_l.max() * 1.05), 200)
        temp_data = {"tp": tp_grid}
        xi_grid = xi_model_from_params(temp_data, np.zeros_like(tp_grid, dtype=np.int64), np.array([fval]), {**record["params"], "F_inf": np.array([finf])})
        y3_grid = xi_grid * (fval ** 2) / np.power(tp_grid, 1.5)

        ax = axes[0, j]
        ax.errorbar(tp_l, y3_l, yerr=y3_e, fmt="ko", ms=3.4, lw=1.0, capsize=2, label="lattice")
        ax.plot(tp_o, y3_o, color="tab:blue", lw=1.6, label="ODE")
        ax.plot(tp_grid, y3_grid, color="tab:red", lw=1.8, label="model")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$Y_3$")
        ax.set_title(rf"$\theta_0={th0:.3g}$")
        ax.grid(alpha=0.25)

        ax = axes[1, j]
        ax.axhline(0.0, color="black", lw=1.0)
        ax.plot(tp_l, resid, "o-", color="tab:red", ms=3.6, lw=1.3)
        ax.set_xscale("log")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$\Delta\xi/\xi$")
        ax.grid(alpha=0.25)

    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_slope_plot(data_h, ode_h, theta_u, record, out_path, dpi, title):
    interps = build_ode_shifted_slope_interps(ode_h)
    interp_keys = np.array(sorted(interps.keys()), dtype=np.float64)
    ntheta = len(theta_u)
    ncols = 3 if ntheta > 4 else 2
    nrows = int(np.ceil(ntheta / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.0 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, th0 in zip(axes, theta_u):
        lmask = np.isclose(data_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
        omask = np.isclose(ode_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
        tp_l = data_h["tp"][lmask]
        y3_l = data_h["y3"][lmask]
        tp_o = ode_h["tp"][omask]
        y3_o = ode_h["y3"][omask]
        order_l = np.argsort(tp_l)
        order_o = np.argsort(tp_o)
        mids_l, slopes_l = log_slope(tp_l[order_l], y3_l[order_l])
        mids_o, slopes_o = log_slope(tp_o[order_o], y3_o[order_o])
        th_key = float(interp_keys[nearest_theta(interp_keys, th0)])
        shifted = interps[th_key](np.log(mids_l) + np.log(record["params"]["s"]))

        ax.plot(mids_o, slopes_o, color="tab:blue", lw=1.3, alpha=0.55, label="ODE")
        ax.plot(mids_l, slopes_l, "ko", ms=3.4, label="lattice")
        ax.plot(mids_l, shifted, color="tab:red", ls="--", lw=1.6, label=rf"ODE shifted, $s={record['params']['s']:.3f}$")
        ax.set_xscale("log")
        ax.set_xlabel(r"$t_p$ midpoint")
        ax.set_ylabel(r"$d\log Y_3 / d\log t_p$")
        ax.set_title(rf"$\theta_0={th0:.3g}$")
        ax.grid(alpha=0.25)

    for ax in axes[ntheta:]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_collapse_plot(data_h, theta_u, theta_index, f_theta, record, out_path, dpi, title):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    z_latt = data_h["y3"] * np.power(data_h["tp"], 1.5) / f_theta[theta_index]
    z_model = record["y3_fit"] * np.power(data_h["tp"], 1.5) / f_theta[theta_index]
    for th0 in theta_u:
        mask = np.isclose(data_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
        order = np.argsort(data_h["beta_over_h"][mask])
        ax.plot(data_h["beta_over_h"][mask][order], z_latt[mask][order], "o", ms=3.8, label=rf"lattice $\theta_0={th0:.3g}$")
        ax.plot(data_h["beta_over_h"][mask][order], z_model[mask][order], "-", lw=1.4)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta/H_*$")
    ax.set_ylabel(r"$Z=Y_3 t_p^{3/2}/F=\xi F$")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_heatmap(data_h, theta_u, record, out_path, dpi, title):
    beta_u = np.array(sorted(np.unique(data_h["beta_over_h"])), dtype=np.float64)
    resid_grid = np.full((len(theta_u), len(beta_u)), np.nan, dtype=np.float64)
    for i, th0 in enumerate(theta_u):
        for j, beta in enumerate(beta_u):
            mask = np.isclose(data_h["theta0"], th0, atol=5.0e-4, rtol=0.0) & np.isclose(data_h["beta_over_h"], beta, atol=1.0e-12, rtol=0.0)
            resid_grid[i, j] = float(record["rel_resid"][mask][0])

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    im = ax.imshow(resid_grid, aspect="auto", cmap="coolwarm", vmin=-0.08, vmax=0.08)
    ax.set_xticks(np.arange(len(beta_u)))
    ax.set_xticklabels([f"{beta:g}" for beta in beta_u], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(theta_u)))
    ax.set_yticklabels([f"{np.degrees(th):.0f}" for th in theta_u])
    ax.set_xlabel(r"$\beta/H_*$")
    ax.set_ylabel(r"$\theta_0$ [deg]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, pad=0.02, label=r"$(\xi_{\rm model}-\xi_{\rm latt})/\xi_{\rm latt}$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_bootstrap_plot(theta_u, chosen, boot, out_path, dpi, title):
    global_names = ["s", "r"]
    global_best = [chosen["params"]["s"], chosen["params"]["r"]]
    if chosen["spec"].tc_mode == "free":
        global_names.append("t_c")
        global_best.append(chosen["params"]["tc"])
    if chosen["spec"].name == "B":
        global_names.append("c")
        global_best.append(chosen["params"]["c"])
    elif chosen["spec"].name == "C":
        global_names.extend(["A", "alpha"])
        global_best.extend([chosen["params"]["A"], chosen["params"]["alpha"]])

    global_med = []
    global_lo = []
    global_hi = []
    for name in global_names:
        vals = np.asarray(boot["globals"][name], dtype=np.float64)
        global_med.append(np.nanmedian(vals))
        global_lo.append(np.nanpercentile(vals, 16.0))
        global_hi.append(np.nanpercentile(vals, 84.0))

    finf_samples = np.asarray(boot["F_inf"], dtype=np.float64)
    finf_med = np.nanmedian(finf_samples, axis=0)
    finf_lo = np.nanpercentile(finf_samples, 16.0, axis=0)
    finf_hi = np.nanpercentile(finf_samples, 84.0, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.2), gridspec_kw={"height_ratios": [1.0, 1.2]})

    xpos = np.arange(len(global_names))
    axes[0].errorbar(
        xpos,
        global_med,
        yerr=[np.array(global_med) - np.array(global_lo), np.array(global_hi) - np.array(global_med)],
        fmt="o",
        color="tab:red",
        ms=5.0,
        lw=1.5,
        capsize=3,
    )
    axes[0].plot(xpos, global_best, "ks", ms=4.5, label="best fit")
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels(global_names)
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    deg = np.degrees(theta_u)
    axes[1].errorbar(
        deg,
        finf_med,
        yerr=[finf_med - finf_lo, finf_hi - finf_med],
        fmt="o-",
        color="tab:blue",
        ms=4.8,
        lw=1.5,
        capsize=3,
    )
    axes[1].plot(deg, chosen["params"]["F_inf"], "ks", ms=4.2, label="best fit")
    axes[1].set_xlabel(r"$\theta_0$ [deg]")
    axes[1].set_ylabel(r"$F_\infty(\theta_0)$")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def bootstrap_worker(payload):
    spec = ModelSpec(*payload["spec"])
    data_h = payload["data_h"]
    theta_index = payload["theta_index"]
    f_theta = payload["f_theta"]
    finf_ode = payload["finf_ode"]
    fixed_tc = payload["fixed_tc"]
    f_inf_reg = payload["f_inf_reg"]
    alpha_reg = payload["alpha_reg"]
    huber_scale = payload["huber_scale"]
    grid_step = payload["grid_step"]
    s_bounds = payload["s_bounds"]
    idx = payload["sample_idx"]
    xbest = payload["xbest"]
    data_boot = {key: np.asarray(val[idx], dtype=np.float64) for key, val in data_h.items()}
    theta_boot = np.asarray(theta_index[idx], dtype=np.int64)
    starts = [xbest.copy()]
    starts.append(xbest.copy())
    starts[-1][0] = np.clip(starts[-1][0] * 0.95, s_bounds[0], s_bounds[1])
    starts.append(xbest.copy())
    starts[-1][0] = np.clip(starts[-1][0] * 1.05, s_bounds[0], s_bounds[1])
    try:
        rec = fit_one_spec(
            spec,
            data_boot,
            theta_boot,
            f_theta,
            xbest[-len(f_theta) :],
            finf_ode,
            fixed_tc,
            f_inf_reg,
            alpha_reg,
            huber_scale,
            grid_step,
            s_bounds,
            starts_override=starts,
        )
    except Exception:
        return None
    params = rec["params"]
    out = {
        "F_inf": params["F_inf"],
        "s": params["s"],
        "r": params["r"],
    }
    if spec.tc_mode == "free":
        out["t_c"] = params["tc"]
    if spec.name == "B":
        out["c"] = params["c"]
    if spec.name == "C":
        out["A"] = params["A"]
        out["alpha"] = params["alpha"]
    return out


def run_bootstrap(chosen, data_h, theta_index, f_theta, finf_ode, args):
    rng = np.random.default_rng(args.bootstrap_seed)
    samples = [
        rng.integers(0, len(data_h["theta0"]), size=len(data_h["theta0"]), endpoint=False, dtype=np.int64)
        for _ in range(args.bootstrap)
    ]
    payloads = [
        {
            "spec": (chosen["spec"].name, chosen["spec"].tc_mode, chosen["spec"].loss),
            "data_h": data_h,
            "theta_index": theta_index,
            "f_theta": f_theta,
            "finf_ode": finf_ode,
            "fixed_tc": args.fixed_tc,
            "f_inf_reg": args.f_inf_reg,
            "alpha_reg": args.alpha_reg,
            "huber_scale": args.huber_scale,
            "grid_step": args.grid_init_step,
            "s_bounds": (args.s_min, args.s_max),
            "sample_idx": idx,
            "xbest": chosen["x"],
        }
        for idx in samples
    ]
    results = []
    if args.bootstrap_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.bootstrap_jobs) as ex:
            for rec in ex.map(bootstrap_worker, payloads):
                if rec is not None:
                    results.append(rec)
    else:
        for payload in payloads:
            rec = bootstrap_worker(payload)
            if rec is not None:
                results.append(rec)

    globals_out = {"s": [], "r": []}
    if chosen["spec"].tc_mode == "free":
        globals_out["t_c"] = []
    if chosen["spec"].name == "B":
        globals_out["c"] = []
    if chosen["spec"].name == "C":
        globals_out["A"] = []
        globals_out["alpha"] = []
    finf_rows = []

    for rec in results:
        globals_out["s"].append(rec["s"])
        globals_out["r"].append(rec["r"])
        if "t_c" in rec:
            globals_out["t_c"].append(rec["t_c"])
        if "c" in rec:
            globals_out["c"].append(rec["c"])
        if "A" in rec:
            globals_out["A"].append(rec["A"])
        if "alpha" in rec:
            globals_out["alpha"].append(rec["alpha"])
        finf_rows.append(rec["F_inf"])

    return {
        "globals": globals_out,
        "F_inf": np.array(finf_rows, dtype=np.float64) if finf_rows else np.empty((0, len(f_theta)), dtype=np.float64),
        "n_success": len(results),
        "n_total": args.bootstrap,
    }


def format_interval(vals):
    vals = np.asarray(vals, dtype=np.float64)
    med = np.nanmedian(vals)
    lo = np.nanpercentile(vals, 16.0)
    hi = np.nanpercentile(vals, 84.0)
    return med, lo, hi


def summarize_alpha_correlation(data_h, theta_index, f_theta, record):
    if record["spec"].name != "C":
        return None
    if abs(record["params"]["alpha"] - 1.0) < 0.05:
        return None
    frow = f_theta[theta_index]
    resid = record["rel_resid"]
    corr = float(np.corrcoef(np.log(frow), resid)[0, 1])
    return corr


def save_bootstrap_samples(path, chosen, boot):
    with open(path, "w") as f:
        header = ["s", "r"]
        if chosen["spec"].tc_mode == "free":
            header.append("t_c")
        if chosen["spec"].name == "B":
            header.append("c")
        if chosen["spec"].name == "C":
            header += ["A", "alpha"]
        header += [f"F_inf_{i}" for i in range(boot["F_inf"].shape[1])]
        f.write("# " + " ".join(header) + "\n")
        for i in range(boot["F_inf"].shape[0]):
            row = [boot["globals"]["s"][i], boot["globals"]["r"][i]]
            if chosen["spec"].tc_mode == "free":
                row.append(boot["globals"]["t_c"][i])
            if chosen["spec"].name == "B":
                row.append(boot["globals"]["c"][i])
            if chosen["spec"].name == "C":
                row += [boot["globals"]["A"][i], boot["globals"]["alpha"][i]]
            row += list(boot["F_inf"][i])
            f.write(" ".join(f"{float(val):.10e}" for val in row) + "\n")


def save_summary(path, hstar, data_h, ode_h, theta_u, theta_index, f_theta, finf_guess, finf_ode, ode_baseline, records, chosen, by_model, boot):
    records_sorted = sorted(records, key=lambda rec: (rec["spec"].loss, rec["spec"].name, rec["spec"].tc_mode))
    with open(path, "w") as f:
        f.write(f"# Lattice transition-model scan at H*={float(hstar):g}\n")
        f.write("# F(theta) is reconstructed from lattice rho_noPT_data.txt and held fixed.\n")
        f.write("# Models use the explicit xi equations from the task prompt; Y3 is derived from xi via Y3=xi*F^2/tp^(3/2).\n")
        f.write(f"# ODE baseline rel_rmse_xi={ode_baseline['xi_rel_rmse']:.10e} rel_rmse_Y3={ode_baseline['y3_rel_rmse']:.10e}\n")
        f.write("# theta0 F(theta) F_inf_init_lattice F_inf_ode_prior\n")
        for th0, fval, f0, fo in zip(theta_u, f_theta, finf_guess, finf_ode):
            f.write(f"{th0:.10f} {fval:.10e} {f0:.10e} {fo:.10e}\n")
        f.write("\n")
        f.write("# Fits: chi2_rel = sum(((xi_model-xi_latt)/xi_latt)^2)\n")
        f.write("# loss model tc_mode rel_rmse chi2_rel chi2_rel_red success s r tc c A alpha\n")
        for rec in records_sorted:
            p = rec["params"]
            cval = p["c"] if p["c"] is not None else np.nan
            aval = p["A"] if p["A"] is not None else np.nan
            alph = p["alpha"] if p["alpha"] is not None else np.nan
            f.write(
                f"{rec['spec'].loss} {rec['spec'].name} {rec['spec'].tc_mode} "
                f"{rec['rel_rmse']:.10e} {rec['chi2_rel']:.10e} {rec['chi2_rel_red']:.10e} "
                f"{int(rec['success'])} {p['s']:.10e} {p['r']:.10e} {p['tc']:.10e} "
                f"{cval:.10e} {aval:.10e} {alph:.10e}\n"
            )
        f.write("\n")
        f.write("# Chosen linear models after simplicity filter\n")
        for model_name in ["A", "B", "C"]:
            rec = by_model[model_name]
            f.write(
                f"# {model_name}: tc_mode={rec['spec'].tc_mode} rel_rmse={rec['rel_rmse']:.10e} "
                f"s={rec['params']['s']:.6f} r={rec['params']['r']:.6f}\n"
            )
        f.write(
            f"# final_choice model={chosen['spec'].name} tc_mode={chosen['spec'].tc_mode} "
            f"loss={chosen['spec'].loss} rel_rmse={chosen['rel_rmse']:.10e}\n\n"
        )

        f.write("# Final chosen per-theta F_inf\n")
        f.write("# theta0 F_inf_best\n")
        for th0, finf in zip(theta_u, chosen["params"]["F_inf"]):
            f.write(f"{th0:.10f} {finf:.10e}\n")
        f.write("\n")

        f.write("# Final chosen xi at smallest/largest tp per theta\n")
        f.write("# theta0 tp_small xi_latt_small xi_model_small tp_large xi_latt_large xi_model_large\n")
        for th0 in theta_u:
            mask = np.isclose(data_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
            order = np.argsort(data_h["tp"][mask])
            tp = data_h["tp"][mask][order]
            xi_l = data_h["xi"][mask][order]
            xi_m = chosen["xi_fit"][mask][order]
            f.write(
                f"{th0:.10f} {tp[0]:.10e} {xi_l[0]:.10e} {xi_m[0]:.10e} "
                f"{tp[-1]:.10e} {xi_l[-1]:.10e} {xi_m[-1]:.10e}\n"
            )
        f.write("\n")

        alpha_corr = summarize_alpha_correlation(data_h, theta_index, f_theta, chosen)
        if alpha_corr is not None:
            f.write(f"# residual_vs_logF_corr {alpha_corr:.10e}\n\n")

        f.write("# Bootstrap summary for final chosen model\n")
        f.write(f"# n_success={boot['n_success']} n_total={boot['n_total']}\n")
        for name, vals in boot["globals"].items():
            med, lo, hi = format_interval(vals)
            f.write(f"# {name}: median={med:.10e} p16={lo:.10e} p84={hi:.10e}\n")
        if boot["F_inf"].shape[0] > 0:
            f.write("# theta0 F_inf_median F_inf_p16 F_inf_p84\n")
            for th0, col in zip(theta_u, boot["F_inf"].T):
                med, lo, hi = format_interval(col)
                f.write(f"{th0:.10f} {med:.10e} {lo:.10e} {hi:.10e}\n")
        f.write("\n")

        global_parts = [f"s={chosen['params']['s']:.3f}", f"r={chosen['params']['r']:.3f}"]
        if chosen["spec"].tc_mode == "free":
            global_parts.append(f"t_c={chosen['params']['tc']:.3f}")
        if chosen["spec"].name == "B":
            global_parts.append(f"c={chosen['params']['c']:.3f}")
        elif chosen["spec"].name == "C":
            global_parts.append(f"A={chosen['params']['A']:.3f}")
            global_parts.append(f"alpha={chosen['params']['alpha']:.3f}")

        before = ode_baseline["xi_rel_rmse"]
        after = chosen["rel_rmse"]
        f.write(
            "SUMMARY: "
            f"model {chosen['spec'].name} with t_c {chosen['spec'].tc_mode} was selected at H*={float(hstar):g}; "
            f"best-fit globals are {', '.join(global_parts)}. "
            f"The first six F_inf(theta) values are "
            + ", ".join(
                f"theta={th0:.3f}: {finf:.3e}" for th0, finf in zip(theta_u[:6], chosen["params"]["F_inf"][:6])
            )
            + f". Relative RMSE improves from the direct-ODE baseline {before:.3e} to {after:.3e}. "
            + (
                "Bootstrap uncertainties indicate a significant theta-scaling amplitude effect."
                if chosen["spec"].name == "C" and boot["n_success"] > 0 and abs(np.nanmedian(boot["globals"]["alpha"]) - 1.0) > 0.05
                else "Bootstrap uncertainties are consistent with a predominantly transition-shape correction rather than a large theta-dependent amplitude renormalization."
            )
            + "\n"
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

    for hstar in args.target_h:
        tag = f"H{float(hstar):.1f}".replace(".", "p")
        hdir = outdir / tag
        hdir.mkdir(parents=True, exist_ok=True)

        data_h = select_h(lat_data, hstar)
        ode_h = select_h(ode_data, hstar)
        theta_u, theta_index, f_theta = build_theta_arrays(data_h)
        finf_guess = estimate_finf_guess(data_h, theta_u, source_key="y3", top_n=3)
        finf_ode = estimate_finf_guess(ode_h, theta_u, source_key="y3", top_n=5)

        ode_xi_interp = interpolate_ode_on_lattice(data_h, ode_h, "xi")
        ode_y3_interp = interpolate_ode_on_lattice(data_h, ode_h, "y3")
        ode_baseline = {
            "xi_rel_rmse": rel_rmse(data_h["xi"], ode_xi_interp),
            "y3_rel_rmse": rel_rmse(data_h["y3"], ode_y3_interp),
        }

        records = []
        for loss in ["linear", "huber"]:
            for model_name in ["A", "B", "C"]:
                for tc_mode in ["fixed", "free"]:
                    spec = ModelSpec(model_name, tc_mode, loss)
                    rec = fit_one_spec(
                        spec,
                        data_h,
                        theta_index,
                        f_theta,
                        finf_guess,
                        finf_ode,
                        args.fixed_tc,
                        args.f_inf_reg,
                        args.alpha_reg,
                        args.huber_scale,
                        args.grid_init_step,
                        (args.s_min, args.s_max),
                    )
                    records.append(rec)

                    if loss == "linear":
                        stem = f"model_{model_name}_tc_{tc_mode}"
                        make_model_panel_plot(
                            data_h,
                            ode_h,
                            theta_u,
                            theta_index,
                            f_theta,
                            rec,
                            hdir / f"{stem}_overlay.png",
                            args.dpi,
                            rf"Lattice $Y_3$ fit: model {model_name}, $t_c$ {tc_mode}, $H_*={float(hstar):g}$",
                        )
                        make_slope_plot(
                            data_h,
                            ode_h,
                            theta_u,
                            rec,
                            hdir / f"{stem}_slopes.png",
                            args.dpi,
                            rf"Local $Y_3$ slopes with fitted $s$ shift, model {model_name}, $H_*={float(hstar):g}$",
                        )
                        make_collapse_plot(
                            data_h,
                            theta_u,
                            theta_index,
                            f_theta,
                            rec,
                            hdir / f"{stem}_collapse.png",
                            args.dpi,
                            rf"Collapse diagnostic $Z=\xi F$ for model {model_name}, $H_*={float(hstar):g}$",
                        )
                        make_heatmap(
                            data_h,
                            theta_u,
                            rec,
                            hdir / f"{stem}_heatmap.png",
                            args.dpi,
                            rf"Residual heatmap for model {model_name}, $t_c$ {tc_mode}, $H_*={float(hstar):g}$",
                        )

        records_linear = [rec for rec in records if rec["spec"].loss == "linear"]
        chosen, by_model = choose_simple_model(records_linear, args.simplicity_tol)
        boot = run_bootstrap(chosen, data_h, theta_index, f_theta, finf_ode, args)

        make_bootstrap_plot(
            theta_u,
            chosen,
            boot,
            hdir / "bootstrap_summary.png",
            args.dpi,
            rf"Bootstrap summary for chosen model at $H_*={float(hstar):g}$",
        )
        save_bootstrap_samples(hdir / "bootstrap_samples.txt", chosen, boot)
        save_summary(
            hdir / "fit_summary.txt",
            hstar,
            data_h,
            ode_h,
            theta_u,
            theta_index,
            f_theta,
            finf_guess,
            finf_ode,
            ode_baseline,
            records,
            chosen,
            by_model,
            boot,
        )
        print(hdir / "fit_summary.txt")


if __name__ == "__main__":
    main()
