#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fit_vw_amplitude as base_fit


OUTDIR_DEFAULT = ROOT / "results_vw_timewarp_refit_h"
DEFAULT_H_VALUES = [1.5, 2.0]
DEFAULT_VW_TAGS = ["v3", "v5", "v7", "v9"]


def to_native(obj):
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_native(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_native(payload), indent=2, sort_keys=True))


def error_exit(outdir: Path, exc: Exception):
    payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
    save_json(outdir / "_error.json", payload)
    print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True))
    return 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refit analytic h(theta) amplitudes together with the time-warp-only lattice model."
    )
    parser.add_argument("--outdir", type=str, default=str(OUTDIR_DEFAULT))
    parser.add_argument("--vw-folders", nargs="*", default=DEFAULT_VW_TAGS)
    parser.add_argument("--h-values", type=float, nargs="+", default=DEFAULT_H_VALUES)
    parser.add_argument("--rho", type=str, default="")
    parser.add_argument("--fit-table", type=str, default="results_hf/fit_table.csv")
    parser.add_argument("--timewarp-json", type=str, default="results_vw_timewarp/params_optionB.json")
    parser.add_argument("--t-osc", type=float, default=1.5)
    parser.add_argument("--tc", type=float, default=1.5)
    parser.add_argument("--weight-f0", type=float, default=None)
    parser.add_argument("--tp-min", type=float, default=None)
    parser.add_argument("--tp-max", type=float, default=None)
    return parser.parse_args()


def h_theta(theta):
    theta = np.asarray(theta, dtype=np.float64)
    cos_half = np.cos(theta / 2.0)
    h = np.full_like(theta, np.nan, dtype=np.float64)
    good = np.abs(cos_half) > 1.0e-12
    h[good] = np.log(np.e / np.maximum(cos_half[good] ** 2, 1.0e-300))
    return h


def rel_rmse(y_true, y_fit):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square((y_fit - y_true) / np.maximum(np.abs(y_true), 1.0e-12)))))


def aic_bic(resid, k):
    resid = np.asarray(resid, dtype=np.float64)
    n = max(int(resid.size), 1)
    rss = max(float(np.sum(np.square(resid))), 1.0e-18)
    aic = n * math.log(rss / n) + 2.0 * k
    bic = n * math.log(rss / n) + k * math.log(n)
    return float(aic), float(bic)


def load_fit_table(path: Path, t_osc: float):
    df = pd.read_csv(path)
    rows = {str(row["dataset"]).strip(): row for _, row in df.iterrows()}
    if "noPT" not in rows or "Finf" not in rows:
        raise RuntimeError(f"{path} must contain dataset rows noPT and Finf.")
    return {
        "A0": float(rows["noPT"]["A"]),
        "gamma0": float(rows["noPT"]["gamma"]),
        "Ainf": float(rows["Finf"]["A"]) * (float(t_osc) ** 1.5),
        "gamma_inf": float(rows["Finf"]["gamma"]),
    }


def load_timewarp_optionB(path: Path):
    payload = json.loads(path.read_text())
    if "scales" in payload:
        return payload
    if "optionB" in payload:
        return payload["optionB"]
    raise RuntimeError(f"Could not find optionB time-warp parameters in {path}")


def prepare_dataframe(args, outdir: Path):
    load_args = SimpleNamespace(
        rho=args.rho,
        vw_folders=args.vw_folders,
        h_values=args.h_values,
        tp_min=args.tp_min,
        tp_max=args.tp_max,
        bootstrap=0,
        n_jobs=1,
        reg_Finf=0.0,
        tc0=args.tc,
        fix_tc=True,
        dpi=220,
        outdir=str(outdir),
    )
    df, f0_table, theta_values = base_fit.prepare_dataframe(load_args, outdir)
    df["h"] = h_theta(df["theta"].to_numpy(dtype=np.float64))
    bad = ~np.isfinite(df["h"]) | (df["h"] <= 0.0)
    if np.any(bad):
        df = df.loc[~bad].copy()
    theta_values = np.sort(df["theta"].unique())
    return df.reset_index(drop=True), f0_table, theta_values


def build_f0_anchor(f0_table):
    out = f0_table.copy()
    out["h"] = h_theta(out["theta"].to_numpy(dtype=np.float64))
    out = out[np.isfinite(out["h"]) & np.isfinite(out["F0"]) & (out["h"] > 0.0) & (out["F0"] > 0.0)].copy()
    return out.sort_values("theta").reset_index(drop=True)


def unpack(params, n_scales):
    idx = 0
    lnA0 = float(params[idx])
    gamma0 = float(params[idx + 1])
    lnAinf = float(params[idx + 2])
    gamma_inf = float(params[idx + 3])
    idx += 4
    scales = np.asarray(params[idx : idx + n_scales], dtype=np.float64)
    idx += n_scales
    r = float(params[idx])
    return lnA0, gamma0, lnAinf, gamma_inf, scales, r


def build_param_vector(init, vw_values):
    x0 = np.array(
        [
            math.log(init["A0"]),
            init["gamma0"],
            math.log(init["Ainf"]),
            init["gamma_inf"],
            *[init["scales"][f"{vw:.1f}"] for vw in vw_values],
            init["r"],
        ],
        dtype=np.float64,
    )
    lower = np.array(
        [
            math.log(1.0e-8),
            0.2,
            math.log(1.0e-10),
            0.2,
            *([0.5] * len(vw_values)),
            0.1,
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            math.log(1.0),
            3.0,
            math.log(1.0e-1),
            3.0,
            *([1.5] * len(vw_values)),
            20.0,
        ],
        dtype=np.float64,
    )
    return x0, lower, upper


def model_components(params, df, vw_values, t_osc, tc):
    lnA0, gamma0, lnAinf, gamma_inf, scales, r = unpack(params, len(vw_values))
    A0 = math.exp(lnA0)
    Ainf = math.exp(lnAinf)
    h = df["h"].to_numpy(dtype=np.float64)
    F0 = A0 * np.power(h, gamma0)
    Finf = Ainf * np.power(h, gamma_inf)
    scale_map = {float(vw): float(sc) for vw, sc in zip(vw_values, scales)}
    scale = np.asarray([scale_map[float(vw)] for vw in df["v_w"].to_numpy(dtype=np.float64)], dtype=np.float64)
    tp = df["tp"].to_numpy(dtype=np.float64)
    tp_scaled = tp * scale
    plateau = np.power(tp / t_osc, 1.5) * Finf / np.maximum(F0 * F0, 1.0e-18)
    transient = np.power(np.maximum(scale, 1.0e-18), -1.5) / (
        1.0 + np.power(np.maximum(tp_scaled, 1.0e-18) / max(tc, 1.0e-18), r)
    )
    xi_model = plateau + transient
    return xi_model, F0, Finf, scale


def residual_vector(params, df, f0_anchor, vw_values, t_osc, tc, weight_f0):
    xi_model, _, _, _ = model_components(params, df, vw_values, t_osc, tc)
    resid_xi = (xi_model - df["xi"].to_numpy(dtype=np.float64)) / np.maximum(df["xi"].to_numpy(dtype=np.float64), 1.0e-12)

    lnA0, gamma0, _, _, _, _ = unpack(params, len(vw_values))
    A0 = math.exp(lnA0)
    f0_model = A0 * np.power(f0_anchor["h"].to_numpy(dtype=np.float64), gamma0)
    resid_f0 = weight_f0 * (f0_model - f0_anchor["F0"].to_numpy(dtype=np.float64)) / np.maximum(f0_anchor["F0"].to_numpy(dtype=np.float64), 1.0e-12)
    return np.concatenate([resid_xi, resid_f0])


def per_theta_rmse(df, xi_model):
    work = df.copy()
    work["yfit"] = xi_model
    work["frac"] = (work["yfit"] - work["xi"]) / work["xi"]
    rows = []
    for theta, sub in work.groupby("theta"):
        rows.append(
            {
                "theta": float(theta),
                "rel_rmse": float(np.sqrt(np.mean(np.square(sub["frac"])))),
                "mean_frac": float(sub["frac"].mean()),
                "max_abs": float(np.max(np.abs(sub["frac"]))),
            }
        )
    return pd.DataFrame(rows).sort_values("theta").reset_index(drop=True)


def plot_h_fits(f0_anchor, fit, old_fit, outpath):
    h_grid = np.geomspace(f0_anchor["h"].min(), f0_anchor["h"].max(), 300)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    lnA0, gamma0, lnAinf, gamma_inf, _, _ = unpack(fit["params"], len(fit["vw_values"]))
    A0 = math.exp(lnA0)
    Ainf = math.exp(lnAinf)

    old_A0 = old_fit["A0"]
    old_g0 = old_fit["gamma0"]
    old_Ainf = old_fit["Ainf"]
    old_ginf = old_fit["gamma_inf"]

    axes[0].scatter(f0_anchor["h"], f0_anchor["F0"], s=35, color="black", label="F0 table")
    axes[0].plot(h_grid, A0 * np.power(h_grid, gamma0), lw=2.2, color="tab:blue", label="refit")
    axes[0].plot(h_grid, old_A0 * np.power(h_grid, old_g0), lw=1.8, color="tab:orange", ls="--", label="old")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$h(\theta)$")
    axes[0].set_ylabel(r"$F_0(\theta)$")
    axes[0].set_title(r"$F_0 = A_0 h^{\gamma_0}$")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    theta_ref = np.asarray(fit["theta_values"], dtype=np.float64)
    h_ref = h_theta(theta_ref)
    finf_refit = Ainf * np.power(h_ref, gamma_inf)
    finf_old = old_Ainf * np.power(h_ref, old_ginf)
    axes[1].scatter(h_ref, finf_refit, s=35, color="tab:blue", label="refit Finf")
    axes[1].scatter(h_ref, finf_old, s=35, color="tab:orange", marker="x", label="old Finf")
    axes[1].plot(h_grid, Ainf * np.power(h_grid, gamma_inf), lw=2.2, color="tab:blue")
    axes[1].plot(h_grid, old_Ainf * np.power(h_grid, old_ginf), lw=1.8, color="tab:orange", ls="--")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$h(\theta)$")
    axes[1].set_ylabel(r"$F_\infty(\theta)$")
    axes[1].set_title(r"$F_\infty = A_\infty h^{\gamma_\infty}$")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def xi_curve(beta_grid, theta, H_value, vw, fit_payload, t_osc, tc):
    perc = base_fit.PercolationCache()
    tp_grid = np.asarray([perc.get(float(H_value), float(beta), float(vw)) for beta in beta_grid], dtype=np.float64)
    tp_grid = np.maximum(tp_grid, 1.0e-18)
    h_val = float(h_theta(np.asarray([theta], dtype=np.float64))[0])
    A0 = fit_payload["A0"]
    gamma0 = fit_payload["gamma0"]
    Ainf = fit_payload["Ainf"]
    gamma_inf = fit_payload["gamma_inf"]
    F0 = A0 * (h_val ** gamma0)
    Finf = Ainf * (h_val ** gamma_inf)
    scale = fit_payload["scales"][f"{vw:.1f}"]
    tp_scaled = tp_grid * scale
    plateau = np.power(tp_grid / t_osc, 1.5) * Finf / max(F0 * F0, 1.0e-18)
    transient = np.power(max(scale, 1.0e-18), -1.5) / (
        1.0 + np.power(tp_scaled / max(tc, 1.0e-18), fit_payload["r"])
    )
    return plateau + transient


def plot_xi_overlays(df, fit_payload, t_osc, tc, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    theta_values = np.sort(df["theta"].unique())
    vw_values = np.sort(df["v_w"].unique())
    colors = {vw: plt.get_cmap("viridis")(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}

    files = []
    for H_value in np.sort(df["H"].unique()):
        fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex=True, sharey=False)
        axes = axes.flatten()
        sub_h = df[np.isclose(df["H"], float(H_value))].copy()
        for ax, theta in zip(axes, theta_values):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta))].copy()
            for vw in vw_values:
                sub_vw = sub[np.isclose(sub["v_w"], float(vw))].sort_values("beta_over_H").copy()
                if sub_vw.empty:
                    continue
                ax.scatter(sub_vw["beta_over_H"], sub_vw["xi"], s=28, color=colors[vw], alpha=0.9, label=rf"$v_w={vw:.1f}$")
                beta_grid = np.geomspace(sub_vw["beta_over_H"].min(), sub_vw["beta_over_H"].max(), 250)
                ax.plot(beta_grid, xi_curve(beta_grid, float(theta), float(H_value), float(vw), fit_payload, t_osc, tc), color=colors[vw], lw=2.0)
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$\beta/H_*$")
            ax.set_ylabel(r"$\xi$")
        handles, labels = axes[0].get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=4, frameon=False, fontsize=9)
        fig.suptitle(
            rf"Time-warp + refit $h(\theta)$, $H_*={H_value:.1f}$"
            + "\n"
            + rf"$r={fit_payload['r']:.3f}$, $t_c={tc:.3f}$",
            y=0.98,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
        outfile = outdir / f"xi_vs_betaH_refit_h_H{str(H_value).replace('.', 'p')}.png"
        fig.savefig(outfile, dpi=220)
        plt.close(fig)
        files.append(str(outfile))
    return files


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df, f0_table, theta_values = prepare_dataframe(args, outdir)
    f0_anchor = build_f0_anchor(f0_table)

    fit_table_path = Path(args.fit_table).resolve()
    analytic_init = load_fit_table(fit_table_path, args.t_osc)

    tw_path = Path(args.timewarp_json).resolve()
    tw_option = load_timewarp_optionB(tw_path)
    vw_values = np.sort(df["v_w"].unique())

    init = {
        "A0": analytic_init["A0"],
        "gamma0": analytic_init["gamma0"],
        "Ainf": analytic_init["Ainf"],
        "gamma_inf": analytic_init["gamma_inf"],
        "scales": {f"{float(vw):.1f}": float(tw_option["scales"][f"{float(vw):.1f}"]) for vw in vw_values},
        "r": float(tw_option["r"]),
    }
    x0, lower, upper = build_param_vector(init, vw_values)
    if args.weight_f0 is None:
        weight_f0 = math.sqrt(len(df) / max(len(f0_anchor), 1))
    else:
        weight_f0 = float(args.weight_f0)

    fun = lambda p: residual_vector(p, df, f0_anchor, vw_values, args.t_osc, args.tc, weight_f0)
    huber = least_squares(fun, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=40000)
    final = least_squares(fun, huber.x, bounds=(lower, upper), loss="linear", max_nfev=40000)

    xi_model, _, _, _ = model_components(final.x, df, vw_values, args.t_osc, args.tc)
    frac = (xi_model - df["xi"].to_numpy(dtype=np.float64)) / df["xi"].to_numpy(dtype=np.float64)
    lnA0, gamma0, lnAinf, gamma_inf, scales, r = unpack(final.x, len(vw_values))
    payload = {
        "status": "ok",
        "A0": float(math.exp(lnA0)),
        "gamma0": float(gamma0),
        "Ainf": float(math.exp(lnAinf)),
        "gamma_inf": float(gamma_inf),
        "r": float(r),
        "t_c": float(args.tc),
        "scales": {f"{float(vw):.1f}": float(sc) for vw, sc in zip(vw_values, scales)},
        "rel_rmse": rel_rmse(df["xi"].to_numpy(dtype=np.float64), xi_model),
        "AIC": aic_bic(frac, len(final.x))[0],
        "BIC": aic_bic(frac, len(final.x))[1],
        "weight_f0": float(weight_f0),
        "params": final.x.tolist(),
        "theta_values": [float(theta) for theta in theta_values],
        "source_fit_table": str(fit_table_path),
        "source_timewarp": str(tw_path),
    }
    save_json(outdir / "summary.json", payload)

    rmse_df = per_theta_rmse(df, xi_model)
    rmse_df.to_csv(outdir / "per_theta_rmse.csv", index=False)
    plot_h_fits(
        f0_anchor,
        {"params": final.x, "theta_values": theta_values, "vw_values": vw_values},
        analytic_init,
        outdir / "refit_h_functions.png",
    )
    xi_files = plot_xi_overlays(
        df,
        {
            "A0": float(math.exp(lnA0)),
            "gamma0": float(gamma0),
            "Ainf": float(math.exp(lnAinf)),
            "gamma_inf": float(gamma_inf),
            "r": float(r),
            "scales": {f"{float(vw):.1f}": float(sc) for vw, sc in zip(vw_values, scales)},
        },
        args.t_osc,
        args.tc,
        outdir,
    )

    final_summary = {
        "status": "ok",
        "old_timewarp_rel_rmse": float(tw_option["rel_rmse"]),
        "new_rel_rmse": payload["rel_rmse"],
        "old_r": float(tw_option["r"]),
        "new_r": float(r),
        "old_scales": {k: float(v) for k, v in tw_option["scales"].items()},
        "new_scales": payload["scales"],
        "A0": payload["A0"],
        "gamma0": payload["gamma0"],
        "Ainf": payload["Ainf"],
        "gamma_inf": payload["gamma_inf"],
        "t_c": float(args.tc),
        "weight_f0": float(weight_f0),
        "per_theta_rmse_csv": str(outdir / "per_theta_rmse.csv"),
        "overlay_files": xi_files,
    }
    save_json(outdir / "final_summary.json", final_summary)
    print(json.dumps(to_native(final_summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        outdir = Path(OUTDIR_DEFAULT).resolve()
        raise SystemExit(error_exit(outdir, exc))
