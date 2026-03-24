#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import least_squares

import collapse_and_fit_fanh_tosc as collapse
import fit_lattice_fixed_ode_amplitudes as shared


ROOT = Path(__file__).resolve().parent
OUTDIR_DEFAULT = "results_lattice_fixed_ode_amplitudes_vw0p9_H1p0H1p5H2p0_beta0_theta_tc"
REFERENCE_SUMMARY_CANDIDATES = [
    ROOT / "results_lattice_fixed_ode_amplitudes_vw0p9_H1p0H1p5H2p0_beta0" / "final_summary.json",
    ROOT / "results_lattice_fixed_ode_amplitudes_vw0p9_H1p0H1p5H2p0" / "final_summary.json",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit lattice collapse with ODE-fixed F0(theta), F_inf(theta), beta fixed, and per-theta t_c(theta0)."
    )
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--h-values", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    p.add_argument("--fixed-beta", type=float, default=0.0)
    p.add_argument("--t-osc", type=float, default=1.5)
    p.add_argument("--ode-summary", type=str, default=str(shared.ODE_SUMMARY_DEFAULT))
    p.add_argument("--reference-summary", type=str, default="")
    p.add_argument("--bootstrap", type=int, default=120)
    p.add_argument("--n-jobs", type=int, default=min(8, max(1, (shared.os_cpu_count() or 1))))
    p.add_argument("--tc-max", type=float, default=20.0)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--outdir", type=str, default=OUTDIR_DEFAULT)
    return p.parse_args()


def resolve_reference_summary(user_value: str) -> Path | None:
    if user_value:
        path = Path(user_value).resolve()
        return path if path.exists() else None
    for path in REFERENCE_SUMMARY_CANDIDATES:
        if path.exists():
            return path.resolve()
    return None


def h_alt(theta_values: np.ndarray) -> np.ndarray:
    theta_values = np.asarray(theta_values, dtype=np.float64)
    return np.log(np.e / np.maximum(1.0 - np.square(theta_values / np.pi), 1.0e-12))


def error_exit(outdir: Path, exc: Exception):
    payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
    outdir.mkdir(parents=True, exist_ok=True)
    shared.save_json(outdir / "_error.json", payload)
    print(json.dumps(payload, sort_keys=True))


def model_eval(meta, theta_values: np.ndarray, fixed_beta: float, r: float, tc_theta: np.ndarray, c_calib: float = 1.0):
    theta_idx = np.asarray(meta["theta_idx"], dtype=np.int64)
    x = meta["tp"] * np.power(meta["H"], float(fixed_beta))
    plateau = np.power(np.maximum(x / meta["t_osc"], 1.0e-18), 1.5) * meta["F_inf"] / meta["F0_sq"]
    transient = 1.0 / (1.0 + np.power(np.maximum(x / np.maximum(tc_theta[theta_idx], 1.0e-18), 1.0e-18), float(r)))
    return float(c_calib) * (plateau + transient)


def fit_case(meta, theta_values: np.ndarray, fixed_beta: float, init_r: float, init_tc: np.ndarray, with_calib: bool, tc_max: float):
    ntheta = len(theta_values)
    init_tc = np.asarray(init_tc, dtype=np.float64)
    if init_tc.size != ntheta:
        raise ValueError("Initial tc_theta array must match theta grid.")
    if with_calib:
        x0 = np.concatenate([[init_r], init_tc, [1.0]]).astype(np.float64)
        lower = np.concatenate([[0.1], np.full(ntheta, 0.1, dtype=np.float64), [0.1]]).astype(np.float64)
        upper = np.concatenate([[20.0], np.full(ntheta, float(tc_max), dtype=np.float64), [10.0]]).astype(np.float64)
    else:
        x0 = np.concatenate([[init_r], init_tc]).astype(np.float64)
        lower = np.concatenate([[0.1], np.full(ntheta, 0.1, dtype=np.float64)]).astype(np.float64)
        upper = np.concatenate([[20.0], np.full(ntheta, float(tc_max), dtype=np.float64)]).astype(np.float64)

    def resid(par: np.ndarray) -> np.ndarray:
        r = float(par[0])
        tc_theta = np.asarray(par[1 : 1 + ntheta], dtype=np.float64)
        c_calib = float(par[-1]) if with_calib else 1.0
        y_fit = model_eval(meta, theta_values, fixed_beta, r, tc_theta, c_calib=c_calib)
        return (y_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)

    huber = least_squares(resid, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=40000)
    final = least_squares(resid, huber.x, bounds=(lower, upper), loss="linear", max_nfev=40000)
    r = float(final.x[0])
    tc_theta = np.asarray(final.x[1 : 1 + ntheta], dtype=np.float64)
    c_calib = float(final.x[-1]) if with_calib else 1.0
    y_fit = model_eval(meta, theta_values, fixed_beta, r, tc_theta, c_calib=c_calib)
    frac_resid = (y_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    aic, bic = shared.aic_bic(frac_resid, len(final.x))
    out = {
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "beta": float(fixed_beta),
        "r": float(r),
        "tc_theta": tc_theta,
        "rel_rmse": shared.rel_rmse(meta["xi"], y_fit),
        "AIC": float(aic),
        "BIC": float(bic),
        "n_points": int(meta["xi"].size),
        "n_params": int(len(final.x)),
        "result_x": np.asarray(final.x, dtype=np.float64),
        "y_fit": y_fit,
        "frac_resid": frac_resid,
    }
    if with_calib:
        out["c_calib"] = float(c_calib)
    return out


def bootstrap_case(meta, theta_values: np.ndarray, fixed_beta: float, fit_payload: dict, nboot: int, n_jobs: int, with_calib: bool, tc_max: float):
    if nboot <= 0:
        return {"status": "skipped", "n_samples": 0}
    y_fit = fit_payload["y_fit"]
    resid = meta["xi"] - y_fit
    init_tc = np.asarray(fit_payload["tc_theta"], dtype=np.float64)
    init_r = float(fit_payload["r"])

    def one(seed):
        rng = np.random.default_rng(seed)
        boot_y = np.maximum(y_fit + resid[rng.integers(0, resid.size, size=resid.size)], 1.0e-12)
        boot_meta = dict(meta)
        boot_meta["xi"] = boot_y
        payload = fit_case(boot_meta, theta_values, fixed_beta, init_r, init_tc, with_calib=with_calib, tc_max=tc_max)
        if not payload["success"]:
            return None
        row = {
            "r": float(payload["r"]),
            "tc_theta": np.asarray(payload["tc_theta"], dtype=np.float64),
        }
        if with_calib:
            row["c_calib"] = float(payload["c_calib"])
        return row

    rows = Parallel(n_jobs=n_jobs)(delayed(one)(38127 + i) for i in range(nboot))
    rows = [row for row in rows if row is not None]
    if not rows:
        return {"status": "no_successful_bootstrap_samples", "n_samples": 0}

    out = {
        "status": "ok",
        "n_samples": int(len(rows)),
        "beta_fixed": float(fixed_beta),
        "tc_max": float(tc_max),
        "r": {},
        "tc_theta": {},
    }
    r_vals = np.array([row["r"] for row in rows], dtype=np.float64)
    out["r"] = {
        "p16": float(np.percentile(r_vals, 16)),
        "p50": float(np.percentile(r_vals, 50)),
        "p84": float(np.percentile(r_vals, 84)),
    }
    if with_calib:
        c_vals = np.array([row["c_calib"] for row in rows], dtype=np.float64)
        out["c_calib"] = {
            "p16": float(np.percentile(c_vals, 16)),
            "p50": float(np.percentile(c_vals, 50)),
            "p84": float(np.percentile(c_vals, 84)),
        }
    tc_arr = np.stack([row["tc_theta"] for row in rows], axis=0)
    for i, theta in enumerate(theta_values):
        vals = tc_arr[:, i]
        out["tc_theta"][f"{float(theta):.10f}"] = {
            "p16": float(np.percentile(vals, 16)),
            "p50": float(np.percentile(vals, 50)),
            "p84": float(np.percentile(vals, 84)),
        }
    return out


def plot_overlay(df, meta, theta_values: np.ndarray, fit_payload: dict, outpath: Path, title: str, dpi: int):
    beta = float(fit_payload["beta"])
    r = float(fit_payload["r"])
    tc_theta = np.asarray(fit_payload["tc_theta"], dtype=np.float64)
    c_calib = float(fit_payload.get("c_calib", 1.0))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    colors = {float(h): c for h, c in zip(sorted(df["H"].unique()), ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"])}
    for ax, theta in zip(axes.ravel(), theta_values):
        sub = df[np.isclose(df["theta"], float(theta), atol=1.0e-8)].copy().sort_values("tp")
        x_data = sub["tp"].to_numpy(dtype=np.float64) * np.power(sub["H"].to_numpy(dtype=np.float64), beta)
        order = np.argsort(x_data)
        x_data = x_data[order]
        y_data = sub["xi"].to_numpy(dtype=np.float64)[order]
        h_data = sub["H"].to_numpy(dtype=np.float64)[order]
        for h in sorted(sub["H"].unique()):
            mask = np.isclose(h_data, float(h), atol=1.0e-8)
            ax.scatter(x_data[mask], y_data[mask], s=22, color=colors[float(h)], alpha=0.9, label=f"$H_*={h:g}$")
        x_grid = np.geomspace(max(x_data.min() * 0.95, 1.0e-4), x_data.max() * 1.05, 300)
        idx = collapse.nearest_theta(theta_values, float(theta))
        plateau = np.power(np.maximum(x_grid / meta["t_osc"], 1.0e-18), 1.5) * meta["F_inf"][meta["theta_idx"] == idx][0] / max(
            meta["F0_sq"][meta["theta_idx"] == idx][0],
            1.0e-18,
        )
        transient = 1.0 / (1.0 + np.power(np.maximum(x_grid / max(tc_theta[idx], 1.0e-18), 1.0e-18), r))
        ax.plot(x_grid, c_calib * (plateau + transient), color="black", lw=2.0)
        ax.set_xscale("log")
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$t_p$" if np.isclose(beta, 0.0, atol=1.0e-12) else r"$x=t_p H^\beta$")
        ax.set_ylabel(r"$\xi$")
        ax.grid(True, alpha=0.2)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.suptitle(title, fontsize=14)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_raw_betaH(df, meta, theta_values: np.ndarray, fit_payload: dict, outdir: Path, stem: str, dpi: int):
    beta = float(fit_payload["beta"])
    r = float(fit_payload["r"])
    tc_theta = np.asarray(fit_payload["tc_theta"], dtype=np.float64)
    c_calib = float(fit_payload.get("c_calib", 1.0))
    if "v_w" in df.columns:
        vw_values = sorted(df["v_w"].unique())
    else:
        vw_values = [0.9]
    colors = {float(vw): c for vw, c in zip(vw_values, ["#440154", "#31688e", "#35b779", "#fde725"])}
    for h in sorted(df["H"].unique()):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        sub_h = df[np.isclose(df["H"], float(h), atol=1.0e-8)].copy()
        for ax, theta in zip(axes.ravel(), theta_values):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=1.0e-8)].copy().sort_values("beta_over_H")
            idx = collapse.nearest_theta(theta_values, float(theta))
            if "v_w" in sub.columns:
                groups = [
                    (float(vw), sub[np.isclose(sub["v_w"], float(vw), atol=1.0e-8)].copy())
                    for vw in sorted(sub["v_w"].unique())
                ]
            else:
                groups = [(0.9, sub.copy())]
            for vw, hh in groups:
                tp = hh["tp"].to_numpy(dtype=np.float64)
                x = tp * np.power(h, beta)
                y_fit = c_calib * (
                    np.power(np.maximum(x / meta["t_osc"], 1.0e-18), 1.5) * meta["F_inf"][meta["theta_idx"] == idx][0] / max(
                        meta["F0_sq"][meta["theta_idx"] == idx][0],
                        1.0e-18,
                    )
                    + 1.0 / (1.0 + np.power(np.maximum(x / max(tc_theta[idx], 1.0e-18), 1.0e-18), r))
                )
                ax.scatter(hh["beta_over_H"], hh["xi"], s=22, color=colors[float(vw)], alpha=0.85)
                ax.plot(hh["beta_over_H"], y_fit, color=colors[float(vw)], lw=1.8)
            ax.set_xscale("log")
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$\beta/H_*$")
            ax.set_ylabel(r"$\xi$")
            ax.grid(True, alpha=0.2)
        fig.suptitle(f"{stem}, H*={h:g}", fontsize=14)
        fig.savefig(outdir / f"{stem}_H{str(h).replace('.', 'p')}.png", dpi=dpi)
        plt.close(fig)


def plot_tc_theta(theta_values: np.ndarray, fit_payloads: list[tuple[str, dict]], outdir: Path, dpi: int):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), constrained_layout=True)
    hvals = h_alt(theta_values)
    for label, payload in fit_payloads:
        tc_theta = np.asarray(payload["tc_theta"], dtype=np.float64)
        axes[0].plot(theta_values, tc_theta, "o-", ms=4.0, lw=1.4, label=label)
        axes[1].plot(hvals, tc_theta, "o-", ms=4.0, lw=1.4, label=label)
    axes[0].set_xlabel(r"$\theta_0$")
    axes[0].set_ylabel(r"$t_c(\theta_0)$")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel(r"$h(\theta_0)=\log\!\left(e/[1-(\theta/\pi)^2]\right)$")
    axes[1].set_ylabel(r"$t_c(\theta_0)$")
    axes[1].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(outdir / "tc_theta_scan.png", dpi=dpi)
    plt.close(fig)


def summarize_payload(payload: dict, theta_values: np.ndarray, ode: dict, bootstrap: dict):
    out = {
        k: v for k, v in payload.items() if k not in {"result_x", "y_fit", "frac_resid", "tc_theta"}
    }
    out["tc_theta"] = {f"{float(theta):.10f}": float(val) for theta, val in zip(theta_values, payload["tc_theta"])}
    out["bootstrap"] = bootstrap
    out["F0_ode"] = {f"{float(theta):.10f}": float(val) for theta, val in zip(theta_values, ode["F0"])}
    out["F_inf_ode"] = {f"{float(theta):.10f}": float(val) for theta, val in zip(theta_values, ode["F_inf"])}
    return out


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        df = shared.load_lattice_dataset(args.fixed_vw, args.h_values)
        theta_values = np.sort(df["theta"].unique())
        ode = shared.load_ode_amplitudes(Path(args.ode_summary).resolve(), theta_values)
        meta = shared.make_meta(df, theta_values, ode["F0"], ode["F_inf"], args.t_osc)
        reference_path = resolve_reference_summary(args.reference_summary)
        ref_summary = json.loads(reference_path.read_text()) if reference_path is not None else None

        if ref_summary is not None:
            init_r = float(ref_summary["free_tc"].get("r", 2.0))
            init_tc_scalar = float(ref_summary["free_tc"].get("t_c", 2.0))
        else:
            init_r = 2.0
            init_tc_scalar = 2.0
        init_tc = np.full(len(theta_values), init_tc_scalar, dtype=np.float64)

        fit_plain = fit_case(meta, theta_values, args.fixed_beta, init_r, init_tc, with_calib=False, tc_max=args.tc_max)
        fit_calib = fit_case(
            meta,
            theta_values,
            args.fixed_beta,
            float(fit_plain["r"]),
            np.asarray(fit_plain["tc_theta"], dtype=np.float64),
            with_calib=True,
            tc_max=args.tc_max,
        )
        boot_plain = bootstrap_case(meta, theta_values, args.fixed_beta, fit_plain, args.bootstrap, args.n_jobs, with_calib=False, tc_max=args.tc_max)
        boot_calib = bootstrap_case(meta, theta_values, args.fixed_beta, fit_calib, args.bootstrap, args.n_jobs, with_calib=True, tc_max=args.tc_max)

        plain_payload = summarize_payload(fit_plain, theta_values, ode, boot_plain)
        calib_payload = summarize_payload(fit_calib, theta_values, ode, boot_calib)

        plot_overlay(
            df,
            meta,
            theta_values,
            fit_plain,
            outdir / "collapse_overlay_theta_tc.png",
            rf"Lattice v_w={args.fixed_vw:.1f} with ODE-fixed $F_0,F_\infty$, $\beta={args.fixed_beta:.1f}$, and $t_c(\theta_0)$",
            args.dpi,
        )
        plot_overlay(
            df,
            meta,
            theta_values,
            fit_calib,
            outdir / "collapse_overlay_theta_tc_calib.png",
            rf"Lattice v_w={args.fixed_vw:.1f} with ODE-fixed $F_0,F_\infty$, $\beta={args.fixed_beta:.1f}$, $t_c(\theta_0)$, and $c_{{calib}}$",
            args.dpi,
        )
        plot_raw_betaH(df, meta, theta_values, fit_plain, outdir, "xi_vs_betaH_odefixed_theta_tc", args.dpi)
        plot_raw_betaH(df, meta, theta_values, fit_calib, outdir, "xi_vs_betaH_odefixed_theta_tc_calib", args.dpi)
        plot_tc_theta(theta_values, [("no calib", fit_plain), ("with calib", fit_calib)], outdir, args.dpi)

        comparison = {
            "theta_tc": {
                "beta": float(args.fixed_beta),
                "r": float(fit_plain["r"]),
                "rel_rmse": float(fit_plain["rel_rmse"]),
                "AIC": float(fit_plain["AIC"]),
                "BIC": float(fit_plain["BIC"]),
                "tc_theta_min": float(np.min(fit_plain["tc_theta"])),
                "tc_theta_max": float(np.max(fit_plain["tc_theta"])),
            },
            "theta_tc_calib": {
                "beta": float(args.fixed_beta),
                "r": float(fit_calib["r"]),
                "c_calib": float(fit_calib["c_calib"]),
                "rel_rmse": float(fit_calib["rel_rmse"]),
                "AIC": float(fit_calib["AIC"]),
                "BIC": float(fit_calib["BIC"]),
                "tc_theta_min": float(np.min(fit_calib["tc_theta"])),
                "tc_theta_max": float(np.max(fit_calib["tc_theta"])),
            },
        }
        if ref_summary is not None:
            comparison["reference_shared_tc"] = {
                "source": str(reference_path),
                "free_tc": ref_summary.get("free_tc"),
                "free_tc_calib": ref_summary.get("free_tc_calib"),
            }

        theta_table = []
        for i, theta in enumerate(theta_values):
            row = {
                "theta": float(theta),
                "h_alt": float(h_alt(np.array([theta]))[0]),
                "F0_ode": float(ode["F0"][i]),
                "F_inf_ode": float(ode["F_inf"][i]),
                "tc_theta": float(fit_plain["tc_theta"][i]),
                "tc_theta_calib": float(fit_calib["tc_theta"][i]),
            }
            if boot_plain.get("status") == "ok":
                row["tc_theta_p16"] = float(boot_plain["tc_theta"][f"{float(theta):.10f}"]["p16"])
                row["tc_theta_p84"] = float(boot_plain["tc_theta"][f"{float(theta):.10f}"]["p84"])
            if boot_calib.get("status") == "ok":
                row["tc_theta_calib_p16"] = float(boot_calib["tc_theta"][f"{float(theta):.10f}"]["p16"])
                row["tc_theta_calib_p84"] = float(boot_calib["tc_theta"][f"{float(theta):.10f}"]["p84"])
            theta_table.append(row)

        final_summary = {
            "status": "ok",
            "fixed_vw": float(args.fixed_vw),
            "fixed_beta": float(args.fixed_beta),
            "tc_max": float(args.tc_max),
            "H_values": [float(v) for v in sorted(df["H"].unique())],
            "theta_values": [float(v) for v in theta_values],
            "n_points": int(len(df)),
            "ode_amplitude_source": ode["source"],
            "ode_fit_summary": ode["ode_fit_summary"],
            "theta_tc": plain_payload,
            "theta_tc_calib": calib_payload,
            "comparison": comparison,
            "theta_table": theta_table,
        }

        shared.save_json(outdir / "fit_theta_tc.json", plain_payload)
        shared.save_json(outdir / "fit_theta_tc_calib.json", calib_payload)
        shared.save_json(outdir / "model_comparison.json", comparison)
        shared.save_json(outdir / "final_summary.json", final_summary)
        print(json.dumps(shared.to_native(final_summary), indent=2, sort_keys=True))
    except Exception as exc:
        error_exit(outdir, exc)
        raise


if __name__ == "__main__":
    main()
