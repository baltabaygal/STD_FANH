#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from scipy.optimize import least_squares

import fit_lattice_fixed_ode_amplitudes as shared
import fit_lattice_fixed_ode_amplitudes_theta_tc as theta_tc
import fit_lattice_fixed_ode_amplitudes_theta_tc_shared_r_by_vw as shared_rvw
import fit_lattice_quadwarp_universal as quad


ROOT = Path(__file__).resolve().parent
OUTDIR_DEFAULT = "results_lattice_theta_tc_sharedr_hstar_tests_beta0_tcmax300"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]
BASELINE_INIT_DEFAULT = ROOT / "results_lattice_theta_tc_sharedr_free_finf_by_vw_beta0_tcmax300" / "final_summary.json"
TC_INIT_DEFAULT = ROOT / "results_lattice_fixed_ode_amplitudes_theta_tc_by_vw_beta0_tcmax100" / "final_summary.json"


CASE_CONFIGS = [
    ("baseline", "Baseline", "baseline"),
    ("tc_hpow", r"$t_c \to t_c H_*^\alpha$", "tc_hpow"),
    ("gamma_amp", r"$\tilde f \to H_*^\gamma \tilde f$", "gamma_amp"),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare high-H* remedies for the canonical shared-r lattice fit: baseline, tc(theta0,vw) H*-timing power, and overall H*^gamma amplitude."
    )
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--fixed-beta", type=float, default=0.0)
    p.add_argument("--t-osc", type=float, default=1.5)
    p.add_argument("--ode-summary", type=str, default=str(shared.ODE_SUMMARY_DEFAULT))
    p.add_argument("--baseline-init-summary", type=str, default=str(BASELINE_INIT_DEFAULT))
    p.add_argument("--tc-init-summary", type=str, default=str(TC_INIT_DEFAULT))
    p.add_argument("--tc-max", type=float, default=300.0)
    p.add_argument(
        "--transient-kernel",
        type=str,
        choices=["power_inside", "shifted"],
        default="power_inside",
    )
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--outdir", type=str, default=OUTDIR_DEFAULT)
    return p.parse_args()


def error_exit(outdir: Path, exc: Exception):
    payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
    outdir.mkdir(parents=True, exist_ok=True)
    shared.save_json(outdir / "_error.json", payload)
    print(json.dumps(payload, sort_keys=True))


def load_json_if_exists(path: Path | None):
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def baseline_init_from_summary(summary: dict | None, theta_values: np.ndarray, vw_values: np.ndarray, ode: dict):
    init_r = 2.0
    init_tc = np.full((len(vw_values), len(theta_values)), 2.0, dtype=np.float64)
    init_f_infty = np.asarray(ode["F_inf"], dtype=np.float64) / np.maximum(np.asarray(ode["F0"], dtype=np.float64), 1.0e-18)
    if summary is None:
        return init_r, init_tc, init_f_infty
    payload = summary.get("shared_r_free_finf", {})
    if "r" in payload:
        init_r = float(payload["r"])
    table = payload.get("tc_by_vw_theta", {})
    for i, vw in enumerate(vw_values):
        block = table.get(f"{float(vw):.1f}", {})
        for j, theta in enumerate(theta_values):
            key = f"{float(theta):.10f}"
            if key in block:
                init_tc[i, j] = float(block[key])
    f_inf_table = payload.get("f_infty_lattice", {})
    for j, theta in enumerate(theta_values):
        key = f"{float(theta):.10f}"
        if key in f_inf_table:
            init_f_infty[j] = float(f_inf_table[key])
    return init_r, init_tc, init_f_infty


def tc_grid_fallback_from_summary(summary: dict | None, theta_values: np.ndarray, vw_values: np.ndarray):
    if summary is None:
        return np.full((len(vw_values), len(theta_values)), 2.0, dtype=np.float64)
    return shared_rvw.tc_grid_from_reference(summary, vw_values, theta_values, "plain", default_tc=2.0)


def unpack_params(params: np.ndarray, n_vw: int, n_theta: int, mode: str):
    idx = 0
    r = float(params[idx])
    idx += 1
    tc_by_vw_theta = np.asarray(params[idx : idx + n_vw * n_theta], dtype=np.float64).reshape(n_vw, n_theta)
    idx += n_vw * n_theta
    f_infty_theta = np.asarray(params[idx : idx + n_theta], dtype=np.float64)
    idx += n_theta
    extra = {}
    if mode == "tc_hpow":
        extra["alpha_tc"] = float(params[idx])
        idx += 1
    elif mode == "gamma_amp":
        extra["gamma"] = float(params[idx])
        idx += 1
    return r, tc_by_vw_theta, f_infty_theta, extra


def model_details(meta, fixed_beta: float, params: np.ndarray, kernel: str, mode: str):
    n_vw = len(meta["vw_values"])
    n_theta = len(meta["theta_values"])
    r, tc_by_vw_theta, f_infty_theta, extra = unpack_params(params, n_vw, n_theta, mode)
    x, xi_scale = shared_rvw.x_and_xi_scale(meta, fixed_beta)
    tc = tc_by_vw_theta[meta["vw_idx"], meta["theta_idx"]]
    if mode == "tc_hpow":
        tc = tc * np.power(meta["H"], float(extra["alpha_tc"]))
    denom = shared_rvw.transition_denom(x, tc, r, kernel)
    f_tilde = f_infty_theta[meta["theta_idx"]] + meta["F0"] / np.maximum(xi_scale * denom, 1.0e-18)
    if mode == "gamma_amp":
        f_tilde = np.power(meta["H"], float(extra["gamma"])) * f_tilde
    xi_fit = shared_rvw.xi_from_f_tilde(xi_scale, f_tilde, meta["F0"])
    return x, xi_scale, tc, f_tilde, xi_fit, r, tc_by_vw_theta, f_infty_theta, extra


def build_param_vector(init_r: float, init_tc: np.ndarray, init_f_infty: np.ndarray, mode: str):
    parts = [
        np.array([float(init_r)], dtype=np.float64),
        np.asarray(init_tc, dtype=np.float64).ravel(),
        np.asarray(init_f_infty, dtype=np.float64),
    ]
    lower = [
        np.array([0.1], dtype=np.float64),
        np.full(init_tc.size, 0.1, dtype=np.float64),
        np.full(init_f_infty.size, 1.0e-8, dtype=np.float64),
    ]
    upper = [
        np.array([20.0], dtype=np.float64),
        None,
        np.full(init_f_infty.size, 1.0e4, dtype=np.float64),
    ]
    if mode == "tc_hpow":
        parts.append(np.array([0.0], dtype=np.float64))
        lower.append(np.array([-4.0], dtype=np.float64))
        upper.append(np.array([4.0], dtype=np.float64))
    elif mode == "gamma_amp":
        parts.append(np.array([0.0], dtype=np.float64))
        lower.append(np.array([-2.0], dtype=np.float64))
        upper.append(np.array([2.0], dtype=np.float64))
    return parts, lower, upper


def fit_case(meta, fixed_beta: float, init_r: float, init_tc: np.ndarray, init_f_infty: np.ndarray, tc_max: float, kernel: str, mode: str):
    parts, lower_parts, upper_parts = build_param_vector(init_r, init_tc, init_f_infty, mode)
    x0 = np.concatenate(parts)
    lower_parts[1] = np.full(init_tc.size, 0.1, dtype=np.float64)
    upper_parts[1] = np.full(init_tc.size, float(tc_max), dtype=np.float64)
    lower = np.concatenate(lower_parts)
    upper = np.concatenate(upper_parts)

    def resid(par: np.ndarray) -> np.ndarray:
        _, _, _, _, xi_fit, *_ = model_details(meta, fixed_beta, par, kernel, mode)
        return (xi_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)

    huber = least_squares(resid, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=60000)
    final = least_squares(resid, huber.x, bounds=(lower, upper), loss="linear", max_nfev=60000)
    x, xi_scale, tc_eff, f_tilde, xi_fit, r, tc_by_vw_theta, f_infty_theta, extra = model_details(meta, fixed_beta, final.x, kernel, mode)
    frac_resid = (xi_fit - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    aic, bic = shared.aic_bic(frac_resid, len(final.x))
    payload = {
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "mode": str(mode),
        "beta": float(fixed_beta),
        "transient_kernel": str(kernel),
        "canonical_xi_form": "xi = (2 x / (3 t_osc))^(3/2) * f_tilde / F0",
        "transition_denominator": shared_rvw.kernel_denom_text(kernel),
        "r": float(r),
        "tc_by_vw_theta": tc_by_vw_theta,
        "f_infty": f_infty_theta,
        "rel_rmse": shared_rvw.rel_rmse(meta["xi"], xi_fit),
        "AIC": float(aic),
        "BIC": float(bic),
        "n_points": int(meta["xi"].size),
        "n_params": int(len(final.x)),
        "result_x": np.asarray(final.x, dtype=np.float64),
        "y_fit": xi_fit,
        "f_tilde_fit": f_tilde,
        "x_fit": x,
        "tc_eff_fit": tc_eff,
        "frac_resid": frac_resid,
    }
    if mode == "baseline":
        payload["f_tilde_form"] = "f_tilde = f_infty(theta0) + F0(theta0) / ((2 x / (3 t_osc))^(3/2) * D)"
    elif mode == "tc_hpow":
        payload["alpha_tc"] = float(extra["alpha_tc"])
        payload["f_tilde_form"] = "f_tilde = f_infty(theta0) + F0(theta0) / ((2 x / (3 t_osc))^(3/2) * D), with tc_eff = tc(theta0,vw) * H^alpha_tc"
    elif mode == "gamma_amp":
        payload["gamma"] = float(extra["gamma"])
        payload["f_tilde_form"] = "f_tilde_eff = H^gamma * [f_infty(theta0) + F0(theta0) / ((2 x / (3 t_osc))^(3/2) * D)]"
    payload["per_vw_rel_rmse"] = {}
    for vw in meta["vw_values"]:
        mask = np.isclose(meta["v_w"], float(vw), atol=1.0e-12)
        payload["per_vw_rel_rmse"][f"{float(vw):.1f}"] = shared_rvw.rel_rmse(meta["xi"][mask], xi_fit[mask])
    return payload


def build_prediction_frame(df, meta, fit_payload: dict, fixed_beta: float, kernel: str, mode: str):
    out = df.copy()
    out["x"] = fit_payload["x_fit"]
    out["xi_fit"] = fit_payload["y_fit"]
    out["f_tilde_fit"] = fit_payload["f_tilde_fit"]
    out["tc_eff_fit"] = fit_payload["tc_eff_fit"]
    n_vw = len(meta["vw_values"])
    n_theta = len(meta["theta_values"])
    _, _, f_infty_theta, _ = unpack_params(fit_payload["result_x"], n_vw, n_theta, mode)
    out["f_infty_fit"] = f_infty_theta[np.asarray(meta["theta_idx"], dtype=np.int64)]
    return out


def summarize_payload(payload: dict, theta_values: np.ndarray, vw_values: np.ndarray, ode: dict):
    out = {
        k: v
        for k, v in payload.items()
        if k not in {"result_x", "y_fit", "f_tilde_fit", "x_fit", "tc_eff_fit", "frac_resid", "tc_by_vw_theta", "f_infty"}
    }
    out["tc_by_vw_theta"] = {}
    for i, vw in enumerate(vw_values):
        out["tc_by_vw_theta"][f"{float(vw):.1f}"] = {
            f"{float(theta):.10f}": float(payload["tc_by_vw_theta"][i, j])
            for j, theta in enumerate(theta_values)
        }
    out["f_infty_lattice"] = {
        f"{float(theta):.10f}": float(payload["f_infty"][i]) for i, theta in enumerate(theta_values)
    }
    out["f_infty_ode"] = {
        f"{float(theta):.10f}": float(ode["F_inf"][i] / max(float(ode["F0"][i]), 1.0e-18))
        for i, theta in enumerate(theta_values)
    }
    out["F_infty_raw_lattice"] = {
        f"{float(theta):.10f}": float(payload["f_infty"][i] * float(ode["F0"][i]))
        for i, theta in enumerate(theta_values)
    }
    out["F_infty_raw_ode"] = {f"{float(theta):.10f}": float(val) for theta, val in zip(theta_values, ode["F_inf"])}
    out["f_infty_ratio_lattice_over_ode"] = {
        f"{float(theta):.10f}": float(payload["f_infty"][i] / max(float(ode["F_inf"][i] / max(float(ode["F0"][i]), 1.0e-18)), 1.0e-18))
        for i, theta in enumerate(theta_values)
    }
    out["F0_ode"] = {f"{float(theta):.10f}": float(val) for theta, val in zip(theta_values, ode["F0"])}
    return out


def rmse_tables(df_pred):
    rows = []
    by_h = defaultdict(list)
    by_h_vw = defaultdict(list)
    for h_value in sorted(df_pred["H"].unique()):
        sub_h = df_pred[np.isclose(df_pred["H"], float(h_value), atol=1.0e-8)].copy()
        for theta in sorted(sub_h["theta"].unique()):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=1.0e-8)].copy()
            for vw in sorted(sub["v_w"].unique()):
                cur = sub[np.isclose(sub["v_w"], float(vw), atol=1.0e-8)]
                rmse = shared.rel_rmse(cur["xi"], cur["xi_fit"])
                rows.append({"H": float(h_value), "theta": float(theta), "v_w": float(vw), "rel_rmse": rmse})
                by_h[float(h_value)].append(rmse)
                by_h_vw[(float(h_value), float(vw))].append(rmse)
    mean_by_h = {f"{h:.1f}": float(np.mean(vals)) for h, vals in sorted(by_h.items())}
    mean_by_h_vw = {f"H{h:.1f}_vw{vw:.1f}": float(np.mean(vals)) for (h, vw), vals in sorted(by_h_vw.items())}
    return rows, mean_by_h, mean_by_h_vw


def plot_rmse_by_h(case_summaries: dict, outpath: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    h_values = [1.0, 1.5, 2.0]
    for mode, label, _ in CASE_CONFIGS:
        vals = [case_summaries[mode]["mean_raw_rmse_by_h"][f"{h:.1f}"] for h in h_values]
        ax.plot(h_values, vals, "o-", lw=1.8, label=label)
    ax.set_xlabel(r"$H_*$")
    ax.set_ylabel("mean raw rel_rmse")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_rmse_by_h_vw(case_summaries: dict, outpath: Path, dpi: int):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), constrained_layout=True)
    h_values = [1.0, 1.5, 2.0]
    for ax, vw in zip(axes, [0.3, 0.9]):
        for mode, label, _ in CASE_CONFIGS:
            vals = [case_summaries[mode]["mean_raw_rmse_by_h_vw"][f"H{h:.1f}_vw{vw:.1f}"] for h in h_values]
            ax.plot(h_values, vals, "o-", lw=1.8, label=label)
        ax.set_title(rf"$v_w={vw:.1f}$")
        ax.set_xlabel(r"$H_*$")
        ax.set_ylabel("mean raw rel_rmse")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def make_comparison_sheet(case_dirs: dict[str, Path], outpath: Path, dpi: int):
    items = []
    for mode, label, _ in CASE_CONFIGS:
        subdir = case_dirs[mode]
        items.append((f"{label}: collapse", subdir / "collapse_overlay.png"))
        items.append((f"{label}: raw H*=2.0", subdir / "xi_vs_betaH_H2p0.png"))
    fig, axes = plt.subplots(len(CASE_CONFIGS), 2, figsize=(16, 6 * len(CASE_CONFIGS)))
    for ax, (title, path) in zip(axes.ravel(), items):
        img = imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.suptitle("High-H* comparison: baseline vs timing-H vs amplitude-H", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        df = quad.load_lattice_dataframe(outdir, args.vw_folders, args.h_values)
        theta_values = np.sort(df["theta"].unique())
        vw_values = np.sort(df["v_w"].unique())
        ode = shared.load_ode_amplitudes(Path(args.ode_summary).resolve(), theta_values)
        meta = shared_rvw.build_meta(df, theta_values, vw_values, ode, args.t_osc)

        baseline_ref = load_json_if_exists(Path(args.baseline_init_summary).resolve() if args.baseline_init_summary else None)
        tc_ref = load_json_if_exists(Path(args.tc_init_summary).resolve() if args.tc_init_summary else None)
        init_r, init_tc, init_f_infty = baseline_init_from_summary(baseline_ref, theta_values, vw_values, ode)
        if baseline_ref is None:
            init_tc = tc_grid_fallback_from_summary(tc_ref, theta_values, vw_values)

        case_payloads = {}
        case_summaries = {}
        case_dirs = {}

        baseline = fit_case(meta, args.fixed_beta, init_r, init_tc, init_f_infty, args.tc_max, args.transient_kernel, "baseline")
        case_payloads["baseline"] = baseline

        tc_hpow = fit_case(
            meta,
            args.fixed_beta,
            float(baseline["r"]),
            np.asarray(baseline["tc_by_vw_theta"], dtype=np.float64),
            np.asarray(baseline["f_infty"], dtype=np.float64),
            args.tc_max,
            args.transient_kernel,
            "tc_hpow",
        )
        case_payloads["tc_hpow"] = tc_hpow

        gamma_amp = fit_case(
            meta,
            args.fixed_beta,
            float(baseline["r"]),
            np.asarray(baseline["tc_by_vw_theta"], dtype=np.float64),
            np.asarray(baseline["f_infty"], dtype=np.float64),
            args.tc_max,
            args.transient_kernel,
            "gamma_amp",
        )
        case_payloads["gamma_amp"] = gamma_amp

        for mode, label, slug in CASE_CONFIGS:
            subdir = outdir / slug
            subdir.mkdir(parents=True, exist_ok=True)
            case_dirs[mode] = subdir
            payload = case_payloads[mode]
            pred = build_prediction_frame(df, meta, payload, args.fixed_beta, args.transient_kernel, mode)
            pred.to_csv(subdir / "predictions.csv", index=False)
            shared_rvw.plot_collapse_overlay(
                pred,
                theta_values,
                vw_values,
                payload,
                subdir / "collapse_overlay.png",
                args.dpi,
                args.fixed_beta,
                rf"{label}, kernel {shared_rvw.kernel_label(args.transient_kernel)}, $r={payload['r']:.3f}$",
            )
            shared_rvw.plot_raw(pred, theta_values, vw_values, subdir, "xi_vs_betaH", args.dpi)
            free_finf_plot_path = subdir / "f_infty_theta.png"
            theta_tc.h_alt(theta_values)  # keep import usage explicit
            fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), constrained_layout=True)
            f_infty = np.asarray(payload["f_infty"], dtype=np.float64)
            f_infty_ode = np.asarray(ode["F_inf"], dtype=np.float64) / np.maximum(np.asarray(ode["F0"], dtype=np.float64), 1.0e-18)
            hvals = theta_tc.h_alt(theta_values)
            axes[0].plot(theta_values, f_infty_ode, "k--", lw=1.5, label="ODE")
            axes[0].plot(theta_values, f_infty, "o-", ms=4.2, lw=1.5, color="tab:blue", label="fit")
            axes[0].set_xlabel(r"$\theta_0$")
            axes[0].set_ylabel(r"$f_\infty(\theta_0)$")
            axes[0].set_yscale("log")
            axes[0].grid(alpha=0.25)
            axes[0].legend(frameon=False, fontsize=8)
            ratio = f_infty / np.maximum(f_infty_ode, 1.0e-18)
            axes[1].plot(hvals, ratio, "o-", ms=4.2, lw=1.5, color="tab:red")
            axes[1].axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.7)
            axes[1].set_xlabel(r"$h(\theta_0)=\log\!\left(e/[1-(\theta/\pi)^2]\right)$")
            axes[1].set_ylabel(r"$f_\infty^{fit}/f_\infty^{ODE}$")
            axes[1].grid(alpha=0.25)
            fig.savefig(free_finf_plot_path, dpi=args.dpi)
            plt.close(fig)

            raw_rows, mean_by_h, mean_by_h_vw = rmse_tables(pred)
            summary = summarize_payload(payload, theta_values, vw_values, ode)
            summary["label"] = label
            summary["mean_raw_rmse_by_h"] = mean_by_h
            summary["mean_raw_rmse_by_h_vw"] = mean_by_h_vw
            summary["raw_plot_rmse"] = raw_rows
            case_summaries[mode] = summary
            shared.save_json(subdir / "final_summary.json", summary)

        plot_rmse_by_h(case_summaries, outdir / "rmse_by_h.png", args.dpi)
        plot_rmse_by_h_vw(case_summaries, outdir / "rmse_by_h_vw.png", args.dpi)
        make_comparison_sheet(case_dirs, outdir / "comparison_sheet.png", args.dpi)

        summary = {
            "status": "ok",
            "vw_values": [float(v) for v in vw_values],
            "theta_values": [float(v) for v in theta_values],
            "fixed_beta": float(args.fixed_beta),
            "tc_max": float(args.tc_max),
            "transient_kernel": str(args.transient_kernel),
            "n_points": int(len(df)),
            "ode_amplitude_source": ode["source"],
            "ode_fit_summary": ode["ode_fit_summary"],
            "cases": case_summaries,
            "outputs": {
                "comparison_sheet": str(outdir / "comparison_sheet.png"),
                "rmse_by_h": str(outdir / "rmse_by_h.png"),
                "rmse_by_h_vw": str(outdir / "rmse_by_h_vw.png"),
            },
        }
        shared.save_json(outdir / "final_summary.json", summary)
        print(json.dumps(shared.to_native(summary), indent=2, sort_keys=True))
    except Exception as exc:
        error_exit(outdir, exc)
        raise


if __name__ == "__main__":
    main()
