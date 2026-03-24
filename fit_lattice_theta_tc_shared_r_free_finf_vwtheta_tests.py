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
OUTDIR_DEFAULT = "results_lattice_theta_tc_sharedr_free_finf_vwtheta_tests_beta0_tcmax300"
VW_TAGS = ["v3", "v5", "v7", "v9"]
H_VALUES = [1.0, 1.5, 2.0]
REFERENCE_SUMMARY_DEFAULT = (
    ROOT / "results_lattice_theta_tc_sharedr_free_finf_by_vw_beta0_tcmax300" / "final_summary.json"
)
BENCHMARK_SUMMARY_DEFAULT = (
    ROOT / "results_lattice_theta_tc_tosc_and_s_tests_beta0_tcmax300" / "final_summary.json"
)

CASE_CONFIGS = [
    ("vwtheta_fixed_tosc", r"$f_\infty(\theta_0,v_w)$, fixed $t_{\rm osc}$", "vwtheta_fixed_tosc"),
    ("vwtheta_free_tosc", r"$f_\infty(\theta_0,v_w)$, free $t_{\rm osc}$", "vwtheta_free_tosc"),
]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Fit canonical xi=(2 x / (3 t_osc))^(3/2) * f_tilde / F0 with c=1, ODE-fixed F0(theta), "
            "shared global r, free tc(theta,v_w), and free f_infty(theta,v_w). "
            "Compare fixed vs free global t_osc against the calibrated ODE-fixed benchmark."
        )
    )
    p.add_argument("--vw-folders", nargs="*", default=VW_TAGS)
    p.add_argument("--h-values", type=float, nargs="+", default=H_VALUES)
    p.add_argument("--fixed-beta", type=float, default=0.0)
    p.add_argument("--t-osc", type=float, default=1.5)
    p.add_argument("--ode-summary", type=str, default=str(shared.ODE_SUMMARY_DEFAULT))
    p.add_argument("--reference-summary", type=str, default=str(REFERENCE_SUMMARY_DEFAULT))
    p.add_argument("--benchmark-summary", type=str, default=str(BENCHMARK_SUMMARY_DEFAULT))
    p.add_argument("--tc-max", type=float, default=300.0)
    p.add_argument("--tosc-min", type=float, default=0.2)
    p.add_argument("--tosc-max", type=float, default=10.0)
    p.add_argument(
        "--transient-kernel",
        type=str,
        choices=["power_inside", "shifted"],
        default="power_inside",
        help="Use 1 + (x/t_c)^r (`power_inside`) or (1 + x/t_c)^r (`shifted`) in the transient denominator.",
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


def init_from_reference(summary: dict | None, theta_values: np.ndarray, vw_values: np.ndarray, ode: dict):
    n_vw = len(vw_values)
    n_theta = len(theta_values)
    init_r = 2.0
    init_tc = np.full((n_vw, n_theta), 2.0, dtype=np.float64)
    init_f_infty = np.repeat(
        (
            np.asarray(ode["F_inf"], dtype=np.float64)
            / np.maximum(np.asarray(ode["F0"], dtype=np.float64), 1.0e-18)
        )[None, :],
        n_vw,
        axis=0,
    )
    if summary is None:
        return init_r, init_tc, init_f_infty
    payload = summary.get("shared_r_free_finf", {})
    init_r = float(payload.get("r", init_r))
    tc_map = payload.get("tc_by_vw_theta", {})
    for i, vw in enumerate(vw_values):
        row = tc_map.get(f"{float(vw):.1f}", {})
        for j, theta in enumerate(theta_values):
            key = f"{float(theta):.10f}"
            if key in row:
                init_tc[i, j] = float(row[key])
    f_map = payload.get("f_infty_lattice", {})
    f_theta = np.asarray(
        [float(f_map.get(f"{float(theta):.10f}", init_f_infty[0, j])) for j, theta in enumerate(theta_values)],
        dtype=np.float64,
    )
    init_f_infty = np.repeat(f_theta[None, :], n_vw, axis=0)
    return init_r, init_tc, init_f_infty


def benchmark_from_summary(summary: dict | None):
    if summary is None:
        return None
    payload = summary.get("cases", {}).get("calib_free_tosc")
    if payload is None:
        return None
    return payload


def unpack_params(params: np.ndarray, mode: str, n_vw: int, n_theta: int):
    idx = 0
    r = float(params[idx])
    idx += 1
    tc_by_vw_theta = np.asarray(params[idx : idx + n_vw * n_theta], dtype=np.float64).reshape(n_vw, n_theta)
    idx += n_vw * n_theta
    f_infty_by_vw_theta = np.asarray(params[idx : idx + n_vw * n_theta], dtype=np.float64).reshape(n_vw, n_theta)
    idx += n_vw * n_theta
    extra = {}
    if mode == "vwtheta_free_tosc":
        extra["t_osc"] = float(params[idx])
    return r, tc_by_vw_theta, f_infty_by_vw_theta, extra


def model_details(meta, fixed_beta: float, params: np.ndarray, kernel: str, mode: str):
    n_vw = len(meta["vw_values"])
    n_theta = len(meta["theta_values"])
    r, tc_by_vw_theta, f_infty_by_vw_theta, extra = unpack_params(params, mode, n_vw, n_theta)
    x = meta["tp"] * np.power(meta["H"], float(fixed_beta))
    t_osc = float(meta["t_osc"] if mode == "vwtheta_fixed_tosc" else extra["t_osc"])
    xi_scale = shared_rvw.xi_scale_from_x(x, t_osc)
    tc = tc_by_vw_theta[meta["vw_idx"], meta["theta_idx"]]
    f_infty = f_infty_by_vw_theta[meta["vw_idx"], meta["theta_idx"]]
    denom = shared_rvw.transition_denom(x, tc, r, kernel)
    f_tilde = f_infty + meta["F0"] / np.maximum(xi_scale * denom, 1.0e-18)
    xi_fit = shared_rvw.xi_from_f_tilde(xi_scale, f_tilde, meta["F0"])
    return {
        "r": r,
        "tc_by_vw_theta": tc_by_vw_theta,
        "f_infty_by_vw_theta": f_infty_by_vw_theta,
        "t_osc": t_osc,
        "x": x,
        "xi_scale": xi_scale,
        "f_tilde": f_tilde,
        "xi_fit": xi_fit,
    }


def fit_case(meta, fixed_beta: float, init_r: float, init_tc: np.ndarray, init_f_infty: np.ndarray, args, mode: str):
    n_vw = len(meta["vw_values"])
    n_theta = len(meta["theta_values"])
    if np.asarray(init_tc).shape != (n_vw, n_theta):
        raise ValueError("Initial tc grid must match (n_vw, n_theta).")
    if np.asarray(init_f_infty).shape != (n_vw, n_theta):
        raise ValueError("Initial f_infty grid must match (n_vw, n_theta).")

    x0 = np.concatenate(
        [
            np.array([float(init_r)], dtype=np.float64),
            np.asarray(init_tc, dtype=np.float64).ravel(),
            np.asarray(init_f_infty, dtype=np.float64).ravel(),
        ]
    )
    lower = np.concatenate(
        [
            np.array([0.1], dtype=np.float64),
            np.full(n_vw * n_theta, 0.1, dtype=np.float64),
            np.full(n_vw * n_theta, 1.0e-8, dtype=np.float64),
        ]
    )
    upper = np.concatenate(
        [
            np.array([20.0], dtype=np.float64),
            np.full(n_vw * n_theta, float(args.tc_max), dtype=np.float64),
            np.full(n_vw * n_theta, 1.0e4, dtype=np.float64),
        ]
    )
    if mode == "vwtheta_free_tosc":
        x0 = np.concatenate([x0, np.array([float(args.t_osc)], dtype=np.float64)])
        lower = np.concatenate([lower, np.array([float(args.tosc_min)], dtype=np.float64)])
        upper = np.concatenate([upper, np.array([float(args.tosc_max)], dtype=np.float64)])

    def resid(par: np.ndarray) -> np.ndarray:
        details = model_details(meta, fixed_beta, par, args.transient_kernel, mode)
        return (details["xi_fit"] - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)

    huber = least_squares(resid, x0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=50000)
    final = least_squares(resid, huber.x, bounds=(lower, upper), loss="linear", max_nfev=50000)
    details = model_details(meta, fixed_beta, final.x, args.transient_kernel, mode)
    frac_resid = (details["xi_fit"] - meta["xi"]) / np.maximum(meta["xi"], 1.0e-12)
    aic, bic = shared.aic_bic(frac_resid, len(final.x))
    payload = {
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "mode": mode,
        "beta": float(fixed_beta),
        "transient_kernel": str(args.transient_kernel),
        "canonical_xi_form": "xi = (2 x / (3 t_osc))^(3/2) * f_tilde / F0",
        "f_tilde_form": "f_tilde = f_infty(theta0,v_w) + F0(theta0) / ((2 x / (3 t_osc))^(3/2) * D)",
        "transition_denominator": shared_rvw.kernel_denom_text(args.transient_kernel),
        "r": float(details["r"]),
        "t_osc": float(details["t_osc"]),
        "tc_by_vw_theta": np.asarray(details["tc_by_vw_theta"], dtype=np.float64),
        "f_infty_by_vw_theta": np.asarray(details["f_infty_by_vw_theta"], dtype=np.float64),
        "rel_rmse": shared_rvw.rel_rmse(meta["xi"], details["xi_fit"]),
        "AIC": float(aic),
        "BIC": float(bic),
        "n_points": int(meta["xi"].size),
        "n_params": int(len(final.x)),
        "result_x": np.asarray(final.x, dtype=np.float64),
        "y_fit": details["xi_fit"],
        "f_tilde": details["f_tilde"],
        "x": details["x"],
        "frac_resid": frac_resid,
        "per_vw_rel_rmse": {},
    }
    for vw in meta["vw_values"]:
        mask = np.isclose(meta["v_w"], float(vw), atol=1.0e-12)
        payload["per_vw_rel_rmse"][f"{float(vw):.1f}"] = shared_rvw.rel_rmse(meta["xi"][mask], details["xi_fit"][mask])
    return payload


def summarize_payload(payload: dict, theta_values: np.ndarray, vw_values: np.ndarray, ode: dict):
    out = {k: v for k, v in payload.items() if k not in {"result_x", "y_fit", "f_tilde", "x", "frac_resid", "tc_by_vw_theta", "f_infty_by_vw_theta"}}
    out["tc_by_vw_theta"] = {}
    out["f_infty_by_vw_theta"] = {}
    out["f_infty_ratio_to_ode_by_vw_theta"] = {}
    for i, vw in enumerate(vw_values):
        vw_key = f"{float(vw):.1f}"
        out["tc_by_vw_theta"][vw_key] = {}
        out["f_infty_by_vw_theta"][vw_key] = {}
        out["f_infty_ratio_to_ode_by_vw_theta"][vw_key] = {}
        for j, theta in enumerate(theta_values):
            theta_key = f"{float(theta):.10f}"
            out["tc_by_vw_theta"][vw_key][theta_key] = float(payload["tc_by_vw_theta"][i, j])
            out["f_infty_by_vw_theta"][vw_key][theta_key] = float(payload["f_infty_by_vw_theta"][i, j])
            ode_f = float(ode["F_inf"][j] / max(float(ode["F0"][j]), 1.0e-18))
            out["f_infty_ratio_to_ode_by_vw_theta"][vw_key][theta_key] = float(payload["f_infty_by_vw_theta"][i, j] / max(ode_f, 1.0e-18))
    out["f_infty_ode"] = {
        f"{float(theta):.10f}": float(ode["F_inf"][i] / max(float(ode["F0"][i]), 1.0e-18))
        for i, theta in enumerate(theta_values)
    }
    out["F0_ode"] = {f"{float(theta):.10f}": float(val) for theta, val in zip(theta_values, ode["F0"])}
    out["F_infty_raw_ode"] = {f"{float(theta):.10f}": float(val) for theta, val in zip(theta_values, ode["F_inf"])}
    return out


def build_prediction_frame(df, payload: dict, meta):
    out = df.copy()
    out["x"] = np.asarray(payload["x"], dtype=np.float64)
    out["xi_fit"] = np.asarray(payload["y_fit"], dtype=np.float64)
    out["f_tilde_fit"] = np.asarray(payload["f_tilde"], dtype=np.float64)
    f_grid = np.asarray(payload["f_infty_by_vw_theta"], dtype=np.float64)
    out["f_infty_fit"] = f_grid[meta["vw_idx"], meta["theta_idx"]]
    return out


def plot_f_infty_by_vw(summary: dict, theta_values: np.ndarray, vw_values: np.ndarray, ode: dict, outpath: Path, dpi: int):
    cmap = plt.get_cmap("viridis")
    colors = {float(vw): cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    hvals = theta_tc.h_alt(theta_values)
    ode_vals = np.asarray([ode["F_inf"][i] / max(float(ode["F0"][i]), 1.0e-18) for i in range(len(theta_values))], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    axes[0].plot(theta_values, ode_vals, "k--", lw=1.5, label="ODE")
    for vw in vw_values:
        vals = np.asarray([summary["f_infty_by_vw_theta"][f"{float(vw):.1f}"][f"{float(theta):.10f}"] for theta in theta_values], dtype=np.float64)
        axes[0].plot(theta_values, vals, "o-", ms=4.0, lw=1.4, color=colors[float(vw)], label=rf"$v_w={float(vw):.1f}$")
        axes[1].plot(hvals, vals / np.maximum(ode_vals, 1.0e-18), "o-", ms=4.0, lw=1.4, color=colors[float(vw)], label=rf"$v_w={float(vw):.1f}$")
    axes[0].set_xlabel(r"$\theta_0$")
    axes[0].set_ylabel(r"$f_\infty(\theta_0,v_w)$")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[1].axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.7)
    axes[1].set_xlabel(r"$h(\theta_0)=\log\!\left(e/[1-(\theta/\pi)^2]\right)$")
    axes[1].set_ylabel(r"$f_\infty^{fit}/f_\infty^{ODE}$")
    axes[1].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


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
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    h_values = [1.0, 1.5, 2.0]
    for key, label in case_summaries.items():
        vals = [label["mean_raw_rmse_by_h"][f"{h:.1f}"] for h in h_values]
        ax.plot(h_values, vals, "o-", lw=1.8, label=label["label"])
    ax.set_xlabel(r"$H_*$")
    ax.set_ylabel("mean raw rel_rmse")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def make_comparison_sheet(items: list[tuple[str, Path]], outpath: Path, dpi: int):
    fig, axes = plt.subplots(len(items), 2, figsize=(16, 6 * len(items)))
    for row, (label, base_dir) in enumerate(items):
        for col, (title, name) in enumerate([("collapse", "collapse_overlay.png"), ("raw H*=2.0", "xi_vs_betaH_H2p0.png")]):
            ax = axes[row, col]
            ax.imshow(imread(base_dir / name))
            ax.set_title(f"{label}: {title}", fontsize=12)
            ax.axis("off")
    fig.suptitle("Benchmark vs boundary-consistent free-$f_\\infty(\\theta_0,v_w)$ fits", fontsize=16)
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

        reference_summary = load_json_if_exists(Path(args.reference_summary).resolve() if args.reference_summary else None)
        init_r, init_tc, init_f_infty = init_from_reference(reference_summary, theta_values, vw_values, ode)
        benchmark_summary = benchmark_from_summary(load_json_if_exists(Path(args.benchmark_summary).resolve() if args.benchmark_summary else None))

        case_payloads = {}
        case_summaries = {}
        case_dirs = {}

        for mode, label, slug in CASE_CONFIGS:
            if mode == "vwtheta_fixed_tosc":
                payload = fit_case(meta, args.fixed_beta, init_r, init_tc, init_f_infty, args, mode)
            elif mode == "vwtheta_free_tosc":
                seed = case_payloads["vwtheta_fixed_tosc"]
                payload = fit_case(
                    meta,
                    args.fixed_beta,
                    float(seed["r"]),
                    np.asarray(seed["tc_by_vw_theta"], dtype=np.float64),
                    np.asarray(seed["f_infty_by_vw_theta"], dtype=np.float64),
                    args,
                    mode,
                )
            else:
                raise ValueError(f"Unknown mode {mode}")
            case_payloads[mode] = payload

            subdir = outdir / slug
            subdir.mkdir(parents=True, exist_ok=True)
            case_dirs[mode] = subdir
            pred = build_prediction_frame(df, payload, meta)
            pred.to_csv(subdir / "predictions.csv", index=False)
            shared_rvw.plot_collapse_overlay(
                pred,
                theta_values,
                vw_values,
                payload,
                subdir / "collapse_overlay.png",
                args.dpi,
                args.fixed_beta,
                rf"{label}, kernel {shared_rvw.kernel_label(args.transient_kernel)}, $r={payload['r']:.3f}$, $t_{{osc}}={payload['t_osc']:.3f}$",
            )
            raw_rows = shared_rvw.plot_raw(pred, theta_values, vw_values, subdir, "xi_vs_betaH", args.dpi)
            shared_rvw.plot_tc_by_vw(payload, theta_values, vw_values, subdir / "tc_by_vw.png", args.dpi)
            summary = summarize_payload(payload, theta_values, vw_values, ode)
            summary["label"] = label
            summary["raw_plot_rmse"] = raw_rows
            _, mean_by_h, mean_by_h_vw = rmse_tables(pred)
            summary["mean_raw_rmse_by_h"] = mean_by_h
            summary["mean_raw_rmse_by_h_vw"] = mean_by_h_vw
            case_summaries[mode] = summary
            shared.save_json(subdir / "final_summary.json", summary)
            plot_f_infty_by_vw(summary, theta_values, vw_values, ode, subdir / "f_infty_by_vw.png", args.dpi)

        summary_map = {}
        if benchmark_summary is not None:
            summary_map["benchmark_calib_free_tosc"] = {
                "label": "benchmark: ODE-fixed $f_\\infty$, $c(v_w)$, free $t_{\\rm osc}$",
                "rel_rmse": float(benchmark_summary["rel_rmse"]),
                "AIC": float(benchmark_summary["AIC"]),
                "BIC": float(benchmark_summary["BIC"]),
                "r": float(benchmark_summary["r"]),
                "t_osc": float(benchmark_summary["t_osc"]),
                "c_calib_by_vw": benchmark_summary.get("c_calib_by_vw"),
                "mean_raw_rmse_by_h": benchmark_summary.get("mean_raw_rmse_by_h"),
                "mean_raw_rmse_by_h_vw": benchmark_summary.get("mean_raw_rmse_by_h_vw"),
                "per_vw_rel_rmse": benchmark_summary.get("per_vw_rel_rmse"),
            }
        summary_map.update(case_summaries)

        plot_rmse_by_h(summary_map, outdir / "rmse_by_h.png", args.dpi)
        compare_items = []
        benchmark_dir = ROOT / "results_lattice_theta_tc_tosc_and_s_tests_beta0_tcmax300" / "calib_free_tosc"
        if benchmark_dir.exists():
            compare_items.append(("benchmark calib free $t_{osc}$", benchmark_dir))
        compare_items.extend(
            [
                (case_summaries[mode]["label"], case_dirs[mode])
                for mode, _, _ in CASE_CONFIGS
            ]
        )
        make_comparison_sheet(compare_items, outdir / "comparison_sheet.png", args.dpi)

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
            "benchmark": summary_map.get("benchmark_calib_free_tosc"),
            "cases": case_summaries,
            "outputs": {
                "comparison_sheet": str(outdir / "comparison_sheet.png"),
                "rmse_by_h": str(outdir / "rmse_by_h.png"),
            },
        }
        shared.save_json(outdir / "final_summary.json", summary)
        print(json.dumps(shared.to_native(summary), indent=2, sort_keys=True))
    except Exception as exc:
        error_exit(outdir, exc)
        raise


if __name__ == "__main__":
    main()
