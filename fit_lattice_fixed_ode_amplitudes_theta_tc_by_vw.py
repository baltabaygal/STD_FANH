#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import collapse_and_fit_fanh_tosc as collapse
import fit_lattice_fixed_ode_amplitudes as shared
import fit_lattice_fixed_ode_amplitudes_theta_tc as theta_tc


ROOT = Path(__file__).resolve().parent
OUTDIR_DEFAULT = "results_lattice_fixed_ode_amplitudes_theta_tc_by_vw_beta0_tcmax100"
REFERENCE_SUMMARY_DEFAULT = (
    ROOT / "results_lattice_fixed_ode_amplitudes_vw0p9_H1p0H1p5H2p0_beta0_theta_tc_tcmax100" / "final_summary.json"
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit each v_w slice with ODE-fixed F0,F_inf, beta fixed, and per-theta t_c(theta0; v_w)."
    )
    p.add_argument("--vw-values", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    p.add_argument("--h-values", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    p.add_argument("--fixed-beta", type=float, default=0.0)
    p.add_argument("--t-osc", type=float, default=1.5)
    p.add_argument("--ode-summary", type=str, default=str(shared.ODE_SUMMARY_DEFAULT))
    p.add_argument("--reference-summary", type=str, default=str(REFERENCE_SUMMARY_DEFAULT))
    p.add_argument("--bootstrap", type=int, default=120)
    p.add_argument("--n-jobs", type=int, default=min(8, max(1, (shared.os_cpu_count() or 1))))
    p.add_argument("--tc-max", type=float, default=100.0)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--outdir", type=str, default=OUTDIR_DEFAULT)
    return p.parse_args()


def error_exit(outdir: Path, exc: Exception):
    payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
    outdir.mkdir(parents=True, exist_ok=True)
    shared.save_json(outdir / "_error.json", payload)
    print(json.dumps(payload, sort_keys=True))


def vw_tag(vw: float) -> str:
    return f"vw{str(float(vw)).replace('.', 'p')}"


def load_reference_summary(path: Path | None):
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def tc_theta_array_from_payload(payload: dict | None, theta_values: np.ndarray, key: str) -> np.ndarray | None:
    if payload is None or key not in payload:
        return None
    table = payload[key].get("tc_theta", {})
    vals = []
    for theta in theta_values:
        val = table.get(f"{float(theta):.10f}")
        if val is None:
            return None
        vals.append(float(val))
    return np.asarray(vals, dtype=np.float64)


def prepare_init(theta_values: np.ndarray, ref_summary: dict | None, fit_key: str, default_r: float, default_tc: float):
    init_r = default_r
    init_tc = np.full(len(theta_values), default_tc, dtype=np.float64)
    init_c = 1.0
    if ref_summary is not None and fit_key in ref_summary:
        init_r = float(ref_summary[fit_key].get("r", init_r))
        tc_vals = tc_theta_array_from_payload(ref_summary, theta_values, fit_key)
        if tc_vals is not None:
            init_tc = tc_vals
        init_c = float(ref_summary[fit_key].get("c_calib", init_c))
    return init_r, init_tc, init_c


def aggregate_prediction_curve(x_grid: np.ndarray, theta_idx: int, ode: dict, t_osc: float, fit_payload: dict):
    tc_theta = np.asarray(fit_payload["tc_theta"], dtype=np.float64)
    r = float(fit_payload["r"])
    c_calib = float(fit_payload.get("c_calib", 1.0))
    f0 = float(ode["F0"][theta_idx])
    finf = float(ode["F_inf"][theta_idx])
    plateau = np.power(np.maximum(x_grid / t_osc, 1.0e-18), 1.5) * finf / max(f0 * f0, 1.0e-18)
    transient = 1.0 / (1.0 + np.power(np.maximum(x_grid / max(tc_theta[theta_idx], 1.0e-18), 1.0e-18), r))
    return c_calib * (plateau + transient)


def build_predictions_frame(df, theta_values: np.ndarray, ode: dict, t_osc: float, fit_payload: dict, calib_name: str, vw_value: float):
    meta = shared.make_meta(df, theta_values, ode["F0"], ode["F_inf"], t_osc)
    tc_source = fit_payload["tc_theta"]
    if isinstance(tc_source, dict):
        tc_theta = np.asarray([float(tc_source[f"{float(theta):.10f}"]) for theta in theta_values], dtype=np.float64)
    else:
        tc_theta = np.asarray(tc_source, dtype=np.float64)
    y_fit = theta_tc.model_eval(meta, theta_values, float(fit_payload["beta"]), float(fit_payload["r"]), tc_theta, c_calib=float(fit_payload.get("c_calib", 1.0)))
    out = df.copy()
    out["v_w"] = float(vw_value)
    out["x"] = out["tp"].to_numpy(dtype=np.float64) * np.power(out["H"].to_numpy(dtype=np.float64), float(fit_payload["beta"]))
    out[f"xi_fit_{calib_name}"] = y_fit
    return out


def plot_combined_overlay(run_rows: list[dict], fit_key: str, outpath: Path, dpi: int, fixed_beta: float, t_osc: float):
    theta_values = run_rows[0]["theta_values"]
    cmap = plt.get_cmap("viridis")
    vw_values = [row["vw"] for row in run_rows]
    colors = {float(vw): cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    marker_map = {1.0: "s", 1.5: "^", 2.0: "D", 0.5: "o"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, theta in zip(axes, theta_values):
        idx = collapse.nearest_theta(theta_values, float(theta))
        for row in run_rows:
            vw = float(row["vw"])
            sub = row["df"][np.isclose(row["df"]["theta"], float(theta), atol=1.0e-8)].copy()
            payload = row[fit_key]
            for h in sorted(sub["H"].unique()):
                cur = sub[np.isclose(sub["H"], float(h), atol=1.0e-8)].sort_values("tp")
                if cur.empty:
                    continue
                x_data = cur["tp"].to_numpy(dtype=np.float64) * np.power(float(h), fixed_beta)
                ax.scatter(x_data, cur["xi"], s=20, color=colors[vw], marker=marker_map.get(float(h), "o"), alpha=0.85)
            x_all = sub["tp"].to_numpy(dtype=np.float64) * np.power(sub["H"].to_numpy(dtype=np.float64), fixed_beta)
            x_grid = np.geomspace(max(float(np.min(x_all)) * 0.95, 1.0e-4), float(np.max(x_all)) * 1.05, 250)
            y_curve = aggregate_prediction_curve(x_grid, idx, row["ode"], t_osc, payload)
            ax.plot(x_grid, y_curve, color=colors[vw], lw=1.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.set_xlabel(r"$t_p$" if np.isclose(fixed_beta, 0.0, atol=1.0e-12) else r"$x=t_p H^\beta$")
        ax.set_ylabel(r"$\xi$")
    for ax in axes[len(theta_values):]:
        ax.axis("off")
    vw_handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
    vw_labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
    h_handles = [plt.Line2D([0], [0], color="black", marker=marker_map[h], linestyle="None") for h in sorted(run_rows[0]["df"]["H"].unique())]
    h_labels = [rf"$H_*={h:g}$" for h in sorted(run_rows[0]["df"]["H"].unique())]
    fig.legend(vw_handles + h_handles, vw_labels + h_labels, loc="upper center", ncol=4, frameon=False)
    suffix = "with global $c_{calib}$" if fit_key == "fit_calib" else "no $c_{calib}$"
    fig.suptitle(rf"Per-$v_w$ fits with ODE-fixed $F_0,F_\infty$, $t_c(\theta_0; v_w)$, {suffix}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_combined_raw(run_rows: list[dict], fit_key: str, outdir: Path, dpi: int):
    theta_values = run_rows[0]["theta_values"]
    h_values = sorted(run_rows[0]["df"]["H"].unique())
    cmap = plt.get_cmap("viridis")
    vw_values = [row["vw"] for row in run_rows]
    colors = {float(vw): cmap(i / max(len(vw_values) - 1, 1)) for i, vw in enumerate(vw_values)}
    rmse_rows = []
    fit_col = "xi_fit_calib" if fit_key == "fit_calib" else "xi_fit_plain"
    for h_value in h_values:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
        axes = axes.ravel()
        for ax, theta in zip(axes, theta_values):
            for row in run_rows:
                sub = row["pred"][np.isclose(row["pred"]["H"], float(h_value), atol=1.0e-8) & np.isclose(row["pred"]["theta"], float(theta), atol=1.0e-8)].sort_values("beta_over_H")
                if sub.empty:
                    continue
                ax.scatter(sub["beta_over_H"], sub["xi"], s=22, color=colors[float(row["vw"])], alpha=0.85)
                ax.plot(sub["beta_over_H"], sub[fit_col], color=colors[float(row["vw"])], lw=1.8)
                rmse_rows.append(
                    {
                        "fit_kind": fit_key,
                        "H": float(h_value),
                        "theta": float(theta),
                        "v_w": float(row["vw"]),
                        "rel_rmse": shared.rel_rmse(sub["xi"], sub[fit_col]),
                    }
                )
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$\beta/H_*$")
            ax.set_ylabel(r"$\xi$")
        for ax in axes[len(theta_values):]:
            ax.axis("off")
        handles = [plt.Line2D([0], [0], color=colors[vw], lw=2.0) for vw in vw_values]
        labels = [rf"$v_w={vw:.1f}$" for vw in vw_values]
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
        suffix = "calib" if fit_key == "fit_calib" else "plain"
        fig.suptitle(rf"Per-$v_w$ $t_c(\theta_0)$ fits in raw $\xi(\beta/H_*)$, $H_*={h_value:.1f}$, {suffix}", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        tag = str(float(h_value)).replace(".", "p")
        fig.savefig(outdir / f"xi_vs_betaH_theta_tc_byvw_{suffix}_H{tag}.png", dpi=dpi)
        plt.close(fig)
    return rmse_rows


def plot_tc_theta_by_vw(run_rows: list[dict], fit_key: str, outpath: Path, dpi: int):
    theta_values = run_rows[0]["theta_values"]
    cmap = plt.get_cmap("viridis")
    colors = {float(row["vw"]): cmap(i / max(len(run_rows) - 1, 1)) for i, row in enumerate(run_rows)}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), constrained_layout=True)
    hvals = theta_tc.h_alt(theta_values)
    for row in run_rows:
        vw = float(row["vw"])
        tc_vals = np.asarray(row[fit_key]["tc_theta"], dtype=np.float64)
        axes[0].plot(theta_values, tc_vals, "o-", ms=4.0, lw=1.4, color=colors[vw], label=rf"$v_w={vw:.1f}$")
        axes[1].plot(hvals, tc_vals, "o-", ms=4.0, lw=1.4, color=colors[vw], label=rf"$v_w={vw:.1f}$")
    axes[0].set_xlabel(r"$\theta_0$")
    axes[0].set_ylabel(r"$t_c(\theta_0; v_w)$")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel(r"$h(\theta_0)=\log\!\left(e/[1-(\theta/\pi)^2]\right)$")
    axes[1].set_ylabel(r"$t_c(\theta_0; v_w)$")
    axes[1].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_global_vs_vw(run_rows: list[dict], outpath: Path, dpi: int):
    vw_values = np.asarray([float(row["vw"]) for row in run_rows], dtype=np.float64)
    r_plain = np.asarray([float(row["fit_plain"]["r"]) for row in run_rows], dtype=np.float64)
    r_calib = np.asarray([float(row["fit_calib"]["r"]) for row in run_rows], dtype=np.float64)
    rmse_plain = np.asarray([float(row["fit_plain"]["rel_rmse"]) for row in run_rows], dtype=np.float64)
    rmse_calib = np.asarray([float(row["fit_calib"]["rel_rmse"]) for row in run_rows], dtype=np.float64)
    c_calib = np.asarray([float(row["fit_calib"]["c_calib"]) for row in run_rows], dtype=np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), constrained_layout=True)
    axes[0].plot(vw_values, r_plain, "o-", lw=1.6, label="plain")
    axes[0].plot(vw_values, r_calib, "s--", lw=1.6, label="calib")
    axes[0].set_xlabel(r"$v_w$")
    axes[0].set_ylabel(r"$r(v_w)$")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(vw_values, rmse_plain, "o-", lw=1.6, label="plain")
    axes[1].plot(vw_values, rmse_calib, "s--", lw=1.6, label="calib")
    axes[1].set_xlabel(r"$v_w$")
    axes[1].set_ylabel("rel_rmse")
    axes[1].grid(alpha=0.25)
    axes[2].plot(vw_values, c_calib, "o-", lw=1.6, color="tab:green")
    axes[2].axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.7)
    axes[2].set_xlabel(r"$v_w$")
    axes[2].set_ylabel(r"$c_{calib}(v_w)$")
    axes[2].grid(alpha=0.25)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        ref_summary = load_reference_summary(Path(args.reference_summary).resolve() if args.reference_summary else None)

        run_rows = []
        prev_plain = None
        prev_calib = None
        for vw in sorted([float(v) for v in args.vw_values], reverse=True):
            subdir = outdir / vw_tag(vw)
            subdir.mkdir(parents=True, exist_ok=True)
            df = shared.load_lattice_dataset(vw, args.h_values)
            theta_values = np.sort(df["theta"].unique())
            ode = shared.load_ode_amplitudes(Path(args.ode_summary).resolve(), theta_values)
            meta = shared.make_meta(df, theta_values, ode["F0"], ode["F_inf"], args.t_osc)

            init_r_plain, init_tc_plain, _ = prepare_init(theta_values, ref_summary, "theta_tc", 2.0, 2.0)
            init_r_calib, init_tc_calib, init_c_calib = prepare_init(theta_values, ref_summary, "theta_tc_calib", 2.0, 2.0)
            if prev_plain is not None:
                init_r_plain = float(prev_plain["r"])
                init_tc_plain = np.asarray(prev_plain["tc_theta"], dtype=np.float64)
            if prev_calib is not None:
                init_r_calib = float(prev_calib["r"])
                init_tc_calib = np.asarray(prev_calib["tc_theta"], dtype=np.float64)
                init_c_calib = float(prev_calib["c_calib"])

            fit_plain = theta_tc.fit_case(meta, theta_values, args.fixed_beta, init_r_plain, init_tc_plain, with_calib=False, tc_max=args.tc_max)
            fit_calib = theta_tc.fit_case(
                meta,
                theta_values,
                args.fixed_beta,
                init_r_calib if prev_calib is not None else float(fit_plain["r"]),
                init_tc_calib if prev_calib is not None else np.asarray(fit_plain["tc_theta"], dtype=np.float64),
                with_calib=True,
                tc_max=args.tc_max,
            )
            if prev_calib is None:
                fit_calib["c_calib"] = float(fit_calib["c_calib"])
            boot_plain = theta_tc.bootstrap_case(meta, theta_values, args.fixed_beta, fit_plain, args.bootstrap, args.n_jobs, with_calib=False, tc_max=args.tc_max)
            boot_calib = theta_tc.bootstrap_case(meta, theta_values, args.fixed_beta, fit_calib, args.bootstrap, args.n_jobs, with_calib=True, tc_max=args.tc_max)

            plain_payload = theta_tc.summarize_payload(fit_plain, theta_values, ode, boot_plain)
            calib_payload = theta_tc.summarize_payload(fit_calib, theta_values, ode, boot_calib)

            theta_tc.plot_overlay(
                df,
                meta,
                theta_values,
                fit_plain,
                subdir / "collapse_overlay_theta_tc.png",
                rf"$v_w={vw:.1f}$ with ODE-fixed $F_0,F_\infty$, $\beta={args.fixed_beta:.1f}$, and $t_c(\theta_0)$",
                args.dpi,
            )
            theta_tc.plot_overlay(
                df,
                meta,
                theta_values,
                fit_calib,
                subdir / "collapse_overlay_theta_tc_calib.png",
                rf"$v_w={vw:.1f}$ with ODE-fixed $F_0,F_\infty$, $\beta={args.fixed_beta:.1f}$, $t_c(\theta_0)$, and $c_{{calib}}$",
                args.dpi,
            )
            theta_tc.plot_raw_betaH(df, meta, theta_values, fit_plain, subdir, "xi_vs_betaH_theta_tc", args.dpi)
            theta_tc.plot_raw_betaH(df, meta, theta_values, fit_calib, subdir, "xi_vs_betaH_theta_tc_calib", args.dpi)
            theta_tc.plot_tc_theta(theta_values, [("no calib", fit_plain), ("with calib", fit_calib)], subdir, args.dpi)

            pred_plain = build_predictions_frame(df, theta_values, ode, args.t_osc, plain_payload, "plain", float(vw))
            pred_calib = build_predictions_frame(df, theta_values, ode, args.t_osc, calib_payload, "calib", float(vw))
            pred = pred_plain.merge(
                pred_calib[["H", "theta", "tp", "beta_over_H", "v_w", "xi_fit_calib"]],
                on=["H", "theta", "tp", "beta_over_H", "v_w"],
                how="left",
            )
            pred.to_csv(subdir / "predictions.csv", index=False)

            per_vw_summary = {
                "status": "ok",
                "fixed_vw": float(vw),
                "fixed_beta": float(args.fixed_beta),
                "tc_max": float(args.tc_max),
                "H_values": [float(v) for v in sorted(df["H"].unique())],
                "theta_values": [float(v) for v in theta_values],
                "n_points": int(len(df)),
                "ode_amplitude_source": ode["source"],
                "ode_fit_summary": ode["ode_fit_summary"],
                "theta_tc": plain_payload,
                "theta_tc_calib": calib_payload,
            }
            shared.save_json(subdir / "fit_theta_tc.json", plain_payload)
            shared.save_json(subdir / "fit_theta_tc_calib.json", calib_payload)
            shared.save_json(subdir / "final_summary.json", per_vw_summary)

            run_rows.append(
                {
                    "vw": float(vw),
                    "theta_values": theta_values,
                    "ode": ode,
                    "df": df,
                    "pred": pred,
                    "fit_plain": fit_plain,
                    "fit_calib": fit_calib,
                    "plain_payload": plain_payload,
                    "calib_payload": calib_payload,
                    "subdir": str(subdir),
                }
            )
            prev_plain = fit_plain
            prev_calib = fit_calib

        run_rows = sorted(run_rows, key=lambda row: row["vw"])
        plot_combined_overlay(run_rows, "fit_plain", outdir / "collapse_overlay_allvw_theta_tc.png", args.dpi, args.fixed_beta, args.t_osc)
        plot_combined_overlay(run_rows, "fit_calib", outdir / "collapse_overlay_allvw_theta_tc_calib.png", args.dpi, args.fixed_beta, args.t_osc)
        rmse_rows = []
        rmse_rows.extend(plot_combined_raw(run_rows, "fit_plain", outdir, args.dpi))
        rmse_rows.extend(plot_combined_raw(run_rows, "fit_calib", outdir, args.dpi))
        plot_tc_theta_by_vw(run_rows, "fit_plain", outdir / "tc_theta_by_vw.png", args.dpi)
        plot_tc_theta_by_vw(run_rows, "fit_calib", outdir / "tc_theta_by_vw_calib.png", args.dpi)
        plot_global_vs_vw(run_rows, outdir / "global_params_vs_vw.png", args.dpi)

        global_plain_rmse = shared.rel_rmse(
            np.concatenate([row["pred"]["xi"].to_numpy(dtype=np.float64) for row in run_rows]),
            np.concatenate([row["pred"]["xi_fit_plain"].to_numpy(dtype=np.float64) for row in run_rows]),
        )
        global_calib_rmse = shared.rel_rmse(
            np.concatenate([row["pred"]["xi"].to_numpy(dtype=np.float64) for row in run_rows]),
            np.concatenate([row["pred"]["xi_fit_calib"].to_numpy(dtype=np.float64) for row in run_rows]),
        )

        fit_by_vw = {}
        for row in run_rows:
            fit_by_vw[f"{float(row['vw']):.1f}"] = {
                "plain": row["plain_payload"],
                "calib": row["calib_payload"],
                "outputs": {
                    "subdir": row["subdir"],
                    "collapse_plain": str(Path(row["subdir"]) / "collapse_overlay_theta_tc.png"),
                    "collapse_calib": str(Path(row["subdir"]) / "collapse_overlay_theta_tc_calib.png"),
                    "predictions": str(Path(row["subdir"]) / "predictions.csv"),
                },
            }

        summary = {
            "status": "ok",
            "vw_values": [float(row["vw"]) for row in run_rows],
            "fixed_beta": float(args.fixed_beta),
            "tc_max": float(args.tc_max),
            "global_rel_rmse_plain": float(global_plain_rmse),
            "global_rel_rmse_calib": float(global_calib_rmse),
            "fit_by_vw": fit_by_vw,
            "raw_plot_rmse": rmse_rows,
            "outputs": {
                "collapse_allvw_plain": str(outdir / "collapse_overlay_allvw_theta_tc.png"),
                "collapse_allvw_calib": str(outdir / "collapse_overlay_allvw_theta_tc_calib.png"),
                "tc_theta_by_vw": str(outdir / "tc_theta_by_vw.png"),
                "tc_theta_by_vw_calib": str(outdir / "tc_theta_by_vw_calib.png"),
                "global_params_vs_vw": str(outdir / "global_params_vs_vw.png"),
            },
        }
        shared.save_json(outdir / "final_summary.json", summary)
        print(json.dumps(shared.to_native(summary), indent=2, sort_keys=True))
    except Exception as exc:
        error_exit(outdir, exc)
        raise


if __name__ == "__main__":
    main()
