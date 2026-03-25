#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.image import imread


ROOT = Path(__file__).resolve().parent
BENCHMARK_SUMMARY_DEFAULT = (
    ROOT / "results_lattice_theta_tc_tosc_and_s_tests_beta0_tcmax300" / "final_summary.json"
)
BENCHMARK_PRED_DEFAULT = (
    ROOT / "results_lattice_theta_tc_tosc_and_s_tests_beta0_tcmax300" / "calib_free_tosc" / "predictions.csv"
)
BOUNDARY_SUMMARY_DEFAULT = (
    ROOT / "results_lattice_theta_tc_sharedr_free_finf_vwtheta_tests_beta0_tcmax300" / "final_summary.json"
)
OUTDIR_DEFAULT = ROOT / "results_xi_vs_tp_by_vw_hstar_compare"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot xi vs tp separately for each vw, overlaying H* data and the current "
            "benchmark vs boundary-consistent fits."
        )
    )
    p.add_argument("--benchmark-summary", type=str, default=str(BENCHMARK_SUMMARY_DEFAULT))
    p.add_argument("--benchmark-predictions", type=str, default=str(BENCHMARK_PRED_DEFAULT))
    p.add_argument("--boundary-summary", type=str, default=str(BOUNDARY_SUMMARY_DEFAULT))
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--outdir", type=str, default=str(OUTDIR_DEFAULT))
    return p.parse_args()


def theta_key(theta: float) -> str:
    return f"{float(theta):.10f}"


def vw_key(vw: float) -> str:
    return f"{float(vw):.1f}"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    return json.loads(path.read_text())


def xi_scale(tp: np.ndarray, t_osc: float) -> np.ndarray:
    return np.power(np.maximum((2.0 * np.asarray(tp, dtype=np.float64)) / max(3.0 * float(t_osc), 1.0e-18), 1.0e-18), 1.5)


def benchmark_curve(tp: np.ndarray, theta: float, vw: float, bench_case: dict) -> np.ndarray:
    t_osc = float(bench_case["t_osc"])
    r = float(bench_case["r"])
    c = float(bench_case["c_calib_by_vw"][vw_key(vw)])
    f0 = float(bench_case["F0_ode"][theta_key(theta)])
    f_inf = float(bench_case["f_infty_ode"][theta_key(theta)])
    tc = float(bench_case["tc_by_vw_theta"][vw_key(vw)][theta_key(theta)])
    scale = xi_scale(tp, t_osc)
    denom = 1.0 + np.power(np.maximum(tp / max(tc, 1.0e-18), 1.0e-18), r)
    f_tilde = c * (f_inf + f0 / np.maximum(scale * denom, 1.0e-18))
    return scale * f_tilde / max(f0, 1.0e-18)


def boundary_curve(tp: np.ndarray, theta: float, vw: float, boundary_case: dict) -> np.ndarray:
    t_osc = float(boundary_case["t_osc"])
    r = float(boundary_case["r"])
    f0 = float(boundary_case["F0_ode"][theta_key(theta)])
    f_inf = float(boundary_case["f_infty_by_vw_theta"][vw_key(vw)][theta_key(theta)])
    tc = float(boundary_case["tc_by_vw_theta"][vw_key(vw)][theta_key(theta)])
    scale = xi_scale(tp, t_osc)
    denom = 1.0 + np.power(np.maximum(tp / max(tc, 1.0e-18), 1.0e-18), r)
    f_tilde = f_inf + f0 / np.maximum(scale * denom, 1.0e-18)
    return scale * f_tilde / max(f0, 1.0e-18)


def plot_vw_figure(df: pd.DataFrame, vw: float, bench_case: dict, boundary_case: dict, outpath: Path, dpi: int) -> None:
    theta_values = np.sort(df["theta"].unique())
    h_values = sorted(df["H"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {float(h): cmap(i / max(len(h_values) - 1, 1)) for i, h in enumerate(h_values)}
    marker_map = {1.0: "s", 1.5: "^", 2.0: "D", 0.5: "o"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
    axes = axes.ravel()
    sub_vw = df[np.isclose(df["v_w"], float(vw), atol=1.0e-12)].copy()

    for ax, theta in zip(axes, theta_values):
        sub = sub_vw[np.isclose(sub_vw["theta"], float(theta), atol=1.0e-12)].copy()
        tp_min = float(sub["tp"].min())
        tp_max = float(sub["tp"].max())
        tp_grid = np.geomspace(tp_min * 0.92, tp_max * 1.08, 500)
        for h in h_values:
            cur = sub[np.isclose(sub["H"], float(h), atol=1.0e-12)].sort_values("tp")
            ax.scatter(
                cur["tp"],
                cur["xi"],
                s=23,
                color=colors[float(h)],
                marker=marker_map.get(float(h), "o"),
                alpha=0.9,
            )
        ax.plot(tp_grid, benchmark_curve(tp_grid, float(theta), float(vw), bench_case), color="black", lw=2.0, label="benchmark")
        ax.plot(tp_grid, boundary_curve(tp_grid, float(theta), float(vw), boundary_case), color="tab:orange", lw=1.8, ls="--", label=r"$f_\infty(\theta_0,v_w)$")
        ax.axhline(1.0, color="tab:red", lw=1.0, ls=":", alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta_0={float(theta):.3f}$")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$\xi$")

    for ax in axes[len(theta_values):]:
        ax.axis("off")

    h_handles = [
        plt.Line2D([0], [0], color=colors[float(h)], marker=marker_map.get(float(h), "o"), linestyle="None")
        for h in h_values
    ]
    h_labels = [rf"data $H_*={float(h):.1f}$" for h in h_values]
    fit_handles = [
        plt.Line2D([0], [0], color="black", lw=2.0),
        plt.Line2D([0], [0], color="tab:orange", lw=1.8, ls="--"),
    ]
    fit_labels = [
        rf"benchmark, rel_rmse={float(bench_case['per_vw_rel_rmse'][vw_key(vw)]):.4f}",
        rf"$f_\infty(\theta_0,v_w)$, rel_rmse={float(boundary_case['per_vw_rel_rmse'][vw_key(vw)]):.4f}",
    ]
    fig.legend(h_handles + fit_handles, h_labels + fit_labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(rf"$\xi(t_p)$ diagnostics at $v_w={float(vw):.1f}$", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def make_sheet(image_paths: list[tuple[str, Path]], outpath: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, (label, path) in zip(axes.ravel(), image_paths):
        ax.imshow(imread(path))
        ax.set_title(label, fontsize=12)
        ax.axis("off")
    fig.suptitle(r"$\xi(t_p)$ by $v_w$: diagnosing the $H_*$ dependence", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    benchmark_summary = load_json(Path(args.benchmark_summary).resolve())
    boundary_summary = load_json(Path(args.boundary_summary).resolve())
    df = pd.read_csv(Path(args.benchmark_predictions).resolve())

    bench_case = benchmark_summary["cases"]["calib_free_tosc"]
    boundary_case = boundary_summary["cases"]["vwtheta_free_tosc"]
    vw_values = np.sort(df["v_w"].unique())

    images = []
    for vw in vw_values:
        outpath = outdir / f"xi_vs_tp_compare_vw{str(float(vw)).replace('.', 'p')}.png"
        plot_vw_figure(df, float(vw), bench_case, boundary_case, outpath, args.dpi)
        images.append((rf"$v_w={float(vw):.1f}$", outpath))

    make_sheet(images, outdir / "comparison_sheet.png", args.dpi)

    payload = {
        "status": "ok",
        "benchmark_summary": str(Path(args.benchmark_summary).resolve()),
        "benchmark_predictions": str(Path(args.benchmark_predictions).resolve()),
        "boundary_summary": str(Path(args.boundary_summary).resolve()),
        "outputs": {
            f"vw_{float(vw):.1f}": str(path) for (vw, path) in [(float(lbl.split("=")[1].strip("$")), p) for lbl, p in images]
        },
        "comparison_sheet": str(outdir / "comparison_sheet.png"),
    }
    (outdir / "final_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
