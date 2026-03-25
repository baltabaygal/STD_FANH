#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import gammaincc


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTDIR = ROOT / "results_ode_transition_kernel_compare_fixed_plateau"
DEFAULT_TABLES = [
    ROOT / "ode/analysis/data/dm_tp_fitready_H0p500.txt",
    ROOT / "ode/analysis/data/dm_tp_fitready_H1p000.txt",
    ROOT / "ode/analysis/data/dm_tp_fitready_H1p500.txt",
    ROOT / "ode/analysis/data/dm_tp_fitready_H2p000.txt",
]
T_OSC = 1.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare direct ODE transition kernels with F0 and F_inf fixed from ODE."
    )
    p.add_argument("--tables", nargs="*", default=[str(p) for p in DEFAULT_TABLES])
    p.add_argument("--anchor-h-values", type=float, nargs="+", default=[1.0, 2.0])
    p.add_argument("--tail-frac", type=float, default=0.4)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    return p.parse_args()


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
    return obj


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(to_native(payload), indent=2, sort_keys=True))


def load_table(path: Path) -> pd.DataFrame:
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    cols = [
        "H",
        "t_star",
        "theta",
        "tp",
        "tp_over_tosc",
        "Ea3_PT",
        "Ea3_noPT",
        "fanh_PT",
        "fanh_noPT",
        "xi",
        "nsteps_PT",
        "nsteps_noPT",
    ]
    return pd.DataFrame(arr[:, : len(cols)], columns=cols)


def load_data(paths: list[Path]) -> tuple[pd.DataFrame, np.ndarray]:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing direct ODE fit-ready table: {path}")
        frames.append(load_table(path))
    df = pd.concat(frames, ignore_index=True)
    df = df[
        np.isfinite(df["H"])
        & np.isfinite(df["theta"])
        & np.isfinite(df["tp"])
        & np.isfinite(df["xi"])
        & np.isfinite(df["fanh_noPT"])
        & np.isfinite(df["fanh_PT"])
        & (df["tp"] > 0.0)
        & (df["xi"] > 0.0)
        & (df["fanh_noPT"] > 0.0)
        & (df["fanh_PT"] > 0.0)
    ].copy()
    theta_values = np.sort(df["theta"].unique())
    theta_index = {float(th): i for i, th in enumerate(theta_values)}
    df["theta_idx"] = [theta_index[float(th)] for th in df["theta"]]
    f0_map = df.groupby("theta", as_index=False)["fanh_noPT"].median().rename(columns={"fanh_noPT": "F0"})
    df = df.merge(f0_map, on="theta", how="left")
    return df.sort_values(["H", "theta", "tp"]).reset_index(drop=True), theta_values


def fit_tail_model(tp: np.ndarray, y: np.ndarray) -> dict:
    tp = np.asarray(tp, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def model(par: np.ndarray) -> np.ndarray:
        finf, amp, alpha = par
        return finf + amp / np.power(tp, alpha)

    def resid(par: np.ndarray) -> np.ndarray:
        return (model(par) - y) / np.maximum(y, 1.0e-18)

    finf0 = max(float(np.min(y)), 1.0e-12)
    amp0 = max(float(np.max(y) - finf0), 1.0e-12) * float(np.min(tp))
    x0 = np.array([finf0, amp0, 1.0], dtype=np.float64)
    lower = np.array([1.0e-12, 0.0, 1.0e-3], dtype=np.float64)
    upper = np.array([1.0e6, 1.0e6, 10.0], dtype=np.float64)
    res0 = least_squares(resid, x0=x0, bounds=(lower, upper), loss="soft_l1", f_scale=0.02, max_nfev=8000)
    res = least_squares(resid, x0=res0.x, bounds=(lower, upper), loss="linear", max_nfev=8000)
    yfit = model(res.x)
    return {
        "params": res.x,
        "success": bool(res.success),
        "message": str(res.message),
        "rel_rmse": float(np.sqrt(np.mean(np.square((yfit - y) / np.maximum(y, 1.0e-18))))),
    }


def extract_plateau_anchors(df: pd.DataFrame, theta_values: np.ndarray, anchor_h_values: list[float], tail_frac: float, outdir: Path, dpi: int) -> pd.DataFrame:
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(anchor_h_values)))
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.3), sharex=False, sharey=False)
    rows = []
    for ax, theta in zip(axes.flat, theta_values):
        sub = df[np.isclose(df["theta"], float(theta), atol=1.0e-12)].copy().sort_values("tp")
        per_h = []
        for color, h in zip(colors, anchor_h_values):
            hsub = sub[np.isclose(sub["H"], float(h), atol=1.0e-12)].copy().sort_values("tp")
            ntail = max(8, int(math.ceil(tail_frac * len(hsub))))
            tail = hsub.tail(ntail).copy()
            rec = fit_tail_model(tail["tp"].to_numpy(dtype=np.float64), tail["fanh_PT"].to_numpy(dtype=np.float64))
            finf, amp, alpha = [float(v) for v in rec["params"]]
            xfit = np.geomspace(float(np.min(tail["tp"])), float(np.max(tail["tp"])), 200)
            yfit = finf + amp / np.power(xfit, alpha)
            ax.plot(hsub["tp"], hsub["fanh_PT"], "o", ms=2.5, color=color, alpha=0.75, label=rf"$H={h:g}$" if theta == theta_values[0] else None)
            ax.plot(xfit, yfit, "-", lw=1.6, color=color)
            per_h.append(
                {
                    "H": float(h),
                    "F_inf_fit": finf,
                    "A_fit": amp,
                    "alpha_fit": alpha,
                    "tail_tp_min": float(np.min(tail["tp"])),
                    "tail_tp_max": float(np.max(tail["tp"])),
                    "tail_n": int(ntail),
                    "rel_rmse_tail": float(rec["rel_rmse"]),
                }
            )
        vals = np.array([row["F_inf_fit"] for row in per_h], dtype=np.float64)
        anchor = float(np.mean(vals))
        err = float(0.5 * (np.max(vals) - np.min(vals)))
        ax.axhline(anchor, color="black", ls="--", lw=1.5, label="anchor" if theta == theta_values[0] else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$f_{\rm anh,PT}$")
        ax.set_title(rf"$\theta={theta:.3f}$, $F_\infty={anchor:.4g}$")
        ax.grid(alpha=0.25)
        rows.append({"theta": float(theta), "F_inf_anchor": anchor, "F_inf_err": err, "per_H": per_h})
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "ode_plateau_anchor_fits.png", dpi=dpi)
    plt.close(fig)

    flat_rows = []
    for row in rows:
        payload = {"theta": row["theta"], "F_inf_anchor": row["F_inf_anchor"], "F_inf_err": row["F_inf_err"]}
        for hrec in row["per_H"]:
            tag = f"H{hrec['H']:g}".replace(".", "p")
            payload[f"F_inf_{tag}"] = hrec["F_inf_fit"]
            payload[f"alpha_{tag}"] = hrec["alpha_fit"]
            payload[f"tail_tp_min_{tag}"] = hrec["tail_tp_min"]
            payload[f"tail_tp_max_{tag}"] = hrec["tail_tp_max"]
            payload[f"tail_rel_rmse_{tag}"] = hrec["rel_rmse_tail"]
        flat_rows.append(payload)
    out = pd.DataFrame(flat_rows).sort_values("theta").reset_index(drop=True)
    out.to_csv(outdir / "plateau_anchor_table.csv", index=False)
    save_json(outdir / "plateau_anchor_summary.json", {"anchor_h_values": anchor_h_values, "tail_frac": tail_frac, "rows": rows})
    return out


def kernel_defs():
    return {
        "hill_power": {
            "param_names": ["t_c", "r"],
            "x0": np.array([1.5, 2.0], dtype=np.float64),
            "lower": np.array([0.05, 0.05], dtype=np.float64),
            "upper": np.array([100.0, 20.0], dtype=np.float64),
            "fn": lambda z, p: 1.0 / (1.0 + np.power(z, p[1])),
        },
        "shifted_hill": {
            "param_names": ["t_c", "r"],
            "x0": np.array([1.5, 2.0], dtype=np.float64),
            "lower": np.array([0.05, 0.05], dtype=np.float64),
            "upper": np.array([100.0, 20.0], dtype=np.float64),
            "fn": lambda z, p: np.power(1.0 + z, -p[1]),
        },
        "q_hill": {
            "param_names": ["t_c", "r", "q"],
            "x0": np.array([1.5, 2.0, 1.0], dtype=np.float64),
            "lower": np.array([0.05, 0.05, 0.05], dtype=np.float64),
            "upper": np.array([100.0, 20.0, 10.0], dtype=np.float64),
            "fn": lambda z, p: np.power(1.0 + np.power(z, p[1]), -p[2]),
        },
        "stretched_exp": {
            "param_names": ["t_c", "r"],
            "x0": np.array([1.5, 1.0], dtype=np.float64),
            "lower": np.array([0.05, 0.05], dtype=np.float64),
            "upper": np.array([100.0, 20.0], dtype=np.float64),
            "fn": lambda z, p: np.exp(-np.power(z, p[1])),
        },
        "gamma_survival": {
            "param_names": ["t_c", "k"],
            "x0": np.array([1.5, 2.0], dtype=np.float64),
            "lower": np.array([0.05, 0.05], dtype=np.float64),
            "upper": np.array([100.0, 50.0], dtype=np.float64),
            "fn": lambda z, p: gammaincc(p[1], p[1] * z),
        },
    }


def theta_anchor_array(anchor_df: pd.DataFrame, theta_values: np.ndarray) -> np.ndarray:
    amap = {float(row.theta): float(row.F_inf_anchor) for row in anchor_df.itertuples(index=False)}
    return np.array([amap[float(theta)] for theta in theta_values], dtype=np.float64)


def xi_model(params: np.ndarray, df: pd.DataFrame, theta_values: np.ndarray, finf_anchor: np.ndarray, kernel_name: str) -> np.ndarray:
    kernels = kernel_defs()
    z = df["tp"].to_numpy(dtype=np.float64) / max(float(params[0]), 1.0e-18)
    T = kernels[kernel_name]["fn"](np.maximum(z, 1.0e-18), params)
    theta_idx = df["theta_idx"].to_numpy(dtype=np.int64)
    f0 = df["F0"].to_numpy(dtype=np.float64)
    tp = df["tp"].to_numpy(dtype=np.float64)
    plateau = np.power(tp / T_OSC, 1.5) * finf_anchor[theta_idx] / np.maximum(f0, 1.0e-18)
    return plateau + T


def fanh_model(tp: np.ndarray, f0: np.ndarray, finf: np.ndarray, params: np.ndarray, kernel_name: str) -> np.ndarray:
    kernels = kernel_defs()
    z = tp / max(float(params[0]), 1.0e-18)
    T = kernels[kernel_name]["fn"](np.maximum(z, 1.0e-18), params)
    return finf + f0 * T / np.maximum(np.power(tp / T_OSC, 1.5), 1.0e-18)


def fit_kernel(df: pd.DataFrame, theta_values: np.ndarray, finf_anchor: np.ndarray, kernel_name: str) -> dict:
    kernels = kernel_defs()
    spec = kernels[kernel_name]

    def resid(par: np.ndarray) -> np.ndarray:
        yfit = xi_model(par, df, theta_values, finf_anchor, kernel_name)
        y = df["xi"].to_numpy(dtype=np.float64)
        return (yfit - y) / np.maximum(y, 1.0e-12)

    res0 = least_squares(resid, x0=spec["x0"], bounds=(spec["lower"], spec["upper"]), loss="soft_l1", f_scale=0.03, max_nfev=20000)
    res = least_squares(resid, x0=res0.x, bounds=(spec["lower"], spec["upper"]), loss="linear", max_nfev=20000)
    yfit = xi_model(res.x, df, theta_values, finf_anchor, kernel_name)
    frac = (yfit - df["xi"].to_numpy(dtype=np.float64)) / np.maximum(df["xi"].to_numpy(dtype=np.float64), 1.0e-12)
    rss = float(np.sum(np.square(frac)))
    n = int(len(df))
    k = int(len(res.x))
    aic = float(n * math.log(max(rss, 1.0e-18) / n) + 2.0 * k)
    bic = float(n * math.log(max(rss, 1.0e-18) / n) + k * math.log(n))
    payload = {
        "status": "ok" if res.success else "failed",
        "success": bool(res.success),
        "message": str(res.message),
        "kernel": kernel_name,
        "param_names": spec["param_names"],
        "params": {name: float(val) for name, val in zip(spec["param_names"], res.x)},
        "rel_rmse": float(np.sqrt(np.mean(np.square(frac)))),
        "AIC": aic,
        "BIC": bic,
        "n_points": n,
        "n_params": k,
        "per_H_rel_rmse": {},
    }
    for h in sorted(df["H"].unique()):
        mask = np.isclose(df["H"].to_numpy(dtype=np.float64), float(h), atol=1.0e-12)
        payload["per_H_rel_rmse"][f"{h:.1f}"] = float(
            np.sqrt(np.mean(np.square((yfit[mask] - df.loc[mask, "xi"].to_numpy(dtype=np.float64)) / np.maximum(df.loc[mask, "xi"].to_numpy(dtype=np.float64), 1.0e-12))))
        )
    return {"payload": payload, "params": res.x, "xi_fit": yfit, "frac_resid": frac}


def plot_xi_by_H(df: pd.DataFrame, theta_values: np.ndarray, finf_anchor: np.ndarray, fit_bundle: dict, outdir: Path, dpi: int) -> None:
    kernel = fit_bundle["payload"]["kernel"]
    for h in sorted(df["H"].unique()):
        sub_h = df[np.isclose(df["H"], float(h), atol=1.0e-12)].copy()
        fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0))
        for ax, theta in zip(axes.flat, theta_values):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=1.0e-12)].sort_values("tp")
            tp = sub["tp"].to_numpy(dtype=np.float64)
            xi = sub["xi"].to_numpy(dtype=np.float64)
            f0 = sub["F0"].to_numpy(dtype=np.float64)
            finf = np.full_like(tp, finf_anchor[int(np.argmin(np.abs(theta_values - float(theta))))], dtype=np.float64)
            tp_grid = np.geomspace(float(tp.min()), float(tp.max()), 300)
            f0g = np.full_like(tp_grid, float(np.median(f0)))
            finfg = np.full_like(tp_grid, float(finf[0]))
            xi_fit = (tp_grid / T_OSC) ** 1.5 * finfg / np.maximum(f0g, 1.0e-18) + kernel_defs()[kernel]["fn"](tp_grid / max(float(fit_bundle["params"][0]), 1.0e-18), fit_bundle["params"])
            ax.plot(tp, xi, "o", ms=3.2, color="tab:blue", label="ODE data")
            ax.plot(tp_grid, xi_fit, lw=2.0, color="black", label=kernel)
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$t_p$")
            ax.set_ylabel(r"$\xi$")
        axes.flat[0].legend(frameon=False, fontsize=8)
        param_txt = ", ".join([rf"{k}={v:.3g}" for k, v in fit_bundle["payload"]["params"].items()])
        fig.suptitle(rf"Direct ODE fixed-plateau fit, $H_*={h:.1f}$" + "\n" + rf"{kernel}: {param_txt}", y=0.98)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
        tag = f"H{h:.1f}".replace(".", "p")
        fig.savefig(outdir / f"xi_fit_overlay_{tag}.png", dpi=dpi)
        plt.close(fig)


def plot_fanh_by_H(df: pd.DataFrame, theta_values: np.ndarray, finf_anchor: np.ndarray, fit_bundle: dict, outdir: Path, dpi: int) -> None:
    kernel = fit_bundle["payload"]["kernel"]
    for h in sorted(df["H"].unique()):
        sub_h = df[np.isclose(df["H"], float(h), atol=1.0e-12)].copy()
        fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0))
        for ax, theta in zip(axes.flat, theta_values):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=1.0e-12)].sort_values("tp")
            tp = sub["tp"].to_numpy(dtype=np.float64)
            fanh = sub["fanh_PT"].to_numpy(dtype=np.float64)
            f0 = sub["F0"].to_numpy(dtype=np.float64)
            finf = np.full_like(tp, finf_anchor[int(np.argmin(np.abs(theta_values - float(theta))))], dtype=np.float64)
            tp_grid = np.geomspace(float(tp.min()), float(tp.max()), 300)
            f0g = np.full_like(tp_grid, float(np.median(f0)))
            finfg = np.full_like(tp_grid, float(finf[0]))
            fanh_fit = fanh_model(tp_grid, f0g, finfg, fit_bundle["params"], kernel)
            ax.plot(tp, fanh, "o", ms=3.2, color="tab:green", label="ODE data")
            ax.plot(tp_grid, fanh_fit, lw=2.0, color="black", label=kernel)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$t_p$")
            ax.set_ylabel(r"$f_{\rm anh}$")
        axes.flat[0].legend(frameon=False, fontsize=8)
        param_txt = ", ".join([rf"{k}={v:.3g}" for k, v in fit_bundle["payload"]["params"].items()])
        fig.suptitle(rf"Direct ODE fixed-plateau fanh fit, $H_*={h:.1f}$" + "\n" + rf"{kernel}: {param_txt}", y=0.98)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
        tag = f"H{h:.1f}".replace(".", "p")
        fig.savefig(outdir / f"fanh_fit_overlay_{tag}.png", dpi=dpi)
        plt.close(fig)


def plot_model_comparison(comp_df: pd.DataFrame, outdir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    comp = comp_df.sort_values("rel_rmse").reset_index(drop=True)
    axes[0].bar(comp["kernel"], comp["rel_rmse"], color="tab:blue", alpha=0.8)
    axes[0].set_ylabel("rel-RMSE")
    axes[0].set_title("Kernel comparison by rel-RMSE")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(alpha=0.25, axis="y")
    comp2 = comp_df.sort_values("AIC").reset_index(drop=True)
    axes[1].bar(comp2["kernel"], comp2["AIC"], color="tab:orange", alpha=0.8)
    axes[1].set_ylabel("AIC")
    axes[1].set_title("Kernel comparison by AIC")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(outdir / "kernel_comparison.png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    tables = [Path(p).resolve() if not Path(p).is_absolute() else Path(p) for p in args.tables]
    df, theta_values = load_data(tables)
    anchor_df = extract_plateau_anchors(df, theta_values, [float(v) for v in args.anchor_h_values], float(args.tail_frac), outdir, args.dpi)
    finf_anchor = theta_anchor_array(anchor_df, theta_values)

    comp_rows = []
    best_kernel = None
    best_rmse = np.inf
    for kernel_name in kernel_defs().keys():
        kernel_out = outdir / kernel_name
        kernel_out.mkdir(parents=True, exist_ok=True)
        fit_bundle = fit_kernel(df, theta_values, finf_anchor, kernel_name)
        pred = df.copy()
        pred["xi_model"] = fit_bundle["xi_fit"]
        pred["frac_resid"] = fit_bundle["frac_resid"]
        pred.to_csv(kernel_out / "predictions.csv", index=False)
        save_json(kernel_out / "global_fit.json", fit_bundle["payload"])
        plot_xi_by_H(df, theta_values, finf_anchor, fit_bundle, kernel_out, args.dpi)
        plot_fanh_by_H(df, theta_values, finf_anchor, fit_bundle, kernel_out, args.dpi)
        comp_rows.append(
            {
                "kernel": kernel_name,
                "rel_rmse": fit_bundle["payload"]["rel_rmse"],
                "AIC": fit_bundle["payload"]["AIC"],
                "BIC": fit_bundle["payload"]["BIC"],
                **{f"param_{k}": v for k, v in fit_bundle["payload"]["params"].items()},
            }
        )
        if fit_bundle["payload"]["rel_rmse"] < best_rmse:
            best_rmse = fit_bundle["payload"]["rel_rmse"]
            best_kernel = kernel_name

    comp_df = pd.DataFrame(comp_rows).sort_values("rel_rmse").reset_index(drop=True)
    comp_df.to_csv(outdir / "kernel_comparison.csv", index=False)
    plot_model_comparison(comp_df, outdir, args.dpi)

    summary = {
        "status": "ok",
        "n_points": int(len(df)),
        "theta_values": [float(v) for v in theta_values],
        "anchor_h_values": [float(v) for v in args.anchor_h_values],
        "tail_frac": float(args.tail_frac),
        "best_kernel": str(best_kernel),
        "best_rel_rmse": float(best_rmse),
        "kernel_ranking": comp_df.to_dict(orient="records"),
        "outputs": {
            "anchor_table": str((outdir / "plateau_anchor_table.csv").resolve()),
            "comparison_csv": str((outdir / "kernel_comparison.csv").resolve()),
            "comparison_plot": str((outdir / "kernel_comparison.png").resolve()),
            "best_dir": str((outdir / str(best_kernel)).resolve()) if best_kernel else None,
        },
    }
    save_json(outdir / "final_summary.json", summary)
    print(json.dumps(to_native(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
