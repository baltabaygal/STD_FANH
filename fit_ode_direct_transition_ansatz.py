#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTDIR = ROOT / "results_ode_direct_transition_ansatz"
DEFAULT_TABLES = [
    ROOT / "ode/analysis/data/dm_tp_fitready_H0p500.txt",
    ROOT / "ode/analysis/data/dm_tp_fitready_H1p000.txt",
    ROOT / "ode/analysis/data/dm_tp_fitready_H1p500.txt",
    ROOT / "ode/analysis/data/dm_tp_fitready_H2p000.txt",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit the direct ODE tp-scan data with the current transition ansatz."
    )
    p.add_argument("--tables", nargs="*", default=[str(p) for p in DEFAULT_TABLES])
    p.add_argument("--t-osc", type=float, default=1.5)
    p.add_argument("--tc0", type=float, default=1.5)
    p.add_argument("--r0", type=float, default=3.0)
    p.add_argument("--nboot", type=int, default=120)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    return p.parse_args()


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def error_exit(outdir: Path, exc: Exception) -> None:
    payload = {
        "status": "error",
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "_error.json", payload)
    print(json.dumps(payload, sort_keys=True))


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


def xi_scale_from_tp(tp: np.ndarray, t_osc: float) -> np.ndarray:
    return np.power(
        np.maximum((2.0 * np.asarray(tp, dtype=np.float64)) / max(3.0 * float(t_osc), 1.0e-18), 1.0e-18),
        1.5,
    )


def load_data(paths: list[Path], t_osc: float) -> tuple[pd.DataFrame, np.ndarray]:
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
        & (df["tp"] > 0.0)
        & (df["xi"] > 0.0)
        & (df["fanh_noPT"] > 0.0)
    ].copy()
    theta_values = np.sort(df["theta"].unique())
    theta_index = {float(th): i for i, th in enumerate(theta_values)}
    df["theta_idx"] = [theta_index[float(th)] for th in df["theta"]]
    # Use the direct ODE no-PT amplitude as F0 for the pure ODE ansatz test.
    f0_map = (
        df.groupby("theta", as_index=False)["fanh_noPT"]
        .median()
        .rename(columns={"fanh_noPT": "F0"})
    )
    df = df.merge(f0_map, on="theta", how="left")
    df["fanh_model_data"] = df["xi"] * df["F0"] / np.maximum(xi_scale_from_tp(df["tp"].to_numpy(dtype=np.float64), t_osc), 1.0e-18)
    return df.sort_values(["H", "theta", "tp"]).reset_index(drop=True), theta_values


def estimate_finf_init(df: pd.DataFrame, theta_values: np.ndarray, t_osc: float) -> np.ndarray:
    out = np.zeros(len(theta_values), dtype=np.float64)
    for i, theta in enumerate(theta_values):
        sub = df[np.isclose(df["theta"], float(theta), atol=1.0e-12)].sort_values("tp").copy()
        n_tail = max(5, int(math.ceil(0.10 * len(sub))))
        tail = sub.tail(n_tail)
        coeff = tail["xi"].to_numpy(dtype=np.float64) / np.maximum(xi_scale_from_tp(tail["tp"].to_numpy(dtype=np.float64), t_osc), 1.0e-18)
        out[i] = max(float(np.median(coeff * np.square(tail["F0"].to_numpy(dtype=np.float64)))), 1.0e-8)
    return out


def unpack_params(params: np.ndarray, n_theta: int) -> tuple[float, float, np.ndarray]:
    tc = float(params[0])
    r = float(params[1])
    finf = np.asarray(params[2 : 2 + n_theta], dtype=np.float64)
    return tc, r, finf


def model_eval(params: np.ndarray, df: pd.DataFrame, t_osc: float, theta_values: np.ndarray) -> np.ndarray:
    tc, r, finf = unpack_params(params, len(theta_values))
    tp = df["tp"].to_numpy(dtype=np.float64)
    f0_sq = np.square(df["F0"].to_numpy(dtype=np.float64))
    theta_idx = df["theta_idx"].to_numpy(dtype=np.int64)
    plateau = xi_scale_from_tp(tp, t_osc) * finf[theta_idx] / np.maximum(f0_sq, 1.0e-18)
    transient = 1.0 / (1.0 + np.power(np.maximum(tp, 1.0e-18) / max(tc, 1.0e-18), r))
    return plateau + transient


def residual_vector(
    params: np.ndarray,
    df: pd.DataFrame,
    t_osc: float,
    theta_values: np.ndarray,
    reg_finf: float,
    finf_ref: np.ndarray,
) -> np.ndarray:
    y_model = model_eval(params, df, t_osc, theta_values)
    resid = (y_model - df["xi"].to_numpy(dtype=np.float64)) / np.maximum(df["xi"].to_numpy(dtype=np.float64), 1.0e-12)
    if reg_finf > 0.0:
        _, _, finf = unpack_params(params, len(theta_values))
        reg = math.sqrt(reg_finf) * (finf - finf_ref) / np.maximum(finf_ref, 1.0e-6)
        resid = np.concatenate([resid, reg])
    return resid


def rel_rmse(y: np.ndarray, y_fit: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square((y_fit - y) / np.maximum(y, 1.0e-12)))))


def aic_bic(resid: np.ndarray, k: int) -> tuple[float, float]:
    n = max(int(resid.size), 1)
    rss = max(float(np.sum(np.square(resid))), 1.0e-18)
    aic = float(n * math.log(rss / n) + 2.0 * k)
    bic = float(n * math.log(rss / n) + k * math.log(n))
    return aic, bic


def fit_global(df: pd.DataFrame, theta_values: np.ndarray, t_osc: float, tc0: float, r0: float) -> dict:
    finf0 = estimate_finf_init(df, theta_values, t_osc)
    x0 = np.concatenate([[tc0, r0], finf0])
    lower = np.concatenate([[0.05, 0.1], np.full(len(theta_values), 1.0e-8)])
    upper = np.concatenate([[50.0, 50.0], np.full(len(theta_values), 1.0e3)])
    fun = lambda p: residual_vector(p, df, t_osc, theta_values, 1.0e-4, finf0)
    huber = least_squares(fun, x0, bounds=(lower, upper), loss="huber", f_scale=0.03, max_nfev=30000)
    final = least_squares(fun, huber.x, bounds=(lower, upper), loss="linear", max_nfev=30000)
    y_fit = model_eval(final.x, df, t_osc, theta_values)
    frac_resid = (y_fit - df["xi"].to_numpy(dtype=np.float64)) / np.maximum(df["xi"].to_numpy(dtype=np.float64), 1.0e-12)
    tc, r, finf = unpack_params(final.x, len(theta_values))
    aic, bic = aic_bic(frac_resid, len(final.x))
    payload = {
        "status": "ok" if final.success else "failed",
        "success": bool(final.success),
        "message": str(final.message),
        "t_c": float(tc),
        "r": float(r),
        "rel_rmse": rel_rmse(df["xi"].to_numpy(dtype=np.float64), y_fit),
        "AIC": aic,
        "BIC": bic,
        "n_points": int(len(df)),
        "n_params": int(len(final.x)),
        "theta_values": [float(v) for v in theta_values],
        "F0": {
            f"{theta:.10f}": float(
                df.loc[np.isclose(df["theta"], float(theta), atol=1.0e-12), "F0"].median()
            )
            for theta in theta_values
        },
        "F_inf": {f"{theta:.10f}": float(val) for theta, val in zip(theta_values, finf)},
        "per_H_rel_rmse": {},
    }
    for h in sorted(df["H"].unique()):
        mask = np.isclose(df["H"].to_numpy(dtype=np.float64), float(h), atol=1.0e-12)
        payload["per_H_rel_rmse"][f"{h:.1f}"] = rel_rmse(
            df.loc[mask, "xi"].to_numpy(dtype=np.float64),
            y_fit[mask],
        )
    return {
        "payload": payload,
        "result": final,
        "y_fit": y_fit,
        "frac_resid": frac_resid,
        "finf_ref": finf0,
    }


def bootstrap_fit(df: pd.DataFrame, theta_values: np.ndarray, fit_bundle: dict, t_osc: float, nboot: int, n_jobs: int) -> dict:
    if nboot <= 0:
        return {"status": "skipped", "n_samples": 0}
    x_best = fit_bundle["result"].x
    y_fit = fit_bundle["y_fit"]
    frac_resid = fit_bundle["frac_resid"]
    finf_ref = fit_bundle["finf_ref"]
    lower = np.concatenate([[0.05, 0.1], np.full(len(theta_values), 1.0e-8)])
    upper = np.concatenate([[50.0, 50.0], np.full(len(theta_values), 1.0e3)])

    def one_boot(seed: int):
        rng = np.random.default_rng(seed)
        sampled = rng.choice(frac_resid, size=len(frac_resid), replace=True)
        df_boot = df.copy()
        df_boot["xi"] = y_fit / np.maximum(1.0 + sampled, 0.05)
        fun = lambda p: residual_vector(p, df_boot, t_osc, theta_values, 1.0e-4, finf_ref)
        try:
            huber = least_squares(fun, x_best, bounds=(lower, upper), loss="huber", f_scale=0.03, max_nfev=15000)
            final = least_squares(fun, huber.x, bounds=(lower, upper), loss="linear", max_nfev=15000)
            if not final.success:
                return None
            return final.x
        except Exception:
            return None

    seeds = np.arange(nboot, dtype=np.int64) + 24680
    samples = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(one_boot)(int(seed)) for seed in seeds)
    samples = [s for s in samples if s is not None]
    if not samples:
        return {"status": "failed", "n_samples": 0}
    arr = np.asarray(samples, dtype=np.float64)
    out = {
        "status": "ok",
        "n_samples": int(len(arr)),
        "t_c": {q: float(np.percentile(arr[:, 0], p)) for q, p in [("p16", 16), ("p50", 50), ("p84", 84)]},
        "r": {q: float(np.percentile(arr[:, 1], p)) for q, p in [("p16", 16), ("p50", 50), ("p84", 84)]},
        "F_inf": {},
    }
    for i, theta in enumerate(theta_values):
        vals = arr[:, 2 + i]
        out["F_inf"][f"{theta:.10f}"] = {q: float(np.percentile(vals, p)) for q, p in [("p16", 16), ("p50", 50), ("p84", 84)]}
    return out


def fanh_model(tp: np.ndarray, f0: float, finf: float, t_osc: float, tc: float, r: float) -> np.ndarray:
    return finf / f0 + f0 / (xi_scale_from_tp(tp, t_osc) * (1.0 + np.power(np.maximum(tp, 1.0e-18) / max(tc, 1.0e-18), r)))


def plot_xi_by_H(df: pd.DataFrame, payload: dict, t_osc: float, outdir: Path) -> None:
    tp_plot_min = 1.0e-4
    for h in sorted(df["H"].unique()):
        sub_h = df[np.isclose(df["H"], float(h), atol=1.0e-12)].copy()
        theta_values = np.sort(sub_h["theta"].unique())
        tp_plot_max = float(sub_h["tp"].max()) * 1.05
        fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex=False, sharey=False)
        for ax, theta in zip(axes.flat, theta_values):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=1.0e-12)].sort_values("tp").copy()
            tp = sub["tp"].to_numpy(dtype=np.float64)
            xi = sub["xi"].to_numpy(dtype=np.float64)
            f0 = float(sub["F0"].iloc[0])
            finf = float(payload["F_inf"][f"{float(theta):.10f}"])
            tp_curve = np.geomspace(tp_plot_min, max(tp_plot_max, tp_plot_min * 10.0), 500)
            xi_curve = xi_scale_from_tp(tp_curve, t_osc) * finf / max(f0 * f0, 1.0e-18) + 1.0 / (
                1.0 + np.power(tp_curve / payload["t_c"], payload["r"])
            )
            ax.plot(tp, xi, "o", ms=3.2, color="tab:blue", label="ODE data")
            ax.plot(tp_curve, xi_curve, lw=2.0, color="black", label="ansatz fit")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(tp_plot_min, tp_plot_max)
            ax.axhline(1.0, color="tab:red", lw=1.1, ls="--", alpha=0.8, label=r"$\xi=1$")
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$t_p$")
            ax.set_ylabel(r"$\xi$")
        axes.flat[0].legend(frameon=False, fontsize=8)
        fig.suptitle(
            rf"Direct ODE ansatz fit, $H_*={h:.1f}$" + "\n" + rf"$t_c={payload['t_c']:.3f},\ r={payload['r']:.3f}$",
            y=0.98,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
        tag = f"H{h:.1f}".replace(".", "p")
        fig.savefig(outdir / f"xi_fit_overlay_{tag}.png", dpi=220)
        plt.close(fig)


def plot_fanh_by_H(df: pd.DataFrame, payload: dict, t_osc: float, outdir: Path) -> None:
    for h in sorted(df["H"].unique()):
        sub_h = df[np.isclose(df["H"], float(h), atol=1.0e-12)].copy()
        theta_values = np.sort(sub_h["theta"].unique())
        fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex=False, sharey=False)
        for ax, theta in zip(axes.flat, theta_values):
            sub = sub_h[np.isclose(sub_h["theta"], float(theta), atol=1.0e-12)].sort_values("tp").copy()
            tp = sub["tp"].to_numpy(dtype=np.float64)
            fanh = sub["fanh_model_data"].to_numpy(dtype=np.float64)
            f0 = float(sub["F0"].iloc[0])
            finf = float(payload["F_inf"][f"{float(theta):.10f}"])
            ax.plot(tp, fanh, "o", ms=3.2, color="tab:green", label="ODE data")
            ax.plot(tp, fanh_model(tp, f0, finf, t_osc, payload["t_c"], payload["r"]), lw=2.0, color="black", label="ansatz fit")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$t_p$")
            ax.set_ylabel(r"$f_{\rm anh}$")
        axes.flat[0].legend(frameon=False, fontsize=8)
        fig.suptitle(
            rf"Direct ODE fanh fit, $H_*={h:.1f}$" + "\n" + rf"$t_c={payload['t_c']:.3f},\ r={payload['r']:.3f}$",
            y=0.98,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
        tag = f"H{h:.1f}".replace(".", "p")
        fig.savefig(outdir / f"fanh_fit_overlay_{tag}.png", dpi=220)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        tables = [Path(p).resolve() if not Path(p).is_absolute() else Path(p) for p in args.tables]
        df, theta_values = load_data(tables, args.t_osc)
        fit_bundle = fit_global(df, theta_values, args.t_osc, args.tc0, args.r0)
        payload = fit_bundle["payload"]
        boot = bootstrap_fit(df, theta_values, fit_bundle, args.t_osc, args.nboot, args.n_jobs)
        save_json(outdir / "global_fit.json", payload)
        save_json(outdir / "bootstrap.json", boot)
        df_out = df.copy()
        df_out["xi_model"] = fit_bundle["y_fit"]
        df_out["frac_resid"] = fit_bundle["frac_resid"]
        df_out.to_csv(outdir / "predictions.csv", index=False)
        plot_xi_by_H(df_out, payload, args.t_osc, outdir)
        plot_fanh_by_H(df_out, payload, args.t_osc, outdir)
        summary = {
            "status": "ok",
            "data": {
                "H_values": [float(v) for v in sorted(df["H"].unique())],
                "theta_values": [float(v) for v in theta_values],
                "n_points": int(len(df)),
            },
            "fit": payload,
            "bootstrap": boot,
            "note": "Pure direct-ODE fit of xi(tp) = (2 tp / (3 t_osc))^(3/2) F_inf/F0^2 + 1/(1 + (tp/t_c)^r), with shared t_c,r and per-theta F_inf.",
        }
        save_json(outdir / "final_summary.json", summary)
        print(json.dumps(summary, sort_keys=True))
    except Exception as exc:
        error_exit(outdir, exc)


if __name__ == "__main__":
    main()
