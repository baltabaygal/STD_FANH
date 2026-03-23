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
from sklearn.linear_model import HuberRegressor


ROOT = Path(__file__).resolve().parent

# Override these with CLI flags if needed.
F0_CANDIDATES = [
    ROOT / "F0_table.csv",
    ROOT / "results_collapse" / "F0_table.csv",
    ROOT / "results_gamma" / "F0_table.csv",
    ROOT / "results_hstar" / "F0_table.csv",
    ROOT / "lattice_data" / "data" / "rho_noPT_data.txt",
    ROOT / "lattice_data" / "rho_noPT_data.txt",
]
FINF_CANDIDATES = [
    ROOT / "Finf_tail.csv",
    ROOT / "results_collapse" / "Finf_tail.csv",
    ROOT / "results_hstar" / "Finf_tail_H1p5_table.csv",
    ROOT / "results_hstar" / "Finf_tail_H1p5.csv",
    ROOT / "results_hstar" / "Finf_tail_H2p0_table.csv",
    ROOT / "results_hstar" / "Finf_tail_H2p0.csv",
]


def parse_args():
    p = argparse.ArgumentParser(description="Fit f(theta) ~ A h(theta)^gamma for F0 and F_inf.")
    p.add_argument("--f0", type=str, default="")
    p.add_argument("--finf", type=str, default="")
    p.add_argument("--nboot", type=int, default=500)
    p.add_argument("--outdir", type=str, default="results_hf")
    p.add_argument("--robust", dest="robust", action="store_true")
    p.add_argument("--no-robust", dest="robust", action="store_false")
    p.add_argument("--plot", dest="plot", action="store_true")
    p.add_argument("--no-plot", dest="plot", action="store_false")
    p.set_defaults(robust=True, plot=True)
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


def save_json(path: Path, payload):
    path.write_text(json.dumps(to_native(payload), indent=2, sort_keys=True))


def error_exit(outdir: Path, message: str):
    payload = {"status": "error", "message": message, "traceback": traceback.format_exc()}
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "_error.json", payload)
    print(json.dumps({"status": "error", "message": message}, sort_keys=True))
    return 1


def resolve_existing(path_value: str, candidates):
    if path_value:
        path = Path(path_value).resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"Missing requested file: {path}")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def potential(theta):
    theta = np.asarray(theta, dtype=np.float64)
    return 1.0 - np.cos(theta)


def rel_rmse(y, yfit):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square((yfit - y) / np.maximum(y, 1.0e-18)))))


def compute_h(theta):
    theta = np.asarray(theta, dtype=np.float64)
    close_to_pi = np.abs(theta - np.pi) < 1.0e-6
    denom = np.cos(theta / 2.0) ** 2
    h = np.full_like(theta, np.nan, dtype=np.float64)
    valid = (~close_to_pi) & np.isfinite(theta) & (denom > 0.0)
    h[valid] = np.log(np.e / denom[valid])
    return h, close_to_pi


def load_f0(path: Path):
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep=r"\s+", comment="#")
    cols = {c.lower(): c for c in df.columns}

    if {"theta", "f0"} <= set(cols):
        out = df.rename(columns={cols["theta"]: "theta", cols["f0"]: "f"})[["theta", "f"]].copy()
        return out.sort_values("theta").reset_index(drop=True), str(path)

    theta_col = cols.get("theta0", cols.get("theta"))
    h_col = cols.get("h_pt", cols.get("hstar", cols.get("h")))
    rho_col = cols.get("rho")
    if theta_col is None or h_col is None or rho_col is None:
        raise RuntimeError(
            f"Could not parse F0 input {path}. Need columns [theta,F0] or [theta0/theta, H_PT/H, rho]. Found {list(df.columns)}"
        )

    work = pd.DataFrame(
        {
            "theta": df[theta_col].astype(float),
            "H": df[h_col].astype(float),
            "rho": df[rho_col].astype(float),
        }
    )
    work = work[np.isfinite(work["theta"]) & np.isfinite(work["H"]) & np.isfinite(work["rho"])].copy()
    work["f_raw"] = work["rho"] / np.maximum(potential(work["theta"]) * np.power(work["H"], 1.5), 1.0e-18)
    out = work.groupby("theta", as_index=False)["f_raw"].median().rename(columns={"f_raw": "f"})
    return out.sort_values("theta").reset_index(drop=True), str(path)


def load_finf(path: Path):
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep=r"\s+", comment="#")
    cols = {c.lower(): c for c in df.columns}

    theta_col = cols.get("theta0", cols.get("theta"))
    finf_col = cols.get("f_inf_tail", cols.get("finf_tail", cols.get("f_inf", cols.get("finf"))))
    if theta_col is None or finf_col is None:
        raise RuntimeError(
            f"Could not parse Finf input {path}. Need columns like [theta,F_inf_tail]. Found {list(df.columns)}"
        )

    out = pd.DataFrame({"theta": df[theta_col].astype(float), "f": df[finf_col].astype(float)})
    out = out[np.isfinite(out["theta"]) & np.isfinite(out["f"])].copy()
    return out.sort_values("theta").reset_index(drop=True), str(path)


def prepare_dataset(df: pd.DataFrame, name: str):
    work = df.copy()
    h, close_to_pi = compute_h(work["theta"].to_numpy(dtype=np.float64))
    work["h"] = h
    work["drop_pi"] = close_to_pi

    n_pi = int(np.sum(close_to_pi))
    if n_pi:
        print(f"[warn] {name}: dropped {n_pi} point(s) with theta too close to pi")

    before = len(work)
    work = work[~work["drop_pi"]].copy()
    work = work[np.isfinite(work["h"]) & (work["h"] > 0.0) & np.isfinite(work["f"]) & (work["f"] > 0.0)].copy()
    dropped = before - len(work)

    work["x"] = np.log(work["h"].to_numpy(dtype=np.float64))
    work["y"] = np.log(work["f"].to_numpy(dtype=np.float64))
    work = work[np.isfinite(work["x"]) & np.isfinite(work["y"])].copy()
    dropped = before - len(work)
    if len(work) < 2:
        raise RuntimeError(f"{name}: not enough valid points after filtering")

    return work.sort_values("theta").reset_index(drop=True), dropped


def fit_huber(x, y):
    model = HuberRegressor(alpha=0.0, fit_intercept=True, max_iter=1000)
    model.fit(x.reshape(-1, 1), y)
    gamma = float(model.coef_[0])
    lnA = float(model.intercept_)
    y_pred = model.predict(x.reshape(-1, 1))
    return {"gamma": gamma, "lnA": lnA, "y_pred": y_pred, "model": model}


def fit_ols(x, y):
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    lnA = float(beta[0])
    gamma = float(beta[1])
    y_pred = X @ beta
    return {"gamma": gamma, "lnA": lnA, "y_pred": y_pred}


def r2_score(y, y_pred):
    y = np.asarray(y, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum(np.square(y - y_pred)))
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def summarize_fit(df: pd.DataFrame, fit_result, method: str):
    y = df["y"].to_numpy(dtype=np.float64)
    y_pred = np.asarray(fit_result["y_pred"], dtype=np.float64)
    f = df["f"].to_numpy(dtype=np.float64)
    f_pred = np.exp(y_pred)
    return {
        "method": method,
        "A": float(np.exp(fit_result["lnA"])),
        "gamma": float(fit_result["gamma"]),
        "lnA": float(fit_result["lnA"]),
        "R2": float(r2_score(y, y_pred)),
        "rmse_log": float(np.sqrt(np.mean(np.square(y - y_pred)))),
        "rel_rmse": float(rel_rmse(f, f_pred)),
    }


def bootstrap_one(seed, x, y, robust):
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=n)
    xb = x[idx]
    yb = y[idx]
    try:
        if robust:
            fit = fit_huber(xb, yb)
        else:
            fit = fit_ols(xb, yb)
        return fit["gamma"], fit["lnA"]
    except Exception:
        return None


def run_bootstrap(df: pd.DataFrame, robust: bool, nboot: int):
    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    seeds = np.arange(nboot, dtype=np.int64) + 12345
    results = Parallel(n_jobs=-1)(delayed(bootstrap_one)(int(seed), x, y, robust) for seed in seeds)
    results = [r for r in results if r is not None and np.all(np.isfinite(r))]
    if not results:
        return {"status": "no_successful_samples"}, np.empty((0, 2), dtype=np.float64)
    arr = np.asarray(results, dtype=np.float64)
    gamma = arr[:, 0]
    lnA = arr[:, 1]
    A = np.exp(lnA)
    summary = {
        "status": "ok",
        "n_samples": int(len(arr)),
        "gamma_median": float(np.percentile(gamma, 50)),
        "gamma_p16": float(np.percentile(gamma, 16)),
        "gamma_p84": float(np.percentile(gamma, 84)),
        "gamma_p2p5": float(np.percentile(gamma, 2.5)),
        "gamma_p97p5": float(np.percentile(gamma, 97.5)),
        "A_median": float(np.percentile(A, 50)),
        "A_p16": float(np.percentile(A, 16)),
        "A_p84": float(np.percentile(A, 84)),
        "A_p2p5": float(np.percentile(A, 2.5)),
        "A_p97p5": float(np.percentile(A, 97.5)),
        "lnA_median": float(np.percentile(lnA, 50)),
    }
    return summary, arr


def influence_check(df: pd.DataFrame, robust: bool):
    if len(df) < 3:
        return None
    base = fit_huber(df["x"].to_numpy(), df["y"].to_numpy()) if robust else fit_ols(df["x"].to_numpy(), df["y"].to_numpy())
    idx = int(np.argmax(df["h"].to_numpy(dtype=np.float64)))
    reduced = df.drop(df.index[idx]).reset_index(drop=True)
    alt = fit_huber(reduced["x"].to_numpy(), reduced["y"].to_numpy()) if robust else fit_ols(reduced["x"].to_numpy(), reduced["y"].to_numpy())
    gamma0 = float(base["gamma"])
    gamma1 = float(alt["gamma"])
    frac = abs(gamma1 - gamma0) / max(abs(gamma0), 1.0e-12)
    return {
        "dominant_theta": float(df.iloc[idx]["theta"]),
        "gamma_full": gamma0,
        "gamma_drop_maxh": gamma1,
        "relative_shift": float(frac),
        "warn": bool(frac > 0.10),
        "reduced_df": reduced,
        "reduced_fit": alt,
    }


def fit_dataset(df: pd.DataFrame, name: str, robust: bool, nboot: int, outdir: Path):
    fit_main = fit_huber(df["x"].to_numpy(), df["y"].to_numpy()) if robust else fit_ols(df["x"].to_numpy(), df["y"].to_numpy())
    fit_ols_result = fit_ols(df["x"].to_numpy(), df["y"].to_numpy())
    summary_main = summarize_fit(df, fit_main, "Huber" if robust else "OLS")
    summary_ols = summarize_fit(df, fit_ols_result, "OLS")
    boot_summary, boot_arr = run_bootstrap(df, robust=robust, nboot=nboot)
    influence = influence_check(df, robust=robust)
    alt_payload = None
    if influence and influence["warn"]:
        alt_summary = summarize_fit(influence["reduced_df"], influence["reduced_fit"], "Huber_exclude_maxh" if robust else "OLS_exclude_maxh")
        alt_payload = {
            "dataset": name,
            "warning": "Removing the largest-h point changes gamma by more than 10%",
            "dominant_theta": influence["dominant_theta"],
            "gamma_full": influence["gamma_full"],
            "gamma_drop_maxh": influence["gamma_drop_maxh"],
            "relative_shift": influence["relative_shift"],
            "alternate_fit": alt_summary,
        }
        save_json(outdir / f"robust_exclude_pi_{name}.json", alt_payload)
        print(f"[warn] {name}: largest-h point shifts gamma by {100.0 * influence['relative_shift']:.1f}%")

    return {
        "fit": fit_main,
        "fit_ols": fit_ols_result,
        "summary": summary_main,
        "summary_ols": summary_ols,
        "bootstrap": boot_summary,
        "bootstrap_samples": boot_arr,
        "influence": influence,
        "alternate": alt_payload,
    }


def plot_fit(df: pd.DataFrame, result: dict, name: str, outdir: Path):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    ax.scatter(x, y, color="tab:blue", s=40, zorder=3)
    for _, row in df.iterrows():
        ax.annotate(f"{row['theta']:.3f}", (row["x"], row["y"]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    grid = np.linspace(np.min(x) - 0.05, np.max(x) + 0.05, 300)
    y_huber = result["fit"]["lnA"] + result["fit"]["gamma"] * grid
    y_ols = result["fit_ols"]["lnA"] + result["fit_ols"]["gamma"] * grid
    ax.plot(grid, y_huber, color="tab:red", lw=2.0, label="Huber")
    ax.plot(grid, y_ols, color="black", lw=1.5, ls="--", label="OLS")

    boot_arr = result["bootstrap_samples"]
    if boot_arr.size:
        gamma = boot_arr[:, 0]
        lnA = boot_arr[:, 1]
        band = lnA[:, None] + gamma[:, None] * grid[None, :]
        lo = np.percentile(band, 16, axis=0)
        hi = np.percentile(band, 84, axis=0)
        ax.fill_between(grid, lo, hi, color="tab:red", alpha=0.18, label="bootstrap 68%")

    ax.set_xlabel(r"$\ln h(\theta)$")
    ax.set_ylabel(r"$\ln f$")
    ax.set_title(name)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / f"lnf_vs_lnh_{name}.png", dpi=220)
    plt.close(fig)


def plot_residuals(df: pd.DataFrame, result: dict, name: str, outdir: Path):
    y = df["y"].to_numpy(dtype=np.float64)
    y_pred = np.asarray(result["fit"]["y_pred"], dtype=np.float64)
    resid = y - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].scatter(df["x"], resid, color="tab:blue", s=40)
    axes[0].axhline(0.0, color="black", lw=1.0)
    axes[0].set_xlabel(r"$\ln h(\theta)$")
    axes[0].set_ylabel(r"$y - y_{\rm pred}$")
    axes[0].set_title(f"{name}: residual vs ln h")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(df["theta"], resid, color="tab:blue", s=40)
    axes[1].axhline(0.0, color="black", lw=1.0)
    axes[1].set_xlabel(r"$\theta$")
    axes[1].set_ylabel(r"$y - y_{\rm pred}$")
    axes[1].set_title(f"{name}: residual vs theta")
    axes[1].grid(alpha=0.25)

    rms = math.sqrt(float(np.mean(np.square(resid))))
    fig.suptitle(f"{name} residual RMS = {rms:.4e}")
    fig.tight_layout()
    fig.savefig(outdir / f"residuals_{name}.png", dpi=220)
    plt.close(fig)


def plot_bootstrap_hist(results: dict, outdir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    names = ["noPT", "inf"]
    for i, name in enumerate(names):
        boot_arr = results[name]["bootstrap_samples"]
        gamma = boot_arr[:, 0]
        A = np.exp(boot_arr[:, 1])
        ax_g = axes[i, 0]
        ax_a = axes[i, 1]
        ax_g.hist(gamma, bins=30, color="tab:blue", alpha=0.70)
        ax_a.hist(A, bins=30, color="tab:orange", alpha=0.70)
        for ax, arr, label in [(ax_g, gamma, r"$\gamma$"), (ax_a, A, r"$A$")]:
            if len(arr):
                p16, p50, p84 = np.percentile(arr, [16, 50, 84])
                ax.axvline(p16, color="black", ls="--", lw=1.0)
                ax.axvline(p50, color="black", lw=1.4)
                ax.axvline(p84, color="black", ls="--", lw=1.0)
            ax.set_xlabel(label)
            ax.grid(alpha=0.20)
        ax_g.set_title(f"{name}: gamma bootstrap")
        ax_a.set_title(f"{name}: A bootstrap")
    fig.tight_layout()
    fig.savefig(outdir / "A_gamma_bootstrap_hist.png", dpi=220)
    plt.close(fig)


def write_fit_table(summary_noPT, summary_inf, boot_noPT, boot_inf, outdir: Path):
    rows = []
    for name, summary, boot in [("noPT", summary_noPT, boot_noPT), ("Finf", summary_inf, boot_inf)]:
        rows.append(
            {
                "dataset": name,
                "A": summary["A"],
                "gamma": summary["gamma"],
                "lnA": summary["lnA"],
                "R2": summary["R2"],
                "rmse_log": summary["rmse_log"],
                "rel_rmse": summary["rel_rmse"],
                "gamma_median": boot.get("gamma_median", np.nan),
                "gamma_p16": boot.get("gamma_p16", np.nan),
                "gamma_p84": boot.get("gamma_p84", np.nan),
                "gamma_p2p5": boot.get("gamma_p2p5", np.nan),
                "gamma_p97p5": boot.get("gamma_p97p5", np.nan),
                "A_median": boot.get("A_median", np.nan),
                "A_p16": boot.get("A_p16", np.nan),
                "A_p84": boot.get("A_p84", np.nan),
                "A_p2p5": boot.get("A_p2p5", np.nan),
                "A_p97p5": boot.get("A_p97p5", np.nan),
            }
        )
    pd.DataFrame(rows).to_csv(outdir / "fit_table.csv", index=False)


def main():
    args = parse_args()
    outdir = (ROOT / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        f0_path = resolve_existing(args.f0, F0_CANDIDATES)
        finf_path = resolve_existing(args.finf, FINF_CANDIDATES)
        if f0_path is None:
            raise FileNotFoundError("Missing F0 input. Supply --f0 rho_noPT_data.txt or a F0_table.csv.")
        if finf_path is None:
            raise FileNotFoundError("Missing Finf input. Supply --finf Finf_tail.csv or a Finf_tail_H*.csv file.")

        df_f0_raw, f0_source = load_f0(f0_path)
        df_finf_raw, finf_source = load_finf(finf_path)

        df_f0, dropped_f0 = prepare_dataset(df_f0_raw, "noPT")
        df_finf, dropped_finf = prepare_dataset(df_finf_raw, "inf")

        result_f0 = fit_dataset(df_f0, "noPT", args.robust, args.nboot, outdir)
        result_finf = fit_dataset(df_finf, "inf", args.robust, args.nboot, outdir)

        if args.plot:
            plot_fit(df_f0, result_f0, "noPT", outdir)
            plot_fit(df_finf, result_finf, "inf", outdir)
            plot_residuals(df_f0, result_f0, "noPT", outdir)
            plot_residuals(df_finf, result_finf, "inf", outdir)
            plot_bootstrap_hist({"noPT": result_f0, "inf": result_finf}, outdir)

        write_fit_table(
            result_f0["summary"],
            result_finf["summary"],
            result_f0["bootstrap"],
            result_finf["bootstrap"],
            outdir,
        )

        final_summary = {
            "status": "ok",
            "file_sources": {"F0": f0_source, "Finf": finf_source},
            "noPT": {
                "best_fit": result_f0["summary"],
                "ols_fit": result_f0["summary_ols"],
                "bootstrap": result_f0["bootstrap"],
                "n_points_used": int(len(df_f0)),
                "n_points_dropped": int(dropped_f0),
            },
            "Finf": {
                "best_fit": result_finf["summary"],
                "ols_fit": result_finf["summary_ols"],
                "bootstrap": result_finf["bootstrap"],
                "n_points_used": int(len(df_finf)),
                "n_points_dropped": int(dropped_finf),
            },
        }
        if result_f0["alternate"] is not None:
            final_summary["noPT"]["exclude_maxh_warning"] = result_f0["alternate"]
        if result_finf["alternate"] is not None:
            final_summary["Finf"]["exclude_maxh_warning"] = result_finf["alternate"]

        save_json(outdir / "final_summary.json", final_summary)
        print(json.dumps(to_native(final_summary), sort_keys=True))
        return 0
    except Exception as exc:
        return error_exit(outdir, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
