#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import least_squares

import collapse_and_fit_fanh_tosc as cf


ROOT = Path(__file__).resolve().parent
VW_MAP = {"v3": 0.3, "v5": 0.5, "v7": 0.7, "v9": 0.9}
RAW_TEMPLATE = ROOT / "lattice_data" / "data" / "energy_ratio_by_theta_data_{tag}.txt"
TARGET_H = [1.5, 2.0]
FIXED_TC = 1.5
FIXED_BETA = 0.0
BOOTSTRAP_N = 80
BOOTSTRAP_JOBS = 6
BOOTSTRAP_SEED = 4210
DPI = 220


def compute_x_tp(df: pd.DataFrame):
    out = df.copy()
    out["x"] = out["tp"].to_numpy(dtype=np.float64)
    return out


def xi_model_fixed_tc(df: pd.DataFrame, theta_values: np.ndarray, params_reduced: np.ndarray, fixed_tc: float):
    r = float(params_reduced[0])
    finf = np.asarray(params_reduced[1:], dtype=np.float64)
    theta_index = np.array([cf.nearest_theta(theta_values, th) for th in df["theta"].to_numpy(dtype=np.float64)], dtype=np.int64)
    x = df["x"].to_numpy(dtype=np.float64)
    f0 = df["F0"].to_numpy(dtype=np.float64)
    transient = 1.0 / (1.0 + np.power(x / max(fixed_tc, 1.0e-12), r))
    xi_fit = np.power(x / cf.T_OSC, 1.5) * finf[theta_index] / np.maximum(f0 * f0, 1.0e-18) + transient
    return xi_fit, theta_index


def fit_global_fixed_tc(df: pd.DataFrame, finf_tail_df: pd.DataFrame, fixed_tc: float):
    theta_values = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    tail_map = {float(row.theta): float(row.F_inf_tail) for row in finf_tail_df.itertuples(index=False)}
    finf0 = np.array([max(tail_map.get(float(th), np.nanmedian(finf_tail_df["F_inf_tail"])), 1.0e-8) for th in theta_values], dtype=np.float64)
    x0 = np.concatenate([np.array([1.5], dtype=np.float64), finf0])
    lower = np.concatenate([np.array([0.1], dtype=np.float64), np.full(len(theta_values), 1.0e-6, dtype=np.float64)])
    upper = np.concatenate([np.array([20.0], dtype=np.float64), np.full(len(theta_values), 1.0e4, dtype=np.float64)])
    xi_data = df["xi"].to_numpy(dtype=np.float64)

    def resid(par):
        xi_fit, _ = xi_model_fixed_tc(df, theta_values, par, fixed_tc)
        return (xi_fit - xi_data) / np.maximum(xi_data, 1.0e-12)

    res0 = least_squares(resid, x0=x0, bounds=(lower, upper), loss="soft_l1", f_scale=0.05, max_nfev=6000)
    res = least_squares(resid, x0=res0.x, bounds=(lower, upper), loss="linear", max_nfev=6000)
    xi_fit, theta_index = xi_model_fixed_tc(df, theta_values, res.x, fixed_tc)
    rss = float(np.sum(np.square((xi_fit - xi_data) / np.maximum(xi_data, 1.0e-12))))
    n = int(len(df))
    k = int(len(res.x))
    dof = max(n - k, 1)
    cov_red = np.full((len(res.x), len(res.x)), np.nan, dtype=np.float64)
    try:
        cov_red = (rss / dof) * np.linalg.inv(res.jac.T @ res.jac)
    except np.linalg.LinAlgError:
        pass
    cov_full = np.full((2 + len(theta_values), 2 + len(theta_values)), np.nan, dtype=np.float64)
    cov_full[1:, 1:] = cov_red
    params_full = np.concatenate([[fixed_tc, float(res.x[0])], np.asarray(res.x[1:], dtype=np.float64)])
    aic = float(n * math.log(max(rss, 1.0e-18) / n) + 2.0 * (2 + len(theta_values)))
    bic = float(n * math.log(max(rss, 1.0e-18) / n) + (2 + len(theta_values)) * math.log(n))
    return {
        "success": bool(res.success),
        "message": res.message,
        "params": params_full,
        "theta_values": theta_values,
        "theta_index": theta_index,
        "xi_fit": xi_fit,
        "covariance": cov_full,
        "dof": dof,
        "rel_rmse": cf.rel_rmse(xi_data, xi_fit),
        "rss_frac": rss,
        "AIC": aic,
        "BIC": bic,
    }


def save_global_fit_fixed_tc(result: dict, outdir: Path):
    params = np.asarray(result["params"], dtype=np.float64)
    cov = np.asarray(result["covariance"], dtype=np.float64)
    r_err = float(np.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) else np.nan
    f_err = [float(np.sqrt(cov[2 + i, 2 + i])) if np.isfinite(cov[2 + i, 2 + i]) else np.nan for i in range(len(result["theta_values"]))]
    payload = {
        "success": result["success"],
        "message": result["message"],
        "beta": FIXED_BETA,
        "t_c": float(FIXED_TC),
        "t_c_fixed": True,
        "t_c_err": 0.0,
        "r": float(params[1]),
        "r_err": r_err,
        "F_inf": {
            f"{float(th):.10f}": {"value": float(params[2 + i]), "err": f_err[i]}
            for i, th in enumerate(result["theta_values"])
        },
        "dof": int(result["dof"]),
        "rel_rmse": float(result["rel_rmse"]),
        "rss_frac": float(result["rss_frac"]),
        "AIC": float(result["AIC"]),
        "BIC": float(result["BIC"]),
        "covariance": result["covariance"].tolist(),
    }
    cf.save_json(outdir / "global_fit.json", payload)
    return payload


def bootstrap_fixed_tc(df: pd.DataFrame, fit_result: dict, nboot: int, n_jobs: int, seed0: int):
    if nboot <= 0:
        return {
            "t_c": {"p16": FIXED_TC, "p50": FIXED_TC, "p84": FIXED_TC},
            "r": {"p16": np.nan, "p50": np.nan, "p84": np.nan},
            "F_inf": {},
        }
    theta_values = fit_result["theta_values"]
    xi_fit = fit_result["xi_fit"]
    resid = (df["xi"].to_numpy(dtype=np.float64) - xi_fit) / np.maximum(xi_fit, 1.0e-12)
    theta_index = fit_result["theta_index"]
    finf_seed = np.asarray(fit_result["params"][2:], dtype=np.float64)

    def worker(seed):
        rng = np.random.default_rng(seed)
        boot = df.copy()
        xi_boot = xi_fit.copy()
        for i, _ in enumerate(theta_values):
            mask = theta_index == i
            if np.sum(mask) == 0:
                continue
            sampled = rng.choice(resid[mask], size=np.sum(mask), replace=True)
            xi_boot[mask] = np.maximum(xi_fit[mask] * (1.0 + sampled), 1.0e-10)
        boot["xi"] = xi_boot
        rec = fit_global_fixed_tc(boot, pd.DataFrame({"theta": theta_values, "F_inf_tail": finf_seed}), FIXED_TC)
        return rec["params"]

    seeds = [seed0 + i for i in range(nboot)]
    arr = np.asarray(Parallel(n_jobs=n_jobs)(delayed(worker)(seed) for seed in seeds), dtype=np.float64)
    payload = {
        "t_c": {"p16": FIXED_TC, "p50": FIXED_TC, "p84": FIXED_TC},
        "r": {
            "p16": float(np.percentile(arr[:, 1], 16.0)),
            "p50": float(np.percentile(arr[:, 1], 50.0)),
            "p84": float(np.percentile(arr[:, 1], 84.0)),
        },
        "F_inf": {},
    }
    for i, th in enumerate(theta_values):
        payload["F_inf"][f"{float(th):.10f}"] = {
            "p16": float(np.percentile(arr[:, 2 + i], 16.0)),
            "p50": float(np.percentile(arr[:, 2 + i], 50.0)),
            "p84": float(np.percentile(arr[:, 2 + i], 84.0)),
        }
    return payload


def fit_single_vw(tag: str, vw: float, compare_rows: list[dict]):
    raw_path = RAW_TEMPLATE.with_name(RAW_TEMPLATE.name.format(tag=tag))
    outdir = ROOT / f"results_collapse_fixed_tc_tp_{tag}_H15H20_tosc"
    outdir.mkdir(parents=True, exist_ok=True)
    rho_path = cf.resolve_first_existing(cf.RHO_CANDIDATES, "")
    if rho_path is None:
        raise FileNotFoundError("Missing rho_noPT_data.txt.")
    f0_table = cf.load_f0_table(rho_path, TARGET_H)
    f0_table.to_csv(outdir / "F0_table.csv", index=False)
    lattice_df = cf.load_lattice_data(raw_path, vw, TARGET_H)
    lattice_df = cf.merge_f0(lattice_df, f0_table)
    lattice_df = lattice_df[np.isfinite(lattice_df["F0"]) & (lattice_df["F0"] > 0.0)].copy()

    beta_payload = {
        "beta": FIXED_BETA,
        "collapse_score": float(cf.collapse_score(lattice_df, FIXED_BETA, 120)),
        "beta_scan": [FIXED_BETA],
        "score_scan": [float(cf.collapse_score(lattice_df, FIXED_BETA, 120))],
        "refined_interval": [FIXED_BETA, FIXED_BETA],
        "refined_success": True,
    }
    cf.save_json(outdir / "best_beta.json", beta_payload)
    collapsed = compute_x_tp(lattice_df)
    collapsed["fanh_data"] = collapsed["xi"].to_numpy(dtype=np.float64) * collapsed["F0"].to_numpy(dtype=np.float64) / np.power(
        collapsed["tp"].to_numpy(dtype=np.float64) / cf.T_OSC, 1.5
    )
    finf_tail_df = cf.fit_tail(collapsed, outdir, DPI)
    inversion_rows = []
    for theta, group in collapsed.groupby("theta", sort=True):
        finf_tail = float(finf_tail_df.loc[np.isclose(finf_tail_df["theta"], theta, atol=5.0e-4), "F_inf_tail"].iloc[0])
        inversion_rows.append(cf.inversion_single_theta(group, finf_tail))
    inversion_global = cf.pooled_inversion(inversion_rows)
    cf.plot_inversion(inversion_rows, outdir, DPI)

    fit_result = fit_global_fixed_tc(collapsed, finf_tail_df, FIXED_TC)
    global_payload = save_global_fit_fixed_tc(fit_result, outdir)
    bootstrap_payload = bootstrap_fixed_tc(collapsed, fit_result, BOOTSTRAP_N, BOOTSTRAP_JOBS, BOOTSTRAP_SEED + int(vw * 1000))
    cf.save_json(outdir / "bootstrap_global_fit.json", bootstrap_payload)
    cf.plot_collapse_overlay(collapsed, fit_result, outdir, DPI)
    cf.plot_residual_heatmap(collapsed, fit_result, outdir, DPI)
    cf.plot_fanh_theta(collapsed, fit_result, outdir, DPI)
    for theta in sorted(collapsed["theta"].unique()):
        cf.slope_curvature(float(theta), collapsed, outdir, DPI)

    summary = {
        "status": "ok",
        "beta": FIXED_BETA,
        "collapse_score": beta_payload["collapse_score"],
        "available_H": sorted(float(h) for h in collapsed["H"].unique()),
        "fixed_vw": float(vw),
        "n_lattice_points": int(len(collapsed)),
        "ode_loaded": False,
        "global_fit": {
            "t_c": float(FIXED_TC),
            "r": global_payload["r"],
            "rel_rmse": global_payload["rel_rmse"],
            "AIC": global_payload["AIC"],
            "BIC": global_payload["BIC"],
        },
        "bootstrap_68": {
            "t_c": [FIXED_TC, FIXED_TC],
            "r": [bootstrap_payload["r"]["p16"], bootstrap_payload["r"]["p84"]],
        },
        "inversion_global": inversion_global,
    }
    cf.save_json(outdir / "final_summary.json", summary)
    compare_rows.append(
        {
            "tag": tag,
            "vw": vw,
            "beta": FIXED_BETA,
            "collapse_score": beta_payload["collapse_score"],
            "t_c": FIXED_TC,
            "r": global_payload["r"],
            "rel_rmse": global_payload["rel_rmse"],
            "outdir": str(outdir),
        }
    )


def make_compare_outputs(compare_rows: list[dict]):
    outdir = ROOT / "results_collapse_fixed_tc_tp_vw_compare_tosc"
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(compare_rows).sort_values("vw").reset_index(drop=True)
    df.to_csv(outdir / "vw_collapse_fixed_tc_tp_summary.csv", index=False)
    cf.save_json(outdir / "vw_collapse_fixed_tc_tp_summary.json", df.to_dict(orient="records"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, row in zip(axes.flat, df.itertuples(index=False)):
        img = plt.imread(Path(row.outdir) / "collapse_overlay.png")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(rf"$v_w={row.vw:.1f}$, $x=t_p$, $t_c=1.5$, $r={row.r:.3f}$" + "\n" + rf"rel-RMSE={row.rel_rmse:.4e}")
    fig.tight_layout()
    fig.savefig(outdir / "collapse_overlay_fixed_tc_tp_by_vw.png", dpi=DPI)
    plt.close(fig)


def main():
    compare_rows: list[dict] = []
    for tag, vw in VW_MAP.items():
        fit_single_vw(tag, vw, compare_rows)
    make_compare_outputs(compare_rows)
    summary = {
        "status": "ok",
        "fixed_t_c": FIXED_TC,
        "fixed_beta": FIXED_BETA,
        "t_osc": cf.T_OSC,
        "normalization": "xi = (tp/t_osc)^(3/2) * F_inf/F0^2 + 1/(1 + (tp/t_c)^r), with x = tp",
        "results": sorted(compare_rows, key=lambda row: row["vw"]),
        "compare_dir": str(ROOT / "results_collapse_fixed_tc_tp_vw_compare_tosc"),
    }
    cf.save_json(ROOT / "results_collapse_fixed_tc_tp_vw_compare_tosc" / "final_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
