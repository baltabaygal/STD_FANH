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

import collapse_and_fit_fanh_tosc as latmod


ROOT = Path(__file__).resolve().parent
T_OSC = latmod.T_OSC
ODE_FILES = {
    0.5: ROOT / "ode" / "analysis" / "data" / "dm_tp_fitready_H0p500.txt",
    1.0: ROOT / "ode" / "analysis" / "data" / "dm_tp_fitready_H1p000.txt",
    1.5: ROOT / "ode" / "analysis" / "data" / "dm_tp_fitready_H1p500.txt",
    2.0: ROOT / "ode" / "analysis" / "data" / "dm_tp_fitready_H2p000.txt",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract ODE-anchored F_inf(theta) from long ODE tails and refit the clean vw=0.9 lattice collapse with F_inf fixed."
    )
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--lattice-h-values", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    p.add_argument("--ode-h-values", type=float, nargs="+", default=[1.0, 2.0])
    p.add_argument("--tail-frac", type=float, default=0.4, help="Fraction of each ODE tp range used for the asymptotic fit.")
    p.add_argument("--grid-n", type=int, default=160)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--outdir", type=str, default="results_ode_anchored_finf_vw0p9_H1p0H1p5H2p0")
    return p.parse_args()


def load_direct_ode_table(h: float) -> pd.DataFrame:
    path = ODE_FILES[float(h)]
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    raw_cols = [
        "#",
        "H_star",
        "t_star",
        "theta0",
        "t_p",
        "x_tp_over_tosc",
        "Ea3_PT",
        "Ea3_noPT",
        "f_anh_PT",
        "f_anh_noPT",
        "xi_DM",
        "nsteps_PT",
        "nsteps_noPT",
    ]
    true_cols = raw_cols[1:]
    df = df[raw_cols[:-1]].copy()
    df.columns = true_cols
    for col in true_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.rename(columns={"theta0": "theta", "t_p": "tp", "f_anh_noPT": "F0_ode", "xi_DM": "xi"})
    df["H"] = float(h)
    df["Finf_est"] = df["xi"].to_numpy(dtype=np.float64) * np.square(df["F0_ode"].to_numpy(dtype=np.float64)) / np.power(
        df["tp"].to_numpy(dtype=np.float64) / T_OSC, 1.5
    )
    keep = ["H", "theta", "tp", "F0_ode", "xi", "Finf_est"]
    return df[keep].copy()


def fit_tail_model(tp: np.ndarray, y: np.ndarray) -> dict:
    tp = np.asarray(tp, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def model(par: np.ndarray) -> np.ndarray:
        finf, amp, alpha = par
        return finf + amp / np.power(tp, alpha)

    def resid(par: np.ndarray) -> np.ndarray:
        return (model(par) - y) / np.maximum(y, 1.0e-18)

    finf0 = float(np.min(y))
    alpha0 = 1.0
    amp0 = max(float(np.max(y) - finf0), 1.0e-12) * float(np.min(tp) ** alpha0)
    lower = np.array([1.0e-12, 0.0, 1.0e-3], dtype=np.float64)
    upper = np.array([1.0e6, 1.0e6, 10.0], dtype=np.float64)
    x0 = np.array([max(finf0, 1.0e-12), amp0, alpha0], dtype=np.float64)
    res0 = least_squares(resid, x0=x0, bounds=(lower, upper), loss="soft_l1", f_scale=0.02, max_nfev=8000)
    res = least_squares(resid, x0=res0.x, bounds=(lower, upper), loss="linear", max_nfev=8000)
    yfit = model(res.x)
    rel_rmse = float(np.sqrt(np.mean(np.square((yfit - y) / np.maximum(y, 1.0e-18)))))
    return {
        "params": res.x,
        "success": bool(res.success),
        "message": res.message,
        "rel_rmse": rel_rmse,
        "yfit": yfit,
    }


def extract_ode_anchors(df_ode: pd.DataFrame, ode_h_values: list[float], tail_frac: float, outdir: Path, dpi: int) -> pd.DataFrame:
    theta_values = np.array(sorted(df_ode["theta"].unique()), dtype=np.float64)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ode_h_values)))
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.3), sharex=False, sharey=False)
    rows = []
    for ax, theta in zip(axes.flat, theta_values):
        sub = df_ode[np.isclose(df_ode["theta"], theta, atol=5.0e-4, rtol=0.0)].copy().sort_values("tp")
        per_h = []
        for color, h in zip(colors, ode_h_values):
            hsub = sub[np.isclose(sub["H"], h, atol=1.0e-12, rtol=0.0)].copy().sort_values("tp")
            ntail = max(8, int(math.ceil(tail_frac * len(hsub))))
            tail = hsub.tail(ntail).copy()
            rec = fit_tail_model(tail["tp"].to_numpy(dtype=np.float64), tail["Finf_est"].to_numpy(dtype=np.float64))
            finf, amp, alpha = [float(v) for v in rec["params"]]
            xfit = np.geomspace(float(np.min(tail["tp"])), float(np.max(tail["tp"])), 200)
            yfit = finf + amp / np.power(xfit, alpha)
            ax.plot(hsub["tp"], hsub["Finf_est"], "o", ms=2.5, color=color, alpha=0.75, label=rf"$H={h:g}$" if theta == theta_values[0] else None)
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
        finf_vals = np.array([row["F_inf_fit"] for row in per_h], dtype=np.float64)
        anchor = float(np.mean(finf_vals))
        err = float(0.5 * (np.max(finf_vals) - np.min(finf_vals)))
        ax.axhline(anchor, color="black", ls="--", lw=1.5, label="ODE anchor" if theta == theta_values[0] else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$\xi_{\rm ODE} F_0^2 / (t_p/T_{\rm osc})^{3/2}$")
        ax.set_title(rf"$\theta={theta:.3f}$, $F_\infty^{{\rm ODE}}={anchor:.3e}$")
        ax.grid(alpha=0.25)
        rows.append(
            {
                "theta": float(theta),
                "F_inf_ode_anchor": anchor,
                "F_inf_ode_err": err,
                "per_H": per_h,
            }
        )
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "ode_anchor_tail_fits.png", dpi=dpi)
    plt.close(fig)

    flat_rows = []
    for row in rows:
        payload = {
            "theta": row["theta"],
            "F_inf_ode_anchor": row["F_inf_ode_anchor"],
            "F_inf_ode_err": row["F_inf_ode_err"],
        }
        for hrec in row["per_H"]:
            tag = f"H{hrec['H']:g}".replace(".", "p")
            payload[f"F_inf_{tag}"] = hrec["F_inf_fit"]
            payload[f"A_{tag}"] = hrec["A_fit"]
            payload[f"alpha_{tag}"] = hrec["alpha_fit"]
            payload[f"tail_tp_min_{tag}"] = hrec["tail_tp_min"]
            payload[f"tail_tp_max_{tag}"] = hrec["tail_tp_max"]
            payload[f"tail_rel_rmse_{tag}"] = hrec["rel_rmse_tail"]
        flat_rows.append(payload)
    out = pd.DataFrame(flat_rows).sort_values("theta").reset_index(drop=True)
    out.to_csv(outdir / "ode_anchor_table.csv", index=False)
    with open(outdir / "ode_anchor_summary.json", "w") as f:
        json.dump({"tail_frac": tail_frac, "ode_h_values": ode_h_values, "rows": rows}, f, indent=2)
    return out


def fit_lattice_fixed_finf(df: pd.DataFrame, theta_values: np.ndarray, finf_fixed: np.ndarray) -> dict:
    def model(params: np.ndarray) -> np.ndarray:
        t_c = float(params[0])
        r = float(params[1])
        theta_index = np.array([latmod.nearest_theta(theta_values, th) for th in df["theta"].to_numpy(dtype=np.float64)], dtype=np.int64)
        x = df["x"].to_numpy(dtype=np.float64)
        f0 = df["F0"].to_numpy(dtype=np.float64)
        return np.power(x / T_OSC, 1.5) * finf_fixed[theta_index] / np.maximum(f0 * f0, 1.0e-18) + 1.0 / (
            1.0 + np.power(x / np.maximum(t_c, 1.0e-12), r)
        )

    def resid(params: np.ndarray) -> np.ndarray:
        xi_fit = model(params)
        xi = df["xi"].to_numpy(dtype=np.float64)
        return (xi_fit - xi) / np.maximum(xi, 1.0e-12)

    x0 = np.array([1.5, 1.5], dtype=np.float64)
    lower = np.array([1.0e-3, 0.1], dtype=np.float64)
    upper = np.array([10.0, 20.0], dtype=np.float64)
    res0 = least_squares(resid, x0=x0, bounds=(lower, upper), loss="soft_l1", f_scale=0.05, max_nfev=6000)
    res = least_squares(resid, x0=res0.x, bounds=(lower, upper), loss="linear", max_nfev=6000)
    xi_fit = model(res.x)
    rss = float(np.sum(np.square((xi_fit - df["xi"].to_numpy(dtype=np.float64)) / np.maximum(df["xi"].to_numpy(dtype=np.float64), 1.0e-12))))
    n = int(len(df))
    k = 2
    dof = max(n - k, 1)
    try:
        cov = (rss / dof) * np.linalg.inv(res.jac.T @ res.jac)
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.nan, dtype=np.float64)
    aic = float(n * math.log(max(rss, 1.0e-18) / n) + 2.0 * k)
    bic = float(n * math.log(max(rss, 1.0e-18) / n) + k * math.log(n))
    return {
        "params": np.asarray(res.x, dtype=np.float64),
        "covariance": cov,
        "success": bool(res.success),
        "message": res.message,
        "xi_fit": xi_fit,
        "rel_rmse": float(latmod.rel_rmse(df["xi"].to_numpy(dtype=np.float64), xi_fit)),
        "rss_frac": rss,
        "AIC": aic,
        "BIC": bic,
        "dof": dof,
    }


def plot_collapse_overlay_fixed(df: pd.DataFrame, fit_result: dict, outdir: Path, dpi: int) -> None:
    theta_values = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sorted(df["H"].unique()))))
    xi_fit = fit_result["xi_fit"]
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex=False, sharey=False)
    for ax, theta in zip(axes.flat, theta_values):
        sub = df[np.isclose(df["theta"], theta, atol=5.0e-4, rtol=0.0)].copy()
        for color, (h, hsub) in zip(colors, sub.groupby("H", sort=True)):
            ax.plot(hsub["x"], hsub["xi"], "o", ms=3.2, color=color, label=rf"$H={h:g}$")
            mask = np.isclose(sub["H"], h, atol=1.0e-12, rtol=0.0)
            hfit = xi_fit[sub.index.to_numpy()][mask]
            xfit = hsub["x"].to_numpy(dtype=np.float64)
            order = np.argsort(xfit)
            ax.plot(xfit[order], hfit[order], "-", lw=1.5, color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$x = t_p H^\beta$")
        ax.set_ylabel(r"$\xi$")
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "collapse_overlay_fixed_odeF.png", dpi=dpi)
    plt.close(fig)


def plot_lattice_plateau_estimator(df: pd.DataFrame, anchor_map: dict[float, float], outdir: Path, dpi: int) -> pd.DataFrame:
    theta_values = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sorted(df["H"].unique()))))
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), sharex=False, sharey=False)
    rows = []
    for ax, theta in zip(axes.flat, theta_values):
        sub = df[np.isclose(df["theta"], theta, atol=5.0e-4, rtol=0.0)].copy().sort_values("x")
        sub["Finf_est"] = sub["xi"].to_numpy(dtype=np.float64) * np.square(sub["F0"].to_numpy(dtype=np.float64)) / np.power(
            sub["x"].to_numpy(dtype=np.float64) / T_OSC, 1.5
        )
        for color, (h, hsub) in zip(colors, sub.groupby("H", sort=True)):
            ax.plot(hsub["x"], hsub["Finf_est"], "o-", ms=3.0, lw=1.2, color=color, label=rf"$H={h:g}$")
        anchor = float(anchor_map[float(theta)])
        ax.axhline(anchor, color="black", lw=1.6, ls="--", label=r"ODE anchor $F_\infty$" if theta == theta_values[0] else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$x = t_p H^\beta$")
        ax.set_ylabel(r"$\xi F_0^2 / (x/T_{\rm osc})^{3/2}$")
        ax.set_title(rf"$\theta={theta:.3f}$")
        ax.grid(alpha=0.25)
        tail = sub.tail(max(5, int(math.ceil(0.10 * len(sub))))).copy()
        rows.append(
            {
                "theta": float(theta),
                "F_inf_ode_anchor": anchor,
                "last_point_frac_minus1": float(tail["Finf_est"].iloc[-1] / anchor - 1.0),
                "tail_mean_frac_minus1": float(tail["Finf_est"].mean() / anchor - 1.0),
                "tail_loglog_slope": float(np.polyfit(np.log(tail["x"].to_numpy(dtype=np.float64)), np.log(tail["Finf_est"].to_numpy(dtype=np.float64)), 1)[0]),
            }
        )
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "lattice_Finf_est_vs_x_with_ode_anchor.png", dpi=dpi)
    plt.close(fig)
    out = pd.DataFrame(rows).sort_values("theta").reset_index(drop=True)
    out.to_csv(outdir / "lattice_anchor_tail_diagnostic.csv", index=False)
    return out


def main() -> None:
    args = parse_args()
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df_ode = pd.concat([load_direct_ode_table(h) for h in args.ode_h_values], ignore_index=True)
    ode_anchor_df = extract_ode_anchors(df_ode, [float(h) for h in args.ode_h_values], args.tail_frac, outdir, args.dpi)

    rho_path = latmod.resolve_first_existing(latmod.RHO_CANDIDATES, None)
    raw_lattice_path = latmod.resolve_first_existing(latmod.LATTICE_RAW_CANDIDATES, None)
    f0_table = latmod.load_f0_table(rho_path, [float(h) for h in args.lattice_h_values])
    f0_lat_map = {float(row.theta): float(row.F0) for row in f0_table.drop_duplicates("theta").itertuples(index=False)}
    f0_ode_map = {
        float(theta): float(np.median(sub["F0_ode"].to_numpy(dtype=np.float64)))
        for theta, sub in df_ode.groupby("theta", sort=True)
    }
    ode_anchor_df["F0_ode"] = [f0_ode_map[float(th)] for th in ode_anchor_df["theta"]]
    ode_anchor_df["F0_lattice"] = [
        float(f0_lat_map[min(f0_lat_map.keys(), key=lambda ref: abs(ref - float(th)))]) for th in ode_anchor_df["theta"]
    ]
    ode_anchor_df["F0_ratio_ode_over_lattice"] = ode_anchor_df["F0_ode"] / ode_anchor_df["F0_lattice"]
    ode_anchor_df["F_inf_ode_anchor_lattice_units"] = ode_anchor_df["F_inf_ode_anchor"] / np.square(
        ode_anchor_df["F0_ratio_ode_over_lattice"]
    )
    ode_anchor_df["F_inf_ode_err_lattice_units"] = ode_anchor_df["F_inf_ode_err"] / np.square(
        ode_anchor_df["F0_ratio_ode_over_lattice"]
    )
    ode_anchor_df.to_csv(outdir / "ode_anchor_table.csv", index=False)

    anchor_theta = ode_anchor_df["theta"].to_numpy(dtype=np.float64)
    anchor_vals = ode_anchor_df["F_inf_ode_anchor_lattice_units"].to_numpy(dtype=np.float64)

    lattice_df = latmod.load_lattice_data(raw_lattice_path, float(args.fixed_vw), [float(h) for h in args.lattice_h_values])
    lattice_df = latmod.merge_f0(lattice_df, f0_table)
    lattice_df = lattice_df[np.isfinite(lattice_df["F0"]) & (lattice_df["F0"] > 0.0)].copy()

    beta_payload = latmod.find_best_beta(lattice_df, args.grid_n)
    latmod.save_json(outdir / "best_beta.json", beta_payload)
    collapsed = latmod.compute_x(lattice_df, beta_payload["beta"])
    theta_values = np.array(sorted(collapsed["theta"].unique()), dtype=np.float64)
    finf_fixed = np.array([float(anchor_vals[latmod.nearest_theta(anchor_theta, theta)]) for theta in theta_values], dtype=np.float64)
    fixed_fit = fit_lattice_fixed_finf(collapsed, theta_values, finf_fixed)

    free_run = ROOT / "results_tosc_lattice_vw0p9_H1p0H1p5H2p0" / "collapse_and_fit_fanh" / "global_fit.json"
    with open(free_run) as f:
        free_payload = json.load(f)
    comparison = {
        "free_lattice_Finf": {
            "beta": float(free_payload["beta"]),
            "t_c": float(free_payload["t_c"]),
            "r": float(free_payload["r"]),
            "rel_rmse": float(free_payload["rel_rmse"]),
            "AIC": float(free_payload["AIC"]),
            "BIC": float(free_payload["BIC"]),
        },
        "fixed_ode_anchor_Finf": {
            "beta": float(beta_payload["beta"]),
            "t_c": float(fixed_fit["params"][0]),
            "t_c_err": float(np.sqrt(fixed_fit["covariance"][0, 0])) if np.isfinite(fixed_fit["covariance"][0, 0]) else np.nan,
            "r": float(fixed_fit["params"][1]),
            "r_err": float(np.sqrt(fixed_fit["covariance"][1, 1])) if np.isfinite(fixed_fit["covariance"][1, 1]) else np.nan,
            "rel_rmse": float(fixed_fit["rel_rmse"]),
            "AIC": float(fixed_fit["AIC"]),
            "BIC": float(fixed_fit["BIC"]),
            "dof": int(fixed_fit["dof"]),
        },
    }
    with open(outdir / "comparison_to_free_fit.json", "w") as f:
        json.dump(comparison, f, indent=2)

    with open(outdir / "global_fit_fixed_ode_anchor.json", "w") as f:
        json.dump(
            {
                "success": bool(fixed_fit["success"]),
                "message": fixed_fit["message"],
                "beta": float(beta_payload["beta"]),
                "t_c": float(fixed_fit["params"][0]),
                "r": float(fixed_fit["params"][1]),
                "covariance": fixed_fit["covariance"].tolist(),
                "rel_rmse": float(fixed_fit["rel_rmse"]),
                "rss_frac": float(fixed_fit["rss_frac"]),
                "AIC": float(fixed_fit["AIC"]),
                "BIC": float(fixed_fit["BIC"]),
                "dof": int(fixed_fit["dof"]),
                "F_inf_ode_anchor": {f"{theta:.10f}": float(val) for theta, val in zip(theta_values, finf_fixed)},
            },
            f,
            indent=2,
        )

    plot_collapse_overlay_fixed(collapsed, fixed_fit, outdir, args.dpi)
    anchor_map_for_plot = {float(theta): float(anchor_vals[latmod.nearest_theta(anchor_theta, theta)]) for theta in theta_values}
    lattice_tail_df = plot_lattice_plateau_estimator(collapsed, anchor_map_for_plot, outdir, args.dpi)

    inversion_rows = []
    for theta, group in collapsed.groupby("theta", sort=True):
        finf_anchor = float(anchor_vals[latmod.nearest_theta(anchor_theta, float(theta))])
        inversion_rows.append(latmod.inversion_single_theta(group, finf_anchor))
    latmod.plot_inversion(inversion_rows, outdir, args.dpi)
    inversion_records = [
        {
            "theta": float(row["theta"]),
            "r_eff": float(row["r"]),
            "r_err": float(row["r_err"]) if np.isfinite(row["r_err"]) else np.nan,
            "n_points_used": int(row["n_points_used"]),
        }
        for row in inversion_rows
        if row is not None
    ]
    inversion_table = pd.DataFrame(inversion_records)
    if not inversion_table.empty and "theta" in inversion_table.columns:
        inversion_table = inversion_table.sort_values("theta").reset_index(drop=True)
    inversion_table.to_csv(outdir / "inversion_fixed_ode_anchor.csv", index=False)

    final_summary = {
        "fixed_vw": float(args.fixed_vw),
        "lattice_h_values": [float(h) for h in args.lattice_h_values],
        "ode_h_values": [float(h) for h in args.ode_h_values],
        "tail_frac": float(args.tail_frac),
        "beta": float(beta_payload["beta"]),
        "comparison": comparison,
        "worst_theta_tail_mismatch": float(
            lattice_tail_df.iloc[lattice_tail_df["tail_mean_frac_minus1"].abs().argmax()]["theta"]
        ),
    }
    with open(outdir / "final_summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)

    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
