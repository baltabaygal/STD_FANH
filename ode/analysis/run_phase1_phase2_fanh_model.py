import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ode.hom_ODE.percolation import PercolationCache


EXPECTED_INPUTS = {
    "rho_noPT_data.txt": "lattice_data/data/rho_noPT_data.txt",
    "xi_lattice_scan_H1p5.txt": "lattice_data/data/energy_ratio_by_theta_data_v9.txt",
    "xi_lattice_scan_H2p0.txt": "lattice_data/data/energy_ratio_by_theta_data_v9.txt",
    "xi_ode_scan.txt": "ode/xi_DM_ODE_results.txt",
    "ode_nopt_reference.txt": "ode/analysis/results/noPT_reference_dm_tp_fitready_H1p000.txt",
}

LATTICE_TARGET_H = [1.5, 2.0]
ODE_TARGET_H = [0.5, 1.0, 1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Phase-1 diagnostics and Phase-2 global fits for the fanh/xi model. "
            "Adapts the requested filenames to the current repo layout."
        )
    )
    p.add_argument("--rho-nopt", type=str, default=EXPECTED_INPUTS["rho_noPT_data.txt"])
    p.add_argument("--lattice-ratio", type=str, default="lattice_data/data/energy_ratio_by_theta_data_v9.txt")
    p.add_argument("--ode-xi", type=str, default=EXPECTED_INPUTS["xi_ode_scan.txt"])
    p.add_argument("--ode-nopt", type=str, default=EXPECTED_INPUTS["ode_nopt_reference.txt"])
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--f0-use-all-h", action="store_true", help="Use all H values in rho_noPT_data to reconstruct F0.")
    p.add_argument("--smalltp-min-points", type=int, default=5)
    p.add_argument("--smalltp-max-points", type=int, default=10)
    p.add_argument("--tail-frac", type=float, default=0.10)
    p.add_argument("--huber-fscale", type=float, default=0.05)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--bootstrap-jobs", type=int, default=min(6, max(os.cpu_count() or 1, 1)))
    p.add_argument("--bootstrap-seed", type=int, default=12345)
    p.add_argument("--profile-grid-n", type=int, default=80)
    p.add_argument(
        "--outdir",
        type=str,
        default="ode/analysis/results/phase1_phase2_fanh_model_v9",
    )
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def potential(theta):
    theta = np.asarray(theta, dtype=np.float64)
    return 1.0 - np.cos(theta)


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


def rel_rmse(y, yfit):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square((yfit - y) / y))))


def save_json(path, payload):
    path = Path(path)
    path.write_text(json.dumps(to_native(payload), indent=2, sort_keys=True))


def require_file(path, label):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing required input for {label}: {p}")
    return p


def nearest_theta(values, theta0, atol=5.0e-4):
    values = np.asarray(values, dtype=np.float64)
    idx = int(np.argmin(np.abs(values - float(theta0))))
    if abs(values[idx] - float(theta0)) > atol:
        raise RuntimeError(f"No theta match for theta0={theta0:.10f}")
    return idx


def load_ode_nopt(path):
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    return pd.DataFrame({"theta": arr[:, 0], "F0_ref": arr[:, 2]})


def reconstruct_f0_table(rho_path, ode_nopt_df, outdir, use_all_h):
    rho_df = pd.read_csv(rho_path, sep=r"\s+")
    rho_df = rho_df.rename(columns={"theta0": "theta", "H_PT": "Hstar"})
    if not use_all_h:
        rho_df = rho_df[rho_df["Hstar"].isin(LATTICE_TARGET_H)].copy()
    rho_df["raw_F0"] = rho_df["rho"] / (potential(rho_df["theta"].to_numpy()) * np.power(rho_df["Hstar"].to_numpy(), 1.5))

    theta_ref = ode_nopt_df["theta"].to_numpy()
    f0_ref = ode_nopt_df["F0_ref"].to_numpy()
    ref_vals = []
    raw_vals = []
    for row in rho_df.itertuples(index=False):
        idx = nearest_theta(theta_ref, row.theta)
        ref_vals.append(float(f0_ref[idx]))
        raw_vals.append(float(row.raw_F0))
    ref_vals = np.asarray(ref_vals, dtype=np.float64)
    raw_vals = np.asarray(raw_vals, dtype=np.float64)
    scale = float(np.dot(raw_vals, ref_vals) / np.dot(raw_vals, raw_vals))
    rho_df["F0_scaled"] = scale * rho_df["raw_F0"]

    f0_df = (
        rho_df.groupby("theta", as_index=False)
        .agg(F0=("F0_scaled", "median"), F0_std=("F0_scaled", "std"), nH=("F0_scaled", "size"))
        .sort_values("theta")
        .reset_index(drop=True)
    )
    f0_df["F0_std"] = f0_df["F0_std"].fillna(0.0)
    f0_df[["theta", "F0"]].to_csv(outdir / "F0_table.csv", index=False)
    save_json(
        outdir / "F0_reconstruction_meta.json",
        {
            "input_rho": str(Path(rho_path).resolve()),
            "input_ode_nopt": str(Path(EXPECTED_INPUTS["ode_nopt_reference.txt"]).resolve()),
            "used_all_H": bool(use_all_h),
            "scale_to_ode_reference": scale,
            "rel_rmse_to_ode_reference": rel_rmse(ref_vals, scale * raw_vals),
        },
    )
    return f0_df[["theta", "F0"]].copy(), f0_df.copy()


def load_lattice_scan(path, hstar, vw):
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    df = pd.DataFrame(
        {
            "theta": arr[:, 0],
            "Hstar": arr[:, 1],
            "beta_over_H": arr[:, 2],
            "beta": arr[:, 3],
            "xi": arr[:, 4],
            "std_ratio": arr[:, 5],
            "N_samples": arr[:, 6],
        }
    )
    df = df[np.isclose(df["Hstar"], float(hstar), rtol=0.0, atol=1.0e-12)].copy()
    if df.empty:
        raise RuntimeError(f"No lattice xi rows found for H*={hstar:g} in {path}")
    df["xi_sem"] = df["std_ratio"] / np.sqrt(np.maximum(df["N_samples"], 1.0))
    perc = PercolationCache()
    df["tp"] = [perc.get(float(h), float(bh), float(vw)) for h, bh in zip(df["Hstar"], df["beta_over_H"])]
    df["dataset"] = f"H{str(hstar).replace('.', 'p')}"
    return df.sort_values(["theta", "tp"]).reset_index(drop=True)


def load_ode_scan(path, vw):
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    df = pd.DataFrame(
        {
            "vw": arr[:, 0],
            "theta": arr[:, 1],
            "Hstar": arr[:, 2],
            "beta_over_H": arr[:, 3],
            "tp": arr[:, 4],
            "xi": arr[:, 5],
        }
    )
    df = df[np.isclose(df["vw"], float(vw), rtol=0.0, atol=1.0e-12)].copy()
    df = df[df["Hstar"].isin(ODE_TARGET_H)].copy()
    if df.empty:
        raise RuntimeError(f"No ODE xi rows found for vw={vw:g} in {path}")
    df["dataset"] = "ODE"
    return df.sort_values(["Hstar", "theta", "tp"]).reset_index(drop=True)


def merge_f0(df, f0_table):
    theta_ref = f0_table["theta"].to_numpy(dtype=np.float64)
    f0_ref = f0_table["F0"].to_numpy(dtype=np.float64)
    out = df.copy()
    out["F0"] = [float(f0_ref[nearest_theta(theta_ref, th)]) for th in out["theta"].to_numpy(dtype=np.float64)]
    return out


def export_adapted_scans(df_h15, df_h20, df_ode, outdir):
    df_h15[["theta", "tp", "xi", "Hstar", "beta_over_H", "xi_sem"]].to_csv(outdir / "xi_lattice_scan_H1p5.txt", sep="\t", index=False)
    df_h20[["theta", "tp", "xi", "Hstar", "beta_over_H", "xi_sem"]].to_csv(outdir / "xi_lattice_scan_H2p0.txt", sep="\t", index=False)
    df_ode[["theta", "tp", "xi", "Hstar", "beta_over_H"]].to_csv(outdir / "xi_ode_scan.txt", sep="\t", index=False)


def compute_fanh_and_save(df, out_csv):
    out = df.copy()
    out["fanh_data"] = out["xi"] * out["F0"] / np.power(out["tp"], 1.5)
    out.to_csv(out_csv, index=False)
    return out


def group_keys_for_dataset(name):
    if name == "ODE":
        return ["Hstar", "theta"]
    return ["theta"]


def robust_loglog_fit(tp, delta, huber_fscale):
    x = np.log(np.asarray(tp, dtype=np.float64))
    y = np.log(np.abs(np.asarray(delta, dtype=np.float64)))

    def resid(par):
        a, p = par
        return a + p * x - y

    res = least_squares(resid, x0=np.array([float(np.median(y)), 1.5]), loss="soft_l1", f_scale=huber_fscale)
    jac = res.jac
    dof = max(len(y) - 2, 1)
    rss = float(np.sum(np.square(res.fun)))
    cov = np.full((2, 2), np.nan, dtype=np.float64)
    try:
        jtj_inv = np.linalg.inv(jac.T @ jac)
        cov = (rss / dof) * jtj_inv
    except np.linalg.LinAlgError:
        pass
    p_err = float(np.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) else np.nan
    return {
        "a": float(res.x[0]),
        "p": float(res.x[1]),
        "p_err": p_err,
        "success": bool(res.success),
    }


def select_smalltp_slice(group, min_points, max_points):
    group = group.sort_values("tp").copy()
    delta = group["xi"].to_numpy(dtype=np.float64) - 1.0
    sem = group["xi_sem"].to_numpy(dtype=np.float64) if "xi_sem" in group.columns else np.full(len(group), np.nan)
    noise_floor = max(1.0e-6, float(np.nanmedian(sem[np.isfinite(sem)])) if np.any(np.isfinite(sem)) else 1.0e-6)
    mask = np.abs(delta) > max(1.0e-6, 2.0 * noise_floor)
    cand = group.loc[mask].copy()
    if len(cand) < min_points:
        cand = group.head(min(max_points, len(group))).copy()
    else:
        cand = cand.head(min(max_points, len(cand))).copy()
    return cand


def estimate_smalltp_exponents(df, dataset_name, out_csv, huber_fscale, min_points, max_points):
    rows = []
    for keys, group in df.groupby(group_keys_for_dataset(dataset_name), sort=True):
        small = select_smalltp_slice(group, min_points, max_points)
        delta = small["xi"].to_numpy(dtype=np.float64) - 1.0
        sign = float(np.sign(np.nanmedian(delta))) if len(delta) else np.nan
        fit = robust_loglog_fit(small["tp"].to_numpy(dtype=np.float64), delta, huber_fscale)
        row = {"theta": float(small["theta"].iloc[0]), "p": fit["p"], "p_err": fit["p_err"], "sign": sign}
        if dataset_name == "ODE":
            row["Hstar"] = float(small["Hstar"].iloc[0])
        rows.append(row)
    out = pd.DataFrame(rows).sort_values([c for c in ["Hstar", "theta"] if c in rows[0]]).reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    return out


def make_smalltp_plot(exponent_map, out_path, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), sharey=True)
    configs = [("H1p5", axes[0], "Lattice H*=1.5"), ("H2p0", axes[1], "Lattice H*=2.0"), ("ODE", axes[2], "ODE")]
    for name, ax, title in configs:
        df = exponent_map[name]
        if name == "ODE":
            for hstar, sub in df.groupby("Hstar", sort=True):
                ax.errorbar(sub["theta"], sub["p"], yerr=sub["p_err"], marker="o", ms=4.0, lw=1.2, label=rf"$H_*={hstar:g}$")
        else:
            ax.errorbar(df["theta"], df["p"], yerr=df["p_err"], marker="o", ms=4.5, lw=1.4, label=title)
        ax.axhline(1.5, color="black", lw=1.0, ls="--", alpha=0.6)
        ax.set_xlabel(r"$\theta$")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("local small-$t_p$ exponent")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def fit_tail_amplitude(group, huber_fscale, tail_frac):
    group = group.sort_values("tp").copy()
    ntail = max(3, int(math.ceil(tail_frac * len(group))))
    tail = group.tail(ntail).copy()
    x = np.power(tail["tp"].to_numpy(dtype=np.float64), 1.5)
    y = tail["xi"].to_numpy(dtype=np.float64)

    def resid(par):
        c = par[0]
        return (c * x - y) / np.maximum(y, 1.0e-12)

    c0 = float(np.median(y / np.maximum(x, 1.0e-12)))
    res = least_squares(resid, x0=np.array([c0]), loss="soft_l1", f_scale=huber_fscale)
    jac = res.jac
    dof = max(len(y) - 1, 1)
    rss = float(np.sum(np.square(res.fun)))
    try:
        cov = (rss / dof) * np.linalg.inv(jac.T @ jac)
        c_err = float(np.sqrt(cov[0, 0]))
    except np.linalg.LinAlgError:
        c_err = np.nan
    return {
        "C": float(res.x[0]),
        "C_err": c_err,
        "tail_n": ntail,
        "tp_min_tail": float(np.min(tail["tp"])),
    }


def estimate_finf_tail(df, dataset_name, out_csv, huber_fscale, tail_frac):
    rows = []
    for keys, group in df.groupby(group_keys_for_dataset(dataset_name), sort=True):
        fit = fit_tail_amplitude(group, huber_fscale, tail_frac)
        theta = float(group["theta"].iloc[0])
        f0 = float(group["F0"].iloc[0])
        row = {
            "theta": theta,
            "F_inf_tail": fit["C"] * (f0 ** 2),
            "err": abs(fit["C_err"] * (f0 ** 2)) if np.isfinite(fit["C_err"]) else np.nan,
            "C": fit["C"],
            "C_err": fit["C_err"],
            "tail_n": fit["tail_n"],
            "tp_min_tail": fit["tp_min_tail"],
        }
        if dataset_name == "ODE":
            row["Hstar"] = float(group["Hstar"].iloc[0])
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["Hstar", "theta"] if c in out.columns]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    return out


def make_finf_tail_plot(tail_map, out_path, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), sharey=True)
    configs = [("H1p5", axes[0], "Tail F_inf, H*=1.5"), ("H2p0", axes[1], "Tail F_inf, H*=2.0"), ("ODE", axes[2], "Tail F_inf, ODE")]
    for name, ax, title in configs:
        df = tail_map[name]
        if name == "ODE":
            for hstar, sub in df.groupby("Hstar", sort=True):
                ax.errorbar(sub["theta"], sub["F_inf_tail"], yerr=sub["err"], marker="o", ms=4.0, lw=1.2, label=rf"$H_*={hstar:g}$")
        else:
            ax.errorbar(df["theta"], df["F_inf_tail"], yerr=df["err"], marker="o", ms=4.5, lw=1.4)
        ax.set_xlabel(r"$\theta$")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"$F_{\infty}^{\rm tail}(\theta)$")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def estimate_A_small(df, exp_df, dataset_name):
    rows = []
    merge_cols = ["theta"] + (["Hstar"] if dataset_name == "ODE" else [])
    merged = df.merge(exp_df, on=merge_cols, how="left")
    for keys, group in merged.groupby(group_keys_for_dataset(dataset_name), sort=True):
        group = group.sort_values("tp")
        p_est = float(group["p"].iloc[0])
        if not np.isfinite(p_est) or abs(p_est - 1.5) > 0.2:
            rows.append({**({} if dataset_name != "ODE" else {"Hstar": float(group["Hstar"].iloc[0])}), "theta": float(group["theta"].iloc[0]), "A_small": np.nan, "A_small_err": np.nan, "small_n": 0})
            continue
        small = select_smalltp_slice(group, 5, 8)
        vals = (small["xi"].to_numpy(dtype=np.float64) - 1.0) / np.power(small["tp"].to_numpy(dtype=np.float64), 1.5)
        rows.append(
            {
                **({} if dataset_name != "ODE" else {"Hstar": float(small["Hstar"].iloc[0])}),
                "theta": float(small["theta"].iloc[0]),
                "A_small": float(np.mean(vals)),
                "A_small_err": float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                "small_n": int(len(vals)),
            }
        )
    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["Hstar", "theta"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def make_A_comparison(dataset_name, A_df, tail_df, out_csv, out_png, dpi):
    merge_cols = ["theta"] + (["Hstar"] if dataset_name == "ODE" else [])
    out = A_df.merge(tail_df[merge_cols + ["F_inf_tail"]], on=merge_cols, how="left")
    out["A_tail"] = out["F_inf_tail"] / np.square(out["theta"].map({}))  # placeholder to overwrite below
    out.to_csv(out_csv, index=False)


def robust_sign_majority(exp_df):
    if exp_df.empty:
        return False
    sub = exp_df[np.isfinite(exp_df["p"])]
    if sub.empty:
        return False
    neg_mask = sub["sign"] < 0
    low_mask = sub["p"] < 1.4
    return bool(np.mean(low_mask) > 0.5 and np.mean(neg_mask) > 0.5)


def fit_r_B_global(df_small, huber_fscale):
    tp = df_small["tp"].to_numpy(dtype=np.float64)
    y = df_small["xi"].to_numpy(dtype=np.float64) - 1.0

    def resid(par):
        logB, r = par
        model = -np.exp(logB) * np.power(tp, r)
        return (model - y) / np.maximum(np.abs(y), 1.0e-8)

    res = least_squares(resid, x0=np.array([np.log(0.1), 1.0]), bounds=([-30.0, 0.5], [30.0, 12.0]), loss="soft_l1", f_scale=huber_fscale)
    jac = res.jac
    dof = max(len(y) - 2, 1)
    rss = float(np.sum(np.square(res.fun)))
    try:
        cov = (rss / dof) * np.linalg.inv(jac.T @ jac)
        logB_err = float(np.sqrt(cov[0, 0]))
        r_err = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        logB_err = np.nan
        r_err = np.nan
    B = float(np.exp(res.x[0]))
    B_err = float(B * logB_err) if np.isfinite(logB_err) else np.nan
    return {"r": float(res.x[1]), "r_err": r_err, "B": B, "B_err": B_err, "n_points": int(len(y))}


def huber_rho(residuals, fscale):
    z = np.asarray(residuals, dtype=np.float64) / max(float(fscale), 1.0e-12)
    return np.sum(2.0 * (np.sqrt(1.0 + z * z) - 1.0))


def solve_finf_with_priors(df, theta_values, theta_index, transient, prior_center, prior_sigma):
    xi = df["xi"].to_numpy(dtype=np.float64)
    tp = df["tp"].to_numpy(dtype=np.float64)
    f0 = df["F0"].to_numpy(dtype=np.float64)
    a = np.power(tp, 1.5) / np.square(f0)
    w = 1.0 / np.square(np.maximum(xi, 1.0e-12))
    finf = np.zeros(len(theta_values), dtype=np.float64)
    for i in range(len(theta_values)):
        mask = theta_index == i
        ai = a[mask]
        yi = xi[mask]
        ti = transient[mask]
        wi = w[mask]
        mu = float(prior_center[i])
        sig = float(prior_sigma[i])
        lhs = float(np.sum(wi * ai * ai) + 1.0 / (sig * sig))
        rhs = float(np.sum(wi * ai * (yi - ti)) + mu / (sig * sig))
        finf[i] = max(rhs / max(lhs, 1.0e-18), 1.0e-12)
    return finf


def model_A_transient(tp, tceff, r):
    return 1.0 / (1.0 + np.power(tp / max(float(tceff), 1.0e-12), float(r)))


def model_B_tpeff(tp, tau):
    tau = float(tau)
    tp = np.asarray(tp, dtype=np.float64)
    if tau <= 1.0e-12:
        return tp
    return tp + tau * (1.0 - np.exp(-tp / tau))


def evaluate_model_A(df, theta_values, theta_index, prior_center, prior_sigma, tceff, r, prior_r=None, prior_tceff=None):
    tp = df["tp"].to_numpy(dtype=np.float64)
    transient = model_A_transient(tp, tceff, r)
    finf = solve_finf_with_priors(df, theta_values, theta_index, transient, prior_center, prior_sigma)
    xi_fit = np.power(tp, 1.5) * finf[theta_index] / np.square(df["F0"].to_numpy(dtype=np.float64)) + transient
    data_resid = (xi_fit - df["xi"].to_numpy(dtype=np.float64)) / df["xi"].to_numpy(dtype=np.float64)
    prior_f = (finf - prior_center) / prior_sigma
    res = [data_resid, prior_f]
    if prior_r is not None:
        res.append(np.array([(r - prior_r["center"]) / prior_r["sigma"]], dtype=np.float64))
    if prior_tceff is not None:
        res.append(np.array([(tceff - prior_tceff["center"]) / prior_tceff["sigma"]], dtype=np.float64))
    resid = np.concatenate(res)
    return {
        "t_c_eff": float(tceff),
        "r": float(r),
        "F_inf": finf,
        "xi_fit": xi_fit,
        "residual_vector": resid,
        "data_resid": data_resid,
        "rel_rmse": rel_rmse(df["xi"].to_numpy(dtype=np.float64), xi_fit),
        "rss_frac": float(np.sum(np.square(data_resid))),
    }


def evaluate_model_B(df, theta_values, theta_index, prior_center, prior_sigma, tceff, r, tau, prior_r=None, prior_tceff=None):
    tp = df["tp"].to_numpy(dtype=np.float64)
    tp_eff = model_B_tpeff(tp, tau)
    transient = 1.0 / (1.0 + np.power(tp_eff / max(float(tceff), 1.0e-12), float(r)))
    finf = solve_finf_with_priors(df, theta_values, theta_index, transient, prior_center, prior_sigma)
    xi_fit = np.power(tp, 1.5) * finf[theta_index] / np.square(df["F0"].to_numpy(dtype=np.float64)) + transient
    data_resid = (xi_fit - df["xi"].to_numpy(dtype=np.float64)) / df["xi"].to_numpy(dtype=np.float64)
    prior_f = (finf - prior_center) / prior_sigma
    res = [data_resid, prior_f]
    if prior_r is not None:
        res.append(np.array([(r - prior_r["center"]) / prior_r["sigma"]], dtype=np.float64))
    if prior_tceff is not None:
        res.append(np.array([(tceff - prior_tceff["center"]) / prior_tceff["sigma"]], dtype=np.float64))
    resid = np.concatenate(res)
    return {
        "t_c_eff": float(tceff),
        "r": float(r),
        "tau_p": float(tau),
        "F_inf": finf,
        "xi_fit": xi_fit,
        "residual_vector": resid,
        "data_resid": data_resid,
        "rel_rmse": rel_rmse(df["xi"].to_numpy(dtype=np.float64), xi_fit),
        "rss_frac": float(np.sum(np.square(data_resid))),
    }


def make_modelA_objective(df, theta_values, theta_index, prior_center, prior_sigma, prior_r, prior_tceff, huber_fscale):
    def objective(par, huber=False):
        rec = evaluate_model_A(df, theta_values, theta_index, prior_center, prior_sigma, float(par[0]), float(par[1]), prior_r, prior_tceff)
        if huber:
            return huber_rho(rec["residual_vector"], huber_fscale)
        return rec["residual_vector"]
    return objective


def make_modelB_objective(df, theta_values, theta_index, prior_center, prior_sigma, prior_r, prior_tceff, huber_fscale):
    def objective(par, huber=False):
        rec = evaluate_model_B(df, theta_values, theta_index, prior_center, prior_sigma, float(par[0]), float(par[1]), float(par[2]), prior_r, prior_tceff)
        if huber:
            return huber_rho(rec["residual_vector"], huber_fscale)
        return rec["residual_vector"]
    return objective


def coarse_grid_start(objective, t_bounds, r_bounds, grid_n):
    t_grid = np.linspace(t_bounds[0], t_bounds[1], grid_n)
    r_grid = np.linspace(r_bounds[0], r_bounds[1], grid_n)
    best = None
    for tceff in t_grid:
        for r in r_grid:
            val = objective(np.array([tceff, r], dtype=np.float64), huber=True)
            if best is None or val < best[0]:
                best = (val, np.array([tceff, r], dtype=np.float64))
    return best[1]


def coarse_grid_start_B(objective, t_bounds, r_bounds, tau_bounds, grid_n):
    t_grid = np.linspace(t_bounds[0], t_bounds[1], max(8, grid_n // 2))
    r_grid = np.linspace(r_bounds[0], r_bounds[1], max(8, grid_n // 2))
    tau_grid = np.linspace(tau_bounds[0], tau_bounds[1], max(6, grid_n // 3))
    best = None
    for tceff in t_grid:
        for r in r_grid:
            for tau in tau_grid:
                val = objective(np.array([tceff, r, tau], dtype=np.float64), huber=True)
                if best is None or val < best[0]:
                    best = (val, np.array([tceff, r, tau], dtype=np.float64))
    return best[1]


def finite_difference_cov(fun_resid, xbest):
    xbest = np.asarray(xbest, dtype=np.float64)
    eps = np.maximum(1.0e-4, 1.0e-3 * np.abs(xbest))
    r0 = fun_resid(xbest)
    jac = np.zeros((len(r0), len(xbest)), dtype=np.float64)
    for i in range(len(xbest)):
        xp = xbest.copy()
        xm = xbest.copy()
        xp[i] += eps[i]
        xm[i] -= eps[i]
        jac[:, i] = (fun_resid(xp) - fun_resid(xm)) / (2.0 * eps[i])
    dof = max(len(r0) - len(xbest), 1)
    rss = float(np.sum(np.square(r0)))
    try:
        cov = (rss / dof) * np.linalg.inv(jac.T @ jac)
    except np.linalg.LinAlgError:
        cov = np.full((len(xbest), len(xbest)), np.nan, dtype=np.float64)
    return cov


def fit_model_A(df, priors, huber_fscale, grid_n):
    theta_values = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    theta_index = np.array([nearest_theta(theta_values, th) for th in df["theta"].to_numpy(dtype=np.float64)], dtype=np.int64)
    prior_center = np.array([priors["F_inf"][float(th)]["center"] for th in theta_values], dtype=np.float64)
    prior_sigma = np.array([priors["F_inf"][float(th)]["sigma"] for th in theta_values], dtype=np.float64)
    objective = make_modelA_objective(df, theta_values, theta_index, prior_center, prior_sigma, priors["r"], priors["t_c_eff"], huber_fscale)
    x0 = coarse_grid_start(objective, (0.5, 4.0), (0.5, 12.0), grid_n)
    res0 = least_squares(lambda x: objective(x, huber=False), x0=x0, bounds=([0.5, 0.5], [4.0, 12.0]), loss="soft_l1", f_scale=huber_fscale)
    res = least_squares(lambda x: objective(x, huber=False), x0=res0.x, bounds=([0.5, 0.5], [4.0, 12.0]), loss="linear")
    rec = evaluate_model_A(df, theta_values, theta_index, prior_center, prior_sigma, float(res.x[0]), float(res.x[1]), priors["r"], priors["t_c_eff"])
    cov = finite_difference_cov(lambda x: make_modelA_objective(df, theta_values, theta_index, prior_center, prior_sigma, priors["r"], priors["t_c_eff"], huber_fscale)(x, huber=False), res.x)
    rec["theta_values"] = theta_values
    rec["theta_index"] = theta_index
    rec["cov"] = cov
    rec["param_err"] = {
        "t_c_eff": float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else np.nan,
        "r": float(np.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) else np.nan,
    }
    rec["success"] = bool(res.success)
    rec["n_points"] = int(len(df))
    return rec


def fit_model_B(df, priors, huber_fscale, grid_n):
    theta_values = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    theta_index = np.array([nearest_theta(theta_values, th) for th in df["theta"].to_numpy(dtype=np.float64)], dtype=np.int64)
    prior_center = np.array([priors["F_inf"][float(th)]["center"] for th in theta_values], dtype=np.float64)
    prior_sigma = np.array([priors["F_inf"][float(th)]["sigma"] for th in theta_values], dtype=np.float64)
    objective = make_modelB_objective(df, theta_values, theta_index, prior_center, prior_sigma, priors["r"], priors["t_c_eff"], huber_fscale)
    x0 = coarse_grid_start_B(objective, (0.5, 4.0), (0.5, 12.0), (0.0, 3.0), grid_n)
    res0 = least_squares(lambda x: objective(x, huber=False), x0=x0, bounds=([0.5, 0.5, 0.0], [4.0, 12.0, 3.0]), loss="soft_l1", f_scale=huber_fscale)
    res = least_squares(lambda x: objective(x, huber=False), x0=res0.x, bounds=([0.5, 0.5, 0.0], [4.0, 12.0, 3.0]), loss="linear")
    rec = evaluate_model_B(df, theta_values, theta_index, prior_center, prior_sigma, float(res.x[0]), float(res.x[1]), float(res.x[2]), priors["r"], priors["t_c_eff"])
    cov = finite_difference_cov(lambda x: make_modelB_objective(df, theta_values, theta_index, prior_center, prior_sigma, priors["r"], priors["t_c_eff"], huber_fscale)(x, huber=False), res.x)
    rec["theta_values"] = theta_values
    rec["theta_index"] = theta_index
    rec["cov"] = cov
    rec["param_err"] = {
        "t_c_eff": float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else np.nan,
        "r": float(np.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) else np.nan,
        "tau_p": float(np.sqrt(cov[2, 2])) if np.isfinite(cov[2, 2]) else np.nan,
    }
    rec["success"] = bool(res.success)
    rec["n_points"] = int(len(df))
    return rec


def profile_tceff(df, priors, huber_fscale, grid_n):
    theta_values = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    theta_index = np.array([nearest_theta(theta_values, th) for th in df["theta"].to_numpy(dtype=np.float64)], dtype=np.int64)
    prior_center = np.array([priors["F_inf"][float(th)]["center"] for th in theta_values], dtype=np.float64)
    prior_sigma = np.array([priors["F_inf"][float(th)]["sigma"] for th in theta_values], dtype=np.float64)
    t_grid = np.linspace(0.5, 4.0, grid_n)
    rows = []
    for tceff in t_grid:
        def objective_r(r):
            rec = evaluate_model_A(df, theta_values, theta_index, prior_center, prior_sigma, float(tceff), float(r), priors["r"], priors["t_c_eff"])
            return np.sum(np.square(rec["residual_vector"]))
        probe = np.linspace(0.5, 12.0, 80)
        vals = np.array([objective_r(r) for r in probe], dtype=np.float64)
        idx = int(np.argmin(vals))
        a = probe[max(idx - 1, 0)]
        b = probe[min(idx + 1, len(probe) - 1)]
        res = minimize_scalar(objective_r, bounds=(float(a), float(b)), method="bounded")
        rec = evaluate_model_A(df, theta_values, theta_index, prior_center, prior_sigma, float(tceff), float(res.x), priors["r"], priors["t_c_eff"])
        rows.append({"t_c_eff": float(tceff), "r": float(res.x), "rel_rmse": rec["rel_rmse"], "rss_frac": rec["rss_frac"]})
    return pd.DataFrame(rows)


def contour_tceff_r(df, priors, grid_n):
    theta_values = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    theta_index = np.array([nearest_theta(theta_values, th) for th in df["theta"].to_numpy(dtype=np.float64)], dtype=np.int64)
    prior_center = np.array([priors["F_inf"][float(th)]["center"] for th in theta_values], dtype=np.float64)
    prior_sigma = np.array([priors["F_inf"][float(th)]["sigma"] for th in theta_values], dtype=np.float64)
    t_grid = np.linspace(0.5, 4.0, grid_n)
    r_grid = np.linspace(0.5, 12.0, grid_n)
    Z = np.zeros((len(r_grid), len(t_grid)), dtype=np.float64)
    for ir, r in enumerate(r_grid):
        for it, tceff in enumerate(t_grid):
            rec = evaluate_model_A(df, theta_values, theta_index, prior_center, prior_sigma, float(tceff), float(r), priors["r"], priors["t_c_eff"])
            Z[ir, it] = rec["rel_rmse"]
    return t_grid, r_grid, Z


def save_global_fit_json(path, rec, dataset_label):
    theta_entries = {
        f"{float(th):.10f}": {"F_inf": float(finf)}
        for th, finf in zip(rec["theta_values"], rec["F_inf"])
    }
    payload = {
        "dataset": dataset_label,
        "t_c_eff": rec["t_c_eff"],
        "t_c_eff_err": rec["param_err"]["t_c_eff"],
        "r": rec["r"],
        "r_err": rec["param_err"]["r"],
        "rel_rmse": rec["rel_rmse"],
        "rss_frac": rec["rss_frac"],
        "n_points": rec["n_points"],
        "success": rec["success"],
        "F_inf": theta_entries,
        "covariance": rec["cov"],
    }
    save_json(path, payload)
    return payload


def residual_bootstrap_worker(payload):
    seed = payload["seed"]
    rng = np.random.default_rng(seed)
    df = payload["df"].copy()
    model = payload["model"]
    fitted = payload["fitted"]
    resid = payload["resid"]
    out = []
    for theta in sorted(df["theta"].unique()):
        mask = np.isclose(df["theta"].to_numpy(dtype=np.float64), float(theta), rtol=0.0, atol=5.0e-4)
        rtheta = resid[mask]
        sample = rng.choice(rtheta, size=np.sum(mask), replace=True)
        out.append((mask, sample))
    xi_boot = fitted.copy()
    for mask, sample in out:
        xi_boot[mask] = fitted[mask] * (1.0 + sample)
    xi_boot = np.maximum(xi_boot, 1.0e-8)
    df["xi"] = xi_boot
    priors = payload["priors"]
    huber_fscale = payload["huber_fscale"]
    grid_n = payload["grid_n"]
    if model == "A":
        rec = fit_model_A(df, priors, huber_fscale, grid_n)
        return {"t_c_eff": rec["t_c_eff"], "r": rec["r"], "F_inf": rec["F_inf"]}
    rec = fit_model_B(df, priors, huber_fscale, max(10, grid_n // 2))
    return {"t_c_eff": rec["t_c_eff"], "r": rec["r"], "tau_p": rec["tau_p"], "F_inf": rec["F_inf"]}


def run_bootstrap(df, rec, priors, model, nboot, seed, jobs, huber_fscale, grid_n):
    payloads = []
    for i in range(nboot):
        payloads.append(
            {
                "seed": int(seed + i),
                "df": df,
                "model": model,
                "fitted": rec["xi_fit"],
                "resid": rec["data_resid"],
                "priors": priors,
                "huber_fscale": huber_fscale,
                "grid_n": max(12, grid_n // 2),
            }
        )
    results = []
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for item in ex.map(residual_bootstrap_worker, payloads):
                results.append(item)
    else:
        for payload in payloads:
            results.append(residual_bootstrap_worker(payload))
    return results


def summarize_bootstrap(rec, boot, model):
    theta_values = rec["theta_values"]
    summary = {
        "t_c_eff": {
            "p16": float(np.percentile([b["t_c_eff"] for b in boot], 16.0)),
            "p50": float(np.percentile([b["t_c_eff"] for b in boot], 50.0)),
            "p84": float(np.percentile([b["t_c_eff"] for b in boot], 84.0)),
        },
        "r": {
            "p16": float(np.percentile([b["r"] for b in boot], 16.0)),
            "p50": float(np.percentile([b["r"] for b in boot], 50.0)),
            "p84": float(np.percentile([b["r"] for b in boot], 84.0)),
        },
        "F_inf": {},
    }
    if model == "B":
        summary["tau_p"] = {
            "p16": float(np.percentile([b["tau_p"] for b in boot], 16.0)),
            "p50": float(np.percentile([b["tau_p"] for b in boot], 50.0)),
            "p84": float(np.percentile([b["tau_p"] for b in boot], 84.0)),
        }
    for i, th in enumerate(theta_values):
        vals = [b["F_inf"][i] for b in boot]
        summary["F_inf"][f"{float(th):.10f}"] = {
            "p16": float(np.percentile(vals, 16.0)),
            "p50": float(np.percentile(vals, 50.0)),
            "p84": float(np.percentile(vals, 84.0)),
        }
    return summary


def make_fanh_theta_plots(df, rec, dataset_tag, outdir, dpi):
    for theta in sorted(df["theta"].unique()):
        sub = df[np.isclose(df["theta"], float(theta), rtol=0.0, atol=5.0e-4)].copy().sort_values("tp")
        idx = nearest_theta(rec["theta_values"], theta)
        finf = float(rec["F_inf"][idx])
        if "tau_p" in rec:
            tp_eff = model_B_tpeff(sub["tp"].to_numpy(dtype=np.float64), rec["tau_p"])
            transient = 1.0 / (1.0 + np.power(tp_eff / rec["t_c_eff"], rec["r"]))
        else:
            transient = model_A_transient(sub["tp"].to_numpy(dtype=np.float64), rec["t_c_eff"], rec["r"])
        fanh_model = finf / sub["F0"].to_numpy(dtype=np.float64) + sub["F0"].to_numpy(dtype=np.float64) / (np.power(sub["tp"].to_numpy(dtype=np.float64), 1.5) * (1.0 / transient))
        fig, ax = plt.subplots(figsize=(5.8, 4.4))
        ax.plot(sub["tp"], sub["fanh_data"], "ko", ms=3.6, label="data")
        ax.plot(sub["tp"], fanh_model, color="tab:red", lw=1.8, label="fit")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel("fanh")
        ax.set_title(rf"$\theta={theta:.3f}$, {dataset_tag}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"fanh_theta_{theta:.3f}_{dataset_tag}.png", dpi=dpi)
        plt.close(fig)


def make_residual_heatmap(df, rec, out_path, dpi, title):
    theta_vals = np.array(sorted(df["theta"].unique()), dtype=np.float64)
    tp_vals = np.array(sorted(df["tp"].unique()), dtype=np.float64)
    grid = np.full((len(theta_vals), len(tp_vals)), np.nan, dtype=np.float64)
    resid = (df["xi"].to_numpy(dtype=np.float64) - rec["xi_fit"]) / df["xi"].to_numpy(dtype=np.float64)
    for i, th in enumerate(theta_vals):
        for j, tp in enumerate(tp_vals):
            mask = np.isclose(df["theta"], th, rtol=0.0, atol=5.0e-4) & np.isclose(df["tp"], tp, rtol=0.0, atol=1.0e-12)
            if np.any(mask):
                grid[i, j] = float(np.median(resid[mask]))
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    vmax = np.nanmax(np.abs(grid))
    mesh = ax.pcolormesh(tp_vals, np.arange(len(theta_vals) + 1), np.vstack([grid, grid[-1:]]), cmap="coolwarm", shading="auto", vmin=-vmax, vmax=vmax)
    ax.set_xscale("log")
    ax.set_xlabel(r"$t_p$")
    ax.set_ylabel(r"$\theta$")
    ax.set_yticks(np.arange(len(theta_vals)) + 0.5)
    ax.set_yticklabels([f"{th:.3f}" for th in theta_vals])
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, pad=0.02, label=r"$(\xi_{\rm data}-\xi_{\rm model})/\xi_{\rm data}$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_profile_plot(profile_df, out_path, dpi, title):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(profile_df["t_c_eff"], profile_df["rel_rmse"], color="tab:blue", lw=1.8)
    best = profile_df.iloc[int(np.argmin(profile_df["rel_rmse"]))]
    ax.plot(best["t_c_eff"], best["rel_rmse"], "o", color="tab:red", ms=6)
    ax.set_xlabel(r"$t_{c,\rm eff}$")
    ax.set_ylabel("rel-RMSE")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_contour_plot(t_grid, r_grid, Z, out_path, dpi, title):
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    levels = np.linspace(float(np.min(Z)), float(np.percentile(Z, 95.0)), 16)
    cs = ax.contourf(t_grid, r_grid, Z, levels=levels, cmap="viridis")
    ax.contour(t_grid, r_grid, Z, levels=levels[::2], colors="white", linewidths=0.5, alpha=0.55)
    ax.set_xlabel(r"$t_{c,\rm eff}$")
    ax.set_ylabel(r"$r$")
    ax.set_title(title)
    fig.colorbar(cs, ax=ax, pad=0.02, label="rel-RMSE")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def build_fit_summary_text(dataset_tag, fitA, bootA, model_comp):
    lines = []
    lines.append(f"{dataset_tag}: best-fit t_c_eff = {bootA['t_c_eff']['p50']:.4f} [{bootA['t_c_eff']['p16']:.4f}, {bootA['t_c_eff']['p84']:.4f}], "
                 f"r = {bootA['r']['p50']:.4f} [{bootA['r']['p16']:.4f}, {bootA['r']['p84']:.4f}], "
                 f"rel-RMSE = {fitA['rel_rmse']:.4e}.")
    lines.append("Per-theta F_inf(68% CI):")
    for th in fitA["theta_values"]:
        key = f"{float(th):.10f}"
        ci = bootA["F_inf"][key]
        lines.append(
            f"theta={float(th):.4f}: {ci['p50']:.6e} [{ci['p16']:.6e}, {ci['p84']:.6e}]"
        )
    tau_note = "yes" if model_comp["tau_required"] else "no"
    lines.append(
        f"Bootstrap diagnostics used {model_comp['bootstrap_n']} residual resamples. "
        f"Tau required: {tau_note}; delta rel-RMSE = {model_comp['delta_rel_rmse']:.4e}, "
        f"delta AIC = {model_comp['delta_AIC']:.4f}, delta BIC = {model_comp['delta_BIC']:.4f}."
    )
    return "\n".join(lines) + "\n"


def aic_bic_from_rss(rss, n, k):
    rss = max(float(rss), 1.0e-18)
    n = max(int(n), 1)
    aic = n * math.log(rss / n) + 2.0 * k
    bic = n * math.log(rss / n) + k * math.log(n)
    return aic, bic


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rho_path = require_file(args.rho_nopt, "rho_noPT_data.txt")
    lattice_path = require_file(args.lattice_ratio, "xi_lattice_scan_H1p5/H2p0 source")
    ode_xi_path = require_file(args.ode_xi, "xi_ode_scan.txt source")
    ode_nopt_path = require_file(args.ode_nopt, "ode_nopt_reference.txt")

    ode_nopt_df = load_ode_nopt(ode_nopt_path)
    f0_table, _ = reconstruct_f0_table(rho_path, ode_nopt_df, outdir, args.f0_use_all_h)

    df_h15 = merge_f0(load_lattice_scan(lattice_path, 1.5, args.fixed_vw), f0_table)
    df_h20 = merge_f0(load_lattice_scan(lattice_path, 2.0, args.fixed_vw), f0_table)
    df_ode = merge_f0(load_ode_scan(ode_xi_path, args.fixed_vw), f0_table)
    export_adapted_scans(df_h15, df_h20, df_ode, outdir)

    fanh_map = {
        "H1p5": compute_fanh_and_save(df_h15, outdir / "fanh_H1p5.csv"),
        "H2p0": compute_fanh_and_save(df_h20, outdir / "fanh_H2p0.csv"),
        "ODE": compute_fanh_and_save(df_ode, outdir / "fanh_ODE.csv"),
    }

    exponent_map = {
        "H1p5": estimate_smalltp_exponents(fanh_map["H1p5"], "H1p5", outdir / "smalltp_exponents_H1p5.csv", args.huber_fscale, args.smalltp_min_points, args.smalltp_max_points),
        "H2p0": estimate_smalltp_exponents(fanh_map["H2p0"], "H2p0", outdir / "smalltp_exponents_H2p0.csv", args.huber_fscale, args.smalltp_min_points, args.smalltp_max_points),
        "ODE": estimate_smalltp_exponents(fanh_map["ODE"], "ODE", outdir / "smalltp_exponents_ODE.csv", args.huber_fscale, args.smalltp_min_points, args.smalltp_max_points),
    }
    make_smalltp_plot(exponent_map, outdir / "smalltp_exponent_vs_theta.png", args.dpi)

    tail_map = {
        "H1p5": estimate_finf_tail(fanh_map["H1p5"], "H1p5", outdir / "Finf_tail_H1p5.csv", args.huber_fscale, args.tail_frac),
        "H2p0": estimate_finf_tail(fanh_map["H2p0"], "H2p0", outdir / "Finf_tail_H2p0.csv", args.huber_fscale, args.tail_frac),
        "ODE": estimate_finf_tail(fanh_map["ODE"], "ODE", outdir / "Finf_tail_ODE.csv", args.huber_fscale, args.tail_frac),
    }
    make_finf_tail_plot(tail_map, outdir / "Finf_tail_vs_theta.png", args.dpi)

    for tag in ["H1p5", "H2p0"]:
        A_small = estimate_A_small(fanh_map[tag], exponent_map[tag], tag)
        tail_df = tail_map[tag].copy()
        comp = A_small.merge(tail_df[["theta", "F_inf_tail"]], on="theta", how="left")
        comp = comp.merge(f0_table, on="theta", how="left")
        comp["A_tail"] = comp["F_inf_tail"] / np.square(comp["F0"])
        comp.to_csv(outdir / f"A_comparison_{tag}.csv", index=False)
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        mask = np.isfinite(comp["A_small"])
        ax.errorbar(comp.loc[mask, "theta"], comp.loc[mask, "A_small"], yerr=comp.loc[mask, "A_small_err"], marker="o", ms=4.5, lw=1.2, label=r"$A_{\rm small}$")
        ax.plot(comp["theta"], comp["A_tail"], "s--", color="tab:red", ms=4.0, lw=1.4, label=r"$A_{\rm tail}$")
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$A(\theta)$")
        ax.set_title(f"A_small vs A_tail, {tag}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / f"A_small_vs_A_tail_{tag}.png", dpi=args.dpi)
        plt.close(fig)

    rB_payload = {"executed": False, "reason": "local small-tp exponent does not indicate global crossover dominance"}
    combined_small = pd.concat([fanh_map["H1p5"], fanh_map["H2p0"]], ignore_index=True)
    combined_exp = pd.concat([exponent_map["H1p5"], exponent_map["H2p0"]], ignore_index=True)
    if robust_sign_majority(combined_exp):
        small_rows = []
        for theta, group in combined_small.groupby("theta", sort=True):
            small_rows.append(select_smalltp_slice(group, args.smalltp_min_points, args.smalltp_max_points))
        small_df = pd.concat(small_rows, ignore_index=True)
        rB_payload = {"executed": True, **fit_r_B_global(small_df, args.huber_fscale)}
    save_json(outdir / "r_B_estimate.json", rB_payload)

    fit_results_json = {}
    for tag, df in [("H1p5", fanh_map["H1p5"]), ("H2p0", fanh_map["H2p0"])]:
        tail_df = tail_map[tag]
        exp_df = exponent_map[tag]
        priors = {"F_inf": {}, "r": None, "t_c_eff": {"center": 1.5, "sigma": 1.5}}
        for row in tail_df.itertuples(index=False):
            sigma = max(0.30 * row.F_inf_tail, row.err if np.isfinite(row.err) else 0.0, 1.0e-8)
            priors["F_inf"][float(row.theta)] = {"center": float(row.F_inf_tail), "sigma": float(sigma)}
        stable_small = exp_df[np.isfinite(exp_df["p"]) & (exp_df["p"] < 1.4)]
        if len(stable_small) >= max(3, len(exp_df) // 2) and rB_payload.get("executed", False):
            priors["r"] = {"center": float(rB_payload["r"]), "sigma": max(0.30 * float(rB_payload["r"]), float(rB_payload["r_err"]) if np.isfinite(rB_payload["r_err"]) else 0.5)}

        fitA = fit_model_A(df, priors, args.huber_fscale, args.profile_grid_n // 2)
        fitB = fit_model_B(df, priors, args.huber_fscale, max(12, args.profile_grid_n // 3))
        bootA = run_bootstrap(df, fitA, priors, "A", args.bootstrap, args.bootstrap_seed + (15 if tag == "H1p5" else 20), args.bootstrap_jobs, args.huber_fscale, args.profile_grid_n // 2)
        bootA_summary = summarize_bootstrap(fitA, bootA, "A")
        save_json(outdir / f"bootstrap_{tag}.json", bootA_summary)

        jsonA = save_global_fit_json(outdir / f"global_fit_{tag}.json", fitA, tag)
        profile_df = profile_tceff(df, priors, args.huber_fscale, args.profile_grid_n)
        profile_df.to_csv(outdir / f"profile_tceff_{tag}.csv", index=False)
        make_profile_plot(profile_df, outdir / f"profile_tceff_vs_relRMSE_{tag}.png", args.dpi, f"Profile t_c_eff, {tag}")
        t_grid, r_grid, Z = contour_tceff_r(df, priors, max(30, args.profile_grid_n // 2))
        make_contour_plot(t_grid, r_grid, Z, outdir / f"tceff_r_contour_{tag}.png", args.dpi, f"t_c_eff-r contour, {tag}")
        make_residual_heatmap(df, fitA, outdir / f"residual_heatmap_{tag}.png", args.dpi, f"Residual heatmap, {tag}")
        make_fanh_theta_plots(df, fitA, tag, outdir, args.dpi)

        n = int(len(df))
        kA = 2 + len(fitA["theta_values"])
        kB = 3 + len(fitB["theta_values"])
        aicA, bicA = aic_bic_from_rss(fitA["rss_frac"], n, kA)
        aicB, bicB = aic_bic_from_rss(fitB["rss_frac"], n, kB)
        model_comp = {
            "dataset": tag,
            "ModelA": {
                "t_c_eff": fitA["t_c_eff"],
                "r": fitA["r"],
                "rel_rmse": fitA["rel_rmse"],
                "AIC": aicA,
                "BIC": bicA,
            },
            "ModelB": {
                "t_c_eff": fitB["t_c_eff"],
                "r": fitB["r"],
                "tau_p": fitB["tau_p"],
                "rel_rmse": fitB["rel_rmse"],
                "AIC": aicB,
                "BIC": bicB,
            },
            "delta_rel_rmse": float(fitB["rel_rmse"] - fitA["rel_rmse"]),
            "delta_AIC": float(aicB - aicA),
            "delta_BIC": float(bicB - bicA),
            "tau_required": bool((fitB["rel_rmse"] < 0.95 * fitA["rel_rmse"]) and (aicB < aicA - 2.0) and (bicB < bicA - 2.0)),
            "bootstrap_n": int(args.bootstrap),
        }
        save_json(outdir / f"model_comparison_{tag}.json", model_comp)

        summary_text = build_fit_summary_text(tag, fitA, bootA_summary, model_comp)
        (outdir / f"fit_summary_{tag}.txt").write_text(summary_text)

        fit_results_json[tag] = {
            "global_fit": jsonA,
            "bootstrap": bootA_summary,
            "model_comparison": model_comp,
        }

    combined_df = pd.concat([fanh_map["H1p5"], fanh_map["H2p0"]], ignore_index=True)
    combined_tail = pd.concat([tail_map["H1p5"], tail_map["H2p0"]], ignore_index=True)
    priors_combined = {"F_inf": {}, "r": None, "t_c_eff": {"center": 1.5, "sigma": 1.5}}
    for theta, sub in combined_tail.groupby("theta", sort=True):
        center = float(np.mean(sub["F_inf_tail"]))
        sigma = max(0.30 * center, float(np.mean(np.maximum(sub["err"].fillna(0.0), 0.0))), 1.0e-8)
        priors_combined["F_inf"][float(theta)] = {"center": center, "sigma": sigma}
    fit_combined = fit_model_A(combined_df, priors_combined, args.huber_fscale, args.profile_grid_n // 2)
    save_global_fit_json(outdir / "global_fit_combined.json", fit_combined, "combined_H1p5_H2p0")
    fit_results_json["combined"] = {
        "t_c_eff": fit_combined["t_c_eff"],
        "r": fit_combined["r"],
        "rel_rmse": fit_combined["rel_rmse"],
    }

    print(json.dumps(to_native(fit_results_json), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
