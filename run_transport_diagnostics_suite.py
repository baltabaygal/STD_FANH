import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parent
TOSC = 1.5
PRED_PATH = ROOT / "results_vw0p9_model_applied_all_vw" / "predictions.csv"
BASELINE_PATH = ROOT / "results_tosc_lattice_vw0p9_H1p0H1p5H2p0" / "collapse_and_fit_fanh" / "global_fit.json"
WARP_PATH = ROOT / "results_vw_overlay_H1p0H1p5H2p0_quadwarped_beta0" / "lattice_quadwarp_summary.json"
ODE_DATA = {
    1.0: ROOT / "ode" / "analysis" / "data" / "dm_tp_fitready_H1p000.txt",
    1.5: ROOT / "ode" / "analysis" / "data" / "dm_tp_fitready_H1p500.txt",
    2.0: ROOT / "ode" / "analysis" / "data" / "dm_tp_fitready_H2p000.txt",
}
OUTDIR = ROOT / "results_transport_diagnostics_suite"


def warp_tp(tp, log_s, b, c):
    tp = np.asarray(tp, dtype=np.float64)
    ltp = np.log(np.maximum(tp, 1.0e-18))
    return np.exp(float(log_s) + float(b) * ltp + float(c) * ltp * ltp)


def closest_key(theta, keys):
    vals = np.array([float(k) for k in keys], dtype=np.float64)
    idx = int(np.argmin(np.abs(vals - float(theta))))
    return list(keys)[idx]


def baseline_xi(x, f0, finf, tc, r):
    x = np.asarray(x, dtype=np.float64)
    return np.power(np.maximum(x / TOSC, 1.0e-18), 1.5) * (finf / np.maximum(f0, 1.0e-18) ** 2) + 1.0 / (
        1.0 + np.power(np.maximum(x / tc, 1.0e-18), r)
    )


def log_bin_centers(edges):
    return np.sqrt(edges[:-1] * edges[1:])


def binned_mean_frame(df, x_col, y_col, group_col, bins):
    work = df.copy()
    work["bin"] = pd.cut(work[x_col], bins=bins, include_lowest=True, right=True)
    rows = []
    for (grp, bin_key), sub in work.groupby([group_col, "bin"], observed=True):
        if len(sub) == 0:
            continue
        rows.append(
            {
                group_col: grp,
                "bin_left": float(bin_key.left),
                "bin_right": float(bin_key.right),
                "x_center": float(math.sqrt(bin_key.left * bin_key.right)),
                "y_mean": float(sub[y_col].mean()),
                "y_std": float(sub[y_col].std(ddof=0)),
                "n": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def load_ode_curves():
    curves = {}
    cols = [
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
    for H, path in ODE_DATA.items():
        df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=cols)
        curves[H] = {}
        for theta, sub in df.groupby("theta0"):
            s = sub.sort_values("t_p")
            curves[H][float(theta)] = {
                "tp": s["t_p"].to_numpy(dtype=np.float64),
                "xi": s["xi_DM"].to_numpy(dtype=np.float64),
            }
    return curves


def interp_log(tp_grid, y_grid, tp):
    lgrid = np.log(np.maximum(tp_grid, 1.0e-18))
    ltp = np.log(np.maximum(np.asarray(tp, dtype=np.float64), 1.0e-18))
    return np.exp(np.interp(ltp, lgrid, np.log(np.maximum(y_grid, 1.0e-18))))


def jensen_factor(tp_grid, y_grid, center_tp, sigma):
    center_tp = np.asarray(center_tp, dtype=np.float64)
    if sigma <= 1.0e-8:
        return np.ones_like(center_tp)
    z = np.linspace(-4.0, 4.0, 121)
    w = np.exp(-0.5 * z * z)
    w = w / np.sum(w)
    base = interp_log(tp_grid, y_grid, center_tp)
    out = np.empty_like(center_tp)
    for i, tp0 in enumerate(center_tp):
        tau = tp0 * np.exp(float(sigma) * z)
        ys = interp_log(tp_grid, y_grid, tau)
        out[i] = float(np.sum(w * ys) / max(base[i], 1.0e-18))
    return out


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(PRED_PATH)
    base = json.load(open(BASELINE_PATH))
    warp = json.load(open(WARP_PATH))
    ode_curves = load_ode_curves()

    finf_map = {float(k): float(v["value"]) for k, v in base["F_inf"].items()}
    beta = float(base["beta"])
    tc = float(base["t_c"])
    r = float(base["r"])

    def finf(theta):
        key = min(finf_map, key=lambda t: abs(t - float(theta)))
        return finf_map[key]

    teff = []
    x_eff = []
    xi_teff = []
    for _, row in pred.iterrows():
        rec = warp["params"][f"{float(row['v_w']):.1f}"]
        t_eff = warp_tp(float(row["tp"]), rec["log_s"], rec["b"], rec["c"])
        x = float(t_eff * np.power(float(row["H"]), beta))
        model = float(baseline_xi(x, float(row["F0"]), finf(float(row["theta"])), tc, r))
        teff.append(float(t_eff))
        x_eff.append(x)
        xi_teff.append(model)

    pred["t_eff"] = np.asarray(teff, dtype=np.float64)
    pred["x_eff"] = np.asarray(x_eff, dtype=np.float64)
    pred["xi_teff_model"] = np.asarray(xi_teff, dtype=np.float64)
    pred["ratio_teff"] = pred["xi"] / np.maximum(pred["xi_teff_model"], 1.0e-18)

    pred.to_csv(OUTDIR / "transport_diagnostics_table.csv", index=False)

    # Test 1: ratio vs vw at fixed tp bins
    tp_bins = np.geomspace(pred["tp"].min(), pred["tp"].max(), 7)
    tp_vw = binned_mean_frame(pred, "tp", "ratio_teff", "v_w", tp_bins)
    tp_vw.to_csv(OUTDIR / "ratio_vs_vw_tp_bins.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for x_center, sub in tp_vw.groupby("x_center", sort=True):
        sub = sub.sort_values("v_w")
        ax.plot(sub["v_w"], sub["y_mean"], marker="o", lw=1.8, label=fr"$t_p\sim {x_center:.2f}$")
    ax.axhline(1.0, color="black", lw=1.0, alpha=0.5)
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$\xi_{\rm data}/\xi_{v9}(t_{\rm eff})$")
    ax.set_title(r"Timing-corrected residual vs $v_w$ in $t_p$ bins")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTDIR / "ratio_teff_vs_vw_tp_bins.png", dpi=180)
    plt.close(fig)

    # Tests 2 and 3: ratio vs tp and tp-independence metrics
    ratio_tp = binned_mean_frame(pred, "tp", "ratio_teff", "v_w", tp_bins)
    ratio_tp.to_csv(OUTDIR / "ratio_vs_tp_by_vw.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for vw, sub in ratio_tp.groupby("v_w", sort=True):
        sub = sub.sort_values("x_center")
        ax.plot(sub["x_center"], sub["y_mean"], marker="o", lw=1.8, label=fr"$v_w={vw:.1f}$")
    ax.axhline(1.0, color="black", lw=1.0, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$t_p$")
    ax.set_ylabel(r"$\xi_{\rm data}/\xi_{v9}(t_{\rm eff})$")
    ax.set_title(r"Timing-corrected residual vs $t_p$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "ratio_teff_vs_tp_by_vw.png", dpi=180)
    plt.close(fig)

    dep_rows = []
    for vw, sub in pred.groupby("v_w", sort=True):
        s = sub.sort_values("tp").copy()
        q1 = float(s["tp"].quantile(1.0 / 3.0))
        q2 = float(s["tp"].quantile(2.0 / 3.0))
        low = float(s.loc[s["tp"] <= q1, "ratio_teff"].mean())
        high = float(s.loc[s["tp"] >= q2, "ratio_teff"].mean())
        X = np.log(s["tp"].to_numpy(dtype=np.float64)).reshape(-1, 1)
        y = s["ratio_teff"].to_numpy(dtype=np.float64)
        reg = LinearRegression().fit(X, y)
        y_lin = reg.predict(X)
        y_const = np.full_like(y, float(np.mean(y)))
        dep_rows.append(
            {
                "v_w": float(vw),
                "low_tp_mean_ratio": low,
                "high_tp_mean_ratio": high,
                "low_minus_high": low - high,
                "slope_vs_logtp": float(reg.coef_[0]),
                "intercept": float(reg.intercept_),
                "corr_ratio_logtp": float(np.corrcoef(np.log(s["tp"]), y)[0, 1]),
                "rmse_const": float(np.sqrt(np.mean((y_const - y) ** 2))),
                "rmse_linear": float(np.sqrt(np.mean((y_lin - y) ** 2))),
            }
        )
    dep_df = pd.DataFrame(dep_rows).sort_values("v_w")
    dep_df.to_csv(OUTDIR / "tp_dependence_summary.csv", index=False)

    # Test 4: direct ODE Jensen-smearing on teff residual
    jensen_rows = []
    pred["jensen_factor"] = 1.0
    sigma_summary = []
    for vw, sub in pred.groupby("v_w", sort=True):
        if float(vw) >= 0.9 - 1.0e-12:
            sigma = 0.0
        else:
            idx = sub.index.to_numpy()

            def objective(sigma_):
                facs = []
                for i in idx:
                    row = pred.loc[i]
                    H = float(row["H"])
                    theta = min(ode_curves[H], key=lambda t: abs(t - float(row["theta"])))
                    curve = ode_curves[H][theta]
                    fac = jensen_factor(curve["tp"], curve["xi"], np.array([float(row["t_eff"])]), float(sigma_))[0]
                    facs.append(fac)
                facs = np.asarray(facs, dtype=np.float64)
                target = pred.loc[idx, "ratio_teff"].to_numpy(dtype=np.float64)
                return float(np.mean((facs - target) ** 2))

            res = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
            sigma = float(res.x)

        idx = sub.index.to_numpy()
        facs = []
        for i in idx:
            row = pred.loc[i]
            H = float(row["H"])
            theta = min(ode_curves[H], key=lambda t: abs(t - float(row["theta"])))
            curve = ode_curves[H][theta]
            fac = jensen_factor(curve["tp"], curve["xi"], np.array([float(row["t_eff"])]), sigma)[0]
            facs.append(fac)
        facs = np.asarray(facs, dtype=np.float64)
        pred.loc[idx, "jensen_factor"] = facs
        target = pred.loc[idx, "ratio_teff"].to_numpy(dtype=np.float64)
        sigma_summary.append(
            {
                "v_w": float(vw),
                "sigma_logtp": float(sigma),
                "rmse_vs_one": float(np.sqrt(np.mean((target - 1.0) ** 2))),
                "rmse_vs_jensen": float(np.sqrt(np.mean((target - facs) ** 2))),
                "mean_ratio_teff": float(np.mean(target)),
                "mean_jensen_factor": float(np.mean(facs)),
            }
        )
        for i, fac in zip(idx, facs):
            jensen_rows.append(
                {
                    "row_index": int(i),
                    "v_w": float(vw),
                    "jensen_factor": float(fac),
                    "ratio_teff": float(pred.loc[i, "ratio_teff"]),
                    "t_eff": float(pred.loc[i, "t_eff"]),
                    "tp": float(pred.loc[i, "tp"]),
                    "H": float(pred.loc[i, "H"]),
                    "theta": float(pred.loc[i, "theta"]),
                    "beta_over_H": float(pred.loc[i, "beta_over_H"]),
                }
            )

    sigma_df = pd.DataFrame(sigma_summary).sort_values("v_w")
    sigma_df.to_csv(OUTDIR / "jensen_sigma_fit_summary.csv", index=False)
    pd.DataFrame(jensen_rows).to_csv(OUTDIR / "jensen_row_predictions.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, vw in zip(axes, sorted(pred["v_w"].unique())):
        sub = pred[np.isclose(pred["v_w"], vw)].copy()
        ax.scatter(sub["ratio_teff"], sub["jensen_factor"], s=14, alpha=0.5)
        lo = float(min(sub["ratio_teff"].min(), sub["jensen_factor"].min()))
        hi = float(max(sub["ratio_teff"].max(), sub["jensen_factor"].max()))
        ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, alpha=0.7)
        ax.set_title(rf"$v_w={vw:.1f}$")
        ax.set_xlabel(r"observed $\xi/\xi_{v9}(t_{\rm eff})$")
        ax.set_ylabel("ODE Jensen factor")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "jensen_pred_vs_obs_by_vw.png", dpi=180)
    plt.close(fig)

    teff_bins = np.geomspace(pred["t_eff"].min(), pred["t_eff"].max(), 7)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, vw in zip(axes, sorted(pred["v_w"].unique())):
        sub = pred[np.isclose(pred["v_w"], vw)].copy()
        obs = binned_mean_frame(sub, "t_eff", "ratio_teff", "v_w", teff_bins)
        jn = binned_mean_frame(sub.assign(y=sub["jensen_factor"]), "t_eff", "jensen_factor", "v_w", teff_bins)
        if not obs.empty:
            ax.plot(obs["x_center"], obs["y_mean"], marker="o", lw=1.8, label="observed")
        if not jn.empty:
            ax.plot(jn["x_center"], jn["y_mean"], marker="s", lw=1.8, label="Jensen")
        ax.axhline(1.0, color="black", lw=1.0, alpha=0.5)
        ax.set_xscale("log")
        ax.set_title(rf"$v_w={vw:.1f}$")
        ax.set_xlabel(r"$t_{\rm eff}$")
        ax.set_ylabel(r"$\xi/\xi_{v9}(t_{\rm eff})$")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "ratio_vs_teff_observed_vs_jensen.png", dpi=180)
    plt.close(fig)

    summary = {
        "inputs": {
            "predictions": str(PRED_PATH),
            "baseline_model": str(BASELINE_PATH),
            "warp_params": str(WARP_PATH),
            "ode_data": {str(k): str(v) for k, v in ODE_DATA.items()},
        },
        "baseline": {
            "beta": beta,
            "t_c": tc,
            "r": r,
        },
        "aggregate": {
            "mean_abs_ratio_teff_minus1": float(np.mean(np.abs(pred["ratio_teff"] - 1.0))),
            "median_abs_ratio_teff_minus1": float(np.median(np.abs(pred["ratio_teff"] - 1.0))),
            "std_ratio_teff": float(np.std(pred["ratio_teff"], ddof=0)),
        },
        "outputs": {
            "joined_table": str(OUTDIR / "transport_diagnostics_table.csv"),
            "ratio_vs_vw_tp_bins": str(OUTDIR / "ratio_teff_vs_vw_tp_bins.png"),
            "ratio_vs_tp_by_vw": str(OUTDIR / "ratio_teff_vs_tp_by_vw.png"),
            "tp_dependence_summary": str(OUTDIR / "tp_dependence_summary.csv"),
            "jensen_sigma_summary": str(OUTDIR / "jensen_sigma_fit_summary.csv"),
            "jensen_pred_vs_obs": str(OUTDIR / "jensen_pred_vs_obs_by_vw.png"),
            "jensen_vs_teff": str(OUTDIR / "ratio_vs_teff_observed_vs_jensen.png"),
        },
    }
    with open(OUTDIR / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
