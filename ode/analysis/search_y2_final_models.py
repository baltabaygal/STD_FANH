import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def parse_args():
    p = argparse.ArgumentParser(
        description="Search compact Y2(theta0,tp) models built from per-theta plateau-plus-powerlaw fits."
    )
    p.add_argument("--data", type=str, default="ode/analysis/data/dm_tp_fitready_H1p000.txt")
    p.add_argument("--outdir", type=str, default="ode/analysis/results")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--top", type=int, default=12)
    return p.parse_args()


def y2_model(tp, c0, c1, p):
    return c0 + c1 / np.power(tp, p)


def rel_rmse(y, yfit):
    return float(np.sqrt(np.mean(np.square((yfit - y) / y))))


def log_rmse(y, yfit):
    return float(np.sqrt(np.mean(np.square(np.log(yfit) - np.log(y)))))


def fit_per_theta(theta0, tp, y2):
    rows = []
    for th0 in np.unique(theta0):
        mask = np.isclose(theta0, th0) & np.isfinite(tp) & np.isfinite(y2) & (tp > 0) & (y2 > 0)
        x = tp[mask]
        y = y2[mask]
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

        p0 = [y[-1], max(y[0] - y[-1], 1e-6), 1.0]
        bounds = ([0.0, 0.0, 0.05], [10.0, 1e4, 8.0])
        popt, _ = curve_fit(y2_model, x, y, p0=p0, bounds=bounds, maxfev=200000)
        yfit = y2_model(x, *popt)
        rows.append((th0, popt[0], popt[1], popt[2], log_rmse(y, yfit), rel_rmse(y, yfit)))
    return np.array(rows, dtype=np.float64)


def feature_map(theta0):
    u = theta0 / np.pi
    s2 = np.sin(theta0 / 2.0) ** 2
    one_minus_u2 = np.clip(1.0 - u * u, 1e-12, None)
    cos2 = np.clip(np.cos(theta0 / 2.0) ** 2, 1e-12, None)
    h_u = np.log(np.e / one_minus_u2)
    h_p = np.log(np.e / cos2)
    return {
        "u": u,
        "u2": u * u,
        "s2": s2,
        "s4": s2 * s2,
        "h_u": h_u,
        "h_p": h_p,
        "inv_u": 1.0 / one_minus_u2,
        "inv_cos": 1.0 / cos2,
    }


@dataclass
class Family:
    name: str
    n_params: int
    fit_kind: str


def poly_eval(x, coeffs):
    y = np.zeros_like(np.asarray(x, dtype=np.float64), dtype=np.float64)
    for c in coeffs:
        y = y * x + c
    return y


def family_eval(family_name, theta0, params, c0_eval=None):
    feat = feature_map(theta0)
    if family_name == "poly2_u":
        return poly_eval(feat["u"], params)
    if family_name == "poly1_u":
        return poly_eval(feat["u"], params)
    if family_name == "poly1_s2":
        return poly_eval(feat["s2"], params)
    if family_name == "poly2_s2":
        return poly_eval(feat["s2"], params)
    if family_name == "lin_h_p":
        a0, a1 = params
        return a0 + a1 * feat["h_p"]
    if family_name == "lin_h_u":
        a0, a1 = params
        return a0 + a1 * feat["h_u"]
    if family_name == "pow_h_p":
        amp, alpha = params
        return amp * np.power(feat["h_p"], alpha)
    if family_name == "pow_h_u":
        amp, alpha = params
        return amp * np.power(feat["h_u"], alpha)
    if family_name == "pow_inv_cos":
        amp, alpha = params
        return amp * np.power(feat["inv_cos"], alpha)
    if family_name == "pow_inv_u":
        amp, alpha = params
        return amp * np.power(feat["inv_u"], alpha)
    if family_name == "const":
        return np.full_like(theta0, params[0], dtype=np.float64)
    if family_name == "lin_u":
        return poly_eval(feat["u"], params)
    if family_name == "lin_s2":
        return poly_eval(feat["s2"], params)
    if family_name == "lin_h_p_p":
        return poly_eval(feat["h_p"], params)
    if family_name == "lin_h_u_p":
        return poly_eval(feat["h_u"], params)
    if family_name == "lin_c0":
        return poly_eval(c0_eval, params)
    if family_name == "quad_c0":
        return poly_eval(c0_eval, params)
    if family_name == "lin_h_p_logc1":
        return poly_eval(feat["h_p"], params)
    if family_name == "quad_h_p_logc1":
        return poly_eval(feat["h_p"], params)
    if family_name == "lin_h_u_logc1":
        return poly_eval(feat["h_u"], params)
    if family_name == "quad_h_u_logc1":
        return poly_eval(feat["h_u"], params)
    if family_name == "lin_s2_logc1":
        return poly_eval(feat["s2"], params)
    if family_name == "quad_s2_logc1":
        return poly_eval(feat["s2"], params)
    raise ValueError(f"Unknown family: {family_name}")


def fit_family(theta0, y, family_name):
    feat = feature_map(theta0)
    if family_name == "poly2_u":
        return np.polyfit(feat["u"], y, 2)
    if family_name == "poly1_u":
        return np.polyfit(feat["u"], y, 1)
    if family_name == "poly1_s2":
        return np.polyfit(feat["s2"], y, 1)
    if family_name == "poly2_s2":
        return np.polyfit(feat["s2"], y, 2)
    if family_name == "lin_h_p":
        return np.polyfit(feat["h_p"], y, 1)
    if family_name == "lin_h_u":
        return np.polyfit(feat["h_u"], y, 1)
    if family_name == "pow_h_p":
        popt, _ = curve_fit(
            lambda th, amp, alpha: family_eval("pow_h_p", th, [amp, alpha]),
            theta0,
            y,
            p0=[np.median(y), 0.2],
            bounds=([1e-12, 0.0], [10.0, 5.0]),
            maxfev=200000,
        )
        return popt
    if family_name == "pow_h_u":
        popt, _ = curve_fit(
            lambda th, amp, alpha: family_eval("pow_h_u", th, [amp, alpha]),
            theta0,
            y,
            p0=[np.median(y), 0.2],
            bounds=([1e-12, 0.0], [10.0, 5.0]),
            maxfev=200000,
        )
        return popt
    if family_name == "pow_inv_cos":
        popt, _ = curve_fit(
            lambda th, amp, alpha: family_eval("pow_inv_cos", th, [amp, alpha]),
            theta0,
            y,
            p0=[np.median(y), 0.1],
            bounds=([1e-12, 0.0], [10.0, 3.0]),
            maxfev=200000,
        )
        return popt
    if family_name == "pow_inv_u":
        popt, _ = curve_fit(
            lambda th, amp, alpha: family_eval("pow_inv_u", th, [amp, alpha]),
            theta0,
            y,
            p0=[np.median(y), 0.1],
            bounds=([1e-12, 0.0], [10.0, 3.0]),
            maxfev=200000,
        )
        return popt
    if family_name == "const":
        return np.array([np.mean(y)], dtype=np.float64)
    if family_name == "lin_u":
        return np.polyfit(feat["u"], y, 1)
    if family_name == "lin_s2":
        return np.polyfit(feat["s2"], y, 1)
    if family_name == "lin_h_p_p":
        return np.polyfit(feat["h_p"], y, 1)
    if family_name == "lin_h_u_p":
        return np.polyfit(feat["h_u"], y, 1)
    if family_name == "lin_h_p_logc1":
        return np.polyfit(feat["h_p"], y, 1)
    if family_name == "quad_h_p_logc1":
        return np.polyfit(feat["h_p"], y, 2)
    if family_name == "lin_h_u_logc1":
        return np.polyfit(feat["h_u"], y, 1)
    if family_name == "quad_h_u_logc1":
        return np.polyfit(feat["h_u"], y, 2)
    if family_name == "lin_s2_logc1":
        return np.polyfit(feat["s2"], y, 1)
    if family_name == "quad_s2_logc1":
        return np.polyfit(feat["s2"], y, 2)
    raise ValueError(f"Unknown family: {family_name}")


def fit_c0_related(y, x, degree):
    return np.polyfit(x, y, degree)


def candidate_spaces():
    c0_fams = [
        Family("poly2_u", 3, "direct"),
        Family("pow_h_p", 2, "direct"),
        Family("pow_h_u", 2, "direct"),
        Family("pow_inv_cos", 2, "direct"),
        Family("pow_inv_u", 2, "direct"),
        Family("lin_h_p", 2, "direct"),
        Family("lin_h_u", 2, "direct"),
        Family("poly2_s2", 3, "direct"),
        Family("poly1_s2", 2, "direct"),
    ]
    p_fams = [
        Family("lin_u", 2, "direct"),
        Family("lin_s2", 2, "direct"),
        Family("lin_h_p_p", 2, "direct"),
        Family("lin_h_u_p", 2, "direct"),
        Family("const", 1, "direct"),
        Family("lin_c0", 2, "c0"),
    ]
    logc1_fams = [
        Family("quad_c0", 3, "c0"),
        Family("lin_c0", 2, "c0"),
        Family("lin_h_p_logc1", 2, "direct"),
        Family("quad_h_p_logc1", 3, "direct"),
        Family("lin_h_u_logc1", 2, "direct"),
        Family("quad_h_u_logc1", 3, "direct"),
        Family("lin_s2_logc1", 2, "direct"),
        Family("quad_s2_logc1", 3, "direct"),
    ]
    return c0_fams, p_fams, logc1_fams


def evaluate_combo(theta_rows, c0_rows, c1_rows, p_rows, theta_all, tp_all, y_all, c0_fam, logc1_fam, p_fam):
    c0_params = fit_family(theta_rows, c0_rows, c0_fam.name)
    c0_rows_eval = family_eval(c0_fam.name, theta_rows, c0_params)

    if logc1_fam.fit_kind == "c0":
        degree = 2 if logc1_fam.name == "quad_c0" else 1
        logc1_params = fit_c0_related(np.log(c1_rows), c0_rows_eval, degree)
        logc1_rows_eval = poly_eval(c0_rows_eval, logc1_params)
    else:
        logc1_params = fit_family(theta_rows, np.log(c1_rows), logc1_fam.name)
        logc1_rows_eval = family_eval(logc1_fam.name, theta_rows, logc1_params)

    if p_fam.fit_kind == "c0":
        p_params = fit_c0_related(p_rows, c0_rows_eval, 1)
        p_rows_eval = poly_eval(c0_rows_eval, p_params)
    else:
        p_params = fit_family(theta_rows, p_rows, p_fam.name)
        p_rows_eval = family_eval(p_fam.name, theta_rows, p_params)

    c0_all = family_eval(c0_fam.name, theta_all, c0_params)
    if logc1_fam.fit_kind == "c0":
        logc1_all = poly_eval(c0_all, logc1_params)
    else:
        logc1_all = family_eval(logc1_fam.name, theta_all, logc1_params)
    c1_all = np.exp(logc1_all)
    if p_fam.fit_kind == "c0":
        p_all = poly_eval(c0_all, p_params)
    else:
        p_all = family_eval(p_fam.name, theta_all, p_params)

    yfit = y2_model(tp_all, c0_all, c1_all, p_all)
    entry = {
        "name": f"c0:{c0_fam.name} | logc1:{logc1_fam.name} | p:{p_fam.name}",
        "c0_family": c0_fam.name,
        "logc1_family": logc1_fam.name,
        "p_family": p_fam.name,
        "c0_params": np.asarray(c0_params, dtype=np.float64),
        "logc1_params": np.asarray(logc1_params, dtype=np.float64),
        "p_params": np.asarray(p_params, dtype=np.float64),
        "n_params": c0_fam.n_params + logc1_fam.n_params + p_fam.n_params,
        "global_log_rmse": log_rmse(y_all, yfit),
        "global_rel_rmse": rel_rmse(y_all, yfit),
        "coeff_rel_c0": rel_rmse(c0_rows, c0_rows_eval),
        "coeff_rel_c1": rel_rmse(np.log(c1_rows), logc1_rows_eval),
        "coeff_rel_p": rel_rmse(p_rows, p_rows_eval),
        "yfit": yfit,
    }
    return entry


def save_summary(path, results):
    with open(path, "w") as f:
        f.write("# name | n_params | global_log_rmse | global_rel_rmse | c0_rel | logc1_rel | p_rel\n")
        for r in results:
            f.write(
                f"{r['name']} | {r['n_params']} | {r['global_log_rmse']:.10e} | {r['global_rel_rmse']:.10e} | "
                f"{r['coeff_rel_c0']:.10e} | {r['coeff_rel_c1']:.10e} | {r['coeff_rel_p']:.10e}\n"
            )
            f.write("# c0_params " + " ".join(f"{x:.10e}" for x in r["c0_params"]) + "\n")
            f.write("# logc1_params " + " ".join(f"{x:.10e}" for x in r["logc1_params"]) + "\n")
            f.write("# p_params " + " ".join(f"{x:.10e}" for x in r["p_params"]) + "\n")


def plot_top_theta_slices(theta0, tp, y2, results, out_path, dpi):
    pick_models = results[:4]
    pick_tp = np.unique(tp)[np.linspace(0, len(np.unique(tp)) - 1, 6).astype(int)]
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.2), squeeze=False)
    axes = axes.ravel()
    for ax, tpi in zip(axes, pick_tp):
        mask = np.isclose(tp, tpi)
        idx = np.argsort(theta0[mask])
        th = theta0[mask][idx]
        yd = y2[mask][idx]
        ax.plot(th, yd, "ko", ms=4, label="data")
        for r in pick_models:
            yf = r["yfit"][mask][idx]
            ax.plot(th, yf, lw=1.6, label=f"{r['n_params']}p {r['c0_family']}/{r['logc1_family']}/{r['p_family']}")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\theta_0$")
        ax.set_ylabel(r"$Y_2$")
        ax.set_title(rf"$t_p={tpi:.3g}$")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_complexity_tradeoff(results, out_path, dpi):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    xs = np.array([r["n_params"] for r in results], dtype=np.float64)
    ys = np.array([r["global_rel_rmse"] for r in results], dtype=np.float64)
    ax.plot(xs, ys, "ko", ms=4, alpha=0.75)
    for r in results[:10]:
        ax.annotate(
            f"{r['c0_family']}\n{r['logc1_family']}\n{r['p_family']}",
            (r["n_params"], r["global_rel_rmse"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
        )
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("Global relative RMSE")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_best_params(theta_rows, rows, result, out_path, dpi):
    th_dense = np.linspace(theta_rows.min(), theta_rows.max(), 400)
    c0_dense = family_eval(result["c0_family"], th_dense, result["c0_params"])
    if result["logc1_family"] == "quad_c0":
        logc1_dense = poly_eval(c0_dense, result["logc1_params"])
    elif result["logc1_family"] == "lin_c0":
        logc1_dense = poly_eval(c0_dense, result["logc1_params"])
    else:
        logc1_dense = family_eval(result["logc1_family"], th_dense, result["logc1_params"])
    c1_dense = np.exp(logc1_dense)
    if result["p_family"] == "lin_c0":
        p_dense = poly_eval(c0_dense, result["p_params"])
    else:
        p_dense = family_eval(result["p_family"], th_dense, result["p_params"])

    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.0))
    axes[0].plot(theta_rows, rows[:, 1], "ko", ms=5)
    axes[0].plot(th_dense, c0_dense, "-", lw=1.8)
    axes[0].set_xlabel(r"$\theta_0$")
    axes[0].set_ylabel(r"$c_0(\theta_0)$")
    axes[0].grid(alpha=0.25)

    axes[1].plot(theta_rows, rows[:, 2], "ko", ms=5)
    axes[1].plot(th_dense, c1_dense, "-", lw=1.8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\theta_0$")
    axes[1].set_ylabel(r"$c_1(\theta_0)$")
    axes[1].grid(alpha=0.25)

    axes[2].plot(theta_rows, rows[:, 3], "ko", ms=5)
    axes[2].plot(th_dense, p_dense, "-", lw=1.8)
    axes[2].set_xlabel(r"$\theta_0$")
    axes[2].set_ylabel(r"$p(\theta_0)$")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    data_path = Path(args.data).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    arr = np.loadtxt(data_path, comments="#")
    theta0 = arr[:, 2]
    tp = arr[:, 3]
    fanh_no = arr[:, 8]
    xi = arr[:, 9]
    y2 = xi * fanh_no / np.power(tp, 1.5)

    rows = fit_per_theta(theta0, tp, y2)
    theta_rows = rows[:, 0]
    c0_rows = rows[:, 1]
    c1_rows = rows[:, 2]
    p_rows = rows[:, 3]

    c0_fams, p_fams, logc1_fams = candidate_spaces()
    results = []
    for c0_fam in c0_fams:
        for logc1_fam in logc1_fams:
            for p_fam in p_fams:
                try:
                    results.append(
                        evaluate_combo(
                            theta_rows,
                            c0_rows,
                            c1_rows,
                            p_rows,
                            theta0,
                            tp,
                            y2,
                            c0_fam,
                            logc1_fam,
                            p_fam,
                        )
                    )
                except Exception:
                    continue

    results.sort(key=lambda r: (r["global_rel_rmse"], r["n_params"], r["global_log_rmse"]))
    best = results[: args.top]

    stem = data_path.stem
    save_summary(outdir / f"search_y2_final_models_{stem}.txt", best)
    plot_top_theta_slices(
        theta0,
        tp,
        y2,
        best,
        outdir / f"search_y2_final_models_vs_theta_{stem}.png",
        args.dpi,
    )
    plot_complexity_tradeoff(
        results,
        outdir / f"search_y2_final_models_tradeoff_{stem}.png",
        args.dpi,
    )
    plot_best_params(
        theta_rows,
        rows,
        best[0],
        outdir / f"search_y2_final_models_best_params_{stem}.png",
        args.dpi,
    )

    print(f"Loaded: {data_path}")
    print(f"Saved: {outdir / f'search_y2_final_models_{stem}.txt'}")
    print(f"Saved: {outdir / f'search_y2_final_models_vs_theta_{stem}.png'}")
    print(f"Saved: {outdir / f'search_y2_final_models_tradeoff_{stem}.png'}")
    print(f"Saved: {outdir / f'search_y2_final_models_best_params_{stem}.png'}")
    print("Top models:")
    for r in best[:8]:
        print(
            f"{r['n_params']}p | rel={r['global_rel_rmse']:.4e} | "
            f"{r['c0_family']} / {r['logc1_family']} / {r['p_family']}"
        )


if __name__ == "__main__":
    main()
