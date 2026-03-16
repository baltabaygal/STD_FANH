import argparse
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Fit the asymptotic xi model "
            "xi = t_p^(3/2)/fanh_noPT(theta)^2 * (fanh_infty(theta) + c fanh_noPT(theta)^2 / t_p^q)."
        )
    )
    p.add_argument("--xi-data", type=str, default="ode/analysis/results/xi_dense_bridge_H1p000.txt")
    p.add_argument("--reference-data", type=str, default="ode/analysis/data/dm_tp_fitready_H1p000.txt")
    p.add_argument("--outdir", type=str, default="ode/analysis/results")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--tp-max-smallfit", type=float, default=1.0)
    return p.parse_args()


def h_of_theta(theta0):
    theta0 = np.asarray(theta0, dtype=np.float64)
    return np.log(np.e / np.clip(np.cos(theta0 / 2.0) ** 2, 1e-12, None))


def rel_rmse(y, yfit):
    return float(np.sqrt(np.mean(np.square((yfit - y) / y))))


def log_rmse(y, yfit):
    return float(np.sqrt(np.mean(np.square(np.log(yfit) - np.log(y)))))


def fit_powerlaw_h(h, y):
    coeff = np.polyfit(np.log(h), np.log(y), 1)
    gamma = float(coeff[0])
    amp = float(np.exp(coeff[1]))
    yfit = amp * np.power(h, gamma)
    return {
        "amp": amp,
        "gamma": gamma,
        "fit": yfit,
        "rel_rmse": rel_rmse(y, yfit),
        "log_rmse": log_rmse(y, yfit),
    }


def load_xi_data(path):
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]

    # Dense-bridge format: theta0 tp xi source_id ...
    if arr.shape[1] >= 4 and np.all(np.isin(np.unique(arr[:, 3].astype(int)), [0, 1])):
        return {
            "theta0": arr[:, 0],
            "tp": arr[:, 1],
            "xi": arr[:, 2],
            "source_id": arr[:, 3].astype(int),
        }

    # Fit-ready format: H_star t_star theta0 t_p ... xi_DM ...
    if arr.shape[1] >= 10:
        return {
            "theta0": arr[:, 2],
            "tp": arr[:, 3],
            "xi": arr[:, 9],
            "source_id": np.zeros(arr.shape[0], dtype=int),
        }

    raise RuntimeError(f"Unsupported xi-data format in {path}.")


def load_reference_fanh(path):
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    theta0 = arr[:, 2]
    fanh_no = arr[:, 8]
    thetas = np.array(sorted(np.unique(theta0)), dtype=np.float64)
    fvals = np.array([fanh_no[np.isclose(theta0, th0)][0] for th0 in thetas], dtype=np.float64)
    return thetas, fvals


def xi_model(theta0, tp, af, gf, ainf, ginf, c, q):
    h = h_of_theta(theta0)
    fanh_no = af * np.power(h, gf)
    fanh_inf = ainf * np.power(h, ginf)
    return np.power(tp, 1.5) / np.square(fanh_no) * (fanh_inf + c * np.square(fanh_no) / np.power(tp, q))


def fit_xi_model(theta0, tp, xi, af, gf, q_mode, tp_label):
    if q_mode == "free":
        x0 = np.array([np.log(8.0e-2), 1.5, np.log(1.0), 1.5], dtype=np.float64)
        lower = np.array([np.log(1.0e-8), 0.0, np.log(1.0e-8), 0.5], dtype=np.float64)
        upper = np.array([np.log(10.0), 6.0, np.log(10.0), 3.0], dtype=np.float64)
    elif q_mode == "fixed":
        x0 = np.array([np.log(8.0e-2), 1.5, np.log(1.0)], dtype=np.float64)
        lower = np.array([np.log(1.0e-8), 0.0, np.log(1.0e-8)], dtype=np.float64)
        upper = np.array([np.log(10.0), 6.0, np.log(10.0)], dtype=np.float64)
    else:
        raise ValueError(q_mode)

    def unpack(params):
        ainf = float(np.exp(params[0]))
        ginf = float(params[1])
        c = float(np.exp(params[2]))
        q = 1.5 if q_mode == "fixed" else float(params[3])
        return ainf, ginf, c, q

    def residuals(params):
        ainf, ginf, c, q = unpack(params)
        xi_fit = xi_model(theta0, tp, af, gf, ainf, ginf, c, q)
        return np.log(xi_fit) - np.log(xi)

    res = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        max_nfev=50000,
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
    )

    ainf, ginf, c, q = unpack(res.x)
    xi_fit = xi_model(theta0, tp, af, gf, ainf, ginf, c, q)
    n = len(xi)
    k = 4 if q_mode == "free" else 3
    dof = max(n - k, 1)
    chi2_rel = float(np.sum(np.square((xi_fit - xi) / xi)))
    chi2_log = float(np.sum(np.square(np.log(xi_fit) - np.log(xi))))
    return {
        "label": tp_label + "_" + q_mode,
        "subset": tp_label,
        "q_mode": q_mode,
        "ainf": ainf,
        "ginf": ginf,
        "c": c,
        "q": q,
        "success": bool(res.success),
        "message": str(res.message),
        "xi_fit": xi_fit,
        "rel_rmse": rel_rmse(xi, xi_fit),
        "log_rmse": log_rmse(xi, xi_fit),
        "chi2_rel": chi2_rel,
        "chi2_rel_red": chi2_rel / dof,
        "chi2_log": chi2_log,
        "chi2_log_red": chi2_log / dof,
        "n": n,
        "dof": dof,
    }


def save_summary(path, nofit, results):
    with open(path, "w") as f:
        f.write("# Asymptotic xi model fit\n")
        f.write("# xi(theta,tp) = tp^(3/2)/fanh_noPT(theta)^2 * (fanh_infty(theta) + c * fanh_noPT(theta)^2 / tp^q)\n")
        f.write(
            f"# fanh_noPT(theta) = A_f * h(theta)^gamma_f with "
            f"A_f={nofit['amp']:.10e} gamma_f={nofit['gamma']:.10e} "
            f"rel_rmse={nofit['rel_rmse']:.10e} log_rmse={nofit['log_rmse']:.10e}\n\n"
        )
        f.write(
            "# subset q_mode A_inf gamma_inf c q rel_rmse log_rmse chi2_rel chi2_rel_red chi2_log chi2_log_red n dof success\n"
        )
        for rec in results:
            f.write(
                f"{rec['subset']} {rec['q_mode']} {rec['ainf']:.10e} {rec['ginf']:.10e} "
                f"{rec['c']:.10e} {rec['q']:.10e} {rec['rel_rmse']:.10e} {rec['log_rmse']:.10e} "
                f"{rec['chi2_rel']:.10e} {rec['chi2_rel_red']:.10e} {rec['chi2_log']:.10e} "
                f"{rec['chi2_log_red']:.10e} {rec['n']} {rec['dof']} {int(rec['success'])}\n"
            )
            f.write(f"# message {rec['message']}\n")


def make_plot(data, results, af, gf, out_path, dpi):
    theta_u = np.array(sorted(np.unique(data["theta0"])), dtype=np.float64)
    ntheta = len(theta_u)
    ncols = 3 if ntheta > 4 else 2
    nrows = int(np.ceil(ntheta / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.9 * nrows), squeeze=False)
    axes = axes.ravel()
    free_all = next(rec for rec in results if rec["label"] == "all_free")
    fixed_all = next(rec for rec in results if rec["label"] == "all_fixed")

    for ax, th0 in zip(axes, theta_u):
        mask = np.isclose(data["theta0"], th0)
        idx = np.argsort(data["tp"][mask])
        tp = data["tp"][mask][idx]
        xi = data["xi"][mask][idx]
        src = data["source_id"][mask][idx]
        xi_free = xi_model(th0, tp, af, gf, free_all["ainf"], free_all["ginf"], free_all["c"], free_all["q"])
        xi_fixed = xi_model(th0, tp, af, gf, fixed_all["ainf"], fixed_all["ginf"], fixed_all["c"], fixed_all["q"])

        ax.plot(tp, xi, "-", lw=1.0, color="black", alpha=0.4)
        ax.plot(tp[src == 0], xi[src == 0], "o", ms=3.8, color="black", label="data")
        ax.plot(tp[src == 1], xi[src == 1], ".", ms=4.0, color="black")
        ax.plot(tp, xi_free, "-", lw=1.8, color="tab:blue", label=rf"free $q={free_all['q']:.3f}$")
        ax.plot(tp, xi_fixed, "--", lw=1.8, color="tab:red", label=r"fixed $q=1.5$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(rf"$\theta_0={th0:.3g}$")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$\xi$")
        ax.grid(alpha=0.25)

    for ax in axes[ntheta:]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_residual_plot(data, results, af, gf, out_path, dpi):
    theta_u = np.array(sorted(np.unique(data["theta0"])), dtype=np.float64)
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(theta_u)))
    free_all = next(rec for rec in results if rec["label"] == "all_free")
    fixed_all = next(rec for rec in results if rec["label"] == "all_fixed")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), sharey=True)
    for ax, rec, title in [
        (axes[0], free_all, rf"free $q={free_all['q']:.3f}$"),
        (axes[1], fixed_all, r"fixed $q=1.5$"),
    ]:
        for color, th0 in zip(colors, theta_u):
            mask = np.isclose(data["theta0"], th0)
            idx = np.argsort(data["tp"][mask])
            tp = data["tp"][mask][idx]
            xi = data["xi"][mask][idx]
            xi_fit = xi_model(th0, tp, af, gf, rec["ainf"], rec["ginf"], rec["c"], rec["q"])
            resid = (xi_fit - xi) / xi
            ax.plot(tp, resid, "-o", ms=3.0, lw=1.2, color=color, label=rf"$\theta_0={th0:.3g}$")
        ax.axhline(0.0, color="black", lw=1.0, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel(r"$t_p$")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"relative residual $(\xi_{\rm fit}-\xi)/\xi$")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def infer_stem(path):
    name = Path(path).name
    match = re.search(r"(H\d+p\d+)", name)
    tag = match.group(1) if match else "custom"
    return f"fit_xi_asymptotic_model_{tag}"


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_xi_data(Path(args.xi_data).resolve())
    ref_theta, ref_fanh = load_reference_fanh(Path(args.reference_data).resolve())
    h = h_of_theta(ref_theta)
    nofit = fit_powerlaw_h(h, ref_fanh)
    af = nofit["amp"]
    gf = nofit["gamma"]

    all_mask = np.isfinite(data["tp"]) & np.isfinite(data["xi"]) & (data["tp"] > 0.0) & (data["xi"] > 0.0)
    small_mask = all_mask & (data["tp"] <= float(args.tp_max_smallfit))

    results = []
    for mask, label in [(all_mask, "all"), (small_mask, "smalltp")]:
        theta = data["theta0"][mask]
        tp = data["tp"][mask]
        xi = data["xi"][mask]
        results.append(fit_xi_model(theta, tp, xi, af, gf, "free", label))
        results.append(fit_xi_model(theta, tp, xi, af, gf, "fixed", label))

    stem = infer_stem(args.reference_data)
    summary_out = outdir / f"{stem}.txt"
    plot_out = outdir / f"{stem}.png"
    resid_out = outdir / f"{stem}_residual.png"

    save_summary(summary_out, nofit, results)
    make_plot(data, results, af, gf, plot_out, args.dpi)
    make_residual_plot(data, results, af, gf, resid_out, args.dpi)

    print(f"Saved: {summary_out}")
    print(f"Saved: {plot_out}")
    print(f"Saved: {resid_out}")
    print(
        "fanh_noPT fit: A_f={:.6e}, gamma_f={:.6e}, rel={:.3e}, log={:.3e}".format(
            nofit["amp"], nofit["gamma"], nofit["rel_rmse"], nofit["log_rmse"]
        )
    )
    for rec in results:
        print(
            "{} {}: A_inf={:.6e}, gamma_inf={:.6e}, c={:.6e}, q={:.6f}, rel={:.3e}, log={:.3e}, chi2_rel_red={:.3e}, chi2_log_red={:.3e}".format(
                rec["subset"],
                rec["q_mode"],
                rec["ainf"],
                rec["ginf"],
                rec["c"],
                rec["q"],
                rec["rel_rmse"],
                rec["log_rmse"],
                rec["chi2_rel_red"],
                rec["chi2_log_red"],
            )
        )


if __name__ == "__main__":
    main()
