import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar


TARGET_H_DEFAULT = [1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compare lattice xi(beta/H) data to the direct ODE xi(beta/H) curves "
            "at fixed H_* and v_w, allowing only a horizontal shift in log(beta/H)."
        )
    )
    p.add_argument("--ode-data", type=str, default="ode/xi_DM_ODE_results.txt")
    p.add_argument("--lattice-data", type=str, default="lattice_data/data/energy_ratio_by_theta_data_v9.txt")
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--target-h", type=float, nargs="+", default=TARGET_H_DEFAULT)
    p.add_argument("--shift-min", type=float, default=-0.5, help="Minimum delta in log(beta/H).")
    p.add_argument("--shift-max", type=float, default=0.5, help="Maximum delta in log(beta/H).")
    p.add_argument(
        "--outdir",
        type=str,
        default="ode/analysis/results/lattice_fit/ode_horizontal_shift_v9",
    )
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def rel_rmse(y, yfit):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square((yfit - y) / y))))


def log_rmse(y, yfit):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(np.log(yfit) - np.log(y)))))


def load_lattice(path, target_h):
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    mask = np.zeros(arr.shape[0], dtype=bool)
    for hstar in target_h:
        mask |= np.isclose(arr[:, 1], float(hstar), rtol=0.0, atol=1.0e-12)
    arr = arr[mask]
    sem = arr[:, 5] / np.sqrt(np.maximum(arr[:, 6], 1.0))
    sem = np.maximum(sem, 1.0e-12)
    return {
        "theta0": arr[:, 0].astype(np.float64),
        "hstar": arr[:, 1].astype(np.float64),
        "beta_over_h": arr[:, 2].astype(np.float64),
        "mean_ratio": arr[:, 4].astype(np.float64),
        "sem": sem.astype(np.float64),
    }


def load_ode(path, vw, target_h):
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    mask = np.isclose(arr[:, 0], float(vw), rtol=0.0, atol=1.0e-12)
    hmask = np.zeros(arr.shape[0], dtype=bool)
    for hstar in target_h:
        hmask |= np.isclose(arr[:, 2], float(hstar), rtol=0.0, atol=1.0e-12)
    arr = arr[mask & hmask]
    return {
        "theta0": arr[:, 1].astype(np.float64),
        "hstar": arr[:, 2].astype(np.float64),
        "beta_over_h": arr[:, 3].astype(np.float64),
        "tp": arr[:, 4].astype(np.float64),
        "xi": arr[:, 5].astype(np.float64),
    }


def select_h(data, hstar):
    mask = np.isclose(data["hstar"], float(hstar), rtol=0.0, atol=1.0e-12)
    return {key: val[mask] for key, val in data.items()}


def build_interpolators(ode_h):
    theta_u = np.array(sorted(np.unique(ode_h["theta0"])), dtype=np.float64)
    interps = {}
    for th0 in theta_u:
        mask = np.isclose(ode_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
        beta = ode_h["beta_over_h"][mask]
        xi = ode_h["xi"][mask]
        order = np.argsort(beta)
        interps[float(th0)] = interp1d(
            np.log(beta[order]),
            xi[order],
            kind="linear",
            fill_value="extrapolate",
            assume_sorted=True,
        )
    return interps


def nearest_theta_key(interps, theta0):
    keys = np.array(sorted(interps.keys()), dtype=np.float64)
    idx = int(np.argmin(np.abs(keys - float(theta0))))
    if abs(keys[idx] - float(theta0)) > 5.0e-4:
        raise KeyError(f"No ODE theta match for theta0={theta0:.10f}")
    return float(keys[idx])


def evaluate_shift(lattice_h, interps, delta_logbeta):
    theta = lattice_h["theta0"]
    beta = lattice_h["beta_over_h"]
    xs = np.log(beta) + float(delta_logbeta)
    yfit = np.array(
        [float(interps[nearest_theta_key(interps, th)](x)) for th, x in zip(theta, xs)],
        dtype=np.float64,
    )
    chi2 = float(np.sum(np.square((yfit - lattice_h["mean_ratio"]) / lattice_h["sem"])))
    dof = max(len(yfit) - 1, 1)
    return {
        "delta_logbeta": float(delta_logbeta),
        "scale_beta": float(np.exp(delta_logbeta)),
        "xi_fit": yfit,
        "chi2": chi2,
        "chi2_red": chi2 / dof,
        "rel_rmse": rel_rmse(lattice_h["mean_ratio"], yfit),
        "log_rmse": log_rmse(lattice_h["mean_ratio"], yfit),
        "n": len(yfit),
        "dof": dof,
    }


def fit_shift(lattice_h, interps, shift_min, shift_max, objective):
    def target(delta):
        rec = evaluate_shift(lattice_h, interps, delta)
        return rec["chi2"] if objective == "chi2" else rec["rel_rmse"]

    res = minimize_scalar(target, bounds=(shift_min, shift_max), method="bounded")
    rec = evaluate_shift(lattice_h, interps, res.x)
    rec["objective"] = objective
    rec["success"] = bool(res.success)
    return rec


def fit_per_theta_shift(lattice_h, interps, shift_min, shift_max):
    theta_u = np.array(sorted(np.unique(lattice_h["theta0"])), dtype=np.float64)
    rows = []
    for th0 in theta_u:
        mask = np.isclose(lattice_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
        sub = {key: val[mask] for key, val in lattice_h.items()}
        rec = fit_shift(sub, interps, shift_min, shift_max, "rel")
        rec["theta0"] = float(th0)
        rows.append(rec)
    return rows


def make_overlay_plot(lattice_h, ode_h, records, out_path, dpi, title):
    theta_u = np.array(sorted(np.unique(lattice_h["theta0"])), dtype=np.float64)
    ntheta = len(theta_u)
    ncols = 3 if ntheta > 4 else 2
    nrows = int(np.ceil(ntheta / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.0 * nrows), squeeze=False)
    axes = axes.ravel()

    style = {
        "no_shift": ("tab:gray", "-", "ODE, no shift"),
        "best_rel": ("tab:blue", "--", "ODE, best shift (rel)"),
        "best_chi2": ("tab:red", "-.", "ODE, best shift (chi2)"),
    }

    for ax, th0 in zip(axes, theta_u):
        lmask = np.isclose(lattice_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
        beta = lattice_h["beta_over_h"][lmask]
        y = lattice_h["mean_ratio"][lmask]
        sem = lattice_h["sem"][lmask]
        order = np.argsort(beta)
        ax.errorbar(beta[order], y[order], yerr=sem[order], fmt="ko", ms=3.4, lw=1.0, capsize=2, label="lattice")

        omask = np.isclose(ode_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
        obeta = ode_h["beta_over_h"][omask]
        oxi = ode_h["xi"][omask]
        oorder = np.argsort(obeta)
        ax.plot(obeta[oorder], oxi[oorder], color="black", lw=1.0, alpha=0.45)

        for key, rec in records.items():
            yfit = rec["xi_fit"][lmask]
            color, ls, label = style[key]
            ax.plot(beta[order], yfit[order], color=color, ls=ls, lw=1.8, label=label)

        ax.set_xscale("log")
        ax.set_xlabel(r"$\beta/H_*$")
        ax.set_ylabel(r"$\xi$")
        ax.set_title(rf"$\theta_0={th0:.3g}$")
        ax.grid(alpha=0.25)

    for ax in axes[ntheta:]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_theta_shift_plot(theta_rows, out_path, dpi, title):
    theta = np.array([row["theta0"] for row in theta_rows], dtype=np.float64)
    scale = np.array([row["scale_beta"] for row in theta_rows], dtype=np.float64)
    rel = np.array([row["rel_rmse"] for row in theta_rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True)
    axes[0].plot(np.degrees(theta), scale, "o-", color="tab:blue", lw=1.8, ms=4.8)
    axes[0].axhline(1.0, color="black", lw=1.0, ls="--")
    axes[0].set_ylabel(r"best $s_\beta$")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)

    axes[1].plot(np.degrees(theta), rel, "o-", color="tab:red", lw=1.8, ms=4.8)
    axes[1].set_xlabel(r"$\theta_0$ [deg]")
    axes[1].set_ylabel("best rel RMSE")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_summary(path, args, summaries, theta_summaries):
    with open(path, "w") as f:
        f.write("# Direct ODE curve horizontal-shift test on lattice data\n")
        f.write(f"# ode_data={Path(args.ode_data).resolve()}\n")
        f.write(f"# lattice_data={Path(args.lattice_data).resolve()}\n")
        f.write(f"# fixed_vw={float(args.fixed_vw):.6f}\n")
        f.write(
            "# convention: xi_fit(beta/H) = xi_ODE(s_beta * beta/H), "
            "with s_beta = exp(delta_logbeta)\n"
        )
        f.write("# s_beta < 1 corresponds to moving the ODE curve to the right on a log-beta/H plot\n\n")
        f.write("# hstar case delta_logbeta s_beta chi2 chi2_red rel_rmse log_rmse n dof\n")
        for hstar, recs in summaries.items():
            for key in ["no_shift", "best_rel", "best_chi2"]:
                rec = recs[key]
                f.write(
                    f"{float(hstar):.8g} {key} "
                    f"{rec['delta_logbeta']:.10e} {rec['scale_beta']:.10e} "
                    f"{rec['chi2']:.10e} {rec['chi2_red']:.10e} "
                    f"{rec['rel_rmse']:.10e} {rec['log_rmse']:.10e} "
                    f"{rec['n']} {rec['dof']}\n"
                )
            f.write("\n")
        f.write("# per_theta_best_rel hstar theta0 delta_logbeta s_beta rel_rmse log_rmse chi2_red\n")
        for hstar, rows in theta_summaries.items():
            for row in rows:
                f.write(
                    f"{float(hstar):.8g} {row['theta0']:.10f} "
                    f"{row['delta_logbeta']:.10e} {row['scale_beta']:.10e} "
                    f"{row['rel_rmse']:.10e} {row['log_rmse']:.10e} {row['chi2_red']:.10e}\n"
                )


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    lattice = load_lattice(Path(args.lattice_data).resolve(), args.target_h)
    ode = load_ode(Path(args.ode_data).resolve(), args.fixed_vw, args.target_h)

    summaries = {}
    theta_summaries = {}
    for hstar in args.target_h:
        lattice_h = select_h(lattice, float(hstar))
        ode_h = select_h(ode, float(hstar))
        interps = build_interpolators(ode_h)

        no_shift = evaluate_shift(lattice_h, interps, 0.0)
        best_rel = fit_shift(lattice_h, interps, args.shift_min, args.shift_max, "rel")
        best_chi2 = fit_shift(lattice_h, interps, args.shift_min, args.shift_max, "chi2")
        theta_rows = fit_per_theta_shift(lattice_h, interps, args.shift_min, args.shift_max)

        summaries[float(hstar)] = {
            "no_shift": no_shift,
            "best_rel": best_rel,
            "best_chi2": best_chi2,
        }
        theta_summaries[float(hstar)] = theta_rows

        overlay_out = outdir / f"test_ode_horizontal_shift_H{float(hstar):.1f}".replace(".", "p")
        overlay_out = Path(str(overlay_out) + ".png")
        make_overlay_plot(
            lattice_h,
            ode_h,
            summaries[float(hstar)],
            overlay_out,
            args.dpi,
            rf"Direct ODE shift test at $H_*={float(hstar):g}$, $v_w={args.fixed_vw:g}$",
        )

        theta_out = outdir / f"test_ode_horizontal_shift_theta_H{float(hstar):.1f}".replace(".", "p")
        theta_out = Path(str(theta_out) + ".png")
        make_theta_shift_plot(
            theta_rows,
            theta_out,
            args.dpi,
            rf"Per-theta best horizontal shifts at $H_*={float(hstar):g}$",
        )

        print(
            "H*={:.1f} no_shift rel={:.3e}; best_rel s_beta={:.4f} rel={:.3e}; best_chi2 s_beta={:.4f} rel={:.3e}".format(
                float(hstar),
                no_shift["rel_rmse"],
                best_rel["scale_beta"],
                best_rel["rel_rmse"],
                best_chi2["scale_beta"],
                best_chi2["rel_rmse"],
            )
        )

    summary_out = outdir / "test_ode_horizontal_shift_summary.txt"
    save_summary(summary_out, args, summaries, theta_summaries)
    print(f"Saved: {summary_out}")


if __name__ == "__main__":
    main()
