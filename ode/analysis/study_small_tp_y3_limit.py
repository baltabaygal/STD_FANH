import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ode.hom_ODE.hom_ode import solve_PT, solve_noPT


T_OSC_NOPT = 1.5
THETA0_DEFAULT = [0.2618, 0.7854, 1.309, 1.833, 2.356, 2.88]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Study the small-t_p limit of Y3(theta0, t_p) = "
            "xi * f_anh_noPT(theta0)^2 / t_p^(3/2) from direct ODE solves."
        )
    )
    p.add_argument("--hstar", type=float, default=1.0, help="Used for output naming only.")
    p.add_argument("--outdir", type=str, default="ode/analysis/results")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--theta0-list", type=float, nargs="+", default=THETA0_DEFAULT)
    p.add_argument("--tp-small-min", type=float, default=1.0e-4)
    p.add_argument("--tp-small-max", type=float, default=1.0e-2)
    p.add_argument("--n-small", type=int, default=9)
    p.add_argument("--tp-reference", type=float, nargs="*", default=[1.0e-1, 3.0e-1, 1.0])
    p.add_argument("--tail-fit-count", type=int, default=6)
    p.add_argument("--t-start-nopt", type=float, default=1.0e-6)
    p.add_argument("--t-end-min", type=float, default=800.0)
    p.add_argument("--extra-after", type=float, default=800.0)
    p.add_argument("--method", type=str, default="Radau")
    p.add_argument("--rtol", type=float, default=1.0e-10)
    p.add_argument("--atol", type=float, default=1.0e-12)
    p.add_argument("--late-frac", type=float, default=0.30)
    p.add_argument("--late-mode", type=str, default="time_weighted")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def potential(theta0):
    return 1.0 - np.cos(theta0)


def h_of_theta(theta0):
    theta0 = np.asarray(theta0, dtype=np.float64)
    return np.log(np.e / np.clip(np.cos(theta0 / 2.0) ** 2, 1e-12, None))


def unique_sorted(values):
    return np.array(sorted(set(float(x) for x in values)), dtype=np.float64)


def build_tp_grid(tp_small_min, tp_small_max, n_small, tp_reference):
    small = np.logspace(np.log10(tp_small_min), np.log10(tp_small_max), int(n_small))
    return unique_sorted(list(small) + [float(x) for x in tp_reference if float(x) > 0.0])


def rel_rmse(y, yfit):
    return float(np.sqrt(np.mean(np.square((yfit - y) / y))))


def fit_powerlaw_h(h, y):
    h = np.asarray(h, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    coeff = np.polyfit(np.log(h), np.log(y), 1)
    gamma = float(coeff[0])
    amp = float(np.exp(coeff[1]))
    yfit = amp * np.power(h, gamma)
    return {
        "amp": amp,
        "gamma": gamma,
        "fit": yfit,
        "rel_rmse": rel_rmse(y, yfit),
    }


def tail_model(tp, yinf, b, q):
    tp = np.asarray(tp, dtype=np.float64)
    return yinf + b / np.power(tp, q)


def fit_small_tp_tail(tp, y3, nfit):
    order = np.argsort(tp)
    x = np.asarray(tp[order][:nfit], dtype=np.float64)
    y = np.asarray(y3[order][:nfit], dtype=np.float64)

    if x.size < 3:
        raise RuntimeError("Need at least 3 small-t_p points for the tail fit.")

    b_guess = max(float(np.median(y * np.power(x, 1.5))), 1.0e-14)
    yinf_guesses = [1.0e-16, y[-1] * 1.0e-8, y[-1] * 1.0e-6, y[-1] * 1.0e-4]
    b_guesses = [0.5 * b_guess, b_guess, 2.0 * b_guess]
    q_guesses = [1.5, 1.35, 1.65]

    lower = np.array([np.log(1.0e-16), np.log(1.0e-16), 0.25], dtype=np.float64)
    upper = np.array([np.log(max(float(np.max(y)), 1.0) * 10.0), np.log(1.0e6), 3.5], dtype=np.float64)

    best = None

    def residuals(params):
        yinf = np.exp(params[0])
        b = np.exp(params[1])
        q = params[2]
        yfit = tail_model(x, yinf, b, q)
        return np.log(yfit) - np.log(y)

    for yinf0 in yinf_guesses:
        for b0 in b_guesses:
            for q0 in q_guesses:
                x0 = np.array([np.log(max(yinf0, 1.0e-16)), np.log(max(b0, 1.0e-16)), q0], dtype=np.float64)
                try:
                    res = least_squares(
                        residuals,
                        x0,
                        bounds=(lower, upper),
                        max_nfev=50000,
                        ftol=1.0e-12,
                        xtol=1.0e-12,
                        gtol=1.0e-12,
                    )
                except Exception:
                    continue
                yinf = float(np.exp(res.x[0]))
                b = float(np.exp(res.x[1]))
                q = float(res.x[2])
                yfit = tail_model(x, yinf, b, q)
                cost = float(np.mean(np.square(np.log(yfit) - np.log(y))))
                rec = {
                    "yinf": yinf,
                    "b": b,
                    "q": q,
                    "xfit": x,
                    "yfit": yfit,
                    "success": bool(res.success),
                    "message": str(res.message),
                    "cost": cost,
                    "rel_rmse": rel_rmse(y, yfit),
                }
                if best is None or rec["cost"] < best["cost"]:
                    best = rec

    if best is None:
        raise RuntimeError("Small-t_p tail fit failed.")
    return best


def compute_reference(theta0, args):
    ea3_no, sol_no = solve_noPT(
        theta0,
        t_start=args.t_start_nopt,
        t_end=args.t_end_min,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        late_frac=args.late_frac,
        late_mode=args.late_mode,
    )
    nsteps_no = len(sol_no.t) if (sol_no is not None and hasattr(sol_no, "t")) else 0
    v0 = potential(theta0)
    fanh_no = ea3_no / (v0 * (T_OSC_NOPT ** 1.5))
    return {
        "ea3_no": float(ea3_no),
        "fanh_no": float(fanh_no),
        "v0": float(v0),
        "nsteps_no": int(nsteps_no),
    }


def compute_theta_record(theta0, tp_grid, args):
    ref = compute_reference(theta0, args)
    rows = []
    for tp in tp_grid:
        ea3_pt, sol_pt = solve_PT(
            theta0,
            tp,
            t_end_min=args.t_end_min,
            extra_after=args.extra_after,
            method=args.method,
            rtol=args.rtol,
            atol=args.atol,
            late_frac=args.late_frac,
            late_mode=args.late_mode,
        )
        nsteps_pt = len(sol_pt.t) if (sol_pt is not None and hasattr(sol_pt, "t")) else 0
        xi = ea3_pt / ref["ea3_no"]
        y3 = xi * (ref["fanh_no"] ** 2) / (tp ** 1.5)
        rows.append(
            {
                "theta0": float(theta0),
                "tp": float(tp),
                "xi": float(xi),
                "y3": float(y3),
                "nsteps_pt": int(nsteps_pt),
            }
        )

    tp = np.array([row["tp"] for row in rows], dtype=np.float64)
    y3 = np.array([row["y3"] for row in rows], dtype=np.float64)
    xi = np.array([row["xi"] for row in rows], dtype=np.float64)
    nsteps_pt = np.array([row["nsteps_pt"] for row in rows], dtype=np.int64)
    fit = fit_small_tp_tail(tp, y3, args.tail_fit_count)
    t_scaled = (y3 - fit["yinf"]) * np.power(tp, 1.5)

    order = np.argsort(tp)
    xi_small_mean = float(np.mean(xi[order][: min(3, len(xi))]))
    xi_small_last = float(xi[order][0])
    t_small_mean = float(np.mean(t_scaled[order][: min(3, len(t_scaled))]))

    return {
        "theta0": float(theta0),
        "h": float(h_of_theta(np.array([theta0]))[0]),
        "ref": ref,
        "tp": tp,
        "y3": y3,
        "xi": xi,
        "nsteps_pt": nsteps_pt,
        "tail": fit,
        "t_scaled": t_scaled,
        "xi_small_mean": xi_small_mean,
        "xi_small_last": xi_small_last,
        "t_small_mean": t_small_mean,
    }


def make_y3_plot(records, out_path, dpi):
    ntheta = len(records)
    ncols = min(3, max(2, int(np.ceil(np.sqrt(ntheta)))))
    nrows = int(np.ceil(ntheta / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.8 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, rec in zip(axes, records):
        idx = np.argsort(rec["tp"])
        tp = rec["tp"][idx]
        y3 = rec["y3"][idx]
        fit = rec["tail"]
        yfit = tail_model(tp, fit["yinf"], fit["b"], fit["q"])
        fit_mask = np.isin(tp, fit["xfit"])

        ax.plot(tp, y3, "o", ms=4.2, color="black", label="ODE")
        ax.plot(tp[fit_mask], y3[fit_mask], "o", ms=4.5, color="tab:red", label="fit bins")
        ax.plot(tp, yfit, "-", lw=1.8, color="tab:blue", label=rf"tail fit, $q={fit['q']:.3f}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$Y_3$")
        ax.set_title(rf"$\theta_0={rec['theta0']:.3g}$")
        ax.grid(alpha=0.25)

    for ax in axes[ntheta:]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_tscaled_plot(records, out_path, dpi):
    ntheta = len(records)
    ncols = min(3, max(2, int(np.ceil(np.sqrt(ntheta)))))
    nrows = int(np.ceil(ntheta / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.8 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, rec in zip(axes, records):
        idx = np.argsort(rec["tp"])
        tp = rec["tp"][idx]
        t_scaled = rec["t_scaled"][idx]
        fanh_no_sq = rec["ref"]["fanh_no"] ** 2
        ax.plot(tp, t_scaled, "-o", ms=4.0, lw=1.6, color="tab:blue", label=r"$T=(Y_3-Y_{3,\infty})t_p^{3/2}$")
        ax.axhline(fanh_no_sq, color="black", lw=1.2, ls="--", label=r"$f_{\rm anh}^{\rm noPT}{}^2$")
        ax.axhline(rec["tail"]["b"], color="tab:red", lw=1.2, ls=":", label=r"fitted $B$")
        ax.set_xscale("log")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$T$")
        ax.set_title(rf"$\theta_0={rec['theta0']:.3g}$")
        ax.grid(alpha=0.25)

    for ax in axes[ntheta:]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_q_plot(records, out_path, dpi):
    theta0 = np.array([rec["theta0"] for rec in records], dtype=np.float64)
    q = np.array([rec["tail"]["q"] for rec in records], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(theta0, q, "-o", ms=4.5, lw=1.8, color="tab:purple")
    ax.axhline(1.5, color="black", lw=1.2, ls="--", label=r"$q=3/2$")
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"fitted $q$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_xi_plot(records, out_path, dpi):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(records)))

    for color, rec in zip(colors, records):
        idx = np.argsort(rec["tp"])
        tp = rec["tp"][idx]
        xi = rec["xi"][idx]
        ax.plot(tp, xi, "-o", ms=4.5, lw=1.8, color=color, label=rf"$\theta_0={rec['theta0']:.3g}$")

    ax.axhline(1.0, color="black", lw=1.2, ls="--", label=r"$\xi=1$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$t_p$")
    ax.set_ylabel(r"$\xi$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_scan_table(path, records):
    with open(path, "w") as f:
        f.write("# theta0 h tp xi y3 t_scaled nsteps_pt nsteps_noPT\n")
        for rec in records:
            for tp, xi, y3, t_scaled, nsteps_pt in zip(
                rec["tp"], rec["xi"], rec["y3"], rec["t_scaled"], rec["nsteps_pt"]
            ):
                f.write(
                    f"{rec['theta0']:.8g} {rec['h']:.10e} {tp:.10e} {xi:.10e} "
                    f"{y3:.10e} {t_scaled:.10e} "
                    f"{int(nsteps_pt)} {rec['ref']['nsteps_no']}\n"
                )


def save_summary(path, records, fanh_no_fit, fanh_inf_fit, args):
    with open(path, "w") as f:
        f.write("# Small-t_p Y3 asymptotics study\n")
        f.write("# Y3(theta0,tp) = xi * f_anh_noPT(theta0)^2 / tp^(3/2)\n")
        f.write(
            f"# solver method={args.method} rtol={args.rtol:.3e} atol={args.atol:.3e} "
            f"t_start_nopt={args.t_start_nopt:.3e} t_end_min={args.t_end_min:.3e} extra_after={args.extra_after:.3e}\n"
        )
        f.write(
            f"# tp_small_min={args.tp_small_min:.3e} tp_small_max={args.tp_small_max:.3e} "
            f"n_small={args.n_small} tail_fit_count={args.tail_fit_count}\n\n"
        )

        f.write(
            f"# fanh_noPT(theta0) = A_f * h(theta0)^gamma_f with "
            f"A_f={fanh_no_fit['amp']:.10e} gamma_f={fanh_no_fit['gamma']:.10e} "
            f"rel_rmse={fanh_no_fit['rel_rmse']:.10e}\n"
        )
        f.write(
            f"# fanh_infty(theta0) = A_inf * h(theta0)^gamma_inf with "
            f"A_inf={fanh_inf_fit['amp']:.10e} gamma_inf={fanh_inf_fit['gamma']:.10e} "
            f"rel_rmse={fanh_inf_fit['rel_rmse']:.10e}\n\n"
        )

        f.write(
            "# theta0 h fanh_noPT fanh_noPT_sq Y3_infty_fit B_fit q_fit "
            "B_over_fanh_noPT_sq T_small_mean_over_fanh_noPT_sq xi_small_mean xi_smallest rel_tail_fit\n"
        )
        for rec in records:
            fanh_no = rec["ref"]["fanh_no"]
            fanh_no_sq = fanh_no ** 2
            t_ratio = rec["t_small_mean"] / fanh_no_sq
            b_ratio = rec["tail"]["b"] / fanh_no_sq
            f.write(
                f"{rec['theta0']:.8g} {rec['h']:.10e} {fanh_no:.10e} {fanh_no_sq:.10e} "
                f"{rec['tail']['yinf']:.10e} {rec['tail']['b']:.10e} {rec['tail']['q']:.10e} "
                f"{b_ratio:.10e} {t_ratio:.10e} {rec['xi_small_mean']:.10e} "
                f"{rec['xi_small_last']:.10e} {rec['tail']['rel_rmse']:.10e}\n"
            )


def main():
    args = parse_args()
    if args.smoke:
        args.theta0_list = args.theta0_list[:3]
        args.n_small = 5
        args.tail_fit_count = 4
        args.tp_reference = [1.0e-1, 1.0]
        args.rtol = max(args.rtol, 1.0e-8)
        args.atol = max(args.atol, 1.0e-10)
        args.t_end_min = min(args.t_end_min, 300.0)
        args.extra_after = min(args.extra_after, 300.0)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    tp_grid = build_tp_grid(args.tp_small_min, args.tp_small_max, args.n_small, args.tp_reference)
    theta0_list = [float(x) for x in args.theta0_list]

    records = []
    for theta0 in theta0_list:
        print(f"Solving theta0={theta0:.4f} over {len(tp_grid)} t_p points...")
        records.append(compute_theta_record(theta0, tp_grid, args))

    h = np.array([rec["h"] for rec in records], dtype=np.float64)
    fanh_no = np.array([rec["ref"]["fanh_no"] for rec in records], dtype=np.float64)
    fanh_inf = np.array([rec["tail"]["yinf"] for rec in records], dtype=np.float64)
    fanh_no_fit = fit_powerlaw_h(h, fanh_no)
    fanh_inf_fit = fit_powerlaw_h(h, np.clip(fanh_inf, 1.0e-16, None))

    tag = f"H{args.hstar:.3f}".replace(".", "p")
    stem = f"small_tp_y3_limit_{tag}"

    scan_out = outdir / f"{stem}_scan.txt"
    summary_out = outdir / f"{stem}_summary.txt"
    y3_plot_out = outdir / f"{stem}_logy3_vs_tp.png"
    t_plot_out = outdir / f"{stem}_tscaled_vs_tp.png"
    q_plot_out = outdir / f"{stem}_q_vs_theta.png"
    xi_plot_out = outdir / f"{stem}_xi_vs_tp.png"

    save_scan_table(scan_out, records)
    save_summary(summary_out, records, fanh_no_fit, fanh_inf_fit, args)
    make_y3_plot(records, y3_plot_out, args.dpi)
    make_tscaled_plot(records, t_plot_out, args.dpi)
    make_q_plot(records, q_plot_out, args.dpi)
    make_xi_plot(records, xi_plot_out, args.dpi)

    print(f"Saved: {scan_out}")
    print(f"Saved: {summary_out}")
    print(f"Saved: {y3_plot_out}")
    print(f"Saved: {t_plot_out}")
    print(f"Saved: {q_plot_out}")
    print(f"Saved: {xi_plot_out}")
    print(
        "fanh_noPT fit: A_f={:.6e}, gamma_f={:.6e}, rel={:.3e}".format(
            fanh_no_fit["amp"], fanh_no_fit["gamma"], fanh_no_fit["rel_rmse"]
        )
    )
    print(
        "fanh_infty fit: A_inf={:.6e}, gamma_inf={:.6e}, rel={:.3e}".format(
            fanh_inf_fit["amp"], fanh_inf_fit["gamma"], fanh_inf_fit["rel_rmse"]
        )
    )
    for rec in records:
        print(
            "theta0={:.4f}: q={:.6f}, B/f_no^2={:.6f}, T_small/f_no^2={:.6f}, xi_small={:.6f}".format(
                rec["theta0"],
                rec["tail"]["q"],
                rec["tail"]["b"] / (rec["ref"]["fanh_no"] ** 2),
                rec["t_small_mean"] / (rec["ref"]["fanh_no"] ** 2),
                rec["xi_small_mean"],
            )
        )


if __name__ == "__main__":
    main()
