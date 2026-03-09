import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Test fixed-tp angular-shape ansaetze for "
            "Y2(theta0,tp) with free A(tp), alpha(tp), and optional c0(tp)."
        )
    )
    p.add_argument("--data", type=str, default="ode/analysis/data/dm_tp_fitready_H1p000.txt")
    p.add_argument("--outdir", type=str, default="ode/analysis/results")
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def rel_rmse(y, yfit):
    return float(np.sqrt(np.mean(np.square((yfit - y) / y))))


def log_rmse(y, yfit):
    return float(np.sqrt(np.mean(np.square(np.log(yfit) - np.log(y)))))


def family_base(theta0, family):
    theta0 = np.asarray(theta0, dtype=np.float64)
    u = theta0 / np.pi

    if family == "power":
        return 1.0 / np.clip(1.0 - u * u, 1e-12, None)
    if family == "log_hilltop":
        return np.log(np.e / np.clip(1.0 - u * u, 1e-12, None))
    if family == "pendulum_log":
        return np.log(np.e / np.clip(np.cos(theta0 / 2.0) ** 2, 1e-12, None))
    raise ValueError(f"Unknown family: {family}")


def shape_model(theta0, amp, alpha, family):
    return amp * np.power(family_base(theta0, family), alpha)


def shape_model_plus_const(theta0, c0, amp, alpha, family):
    return c0 + shape_model(theta0, amp, alpha, family)


def fit_one_slice(theta0, y, family, with_const):
    if with_const:
        p0 = [max(float(y.min()) * 0.8, 0.0), max(float(np.median(y) - y.min()), 1e-8), 0.3]
        bounds = ([0.0, 1e-12, 0.0], [10.0, 1e4, 6.0])
        popt, _ = curve_fit(
            lambda th, c0, amp, alpha: shape_model_plus_const(th, c0, amp, alpha, family),
            theta0,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=250000,
        )
        yfit = shape_model_plus_const(theta0, *popt, family)
        return {"c0": popt[0], "amp": popt[1], "alpha": popt[2], "yfit": yfit}

    p0 = [max(float(np.median(y)), 1e-8), 0.3]
    bounds = ([1e-12, 0.0], [1e4, 6.0])
    popt, _ = curve_fit(
        lambda th, amp, alpha: shape_model(th, amp, alpha, family),
        theta0,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=250000,
    )
    yfit = shape_model(theta0, *popt, family)
    return {"c0": 0.0, "amp": popt[0], "alpha": popt[1], "yfit": yfit}


def fit_family(theta0, tp, y2, family, with_const):
    rows = []
    yfit_all = np.full_like(y2, np.nan, dtype=np.float64)

    for tpi in np.unique(tp):
        mask = np.isclose(tp, tpi) & np.isfinite(theta0) & np.isfinite(y2) & (y2 > 0.0)
        th = theta0[mask]
        y = y2[mask]
        if len(th) < 4:
            continue

        idx = np.argsort(th)
        th = th[idx]
        y = y[idx]

        fit = fit_one_slice(th, y, family, with_const)
        lr = log_rmse(y, fit["yfit"])
        rr = rel_rmse(y, fit["yfit"])
        rows.append((tpi, fit["c0"], fit["amp"], fit["alpha"], lr, rr))

        yfit_all[np.where(mask)[0][idx]] = fit["yfit"]

    rows = np.array(rows, dtype=np.float64)
    valid = np.isfinite(yfit_all) & np.isfinite(y2) & (y2 > 0.0)
    theta_err = []
    for th0 in np.unique(theta0):
        mask = np.isclose(theta0, th0) & valid
        theta_err.append((th0, log_rmse(y2[mask], yfit_all[mask]), rel_rmse(y2[mask], yfit_all[mask])))

    tp_err = []
    for label, region in [
        ("tp < 1", tp < 1.0),
        ("1 <= tp < 10", (tp >= 1.0) & (tp < 10.0)),
        ("tp >= 10", tp >= 10.0),
    ]:
        mask = valid & region
        tp_err.append((label, log_rmse(y2[mask], yfit_all[mask]), rel_rmse(y2[mask], yfit_all[mask])))

    return {
        "family": family,
        "with_const": with_const,
        "rows": rows,
        "yfit": yfit_all,
        "global_log_rmse": log_rmse(y2[valid], yfit_all[valid]),
        "global_rel_rmse": rel_rmse(y2[valid], yfit_all[valid]),
        "theta_err": np.array(theta_err, dtype=np.float64),
        "tp_err": tp_err,
    }


def save_summary(path, results):
    with open(path, "w") as f:
        f.write("# Fixed-tp angular-shape tests for Y2(theta0,tp)\n")
        f.write("# model global_log_rmse global_rel_rmse mean_slice_rel max_slice_rel\n")
        for result in results:
            model = result["family"] + ("_plus_const" if result["with_const"] else "")
            rows = result["rows"]
            f.write(
                f"{model} {result['global_log_rmse']:.10e} {result['global_rel_rmse']:.10e} "
                f"{np.mean(rows[:, 5]):.10e} {np.max(rows[:, 5]):.10e}\n"
            )
            f.write("# tp c0_tp A_tp alpha_tp log_rmse rel_rmse\n")
            for row in rows:
                f.write(
                    f"{row[0]:.10e} {row[1]:.10e} {row[2]:.10e} {row[3]:.10e} "
                    f"{row[4]:.10e} {row[5]:.10e}\n"
                )
            f.write("# theta0 log_rmse rel_rmse\n")
            for row in result["theta_err"]:
                f.write(f"{row[0]:.8g} {row[1]:.10e} {row[2]:.10e}\n")
            for label, lr, rr in result["tp_err"]:
                f.write(f"# {label} log_rmse={lr:.10e} rel_rmse={rr:.10e}\n")
            f.write("\n")


def make_tradeoff_plot(results, out_path, dpi):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for result in results:
        model = result["family"] + (" + c0(tp)" if result["with_const"] else "")
        ax.plot(
            result["global_rel_rmse"],
            np.mean(result["rows"][:, 5]),
            "o",
            ms=8,
            label=model,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Global relative RMSE")
    ax.set_ylabel("Mean slice relative RMSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_trend_plot(result, out_path, dpi):
    rows = result["rows"]
    ncols = 3 if result["with_const"] else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4.6 * ncols, 4.2))
    if ncols == 2:
        axes = np.array(axes, dtype=object)

    idx = 0
    if result["with_const"]:
        axes[idx].plot(rows[:, 0], rows[:, 1], "-o", ms=3)
        axes[idx].set_ylabel(r"$c_0(t_p)$")
        axes[idx].grid(alpha=0.25)
        idx += 1

    axes[idx].plot(rows[:, 0], rows[:, 2], "-o", ms=3)
    axes[idx].set_ylabel(r"$A(t_p)$")
    axes[idx].grid(alpha=0.25)

    axes[idx + 1].plot(rows[:, 0], rows[:, 3], "-o", ms=3, color="tab:red")
    axes[idx + 1].set_ylabel(r"$\alpha(t_p)$")
    axes[idx + 1].grid(alpha=0.25)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$t_p$")

    title = result["family"] + (" + c0(tp)" if result["with_const"] else "")
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_theta_compare_plot(theta0, tp, y2, results, out_path, dpi):
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.2), squeeze=False)
    axes = axes.ravel()
    pick_tp = np.unique(tp)[np.linspace(0, len(np.unique(tp)) - 1, 6).astype(int)]
    best = sorted(results, key=lambda r: r["global_rel_rmse"])[:4]

    for ax, tpi in zip(axes, pick_tp):
        mask = np.isclose(tp, tpi)
        idx = np.argsort(theta0[mask])
        th = theta0[mask][idx]
        yd = y2[mask][idx]
        ax.plot(th, yd, "ko", ms=4, label="data")
        for result in best:
            yf = result["yfit"][mask][idx]
            model = result["family"] + ("+c0" if result["with_const"] else "")
            ax.plot(th, yf, "-", lw=1.6, label=f"{model} ({result['global_rel_rmse']:.3f})")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\theta_0$")
        ax.set_ylabel(r"$Y_2$")
        ax.set_title(rf"$t_p={tpi:.3g}$")
        ax.grid(alpha=0.25)

    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_tp_compare_plot(theta0, tp, y2, result, out_path, dpi):
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.2), squeeze=False)
    axes = axes.ravel()
    model = result["family"] + (" + c0(tp)" if result["with_const"] else "")

    for ax, th0 in zip(axes, np.unique(theta0)):
        mask = np.isclose(theta0, th0)
        idx = np.argsort(tp[mask])
        tps = tp[mask][idx]
        yd = y2[mask][idx]
        yf = result["yfit"][mask][idx]

        ax.plot(tps, yd, "ko", ms=4, label="data")
        ax.plot(tps, yf, "-", lw=1.8, color="tab:blue", label="fit")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(r"$Y_2$")
        ax.set_title(rf"$\theta_0={th0:.4g}$")
        ax.grid(alpha=0.25)

    axes[0].legend(frameon=False, fontsize=8, title=model, title_fontsize=8)
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

    mask = np.isfinite(theta0) & np.isfinite(tp) & np.isfinite(y2) & (tp > 0.0) & (y2 > 0.0)
    theta0 = theta0[mask]
    tp = tp[mask]
    y2 = y2[mask]

    results = []
    for family in ["power", "log_hilltop", "pendulum_log"]:
        results.append(fit_family(theta0, tp, y2, family, with_const=False))
        results.append(fit_family(theta0, tp, y2, family, with_const=True))

    stem = data_path.stem
    summary_out = outdir / f"test_y2_tp_shape_ansatz_{stem}.txt"
    tradeoff_out = outdir / f"test_y2_tp_shape_ansatz_tradeoff_{stem}.png"
    theta_out = outdir / f"test_y2_tp_shape_ansatz_vs_theta_{stem}.png"

    save_summary(summary_out, results)
    make_tradeoff_plot(results, tradeoff_out, args.dpi)
    make_theta_compare_plot(theta0, tp, y2, results, theta_out, args.dpi)

    for result in results:
        model = result["family"] + ("_plus_const" if result["with_const"] else "")
        trend_out = outdir / f"test_y2_tp_shape_ansatz_{model}_trends_{stem}.png"
        make_trend_plot(result, trend_out, args.dpi)
        if result["with_const"] and result["family"] in {"log_hilltop", "pendulum_log"}:
            tp_out = outdir / f"test_y2_tp_shape_ansatz_{model}_vs_tp_{stem}.png"
            make_tp_compare_plot(theta0, tp, y2, result, tp_out, args.dpi)

    best = min(results, key=lambda r: r["global_rel_rmse"])
    print(f"Loaded: {data_path}")
    print(f"Saved: {summary_out}")
    print(f"Saved: {tradeoff_out}")
    print(f"Saved: {theta_out}")
    for result in sorted(results, key=lambda r: r["global_rel_rmse"]):
        model = result["family"] + ("_plus_const" if result["with_const"] else "")
        print(
            "{}: global_rel={:.4e}, mean_slice_rel={:.4e}, max_slice_rel={:.4e}".format(
                model,
                result["global_rel_rmse"],
                float(np.mean(result["rows"][:, 5])),
                float(np.max(result["rows"][:, 5])),
            )
        )
    best_model = best["family"] + ("_plus_const" if best["with_const"] else "")
    print(f"Best fixed-tp shape ansatz: {best_model}")


if __name__ == "__main__":
    main()
