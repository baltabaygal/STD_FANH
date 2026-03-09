import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from test_y2_tp_shape_ansatz import fit_family, fit_one_slice, shape_model_plus_const


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Check predictive and parameter stability for fixed-tp angular-shape "
            "fits of Y2(theta0,tp)."
        )
    )
    p.add_argument("--data", type=str, default="ode/analysis/data/dm_tp_fitready_H1p000.txt")
    p.add_argument("--outdir", type=str, default="ode/analysis/results")
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def rel_rmse(y, yfit):
    return float(np.sqrt(np.mean(np.square((yfit - y) / y))))


def fit_slice_jackknife(theta0, y, family):
    n = len(theta0)
    preds = []
    holdout_rel = []
    params = []

    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        fit = fit_one_slice(theta0[keep], y[keep], family, with_const=True)
        ypred = shape_model_plus_const(theta0, fit["c0"], fit["amp"], fit["alpha"], family)
        preds.append(ypred)
        params.append([fit["c0"], fit["amp"], fit["alpha"]])
        holdout_rel.append(abs(ypred[i] - y[i]) / y[i])

    preds = np.asarray(preds, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    holdout_rel = np.asarray(holdout_rel, dtype=np.float64)

    return {
        "pred_rel_spread_mean": float(np.mean(np.std(preds, axis=0) / y)),
        "pred_rel_spread_max": float(np.max(np.std(preds, axis=0) / y)),
        "holdout_rel_rmse": float(np.sqrt(np.mean(np.square(holdout_rel)))),
        "holdout_rel_p95": float(np.percentile(holdout_rel, 95.0)),
        "param_std": np.std(params, axis=0),
    }


def curve_roughness(x, y):
    d1 = np.gradient(y, x)
    d2 = np.gradient(d1, x)
    rough = float(np.sqrt(np.mean(d2 * d2)) / max(np.sqrt(np.mean(d1 * d1)), 1e-12))
    sign = np.sign(d1)
    sign = sign[sign != 0.0]
    sign_changes = int(np.sum(sign[1:] * sign[:-1] < 0.0)) if len(sign) >= 2 else 0
    return rough, sign_changes


def save_summary(path, records):
    with open(path, "w") as f:
        f.write("# Stability checks for fixed-tp Y2 shape ansaetze\n")
        f.write("# model global_rel loo_holdout_rel loo_holdout_p95 pred_spread_mean pred_spread_max\n")
        for rec in records:
            f.write(
                f"{rec['model']} {rec['global_rel']:.10e} {rec['loo_holdout_rel']:.10e} "
                f"{rec['loo_holdout_p95']:.10e} {rec['pred_spread_mean']:.10e} "
                f"{rec['pred_spread_max']:.10e}\n"
            )
            f.write("# param roughness sign_changes mean_jk_std\n")
            for name, rough, sign_changes, mean_std in rec["param_rows"]:
                f.write(f"{name} {rough:.10e} {sign_changes:d} {mean_std:.10e}\n")
            f.write("# tp holdout_rel_rmse pred_rel_spread_mean c0_jk_std A_jk_std alpha_jk_std\n")
            for row in rec["slice_rows"]:
                f.write(
                    f"{row[0]:.10e} {row[1]:.10e} {row[2]:.10e} {row[3]:.10e} "
                    f"{row[4]:.10e} {row[5]:.10e}\n"
                )
            f.write("\n")


def make_plot(records, out_path, dpi):
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.0), sharex=True)
    for rec in records:
        rows = rec["slice_rows"]
        axes[0].plot(rows[:, 0], rows[:, 1], "-o", ms=3, label=rec["model"])
        axes[1].plot(rows[:, 0], rows[:, 2], "-o", ms=3, label=rec["model"])

    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_ylabel("LOO holdout rel. RMSE")
    axes[1].set_ylabel("JK curve rel. spread")
    axes[1].set_xlabel(r"$t_p$")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].legend(frameon=False)
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

    records = []
    for family in ["log_hilltop", "pendulum_log"]:
        fit = fit_family(theta0, tp, y2, family, with_const=True)
        slice_rows = []
        loo_holdout = []
        pred_spreads = []
        param_stds = []

        for tpi in np.unique(tp):
            mask_tp = np.isclose(tp, tpi)
            th = theta0[mask_tp]
            y = y2[mask_tp]
            idx = np.argsort(th)
            th = th[idx]
            y = y[idx]
            jk = fit_slice_jackknife(th, y, family)
            slice_rows.append(
                (
                    tpi,
                    jk["holdout_rel_rmse"],
                    jk["pred_rel_spread_mean"],
                    jk["param_std"][0],
                    jk["param_std"][1],
                    jk["param_std"][2],
                )
            )
            loo_holdout.extend([jk["holdout_rel_rmse"]])
            pred_spreads.extend([jk["pred_rel_spread_mean"]])
            param_stds.append(jk["param_std"])

        slice_rows = np.asarray(slice_rows, dtype=np.float64)
        param_stds = np.asarray(param_stds, dtype=np.float64)
        x = np.log(fit["rows"][:, 0])
        param_rows = []
        for i, name in enumerate(["c0", "A", "alpha"]):
            rough, sign_changes = curve_roughness(x, fit["rows"][:, i + 1])
            param_rows.append((name, rough, sign_changes, float(np.mean(param_stds[:, i]))))

        records.append(
            {
                "model": family + "_plus_const",
                "global_rel": fit["global_rel_rmse"],
                "loo_holdout_rel": float(np.sqrt(np.mean(np.square(loo_holdout)))),
                "loo_holdout_p95": float(np.percentile(slice_rows[:, 1], 95.0)),
                "pred_spread_mean": float(np.mean(pred_spreads)),
                "pred_spread_max": float(np.max(pred_spreads)),
                "param_rows": param_rows,
                "slice_rows": slice_rows,
            }
        )

    stem = data_path.stem
    summary_out = outdir / f"check_y2_tp_shape_stability_{stem}.txt"
    plot_out = outdir / f"check_y2_tp_shape_stability_{stem}.png"
    save_summary(summary_out, records)
    make_plot(records, plot_out, args.dpi)

    best_predictive = min(records, key=lambda r: r["loo_holdout_rel"])
    smooth_scores = {
        rec["model"]: sum(row[1] for row in rec["param_rows"]) for rec in records
    }
    best_smooth = min(records, key=lambda r: smooth_scores[r["model"]])

    print(f"Saved: {summary_out}")
    print(f"Saved: {plot_out}")
    for rec in records:
        print(
            "{}: global_rel={:.4e}, loo_holdout_rel={:.4e}, pred_spread_mean={:.4e}".format(
                rec["model"], rec["global_rel"], rec["loo_holdout_rel"], rec["pred_spread_mean"]
            )
        )
    print(f"Best predictive stability: {best_predictive['model']}")
    print(f"Best parameter smoothness: {best_smooth['model']}")


if __name__ == "__main__":
    main()
