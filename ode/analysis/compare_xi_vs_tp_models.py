import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from refine_y2_physical_models import MODELS, fit_model
from test_y2_tp_shape_ansatz import fit_family


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compare xi(tp) data against the compact global Y2 fit and the "
            "pendulum-log fixed-tp slice benchmark."
        )
    )
    p.add_argument("--data", type=str, default="ode/analysis/data/dm_tp_fitready_H1p000.txt")
    p.add_argument("--outdir", type=str, default="ode/analysis/results")
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def rel_rmse(y, yfit):
    return float(np.sqrt(np.mean(np.square((yfit - y) / y))))


def y2_to_xi(y2, tp, fanh_no):
    return y2 * np.power(tp, 1.5) / fanh_no


def make_plot(theta0, tp, xi, compact_xi, pendulum_xi, compact_rel, pendulum_rel, out_path, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True)
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(np.unique(theta0))))
    panels = [
        (axes[0], compact_xi, f"Compact global fit, rel={compact_rel:.3e}"),
        (axes[1], pendulum_xi, f"Pendulum-log + c0(tp), rel={pendulum_rel:.3e}"),
    ]

    for ax, xifit, title in panels:
        for color, th0 in zip(colors, np.unique(theta0)):
            mask = np.isclose(theta0, th0)
            idx = np.argsort(tp[mask])
            x = tp[mask][idx]
            xd = xi[mask][idx]
            xf = xifit[mask][idx]
            ax.plot(x, xd, "o", ms=3, color=color, alpha=0.75)
            ax.plot(x, xf, "-", lw=1.7, color=color, label=rf"$\theta_0={th0:.3g}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$t_p$")
        ax.set_title(title)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(r"$\xi$")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_summary(path, compact_name, compact_rel, pendulum_rel):
    with open(path, "w") as f:
        f.write("# xi(tp) comparison: compact global fit vs pendulum-log slice benchmark\n")
        f.write(f"compact_model {compact_name}\n")
        f.write(f"compact_rel_rmse {compact_rel:.10e}\n")
        f.write(f"pendulum_slice_rel_rmse {pendulum_rel:.10e}\n")


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

    compact_results = []
    for name, spec in MODELS.items():
        compact_results.append(fit_model(theta0, tp, y2, name, spec))
    compact_results.sort(key=lambda r: (r["global_rel_rmse"], r["global_log_rmse"]))
    compact = compact_results[0]

    pendulum = fit_family(theta0, tp, y2, "pendulum_log", with_const=True)

    compact_xi = y2_to_xi(compact["yfit"], tp, fanh_no)
    pendulum_xi = y2_to_xi(pendulum["yfit"], tp, fanh_no)

    compact_rel = rel_rmse(xi, compact_xi)
    pendulum_rel = rel_rmse(xi, pendulum_xi)

    stem = data_path.stem
    plot_out = outdir / f"xi_compare_models_vs_tp_{stem}.png"
    txt_out = outdir / f"xi_compare_models_vs_tp_{stem}.txt"
    make_plot(theta0, tp, xi, compact_xi, pendulum_xi, compact_rel, pendulum_rel, plot_out, args.dpi)
    save_summary(txt_out, compact["name"], compact_rel, pendulum_rel)

    print(f"Loaded: {data_path}")
    print(f"Saved: {plot_out}")
    print(f"Saved: {txt_out}")
    print(f"compact {compact['name']} rel={compact_rel:.4e}")
    print(f"pendulum_log_plus_const rel={pendulum_rel:.4e}")


if __name__ == "__main__":
    main()
