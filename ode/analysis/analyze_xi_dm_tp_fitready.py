import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Analyze the direct t_p scan table from run_dm_tp_fitready.py "
            "and produce readable xi-derived plots."
        )
    )
    p.add_argument(
        "--data",
        type=str,
        default="ode/analysis/data/dm_tp_fitready_H1p000.txt",
        help="Path to dm_tp_fitready_*.txt",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="ode/analysis/results",
        help="Directory for output plots",
    )
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def load_table(path):
    arr = np.loadtxt(path, comments="#")
    if arr.ndim != 2 or arr.shape[1] < 10:
        raise ValueError(f"Expected at least 10 columns in {path}, got shape {arr.shape}")
    return arr


def make_single_plot(tp, y, theta0, ylabel, title, out_path, dpi, xlim=None):
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    th_unique = np.unique(theta0)
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(th_unique)))

    for color, th0 in zip(colors, th_unique):
        mask = np.isclose(theta0, th0) & np.isfinite(tp) & np.isfinite(y) & (tp > 0) & (y > 0)
        if mask.sum() < 2:
            continue
        idx = np.argsort(tp[mask])
        ax.plot(tp[mask][idx], y[mask][idx], "-o", ms=3.5, lw=1.8, color=color, label=rf"$\theta_0={th0:.3g}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t_p$")
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_panel_plot(tp, curves, theta0, titles, ylabels, suptitle, out_path, dpi):
    th_unique = np.unique(theta0)
    n_panels = len(th_unique)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(11.5, 3.8 * nrows),
        sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(th_unique)))

    for panel_idx, th0 in enumerate(th_unique):
        ax = axes[panel_idx]
        mask = np.isclose(theta0, th0) & np.isfinite(tp) & (tp > 0)
        idx = np.argsort(tp[mask])
        x = tp[mask][idx]

        for color, y, label in zip(colors[: len(curves)], curves, titles):
            y_sel = y[mask][idx]
            good = np.isfinite(y_sel) & (y_sel > 0)
            if good.sum() < 2:
                continue
            ax.plot(x[good], y_sel[good], "-o", ms=3.0, lw=1.5, color=color, label=label)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(rf"$\theta_0={th0:.3g}$")
        ax.grid(alpha=0.25)
        if panel_idx >= (nrows - 1) * ncols:
            ax.set_xlabel(r"$t_p$")
        if panel_idx % 2 == 0:
            ax.set_ylabel(ylabels)
        ax.legend(frameon=False, fontsize=7)

    for panel_idx in range(n_panels, len(axes)):
        axes[panel_idx].axis("off")

    fig.suptitle(suptitle, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_nopt_plot(theta0, fanh_no, h_star, out_path, dpi):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    mask = np.isfinite(theta0) & np.isfinite(fanh_no) & (fanh_no > 0)
    th = theta0[mask]
    y = fanh_no[mask]
    idx = np.argsort(th)
    ax.plot(th[idx], y[idx], "-o", ms=5, lw=1.8, color="tab:blue")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$f_{\rm anh}^{\rm noPT}(\theta_0)$")
    ax.set_title(rf"No-PT anharmonic factor at $H_*={h_star:g}$ reference scan")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    data_path = Path(args.data).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    arr = load_table(data_path)
    h_star = arr[0, 0]
    theta0 = arr[:, 2]
    tp = arr[:, 3]
    ea3_pt = arr[:, 5]
    ea3_no = arr[:, 6]
    fanh_pt = arr[:, 7]
    fanh_no = arr[:, 8]
    xi = arr[:, 9]
    t_star = arr[0, 1]

    xi_over_tp32 = xi / np.power(tp, 1.5)
    xi_over_fanh_no_q1_tp32 = xi / (fanh_no * np.power(tp, 1.5))
    xi_over_fanh_no_q2_tp32 = xi / (np.square(fanh_no) * np.power(tp, 1.5))
    xi_over_fanh_no_q3_tp32 = xi / (np.power(fanh_no, 3) * np.power(tp, 1.5))
    xi_fanh_no_over_tp32 = xi * fanh_no / np.power(tp, 1.5)
    y2_fanh_no = xi * np.square(fanh_no) / np.power(tp, 1.5)
    y4 = xi * np.power(fanh_no, 3) / np.power(tp, 1.5)
    low_tp_max = min(np.nanmax(tp), 4.0 * t_star)

    make_single_plot(
        tp,
        xi,
        theta0,
        r"$\xi_{\rm DM}$",
        rf"Direct $t_p$ scan: $\xi_{{\rm DM}}$ at $H_*={h_star:g}$",
        outdir / f"xi_vs_tp_{data_path.stem}.png",
        args.dpi,
    )
    make_single_plot(
        tp,
        xi_over_tp32,
        theta0,
        r"$\xi_{\rm DM}/t_p^{3/2}$",
        rf"Direct $t_p$ scan: $\xi_{{\rm DM}}/t_p^{{3/2}}$ at $H_*={h_star:g}$",
        outdir / f"xi_over_tp32_vs_tp_{data_path.stem}.png",
        args.dpi,
    )
    make_single_plot(
        tp,
        xi_over_fanh_no_q1_tp32,
        theta0,
        r"$\xi_{\rm DM}/\left(f_{\rm anh}^{\rm noPT} t_p^{3/2}\right)$",
        rf"Direct $t_p$ scan: $\xi_{{\rm DM}}/(f_{{\rm anh}}^{{\rm noPT}} t_p^{{3/2}})$ at $H_*={h_star:g}$",
        outdir / f"xi_over_fanh_noPT_q1_tp32_vs_tp_{data_path.stem}.png",
        args.dpi,
    )
    make_single_plot(
        tp,
        xi_over_fanh_no_q2_tp32,
        theta0,
        r"$\xi_{\rm DM}/\left((f_{\rm anh}^{\rm noPT})^2 t_p^{3/2}\right)$",
        rf"Direct $t_p$ scan: $\xi_{{\rm DM}}/((f_{{\rm anh}}^{{\rm noPT}})^2 t_p^{{3/2}})$ at $H_*={h_star:g}$",
        outdir / f"xi_over_fanh_noPT_q2_tp32_vs_tp_{data_path.stem}.png",
        args.dpi,
    )
    make_single_plot(
        tp,
        xi_over_fanh_no_q3_tp32,
        theta0,
        r"$\xi_{\rm DM}/\left((f_{\rm anh}^{\rm noPT})^3 t_p^{3/2}\right)$",
        rf"Direct $t_p$ scan: $\xi_{{\rm DM}}/((f_{{\rm anh}}^{{\rm noPT}})^3 t_p^{{3/2}})$ at $H_*={h_star:g}$",
        outdir / f"xi_over_fanh_noPT_q3_tp32_vs_tp_{data_path.stem}.png",
        args.dpi,
    )
    make_single_plot(
        tp,
        xi_fanh_no_over_tp32,
        theta0,
        r"$\xi_{\rm DM} f_{\rm anh}^{\rm noPT}(\theta_0)/t_p^{3/2}$",
        rf"Direct $t_p$ scan: $\xi_{{\rm DM}} f_{{\rm anh}}^{{\rm noPT}}/t_p^{{3/2}}$ at $H_*={h_star:g}$",
        outdir / f"xi_fanh_noPT_over_tp32_vs_tp_{data_path.stem}.png",
        args.dpi,
    )
    make_single_plot(
        tp,
        xi_fanh_no_over_tp32,
        theta0,
        r"$\xi_{\rm DM} f_{\rm anh}^{\rm noPT}(\theta_0)/t_p^{3/2}$",
        rf"Low-$t_p$ zoom: $\xi_{{\rm DM}} f_{{\rm anh}}^{{\rm noPT}}/t_p^{{3/2}}$ at $H_*={h_star:g}$",
        outdir / f"xi_fanh_noPT_over_tp32_vs_tp_lowtp_{data_path.stem}.png",
        args.dpi,
        xlim=(max(np.nanmin(tp) * 0.995, 1e-12), low_tp_max),
    )
    make_single_plot(
        tp,
        y2_fanh_no,
        theta0,
        r"$Y_2 f_{\rm anh}^{\rm noPT} = \xi_{\rm DM}\left(f_{\rm anh}^{\rm noPT}\right)^2/t_p^{3/2}$",
        rf"Direct $t_p$ scan: $Y_2 f_{{\rm anh}}^{{\rm noPT}}$ at $H_*={h_star:g}$",
        outdir / f"y2_fanh_noPT_vs_tp_{data_path.stem}.png",
        args.dpi,
    )
    make_single_plot(
        tp,
        y4,
        theta0,
        r"$Y_4 = \xi_{\rm DM}\left(f_{\rm anh}^{\rm noPT}\right)^3/t_p^{3/2}$",
        rf"Direct $t_p$ scan: $Y_4$ at $H_*={h_star:g}$",
        outdir / f"y4_vs_tp_{data_path.stem}.png",
        args.dpi,
    )

    make_panel_plot(
        tp,
        [xi, xi_over_tp32, xi_fanh_no_over_tp32],
        theta0,
        [
            r"$\xi_{\rm DM}$",
            r"$\xi_{\rm DM}/t_p^{3/2}$",
            r"$\xi_{\rm DM} f_{\rm anh}^{\rm noPT}/t_p^{3/2}$",
        ],
        r"log-scaled quantities",
        rf"Per-$\theta_0$ view of the direct $t_p$ scan at $H_*={h_star:g}$",
        outdir / f"xi_panel_{data_path.stem}.png",
        args.dpi,
    )

    ref_out = outdir / f"noPT_reference_{data_path.stem}.txt"
    th_unique = np.unique(theta0)
    with open(ref_out, "w") as f:
        f.write("# theta0 Ea3_noPT f_anh_noPT\n")
        for th0 in th_unique:
            mask = np.isclose(theta0, th0)
            f.write(
                f"{th0:.8g} {ea3_no[mask][0]:.10e} {fanh_no[mask][0]:.10e}\n"
            )

    fanh_no_unique = np.array([fanh_no[np.isclose(theta0, th0)][0] for th0 in th_unique], dtype=np.float64)
    nopt_plot = outdir / f"fanh_noPT_vs_theta0_{data_path.stem}.png"
    make_nopt_plot(th_unique, fanh_no_unique, h_star, nopt_plot, args.dpi)

    print(f"Loaded: {data_path}")
    print(f"H_star={h_star:g}")
    print(f"Saved: {outdir / f'xi_vs_tp_{data_path.stem}.png'}")
    print(f"Saved: {outdir / f'xi_over_tp32_vs_tp_{data_path.stem}.png'}")
    print(f"Saved: {outdir / f'xi_over_fanh_noPT_q1_tp32_vs_tp_{data_path.stem}.png'}")
    print(f"Saved: {outdir / f'xi_over_fanh_noPT_q2_tp32_vs_tp_{data_path.stem}.png'}")
    print(f"Saved: {outdir / f'xi_over_fanh_noPT_q3_tp32_vs_tp_{data_path.stem}.png'}")
    print(f"Saved: {outdir / f'xi_fanh_noPT_over_tp32_vs_tp_{data_path.stem}.png'}")
    print(f"Saved: {outdir / f'xi_fanh_noPT_over_tp32_vs_tp_lowtp_{data_path.stem}.png'}")
    print(f"Saved: {outdir / f'y2_fanh_noPT_vs_tp_{data_path.stem}.png'}")
    print(f"Saved: {outdir / f'y4_vs_tp_{data_path.stem}.png'}")
    print(f"Saved: {outdir / f'xi_panel_{data_path.stem}.png'}")
    print(f"Saved: {ref_out}")
    print(f"Saved: {nopt_plot}")


if __name__ == "__main__":
    main()
