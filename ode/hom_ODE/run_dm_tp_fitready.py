import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hom_ode import solve_noPT, solve_PT


THETA0_DEFAULT = [0.2618, 0.7854, 1.309, 1.833, 2.356, 2.88]
T_OSC_NOPT = 1.5


def potential(theta0):
    return 1.0 - np.cos(theta0)


def unique_sorted(arr):
    return np.array(sorted(set(float(x) for x in arr)), dtype=np.float64)


def build_fit_tp_grid(
    h_star,
    tp_min_factor,
    tp_dense_factor,
    tp_mid_factor,
    tp_max_factor,
    n_dense,
    n_mid,
    n_tail,
):
    t_star = 1.0 / (2.0 * float(h_star))
    tp_min = max(1e-6, tp_min_factor * t_star)
    tp_dense = max(tp_min * 1.001, tp_dense_factor * t_star)
    tp_mid = max(tp_dense * 1.001, tp_mid_factor * t_star)
    tp_max = max(tp_mid * 1.001, tp_max_factor * t_star)

    seg1 = np.logspace(np.log10(tp_min), np.log10(tp_dense), int(n_dense))
    seg2 = np.logspace(np.log10(tp_dense), np.log10(tp_mid), int(n_mid))
    seg3 = np.logspace(np.log10(tp_mid), np.log10(tp_max), int(n_tail))
    return unique_sorted(np.concatenate([seg1, seg2, seg3]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate fit-ready f_anh(theta0, t_p) for fixed H_star with t_p >= t_star cut."
    )
    parser.add_argument("--hstar", type=float, default=1.0, help="Fixed H_*/M_phi.")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory.")

    # fit-ready grid defaults
    parser.add_argument("--tp-min-factor", type=float, default=1.02)
    parser.add_argument("--tp-dense-factor", type=float, default=6.0)
    parser.add_argument("--tp-mid-factor", type=float, default=40.0)
    parser.add_argument("--tp-max-factor", type=float, default=250.0)
    parser.add_argument("--n-dense", type=int, default=32)
    parser.add_argument("--n-mid", type=int, default=22)
    parser.add_argument("--n-tail", type=int, default=18)

    parser.add_argument(
        "--theta0-list",
        type=float,
        nargs="+",
        default=THETA0_DEFAULT,
        help="List of theta0 values (radians).",
    )

    # numerics
    parser.add_argument("--t-start-nopt", type=float, default=1e-3)
    parser.add_argument("--t-end-min", type=float, default=900.0)
    parser.add_argument("--extra-after", type=float, default=500.0)
    parser.add_argument("--method", type=str, default="DOP853")
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--late-frac", type=float, default=0.30)
    parser.add_argument("--late-mode", type=str, default="time_weighted")

    parser.add_argument("--smoke", action="store_true", help="Small fast run for quick validation.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.smoke:
        args.n_dense = 6
        args.n_mid = 4
        args.n_tail = 3
        args.theta0_list = args.theta0_list[:3]
        args.t_end_min = 300.0
        args.extra_after = 150.0
        args.rtol = max(args.rtol, 1e-6)
        args.atol = max(args.atol, 1e-8)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    h_star = float(args.hstar)
    t_star = 1.0 / (2.0 * h_star)
    theta0_list = [float(x) for x in args.theta0_list]
    tp_grid = build_fit_tp_grid(
        h_star,
        args.tp_min_factor,
        args.tp_dense_factor,
        args.tp_mid_factor,
        args.tp_max_factor,
        args.n_dense,
        args.n_mid,
        args.n_tail,
    )

    tag = f"H{h_star:.3f}".replace(".", "p")
    out_txt = outdir / f"dm_tp_fitready_{tag}.txt"
    plot_fanh = outdir / f"dm_fanh_vs_tp_{tag}.png"
    plot_xi = outdir / f"dm_xi_vs_tp_{tag}.png"

    print(f"H*={h_star:g}  t*={t_star:.8g}")
    print(f"tp grid size={len(tp_grid)}  tp range=[{tp_grid.min():.6g}, {tp_grid.max():.6g}]")
    print(f"Output table: {out_txt}")

    header = (
        "# H_star t_star theta0 t_p x_tp_over_tosc Ea3_PT Ea3_noPT "
        "f_anh_PT f_anh_noPT xi_DM nsteps_PT nsteps_noPT\n"
    )
    with open(out_txt, "w") as f:
        f.write(header)

    print("Computing no-PT references...")
    ref = {}
    for th0 in theta0_list:
        ea3_no, sol_no = solve_noPT(
            th0,
            t_start=args.t_start_nopt,
            t_end=args.t_end_min,
            method=args.method,
            rtol=args.rtol,
            atol=args.atol,
            late_frac=args.late_frac,
            late_mode=args.late_mode,
        )
        nsteps_no = len(sol_no.t) if (sol_no is not None and hasattr(sol_no, "t")) else 0
        v0 = potential(th0)
        f_no = ea3_no / (v0 * (T_OSC_NOPT**1.5)) if (np.isfinite(ea3_no) and v0 > 0) else np.nan
        ref[th0] = {
            "Ea3_noPT": ea3_no,
            "f_anh_noPT": f_no,
            "Vtheta": v0,
            "nsteps_noPT": nsteps_no,
        }
        print(
            f"  theta0={th0:.4f} Ea3_noPT={ea3_no:.6g} "
            f"f_anh_noPT={f_no:.6g} nsteps={nsteps_no}"
        )

    rows = []
    for th0 in theta0_list:
        print(f"Scanning theta0={th0:.4f} over {len(tp_grid)} tp points...")
        ref_row = ref[th0]
        ea3_no = ref_row["Ea3_noPT"]
        f_no = ref_row["f_anh_noPT"]
        v0 = ref_row["Vtheta"]
        nsteps_no = ref_row["nsteps_noPT"]

        for tp in tp_grid:
            if tp < t_star:
                continue

            ea3_pt, sol_pt = solve_PT(
                th0,
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

            x = tp / T_OSC_NOPT
            if (not np.isfinite(ea3_pt)) or (ea3_pt <= 0) or (not np.isfinite(v0)) or (v0 <= 0):
                f_pt = np.nan
                xi = np.nan
            else:
                f_pt = ea3_pt / (v0 * (tp**1.5))
                xi = ea3_pt / ea3_no if (np.isfinite(ea3_no) and ea3_no > 0) else np.nan

            row = [
                h_star,
                t_star,
                th0,
                tp,
                x,
                ea3_pt,
                ea3_no,
                f_pt,
                f_no,
                xi,
                float(nsteps_pt),
                float(nsteps_no),
            ]
            rows.append(row)
            with open(out_txt, "a") as f:
                f.write(
                    f"{row[0]:.8g} {row[1]:.8g} {row[2]:.8g} {row[3]:.10e} {row[4]:.10e} "
                    f"{row[5]:.10e} {row[6]:.10e} {row[7]:.10e} {row[8]:.10e} {row[9]:.10e} "
                    f"{int(row[10])} {int(row[11])}\n"
                )

    arr = np.array(rows, dtype=np.float64)
    print(f"Wrote {len(rows)} rows.")

    # Plot f_anh vs tp by theta0
    fig1, ax1 = plt.subplots(figsize=(7.8, 5.2))
    th_unique = np.unique(arr[:, 2])
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(th_unique)))
    for c, th0 in zip(colors, th_unique):
        m = (arr[:, 2] == th0) & np.isfinite(arr[:, 3]) & np.isfinite(arr[:, 7])
        if m.sum() < 2:
            continue
        idx = np.argsort(arr[m, 3])
        tpv = arr[m, 3][idx]
        fav = arr[m, 7][idx]
        ax1.plot(tpv, fav, "-o", ms=3.5, lw=1.8, color=c, label=rf"$\theta_0={th0:.3g}$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$t_p$")
    ax1.set_ylabel(r"$f_{\rm anh}(\theta_0,t_p)$")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, ncol=2, fontsize=8)
    ax1.set_title(rf"Fit-ready $f_{{\rm anh}}$ scan at $H_*={h_star:g}$")
    fig1.tight_layout()
    fig1.savefig(plot_fanh, dpi=240)
    plt.close(fig1)
    print(f"Saved: {plot_fanh}")

    # Plot xi vs tp by theta0
    fig2, ax2 = plt.subplots(figsize=(7.8, 5.2))
    for c, th0 in zip(colors, th_unique):
        m = (arr[:, 2] == th0) & np.isfinite(arr[:, 3]) & np.isfinite(arr[:, 9])
        if m.sum() < 2:
            continue
        idx = np.argsort(arr[m, 3])
        tpv = arr[m, 3][idx]
        xiv = arr[m, 9][idx]
        ax2.plot(tpv, xiv, "-o", ms=3.5, lw=1.8, color=c, label=rf"$\theta_0={th0:.3g}$")
    ax2.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$t_p$")
    ax2.set_ylabel(r"$\xi_{\rm DM}$")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False, ncol=2, fontsize=8)
    ax2.set_title(rf"Fit-ready $\xi_{{\rm DM}}$ scan at $H_*={h_star:g}$")
    fig2.tight_layout()
    fig2.savefig(plot_xi, dpi=240)
    plt.close(fig2)
    print(f"Saved: {plot_xi}")
    print("ALL DONE.")


if __name__ == "__main__":
    main()
