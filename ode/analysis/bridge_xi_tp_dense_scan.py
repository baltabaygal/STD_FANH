import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ode.hom_ODE.hom_ode import solve_PT, solve_noPT


THETA0_DEFAULT = [0.2618, 0.7854, 1.309, 1.833, 2.356, 2.88]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build a dense xi(theta0, t_p) scan by bridging new low-t_p ODE "
            "points to an existing validated higher-t_p dataset."
        )
    )
    p.add_argument("--existing-data", type=str, default="ode/analysis/data/dm_tp_fitready_H1p000.txt")
    p.add_argument("--outdir", type=str, default="ode/analysis/results")
    p.add_argument("--theta0-list", type=float, nargs="+", default=THETA0_DEFAULT)
    p.add_argument("--tp-low-min", type=float, default=1.0e-3)
    p.add_argument("--tp-low-max", type=float, default=1.0)
    p.add_argument("--n-low", type=int, default=16)
    p.add_argument("--tp-high-max", type=float, default=1.0e2)
    p.add_argument("--jobs", type=int, default=0, help="0 means auto.")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--method", type=str, default="DOP853")
    p.add_argument("--rtol", type=float, default=1.0e-10)
    p.add_argument("--atol", type=float, default=1.0e-12)
    p.add_argument("--t-start-nopt", type=float, default=1.0e-5)
    p.add_argument("--t-end-min", type=float, default=300.0)
    p.add_argument("--extra-after", type=float, default=300.0)
    p.add_argument("--late-frac", type=float, default=0.30)
    p.add_argument("--late-mode", type=str, default="time_weighted")
    return p.parse_args()


def unique_sorted(values):
    return np.array(sorted(set(float(x) for x in values)), dtype=np.float64)


def build_tp_low_grid(tp_low_min, tp_low_max, n_low):
    return np.logspace(np.log10(tp_low_min), np.log10(tp_low_max), int(n_low))


def auto_jobs(njobs, ntheta):
    if njobs and njobs > 0:
        return int(njobs)
    cpu = os.cpu_count() or 1
    return max(1, min(ntheta, cpu))


def is_close_theta(theta_arr, theta0):
    return np.isclose(theta_arr, float(theta0), rtol=0.0, atol=1.0e-8)


def compute_low_theta(task):
    theta0, tp_low_grid, settings = task
    ea3_no, sol_no = solve_noPT(
        theta0,
        t_start=settings["t_start_nopt"],
        t_end=settings["t_end_min"],
        method=settings["method"],
        rtol=settings["rtol"],
        atol=settings["atol"],
        late_frac=settings["late_frac"],
        late_mode=settings["late_mode"],
    )
    nsteps_no = len(sol_no.t) if (sol_no is not None and hasattr(sol_no, "t")) else 0

    rows = []
    for tp in tp_low_grid:
        ea3_pt, sol_pt = solve_PT(
            theta0,
            tp,
            t_end_min=settings["t_end_min"],
            extra_after=settings["extra_after"],
            method=settings["method"],
            rtol=settings["rtol"],
            atol=settings["atol"],
            late_frac=settings["late_frac"],
            late_mode=settings["late_mode"],
        )
        nsteps_pt = len(sol_pt.t) if (sol_pt is not None and hasattr(sol_pt, "t")) else 0
        rows.append(
            (
                float(theta0),
                float(tp),
                float(ea3_pt / ea3_no),
                0,
                int(nsteps_pt),
                int(nsteps_no),
            )
        )
    return np.array(rows, dtype=np.float64)


def load_existing_rows(path, theta0_list, tp_low_max, tp_high_max):
    arr = np.loadtxt(path, comments="#")
    theta0 = arr[:, 2]
    tp = arr[:, 3]
    xi = arr[:, 9]
    nsteps_pt = arr[:, 10]
    nsteps_no = arr[:, 11]

    rows = []
    for th0 in theta0_list:
        mask = (
            is_close_theta(theta0, th0)
            & np.isfinite(tp)
            & np.isfinite(xi)
            & (tp > float(tp_low_max))
            & (tp <= float(tp_high_max))
        )
        if not np.any(mask):
            continue
        idx = np.argsort(tp[mask])
        tps = tp[mask][idx]
        xis = xi[mask][idx]
        npt = nsteps_pt[mask][idx]
        nno = nsteps_no[mask][idx]
        for tpi, xii, npti, nnoi in zip(tps, xis, npt, nno):
            rows.append((float(th0), float(tpi), float(xii), 1, float(npti), float(nnoi)))
    return np.array(rows, dtype=np.float64)


def merge_rows(low_rows, high_rows):
    if high_rows.size == 0:
        merged = np.array(low_rows, dtype=np.float64)
    else:
        merged = np.vstack([low_rows, high_rows]).astype(np.float64)
    order = np.lexsort((merged[:, 1], merged[:, 0]))
    return merged[order]


def save_scan(path, rows):
    with open(path, "w") as f:
        f.write("# theta0 tp xi source_id nsteps_pt nsteps_noPT\n")
        for row in rows:
            f.write(
                f"{row[0]:.8g} {row[1]:.10e} {row[2]:.10e} {int(row[3])} "
                f"{int(row[4])} {int(row[5])}\n"
            )


def save_summary(path, rows, args, jobs):
    theta0_list = unique_sorted(rows[:, 0])
    with open(path, "w") as f:
        f.write("# Dense xi(theta0, tp) bridge scan\n")
        f.write(
            f"# method={args.method} rtol={args.rtol:.3e} atol={args.atol:.3e} "
            f"t_start_nopt={args.t_start_nopt:.3e} t_end_min={args.t_end_min:.3e} extra_after={args.extra_after:.3e}\n"
        )
        f.write(
            f"# tp_low_min={args.tp_low_min:.3e} tp_low_max={args.tp_low_max:.3e} "
            f"n_low={args.n_low} tp_high_max={args.tp_high_max:.3e} jobs={jobs}\n"
        )
        f.write("# source_id: 0=new_low_ode 1=existing_data\n\n")
        f.write("# theta0 n_low n_existing tp_first xi_first tp_last xi_last tp_join_low tp_join_high xi_join_low xi_join_high\n")
        for th0 in theta0_list:
            mask = is_close_theta(rows[:, 0], th0)
            sub = rows[mask]
            low = sub[sub[:, 3] == 0]
            high = sub[sub[:, 3] == 1]
            tp_first = sub[0, 1]
            xi_first = sub[0, 2]
            tp_last = sub[-1, 1]
            xi_last = sub[-1, 2]
            if low.size > 0:
                tp_join_low = low[-1, 1]
                xi_join_low = low[-1, 2]
            else:
                tp_join_low = np.nan
                xi_join_low = np.nan
            if high.size > 0:
                tp_join_high = high[0, 1]
                xi_join_high = high[0, 2]
            else:
                tp_join_high = np.nan
                xi_join_high = np.nan
            f.write(
                f"{th0:.8g} {len(low)} {len(high)} {tp_first:.10e} {xi_first:.10e} "
                f"{tp_last:.10e} {xi_last:.10e} {tp_join_low:.10e} {tp_join_high:.10e} "
                f"{xi_join_low:.10e} {xi_join_high:.10e}\n"
            )


def make_plot(rows, out_path, dpi, tp_high_max):
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    theta0_list = unique_sorted(rows[:, 0])
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(theta0_list)))

    for color, th0 in zip(colors, theta0_list):
        mask = is_close_theta(rows[:, 0], th0)
        sub = rows[mask]
        idx = np.argsort(sub[:, 1])
        tp = sub[idx, 1]
        xi = sub[idx, 2]
        src = sub[idx, 3].astype(int)

        ax.plot(tp, xi, "-", lw=1.7, color=color, label=rf"$\theta_0={th0:.3g}$")
        ax.plot(tp[src == 0], xi[src == 0], "o", ms=3.8, color=color)
        ax.plot(tp[src == 1], xi[src == 1], ".", ms=4.0, color=color)

    ax.axhline(1.0, color="black", lw=1.1, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(float(np.min(rows[:, 1])) * 0.95, float(tp_high_max) * 1.02)
    ax.set_xlabel(r"$t_p$")
    ax.set_ylabel(r"$\xi$")
    ax.set_title(r"Dense merged $\xi(\theta_0, t_p)$ scan")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    theta0_list = [float(x) for x in args.theta0_list]
    tp_low_grid = build_tp_low_grid(args.tp_low_min, args.tp_low_max, args.n_low)
    jobs = auto_jobs(args.jobs, len(theta0_list))

    settings = {
        "method": args.method,
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "t_start_nopt": float(args.t_start_nopt),
        "t_end_min": float(args.t_end_min),
        "extra_after": float(args.extra_after),
        "late_frac": float(args.late_frac),
        "late_mode": str(args.late_mode),
    }

    tasks = [(th0, tp_low_grid, settings) for th0 in theta0_list]
    if jobs == 1:
        low_chunks = [compute_low_theta(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            low_chunks = list(ex.map(compute_low_theta, tasks))
    low_rows = np.vstack(low_chunks).astype(np.float64)

    high_rows = load_existing_rows(
        Path(args.existing_data).resolve(),
        theta0_list,
        args.tp_low_max,
        args.tp_high_max,
    )
    merged = merge_rows(low_rows, high_rows)

    stem = "xi_dense_bridge_H1p000"
    scan_out = outdir / f"{stem}.txt"
    summary_out = outdir / f"{stem}_summary.txt"
    plot_out = outdir / f"{stem}.png"

    save_scan(scan_out, merged)
    save_summary(summary_out, merged, args, jobs)
    make_plot(merged, plot_out, args.dpi, args.tp_high_max)

    print(f"Saved: {scan_out}")
    print(f"Saved: {summary_out}")
    print(f"Saved: {plot_out}")
    print(f"Low-grid points per theta={len(tp_low_grid)}; jobs={jobs}")


if __name__ == "__main__":
    main()
