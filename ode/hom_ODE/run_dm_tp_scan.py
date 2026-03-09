import numpy as np
import matplotlib.pyplot as plt

from hom_ode import solve_noPT, solve_PT


# ---------------- CONFIG ----------------
THETA0_LIST = [0.2618, 0.7854, 1.309, 1.833, 2.356, 2.88]
FAST_FIRST_PASS = True

# Direct t_p scan (in M_phi^{-1})
TP_MIN = 5.0e-2
TP_MAX = 1.0e2
N_TP = 16 if FAST_FIRST_PASS else 40

T_OSC_NOPT = 1.5  # from 3H=1 with H=1/(2t)

# ODE numerics
T_START_NOPT = 1e-3
T_END_MIN = 300.0 if FAST_FIRST_PASS else 800.0
EXTRA_AFTER = 150.0 if FAST_FIRST_PASS else 300.0
RTOL = 1e-6 if FAST_FIRST_PASS else 3e-8
ATOL = 1e-8 if FAST_FIRST_PASS else 1e-10
METHOD = "DOP853"
LATE_FRAC = 0.25
LATE_MODE = "time_weighted"  # or "median"

# outputs
OUTDIR = "."
DPI = 220
PLOT_LOGX = True
PLOT_LOGY = False
# ----------------------------------------


def potential(theta0):
    return 1.0 - np.cos(theta0)


def main():
    tp_grid = np.logspace(np.log10(TP_MIN), np.log10(TP_MAX), N_TP)

    print("Precomputing no-PT reference quantities...")
    ref_map = {}
    for th0 in THETA0_LIST:
        ea3_nopt, _ = solve_noPT(
            th0,
            t_start=T_START_NOPT,
            t_end=T_END_MIN,
            method=METHOD,
            rtol=RTOL,
            atol=ATOL,
            late_frac=LATE_FRAC,
            late_mode=LATE_MODE,
        )
        v0 = potential(th0)
        if (not np.isfinite(ea3_nopt)) or (ea3_nopt <= 0.0) or (v0 <= 0.0):
            ref_map[th0] = {
                "Ea3_noPT": np.nan,
                "f_anh_noPT": np.nan,
                "Vtheta": v0,
            }
            print(f"  theta0={th0:.4f}  noPT invalid")
            continue

        f_nopt = ea3_nopt / (v0 * (T_OSC_NOPT ** 1.5))
        ref_map[th0] = {
            "Ea3_noPT": ea3_nopt,
            "f_anh_noPT": f_nopt,
            "Vtheta": v0,
        }
        print(
            f"  theta0={th0:.4f}  Ea3_noPT={ea3_nopt:.6g}  "
            f"f_anh_noPT={f_nopt:.6g}"
        )

    rows = []
    out_txt = f"{OUTDIR}/dm_tp_scan_results.txt"
    header = (
        "# theta0  t_p  x_tp_over_tosc  Ea3_PT  Ea3_noPT  "
        "f_anh_PT  f_anh_noPT  xi_DM\n"
    )
    with open(out_txt, "w") as f:
        f.write(header)

    for th0 in THETA0_LIST:
        print(f"Scanning theta0={th0:.4f} ({N_TP} t_p points)...")
        ref = ref_map[th0]
        ea3_ref = ref["Ea3_noPT"]
        f_ref = ref["f_anh_noPT"]
        v0 = ref["Vtheta"]

        for tp in tp_grid:
            ea3_pt, _ = solve_PT(
                th0,
                tp,
                t_end_min=T_END_MIN,
                extra_after=EXTRA_AFTER,
                method=METHOD,
                rtol=RTOL,
                atol=ATOL,
                late_frac=LATE_FRAC,
                late_mode=LATE_MODE,
            )

            x = tp / T_OSC_NOPT
            if (not np.isfinite(ea3_pt)) or (ea3_pt <= 0.0) or (not np.isfinite(v0)) or (v0 <= 0.0):
                row = [th0, tp, x, np.nan, ea3_ref, np.nan, f_ref, np.nan]
                rows.append(row)
                with open(out_txt, "a") as f:
                    f.write(
                        f"{row[0]:.8g}  {row[1]:.10e}  {row[2]:.10e}  {row[3]:.10e}  "
                        f"{row[4]:.10e}  {row[5]:.10e}  {row[6]:.10e}  {row[7]:.10e}\n"
                    )
                continue

            f_pt = ea3_pt / (v0 * (tp ** 1.5))
            xi = ea3_pt / ea3_ref if np.isfinite(ea3_ref) and ea3_ref > 0 else np.nan
            row = [th0, tp, x, ea3_pt, ea3_ref, f_pt, f_ref, xi]
            rows.append(row)
            with open(out_txt, "a") as f:
                f.write(
                    f"{row[0]:.8g}  {row[1]:.10e}  {row[2]:.10e}  {row[3]:.10e}  "
                    f"{row[4]:.10e}  {row[5]:.10e}  {row[6]:.10e}  {row[7]:.10e}\n"
                )

    arr = np.array(rows, dtype=np.float64)
    print("Saved table:", out_txt)

    # Plot 1: xi_DM vs x for each theta0
    fig1, ax1 = plt.subplots(figsize=(7.5, 5.0))
    colors = plt.cm.plasma(np.linspace(0.0, 0.9, len(THETA0_LIST)))
    for ci, th0 in enumerate(THETA0_LIST):
        m = (arr[:, 0] == th0) & np.isfinite(arr[:, 2]) & np.isfinite(arr[:, 7])
        if m.sum() < 2:
            continue
        idx = np.argsort(arr[m, 2])
        x = arr[m, 2][idx]
        y = arr[m, 7][idx]
        ax1.plot(x, y, lw=2, color=colors[ci], label=rf"$\theta_0={th0:.3g}$")
    ax1.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
    ax1.set_xlabel(r"$x=t_p/t_{\rm osc}^{\rm noPT}$")
    ax1.set_ylabel(r"$\xi_{\rm DM}=Ea^3_{\rm PT}/Ea^3_{\rm noPT}$")
    ax1.grid(alpha=0.25)
    if PLOT_LOGX:
        ax1.set_xscale("log")
    if PLOT_LOGY:
        ax1.set_yscale("log")
    ax1.legend(frameon=False, fontsize=8, ncol=2)
    fig1.tight_layout()
    out1 = f"{OUTDIR}/dm_xi_vs_x.png"
    fig1.savefig(out1, dpi=DPI)
    plt.close(fig1)
    print("Saved:", out1)

    # Plot 2: f_anh_PT vs x for each theta0
    fig2, ax2 = plt.subplots(figsize=(7.5, 5.0))
    for ci, th0 in enumerate(THETA0_LIST):
        m = (arr[:, 0] == th0) & np.isfinite(arr[:, 2]) & np.isfinite(arr[:, 5])
        if m.sum() < 2:
            continue
        idx = np.argsort(arr[m, 2])
        x = arr[m, 2][idx]
        y = arr[m, 5][idx]
        ax2.plot(x, y, lw=2, color=colors[ci], label=rf"$\theta_0={th0:.3g}$")
    ax2.set_xlabel(r"$x=t_p/t_{\rm osc}^{\rm noPT}$")
    ax2.set_ylabel(r"$f_{\rm anh}(\theta_0,t_p)$")
    ax2.grid(alpha=0.25)
    if PLOT_LOGX:
        ax2.set_xscale("log")
    if PLOT_LOGY:
        ax2.set_yscale("log")
    ax2.legend(frameon=False, fontsize=8, ncol=2)
    fig2.tight_layout()
    out2 = f"{OUTDIR}/dm_fanh_vs_x.png"
    fig2.savefig(out2, dpi=DPI)
    plt.close(fig2)
    print("Saved:", out2)

    print("ALL DONE.")


if __name__ == "__main__":
    main()
