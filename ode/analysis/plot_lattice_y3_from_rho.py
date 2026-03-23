import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ode.hom_ODE.percolation import PercolationCache


TARGET_H_DEFAULT = [1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build lattice Y3 from xi and rho_noPT data. "
            "Uses the repo Y3 convention Y3 = xi * fanh_noPT^2 / tp^(3/2), "
            "and also saves the single-power quantity xi * fanh_noPT / tp^(3/2)."
        )
    )
    p.add_argument("--lattice-xi", type=str, default="lattice_data/data/energy_ratio_by_theta_data_v9.txt")
    p.add_argument("--lattice-rho", type=str, default="lattice_data/data/rho_noPT_data.txt")
    p.add_argument("--ode-xi", type=str, default="ode/xi_DM_ODE_results.txt")
    p.add_argument("--ode-nopt", type=str, default="ode/analysis/results/noPT_reference_dm_tp_fitready_H1p000.txt")
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--target-h", type=float, nargs="+", default=TARGET_H_DEFAULT)
    p.add_argument(
        "--outdir",
        type=str,
        default="ode/analysis/results/lattice_fit/lattice_y3_from_rho_v9",
    )
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def potential(theta0):
    theta0 = np.asarray(theta0, dtype=np.float64)
    return 1.0 - np.cos(theta0)


def rel_rmse(y, yfit):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square((yfit - y) / y))))


def load_lattice_ratio(path, target_h):
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
        "xi": arr[:, 4].astype(np.float64),
        "sem": sem.astype(np.float64),
    }


def load_lattice_rho(path):
    arr = np.loadtxt(path, skiprows=1)
    return {
        "theta0": arr[:, 0].astype(np.float64),
        "hstar": arr[:, 1].astype(np.float64),
        "rho": arr[:, 2].astype(np.float64),
    }


def load_ode_ratio(path, vw, target_h):
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


def load_ode_nopt(path):
    arr = np.loadtxt(path)
    return {
        "theta0": arr[:, 0].astype(np.float64),
        "fanh_no": arr[:, 2].astype(np.float64),
    }


def nearest_theta(values, theta0, atol=5.0e-4):
    values = np.asarray(values, dtype=np.float64)
    idx = int(np.argmin(np.abs(values - float(theta0))))
    if abs(values[idx] - float(theta0)) > atol:
        raise RuntimeError(f"No theta match for theta0={theta0:.10f}")
    return idx


def fit_lattice_nopt_scale(rho_lat, ode_nopt):
    raw_vals = []
    ref_vals = []
    for th0, hstar, rho in zip(rho_lat["theta0"], rho_lat["hstar"], rho_lat["rho"]):
        raw = rho / (potential(th0) * (float(hstar) ** 1.5))
        idx = nearest_theta(ode_nopt["theta0"], th0)
        raw_vals.append(raw)
        ref_vals.append(float(ode_nopt["fanh_no"][idx]))
    raw_vals = np.array(raw_vals, dtype=np.float64)
    ref_vals = np.array(ref_vals, dtype=np.float64)
    scale = float(np.dot(raw_vals, ref_vals) / np.dot(raw_vals, raw_vals))
    return {
        "scale": scale,
        "raw_vals": raw_vals,
        "ref_vals": ref_vals,
        "rel_rmse": rel_rmse(ref_vals, scale * raw_vals),
    }


def build_lattice_fanh_lookup(rho_lat, scale):
    return {
        "theta0": rho_lat["theta0"].copy(),
        "hstar": rho_lat["hstar"].copy(),
        "fanh_no": np.array(
            [float(scale) * rho / (potential(th0) * (float(hstar) ** 1.5)) for th0, hstar, rho in zip(rho_lat["theta0"], rho_lat["hstar"], rho_lat["rho"])],
            dtype=np.float64,
        ),
    }


def lookup_lattice_fanh(lookup, theta0, hstar):
    mask = np.isclose(lookup["hstar"], float(hstar), rtol=0.0, atol=1.0e-12)
    if not np.any(mask):
        raise RuntimeError(f"Missing lattice rho_noPT H* entry for H*={hstar:g}")
    theta_vals = lookup["theta0"][mask]
    idx_local = int(np.argmin(np.abs(theta_vals - float(theta0))))
    if abs(theta_vals[idx_local] - float(theta0)) > 5.0e-4:
        raise RuntimeError(f"Missing lattice rho_noPT entry for theta0={theta0:.10f}, H*={hstar:g}")
    return float(lookup["fanh_no"][mask][idx_local])


def build_lattice_dataset(ratio_lat, rho_lookup, vw):
    perc = PercolationCache()
    tp = np.array(
        [perc.get(float(h), float(bh), float(vw)) for h, bh in zip(ratio_lat["hstar"], ratio_lat["beta_over_h"])],
        dtype=np.float64,
    )
    fanh_no = np.array(
        [lookup_lattice_fanh(rho_lookup, th, h) for th, h in zip(ratio_lat["theta0"], ratio_lat["hstar"])],
        dtype=np.float64,
    )
    y3 = ratio_lat["xi"] * np.square(fanh_no) / np.power(tp, 1.5)
    fanh = ratio_lat["xi"] * fanh_no / np.power(tp, 1.5)
    out = dict(ratio_lat)
    out["tp"] = tp
    out["fanh_no"] = fanh_no
    out["y3"] = y3
    out["fanh"] = fanh
    return out


def build_ode_dataset(ode_ratio, ode_nopt):
    fanh_no = []
    for th0 in ode_ratio["theta0"]:
        idx = nearest_theta(ode_nopt["theta0"], th0)
        fanh_no.append(float(ode_nopt["fanh_no"][idx]))
    fanh_no = np.array(fanh_no, dtype=np.float64)
    y3 = ode_ratio["xi"] * np.square(fanh_no) / np.power(ode_ratio["tp"], 1.5)
    fanh = ode_ratio["xi"] * fanh_no / np.power(ode_ratio["tp"], 1.5)
    out = dict(ode_ratio)
    out["fanh_no"] = fanh_no
    out["y3"] = y3
    out["fanh"] = fanh
    return out


def select_h(data, hstar):
    mask = np.isclose(data["hstar"], float(hstar), rtol=0.0, atol=1.0e-12)
    return {key: val[mask] for key, val in data.items()}


def make_panel_plot(lat_h, ode_h, ykey, ylabel, out_path, dpi, title):
    theta_u = np.array(sorted(np.unique(lat_h["theta0"])), dtype=np.float64)
    ntheta = len(theta_u)
    ncols = 3 if ntheta > 4 else 2
    nrows = int(np.ceil(ntheta / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.0 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, th0 in zip(axes, theta_u):
        lmask = np.isclose(lat_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
        omask = np.isclose(ode_h["theta0"], th0, rtol=0.0, atol=5.0e-4)

        ltp = lat_h["tp"][lmask]
        ly = lat_h[ykey][lmask]
        lsem = lat_h["sem"][lmask]
        lidx = np.argsort(ltp)

        otp = ode_h["tp"][omask]
        oy = ode_h[ykey][omask]
        oidx = np.argsort(otp)

        ax.plot(ltp[lidx], ly[lidx], "ko", ms=3.6, label="lattice")
        ax.plot(otp[oidx], oy[oidx], color="tab:blue", lw=1.8, label="ODE")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$t_p$")
        ax.set_ylabel(ylabel)
        ax.set_title(rf"$\theta_0={th0:.3g}$")
        ax.grid(alpha=0.25)

    for ax in axes[ntheta:]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def compare_lattice_to_ode_interp(lat_h, ode_h, ykey):
    theta_u = np.array(sorted(np.unique(lat_h["theta0"])), dtype=np.float64)
    lat_vals = []
    ode_interp_vals = []
    for th0 in theta_u:
        lmask = np.isclose(lat_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
        omask = np.isclose(ode_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
        lbeta = lat_h["beta_over_h"][lmask]
        ly = lat_h[ykey][lmask]
        obeta = ode_h["beta_over_h"][omask]
        oy = ode_h[ykey][omask]
        order = np.argsort(obeta)
        interp = np.interp(np.log(lbeta), np.log(obeta[order]), oy[order])
        lat_vals.append(ly)
        ode_interp_vals.append(interp)
    lat_vals = np.concatenate(lat_vals)
    ode_interp_vals = np.concatenate(ode_interp_vals)
    return rel_rmse(lat_vals, ode_interp_vals)


def save_summary(path, args, scale_fit, lat_data, ode_data):
    with open(path, "w") as f:
        f.write("# Lattice Y3 reconstruction from rho_noPT_data\n")
        f.write(f"# lattice_xi={Path(args.lattice_xi).resolve()}\n")
        f.write(f"# lattice_rho={Path(args.lattice_rho).resolve()}\n")
        f.write(f"# ode_xi={Path(args.ode_xi).resolve()}\n")
        f.write(f"# ode_nopt={Path(args.ode_nopt).resolve()}\n")
        f.write(f"# fixed_vw={float(args.fixed_vw):.6f}\n")
        f.write("# absolute lattice noPT reconstruction:\n")
        f.write("# fanh_noPT^lat(theta,H) = K * rho_noPT(theta,H) / [(1-cos theta) H^(3/2)]\n")
        f.write(f"# fitted_K={scale_fit['scale']:.10e} rel_rmse_vs_ODE={scale_fit['rel_rmse']:.10e}\n")
        f.write("# Y3 convention used here: Y3 = xi * fanh_noPT^2 / tp^(3/2)\n")
        f.write("# single-power quantity also saved: fanh = xi * fanh_noPT / tp^(3/2)\n\n")
        f.write("# hstar theta0 fanh_no_lattice fanh_no_ode rel_delta_fanh_no\n")
        for hstar in args.target_h:
            lat_h = select_h(lat_data, float(hstar))
            ode_h = select_h(ode_data, float(hstar))
            theta_u = np.array(sorted(np.unique(lat_h["theta0"])), dtype=np.float64)
            for th0 in theta_u:
                lmask = np.isclose(lat_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
                omask = np.isclose(ode_h["theta0"], th0, rtol=0.0, atol=5.0e-4)
                f_lat = float(lat_h["fanh_no"][lmask][0])
                f_ode = float(ode_h["fanh_no"][omask][0])
                f.write(
                    f"{float(hstar):.8g} {float(th0):.10f} {f_lat:.10e} {f_ode:.10e} {(f_lat / f_ode - 1.0):.10e}\n"
                )
        f.write("\n")
        f.write("# hstar metric rel_rmse_lattice_vs_ODE_Y3 rel_rmse_lattice_vs_ODE_fanh\n")
        for hstar in args.target_h:
            lat_h = select_h(lat_data, float(hstar))
            ode_h = select_h(ode_data, float(hstar))
            rel_y3 = compare_lattice_to_ode_interp(lat_h, ode_h, "y3")
            rel_fanh = compare_lattice_to_ode_interp(lat_h, ode_h, "fanh")
            f.write(f"{float(hstar):.8g} rel {rel_y3:.10e} {rel_fanh:.10e}\n")


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ratio_lat = load_lattice_ratio(Path(args.lattice_xi).resolve(), args.target_h)
    rho_lat = load_lattice_rho(Path(args.lattice_rho).resolve())
    ode_ratio = load_ode_ratio(Path(args.ode_xi).resolve(), args.fixed_vw, args.target_h)
    ode_nopt = load_ode_nopt(Path(args.ode_nopt).resolve())

    scale_fit = fit_lattice_nopt_scale(rho_lat, ode_nopt)
    rho_lookup = build_lattice_fanh_lookup(rho_lat, scale_fit["scale"])
    lat_data = build_lattice_dataset(ratio_lat, rho_lookup, args.fixed_vw)
    ode_data = build_ode_dataset(ode_ratio, ode_nopt)

    for hstar in args.target_h:
        lat_h = select_h(lat_data, float(hstar))
        ode_h = select_h(ode_data, float(hstar))
        y3_out = outdir / f"lattice_y3_from_rho_H{float(hstar):.1f}".replace(".", "p")
        y3_out = Path(str(y3_out) + ".png")
        make_panel_plot(
            lat_h,
            ode_h,
            "y3",
            r"$Y_3=\xi (f_{\rm anh}^{\rm noPT})^2/t_p^{3/2}$",
            y3_out,
            args.dpi,
            rf"Lattice $Y_3$ from $\rho_{{\rm noPT}}$ at $H_*={float(hstar):g}$, $v_w={args.fixed_vw:g}$",
        )

        fanh_out = outdir / f"lattice_fanh_from_rho_H{float(hstar):.1f}".replace(".", "p")
        fanh_out = Path(str(fanh_out) + ".png")
        make_panel_plot(
            lat_h,
            ode_h,
            "fanh",
            r"$\xi f_{\rm anh}^{\rm noPT}/t_p^{3/2}$",
            fanh_out,
            args.dpi,
            rf"Single-power quantity from $\rho_{{\rm noPT}}$ at $H_*={float(hstar):g}$, $v_w={args.fixed_vw:g}$",
        )

    summary_out = outdir / "lattice_y3_from_rho_summary.txt"
    save_summary(summary_out, args, scale_fit, lat_data, ode_data)
    print(
        "fanh_noPT scale fit: K={:.6e}, rel_rmse_vs_ODE={:.3e}".format(
            scale_fit["scale"],
            scale_fit["rel_rmse"],
        )
    )
    print(f"Saved: {summary_out}")


if __name__ == "__main__":
    main()
