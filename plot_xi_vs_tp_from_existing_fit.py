#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import collapse_and_fit_fanh_tosc as collapse


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "results_tosc_lattice_vw0p9_H0p5H1p0H1p5H2p0" / "collapse_and_fit_fanh" / "global_fit.json"
DEFAULT_OUTDIR = ROOT / "results_xi_vs_tp_from_existing_fit_vw0p9_H0p5H1p0H1p5H2p0"
T_OSC = collapse.T_OSC


def parse_args():
    p = argparse.ArgumentParser(description="Plot raw xi vs tp using an existing collapse fit.")
    p.add_argument("--model-json", type=str, default=str(DEFAULT_MODEL))
    p.add_argument("--fixed-vw", type=float, default=0.9)
    p.add_argument("--h-values", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    p.add_argument("--rho", type=str, default="")
    p.add_argument("--lattice-raw", type=str, default="")
    p.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def to_native(obj):
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_native(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def save_json(path: Path, payload):
    path.write_text(json.dumps(to_native(payload), indent=2, sort_keys=True))


def rel_rmse(y, yfit):
    y = np.asarray(y, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(yfit) & (y > 0.0)
    if np.count_nonzero(mask) == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square((yfit[mask] - y[mask]) / np.maximum(y[mask], 1.0e-12)))))


def choose_theta_subset(theta_values):
    targets = np.array([0.262, 0.785, 1.309, 1.833, 2.356, 2.880], dtype=np.float64)
    theta_values = np.asarray(sorted(theta_values), dtype=np.float64)
    out = []
    for target in targets:
        idx = int(np.argmin(np.abs(theta_values - target)))
        val = float(theta_values[idx])
        if val not in out:
            out.append(val)
    return np.asarray(out, dtype=np.float64)


def load_data(args, outdir: Path):
    rho_path = collapse.resolve_first_existing(collapse.RHO_CANDIDATES, args.rho)
    raw_lattice_path = collapse.resolve_first_existing(collapse.LATTICE_RAW_CANDIDATES, args.lattice_raw)
    target_h = sorted(set(float(h) for h in args.h_values))
    f0_table = collapse.load_f0_table(rho_path, target_h)
    lattice_df = collapse.load_lattice_data(raw_lattice_path, args.fixed_vw, target_h)
    lattice_df = collapse.merge_f0(lattice_df, f0_table)
    lattice_df = lattice_df[np.isfinite(lattice_df["F0"]) & (lattice_df["F0"] > 0.0)].copy()
    lattice_df.to_csv(outdir / "lattice_df.csv", index=False)
    f0_table.to_csv(outdir / "F0_table.csv", index=False)
    return lattice_df, f0_table


def load_fit(path: Path):
    payload = json.loads(path.read_text())
    theta_values = np.array(sorted(float(k) for k in payload["F_inf"].keys()), dtype=np.float64)
    finf = np.array([float(payload["F_inf"][f"{theta:.10f}"]["value"]) for theta in theta_values], dtype=np.float64)
    params = np.concatenate([[float(payload["t_c"]), float(payload["r"])], finf])
    return {"payload": payload, "theta_values": theta_values, "params": params}


def add_predictions(df: pd.DataFrame, fit: dict):
    out = df.copy()
    beta = float(fit["payload"]["beta"])
    out["x"] = out["tp"].to_numpy(dtype=np.float64) * np.power(out["H"].to_numpy(dtype=np.float64), beta)
    xi_fit, _ = collapse.xi_model_from_params(out, fit["theta_values"], fit["params"])
    out["xi_fit"] = xi_fit
    return out


def xi_model_grid(theta: float, H: float, f0: float, tp_grid: np.ndarray, fit: dict):
    beta = float(fit["payload"]["beta"])
    x = tp_grid * np.power(float(H), beta)
    work = pd.DataFrame(
        {
            "theta": np.full_like(tp_grid, float(theta), dtype=np.float64),
            "H": np.full_like(tp_grid, float(H), dtype=np.float64),
            "tp": tp_grid,
            "x": x,
            "F0": np.full_like(tp_grid, float(f0), dtype=np.float64),
        }
    )
    xi, _ = collapse.xi_model_from_params(work, fit["theta_values"], fit["params"])
    return xi


def plot_by_h(df: pd.DataFrame, fit: dict, outdir: Path, dpi: int):
    theta_values = choose_theta_subset(df["theta"].unique())
    h_values = np.sort(df["H"].unique())
    rows = []
    for h_value in h_values:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
        axes = axes.ravel()
        sub_h = df[np.isclose(df["H"], float(h_value), atol=1.0e-12, rtol=0.0)].copy()
        for ax, theta in zip(axes, theta_values):
            cur = sub_h[np.isclose(sub_h["theta"], float(theta), atol=5.0e-4, rtol=0.0)].sort_values("tp").copy()
            if cur.empty:
                ax.axis("off")
                continue
            f0 = float(cur["F0"].iloc[0])
            tp_grid = np.geomspace(float(cur["tp"].min()), float(cur["tp"].max()), 300)
            xi_grid = xi_model_grid(float(theta), float(h_value), f0, tp_grid, fit)
            ax.scatter(cur["tp"], cur["xi"], s=24, color="tab:blue", alpha=0.9, label="data")
            ax.plot(tp_grid, xi_grid, color="black", lw=1.8, label="fit")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(rf"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$t_p$")
            ax.set_ylabel(r"$\xi$")
            rows.append(
                {
                    "H": float(h_value),
                    "theta": float(theta),
                    "rel_rmse": rel_rmse(cur["xi"], cur["xi_fit"]),
                }
            )
        for ax in axes[len(theta_values):]:
            ax.axis("off")
        axes[0].legend(frameon=False, fontsize=8)
        beta = float(fit["payload"]["beta"])
        fig.suptitle(rf"Raw $\xi(t_p)$ vs fit, $v_w={args.fixed_vw:.1f}$, $H_*={h_value:.1f}$, $\beta={beta:.4f}$", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        tag = str(float(h_value)).replace(".", "p")
        fig.savefig(outdir / f"xi_vs_tp_H{tag}.png", dpi=dpi)
        plt.close(fig)
    return pd.DataFrame(rows)


def plot_separate(df: pd.DataFrame, fit: dict, outdir: Path, dpi: int):
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for h_value in np.sort(df["H"].unique()):
        sub_h = df[np.isclose(df["H"], float(h_value), atol=1.0e-12, rtol=0.0)].copy()
        for theta in np.sort(sub_h["theta"].unique()):
            cur = sub_h[np.isclose(sub_h["theta"], float(theta), atol=5.0e-4, rtol=0.0)].sort_values("tp").copy()
            if cur.empty:
                continue
            f0 = float(cur["F0"].iloc[0])
            tp_grid = np.geomspace(float(cur["tp"].min()), float(cur["tp"].max()), 300)
            xi_grid = xi_model_grid(float(theta), float(h_value), f0, tp_grid, fit)
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            ax.scatter(cur["tp"], cur["xi"], s=24, color="tab:blue", alpha=0.9, label="data")
            ax.plot(tp_grid, xi_grid, color="black", lw=1.8, label="fit")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.set_xlabel(r"$t_p$")
            ax.set_ylabel(r"$\xi$")
            ax.set_title(rf"$v_w={args.fixed_vw:.1f},\ H_*={h_value:.1f},\ \theta={theta:.3f}$")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            h_tag = str(float(h_value)).replace(".", "p")
            theta_tag = f"{float(theta):.10f}".replace(".", "p")
            path = outdir / f"xi_vs_tp_H{h_tag}_theta_{theta_tag}.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            rows.append({"H": float(h_value), "theta": float(theta), "path": str(path.resolve())})
    return pd.DataFrame(rows)


def main():
    global args
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    lattice_df, _ = load_data(args, outdir)
    fit = load_fit(Path(args.model_json).resolve())
    pred = add_predictions(lattice_df, fit)
    pred.to_csv(outdir / "predictions.csv", index=False)

    by_h = plot_by_h(pred, fit, outdir, args.dpi)
    by_h.to_csv(outdir / "fit_rows.csv", index=False)
    separate = plot_separate(pred, fit, outdir / "xi_vs_tp_separate", args.dpi)
    separate.to_csv(outdir / "xi_vs_tp_index.csv", index=False)

    summary = {
        "status": "ok",
        "model_json": str(Path(args.model_json).resolve()),
        "fixed_vw": float(args.fixed_vw),
        "beta": float(fit["payload"]["beta"]),
        "t_c": float(fit["payload"]["t_c"]),
        "r": float(fit["payload"]["r"]),
        "global_rel_rmse_on_points": rel_rmse(pred["xi"], pred["xi_fit"]),
        "by_H": [
            {
                "H": float(h),
                "rel_rmse": rel_rmse(sub["xi"], sub["xi_fit"]),
            }
            for h, sub in pred.groupby("H", sort=True)
        ],
        "outputs": {
            "predictions": str((outdir / "predictions.csv").resolve()),
            "fit_rows": str((outdir / "fit_rows.csv").resolve()),
            "index": str((outdir / "xi_vs_tp_index.csv").resolve()),
            "by_H": [str((outdir / f"xi_vs_tp_H{str(float(h)).replace('.', 'p')}.png").resolve()) for h in sorted(pred["H"].unique())],
            "separate_dir": str((outdir / "xi_vs_tp_separate").resolve()),
        },
    }
    save_json(outdir / "final_summary.json", summary)
    print(json.dumps(to_native(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
