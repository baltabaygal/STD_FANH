#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import collapse_and_fit_fanh_tosc as cf


OUTDIR = ROOT / "results_tosc_lattice_vw0p9_H1p0H1p5H2p0_beta0" / "collapse_and_fit_fanh"
FIXED_VW = 0.9
TARGET_H = [1.0, 1.5, 2.0]


def parse_args():
    p = argparse.ArgumentParser(description="Run the lattice-only collapse fit with beta fixed to zero.")
    p.add_argument("--fixed-vw", type=float, default=FIXED_VW)
    p.add_argument("--h-values", type=float, nargs="+", default=TARGET_H)
    p.add_argument("--outdir", type=str, default=str(OUTDIR))
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def to_native(obj):
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_native(v) for v in obj.tolist()]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def save_json(path: Path, payload):
    path.write_text(json.dumps(to_native(payload), indent=2, sort_keys=True))


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    raw_lattice_path = ROOT / "lattice_data" / "data" / "energy_ratio_by_theta_data_v9.txt"
    rho_path = ROOT / "lattice_data" / "data" / "rho_noPT_data.txt"
    target_h = [float(h) for h in args.h_values]
    fixed_vw = float(args.fixed_vw)
    print(f"[load] loading lattice data for v_w={fixed_vw:.1f}")
    lattice_df = cf.load_lattice_data(raw_lattice_path, fixed_vw, target_h)
    f0_table = cf.load_f0_table(rho_path, target_h)
    f0_table.to_csv(outdir / "F0_table.csv", index=False)
    lattice_df = cf.merge_f0(lattice_df, f0_table)
    lattice_df = lattice_df[np.isfinite(lattice_df["F0"]) & (lattice_df["F0"] > 0.0)].copy()

    beta = 0.0
    print("[fit] fixing beta=0 and fitting the same collapse kernel")
    collapsed = cf.compute_x(lattice_df, beta)
    collapsed["fanh_data"] = collapsed["xi"].to_numpy(dtype=np.float64) * collapsed["F0"].to_numpy(dtype=np.float64) / np.power(
        np.maximum(collapsed["x"].to_numpy(dtype=np.float64) / cf.T_OSC, 1.0e-18), 1.5
    )

    finf_tail_df = cf.fit_tail(collapsed, outdir, dpi=args.dpi)
    fit_result = cf.fit_global(collapsed, finf_tail_df)
    global_payload = cf.save_global_fit(fit_result, beta, outdir)

    cf.plot_collapse_overlay(collapsed, fit_result, outdir, dpi=args.dpi)
    cf.plot_fanh_theta(collapsed, fit_result, outdir, dpi=args.dpi)

    # Save best-fit table in the same format as the usual run.
    save_json(outdir / "best_beta.json", {"beta": beta, "collapse_score": cf.collapse_score(lattice_df, beta, 120), "fixed": True})

    summary = {
        "status": "ok",
        "fixed_vw": fixed_vw,
        "available_H": target_h,
        "beta": beta,
        "collapse_score": cf.collapse_score(lattice_df, beta, 120),
        "n_lattice_points": int(len(collapsed)),
        "global_fit": {
            "t_c": float(global_payload["t_c"]),
            "r": float(global_payload["r"]),
            "rel_rmse": float(global_payload["rel_rmse"]),
            "AIC": float(global_payload["AIC"]),
            "BIC": float(global_payload["BIC"]),
        },
        "outputs": {
            "collapse_overlay": str(outdir / "collapse_overlay.png"),
            "global_fit": str(outdir / "global_fit.json"),
            "best_beta": str(outdir / "best_beta.json"),
            "fanh_theta_2p356": str(outdir / "fanh_theta_2.356.png"),
            "fanh_theta_2p880": str(outdir / "fanh_theta_2.880.png"),
        },
    }
    save_json(outdir / "final_summary.json", summary)
    print(json.dumps(to_native(summary), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        outdir = OUTDIR.resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        payload = {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
        save_json(outdir / "_error.json", payload)
        print(json.dumps(payload, sort_keys=True))
        raise
