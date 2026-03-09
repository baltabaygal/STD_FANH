import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pysr import PySRRegressor


def parse_args():
    parser = argparse.ArgumentParser(description="Run PySR for f_anh(theta0, tp).")
    parser.add_argument(
        "--data",
        type=str,
        default="../analysis/data/dm_tp_fitready_H1p000.txt",
        help="Path to fit-ready table.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="../analysis/results",
        help="Output directory for equations.",
    )
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--populations", type=int, default=10)
    parser.add_argument("--model-selection", type=str, default="best")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    arr = np.loadtxt(data_path, comments="#")
    # cols: H_star t_star theta0 t_p x_tp_over_tosc Ea3_PT Ea3_noPT f_anh_PT f_anh_noPT xi_DM nsteps_PT nsteps_noPT
    theta0 = arr[:, 2]
    tp = arr[:, 3]
    fanh = arr[:, 7]

    m = np.isfinite(theta0) & np.isfinite(tp) & np.isfinite(fanh) & (fanh > 0)
    theta0 = theta0[m]
    tp = tp[m]
    fanh = fanh[m]

    # Feature engineering keeps physically useful structure visible.
    u = theta0 / np.pi
    one_minus_u2 = np.clip(1.0 - u**2, 1e-8, None)
    log_tp = np.log(tp)
    inv_one_minus_u2 = 1.0 / one_minus_u2

    X = np.column_stack([u, one_minus_u2, log_tp, inv_one_minus_u2])
    y = np.log(fanh)
    feature_names = ["u", "one_minus_u2", "log_tp", "inv_one_minus_u2"]

    model = PySRRegressor(
        niterations=args.iterations,
        populations=args.populations,
        model_selection=args.model_selection,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "sqrt"],
        constraints={"log": 8, "exp": 8, "sqrt": 8},
        maxsize=28,
        parsimony=1e-4,
        loss="loss(x, y) = (x - y)^2",
        progress=True,
    )

    model.fit(X, y, variable_names=feature_names)

    eq_df = model.equations_
    out_csv = outdir / "pysr_fanh_equations.csv"
    eq_df.to_csv(out_csv, index=False)

    best_latex = str(model.latex())
    out_txt = outdir / "pysr_fanh_best_equation.txt"
    with open(out_txt, "w") as f:
        f.write("# Target: log(f_anh)\n")
        f.write(f"# Data: {data_path}\n")
        f.write("# Features: u, one_minus_u2, log_tp, inv_one_minus_u2\n")
        f.write(f"Best (latex): {best_latex}\n")

    print("Saved equations:", out_csv)
    print("Saved best eq:", out_txt)
    print("Best sympy:", model.get_best()["sympy_format"])


if __name__ == "__main__":
    main()
