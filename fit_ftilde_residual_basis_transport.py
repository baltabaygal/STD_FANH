import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


TOSC = 1.5
ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results_ftilde_gvw_transport" / "ftilde_table.csv"
OUTDIR = ROOT / "results_ftilde_residual_basis"


def h2(theta):
    return np.log(np.e / (1.0 - (theta / np.pi) ** 2))


def fit_no_intercept(df, feature_defs):
    train = df["v_w"].to_numpy() < 0.9
    X = np.column_stack([feature_defs[name](df) for name in feature_defs])
    y = df["ftilde_frac_resid"].to_numpy()
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X[train], y[train])
    yhat = reg.predict(X)
    return {
        "feature_names": list(feature_defs),
        "coef": reg.coef_.copy(),
        "yhat": yhat,
        "train_mask": train,
    }


def rel_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean(((y_pred - y_true) / y_true) ** 2)))


def write_xi_plots(df, value_col, suffix, title):
    for H in sorted(df["H"].unique()):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False, sharey=False)
        axes = axes.ravel()
        for ax, theta in zip(axes, sorted(df["theta"].unique())):
            sub = df[(df["H"] == H) & (df["theta"] == theta)].copy()
            for vw, g in sorted(sub.groupby("v_w")):
                g = g.sort_values("beta_over_H")
                ax.scatter(g["beta_over_H"], g["xi"], s=18, label=fr"$v_w={vw:.1f}$")
                ax.plot(g["beta_over_H"], g[value_col], lw=1.8)
            ax.set_xscale("log")
            ax.set_title(fr"$\theta={theta:.3f}$")
            ax.set_xlabel(r"$\beta/H_*$")
            ax.set_ylabel(r"$\xi$")
            ax.grid(alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles[:4], labels[:4], loc="upper center", ncol=4, frameon=False)
        fig.suptitle(f"{title}, $H_*={H:.1f}$", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(OUTDIR / f"xi_vs_betaH_{suffix}_H{str(H).replace('.', 'p')}.png", dpi=180)
        plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)

    h2c_mean = float(h2(df["theta"].to_numpy()).mean())

    feature_defs_compact = {
        "logv_h2c": lambda d: np.log(d["v_w"].to_numpy() / 0.9) * (h2(d["theta"].to_numpy()) - h2c_mean),
        "logv_logx": lambda d: np.log(d["v_w"].to_numpy() / 0.9) * np.log(d["x"].to_numpy()),
    }
    feature_defs_best = {
        "logv": lambda d: np.log(d["v_w"].to_numpy() / 0.9),
        "logv_h2c": lambda d: np.log(d["v_w"].to_numpy() / 0.9) * (h2(d["theta"].to_numpy()) - h2c_mean),
        "logv_logx": lambda d: np.log(d["v_w"].to_numpy() / 0.9) * np.log(d["x"].to_numpy()),
        "logv_logb": lambda d: np.log(d["v_w"].to_numpy() / 0.9) * np.log(d["beta_over_H"].to_numpy() / 12.0),
    }

    models = {
        "compact2": fit_no_intercept(df, feature_defs_compact),
        "best4": fit_no_intercept(df, feature_defs_best),
    }

    rows = []
    summary = {
        "input": str(INPUT),
        "h2_definition": "log(e / (1 - (theta/pi)^2))",
        "h2_center": h2c_mean,
        "models": {},
    }

    base_pred = df["xi_fit_vw0p9_model"].to_numpy()
    g_pred = df["xi_fit_gvw"].to_numpy()
    xi_true = df["xi"].to_numpy()

    baseline_metrics = {
        "rel_rmse": rel_rmse(xi_true, base_pred),
        "mean_abs_frac_resid": float(np.mean(np.abs((base_pred - xi_true) / xi_true))),
        "median_abs_frac_resid": float(np.median(np.abs((base_pred - xi_true) / xi_true))),
    }
    gvw_metrics = {
        "rel_rmse": rel_rmse(xi_true, g_pred),
        "mean_abs_frac_resid": float(np.mean(np.abs((g_pred - xi_true) / xi_true))),
        "median_abs_frac_resid": float(np.median(np.abs((g_pred - xi_true) / xi_true))),
    }
    summary["baseline"] = {"frozen_vw0p9": baseline_metrics, "gvw_only": gvw_metrics}

    for name, model in models.items():
        delta_hat = model["yhat"]
        xi_fit = g_pred * (1.0 + delta_hat)
        frac_resid = (xi_fit - xi_true) / xi_true

        model_df = df.copy()
        model_df["delta_hat"] = delta_hat
        model_df["xi_fit_basis"] = xi_fit
        model_df["frac_resid_basis"] = frac_resid
        model_df.to_csv(OUTDIR / f"predictions_{name}.csv", index=False)

        metrics = {
            "feature_names": model["feature_names"],
            "coef": {k: float(v) for k, v in zip(model["feature_names"], model["coef"])},
            "rel_rmse": rel_rmse(xi_true, xi_fit),
            "mean_abs_frac_resid": float(np.mean(np.abs(frac_resid))),
            "median_abs_frac_resid": float(np.median(np.abs(frac_resid))),
            "train_rel_rmse_delta": float(np.sqrt(np.mean((delta_hat[model["train_mask"]] - df.loc[model["train_mask"], "ftilde_frac_resid"].to_numpy()) ** 2))),
        }
        by_theta = (
            model_df.groupby("theta")["frac_resid_basis"]
            .apply(lambda s: float(np.mean(np.abs(s))))
            .reset_index(name="mean_abs_frac_resid")
            .sort_values("mean_abs_frac_resid", ascending=False)
        )
        by_theta.to_csv(OUTDIR / f"by_theta_{name}.csv", index=False)
        metrics["worst_theta"] = by_theta.head(3).to_dict(orient="records")
        summary["models"][name] = metrics

        rows.append(
            {
                "model": name,
                "rel_rmse": metrics["rel_rmse"],
                "mean_abs_frac_resid": metrics["mean_abs_frac_resid"],
                "median_abs_frac_resid": metrics["median_abs_frac_resid"],
            }
        )

        # Residual-by-theta plot
        fig, ax = plt.subplots(figsize=(7, 4.5))
        tmp = by_theta.sort_values("theta")
        ax.plot(tmp["theta"], tmp["mean_abs_frac_resid"], marker="o")
        ax.set_xlabel(r"$\theta_0$")
        ax.set_ylabel("mean |fractional residual|")
        ax.set_title(f"Residual by theta: {name}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUTDIR / f"residual_vs_theta_{name}.png", dpi=180)
        plt.close(fig)

        # Delta vs log x plot
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(np.log(model_df["x"]), model_df["ftilde_frac_resid"], s=10, alpha=0.35, label="data")
        order = np.argsort(model_df["x"].to_numpy())
        ax.plot(np.log(model_df["x"].to_numpy()[order]), delta_hat[order], lw=1.8, alpha=0.9, label="fit")
        ax.set_xlabel(r"$\log x$")
        ax.set_ylabel(r"$\delta$")
        ax.set_title(f"Residual correction: {name}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUTDIR / f"delta_vs_logx_{name}.png", dpi=180)
        plt.close(fig)

        write_xi_plots(model_df, "xi_fit_basis", name, f"f_tilde residual-basis correction ({name})")

    pd.DataFrame(rows).sort_values("rel_rmse").to_csv(OUTDIR / "model_comparison.csv", index=False)
    with open(OUTDIR / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
