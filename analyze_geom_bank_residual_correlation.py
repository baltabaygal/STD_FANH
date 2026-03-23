import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parent
PRED_PATH = ROOT / "results_ftilde_gvw_transport" / "ftilde_table.csv"
PILOT_PATH = ROOT / "lattice_data" / "data" / "pilot_kappa.csv"
GEOM_DIR = ROOT / "lattice_data" / "geom_bank"
OUTDIR = ROOT / "results_geom_bank_residuals"

KAPPA_RE = re.compile(r"BM_geometry_RD_kappa_([0-9.]+)_vw([0-9.]+)_oneloop\.json")


def h2(theta):
    return np.log(np.e / (1.0 - (theta / np.pi) ** 2))


def build_geom_index():
    out = {}
    for path in GEOM_DIR.glob("BM_geometry_RD_kappa_*_vw*_oneloop.json"):
        m = KAPPA_RE.match(path.name)
        if not m:
            continue
        kappa = float(m.group(1))
        vw = float(m.group(2))
        out.setdefault(vw, []).append((kappa, path))
    for vw in out:
        out[vw].sort(key=lambda t: t[0])
    return out


def nearest_geom_path(index, vw, kappa):
    choices = index[float(vw)]
    kappa_vals = np.array([x[0] for x in choices], dtype=float)
    idx = int(np.argmin(np.abs(kappa_vals - float(kappa))))
    return choices[idx]


def nearest_pilot_rows(pilot, theta, vw, hb):
    sub = pilot[np.isclose(pilot["theta0"], float(theta), atol=1.0e-10) & np.isclose(pilot["vw"], float(vw), atol=1.0e-12)]
    arr = sub["Hb"].to_numpy(dtype=float)
    idx = int(np.argmin(np.abs(arr - float(hb))))
    return sub.iloc[idx]


def load_geom_payload(path):
    return json.loads(Path(path).read_text())


def get_geom_entry(payload, H, beta_over_H):
    h_key = f"{float(H):.2f}"
    b_key = f"{float(beta_over_H):.2f}"
    return payload[h_key][b_key]


def fit_metrics(y, yhat, p):
    n = len(y)
    rss = float(np.sum((y - yhat) ** 2))
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    mae = float(np.mean(np.abs(y - yhat)))
    aic = float(n * np.log(rss / n) + 2 * p)
    bic = float(n * np.log(rss / n) + p * np.log(n))
    return {"rmse": rmse, "mae": mae, "aic": aic, "bic": bic}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PRED_PATH)
    pilot = pd.read_csv(PILOT_PATH)
    geom_index = build_geom_index()

    rows = []
    cache = {}
    for _, row in df.iterrows():
        hb = float(row["H"]) ** 2 / float(row["beta_over_H"])
        p_row = nearest_pilot_rows(pilot, row["theta"], row["v_w"], hb)
        kappa_target = float(p_row["kappa"])
        kappa_bank, geom_path = nearest_geom_path(geom_index, row["v_w"], kappa_target)
        if geom_path not in cache:
            cache[geom_path] = load_geom_payload(geom_path)
        entry = get_geom_entry(cache[geom_path], row["H"], row["beta_over_H"])
        rows.append(
            {
                "Hb_geom": hb,
                "pilot_Hb": float(p_row["Hb"]),
                "pilot_tau_n": float(p_row["tau_n"]),
                "pilot_t_trig": float(p_row["t_trig"]),
                "kappa_target": kappa_target,
                "kappa_bank": float(kappa_bank),
                "geom_path": str(geom_path),
                "f_BM": float(entry["f_BM"]),
                "A_BM": float(entry["A_BM"]),
                "G_BM": float(entry["G_BM"]),
                "R_min": float(entry["R_min"]),
                "Rmax_global": float(entry["Rmax_global"]),
            }
        )

    geom = pd.DataFrame(rows)
    work = pd.concat([df.reset_index(drop=True), geom], axis=1)
    work["logv"] = np.log(work["v_w"] / 0.9)
    work["h2c"] = h2(work["theta"]) - float(np.mean(h2(work["theta"])))
    work["logx"] = np.log(work["x"])
    work["logb"] = np.log(work["beta_over_H"] / 12.0)
    work["logG_BM"] = np.log(np.maximum(work["G_BM"], 1.0e-18))
    work["logA_BM"] = np.log(np.maximum(work["A_BM"], 1.0e-18))
    work["logf_BM"] = np.log(np.maximum(work["f_BM"], 1.0e-18))

    work.to_csv(OUTDIR / "joined_geom_residual_table.csv", index=False)

    y = work["ftilde_frac_resid"].to_numpy()

    model_defs = {
        "vw_only": ["logv"],
        "vw_shape": ["logv", "logv*h2c", "logv*logx", "logv*logb"],
        "G_only": ["logG_BM"],
        "G_shape": ["logG_BM", "logG_BM*h2c", "logG_BM*logx", "logG_BM*logb"],
        "A_shape": ["logA_BM", "logA_BM*h2c", "logA_BM*logx", "logA_BM*logb"],
        "f_shape": ["logf_BM", "logf_BM*h2c", "logf_BM*logx", "logf_BM*logb"],
    }

    def cols_from(names):
        cols = []
        for name in names:
            if "*" in name:
                a, b = name.split("*")
                cols.append(work[a].to_numpy() * work[b].to_numpy())
            else:
                cols.append(work[name].to_numpy())
        return np.column_stack(cols)

    model_rows = []
    model_payload = {}
    for name, names in model_defs.items():
        X = cols_from(names)
        reg = LinearRegression().fit(X, y)
        yhat = reg.predict(X)
        met = fit_metrics(y, yhat, X.shape[1] + 1)
        model_rows.append({"model": name, **met})
        model_payload[name] = {
            "features": names,
            "intercept": float(reg.intercept_),
            "coef": {k: float(v) for k, v in zip(names, reg.coef_)},
            **met,
        }

    pd.DataFrame(model_rows).sort_values("bic").to_csv(OUTDIR / "model_comparison.csv", index=False)

    corrs = {}
    for key in ["logv", "logG_BM", "logA_BM", "logf_BM", "G_BM", "A_BM", "f_BM"]:
        corrs[key] = {
            "pearson_with_signed_resid": float(np.corrcoef(work[key], work["ftilde_frac_resid"])[0, 1]),
            "pearson_with_abs_resid": float(np.corrcoef(work[key], np.abs(work["ftilde_frac_resid"]))[0, 1]),
        }

    # Plots
    for key, label in [("G_BM", "G_BM"), ("A_BM", "A_BM"), ("f_BM", "f_BM")]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
        axes = axes.ravel()
        for ax, theta in zip(axes, sorted(work["theta"].unique())):
            sub = work[np.isclose(work["theta"], theta)].copy()
            sc = ax.scatter(sub[key], sub["ftilde_frac_resid"], c=sub["v_w"], s=18, cmap="viridis")
            ax.set_title(fr"$\theta={theta:.3f}$")
            ax.set_xlabel(label)
            ax.set_ylabel(r"$\delta \tilde f / \tilde f$")
            ax.grid(alpha=0.25)
        fig.colorbar(sc, ax=axes.tolist(), label=r"$v_w$")
        fig.tight_layout()
        fig.savefig(OUTDIR / f"resid_vs_{key}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    by_v = work.groupby("v_w")["G_BM"].mean().reset_index()
    ax.plot(by_v["v_w"], by_v["G_BM"], marker="o")
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel("mean G_BM")
    ax.set_title("Geometry proxy vs wall speed")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "G_BM_vs_vw.png", dpi=180)
    plt.close(fig)

    summary = {
        "input_predictions": str(PRED_PATH),
        "pilot_mapping": "Hb = H_*^2 / (beta/H_*) ; nearest (theta0, v_w, Hb) row in pilot_kappa.csv ; nearest kappa bank file",
        "correlations": corrs,
        "models": model_payload,
        "outputs": {
            "joined_table": str(OUTDIR / "joined_geom_residual_table.csv"),
            "model_comparison": str(OUTDIR / "model_comparison.csv"),
            "resid_vs_G_BM": str(OUTDIR / "resid_vs_G_BM.png"),
            "resid_vs_A_BM": str(OUTDIR / "resid_vs_A_BM.png"),
            "resid_vs_f_BM": str(OUTDIR / "resid_vs_f_BM.png"),
            "G_BM_vs_vw": str(OUTDIR / "G_BM_vs_vw.png"),
        },
    }
    with open(OUTDIR / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
