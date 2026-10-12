"""Build every figure of the paper from results/ (run after run_experiments.py and
train_final.py).        python scripts/make_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qosf.config import FIG_DIR, MODELS_DIR, RESULTS_DIR
from qosf.data import load_processed
from qosf.features import SpatialPrior, build_features

COLORS = {"Persistence": "#2a78d6", "Harmonic mean (5 s)": "#eda100", "Ridge": "#1baf7a",
          "MLP": "#e87ba4", "LightGBM": "#eb6834", "LightGBM lower bound (CQR)": "#6250d6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titleweight": "bold", "axes.titlesize": 9.5, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "legend.fontsize": 8,
})


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png")
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)


def fig_map(raw):
    d = raw[(raw.mobility == "Driving")].dropna(subset=["Latitude", "Longitude", "thr"])
    s = raw[(raw.mobility == "Static")].dropna(subset=["Latitude", "Longitude"])
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    sc = ax.scatter(d.Longitude, d.Latitude, c=np.log10(d.thr.clip(lower=0.1)), s=1.2,
                    cmap="Blues", vmin=-1, vmax=2.7, rasterized=True)
    sp = s.groupby("trace")[["Longitude", "Latitude"]].median()
    ax.scatter(sp.Longitude, sp.Latitude, marker="^", s=40, color="#eb6834",
               edgecolor="white", linewidth=0.8, label="static measurement sites", zorder=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_ticks([-1, 0, 1, 2]); cb.set_ticklabels(["0.1", "1", "10", "100"])
    cb.set_label("Downlink throughput (Mbit/s, log scale)")
    ax.set_aspect(1 / np.cos(np.deg2rad(51.9)))
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    ax.set_title("Measured throughput along the 16 drives (Cork, Ireland)")
    ax.legend(loc="lower left")
    save(fig, "fig_map")


def fig_distribution(raw):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for mob, col in [("Driving", "#2a78d6"), ("Static", "#eb6834")]:
        x = np.sort(raw.loc[raw.mobility == mob, "thr"].dropna().clip(lower=0.01))
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=col, lw=2,
                label=f"{mob} (median {np.median(x):.0f} Mbit/s)")
    ax.set_xscale("log"); ax.set_xlim(0.1, 600)
    ax.set_xlabel("Downlink throughput (Mbit/s, log scale)"); ax.set_ylabel("Share of seconds ≤ x")
    ax.set_title("Throughput distribution (CDF)")
    ax.legend(loc="upper left")
    save(fig, "fig_distribution")


def fig_horizon(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for name in ["Persistence", "Harmonic mean (5 s)", "Ridge", "MLP", "LightGBM"]:
        m = metrics[metrics.model == name].sort_values("h")
        for ax, col in zip(axes, ["mae", "rmse"]):
            ax.plot(m.h, m[col], marker="o", ms=5, lw=2, color=COLORS[name], label=name)
    for ax, t in zip(axes, ["MAE", "RMSE"]):
        ax.set_xticks([1, 3, 5]); ax.set_xlabel("Forecast horizon h (s)")
        ax.set_ylabel(f"{t} (Mbit/s)"); ax.set_title(f"{t} vs horizon (lower is better)")
        ax.set_ylim(bottom=0)
    axes[1].legend(loc="upper center", bbox_to_anchor=(-0.15, -0.22), ncol=5)
    save(fig, "fig_horizon")


def fig_example(oof):
    g = oof[(oof.mobility == "Driving")]
    # the drive with the most handovers in a 300-s window
    best, best_n = None, -1
    for tr, d in g.groupby("trace"):
        d = d.sort_values("time").reset_index(drop=True)
        ho = d["ho_last5"].gt(0).astype(int).rolling(300).sum()
        if ho.max() > best_n and len(d) > 400:
            best_n, best = ho.max(), (tr, int(ho.idxmax()))
    tr, end = best
    d = g[g.trace == tr].sort_values("time").reset_index(drop=True).iloc[end - 300:end]
    t = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.fill_between(t, d.q_lo, d.q_hi, color="#eb6834", alpha=0.15, lw=0, label="80 % interval (CQR)")
    ax.plot(t, d.y_h3, color=INK, lw=1.4, label="Actual mean throughput over next 3 s")
    ax.plot(t, d.pred_Persistence_h3, color=COLORS["Persistence"], lw=1, alpha=0.9, label="Persistence")
    ax.plot(t, d.pred_LightGBM_h3, color=COLORS["LightGBM"], lw=1.6, label="LightGBM")
    ho_t = t[(d.time_since_ho == 0).values & (t > 0)]
    for x in ho_t:
        ax.axvline(x, color=INK2, lw=0.6, ls=":")
    ax.set_xlabel(f"Time (s) — drive {tr}, dotted lines = cell changes (handovers)")
    ax.set_ylabel("Throughput (Mbit/s)"); ax.set_ylim(bottom=0)
    ax.set_title("Example: 3-second-ahead forecast on a drive never seen in training")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4)
    save(fig, "fig_example")


def fig_ablation(ab):
    order = ["history", "+ radio", "+ mobility", "+ space"]
    a = ab.set_index("setting")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for ax, col, title in [(axes[0], "mae", "All test seconds"),
                           (axes[1], "mae_near_handover", "≤ 5 s after a handover")]:
        vals = a.loc[order, col].values
        ax.plot(range(4), vals, color="#eb6834", lw=1.5, marker="o", ms=7, label="LightGBM")
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, color=INK2)
        if col == "mae_near_handover":
            ref = a.loc["persistence", col]
            ax.axhline(ref, color=COLORS["Persistence"], lw=1.5, ls="--", label=f"Persistence ({ref:.2f})")
        ax.legend(loc="upper right")
        ax.set_xticks(range(4), order); ax.set_xlim(-0.4, 3.4)
        ax.set_ylabel("MAE (Mbit/s), h = 3 s")
        ax.set_ylim(vals.min() - 0.6, vals.max() + 0.6)
        ax.set_title(title)
    save(fig, "fig_ablation")


def binned(oof, by, bins, labels):
    d = oof[oof.valid_h3.astype(bool) & (oof.mobility == "Driving")].copy()
    d["bin"] = pd.cut(d[by], bins=bins, labels=labels, right=False)
    rows = []
    for b, g in d.groupby("bin", observed=True):
        rows.append({"bin": b, "n": len(g),
                     **{m: np.mean(np.abs(g.y_h3 - g[f"pred_{m}_h3"])) for m in ["Persistence", "LightGBM"]},
                     "mean_thr": g.y_h3.mean()})
    return pd.DataFrame(rows)


def fig_speed_handover(oof):
    sp = binned(oof, "Speed", [0, 15, 30, 45, 200], ["0–15", "15–30", "30–45", "≥45"])
    sp["doppler"] = ["0–49", "49–97", "97–146", "≥146"]
    ho = binned(oof, "time_since_ho", [0, 3, 6, 11, 31, 1000], ["0–2", "3–5", "6–10", "11–30", ">30"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, d, xl in [(axes[0], sp, "Speed (km/h) / max. Doppler at 3.5 GHz"),
                      (axes[1], ho, "Time since last handover (s)")]:
        x = np.arange(len(d))
        w = 0.38
        for i, m in enumerate(["Persistence", "LightGBM"]):
            ax.bar(x + (i - 0.5) * w, d[m] / d["mean_thr"], w * 0.95, color=COLORS[m], label=m)
        labels = [f"{a}\n{b} Hz" for a, b in zip(d.bin, d.doppler)] if "doppler" in d else list(d.bin)
        ax.set_xticks(x, labels); ax.set_xlabel(xl)
        ax.set_ylabel("Normalised MAE (MAE / mean)"); ax.set_ylim(0, 0.7)
        ax.grid(axis="x", visible=False)
    axes[0].set_title("Error vs speed, driving traces, h = 3 s")
    axes[1].set_title("Error vs time since handover, h = 3 s")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig_speed_handover")
    for d in (sp, ho):
        d["nmae_persistence"] = d["Persistence"] / d["mean_thr"]
        d["nmae_lightgbm"] = d["LightGBM"] / d["mean_thr"]
    sp.round(4).to_csv(RESULTS_DIR / "error_by_speed.csv", index=False)
    ho.round(4).to_csv(RESULTS_DIR / "error_by_handover.csv", index=False)


def fig_uncertainty(oof):
    pf = pd.read_csv(RESULTS_DIR / "uncertainty.csv")
    d = oof[oof.valid_h3.astype(bool)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    cov = d.groupby("fold").apply(lambda g: np.mean((g.y_h3 >= g.q_lo) & (g.y_h3 <= g.q_hi)))
    ax.plot(cov.index, cov.values, "o", color="#6250d6", ms=8, label="CQR coverage per fold")
    ax.axhline(0.8, color=INK, lw=1, ls="--", label="target 80 %")
    raw = pf.set_index("method").loc["quantile regression", "coverage"]
    ax.axhline(raw, color="#eda100", lw=1.5, label=f"uncalibrated quantile reg. ({raw:.1%})")
    ax.set_ylim(0.6, 0.95); ax.set_xlabel("Cross-validation fold"); ax.set_ylabel("Coverage")
    ax.set_title("Interval coverage on unseen traces"); ax.set_xticks(cov.index)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    bins = [0, 5, 20, 50, 1000]
    lab = ["<5", "5–20", "20–50", "≥50"]
    d = d.assign(b=pd.cut(d.pred_LightGBM_h3, bins, labels=lab, right=False))
    w = d.groupby("b", observed=True).apply(lambda g: (g.q_hi - g.q_lo).median())
    ax.bar(range(len(w)), w.values, color="#6250d6", width=0.6)
    ax.set_xticks(range(len(w)), lab); ax.set_xlabel("Point forecast (Mbit/s)")
    ax.set_ylabel("Median interval width (Mbit/s)"); ax.set_title("Intervals widen when throughput is high")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_uncertainty")


def fig_abr(abr):
    pols = ["Harmonic mean (5 s)", "LightGBM", "LightGBM lower bound (CQR)"]
    s = abr.groupby("policy")[["avg_bitrate", "rebuffer_ratio", "switches_per_min", "qoe_per_chunk"]].mean().loc[pols]
    titles = ["Mean bitrate (Mbit/s) ↑", "Stall time (% of playback) ↓",
              "Quality switches per minute ↓", "QoE per chunk (Yin et al.) ↑"]
    fig, axes = plt.subplots(2, 2, figsize=(3.7, 3.6))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)
    for ax, col, t in zip(axes.ravel(), s.columns, titles):
        v = s[col] * (100 if col == "rebuffer_ratio" else 1)
        ax.bar(range(3), v, color=[COLORS[p] for p in pols], width=0.65)
        for i, x in enumerate(v):
            ax.text(i, x, f"{x:.1f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks([]); ax.set_title(t, fontsize=7.5); ax.grid(axis="x", visible=False)
        ax.set_ylim(0, v.max() * 1.22); ax.tick_params(labelsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[p]) for p in pols]
    fig.legend(handles, ["Harmonic mean of last 5 s (classic)",
                         "LightGBM forecast", "LightGBM 80 % lower bound (CQR)"],
               loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.12), fontsize=7)
    save(fig, "fig_abr")


def fig_shap():
    import shap
    art = joblib.load(MODELS_DIR / "qos_forecaster.joblib")
    raw = load_processed()
    F = art["prior"].transform(build_features(raw[raw.trace != art["demo_trace"]]))
    cols = art["features"][3]
    X = F[cols].sample(3000, random_state=0)
    sv = shap.TreeExplainer(art["point"][3]).shap_values(X)
    imp = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)
    imp.head(20).round(4).to_csv(RESULTS_DIR / "shap_importance.csv", header=["mean_abs_shap"])
    groups = {"history": "History", "radio": "Radio", "mobility": "Mobility", "space": "Space"}
    from qosf.features import feature_groups
    fg = feature_groups(3)
    gmap = {c: groups[g] for g, cs in fg.items() for c in cs}
    gcol = {"History": "#2a78d6", "Radio": "#1baf7a", "Mobility": "#e87ba4", "Space": "#eb6834"}
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.barh(range(len(top)), top.values, color=[gcol[gmap[c]] for c in top.index], height=0.7)
    ax.set_yticks(range(len(top)), top.index, fontsize=7.5)
    ax.set_xlabel("Mean |SHAP value| (Mbit/s)"); ax.grid(axis="y", visible=False)
    ax.set_title("Top-15 features, LightGBM h = 3 s")
    share = {g: imp[[c for c in imp.index if gmap[c] == g]].sum() / imp.sum() for g in gcol}
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in gcol.values()],
              [f"{g} ({share[g]:.0%} of total)" for g in gcol], loc="lower right", title="Feature group")
    save(fig, "fig_shap")
    share = pd.Series({g: imp[[c for c in imp.index if gmap[c] == g]].sum() for g in gcol}) / imp.sum()
    share.round(4).to_csv(RESULTS_DIR / "shap_group_share.csv", header=["share"])


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_processed()
    oof = pd.read_csv(RESULTS_DIR / "oof_predictions.csv.gz", parse_dates=["time"])
    fig_map(raw)
    fig_distribution(raw)
    fig_horizon(pd.read_csv(RESULTS_DIR / "metrics.csv"))
    fig_example(oof)
    fig_ablation(pd.read_csv(RESULTS_DIR / "ablation.csv"))
    fig_speed_handover(oof)
    fig_uncertainty(oof)
    fig_abr(pd.read_csv(RESULTS_DIR / "abr_per_trace.csv"))
    fig_shap()
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
