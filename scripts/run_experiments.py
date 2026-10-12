"""Run every experiment of the paper and save results to results/.

    python scripts/run_experiments.py          (about 15 minutes on a laptop)

Protocol: 5-fold cross-validation where whole traces are held out
(StratifiedGroupKFold by trace, stratified on mobility). Every number is computed on
traces never seen during training, and the space-time prior is re-learned inside each
fold from the training traces only.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from qosf.abr import run_policies
from qosf.conformal import CQR
from qosf.config import HORIZONS, RANDOM_STATE, RESULTS_DIR
from qosf.data import load_processed, summary
from qosf.features import SpatialPrior, all_features, build_features, feature_groups
from qosf.models import lgbm, make_models

ALPHA = 0.2          # 80 % prediction intervals
ABL_H = 3            # horizon used for ablation, uncertainty, SHAP and ABR


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    raw = load_processed()
    summary(raw).round(3).to_csv(RESULTS_DIR / "dataset_summary.csv")
    F = build_features(raw)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(F, F["mobility"], F["trace"]))

    oof = F[["trace", "time", "mobility", "thr", "Speed", "doppler_hz", "time_since_ho",
             "ho_last5", "Latitude", "Longitude", "net"] +
            [f"y_h{h}" for h in HORIZONS] + [f"valid_h{h}" for h in HORIZONS]].copy()
    oof["fold"] = -1
    per_fold, ablation, cqr_rows = [], [], []

    for k, (tr_idx, te_idx) in enumerate(folds):
        print(f"--- fold {k}: {F.iloc[te_idx]['trace'].nunique()} test traces")
        oof.loc[oof.index[te_idx], "fold"] = k
        sp = SpatialPrior().fit(F.iloc[tr_idx])
        tr_all, te_all = sp.transform(F.iloc[tr_idx]), sp.transform(F.iloc[te_idx])

        for h in HORIZONS:
            cols = all_features(h)
            tr = tr_all[tr_all[f"y_h{h}"].notna()]
            valid = te_all[f"valid_h{h}"].values
            y_te = te_all[f"y_h{h}"].values
            for name, model in make_models().items():
                model.fit(tr[cols], tr[f"y_h{h}"])
                p = np.clip(model.predict(te_all[cols]), 0, None)
                oof.loc[oof.index[te_idx], f"pred_{name}_h{h}"] = p
                per_fold.append({"fold": k, "h": h, "model": name,
                                 "mae": mae(y_te[valid], p[valid]),
                                 "rmse": rmse(y_te[valid], p[valid]), "n": int(valid.sum())})

        # ---------------- ablation (cumulative feature groups), h = ABL_H
        h = ABL_H
        tr = tr_all[tr_all[f"y_h{h}"].notna()]
        valid = te_all[f"valid_h{h}"].values
        y_te = te_all[f"y_h{h}"].values[valid]
        near_ho = (te_all["ho_last5"].values[valid] > 0)
        groups = feature_groups(h)
        cum = []
        for gname in ["history", "radio", "mobility", "space"]:
            cum = cum + groups[gname]
            m = lgbm().fit(tr[cum], tr[f"y_h{h}"])
            p = np.clip(m.predict(te_all[cum][valid]), 0, None)
            ablation.append({"fold": k, "setting": "+ " + gname if gname != "history" else "history",
                             "mae": mae(y_te, p), "rmse": rmse(y_te, p), "n": len(y_te),
                             "mae_ho": mae(y_te[near_ho], p[near_ho]), "n_ho": int(near_ho.sum())})
        for gname in ["radio", "mobility", "space"]:          # leave-one-group-out
            cols = [c for c in all_features(h) if c not in groups[gname]]
            m = lgbm().fit(tr[cols], tr[f"y_h{h}"])
            p = np.clip(m.predict(te_all[cols][valid]), 0, None)
            ablation.append({"fold": k, "setting": f"all - {gname}",
                             "mae": mae(y_te, p), "rmse": rmse(y_te, p), "n": len(y_te),
                             "mae_ho": mae(y_te[near_ho], p[near_ho]), "n_ho": int(near_ho.sum())})

        # ---------------- conformalized quantile regression, h = ABL_H
        cols = all_features(h)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
        fit_i, cal_i = next(gss.split(tr, groups=tr["trace"]))
        fit, cal = tr.iloc[fit_i], tr.iloc[cal_i]
        cal = cal[cal[f"valid_h{h}"]]
        cqr = CQR(ALPHA).fit(fit[cols], fit[f"y_h{h}"], cal[cols], cal[f"y_h{h}"].values)
        lo_raw, hi_raw = cqr.raw_interval(te_all[cols])
        lo, hi = cqr.predict_interval(te_all[cols])
        oof.loc[oof.index[te_idx], "q_lo"] = lo
        oof.loc[oof.index[te_idx], "q_hi"] = hi
        yv = te_all[f"y_h{h}"].values
        for kind, a, b in [("quantile regression", np.clip(lo_raw, 0, None), hi_raw), ("CQR", lo, hi)]:
            cqr_rows.append({"fold": k, "method": kind,
                             "coverage": float(np.mean((yv[valid] >= a[valid]) & (yv[valid] <= b[valid]))),
                             "width": float(np.mean(b[valid] - a[valid])), "n": int(valid.sum())})
        print(f"    done in {time.time() - t_start:.0f} s")

    # ------------------------------------------------------------------ aggregate
    pf = pd.DataFrame(per_fold)
    pf.to_csv(RESULTS_DIR / "metrics_per_fold.csv", index=False)
    rows = []
    for (h, model), g in pf.groupby(["h", "model"]):
        sel = oof[oof[f"valid_h{h}"].astype(bool)]
        y, p = sel[f"y_h{h}"].values, sel[f"pred_{model}_h{h}"].values
        rows.append({"h": h, "model": model, "mae": mae(y, p), "rmse": rmse(y, p),
                     "mae_fold_std": g["mae"].std(), "n": len(y)})
    metrics = pd.DataFrame(rows)
    base = metrics[metrics.model == "Persistence"].set_index("h")
    metrics["mae_skill_vs_persistence"] = 1 - metrics["mae"] / metrics["h"].map(base["mae"])
    metrics["rmse_skill_vs_persistence"] = 1 - metrics["rmse"] / metrics["h"].map(base["rmse"])
    metrics.round(4).to_csv(RESULTS_DIR / "metrics.csv", index=False)

    wins = (pf.pivot_table(index=["h", "fold"], columns="model", values="mae")
            .pipe(lambda d: (d["LightGBM"] < d["Persistence"]).groupby("h").sum()))

    # per mobility
    h = ABL_H
    sel = oof[oof[f"valid_h{h}"].astype(bool)]
    mob = []
    for m in ["Driving", "Static"]:
        for model in ["Persistence", "LightGBM"]:
            s = sel[sel.mobility == m]
            mob.append({"mobility": m, "model": model,
                        "mae": mae(s[f"y_h{h}"], s[f"pred_{model}_h{h}"]),
                        "rmse": rmse(s[f"y_h{h}"], s[f"pred_{model}_h{h}"]), "n": len(s)})
    pd.DataFrame(mob).round(4).to_csv(RESULTS_DIR / "metrics_by_mobility.csv", index=False)

    ab = pd.DataFrame(ablation)
    ab_sum = (ab.groupby("setting", sort=False)
              .apply(lambda g: pd.Series({"mae": np.average(g["mae"], weights=g["n"]),
                                          "rmse": np.sqrt(np.average(g["rmse"] ** 2, weights=g["n"])),
                                          "mae_near_handover": np.average(g["mae_ho"], weights=g["n_ho"]),
                                          "n_near_handover": g["n_ho"].sum()})))
    # persistence near handover, same rows
    sel = oof[oof[f"valid_h{ABL_H}"].astype(bool) & (oof["ho_last5"] > 0)]
    ab_sum.loc["persistence", ["mae_near_handover", "n_near_handover"]] = [
        mae(sel[f"y_h{ABL_H}"], sel[f"pred_Persistence_h{ABL_H}"]), len(sel)]
    ab_sum.round(4).to_csv(RESULTS_DIR / "ablation.csv")

    cq = pd.DataFrame(cqr_rows)
    cq_sum = cq.groupby("method").apply(lambda g: pd.Series({
        "coverage": np.average(g["coverage"], weights=g["n"]),
        "width": np.average(g["width"], weights=g["n"]),
        "coverage_min_fold": g["coverage"].min(), "coverage_max_fold": g["coverage"].max()}))
    cq_sum.round(4).to_csv(RESULTS_DIR / "uncertainty.csv")

    # ------------------------------------------------------------------ ABR use case
    oof["thr_hm5"] = F["thr_hm5"].values
    policies = {
        "Harmonic mean (5 s)": ("thr_hm5", 0.9),
        "LightGBM": (f"pred_LightGBM_h{ABL_H}", 0.9),
        "LightGBM lower bound (CQR)": ("q_lo", 1.0),
    }
    abr = []
    for trace, g in oof.groupby("trace"):
        r = run_policies(g.sort_values("time"), policies)
        r["trace"], r["mobility"] = trace, g["mobility"].iloc[0]
        abr.append(r)
    abr = pd.concat(abr, ignore_index=True)
    abr.round(4).to_csv(RESULTS_DIR / "abr_per_trace.csv", index=False)
    abr_sum = abr.groupby("policy", sort=False)[["avg_bitrate", "rebuffer_ratio",
                                                 "switches_per_min", "qoe_per_chunk"]].mean()
    abr_sum.round(4).to_csv(RESULTS_DIR / "abr_summary.csv")
    qoe = abr.pivot(index="trace", columns="policy", values="qoe_per_chunk")
    abr_wins = {p: int((qoe[p] > qoe["Harmonic mean (5 s)"]).sum()) for p in qoe.columns
                if p != "Harmonic mean (5 s)"}

    oof.to_csv(RESULTS_DIR / "oof_predictions.csv.gz", index=False)
    json.dump({"lightgbm_beats_persistence_folds": {int(k): int(v) for k, v in wins.items()},
               "abr_traces_with_higher_qoe_than_harmonic_mean": abr_wins, "abr_traces": int(len(qoe)),
               "runtime_s": round(time.time() - t_start)},
              open(RESULTS_DIR / "summary.json", "w"), indent=2)

    pd.set_option("display.width", 160)
    print(metrics.round(3).to_string(index=False))
    print(ab_sum.round(3))
    print(cq_sum.round(3))
    print(abr_sum.round(3))
    print("LightGBM beats persistence in folds:", wins.to_dict())


if __name__ == "__main__":
    main()
