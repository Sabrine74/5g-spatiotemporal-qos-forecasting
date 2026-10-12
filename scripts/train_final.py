"""Train the deployable model used by the real-time API.

One driving trace is held out as the demo trace replayed in the dashboard, so the live
demo shows predictions on data the model has never seen.

    python scripts/train_final.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from qosf.conformal import CQR
from qosf.config import HORIZONS, MODELS_DIR, RANDOM_STATE, ROOT
from qosf.data import load_processed
from qosf.features import SpatialPrior, all_features, build_features
from qosf.models import lgbm

DEMO_TRACE_FILE = ROOT / "realtime" / "demo_trace.csv"
RAW_COLUMNS = ["time", "thr", "RSRP", "RSRQ", "SNR", "CQI", "RSSI", "NRxRSRP", "NRxRSRQ",
               "Latitude", "Longitude", "Speed", "speed_ms", "CellID", "net", "missing"]


def main():
    raw = load_processed()
    driving = raw[raw.mobility == "Driving"].groupby("trace").size()
    demo = driving.sort_values().index[len(driving) // 2]          # a median-length drive
    raw[raw.trace == demo][RAW_COLUMNS].to_csv(DEMO_TRACE_FILE, index=False)

    train = raw[raw.trace != demo]
    F = build_features(train)
    prior = SpatialPrior().fit(F)
    X = prior.transform(F)

    point = {}
    for h in HORIZONS:
        d = X[X[f"y_h{h}"].notna()]
        point[h] = lgbm().fit(d[all_features(h)], d[f"y_h{h}"])

    h = 3
    d = X[X[f"y_h{h}"].notna()]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    fit_i, cal_i = next(gss.split(d, groups=d["trace"]))
    cal = d.iloc[cal_i]
    cal = cal[cal[f"valid_h{h}"]]
    cqr = CQR(0.2).fit(d.iloc[fit_i][all_features(h)], d.iloc[fit_i][f"y_h{h}"],
                       cal[all_features(h)], cal[f"y_h{h}"].values)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"point": point, "cqr": cqr, "cqr_h": h, "prior": prior,
                 "features": {k: all_features(k) for k in HORIZONS},
                 "demo_trace": demo}, MODELS_DIR / "qos_forecaster.joblib")
    print(f"Saved model (demo trace {demo}, {len(pd.read_csv(DEMO_TRACE_FILE))} s) "
          f"to {MODELS_DIR / 'qos_forecaster.joblib'}")


if __name__ == "__main__":
    main()
