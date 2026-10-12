"""Online inference: turns the last seconds of measurements into forecasts.

Shared by the FastAPI service and the tests, so that online features are computed by
exactly the same code as offline features.
"""
import joblib
import numpy as np
import pandas as pd

from .features import per_trace_features

WINDOW_S = 30          # seconds of history needed by the rolling features


class Forecaster:
    def __init__(self, path):
        a = joblib.load(path)
        self.point, self.cqr, self.cqr_h = a["point"], a["cqr"], a["cqr_h"]
        self.prior, self.features = a["prior"], a["features"]

    def predict(self, records: list) -> dict:
        w = pd.DataFrame(records)
        w["time"] = pd.to_datetime(w["time"])
        w = w.sort_values("time").tail(WINDOW_S).reset_index(drop=True)
        if "missing" not in w:
            w["missing"] = False
        if "speed_ms" not in w:
            w["speed_ms"] = w["Speed"] / 3.6
        f = self.prior.transform(per_trace_features(w)).tail(1)
        out = {"time": str(w["time"].iloc[-1]), "throughput_now": float(w["thr"].iloc[-1])}
        for h, model in self.point.items():
            out[f"forecast_h{h}"] = float(max(0.0, model.predict(f[self.features[h]])[0]))
        lo, hi = self.cqr.predict_interval(f[self.features[self.cqr_h]])
        out[f"lower_h{self.cqr_h}"] = float(lo[0])
        out[f"upper_h{self.cqr_h}"] = float(hi[0])
        out["prior_ahead"] = float(f[f"prior_ahead{self.cqr_h}"].iloc[0])
        return out
