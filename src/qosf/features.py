"""Feature engineering: only information available at time t is used to predict t+h.

Feature groups (used in the ablation study):
  history  - past throughput (lags, rolling statistics, harmonic mean)
  radio    - channel KPIs (RSRP, RSRQ, SNR, CQI, RSSI, NR-specific) and their trends
  mobility - speed, Doppler shift, handover indicators, network technology
  space    - position and a space-time throughput prior learned on training traces,
             evaluated at the current position and at the position predicted h s ahead
"""
import numpy as np
import pandas as pd

from .config import CARRIER_HZ, GRID_LAT, GRID_LON, HORIZONS, SPEED_OF_LIGHT

RADIO = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI", "NRxRSRP", "NRxRSRQ"]
EARTH_M_PER_DEG_LAT = 111_320.0


def _harmonic_mean(x: pd.Series, w: int) -> pd.Series:
    inv = 1.0 / x.clip(lower=0.1)
    return w / inv.rolling(w, min_periods=1).sum() * (inv.rolling(w, min_periods=1).count() / w)


def per_trace_features(t: pd.DataFrame) -> pd.DataFrame:
    """Features for one trace (rows ordered in time, 1-second grid)."""
    t = t.copy()
    thr = t["thr"]

    # ---------------------------------------------------------------- history
    for k in range(0, 5):
        t[f"thr_lag{k}"] = thr.shift(k)
    for w in (3, 5, 10, 30):
        r = thr.rolling(w, min_periods=1)
        t[f"thr_mean{w}"] = r.mean()
        t[f"thr_std{w}"] = r.std()
    t["thr_min5"] = thr.rolling(5, min_periods=1).min()
    t["thr_max5"] = thr.rolling(5, min_periods=1).max()
    t["thr_hm5"] = _harmonic_mean(thr, 5)           # classic ABR throughput estimator
    t["thr_ewma"] = thr.ewm(alpha=0.3).mean()

    # ---------------------------------------------------------------- radio
    for c in RADIO:
        t[f"{c}_d3"] = t[c] - t[c].shift(3)          # trend over the last 3 s
    t["SNR_mean5"] = t["SNR"].rolling(5, min_periods=1).mean()
    t["RSRP_mean5"] = t["RSRP"].rolling(5, min_periods=1).mean()

    # ---------------------------------------------------------------- mobility
    fc = t["net"].map(CARRIER_HZ).fillna(CARRIER_HZ["LTE"])
    t["doppler_hz"] = t["speed_ms"] / SPEED_OF_LIGHT * fc          # max Doppler shift
    t["accel"] = t["speed_ms"].diff()
    t["is_5g"] = (t["net"] == "5G").astype(int)
    t["is_lte"] = (t["net"] == "LTE").astype(int)
    cell_change = t["CellID"].diff().fillna(0).ne(0).astype(int)
    tech_change = t["net"].ne(t["net"].shift()).astype(int).iloc[1:].reindex(t.index, fill_value=0)
    t["ho_last5"] = cell_change.rolling(5, min_periods=1).sum()
    t["ho_last30"] = cell_change.rolling(30, min_periods=1).sum()
    t["tech_change_last10"] = tech_change.rolling(10, min_periods=1).sum()
    grp = cell_change.cumsum()
    t["time_since_ho"] = t.groupby(grp).cumcount().clip(upper=120)

    # ---------------------------------------------------------------- space (raw)
    # heading from the displacement over the last 3 s, used to extrapolate position
    dlat = t["Latitude"] - t["Latitude"].shift(3)
    dlon = t["Longitude"] - t["Longitude"].shift(3)
    t["vel_lat"] = (dlat / 3).fillna(0)          # degrees per second
    t["vel_lon"] = (dlon / 3).fillna(0)
    t["hour"] = t["time"].dt.hour

    # ---------------------------------------------------------------- targets
    real = (~t["missing"].astype(bool)).astype(float)
    for h in HORIZONS:
        # mean throughput over the next h seconds (t, t+h]
        fut = thr[::-1].rolling(h, min_periods=h).mean()[::-1].shift(-1)
        t[f"y_h{h}"] = fut
        # a target is scored only if t and the whole window (t, t+h] were measured,
        # not forward-filled
        real_fut = real[::-1].rolling(h, min_periods=h).min()[::-1].shift(-1)
        t[f"valid_h{h}"] = (real_fut == 1) & (real == 1) & fut.notna()
    return t


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = [per_trace_features(g.sort_values("time")) for _, g in df.groupby("trace", sort=False)]
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ space-time prior
def grid_key(lat, lon):
    return (np.floor(lat / GRID_LAT).astype("Int64").astype(str) + "_" +
            np.floor(lon / GRID_LON).astype("Int64").astype(str))


class SpatialPrior:
    """Median throughput per ~110 m grid cell, learned on training traces only.

    Evaluated at the current position and at the position extrapolated h seconds
    ahead (dead reckoning from the recent heading): "what throughput do vehicles
    usually get where I will be in h seconds?"
    """

    def __init__(self, min_count: int = 5):
        self.min_count = min_count

    def fit(self, df: pd.DataFrame):
        d = df.dropna(subset=["Latitude", "Longitude", "thr"])
        key = grid_key(d["Latitude"], d["Longitude"])
        g = d.groupby(key)["thr"]
        stats = pd.DataFrame({"med": g.median(), "n": g.size()})
        self.table_ = stats[stats["n"] >= self.min_count]
        self.global_ = float(d["thr"].median())
        return self

    def lookup(self, lat, lon):
        key = grid_key(pd.Series(lat), pd.Series(lon))
        med = key.map(self.table_["med"]).astype(float)
        cnt = key.map(self.table_["n"]).fillna(0).astype(float)
        return med.fillna(self.global_).values, cnt.values

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["prior_now"], X["prior_cnt"] = self.lookup(X["Latitude"].values, X["Longitude"].values)
        for h in HORIZONS:
            lat = X["Latitude"] + X["vel_lat"] * h
            lon = X["Longitude"] + X["vel_lon"] * h
            X[f"prior_ahead{h}"], _ = self.lookup(lat.values, lon.values)
        return X


# ------------------------------------------------------------------ feature lists
HISTORY = ([f"thr_lag{k}" for k in range(5)] +
           [f"thr_{s}{w}" for w in (3, 5, 10, 30) for s in ("mean", "std")] +
           ["thr_min5", "thr_max5", "thr_hm5", "thr_ewma"])
RADIO_F = RADIO + [f"{c}_d3" for c in RADIO] + ["SNR_mean5", "RSRP_mean5"]
MOBILITY = ["speed_ms", "doppler_hz", "accel", "is_5g", "is_lte", "ho_last5", "ho_last30",
            "tech_change_last10", "time_since_ho", "hour"]


def space_features(h: int):
    return ["Latitude", "Longitude", "prior_now", "prior_cnt", f"prior_ahead{h}"]


def feature_groups(h: int) -> dict:
    return {"history": HISTORY, "radio": RADIO_F, "mobility": MOBILITY,
            "space": space_features(h)}


def all_features(h: int) -> list:
    return HISTORY + RADIO_F + MOBILITY + space_features(h)
