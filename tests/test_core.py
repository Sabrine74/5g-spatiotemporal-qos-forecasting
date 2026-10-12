"""Unit tests:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qosf.abr import LADDER, simulate
from qosf.features import HISTORY, MOBILITY, RADIO_F, per_trace_features


def toy_trace(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=n, freq="1s"),
        "thr": rng.gamma(2, 20, n), "RSRP": rng.normal(-90, 5, n), "RSRQ": rng.normal(-11, 2, n),
        "SNR": rng.normal(8, 4, n), "CQI": rng.integers(1, 16, n).astype(float),
        "RSSI": rng.normal(-85, 5, n), "NRxRSRP": rng.normal(-90, 5, n), "NRxRSRQ": rng.normal(-11, 2, n),
        "Latitude": 51.9 + np.arange(n) * 1e-4, "Longitude": -8.47 + np.arange(n) * 1e-4,
        "Speed": np.full(n, 36.0), "speed_ms": np.full(n, 10.0),
        "CellID": np.repeat([1.0, 2.0], n // 2), "net": "5G", "missing": False,
    })


def test_features_use_only_the_past():
    """Changing the future must not change the features at time t (no leakage)."""
    a = toy_trace()
    b = a.copy()
    b.loc[40:, ["thr", "SNR", "RSRP", "CellID"]] = [999.0, 30.0, -50.0, 7.0]
    cols = HISTORY + RADIO_F + MOBILITY
    fa, fb = per_trace_features(a), per_trace_features(b)
    pd.testing.assert_frame_equal(fa.loc[:39, cols], fb.loc[:39, cols])


def test_target_is_future_mean():
    t = toy_trace()
    f = per_trace_features(t)
    assert np.isclose(f.loc[10, "y_h3"], t.loc[11:13, "thr"].mean())


def test_doppler_value():
    f = per_trace_features(toy_trace())
    assert np.isclose(f["doppler_hz"].iloc[0], 10.0 / 299_792_458.0 * 3.5e9)   # ~117 Hz


def test_abr_perfect_estimate_never_stalls():
    thr = np.full(300, 100.0)
    res = simulate(thr, thr)
    assert res["rebuffer_s"] == 0 and res["avg_bitrate"] == LADDER.max()


def test_abr_overestimate_stalls():
    thr = np.full(300, 3.0)
    res = simulate(thr, np.full(300, 100.0))
    assert res["rebuffer_s"] > 0
