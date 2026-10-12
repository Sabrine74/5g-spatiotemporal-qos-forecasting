"""API test (needs models/qos_forecaster.joblib: run scripts/train_final.py first)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "realtime" / "api"))
MODEL = ROOT / "models" / "qos_forecaster.joblib"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="model not trained")


def test_predict_endpoint(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(MODEL))
    from fastapi.testclient import TestClient
    import app as api
    client = TestClient(api.app)
    trace = pd.read_csv(ROOT / "realtime" / "demo_trace.csv").dropna(subset=["thr", "Latitude"]).head(30)
    cols = ["thr", "RSRP", "RSRQ", "SNR", "CQI", "RSSI", "NRxRSRP", "NRxRSRQ",
            "Latitude", "Longitude", "Speed", "CellID"]
    recs = [{**{c: (None if pd.isna(r[c]) else float(r[c])) for c in cols},
             "time": str(r["time"]), "net": r["net"]} for _, r in trace.iterrows()]
    r = client.post("/predict", json={"measurements": recs})
    assert r.status_code == 200
    out = r.json()
    assert 0 <= out["lower_h3"] <= out["forecast_h3"] <= out["upper_h3"] or out["lower_h3"] <= out["upper_h3"]
    assert all(out[f"forecast_h{h}"] >= 0 for h in (1, 3, 5))
