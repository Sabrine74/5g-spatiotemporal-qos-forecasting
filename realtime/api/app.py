"""QoS forecasting API.

POST /predict with the last seconds of KPI measurements (up to 30), returns the mean
throughput forecast for the next 1, 3 and 5 s and an 80 % interval for the next 3 s.
"""
import os
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from qosf.service import Forecaster

MODEL_PATH = os.environ.get("MODEL_PATH", "models/qos_forecaster.joblib")
app = FastAPI(title="5G QoS forecasting API", version="1.0")
forecaster = Forecaster(MODEL_PATH)


class Measurement(BaseModel):
    time: str
    thr: float                       # downlink throughput, Mbit/s
    RSRP: Optional[float] = None
    RSRQ: Optional[float] = None
    SNR: Optional[float] = None
    CQI: Optional[float] = None
    RSSI: Optional[float] = None
    NRxRSRP: Optional[float] = None
    NRxRSRQ: Optional[float] = None
    Latitude: float
    Longitude: float
    Speed: float                     # km/h
    CellID: float
    net: str                         # "5G", "LTE" or "3G"


class Window(BaseModel):
    measurements: List[Measurement]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(window: Window):
    return forecaster.predict([m.model_dump() for m in window.measurements])
