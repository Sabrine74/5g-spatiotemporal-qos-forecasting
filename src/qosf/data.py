"""Download, parse and clean the 5G production dataset.

Output: one row per second per trace, for the continuous file-download traces.
Video-streaming traces are excluded from forecasting because their throughput is
limited by the player (on/off downloading), not by the network.
"""
import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from .config import DATA_URL, MAX_GAP_FILL, PROCESSED_FILE, RAW_DIR

# Physically valid ranges (3GPP TS 36.133 / 38.133 reporting ranges). Values outside,
# such as Android's "unavailable" code 2147483647 or RSRP = -200 dBm, become missing.
VALID_RANGE = {"RSRP": (-140, -44), "RSRQ": (-43, 20), "SNR": (-30, 40), "CQI": (0, 15),
               "RSSI": (-130, -20), "NRxRSRP": (-156, -31), "NRxRSRQ": (-43, 20)}

NUMERIC = ["Longitude", "Latitude", "Speed", "RSRP", "RSRQ", "SNR", "CQI", "RSSI",
           "DL_bitrate", "UL_bitrate", "NRxRSRP", "NRxRSRQ"]


def download_raw() -> None:
    if RAW_DIR.exists() and any(RAW_DIR.rglob("*.csv")):
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading the 5G production dataset (2 MB) ...")
    with urllib.request.urlopen(DATA_URL) as r:
        zf = zipfile.ZipFile(io.BytesIO(r.read()))
    for name in zf.namelist():
        if name.endswith(".csv") and "__MACOSX" not in name:
            target = RAW_DIR / name.split("5G-production-dataset/")[-1]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))


def _network_family(mode: str) -> str:
    if mode == "5G":
        return "5G"
    if mode == "LTE":
        return "LTE"
    return "3G"


def load_trace(path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["-", ""])
    df["time"] = pd.to_datetime(df["Timestamp"], format="%Y.%m.%d_%H.%M.%S")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c, (lo, hi) in VALID_RANGE.items():
        df[c] = df[c].where(df[c].between(lo, hi))
    df["net"] = df["NetworkMode"].map(_network_family)

    # Several rows can share one timestamp: keep one row per second (mean of numeric
    # KPIs, last categorical value), then put the trace on a regular 1-second grid.
    agg = {c: "mean" for c in NUMERIC}
    agg.update({"CellID": "last", "net": "last", "State": "last"})
    df = df.groupby("time").agg(agg)
    grid = pd.date_range(df.index.min(), df.index.max(), freq="1s")
    df = df.reindex(grid)
    df["missing"] = df["DL_bitrate"].isna()
    df = df.ffill(limit=MAX_GAP_FILL)
    df.index.name = "time"
    return df.reset_index()


def build_dataset(app: str = "Download") -> pd.DataFrame:
    download_raw()
    frames = []
    for i, path in enumerate(sorted((RAW_DIR / app).rglob("*.csv"))):
        t = load_trace(path)
        t["trace"] = f"{app[:2]}{i:02d}"
        t["mobility"] = "Driving" if "Driving" in path.parts else "Static"
        t["app"] = app
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    df["thr"] = df["DL_bitrate"] / 1000.0          # kbit/s -> Mbit/s
    df["speed_ms"] = df["Speed"] / 3.6             # km/h -> m/s
    df["CellID"] = df["CellID"].astype("float")
    return df


def load_processed() -> pd.DataFrame:
    if PROCESSED_FILE.exists():
        return pd.read_csv(PROCESSED_FILE, parse_dates=["time"])
    df = build_dataset()
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("mobility")
    return pd.DataFrame({
        "traces": g["trace"].nunique(),
        "seconds": g.size(),
        "thr_median_mbps": g["thr"].median(),
        "thr_p90_mbps": g["thr"].quantile(0.9),
        "share_5G": g["net"].apply(lambda s: (s == "5G").mean()),
        "speed_mean_kmh": g["Speed"].mean(),
        "cell_changes": g.apply(lambda d: int((d.groupby("trace")["CellID"].diff().fillna(0) != 0).sum())),
    })
