"""Replays a recorded drive as if it were live.

Every (simulated) second it:
  1. writes the measured KPIs to InfluxDB (measurement "kpi"),
  2. sends the last 30 s to the forecasting API,
  3. writes the forecast to InfluxDB (measurement "forecast"), time-stamped at the time
     it refers to (t + h), so that forecast and measurement line up in Grafana,
  4. raises a "qos_alert" flag when the lower bound of the 3-s forecast drops below the
     threshold needed for the service (default 10 Mbit/s, i.e. 4K video).

    python realtime/replayer/replay.py --dry-run          (no InfluxDB, prints instead)
"""
import argparse
import math
import os
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd

FIELDS = ["thr", "RSRP", "RSRQ", "SNR", "CQI", "RSSI", "NRxRSRP", "NRxRSRQ",
          "Latitude", "Longitude", "Speed", "CellID"]


def clean(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=os.environ.get("TRACE", "realtime/demo_trace.csv"))
    ap.add_argument("--api", default=os.environ.get("API_URL", "http://localhost:8000"))
    ap.add_argument("--speed", type=float, default=float(os.environ.get("REPLAY_SPEED", 1)))
    ap.add_argument("--threshold", type=float, default=float(os.environ.get("QOS_THRESHOLD", 10)))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--loop", action="store_true", default=os.environ.get("LOOP", "1") == "1")
    args = ap.parse_args()

    trace = pd.read_csv(args.trace)
    writer = None
    if not args.dry_run:
        from influxdb_client import InfluxDBClient, Point
        from influxdb_client.client.write_api import SYNCHRONOUS
        client = InfluxDBClient(url=os.environ["INFLUX_URL"], token=os.environ["INFLUX_TOKEN"],
                                org=os.environ["INFLUX_ORG"])
        writer = client.write_api(write_options=SYNCHRONOUS)
        bucket = os.environ["INFLUX_BUCKET"]

    http = httpx.Client(base_url=args.api, timeout=10)
    for _ in range(60):                                    # wait for the API to start
        try:
            if http.get("/health").status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(2)

    step = 0
    while True:
        window = deque(maxlen=30)
        for _, row in trace.iterrows():
            now = datetime.now(timezone.utc)
            rec = {c: clean(row[c]) for c in FIELDS}
            if rec["thr"] is None or rec["Latitude"] is None:
                time.sleep(1 / args.speed)
                continue
            rec.update(time=now.isoformat(), net=str(row["net"]))
            window.append(rec)
            fc = http.post("/predict", json={"measurements": list(window)}).json()
            alert = int(fc["lower_h3"] < args.threshold)

            if writer is None:
                print(f"{now:%H:%M:%S} thr={rec['thr']:7.1f}  forecast3s={fc['forecast_h3']:7.1f} "
                      f"[{fc['lower_h3']:6.1f}, {fc['upper_h3']:6.1f}]  alert={alert}")
            else:
                p = Point("kpi").time(now).tag("net", rec["net"])
                for c in FIELDS:
                    if rec[c] is not None:
                        p = p.field(c, float(rec[c]))
                pts = [p]
                for h in (1, 3, 5):
                    pts.append(Point("forecast").time(now + timedelta(seconds=h)).tag("h", str(h))
                               .field("thr", fc[f"forecast_h{h}"]))
                pts.append(Point("forecast_interval").time(now + timedelta(seconds=3))
                           .field("lower", fc["lower_h3"]).field("upper", fc["upper_h3"]))
                pts.append(Point("qos_alert").time(now).field("alert", alert)
                           .field("threshold", args.threshold))
                writer.write(bucket=bucket, record=pts)

            step += 1
            if args.max_steps and step >= args.max_steps:
                return
            time.sleep(1 / args.speed)
        if not args.loop:
            return


if __name__ == "__main__":
    main()
