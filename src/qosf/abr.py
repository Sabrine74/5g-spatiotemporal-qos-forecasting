"""Trace-driven adaptive-bitrate (ABR) video streaming simulator.

A player downloads 2-second video chunks over the measured throughput of a trace and
picks the quality of each chunk from a throughput estimate. Better estimates mean
higher quality without stalls. QoE follows Yin et al. (SIGCOMM 2015):
    QoE = sum(bitrate) - mu * rebuffering_time - sum(|bitrate change|)
"""
import numpy as np
import pandas as pd

LADDER = np.array([1.0, 2.5, 5.0, 8.0, 16.0, 25.0, 40.0])   # Mbit/s (HD to 4K)
CHUNK_S = 2.0
BUFFER_MAX_S = 20.0
MU = LADDER.max()          # one second of stall costs as much as one top-quality chunk-second


def _download_time(thr, t0, size_mbit):
    """Time to download size_mbit starting at t0 over a 1-second throughput series."""
    t, remaining = t0, size_mbit
    n = len(thr)
    while remaining > 1e-9:
        i = int(t)
        if i >= n:
            return None
        rate = max(thr[i], 1e-3)
        dt = min(i + 1 - t, remaining / rate)
        remaining -= rate * dt
        t += dt
    return t - t0


def simulate(thr, estimate, safety=1.0):
    """thr: true throughput per second (Mbit/s); estimate: estimate available at second i."""
    thr = np.nan_to_num(np.asarray(thr, float), nan=0.0)
    estimate = np.nan_to_num(np.asarray(estimate, float), nan=0.0)
    t, buffer, rebuf, started = 0.0, 0.0, 0.0, False
    rates = []
    while True:
        est = estimate[min(int(t), len(estimate) - 1)] * safety
        ok = LADDER[LADDER <= est]
        r = ok.max() if len(ok) else LADDER[0]
        d = _download_time(thr, t, r * CHUNK_S)
        if d is None:
            break
        if started:
            stall = max(0.0, d - buffer)
            rebuf += stall
            buffer = max(0.0, buffer - d)
        t += d
        buffer += CHUNK_S
        started = True
        rates.append(r)
        if buffer > BUFFER_MAX_S:                 # wait for space in the buffer
            t += buffer - BUFFER_MAX_S
            buffer = BUFFER_MAX_S
    rates = np.array(rates)
    if len(rates) < 2:
        return None
    play_s = len(rates) * CHUNK_S
    switches = np.abs(np.diff(rates))
    return {
        "chunks": len(rates),
        "avg_bitrate": rates.mean(),
        "rebuffer_s": rebuf,
        "rebuffer_ratio": rebuf / play_s,
        "switches_per_min": (switches > 0).sum() / (play_s / 60),
        "qoe_per_chunk": (rates.sum() - MU * rebuf - switches.sum()) / len(rates),
    }


def run_policies(trace_df: pd.DataFrame, policies: dict) -> pd.DataFrame:
    rows = []
    for name, (col, safety) in policies.items():
        res = simulate(trace_df["thr"].values, trace_df[col].values, safety)
        if res:
            rows.append({"policy": name, **res})
    return pd.DataFrame(rows)
