"""SIMULATED heart rate and step count (the Shanghai dataset has no wearable data).

Design rule: the simulator never looks at glucose. HR and steps depend only on time of day,
patient traits (age, BMI), logged meals and random activity bursts. Any link to glucose is
therefore indirect (through meals), so the fused model cannot "learn the generator".
Document this in the README as a limitation.
"""
import numpy as np
import pandas as pd


def simulate_wearables(ts, static_row, seed=0):
    rng = np.random.default_rng(seed)
    n = len(ts)
    hour = ts["time"].dt.hour.values + ts["time"].dt.minute.values / 60
    age = float(static_row.get("age", 55) or 55)
    bmi = float(static_row.get("bmi", 25) or 25)
    rest = 78 - 0.15 * (age - 55) + 0.3 * (bmi - 25) + rng.normal(0, 3)
    # circadian: low at night, higher in day
    circ = 7 * np.sin((hour - 9) / 24 * 2 * np.pi)
    sleeping = (hour < 6.5) | (hour > 22.5)
    # activity bursts (walks) during waking hours, random start; at most a few per day
    activity = np.zeros(n)
    p = np.where(sleeping, 0.0, 0.035)
    i = 0
    while i < n:
        if rng.random() < p[i]:
            L = rng.integers(1, 5); activity[i:i + L] = rng.uniform(0.4, 1.0); i += L
        else:
            i += 1
    # small post-meal bump in heart rate (thermic effect) from logged meals
    meal = (ts["meal_g"].values > 0).astype(float)
    bump = np.convolve(meal, np.exp(-np.arange(0, 8) / 3.0))[:n] * 5
    hr = rest + circ + 35 * activity + bump - np.where(sleeping, 10, 0) + rng.normal(0, 3, n)
    steps = activity * rng.uniform(250, 900, n) + (~sleeping) * rng.poisson(8, n)
    out = ts.copy()
    out["heart_rate"] = np.clip(hr, 45, 170).round(1)
    out["steps"] = steps.round(0)
    out["wearable_is_simulated"] = 1
    return out
