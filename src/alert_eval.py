"""Alert-level evaluation: what a doctor/patient would actually experience.

An alert fires when risk >= threshold, then is silenced for 2 h (refractory period).
Reports alerts/day, alert precision (a crossing of 180 mg/dL follows within 2 h),
crossing-event recall (an alert fired in the 2 h before the crossing), and how many
false alerts were 'near misses' (glucose still reached >= 160 mg/dL).
Run after train.py:  python src/alert_eval.py
"""
import argparse
import numpy as np, pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="results/oof_predictions.csv.gz")
    ap.add_argument("--ts", default="data/processed/timeseries.csv.gz")
    ap.add_argument("--out", default="results/alert_tradeoff.csv"); a = ap.parse_args()
    o = pd.read_csv(a.oof, parse_dates=["time"])
    ts = pd.read_csv(a.ts, parse_dates=["time"], usecols=["file_id", "time", "cgm"])
    fm, ev = [], {}
    for fid, d in ts.groupby("file_id"):
        d = d.sort_values("time").reset_index(drop=True); g = d.cgm.values
        mx = pd.Series(g[::-1]).rolling(8, min_periods=1).max().values[::-1]
        d["fmax"] = np.r_[mx[1:], np.nan]
        ev[fid] = d.time.values[np.where((g[1:] > 180) & (g[:-1] <= 180))[0] + 1]
        fm.append(d[["file_id", "time", "fmax"]])
    o = o.merge(pd.concat(fm), on=["file_id", "time"], how="left")
    days = len(o) * 15 / 1440; nev = sum(len(v) for v in ev.values()); rows = []
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        A, caught = [], 0
        for fid, d in o.groupby("file_id"):
            d = d.sort_values("time"); last = None; al = []
            for t, r, y, fx in zip(d.time, d.risk, d.y, d.fmax):
                if r >= thr and (last is None or (t - last).total_seconds() / 60 >= 120):
                    al.append((t, y, fx)); last = t
            A += al
            caught += sum(any(0 < (e - np.datetime64(t)) / np.timedelta64(1, "m") <= 120 for t, _, _ in al) for e in ev[fid])
        fp = [x[2] for x in A if x[1] == 0]
        rows.append(dict(risk_threshold=thr, alerts_per_day=len(A) / days, alert_precision=sum(x[1] for x in A) / len(A),
                         event_recall=caught / nev, false_alerts_near_miss_ge160=np.mean([f >= 160 for f in fp])))
    r = pd.DataFrame(rows).round(3); r.to_csv(a.out, index=False); print(r.to_string(index=False))


if __name__ == "__main__":
    main()
