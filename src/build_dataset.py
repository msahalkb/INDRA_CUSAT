"""Build the GlucoTwin modelling dataset from the Shanghai T2DM files.

Inputs : data/raw/Shanghai_T2DM/*.xls(x)  (CGM every 15 min, meals, insulin)
         data/raw/Shanghai_T2DM_Summary.xlsx (static clinical record)
Outputs: data/processed/timeseries.csv.gz  (aligned 15-min series, for sequence models)
         data/processed/windows.csv.gz     (one row per prediction window, for GBM / fusion)

Task: at time t with CGM(t) <= 180 mg/dL, will CGM exceed 180 mg/dL within the next
2 hours (8 steps)?  Heart rate and steps are SIMULATED (see wearable_sim.py).
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd
from wearable_sim import simulate_wearables

STEP_MIN = 15
HORIZON = 8          # 2 h ahead
LOOKBACK = 8         # 2 h history
THRESH = 180.0
MAX_INTERP = 2       # fill CGM gaps up to 30 min

CARB_WORDS = ["rice", "noodle", "bread", "bun", "porridge", "congee", "dumpling", "potato",
              "corn", "pasta", "cake", "biscuit", "cookie", "cracker", "oat", "millet", "pancake",
              "wonton", "sweet", "yam", "taro", "pumpkin", "apple", "banana", "orange", "pear",
              "grape", "watermelon", "peach", "fruit", "milk", "yogurt", "sugar", "juice",
              "dough", "steamed", "fried rice", "baozi", "mantou", "lotus"]


def parse_meal(text):
    """Return (total_g, carb_food_g, n_items) from text like 'Rice 150 g\\nVegetable 100 g'."""
    if not isinstance(text, str):
        return (np.nan, np.nan, np.nan)
    tot = carb = n = 0.0
    for line in text.split("\n"):
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:g|ml)\b", line, flags=re.I)
        if not m:
            continue
        g = float(m.group(1)); n += 1; tot += g
        if any(w in line.lower() for w in CARB_WORDS):
            carb += g
    return (tot, carb, n)


def parse_insulin(text):
    if not isinstance(text, str):
        return np.nan
    v = re.findall(r"(\d+(?:\.\d+)?)\s*IU", text, flags=re.I)
    return float(np.sum([float(x) for x in v])) if v else np.nan


def has_any(s, words):
    s = str(s).lower()
    return int(any(w in s for w in words))


def static_features(path):
    s = pd.read_excel(path)
    s.columns = [c.strip() for c in s.columns]
    o = pd.DataFrame({"file_id": s["Patient Number"].astype(str)})
    o["patient_id"] = o["file_id"].str.split("_").str[0]
    o["sex_male"] = (s["Gender (Female=1, Male=2)"] == 2).astype(int)
    num = lambda c: pd.to_numeric(s[c], errors="coerce")
    o["age"] = num("Age (years)")
    o["bmi"] = num("BMI (kg/m2)")
    o["smoking_pack_years"] = num("Smoking History (pack year)")
    o["drinker"] = (s["Alcohol Drinking History (drinker/non-drinker)"].astype(str).str.lower() == "drinker").astype(int)
    o["diabetes_years"] = num("Duration of diabetes (years)")
    o["hba1c_pct"] = 0.0915 * num("HbA1c (mmol/mol)") + 2.15
    for k, c in {"fpg": "Fasting Plasma Glucose (mg/dl)", "pp2h_glucose": "2-hour Postprandial Plasma Glucose (mg/dl)",
                 "cpep_fast": "Fasting C-peptide (nmol/L)", "cpep_2h": "2-hour Postprandial C-peptide (nmol/L)",
                 "ins_fast": "Fasting Insulin (pmol/L)", "ins_2h": "2-hour Postprandial insulin (pmol/L)",
                 "glycated_albumin": "Glycated Albumin (%)", "chol": "Total Cholesterol (mmol/L)",
                 "tg": "Triglyceride (mmol/L)", "hdl": "High-Density Lipoprotein Cholesterol (mmol/L)",
                 "ldl": "Low-Density Lipoprotein Cholesterol (mmol/L)", "creatinine": "Creatinine (umol/L)",
                 "egfr": "Estimated Glomerular Filtration Rate  (ml/min/1.73m2)", "uric_acid": "Uric Acid (mmol/L)",
                 "bun": "Blood Urea Nitrogen (mmol/L)"}.items():
        o[k] = pd.to_numeric(s[c], errors="coerce")
    o["hx_hypoglycemia"] = (s["Hypoglycemia (yes/no)"].astype(str).str.lower() == "yes").astype(int)
    ag = s["Hypoglycemic Agents"].astype(str)
    o["rx_insulin"] = ag.apply(lambda x: has_any(x, ["insulin", "gansulin", "humulin", "novolin", "lantus", "levemir", "tresiba", "glargine", "detemir", "degludec", "lispro", "aspart", "novomix", "humalog", "novorapid"]))
    o["rx_metformin"] = ag.apply(lambda x: has_any(x, ["metformin"]))
    o["rx_sulfonylurea"] = ag.apply(lambda x: has_any(x, ["glimepiride", "gliclazide", "glipizide", "gliquidone", "glibenclamide"]))
    o["rx_acarbose"] = ag.apply(lambda x: has_any(x, ["acarbose", "voglibose"]))
    o["rx_sglt2"] = ag.apply(lambda x: has_any(x, ["gliflozin"]))
    o["rx_dpp4"] = ag.apply(lambda x: has_any(x, ["gliptin"]))
    o["rx_glp1"] = ag.apply(lambda x: has_any(x, ["glutide", "exenatide"]))
    o["n_antidiabetic_drugs"] = ag.apply(lambda x: 0 if x.lower() in ("none", "nan") else len(re.split(r"[,;]", x)))
    co = s["Comorbidities"].astype(str)
    o["hypertension"] = co.apply(lambda x: has_any(x, ["hypertension"]))
    o["dyslipidemia"] = co.apply(lambda x: has_any(x, ["lipid", "cholesterol"]))
    for k, c in {"micro_nephropathy": ("Diabetic Microvascular Complications", "nephropathy"),
                 "micro_retinopathy": ("Diabetic Microvascular Complications", "retinopathy"),
                 "micro_neuropathy": ("Diabetic Microvascular Complications", "neuropathy")}.items():
        o[k] = s[c[0]].apply(lambda x: has_any(x, [c[1]]))
    o["macrovascular"] = s["Diabetic Macrovascular  Complications"].astype(str).str.lower().apply(lambda x: int(x not in ("none", "nan")))
    return o


def load_series(path):
    d = pd.read_excel(path)
    d.columns = [str(c).strip() for c in d.columns]
    cgm = [c for c in d.columns if c.startswith("CGM")][0]
    d = d.rename(columns={cgm: "cgm", "Date": "time"})
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d = d.dropna(subset=["time"]).sort_values("time").drop_duplicates("time")
    meal = d["Dietary intake"].apply(parse_meal) if "Dietary intake" in d else pd.Series([(np.nan,) * 3] * len(d), index=d.index)
    d["meal_g"] = [m[0] for m in meal]; d["meal_carb_g"] = [m[1] for m in meal]
    d["insulin_iu"] = d["Insulin dose - s.c."].apply(parse_insulin) if "Insulin dose - s.c." in d else np.nan
    d["cbg"] = d["CBG (mg / dl)"] if "CBG (mg / dl)" in d else np.nan
    d = d[["time", "cgm", "meal_g", "meal_carb_g", "insulin_iu"]].set_index("time")
    # snap to a regular 15-minute grid; meals/insulin are events -> sum into the nearest bin
    grid = pd.date_range(d.index.min().ceil("15min"), d.index.max().floor("15min"), freq="15min")
    cg = d["cgm"].astype(float)
    cg = cg.groupby(cg.index.round("15min")).mean().reindex(grid)
    cg = cg.interpolate(limit=MAX_INTERP, limit_area="inside")
    ev = d[["meal_g", "meal_carb_g", "insulin_iu"]].copy()
    ev.index = ev.index.round("15min")
    ev = ev.groupby(level=0).sum(min_count=1).reindex(grid)
    out = pd.concat([cg.rename("cgm"), ev], axis=1)
    out["has_insulin_record"] = int(d["insulin_iu"].notna().any())
    for c in ["meal_g", "meal_carb_g", "insulin_iu"]:
        out[c] = out[c].fillna(0.0)
    out.index.name = "time"
    return out.reset_index()


def time_since(event_mask, cap=96):
    """Steps since the last True (capped)."""
    out = np.full(len(event_mask), cap, dtype=float); last = None
    for i, e in enumerate(event_mask):
        if e: last = i
        out[i] = cap if last is None else min(i - last, cap)
    return out


def make_windows(ts):
    g = ts["cgm"].values; n = len(g); rows = []
    t_meal = time_since(ts["meal_g"].values > 0)
    t_ins = time_since(ts["insulin_iu"].values > 0)
    meal_c = np.minimum(ts["meal_carb_g"].rolling(12, min_periods=1).sum().values, 400)   # last 3 h, clipped (parsing outliers)
    meal_t = np.minimum(ts["meal_g"].rolling(12, min_periods=1).sum().values, 1000)
    ins_4h = ts["insulin_iu"].rolling(16, min_periods=1).sum().values
    hr, st = ts["heart_rate"].values, ts["steps"].values
    hr1 = pd.Series(hr).rolling(4, min_periods=1).mean().values
    hr_sd = pd.Series(hr).rolling(8, min_periods=2).std().values
    st1 = pd.Series(st).rolling(4, min_periods=1).sum().values
    st2 = pd.Series(st).rolling(8, min_periods=1).sum().values
    for i in range(LOOKBACK - 1, n - HORIZON):
        hist = g[i - LOOKBACK + 1:i + 1]; fut = g[i + 1:i + 1 + HORIZON]
        if np.isnan(hist).any() or np.isnan(fut).any() or g[i] > THRESH:
            continue
        over = np.where(fut > THRESH)[0]
        t = ts["time"].iloc[i]
        rows.append(dict(
            time=t, cgm=g[i], d15=g[i] - g[i - 1], d30=g[i] - g[i - 2], d60=g[i] - g[i - 4], d120=g[i] - g[i - 7],
            mean2h=hist.mean(), std2h=hist.std(), min2h=hist.min(), max2h=hist.max(),
            slope2h=np.polyfit(np.arange(LOOKBACK), hist, 1)[0],
            min_since_meal=t_meal[i] * STEP_MIN, min_since_insulin=t_ins[i] * STEP_MIN,
            meal_carb_3h=meal_c[i], meal_g_3h=meal_t[i], insulin_iu_4h=ins_4h[i],
            has_insulin_record=ts["has_insulin_record"].iloc[0],
            hr_1h=hr1[i], hr_sd_2h=hr_sd[i], steps_1h=st1[i], steps_2h=st2[i],
            hour_sin=np.sin(2 * np.pi * (t.hour + t.minute / 60) / 24), hour_cos=np.cos(2 * np.pi * (t.hour + t.minute / 60) / 24),
            y=int(len(over) > 0), minutes_to_cross=(over[0] + 1) * STEP_MIN if len(over) else np.nan))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/Shanghai_T2DM")
    ap.add_argument("--summary", default="data/raw/Shanghai_T2DM_Summary.xlsx")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stat = static_features(a.summary).set_index("file_id")
    ts_all, win_all = [], []
    for f in sorted(glob.glob(os.path.join(a.raw, "*.xls*"))):
        fid = os.path.splitext(os.path.basename(f))[0]
        if fid not in stat.index:
            print("skip (no summary row):", fid); continue
        ts = load_series(f)
        ts = simulate_wearables(ts, stat.loc[fid], seed=a.seed + int(fid.split("_")[0]))
        ts.insert(0, "file_id", fid); ts.insert(1, "patient_id", fid.split("_")[0])
        w = make_windows(ts)
        if w.empty: continue
        w.insert(0, "file_id", fid); w.insert(1, "patient_id", fid.split("_")[0])
        ts_all.append(ts); win_all.append(w)
    ts = pd.concat(ts_all); win = pd.concat(win_all)
    win = win.merge(stat.drop(columns=["patient_id"]).reset_index(), on="file_id", how="left")
    ts.to_csv(os.path.join(a.out, "timeseries.csv.gz"), index=False)
    win.to_csv(os.path.join(a.out, "windows.csv.gz"), index=False)
    print(f"files={win.file_id.nunique()} patients={win.patient_id.nunique()} windows={len(win)} positive={win.y.sum()} ({win.y.mean():.1%})")


if __name__ == "__main__":
    main()
