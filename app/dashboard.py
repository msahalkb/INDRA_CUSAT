"""GlucoTwin - doctor dashboard (conceptual UI for the virtual patient).

Run from the repo root:  streamlit run app/dashboard.py
Replay mode: pick a patient recording, move the "now" slider, and the twin shows the
2-hour risk forecast, the factors behind it, and what actually happened next.
Heart rate and steps are SIMULATED. Not a medical device.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import lightgbm as lgb
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
st.set_page_config(page_title="GlucoTwin", page_icon="🩸", layout="wide")

NAMES = {"cgm": "Current glucose", "d15": "Change, last 15 min", "d30": "Change, last 30 min", "d60": "Change, last 1 h",
         "d120": "Change, last 2 h", "mean2h": "Average glucose, 2 h", "std2h": "Glucose variability, 2 h",
         "min2h": "Lowest glucose, 2 h", "max2h": "Highest glucose, 2 h", "slope2h": "Glucose slope, 2 h",
         "min_since_meal": "Time since last meal", "min_since_insulin": "Time since last insulin",
         "meal_carb_3h": "Carbohydrate food, last 3 h", "meal_g_3h": "Food eaten, last 3 h",
         "insulin_iu_4h": "Insulin, last 4 h", "has_insulin_record": "Insulin is recorded for this patient",
         "hour_sin": "Time of day", "hour_cos": "Time of day", "age": "Age", "bmi": "BMI", "hba1c_pct": "HbA1c",
         "fpg": "Fasting glucose (lab)", "diabetes_years": "Years with diabetes", "glycated_albumin": "Glycated albumin",
         "cpep_fast": "Fasting C-peptide", "hdl": "HDL cholesterol", "ldl": "LDL cholesterol", "tg": "Triglycerides", "chol": "Total cholesterol", "creatinine": "Creatinine", "uric_acid": "Uric acid", "bun": "Blood urea nitrogen", "ins_fast": "Fasting insulin (lab)", "ins_2h": "Post-meal insulin (lab)", "pp2h_glucose": "Post-meal glucose (lab)", "sex_male": "Sex", "smoking_pack_years": "Smoking history", "drinker": "Alcohol use", "hx_hypoglycemia": "History of low glucose", "cpep_2h": "Post-meal C-peptide", "egfr": "Kidney function (eGFR)"}
RISK_COLOR = {"Low": "#2a9d8f", "Moderate": "#e9a23b", "High": "#d1495b"}


@st.cache_data(show_spinner=False)
def load():
    w = pd.read_csv(ROOT / "data/processed/windows.csv.gz", parse_dates=["time"])
    ts = pd.read_csv(ROOT / "data/processed/timeseries.csv.gz", parse_dates=["time"])
    oof = pd.read_csv(ROOT / "results/oof_predictions.csv.gz", parse_dates=["time"])[["file_id", "time", "risk"]]
    w = w.merge(oof, on=["file_id", "time"], how="left")
    return w, ts


@st.cache_resource(show_spinner=False)
def load_model():
    card = json.load(open(ROOT / "models/model_card.json"))
    return lgb.Booster(model_file=str(ROOT / "models/glucotwin_lgbm.txt")), card


def band(r, thr):
    return "High" if r >= thr else ("Moderate" if r >= 0.5 * thr else "Low")


def alerts_for(d, thr):
    out, last = [], None
    for t, r in zip(d.time, d.risk):
        if r >= thr and (last is None or (t - last).total_seconds() >= 7200):
            out.append(t); last = t
    return out


w, ts = load(); model, card = load_model()

# ---------------- sidebar ----------------
st.sidebar.title("🩸 GlucoTwin")
st.sidebar.caption("Digital twin for 2-hour glucose spike prediction in type 2 diabetes")
files = sorted(w.file_id.unique())
fid = st.sidebar.selectbox("Patient recording", files, index=files.index("2010_0_20220111") if "2010_0_20220111" in files else 0)
thr = st.sidebar.slider("Alert sensitivity threshold", 0.30, 0.80, 0.50, 0.05,
                        help="Lower = more alerts and more spikes caught. Higher = fewer alerts, more are true.")
st.sidebar.markdown("**Held-out accuracy at this threshold** (patient-grouped CV)")
try:
    tr = pd.read_csv(ROOT / "results/alert_tradeoff.csv"); row = tr.iloc[(tr.risk_threshold - thr).abs().argmin()]
    st.sidebar.metric("Alerts per day", f"{row.alerts_per_day:.1f}")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Alerts that were right", f"{row.alert_precision:.0%}"); c2.metric("Spikes warned", f"{row.event_recall:.0%}")
except Exception:
    pass
st.sidebar.info("Replay of recorded data. Heart rate and steps are **simulated**. Research prototype, not a medical device.")

d = w[w.file_id == fid].sort_values("time").reset_index(drop=True)
t_all = ts[ts.file_id == fid].sort_values("time").reset_index(drop=True)
p = d.iloc[0]

# ---------------- patient card ----------------
st.title(f"Virtual patient {fid.split('_')[0]}")
rx = [n for n, c in [("insulin", "rx_insulin"), ("metformin", "rx_metformin"), ("sulfonylurea", "rx_sulfonylurea"),
                     ("acarbose", "rx_acarbose"), ("SGLT2i", "rx_sglt2"), ("DPP-4i", "rx_dpp4"), ("GLP-1", "rx_glp1")] if p[c] == 1]
cx = [n for n, c in [("hypertension", "hypertension"), ("nephropathy", "micro_nephropathy"), ("retinopathy", "micro_retinopathy"),
                     ("neuropathy", "micro_neuropathy"), ("macrovascular disease", "macrovascular")] if p[c] == 1]
k = st.columns(6)
k[0].metric("Age / sex", f"{int(p.age)} / {'M' if p.sex_male else 'F'}")
k[1].metric("BMI", f"{p.bmi:.1f}")
k[2].metric("HbA1c", f"{p.hba1c_pct:.1f} %")
k[3].metric("Diabetes duration", f"{p.diabetes_years:.0f} y" if p.diabetes_years >= 1 else f"{max(round(p.diabetes_years * 12), 1)} mo")
k[4].metric("Time above 180", f"{(t_all.cgm > 180).mean():.0%}")
k[5].metric("Recording", f"{(t_all.time.max() - t_all.time.min()).days + 1} days")
st.caption(f"**Medication:** {', '.join(rx) or 'none recorded'}  ·  **Conditions:** {', '.join(cx) or 'none recorded'}")

# ---------------- day + now selector ----------------
days = sorted(t_all.time.dt.date.unique())
cand = d[(d.y == 1) & (d.risk >= 0.4)]                       # an early-warning moment: spike ahead, risk already high
best = cand.sort_values(["minutes_to_cross", "risk"], ascending=False).iloc[0] if len(cand) else d.loc[d.risk.fillna(0).idxmax()]
dcol, ncol = st.columns([1, 3])
day = dcol.selectbox("Day", days, index=days.index(best.time.date()), key=f"day_{fid}", format_func=lambda x: x.strftime("%a %d %b"))
dd = d[d.time.dt.date == day]
if dd.empty:
    st.warning("No forecastable moments on this day (glucose was above 180 or data gaps)."); st.stop()
times = list(dd.time)
default = best.time if best.time in set(dd.time) else dd.assign(r=dd.risk).sort_values("r").time.iloc[-1]
now = ncol.select_slider("Move 'now' through the day", options=times, value=default, key=f"now_{fid}_{day}", format_func=lambda x: x.strftime("%H:%M"))
row = d[d.time == now].iloc[0]

# ---------------- timeline ----------------
td = t_all[t_all.time.dt.date == day]
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.25, 0.20], vertical_spacing=0.06, specs=[[{}], [{}], [{"secondary_y": True}]],
                    subplot_titles=("Glucose (CGM)", "Predicted risk of exceeding 180 mg/dL within 2 h", "Heart rate & steps (simulated)"))
fig.add_hline(y=180, line_dash="dot", line_color="#d1495b", row=1, col=1)
fig.add_hrect(y0=70, y1=180, fillcolor="#2a9d8f", opacity=0.08, line_width=0, row=1, col=1)
fig.add_trace(go.Scatter(x=td.time, y=td.cgm, mode="lines", name="CGM", line=dict(color="#1f3a5f", width=2.2)), row=1, col=1)
meals = td[td.meal_g > 0]; ins = td[td.insulin_iu > 0]
fig.add_trace(go.Scatter(x=meals.time, y=meals.cgm, mode="markers", name="Meal", marker=dict(symbol="diamond", size=10, color="#e9a23b")), row=1, col=1)
fig.add_trace(go.Scatter(x=ins.time, y=ins.cgm, mode="markers", name="Insulin", marker=dict(symbol="triangle-up", size=10, color="#7b5ea7")), row=1, col=1)
al = [a for a in alerts_for(dd, thr)]
if al:
    ay = td.set_index("time").cgm.reindex(al)
    fig.add_trace(go.Scatter(x=al, y=ay, mode="markers", name="Alert fired", marker=dict(symbol="star", size=14, color="#d1495b", line=dict(width=1, color="white"))), row=1, col=1)
fig.add_trace(go.Scatter(x=dd.time, y=dd.risk, mode="lines", name="Risk", line=dict(color="#d1495b", width=2), fill="tozeroy",
                         fillcolor="rgba(209,73,91,0.12)"), row=2, col=1)
fig.add_hline(y=thr, line_dash="dash", line_color="#555", row=2, col=1)
fig.add_trace(go.Bar(x=td.time, y=td.steps, name="Steps (sim.)", marker_color="rgba(120,120,120,0.40)"), row=3, col=1, secondary_y=True)
fig.add_trace(go.Scatter(x=td.time, y=td.heart_rate, name="Heart rate (sim.)", line=dict(color="#4c78a8", width=1.6)), row=3, col=1, secondary_y=False)
fig.add_vrect(x0=now, x1=now + pd.Timedelta(hours=2), fillcolor="#d1495b", opacity=0.07, line_width=0)
fig.add_vline(x=now, line_color="#222", line_width=1.5)
fig.update_yaxes(title_text="mg/dL", row=1, col=1); fig.update_yaxes(title_text="bpm", row=3, col=1, secondary_y=False); fig.update_yaxes(title_text="steps", showgrid=False, row=3, col=1, secondary_y=True); fig.update_yaxes(range=[0, 1], tickformat=".0%", row=2, col=1)
fig.update_layout(height=640, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.08), hovermode="x unified",
                  plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(fig, width="stretch")

# ---------------- forecast card ----------------
risk = float(row.risk); b = band(risk, thr)
fut = t_all[(t_all.time > now) & (t_all.time <= now + pd.Timedelta(hours=2))]
f1, f2 = st.columns([1, 1.4])
with f1:
    st.subheader(f"Forecast at {now.strftime('%H:%M')}")
    st.markdown(f"<div style='font-size:46px;font-weight:700;color:{RISK_COLOR[b]}'>{risk:.0%}</div>"
                f"<div style='font-size:16px'>chance glucose exceeds 180 mg/dL in the next 2 hours · <b style='color:{RISK_COLOR[b]}'>{b.upper()}</b></div>",
                unsafe_allow_html=True)
    st.write(f"Glucose now **{row.cgm:.0f} mg/dL** ({row.d60:+.0f} over the last hour).")
    if b == "High":
        st.error("Alert: consider an early intervention (activity, meal adjustment, clinician review).")
    elif b == "Moderate":
        st.warning("Watch: risk is rising.")
    else:
        st.success("No action suggested.")
    if len(fut):
        st.caption(f"Replay: what actually happened next, peak **{fut.cgm.max():.0f} mg/dL** → "
                   f"{'exceeded 180' if fut.cgm.max() > 180 else 'stayed under 180'}.")
with f2:
    st.subheader("Why the twin thinks so")
    X = pd.DataFrame([row[card["features"]]]).astype(float)
    contrib = model.predict(X, pred_contrib=True)[0][:-1]
    c = pd.Series(contrib, index=card["features"])
    c.index = [NAMES.get(i, i) for i in c.index]; c = c.groupby(level=0).sum()
    c = c.reindex(c.abs().sort_values(ascending=False).index)[:6][::-1]
    fb = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h", marker_color=["#d1495b" if v > 0 else "#2a9d8f" for v in c.values]))
    fb.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="← lowers risk   |   raises risk →",
                     plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fb, width="stretch")

# ---------------- what-if ----------------
st.divider(); st.subheader("What-if: change the recent meal or insulin")
w1, w2, w3 = st.columns([1.3, 1.3, 1.2])
mode = w1.radio("Meal in the last 3 hours", ["As recorded", "No meal", "Add a meal"], horizontal=True)
X2 = X.copy()
if mode == "No meal":
    X2["meal_carb_3h"] = 0; X2["meal_g_3h"] = 0; X2["min_since_meal"] = 1440
elif mode == "Add a meal":
    ago = w1.slider("Eaten how many minutes ago", 0, 180, 30, 15)
    carbs = w1.slider("Carbohydrate food (g)", 20, 150, 60, 10)
    X2["meal_carb_3h"] = row.meal_carb_3h + carbs; X2["meal_g_3h"] = row.meal_g_3h + 2 * carbs
    X2["min_since_meal"] = min(float(row.min_since_meal), float(ago))
if row.has_insulin_record == 1:
    ex = w2.slider("Extra insulin in last 4 h (IU)", 0.0, 10.0, 0.0, 0.5); X2["insulin_iu_4h"] = row.insulin_iu_4h + ex
else:
    w2.caption("No insulin is recorded for this patient, so the insulin what-if is switched off.")
r0 = float(model.predict(X)[0]); r1 = float(model.predict(X2)[0])
w3.metric("Risk with this change", f"{r1:.0%}", f"{(r1 - r0) * 100:+.0f} points vs. as recorded ({r0:.0%})", delta_color="inverse")
st.caption("The twin responds mainly to **when** a meal happened, much less to its size or to small insulin changes, which is what the model "
           "learned from these 100 patients. This shows the model's learned behaviour, not a proven causal effect, and is not treatment advice. "
           "Baseline here comes from the final model (trained on all patients), so it can differ slightly from the held-out curve above.")
