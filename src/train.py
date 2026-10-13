"""Train and evaluate GlucoTwin models with patient-grouped cross-validation.

Models : LightGBM (gradient boosting) and an MLP fusion network (scikit-learn).
Ablation: current glucose only / dynamic only / static only / dynamic+static / +simulated wearables.
Outputs: results/metrics.csv, results/*.png, results/oof_predictions.csv.gz, models/*
Run    : python src/train.py
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DYN = ["cgm", "d15", "d30", "d60", "d120", "mean2h", "std2h", "min2h", "max2h", "slope2h",
       "min_since_meal", "min_since_insulin", "meal_carb_3h", "meal_g_3h", "insulin_iu_4h",
       "has_insulin_record", "hour_sin", "hour_cos"]
WEAR = ["hr_1h", "hr_sd_2h", "steps_1h", "steps_2h"]
EXCL = ["file_id", "patient_id", "time", "y", "minutes_to_cross"]


def gbm():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                              random_state=0, verbose=-1, n_jobs=2)


def mlp():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                       max_iter=200, random_state=0))


def oof(make, X, y, groups, k=5):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(k).split(X, y, groups):
        m = make().fit(X.iloc[tr], y.iloc[tr]); p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p


def boot_ci(y, p, groups, n=300, seed=0):
    """95% CI of AUROC by resampling PATIENTS (not windows)."""
    rng = np.random.default_rng(seed); ids = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ids}
    s = []
    for _ in range(n):
        pick = np.concatenate([idx[g] for g in rng.choice(ids, len(ids))])
        if y[pick].min() != y[pick].max(): s.append(roc_auc_score(y[pick], p[pick]))
    return np.percentile(s, [2.5, 97.5])


def alert_stats(y, p, mtc, recall_target=0.80):
    """Threshold giving >= recall_target; precision, alert rate and lead time of caught events."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= recall_target)[0]; t = thr[ok[-1]]
    a = p >= t; tp = a & (y == 1)
    return dict(threshold=float(t), recall=float(tp.sum() / y.sum()), precision=float(tp.sum() / max(a.sum(), 1)),
                alerts_per_day=float(a.mean() * 96), median_lead_min=float(np.nanmedian(mtc[tp])),
                mean_lead_min=float(np.nanmean(mtc[tp])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/windows.csv.gz"); ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="models"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.models, exist_ok=True)
    w = pd.read_csv(a.data); y = w["y"]; g = w["patient_id"].values; mtc = w["minutes_to_cross"].values
    feats = [c for c in w.columns if c not in EXCL]
    STAT = [c for c in feats if c not in DYN + WEAR]
    sets = {"1 Current glucose only": ["cgm"], "2 Dynamic only (CGM+meals+insulin)": DYN, "3 Static EHR only": STAT,
            "4 Fused: dynamic + static EHR": DYN + STAT, "5 Fused + simulated wearables": DYN + STAT + WEAR}
    rows, preds = [], {}
    for mname, make in [("LightGBM", gbm), ("MLP fusion net", mlp)]:
        for sname, f in sets.items():
            if mname == "MLP fusion net" and sname.startswith("1"): continue
            p = oof(make, w[f], y, g); preds[(mname, sname)] = p
            lo, hi = boot_ci(y.values, p, g)
            r = dict(model=mname, features=sname, auroc=roc_auc_score(y, p), auroc_ci_low=lo, auroc_ci_high=hi,
                     auprc=average_precision_score(y, p)); r.update(alert_stats(y.values, p, mtc)); rows.append(r)
            print(f"{mname:15s} {sname:38s} AUROC {r['auroc']:.3f} [{lo:.3f},{hi:.3f}] AUPRC {r['auprc']:.3f}", flush=True)
    m = pd.DataFrame(rows); m.to_csv(os.path.join(a.out, "metrics.csv"), index=False)
    print("prevalence (random AUPRC) = %.3f" % y.mean())

    # ---- plots (colour-blind-safe, labelled) ----
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    C = {"1": "#7f7f7f", "2": "#0072B2", "3": "#E69F00", "4": "#009E73", "5": "#CC79A7"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for (mn, sn), p in preds.items():
        if mn != "LightGBM": continue
        fpr, tpr, _ = roc_curve(y, p); ax[0].plot(fpr, tpr, color=C[sn[0]], label=sn[2:])
        pr, rc, _ = precision_recall_curve(y, p); ax[1].plot(rc, pr, color=C[sn[0]], label=sn[2:])
    ax[0].plot([0, 1], [0, 1], "k:", lw=1); ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (patient-grouped CV)")
    ax[1].axhline(y.mean(), color="k", ls=":", lw=1); ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(os.path.join(a.out, "roc_pr.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4))
    mm = m[m.model == "LightGBM"]; yy = np.arange(len(mm))[::-1]
    ax.barh(yy, mm.auroc, color=[C[s[0]] for s in mm.features], xerr=[mm.auroc - mm.auroc_ci_low, mm.auroc_ci_high - mm.auroc], capsize=3)
    ax.set_yticks(yy, [s[2:] for s in mm.features]); ax.set_xlim(0.5, 0.95); ax.set_xlabel("AUROC (95% CI by patient bootstrap)")
    for yi, v, hi in zip(yy, mm.auroc, mm.auroc_ci_high): ax.text(hi + 0.008, yi, f"{v:.3f}", va="center", fontsize=9)
    ax.set_title("Ablation: what each data stream adds"); fig.tight_layout(); fig.savefig(os.path.join(a.out, "ablation.png"), dpi=160); plt.close(fig)

    key = ("LightGBM", "4 Fused: dynamic + static EHR"); p = preds[key]; st = alert_stats(y.values, p, mtc)
    tp = (p >= st["threshold"]) & (y.values == 1)
    fig, ax = plt.subplots(figsize=(6.5, 4)); ax.hist(mtc[tp], bins=np.arange(7.5, 135, 15), color="#009E73", edgecolor="white")
    ax.set(xlabel="Warning lead time before glucose crosses 180 mg/dL (minutes)", ylabel="Caught events",
           title=f"Lead time at {st['recall']:.0%} recall (median {st['median_lead_min']:.0f} min)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "lead_time.png"), dpi=160); plt.close(fig)

    # ---- artefacts for the dashboard + final model on all data ----
    o = w[["file_id", "patient_id", "time", "cgm", "y", "minutes_to_cross"]].copy(); o["risk"] = p
    o.to_csv(os.path.join(a.out, "oof_predictions.csv.gz"), index=False)
    f = DYN + STAT; fin = gbm().fit(w[f], y); fin.booster_.save_model(os.path.join(a.models, "glucotwin_lgbm.txt"))
    json.dump({"features": f, "alert_threshold": st["threshold"], "target": "CGM > 180 mg/dL within 120 min, given CGM <= 180"},
              open(os.path.join(a.models, "model_card.json"), "w"), indent=2)
    imp = pd.Series(fin.booster_.feature_importance("gain"), index=f).sort_values(ascending=False)
    imp.head(15).to_csv(os.path.join(a.out, "feature_importance.csv"))
    print("\nTop features by gain:\n", (imp / imp.sum()).head(10).round(3).to_string())


if __name__ == "__main__":
    main()
