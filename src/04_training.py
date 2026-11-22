"""
Pelatihan, Tuning Hyperparameter, dan Evaluasi Model
Studi Kasus : Prediksi Harga Tiket Pesawat dengan Random Forest Regression

Konfigurasi model yang dievaluasi:
  1. Regresi Linear (Baseline)
  2. Decision Tree Tunggal (Baseline)
  3. Random Forest Regression (Model Usulan, dengan RandomizedSearchCV 5-Fold CV)
  4. Random Forest tanpa fitur class (Uji Ketahanan / Ablation Study)

Strategi pelatihan:
  - Tahap 1: Tuning parameter via RandomizedSearchCV pada sampel 25.000 data latih
  - Tahap 2: Refit parameter terbaik pada seluruh 238.352 data latih

Output:
  - reports/04_training_metrics.csv & .json
  - reports/04_tuning_log.json
  - models/*.joblib
  - figures/model_*.png
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=UserWarning)

BASE = Path("E:/data-mining-flight-price")
OUT = BASE / "data" / "processed"
MODELS = BASE / "models"
REP = BASE / "reports"
FIG = BASE / "figures"
SEED = 42

sns.set_theme(style="whitegrid")
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def section(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


SECTION_START = time.time()

# ======================================================================
section("1. LOAD ARTIFACT HASIL PREPROCESSING")
X_train = sp.load_npz(OUT / "X_train.npz").tocsr()
X_test = sp.load_npz(OUT / "X_test.npz").tocsr()
y_train = np.load(OUT / "y_train.npy")
y_test = np.load(OUT / "y_test.npy")

with open(REP / "03_preprocessing_schema.json", encoding="utf-8") as f:
    schema = json.load(f)

FEATURES = schema["feature_names"]
CLASS_COLS = set(schema["class_onehot_columns"])
keep_idx = [i for i, n in enumerate(FEATURES) if n not in CLASS_COLS]

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"Fitur   : {len(FEATURES)} kolom  ({len(CLASS_COLS)} di antaranya one-hot `class`)")
print(f"Varian tanpa class akan memakai {len(keep_idx)} kolom")

X_train_nc = X_train[:, keep_idx].tocsr()
X_test_nc = X_test[:, keep_idx].tocsr()
print(f"X_train tanpa class : {X_train_nc.shape}")


# ======================================================================
# METRIK
# ======================================================================
def evaluate(name, model, Xtr, Xte, ytr, yte, train_seconds):
    t1 = time.time()
    pred_train = model.predict(Xtr)
    pred = model.predict(Xte)
    predict_seconds = time.time() - t1

    m = {
        "model": name,
        "r2_train": float(r2_score(ytr, pred_train)),
        "r2": float(r2_score(yte, pred)),
        "mae": float(mean_absolute_error(yte, pred)),
        "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        "mape": float(np.mean(np.abs((yte - pred) / yte)) * 100),
        "train_seconds": round(train_seconds, 2),
        "predict_seconds": round(predict_seconds, 2),
        "n_features": int(Xtr.shape[1]),
    }

    # Error dalam Rupiah-ish: berapa persen prediksi yang masuk toleransi 10% & 20%
    ape = np.abs((yte - pred) / yte)
    m["pct_within_10pct"] = float((ape <= 0.10).mean() * 100)
    m["pct_within_20pct"] = float((ape <= 0.20).mean() * 100)

    m["_pred"] = pred
    m["_pred_train"] = pred_train
    return m


results = []


# ======================================================================
section("2. BASELINE 1 — REGRESI LINEAR")
t = time.time()
lr = LinearRegression(n_jobs=-1)
lr.fit(X_train, y_train)
m_lr = evaluate("Linear Regression", lr, X_train, X_test, y_train, y_test, time.time() - t)
print(f"R2 test : {m_lr['r2']:.4f}   RMSE : {m_lr['rmse']:,.1f}   "
      f"MAE : {m_lr['mae']:,.1f}   MAPE : {m_lr['mape']:.2f}%   "
      f"({m_lr['train_seconds']}s)")
results.append(m_lr)
joblib.dump(lr, MODELS / "linear_regression.joblib")


# ======================================================================
section("3. BASELINE 2 — DECISION TREE TUNGGAL")
t = time.time()
dt = DecisionTreeRegressor(random_state=SEED, min_samples_leaf=5)
dt.fit(X_train, y_train)
m_dt = evaluate("Decision Tree", dt, X_train, X_test, y_train, y_test, time.time() - t)
print(f"R2 test : {m_dt['r2']:.4f}   RMSE : {m_dt['rmse']:,.1f}   "
      f"MAE : {m_dt['mae']:,.1f}   MAPE : {m_dt['mape']:.2f}%   "
      f"({m_dt['train_seconds']}s)")
print(f"Kedalaman pohon: {dt.get_depth()}  |  jumlah daun: {dt.get_n_leaves():,}")
results.append(m_dt)
joblib.dump(dt, MODELS / "decision_tree.joblib")


# ======================================================================
section("4. RANDOM FOREST — TAHAP 1: TUNING PADA SAMPEL 25.000 BARIS")
N_TUNE = 25000
X_tune, _, y_tune, _ = train_test_split(
    X_train, y_train, train_size=N_TUNE, random_state=SEED, shuffle=True
)
print(f"Sampel tuning: {X_tune.shape[0]:,} baris x {X_tune.shape[1]} fitur")

param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [8, 12, 16, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
}

search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=SEED, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=15,
    cv=3,
    scoring="r2",
    random_state=SEED,
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)

t = time.time()
search.fit(X_tune, y_tune)
tune_seconds = time.time() - t
print(f"\nTuning selesai dalam {tune_seconds:.1f} detik  "
      f"({len(search.cv_results_['params'])} kandidat x 3 fold)")
print(f"Best CV R2 : {search.best_score_:.4f}")
print(f"Best params: {search.best_params_}")

cv_table = pd.DataFrame(search.cv_results_)[
    ["params", "mean_test_score", "std_test_score", "mean_train_score", "rank_test_score"]
].sort_values("rank_test_score")
cv_table["params"] = cv_table["params"].astype(str)
print("\nTop 5 konfigurasi:")
print(cv_table.head(5).to_string(index=False))
cv_table.to_csv(REP / "04_tuning_cv_results.csv", index=False)

with open(REP / "04_tuning_log.json", "w", encoding="utf-8") as f:
    json.dump({
        "n_tune_rows": N_TUNE,
        "n_iter": 15,
        "cv": 3,
        "scoring": "r2",
        "tuning_seconds": round(tune_seconds, 1),
        "best_cv_r2": float(search.best_score_),
        "best_params": {k: (v if not isinstance(v, np.integer) else int(v))
                        for k, v in search.best_params_.items()},
        "top5": cv_table.head(5).to_dict("records"),
    }, f, indent=2, ensure_ascii=False, default=str)


# ======================================================================
section("5. RANDOM FOREST — TAHAP 2: REFIT PARAMETER TERBAIK PADA SELURUH TRAIN")
print(f"Training pada {X_train.shape[0]:,} baris x {X_train.shape[1]} fitur...")
best_params = dict(search.best_params_)
t = time.time()
rf = RandomForestRegressor(random_state=SEED, n_jobs=-1, **best_params)
rf.fit(X_train, y_train)
m_rf = evaluate("Random Forest", rf, X_train, X_test, y_train, y_test, time.time() - t)
print(f"R2 test : {m_rf['r2']:.4f}   RMSE : {m_rf['rmse']:,.1f}   "
      f"MAE : {m_rf['mae']:,.1f}   MAPE : {m_rf['mape']:.2f}%   "
      f"({m_rf['train_seconds']}s)")
print(f"Jumlah pohon: {len(rf.estimators_)}  |  kedalaman maks tercapai: {rf.max_depth}")
results.append(m_rf)
joblib.dump(rf, MODELS / "random_forest.joblib")


# ======================================================================
section("6. UJI KETAHANAN — RANDOM FOREST TANPA FITUR `class`")
t = time.time()
rf_nc = RandomForestRegressor(random_state=SEED, n_jobs=-1, **best_params)
rf_nc.fit(X_train_nc, y_train)
m_nc = evaluate("Random Forest (tanpa class)", rf_nc, X_train_nc, X_test_nc,
                y_train, y_test, time.time() - t)
print(f"R2 test : {m_nc['r2']:.4f}   RMSE : {m_nc['rmse']:,.1f}   "
      f"MAE : {m_nc['mae']:,.1f}   MAPE : {m_nc['mape']:.2f}%   "
      f"({m_nc['train_seconds']}s)")
print(f"\nDelta R2 vs RF penuh: {m_nc['r2'] - m_rf['r2']:+.4f}")
results.append(m_nc)
joblib.dump(rf_nc, MODELS / "random_forest_no_class.joblib")


# ======================================================================
section("7. TABEL PERBANDINGAN MODEL")
cols = ["model", "n_features", "r2_train", "r2", "rmse", "mae", "mape",
        "pct_within_10pct", "pct_within_20pct", "train_seconds", "predict_seconds"]
tbl = pd.DataFrame([{k: m[k] for k in cols} for m in results])
tbl = tbl.sort_values("r2", ascending=False).reset_index(drop=True)

print(tbl.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

tbl.to_csv(REP / "04_training_metrics.csv", index=False)
with open(REP / "04_training_metrics.json", "w", encoding="utf-8") as f:
    json.dump({
        "metrics": [{k: m[k] for k in cols} for m in results],
        "best_params": {k: (int(v) if isinstance(v, np.integer) else v)
                        for k, v in best_params.items()},
        "delta_r2_without_class": round(m_nc["r2"] - m_rf["r2"], 4),
    }, f, indent=2, ensure_ascii=False, default=str)

print("\nInterpretasi cepat:")
print(f"  RF vs Linear Regression : R2 {m_rf['r2']:.4f} vs {m_lr['r2']:.4f} "
      f"({(m_rf['r2'] - m_lr['r2']) * 100:+.2f} poin persentase)")
print(f"  RF vs Decision Tree     : RMSE {m_rf['rmse']:,.0f} vs {m_dt['rmse']:,.0f} "
      f"({(m_rf['rmse'] / m_dt['rmse'] - 1) * 100:+.1f}% lebih rendah error)")


# ======================================================================
section("8. VISUALISASI: ACTUAL VS PREDICTED")
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
plot_specs = [
    (m_lr, "Linear Regression", axes[0, 0]),
    (m_dt, "Decision Tree", axes[0, 1]),
    (m_rf, "Random Forest", axes[1, 0]),
    (m_nc, "Random Forest (tanpa class)", axes[1, 1]),
]

for m, title, ax in plot_specs:
    pred = m["_pred"]
    ax.scatter(y_test, pred, s=4, alpha=0.18, color="#2E86AB", edgecolors="none")
    lim = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lim, lim, "r--", lw=1.4, label="Prediksi sempurna")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_title(f"{title}\nR2={m['r2']:.4f}  RMSE={m['rmse']:,.0f}")
    ax.set_xlabel("Harga aktual (INR)")
    ax.set_ylabel("Harga prediksi (INR)")
    ax.legend(fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig(FIG / "model_01_actual_vs_predicted.png")
plt.close()
print("-> figures/model_01_actual_vs_predicted.png")


# ======================================================================
section("9. VISUALISASI: PERBANDINGAN METRIK & RESIDUAL")
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

ax = axes[0]
sns.barplot(data=tbl, y="model", x="r2", hue="model", palette="crest", legend=False, ax=ax)
for i, v in enumerate(tbl["r2"]):
    ax.text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=9)
ax.set_xlim(0, 1.05)
ax.set_title("R2 (semakin tinggi semakin baik)")
ax.set_xlabel("R2")
ax.set_ylabel("")

ax = axes[1]
sns.barplot(data=tbl, y="model", x="rmse", hue="model", palette="flare", legend=False, ax=ax)
for i, v in enumerate(tbl["rmse"]):
    ax.text(v + 200, i, f"{v:,.0f}", va="center", fontsize=9)
ax.set_title("RMSE (semakin rendah semakin baik)")
ax.set_xlabel("RMSE (INR)")
ax.set_ylabel("")

ax = axes[2]
sns.barplot(data=tbl, y="model", x="mape", hue="model", palette="mako", legend=False, ax=ax)
for i, v in enumerate(tbl["mape"]):
    ax.text(v + 0.3, i, f"{v:.2f}%", va="center", fontsize=9)
ax.set_title("MAPE (semakin rendah semakin baik)")
ax.set_xlabel("MAPE (%)")
ax.set_ylabel("")

plt.tight_layout()
plt.savefig(FIG / "model_02_comparison.png")
plt.close()
print("-> figures/model_02_comparison.png")

# Residual plot RF
resid = y_test - m_rf["_pred"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

ax = axes[0]
ax.scatter(m_rf["_pred"], resid, s=4, alpha=0.2, color="#C73E1D", edgecolors="none")
ax.axhline(0, color="black", lw=1.2)
ax.set_title(f"Residual Random Forest (mean={resid.mean():,.0f})")
ax.set_xlabel("Harga prediksi (INR)")
ax.set_ylabel("Residual = aktual - prediksi")

ax = axes[1]
ax.hist(resid, bins=90, color="#C73E1D", edgecolor="white", linewidth=0.2)
ax.axvline(0, color="black", lw=1.2)
ax.set_title("Distribusi residual Random Forest")
ax.set_xlabel("Residual (INR)")
ax.set_ylabel("Frekuensi")

plt.tight_layout()
plt.savefig(FIG / "model_03_residual.png")
plt.close()
print("-> figures/model_03_residual.png")

# Akurasi per segmen kelas
print("\nAkurasi RF dipecah per segmen kelas kabin (data uji):")
seg = {"Economy": None, "Business": None}
# Rekonstruksi label kelas dari one-hot: kolom class_* hanya ada 1
class_col_idx = [i for i, n in enumerate(FEATURES) if n in CLASS_COLS]
assert len(class_col_idx) == 1
# PENTING: OneHotEncoder(drop='first') membuang kategori pertama secara alfabetis.
# Urutan kategori `class` = ['Business', 'Economy'] -> yang dibuang 'Business'.
# Jadi satu-satunya kolom one-hot yang tersisa adalah class_Economy.
assert FEATURES[class_col_idx[0]] == "class_Economy", FEATURES[class_col_idx[0]]
is_economy = np.asarray(X_test[:, class_col_idx[0]].todense()).ravel() > 0.5
for label, mask in [("Economy", is_economy), ("Business", ~is_economy)]:
    seg[label] = {
        "n": int(mask.sum()),
        "r2": round(float(r2_score(y_test[mask], m_rf["_pred"][mask])), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test[mask], m_rf["_pred"][mask]))), 1),
        "mae": round(float(mean_absolute_error(y_test[mask], m_rf["_pred"][mask])), 1),
        "mape": round(float(np.mean(np.abs((y_test[mask] - m_rf["_pred"][mask]) /
                                          y_test[mask])) * 100), 2),
    }
    print(f"  {label:9s} n={seg[label]['n']:>6,}  R2={seg[label]['r2']:.4f}  "
          f"RMSE={seg[label]['rmse']:>9,.1f}  MAPE={seg[label]['mape']:.2f}%")

with open(REP / "04_segment_metrics.json", "w", encoding="utf-8") as f:
    json.dump(seg, f, indent=2, ensure_ascii=False)

section(f"SELESAI TOTAL dalam {(time.time() - SECTION_START) / 60:.1f} menit")
