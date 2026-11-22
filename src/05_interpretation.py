"""
Interpretasi Model (Model Interpretation & Explainability)
Studi Kasus : Prediksi Harga Tiket Pesawat dengan Random Forest Regression

Tahapan analisis kepentingan fitur:
  1. Gini Importance (penurunan impuritas bawaan Random Forest)
  2. Permutation Feature Importance pada data pengujian
  3. Permutation Importance per segmen (Ekonomi vs Bisnis)
  4. Partial Dependence Plot (PDP): analisis non-linear days_left dan duration

Output:
  - reports/05_importance_gini.csv
  - reports/05_importance_permutation.csv
  - reports/05_importance_by_segment.csv
  - reports/05_interpretation.json
  - figures/interp_01_gini.png
  - figures/interp_02_permutation.png
  - figures/interp_03_segment.png
  - figures/interp_04_pdp_days_left.png
  - figures/interp_05_pdp_duration.png
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import r2_score

BASE = Path("E:/data-mining-flight-price")
OUT = BASE / "data" / "processed"
REP = BASE / "reports"
FIG = BASE / "figures"
SEED = 42

sns.set_theme(style="whitegrid")
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["figure.dpi"] = 110

T0 = time.time()


def section(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


# ======================================================================
section("1. LOAD MODEL & DATA")
rf = joblib.load(BASE / "models" / "random_forest.joblib")
X_test = sp.load_npz(OUT / "X_test.npz").tocsr()
y_test = np.load(OUT / "y_test.npy")
schema = json.load(open(REP / "03_preprocessing_schema.json", encoding="utf-8"))
FEATURES = schema["feature_names"]

assert FEATURES[29] == "class_Economy"
is_eco = np.asarray(X_test[:, 29].todense()).ravel() > 0.5
print(f"X_test: {X_test.shape}   Economy={is_eco.sum():,}  Business={(~is_eco).sum():,}")

# fungsi bantu: petakan nama one-hot -> nama kolom asli
CAT_FEATURES = schema["cat_features"]
NUM_FEATURES = schema["num_features"]


def original_feature(name):
    for c in CAT_FEATURES:
        if name.startswith(c + "_"):
            return c
    return name


# ======================================================================
section("2. GINI IMPORTANCE (BAWAAN RANDOM FOREST)")
gini = pd.DataFrame({
    "feature": FEATURES,
    "gini_importance": rf.feature_importances_,
}).sort_values("gini_importance", ascending=False).reset_index(drop=True)
gini["pct"] = (gini["gini_importance"] * 100).round(2)
gini["original"] = gini["feature"].map(original_feature)

print(gini.head(15).to_string(index=False))

gini.to_csv(REP / "05_importance_gini.csv", index=False)

# Agregasi per kolom asli
gini_agg = (gini.groupby("original")["gini_importance"].sum()
              .sort_values(ascending=False).reset_index())
gini_agg["pct"] = (gini_agg["gini_importance"] * 100).round(2)
print("\nAgregasi per kolom asli (one-hot digabung):")
print(gini_agg.to_string(index=False))

# ======================================================================
section("3. PERMUTATION IMPORTANCE (DATA UJI, 10.000 SAMPEL)")
rng = np.random.default_rng(SEED)
idx = rng.choice(X_test.shape[0], size=10000, replace=False)
Xp = np.asarray(X_test[idx].todense())
yp = y_test[idx]

t = time.time()
perm = permutation_importance(
    rf, Xp, yp, n_repeats=5, random_state=SEED,
    scoring="r2", n_jobs=-1,
)
print(f"Selesai dalam {time.time() - t:.1f} detik "
      f"({len(FEATURES)} fitur x 5 ulangan)")

perm_df = pd.DataFrame({
    "feature": FEATURES,
    "perm_mean": perm.importances_mean,
    "perm_std": perm.importances_std,
}).sort_values("perm_mean", ascending=False).reset_index(drop=True)
perm_df["original"] = perm_df["feature"].map(original_feature)
perm_df["drop_r2_pp"] = (perm_df["perm_mean"] * 100).round(3)

print(perm_df.head(15).to_string(index=False))
perm_df.to_csv(REP / "05_importance_permutation.csv", index=False)

perm_agg = (perm_df.groupby("original")["perm_mean"].sum()
              .sort_values(ascending=False).reset_index())
perm_agg["drop_r2_pp"] = (perm_agg["perm_mean"] * 100).round(3)
print("\nAgregasi per kolom asli:")
print(perm_agg.to_string(index=False))

print("\nPerbandingan peringkat Gini vs Permutation (top 8):")
cmp = pd.DataFrame({
    "gini_rank": gini.head(8).reset_index().set_index("feature")["index"].add(1),
    "perm_rank": perm_df.reset_index().set_index("feature")["index"].add(1),
}).fillna("-")
print(cmp.to_string())

# ======================================================================
section("4. PERMUTATION IMPORTANCE PER SEGMEN")
seg_imp = {}
for label, mask in [("Economy", is_eco), ("Business", ~is_eco)]:
    pool = np.where(mask)[0]
    take = rng.choice(pool, size=min(8000, len(pool)), replace=False)
    t = time.time()
    p = permutation_importance(rf, np.asarray(X_test[take].todense()), y_test[take],
                               n_repeats=3, random_state=SEED, scoring="r2", n_jobs=-1)
    d = pd.DataFrame({
        "feature": FEATURES,
        "perm_mean": p.importances_mean,
        "perm_std": p.importances_std,
    }).sort_values("perm_mean", ascending=False).reset_index(drop=True)
    d["original"] = d["feature"].map(original_feature)
    seg_imp[label] = d
    print(f"\n{label} (n={len(take):,}, {time.time() - t:.1f}s) — top 10:")
    print(d.head(10).to_string(index=False))

seg_table = pd.DataFrame({
    "feature": seg_imp["Economy"]["feature"],
    "Economy": seg_imp["Economy"]["perm_mean"].round(5),
    "Business": seg_imp["Business"].set_index("feature").loc[
        seg_imp["Economy"]["feature"], "perm_mean"].values.round(5),
}).sort_values("Economy", ascending=False)
seg_table["original"] = seg_table["feature"].map(original_feature)
seg_table.to_csv(REP / "05_importance_by_segment.csv", index=False)

# Agregasi segmen per kolom asli
seg_agg = pd.DataFrame({
    "Economy": seg_imp["Economy"].groupby("original")["perm_mean"].sum(),
    "Business": seg_imp["Business"].groupby("original")["perm_mean"].sum(),
}).sort_values("Economy", ascending=False)
print("\nAgregasi per kolom asli — Economy vs Business:")
print((seg_agg * 100).round(3).to_string())

# ======================================================================
section("5. PARTIAL DEPENDENCE — days_left & duration PER KELAS")
days_left_idx = FEATURES.index("days_left")
duration_idx = FEATURES.index("duration")

# subsample dense untuk PDP
pd_sets = {}
for label, mask in [("Economy", is_eco), ("Business", ~is_eco)]:
    pool = np.where(mask)[0]
    take = rng.choice(pool, size=min(4000, len(pool)), replace=False)
    pd_sets[label] = np.asarray(X_test[take].todense())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (label, Xs) in zip(axes, pd_sets.items()):
    PartialDependenceDisplay.from_estimator(
        rf, Xs, [days_left_idx], kind="average", grid_resolution=49,
        ax=ax, line_kw={"lw": 2.2, "color": "#C73E1D" if label == "Business" else "#2E86AB"},
        n_jobs=-1, random_state=SEED,
    )
    ax.set_title(f"Partial dependence: days_left -> price ({label})")
    ax.set_xlabel("days_left (hari sebelum keberangkatan)")
    ax.set_ylabel("Harga prediksi (INR)")
    ax.invert_xaxis()
plt.tight_layout()
plt.savefig(FIG / "interp_04_pdp_days_left.png")
plt.close()
print("-> figures/interp_04_pdp_days_left.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (label, Xs) in zip(axes, pd_sets.items()):
    PartialDependenceDisplay.from_estimator(
        rf, Xs, [duration_idx], kind="average", grid_resolution=60,
        ax=ax, line_kw={"lw": 2.2, "color": "#C73E1D" if label == "Business" else "#2E86AB"},
        n_jobs=-1, random_state=SEED,
    )
    ax.set_title(f"Partial dependence: duration -> price ({label})")
    ax.set_xlabel("Durasi penerbangan (jam)")
    ax.set_ylabel("Harga prediksi (INR)")
plt.tight_layout()
plt.savefig(FIG / "interp_05_pdp_duration.png")
plt.close()
print("-> figures/interp_05_pdp_duration.png")

# Angka PDP untuk laporan
pd_numbers = {}
for label, Xs in pd_sets.items():
    grid = np.unique(np.percentile(Xs[:, days_left_idx], np.linspace(1, 99, 25)))
    grid = np.round(grid).astype(int)
    rows = []
    for g in grid:
        Xm = Xs.copy()
        Xm[:, days_left_idx] = g
        rows.append({"days_left": int(g), "pred": float(rf.predict(Xm).mean())})
    d = pd.DataFrame(rows)
    pd_numbers[label] = d.set_index("days_left")["pred"].round(0).to_dict()
    print(f"\nPDP days_left — {label} (harga prediksi rata-rata INR):")
    print(d.assign(pred=d["pred"].round(0).astype(int)).to_string(index=False))

# ======================================================================
section("6. VISUALISASI IMPORTANCE")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]
top = gini_agg.head(10).iloc[::-1]
ax.barh(top["original"], top["pct"], color="#4C72B0")
for i, v in enumerate(top["pct"]):
    ax.text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=9)
ax.set_title("Gini importance (agregasi per kolom asli)")
ax.set_xlabel("Kontribusi terhadap penurunan impurity (%)")
ax.set_xlim(0, top["pct"].max() * 1.18)

ax = axes[1]
top2 = perm_agg.head(10).iloc[::-1]
ax.barh(top2["original"], top2["drop_r2_pp"], color="#C73E1D")
for i, v in enumerate(top2["drop_r2_pp"]):
    ax.text(v + max(top2["drop_r2_pp"]) * 0.02, i, f"{v:.2f}", va="center", fontsize=9)
ax.set_title("Permutation importance (penurunan R2, poin persentase)")
ax.set_xlabel("Penurunan R2 saat fitur diacak (pp)")

plt.tight_layout()
plt.savefig(FIG / "interp_01_gini.png")
plt.close()
print("-> figures/interp_01_gini.png")

fig, ax = plt.subplots(figsize=(9, 7))
sub = perm_df.head(15).iloc[::-1]
ax.barh(sub["feature"], sub["perm_mean"] * 100,
        xerr=sub["perm_std"] * 100, color="#2E7D32", alpha=0.85)
ax.set_title("Permutation importance top 15 (level kolom one-hot)")
ax.set_xlabel("Penurunan R2 (poin persentase)")

plt.tight_layout()
plt.savefig(FIG / "interp_02_permutation.png")
plt.close()
print("-> figures/interp_02_permutation.png")

fig, ax = plt.subplots(figsize=(9, 6))
sagg = (seg_agg * 100)
sagg = sagg.loc[sagg.abs().max(axis=1).sort_values(ascending=False).index[:10]].iloc[::-1]
ypos = np.arange(len(sagg))
ax.barh(ypos - 0.2, sagg["Economy"], height=0.4, label="Economy", color="#2E86AB")
ax.barh(ypos + 0.2, sagg["Business"], height=0.4, label="Business", color="#C73E1D")
ax.set_yticks(ypos)
ax.set_yticklabels(sagg.index)
ax.set_title("Permutation importance per segmen (penurunan R2, pp)")
ax.set_xlabel("Penurunan R2 (poin persentase)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "interp_03_segment.png")
plt.close()
print("-> figures/interp_03_segment.png")

# ======================================================================
section("7. SIMPAN RINGKASAN")
summary = {
    "gini_top10": gini_agg.head(10).to_dict("records"),
    "permutation_top10": perm_agg.head(10).to_dict("records"),
    "segment_top10": (seg_agg * 100).round(3).head(10).reset_index().to_dict("records"),
    "pdp_days_left": pd_numbers,
    "n_permutation_samples": int(len(idx)),
    "n_repeats": 5,
}
with open(REP / "05_interpretation.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
print("-> reports/05_interpretation.json")

section(f"SELESAI dalam {(time.time() - T0) / 60:.1f} menit")
