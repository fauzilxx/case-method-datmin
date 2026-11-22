"""
Pra-pemrosesan Data (Data Preprocessing)
Studi Kasus : Prediksi Harga Tiket Pesawat dengan Random Forest Regression

Tahapan preprocessing:
  - 7 fitur kategorikal diubah menggunakan One-Hot Encoding (OHE)
  - Fitur numerik (duration, days_left) dipertahankan tanpa scaling (scale-invariant)
  - Fit preprocessor hanya pada data latih untuk mencegah data leakage
  - Pembagian data train-test 80:20 secara terstratifikasi berdasarkan class

Output:
  - data/processed/X_train.npz, X_test.npz, y_train.npy, y_test.npy
  - models/preprocessor.joblib
  - reports/03_preprocessing_schema.json
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path("E:/data-mining-flight-price")
DATA = BASE / "data" / "processed" / "flight_clean.csv"
OUT = BASE / "data" / "processed"
MODELS = BASE / "models"
MODELS.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET = "price"

CAT_FEATURES = ["airline", "source_city", "departure_time", "stops",
                "arrival_time", "destination_city", "class"]
NUM_FEATURES = ["duration", "days_left"]


def section(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


t0 = time.time()

# ======================================================================
section("1. LOAD & DEFINISI FITUR")
df = pd.read_csv(DATA)
print(f"Dataset: {df.shape[0]:,} baris x {df.shape[1]} kolom")

assert set(CAT_FEATURES + NUM_FEATURES + [TARGET]) == set(df.columns), "Kolom tidak cocok"
print(f"Fitur kategorikal ({len(CAT_FEATURES)}): {CAT_FEATURES}")
print(f"Fitur numerik     ({len(NUM_FEATURES)}): {NUM_FEATURES}")
print(f"Target            : {TARGET}")

X = df[CAT_FEATURES + NUM_FEATURES]
y = df[TARGET]

# ======================================================================
section("2. SPLIT 80/20 (STRATIFIED PER CLASS)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=df["class"]
)
print(f"Train: {X_train.shape[0]:,} baris")
print(f"Test : {X_test.shape[0]:,} baris")
print(f"Rasio: {X_train.shape[0] / len(df):.1%} / {X_test.shape[0] / len(df):.1%}")

print("\nDistribusi class — TRAIN:")
print((X_train["class"].value_counts(normalize=True) * 100).round(2).to_string())
print("Distribusi class — TEST:")
print((X_test["class"].value_counts(normalize=True) * 100).round(2).to_string())

print("\nStatistik target — TRAIN vs TEST:")
cmp = pd.DataFrame({
    "train": y_train.describe(),
    "test": y_test.describe(),
}).round(1)
print(cmp.to_string())

# ======================================================================
section("3. FIT PREPROCESSOR (HANYA PADA TRAIN)")
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first",
                              sparse_output=True, dtype=np.float64), CAT_FEATURES),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

preprocessor.fit(X_train)
print("Preprocessor selesai di-fit pada data train.")

cat_encoder = preprocessor.named_transformers_["cat"]
cat_names = cat_encoder.get_feature_names_out(CAT_FEATURES).tolist()
feature_names = NUM_FEATURES + cat_names

print(f"\nJumlah fitur setelah One-Hot Encoding: {len(feature_names)}")
print(f"  - numerik passthrough : {len(NUM_FEATURES)}")
print(f"  - hasil one-hot       : {len(cat_names)}")
print("\nDaftar fitur hasil encoding:")
for i, n in enumerate(feature_names):
    print(f"  [{i:2d}] {n}")

# ======================================================================
section("4. TRANSFORM")
X_train_t = preprocessor.transform(X_train)
X_test_t = preprocessor.transform(X_test)

if not sp.issparse(X_train_t):
    X_train_t = sp.csr_matrix(X_train_t)
    X_test_t = sp.csr_matrix(X_test_t)
else:
    X_train_t = X_train_t.tocsr()
    X_test_t = X_test_t.tocsr()

print(f"X_train: {X_train_t.shape}  sparse={sp.issparse(X_train_t)}  "
      f"nnz={X_train_t.nnz:,}  density={X_train_t.nnz / (X_train_t.shape[0] * X_train_t.shape[1]):.3%}")
print(f"X_test : {X_test_t.shape}")
print(f"Dtype  : {X_train_t.dtype}")

# Sanity: pastikan tidak ada NaN / inf
def check(name, M):
    data = M.data if sp.issparse(M) else M
    n_nan = int(np.isnan(data).sum())
    n_inf = int(np.isinf(data).sum())
    print(f"  {name}: NaN={n_nan}  Inf={n_inf}")
    return n_nan, n_inf

nan_inf = {"train": check("X_train", X_train_t), "test": check("X_test", X_test_t)}

# ======================================================================
section("5. VERIFIKASI KONSISTENSI FITUR: train vs test")
print(f"Jumlah kolom train == test ? {X_train_t.shape[1] == X_test_t.shape[1]}")
# Cek apakah ada kolom kategori di test yang tidak dikenal train
unseen = set(X_test["airline"].unique()) - set(X_train["airline"].unique())
unseen2 = set(X_test["source_city"].unique()) - set(X_train["source_city"].unique())
print(f"Kategori tak dikenal (airline)     : {unseen if unseen else 'tidak ada'}")
print(f"Kategori tak dikenal (source_city) : {unseen2 if unseen2 else 'tidak ada'}")

# ======================================================================
section("6. SIMPAN ARTIFACT")
sp.save_npz(OUT / "X_train.npz", X_train_t)
sp.save_npz(OUT / "X_test.npz", X_test_t)
np.save(OUT / "y_train.npy", y_train.to_numpy())
np.save(OUT / "y_test.npy", y_test.to_numpy())
joblib.dump(preprocessor, MODELS / "preprocessor.joblib")

class_mask = [not name.startswith("class_") for name in feature_names]
n_class_cols = len(feature_names) - sum(class_mask)

schema = {
    "seed": SEED,
    "n_train": int(X_train_t.shape[0]),
    "n_test": int(X_test_t.shape[0]),
    "n_features": len(feature_names),
    "feature_names": feature_names,
    "cat_features": CAT_FEATURES,
    "num_features": NUM_FEATURES,
    "target": TARGET,
    "class_onehot_columns": [n for n in feature_names if n.startswith("class_")],
    "features_without_class": [n for n, m in zip(feature_names, class_mask) if m],
    "n_features_without_class": int(sum(class_mask)),
    "scaling": "none (tree-based model)",
    "encoding": "OneHotEncoder(drop='first', handle_unknown='ignore')",
    "stratified_on": "class",
}
with open(BASE / "reports" / "03_preprocessing_schema.json", "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2, ensure_ascii=False)

print(f"X_train.npz / X_test.npz  -> data/processed/")
print(f"y_train.npy / y_test.npy  -> data/processed/")
print(f"preprocessor.joblib       -> models/")
print(f"schema                    -> reports/03_preprocessing_schema.json")
print(f"\nKolom one-hot untuk `class`: {schema['class_onehot_columns']} ({n_class_cols} kolom)")
print(f"Jumlah fitur untuk varian TANPA class: {schema['n_features_without_class']}")

section(f"SELESAI dalam {time.time() - t0:.1f} detik")
