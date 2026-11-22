"""
Pembersihan Data (Data Cleaning & Audit)
Studi Kasus : Prediksi Harga Tiket Pesawat dengan Random Forest Regression
Dataset     : Flight Price Prediction (EaseMyTrip)

Tahapan pembersihan:
  1. Hapus kolom indeks bawaan 'Unnamed: 0'
  2. Hapus nomor penerbangan 'flight' (mencegah data leakage)
  3. Hapus baris duplikat penuh
  4. Validasi nilai target (price > 0)
  5. Validasi tipe data dan konsistensi kategori

Output:
  - data/processed/flight_clean.csv
  - reports/cleaning_log.json
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("E:/data-mining-flight-price")
RAW = BASE / "data" / "raw" / "Clean_Dataset.csv"
OUT_CSV = BASE / "data" / "processed" / "flight_clean.csv"

log = {}


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------
section("1. LOAD DATASET")
df = pd.read_csv(RAW)
log["rows_raw"] = int(df.shape[0])
log["cols_raw"] = int(df.shape[1])
print(f"Shape awal : {df.shape[0]:,} baris x {df.shape[1]} kolom")
print(f"Kolom      : {list(df.columns)}")

# ----------------------------------------------------------------------
# 2. AUDIT TIPE DATA & MISSING VALUE
# ----------------------------------------------------------------------
section("2. AUDIT TIPE DATA & MISSING VALUE")
audit = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "non_null": df.notna().sum(),
    "null": df.isna().sum(),
    "null_pct": (df.isna().mean() * 100).round(3),
    "n_unique": df.nunique(),
})
print(audit.to_string())
log["missing_total"] = int(df.isna().sum().sum())
log["audit_table"] = audit.reset_index().rename(columns={"index": "column"}).to_dict("records")

# ----------------------------------------------------------------------
# 3. VALIDASI NILAI KATEGORI
# ----------------------------------------------------------------------
section("3. VALIDASI NILAI KATEGORI")
cat_cols = ["airline", "source_city", "departure_time", "stops",
            "arrival_time", "destination_city", "class"]
cat_values = {}
for c in cat_cols:
    vals = sorted(df[c].dropna().astype(str).unique().tolist())
    cat_values[c] = vals
    print(f"{c:20s} ({len(vals)}): {vals}")
log["categorical_values"] = cat_values

# ----------------------------------------------------------------------
# 4. DISTRIBUSI TARGET SEBELUM CLEANING
# ----------------------------------------------------------------------
section("4. DISTRIBUSI TARGET 'price' (SEBELUM CLEANING)")
price_stats_before = df["price"].describe().to_dict()
print(df["price"].describe().to_string())
log["price_stats_before"] = {k: float(v) for k, v in price_stats_before.items()}

n_price_le0 = int((df["price"] <= 0).sum())
n_price_null = int(df["price"].isna().sum())
print(f"\nprice <= 0 : {n_price_le0}")
print(f"price null : {n_price_null}")
log["n_price_le0"] = n_price_le0
log["n_price_null"] = n_price_null

print("\nHarga rata-rata per kelas kabin:")
per_class = df.groupby("class")["price"].agg(["count", "mean", "median", "min", "max"]).round(2)
print(per_class.to_string())
log["price_by_class"] = {
    str(idx): {k: float(v) for k, v in row.items()}
    for idx, row in per_class.iterrows()
}

# ----------------------------------------------------------------------
# 5. CLEANING - STEP 1: DROP KOLOM INDEX & ID
# ----------------------------------------------------------------------
section("5. CLEANING — DROP KOLOM INDEX & ID")
dropped = []
for col in df.columns:
    if col.startswith("Unnamed") or col.strip() == "":
        dropped.append(col)
df = df.drop(columns=dropped, errors="ignore")
print(f"Kolom index dibuang : {dropped}")

if "flight" in df.columns:
    df = df.drop(columns=["flight"])
    print("Kolom 'flight' dibuang (ID unik -> leakage)")
    dropped.append("flight")
log["dropped_columns"] = dropped

# ----------------------------------------------------------------------
# 6. CLEANING - STEP 2: HAPUS DUPLIKAT
# ----------------------------------------------------------------------
section("6. CLEANING — HAPUS DUPLIKAT BARIS PENUH")
n_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
n_dup = n_before - len(df)
print(f"Duplikat dihapus : {n_dup:,}  ({n_dup / n_before * 100:.2f}%)")
log["duplicates_removed"] = int(n_dup)

# ----------------------------------------------------------------------
# 7. CLEANING - STEP 3: TARGET TIDAK VALID
# ----------------------------------------------------------------------
section("7. CLEANING — TARGET TIDAK VALID")
n_before = len(df)
df = df[df["price"].notna()]
df = df[df["price"] > 0]
n_invalid = n_before - len(df)
print(f"Baris price tidak valid dihapus : {n_invalid}")
log["invalid_price_removed"] = int(n_invalid)

# ----------------------------------------------------------------------
# 8. CLEANING - STEP 4: MISSING VALUE PADA FITUR
# ----------------------------------------------------------------------
section("8. CLEANING — MISSING VALUE PADA FITUR")
n_before = len(df)
missing_before = int(df.isna().sum().sum())
# Semua kolom fitur wajib terisi; dataset bersih ini seharusnya 0
df = df.dropna(subset=[c for c in df.columns])
n_null_rows = n_before - len(df)
print(f"Total missing sebelum : {missing_before}")
print(f"Baris dihapus karena missing : {n_null_rows}")
log["missing_value_rows_removed"] = int(n_null_rows)

# ----------------------------------------------------------------------
# 9. CLEANING - STEP 5: PASTIKAN TIPE NUMERIK BENAR
# ----------------------------------------------------------------------
section("9. CLEANING — PASTIKAN TIPE NUMERIK")
for c in ["duration", "days_left", "price"]:
    if df[c].dtype == object:
        print(f"Kolom '{c}' bertipe object -> dikonversi ke numerik")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    else:
        print(f"Kolom '{c}' sudah numerik ({df[c].dtype})")

n_before = len(df)
df = df.dropna(subset=["duration", "days_left", "price"])
n_coerce = n_before - len(df)
if n_coerce:
    print(f"Baris tidak valid setelah konversi numerik dihapus : {n_coerce}")
log["numeric_coercion_rows_removed"] = int(n_coerce)

df["days_left"] = df["days_left"].astype(int)

# ----------------------------------------------------------------------
# 10. OUTLIER AUDIT (TIDAK DIHAPUS - hanya didokumentasikan)
# ----------------------------------------------------------------------
section("10. AUDIT OUTLIER 'price' (IQR) — TIDAK DIHAPUS")
q1, q3 = df["price"].quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_out = int(((df["price"] < lo) | (df["price"] > hi)).sum())
print(f"Q1={q1:.0f}  Q3={q3:.0f}  IQR={iqr:.0f}")
print(f"Batas bawah={lo:.0f}  batas atas={hi:.0f}")
print(f"Jumlah outlier : {n_out:,} ({n_out / len(df) * 100:.2f}%)")
print("-> DIPERTAHANKAN: harga mahal kelas Business adalah sinyal nyata, bukan noise.")
log["outliers_price_iqr"] = {"q1": float(q1), "q3": float(q3), "iqr": float(iqr),
                              "lower": float(lo), "upper": float(hi),
                              "count": n_out, "pct": round(n_out / len(df) * 100, 3),
                              "action": "retained"}

# ----------------------------------------------------------------------
# 11. HASIL AKHIR
# ----------------------------------------------------------------------
section("11. HASIL AKHIR CLEANING")
log["rows_final"] = int(df.shape[0])
log["cols_final"] = int(df.shape[1])
log["columns_final"] = list(df.columns)
log["price_stats_after"] = {k: float(v) for k, v in df["price"].describe().to_dict().items()}

print(f"Shape akhir : {df.shape[0]:,} baris x {df.shape[1]} kolom")
print(f"Baris terbuang total : {log['rows_raw'] - log['rows_final']:,} "
      f"({(log['rows_raw'] - log['rows_final']) / log['rows_raw'] * 100:.3f}%)")
print(f"\nKolom final : {list(df.columns)}")
print("\nStatistik target setelah cleaning:")
print(df["price"].describe().to_string())

print("\nSebaran kelas kabin:")
print(df["class"].value_counts().to_string())

print("\n5 baris pertama hasil cleaning:")
print(df.head().to_string())

# ----------------------------------------------------------------------
# 12. SIMPAN
# ----------------------------------------------------------------------
section("12. SIMPAN OUTPUT")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
size_mb = os.path.getsize(OUT_CSV) / (1024 * 1024)
print(f"CSV bersih disimpan : {OUT_CSV}  ({size_mb:.2f} MB)")

with open(BASE / "reports" / "cleaning_log.json", "w", encoding="utf-8") as f:
    json.dump(log, f, indent=2, ensure_ascii=False)
print(f"Log JSON disimpan   : {BASE / 'reports' / 'cleaning_log.json'}")

print("\nSELESAI.")
