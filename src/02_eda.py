"""
Analisis Eksplorasi Data (Exploratory Data Analysis / EDA)
Studi Kasus : Prediksi Harga Tiket Pesawat dengan Random Forest Regression

Visualisasi dan analisis statistik:
  1. Distribusi variabel target price (sebaran bimodal)
  2. Perbandingan harga tiket antara kelas Ekonomi dan Bisnis
  3. Pola non-linear sisa hari pemesanan (days_left) terhadap harga
  4. Perbandingan sebaran harga antaran maskapai
  5. Pengaruh jumlah transit (stops) terhadap harga
  6. Korelasi antar fitur numerik

Output:
  - figures/eda_01_distribusi_price.png
  - figures/eda_02_price_by_class.png
  - figures/eda_03_days_left_vs_price.png
  - figures/eda_04_price_by_airline.png
  - figures/eda_05_price_by_stops.png
  - figures/eda_06_correlation.png
  - reports/eda_stats.json
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend headless: wajib untuk simpan file di Windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE = Path("E:/data-mining-flight-price")
DATA = BASE / "data" / "processed" / "flight_clean.csv"
FIG = BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 10

stats = {}


def section(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


# ======================================================================
section("0. LOAD")
df = pd.read_csv(DATA)
print(f"Shape: {df.shape[0]:,} baris x {df.shape[1]} kolom")
stats["shape"] = list(df.shape)

PALETTE_CLASS = {"Economy": "#2E86AB", "Business": "#C73E1D"}


# ======================================================================
# Q1 - DISTRIBUSI TARGET
# ======================================================================
section("Q1. DISTRIBUSI TARGET price")

skew = float(df["price"].skew())
kurt = float(df["price"].kurt())
print(f"Skewness : {skew:.3f}")
print(f"Kurtosis : {kurt:.3f}")
stats["skew"] = round(skew, 4)
stats["kurtosis"] = round(kurt, 4)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

ax = axes[0]
ax.hist(df["price"], bins=80, color="#4C72B0", edgecolor="white", linewidth=0.3)
ax.axvline(df["price"].mean(), color="crimson", ls="--", lw=1.6,
           label=f"Mean = {df['price'].mean():,.0f}")
ax.axvline(df["price"].median(), color="darkgreen", ls="-.", lw=1.6,
           label=f"Median = {df['price'].median():,.0f}")
ax.set_title("Q1a. Distribusi price (seluruh data)")
ax.set_xlabel("Harga tiket (INR)")
ax.set_ylabel("Frekuensi")
ax.legend(fontsize=9)

ax = axes[1]
for cls, color in PALETTE_CLASS.items():
    sub = df.loc[df["class"] == cls, "price"]
    ax.hist(sub, bins=70, alpha=0.72, color=color, label=f"{cls} (n={len(sub):,})",
            edgecolor="white", linewidth=0.2)
ax.set_title("Q1b. Distribusi price dipisah per kelas kabin")
ax.set_xlabel("Harga tiket (INR)")
ax.set_ylabel("Frekuensi")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(FIG / "eda_01_distribusi_price.png")
plt.close()
print("-> figures/eda_01_distribusi_price.png")

# Porsi harga di bawah median overall, per kelas
below = df["price"] < df["price"].median()
stats["pct_below_overall_median"] = {
    cls: round(float(below[df["class"] == cls].mean() * 100), 2)
    for cls in ["Economy", "Business"]
}
print("Porsi harga < median overall, per kelas:", stats["pct_below_overall_median"])


# ======================================================================
# Q2 - JURANG HARGA ANTAR KELAS
# ======================================================================
section("Q2. JURANG HARGA ANTAR KELAS KABIN")

desc = df.groupby("class")["price"].describe().round(1)
print(desc.to_string())
stats["price_by_class"] = {
    str(i): {k: float(v) for k, v in row.items()} for i, row in desc.iterrows()
}
ratio = float(desc.loc["Business", "mean"] / desc.loc["Economy", "mean"])
stats["ratio_business_economy"] = round(ratio, 3)
print(f"\nRasio mean Business : Economy = {ratio:.2f} : 1")

# Overlap check: berapa persen tiket Business yang lebih murah dari Business termahal Economy?
eco_max = df.loc[df["class"] == "Economy", "price"].max()
biz_min = df.loc[df["class"] == "Business", "price"].min()
n_overlap = int(((df["class"] == "Business") & (df["price"] <= eco_max)).sum())
print(f"Economy max = {eco_max:,.0f} | Business min = {biz_min:,.0f}")
print(f"Tiket Business yang harganya <= Economy termahal : {n_overlap:,} "
      f"({n_overlap / (df['class'] == 'Business').sum() * 100:.2f}% dari Business)")
stats["economy_max"] = float(eco_max)
stats["business_min"] = float(biz_min)
stats["business_overlap_count"] = n_overlap

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

ax = axes[0]
sns.boxplot(data=df, x="class", y="price", hue="class", palette=PALETTE_CLASS,
            legend=False, ax=ax, width=0.5, fliersize=2)
ax.set_title("Q2a. Sebaran harga per kelas kabin")
ax.set_xlabel("Kelas kabin")
ax.set_ylabel("Harga tiket (INR)")

ax = axes[1]
sns.violinplot(data=df, x="class", y="price", hue="class", palette=PALETTE_CLASS,
               legend=False, ax=ax, inner="quartile", cut=0)
ax.set_title("Q2b. Bentuk distribusi (violin) per kelas")
ax.set_xlabel("Kelas kabin")
ax.set_ylabel("Harga tiket (INR)")

plt.tight_layout()
plt.savefig(FIG / "eda_02_price_by_class.png")
plt.close()
print("-> figures/eda_02_price_by_class.png")


# ======================================================================
# Q3 - days_left VS price (UJI LINEARITAS)
# ======================================================================
section("Q3. HUBUNGAN days_left -> price")

agg = df.groupby("days_left")["price"].agg(["mean", "median", "count"]).round(1)
agg.columns = ["mean_price", "median_price", "n"]

# Korelasi linear mentah
r_raw = float(df["days_left"].corr(df["price"]))
print(f"Korelasi Pearson days_left vs price (mentah) : {r_raw:.4f}")
stats["pearson_days_left_price"] = round(r_raw, 4)

# Rata-rata per kelas
agg_cls = df.groupby(["days_left", "class"])["price"].mean().unstack().round(1)
print("\nRata-rata harga per days_left (per kelas) - sampel H1..H10 dan H45..H49:")
print(pd.concat([agg_cls.head(10), agg_cls.tail(5)]).to_string())

peak_day = int(agg["mean_price"].idxmax())
peak_val = float(agg["mean_price"].max())
trough_day = int(agg["mean_price"].idxmin())
trough_val = float(agg["mean_price"].min())
print(f"\nPuncak harga rata-rata : H-{peak_day} = {peak_val:,.0f} INR")
print(f"Terendah harga rata-rata: H-{trough_day} = {trough_val:,.0f} INR")
print(f"Amplitudo (puncak-terendah): {peak_val - trough_val:,.0f} INR "
      f"({(peak_val / trough_val - 1) * 100:.1f}%)")
stats["peak_days_left"] = peak_day
stats["peak_mean_price"] = round(peak_val, 1)
stats["trough_days_left"] = trough_day
stats["trough_mean_price"] = round(trough_val, 1)
stats["amplitude_pct"] = round((peak_val / trough_val - 1) * 100, 2)

# Median per kelas untuk lihat tren murni
agg_cls_median = df.groupby(["days_left", "class"])["price"].median().unstack()
stats["days_left_table"] = agg.reset_index().to_dict("records")

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

ax = axes[0]
ax.plot(agg.index, agg["mean_price"], marker="o", ms=4, lw=1.8,
        color="#C73E1D", label="Rata-rata")
ax.plot(agg.index, agg["median_price"], marker="s", ms=3.5, lw=1.6,
        color="#2E86AB", label="Median")
ax.axvline(peak_day, color="gray", ls=":", lw=1.2)
ax.annotate(f"Puncak H-{peak_day}\n{peak_val:,.0f}",
            xy=(peak_day, peak_val), xytext=(peak_day + 6, peak_val * 1.03),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="gray", lw=1))
ax.set_title("Q3a. Harga vs sisa hari pemesanan (seluruh data)")
ax.set_xlabel("days_left (hari sebelum keberangkatan)")
ax.set_ylabel("Harga tiket (INR)")
ax.legend(fontsize=9)
ax.invert_xaxis()

ax = axes[1]
for cls, color in PALETTE_CLASS.items():
    ax.plot(agg_cls_median.index, agg_cls_median[cls], marker="o", ms=3,
            lw=1.6, color=color, label=cls)
ax.set_title("Q3b. Median harga vs days_left, per kelas kabin")
ax.set_xlabel("days_left (hari sebelum keberangkatan)")
ax.set_ylabel("Median harga (INR)")
ax.legend(fontsize=9)
ax.invert_xaxis()

plt.tight_layout()
plt.savefig(FIG / "eda_03_days_left_vs_price.png")
plt.close()
print("-> figures/eda_03_days_left_vs_price.png")

# Uji korelasi di dalam masing-masing kelas (menghapus efek kelas)
r_eco = float(df.loc[df["class"] == "Economy", "days_left"].corr(
    df.loc[df["class"] == "Economy", "price"]))
r_biz = float(df.loc[df["class"] == "Business", "days_left"].corr(
    df.loc[df["class"] == "Business", "price"]))
print(f"\nKorelasi Pearson days_left-price di DALAM Economy : {r_eco:.4f}")
print(f"Korelasi Pearson days_left-price di DALAM Business: {r_biz:.4f}")
stats["pearson_within_economy"] = round(r_eco, 4)
stats["pearson_within_business"] = round(r_biz, 4)


# ======================================================================
# Q4 - HARGA PER MASKAPAI
# ======================================================================
section("Q4. HARGA PER MASKAPAI")

air = (df.groupby("airline")["price"]
         .agg(["count", "mean", "median", "std", "min", "max"])
         .round(1)
         .sort_values("median", ascending=False))
print(air.to_string())
stats["price_by_airline"] = {
    str(i): {k: float(v) for k, v in row.items()} for i, row in air.iterrows()
}

# Maskapai mana yang konsisten mahal DI DALAM segmen Economy saja
air_eco = (df[df["class"] == "Economy"].groupby("airline")["price"]
             .agg(["count", "median"]).round(1).sort_values("median", ascending=False))
print("\nMaskapai termahal di segmen Economy saja (median INR):")
print(air_eco.to_string())
stats["price_by_airline_economy"] = {
    str(i): {k: float(v) for k, v in row.items()} for i, row in air_eco.iterrows()
}

order = air.index.tolist()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
sns.boxplot(data=df, x="airline", y="price", order=order, hue="airline",
            palette="crest", legend=False, ax=ax, width=0.6, fliersize=2)
ax.set_title("Q4a. Sebaran harga per maskapai")
ax.set_xlabel("Maskapai")
ax.set_ylabel("Harga tiket (INR)")
ax.tick_params(axis="x", rotation=20)

ax = axes[1]
sns.boxplot(data=df[df["class"] == "Economy"], x="airline", y="price",
            hue="airline", palette="flare",
            order=air_eco.index.tolist(), legend=False, ax=ax, width=0.6, fliersize=2)
ax.set_title("Q4b. Khusus Economy - perbandingan adil antar maskapai")
ax.set_xlabel("Maskapai")
ax.set_ylabel("Harga tiket (INR)")
ax.tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(FIG / "eda_04_price_by_airline.png")
plt.close()
print("-> figures/eda_04_price_by_airline.png")


# ======================================================================
# Q5 - PENGARUH TRANSIT
# ======================================================================
section("Q5. PENGARUH JUMLAH TRANSIT")

stops_order = ["zero", "one", "two_or_more"]
st = (df.groupby("stops")["price"]
        .agg(["count", "mean", "median"]).round(1).reindex(stops_order))
print(st.to_string())
stats["price_by_stops"] = {
    str(i): {k: float(v) for k, v in row.items()} for i, row in st.iterrows()
}

st_eco = (df[df["class"] == "Economy"].groupby("stops")["price"]
            .agg(["count", "mean", "median"]).round(1).reindex(stops_order))
print("\nKhusus Economy:")
print(st_eco.to_string())
stats["price_by_stops_economy"] = {
    str(i): {k: float(v) for k, v in row.items()} for i, row in st_eco.iterrows()
}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

ax = axes[0]
sns.barplot(data=st.reset_index(), x="stops", y="mean", order=stops_order,
            hue="stops", palette="mako", legend=False, ax=ax)
for i, v in enumerate(st["mean"]):
    ax.text(i, v * 1.01, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
ax.set_title("Q5a. Rata-rata harga per jumlah transit")
ax.set_xlabel("Jumlah transit")
ax.set_ylabel("Rata-rata harga (INR)")

ax = axes[1]
sns.boxplot(data=df[df["class"] == "Economy"], x="stops", y="price",
            order=stops_order, hue="stops", palette="mako", legend=False,
            ax=ax, width=0.55, fliersize=2)
ax.set_title("Q5b. Sebaran harga per transit (khusus Economy)")
ax.set_xlabel("Jumlah transit")
ax.set_ylabel("Harga tiket (INR)")

plt.tight_layout()
plt.savefig(FIG / "eda_05_price_by_stops.png")
plt.close()
print("-> figures/eda_05_price_by_stops.png")


# ======================================================================
# Q6 - KORELASI FITUR NUMERIK
# ======================================================================
section("Q6. KORELASI FITUR NUMERIK")

corr_df = df[["duration", "days_left", "price"]].copy()
corr_df["class_bin"] = (df["class"] == "Business").astype(int)
corr = corr_df.corr().round(4)
print(corr.to_string())
stats["correlation"] = corr.to_dict()

print("\nKorelasi duration vs price (keseluruhan) :",
      f"{corr.loc['duration', 'price']:.4f}")
print("Korelasi class_bin vs price             :",
      f"{corr.loc['class_bin', 'price']:.4f}")
print("Korelasi duration vs price (Economy saja):",
      f"{df.loc[df['class']=='Economy', 'duration'].corr(df.loc[df['class']=='Economy','price']):.4f}")
print("Korelasi duration vs price (Business saja):",
      f"{df.loc[df['class']=='Business','duration'].corr(df.loc[df['class']=='Business','price']):.4f}")
stats["pearson_duration_economy"] = round(float(
    df.loc[df['class'] == 'Economy', 'duration'].corr(df.loc[df['class'] == 'Economy', 'price'])), 4)
stats["pearson_duration_business"] = round(float(
    df.loc[df['class'] == 'Business', 'duration'].corr(df.loc[df['class'] == 'Business', 'price'])), 4)

fig, ax = plt.subplots(figsize=(6.2, 5))
labels = ["duration", "days_left", "class_bin", "price"]
sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.6, cbar_kws={"shrink": 0.8}, ax=ax,
            xticklabels=["duration", "days_left", "class (B=1)", "price"],
            yticklabels=["duration", "days_left", "class (B=1)", "price"])
ax.set_title("Q6. Matriks korelasi fitur numerik + kelas (binary)")
plt.tight_layout()
plt.savefig(FIG / "eda_06_correlation.png")
plt.close()
print("-> figures/eda_06_correlation.png")


# ======================================================================
# TAMBAHAN UNTUK LAPORAN: rute & waktu
# ======================================================================
section("TAMBAHAN. RUTE & WAKTU KEBERANGKATAN")

df["route"] = df["source_city"] + " -> " + df["destination_city"]
route_stats = (df.groupby("route")["price"].agg(["count", "median"])
                 .round(1).sort_values("median", ascending=False))
print(f"Jumlah rute unik: {df['route'].nunique()}")
print("Top 5 rute termahal (median):")
print(route_stats.head(5).to_string())
print("\nBottom 5 rute termurah (median):")
print(route_stats.tail(5).to_string())
stats["route_top5"] = route_stats.head(5).reset_index().to_dict("records")
stats["route_bottom5"] = route_stats.tail(5).reset_index().to_dict("records")

dep = (df.groupby("departure_time")["price"].agg(["count", "median"])
         .round(1).sort_values("median", ascending=False))
print("\nMedian harga per waktu keberangkatan:")
print(dep.to_string())
stats["price_by_departure"] = {
    str(i): {k: float(v) for k, v in row.items()} for i, row in dep.iterrows()
}

# Simpan stats
with open(BASE / "reports" / "eda_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print("\nStats disimpan: reports/eda_stats.json")

# Simpan tabel days_left untuk lampiran
agg.to_csv(BASE / "reports" / "table_days_left_price.csv")
print("Tabel days_left disimpan: reports/table_days_left_price.csv")

section("SELESAI")
print(f"Total grafik tersimpan: {len(list(FIG.glob('eda_*.png')))}")
