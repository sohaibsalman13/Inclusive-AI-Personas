import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =============================
# CONFIG
# =============================
DATA_DIR = Path("data_raw")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

AGE_CSV = DATA_DIR / "altersgruppen.csv"
NONGER_CSV = DATA_DIR / "nichtdeutsche.csv"
MIGBG_CSV = DATA_DIR / "migrationshintergrund.csv"
DENSITY_CSV = DATA_DIR / "bevoelkerungsdichte.csv"
UNEMP_CSV = DATA_DIR / "arbeitslosendichte.csv"
SINGLE_HH_CSV = DATA_DIR / "einpersonenhaushalte.csv"
HH_WITH_KIDS_CSV = DATA_DIR / "haushalte_mit_kindern.csv"
SINGLE_PARENT_CSV = DATA_DIR / "haushalte_alleinerziehend.csv"
HH_SIZE_CSV = DATA_DIR / "privathaushalte.csv"

YEAR_MODE = "latest_common"  # or set YEAR_MODE = 2024 (int)

# With ~25 districts, large K is not meaningful, and it slows selection
K_MIN, K_MAX = 2, 6
MIN_CLUSTER_SIZE = 3

OUT_JSON = OUT_DIR / "munich_clusters_extended.json"
OUT_JSONL = OUT_DIR / "munich_clusters_extended.jsonl"

# Approximate the "5 and more people" household bucket with 5.0
FIVEPLUS_VALUE = 5.0

# silhouette sampling for speed (important on Windows)
SIL_SAMPLE_MAX = 10


# =============================
# HELPERS
# =============================
def safe_text(x) -> str:
    return str(x).strip()

def norm_space(s: str) -> str:
    return " ".join(safe_text(s).split())

def district_key_from_raumbezug(raumbezug: str) -> str:
    """
    Indikatorenatlas example: '01 Altstadt - Lehel'
    -> 'altstadt-lehel'
    """
    s = norm_space(raumbezug)

    # remove leading '01 ' if present
    if len(s) >= 3 and s[0:2].isdigit() and s[2] == " ":
        s = s[3:]

    # normalize hyphens/spaces
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace(" - ", "-")
    s = s.replace(" -", "-").replace("- ", "-")
    return s.lower()

def district_key_from_stadtbezirk(stadtbezirk: str) -> str:
    """
    privathaushalte example: 'Altstadt-Lehel'
    -> 'altstadt-lehel'
    """
    s = norm_space(stadtbezirk)
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace(" - ", "-")
    s = s.replace(" -", "-").replace("- ", "-")
    return s.lower()

def assert_cols(df: pd.DataFrame, cols, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}. Found: {list(df.columns)}")

def pick_year(dfs, year_mode):
    if isinstance(year_mode, int):
        return year_mode
    if year_mode == "latest_common":
        max_years = [int(df["Jahr"].max()) for df in dfs]
        return int(min(max_years))
    raise ValueError("YEAR_MODE must be int or 'latest_common'")

def normalize_age_label(label: str) -> str:
    mapping = {
        "bis 17 Jahre": "under18",
        "0 bis unter 18 Jahre": "under18",
        "65 Jahre und älter": "65plus",
        "65 Jahre und älter ": "65plus",
    }
    return mapping.get(label, label)

def load_indikatorenatlas_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert_cols(df, ["Jahr", "Raumbezug", "Indikatorwert"], path.name)
    df["Raumbezug"] = df["Raumbezug"].map(norm_space)
    df["district_key"] = df["Raumbezug"].map(district_key_from_raumbezug)
    return df

def to_numeric(df: pd.DataFrame, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def ensure_one_row_per_district(df: pd.DataFrame, value_cols):
    """
    Some datasets contain multiple rows per district (different Ausprägung categories).
    To prevent merge explosions, aggregate to ONE row per district.
    """
    return (
        df.groupby("district_key", as_index=False)[value_cols]
          .mean()
    )

def choose_k_with_min_cluster_size(Xs, k_min, k_max, min_cluster_size):
    report = []
    best = None

    for k in range(k_min, k_max + 1):
        print(f"Testing k={k}...")
        km = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels = km.fit_predict(Xs)

        counts = np.bincount(labels)
        min_size = int(counts.min())
        valid = min_size >= min_cluster_size

        sil = None
        if k >= 2 and len(np.unique(labels)) > 1:
            sample_size = min(SIL_SAMPLE_MAX, len(Xs))
            sil = float(silhouette_score(Xs, labels, sample_size=sample_size, random_state=42))

        report.append({"k": k, "silhouette": sil, "min_cluster_size": min_size, "valid": valid})

        if valid and sil is not None:
            if best is None:
                best = (k, sil)
            else:
                best_k, best_sil = best
                if sil > best_sil or (abs(sil - best_sil) < 1e-9 and k < best_k):
                    best = (k, sil)

    if best is None:
        best_k = sorted(report, key=lambda r: (r["min_cluster_size"], r["silhouette"] or -1), reverse=True)[0]["k"]
        return best_k, report

    return best[0], report

def compute_avg_household_size(privhh_df: pd.DataFrame) -> pd.DataFrame:
    """
    privathaushalte.csv snapshot (no Jahr):
      - stadtbezirk
      - haushalte_zusammen
      - haushalte_mit_1_person
      - haushalte_mit_2_personen
      - haushalte_mit_3_personen
      - haushalte_mit_4_personen
      - haushalte_mit_5_personen_und_mehr

    avg = weighted average household size per district.
    """
    df = privhh_df.copy()

    assert_cols(
        df,
        [
            "stadtbezirk",
            "haushalte_zusammen",
            "haushalte_mit_1_person",
            "haushalte_mit_2_personen",
            "haushalte_mit_3_personen",
            "haushalte_mit_4_personen",
            "haushalte_mit_5_personen_und_mehr",
        ],
        "privathaushalte.csv"
    )

    df["stadtbezirk"] = df["stadtbezirk"].map(norm_space)
    df["district_key"] = df["stadtbezirk"].map(district_key_from_stadtbezirk)

    num_cols = [
        "haushalte_zusammen",
        "haushalte_mit_1_person",
        "haushalte_mit_2_personen",
        "haushalte_mit_3_personen",
        "haushalte_mit_4_personen",
        "haushalte_mit_5_personen_und_mehr",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    total = df["haushalte_zusammen"]
    avg = (
        1 * df["haushalte_mit_1_person"]
        + 2 * df["haushalte_mit_2_personen"]
        + 3 * df["haushalte_mit_3_personen"]
        + 4 * df["haushalte_mit_4_personen"]
        + FIVEPLUS_VALUE * df["haushalte_mit_5_personen_und_mehr"]
    ) / total

    out = pd.DataFrame({"district_key": df["district_key"], "avg_household_size": avg})
    out = out.dropna(subset=["avg_household_size"]).drop_duplicates(subset=["district_key"], keep="first")
    return out


# =============================
# MAIN
# =============================
def main():
    # --- load time-series datasets ---
    age = pd.read_csv(AGE_CSV)
    assert_cols(age, ["Jahr", "Raumbezug", "Ausprägung", "Indikatorwert"], "altersgruppen.csv")
    age["Raumbezug"] = age["Raumbezug"].map(norm_space)
    age["district_key"] = age["Raumbezug"].map(district_key_from_raumbezug)
    age["Ausprägung"] = age["Ausprägung"].map(safe_text)

    nat = load_indikatorenatlas_csv(NONGER_CSV)
    mig = load_indikatorenatlas_csv(MIGBG_CSV)
    dens = load_indikatorenatlas_csv(DENSITY_CSV)
    unemp = load_indikatorenatlas_csv(UNEMP_CSV)
    singlehh = load_indikatorenatlas_csv(SINGLE_HH_CSV)
    kids = load_indikatorenatlas_csv(HH_WITH_KIDS_CSV)
    singleparent = load_indikatorenatlas_csv(SINGLE_PARENT_CSV)

    # snapshot dataset
    privhh = pd.read_csv(HH_SIZE_CSV)

    # --- pick a year ---
    year = pick_year([age, nat, mig, dens, unemp, singlehh, kids, singleparent], YEAR_MODE)

    # --- Age: under18 + 65plus (already filtered) ---
    age_y = age[age["Jahr"] == year].copy()
    age_y["age_key"] = age_y["Ausprägung"].map(normalize_age_label)

    under18 = (
        age_y[age_y["age_key"] == "under18"][["district_key", "Raumbezug", "Indikatorwert"]]
        .rename(columns={"Indikatorwert": "pct_under18"})
    )
    plus65 = (
        age_y[age_y["age_key"] == "65plus"][["district_key", "Indikatorwert"]]
        .rename(columns={"Indikatorwert": "pct_65plus"})
    )
    under18 = to_numeric(under18, ["pct_under18"])
    plus65 = to_numeric(plus65, ["pct_65plus"])

    # keep a readable district display name from under18 rows
    district_display = under18[["district_key", "Raumbezug"]].drop_duplicates("district_key").rename(
        columns={"Raumbezug": "district_display"}
    )

    # --- Other indicators: aggregate to one row per district ---
    nong = nat[nat["Jahr"] == year][["district_key", "Indikatorwert"]]
    nong = ensure_one_row_per_district(to_numeric(nong, ["Indikatorwert"]), ["Indikatorwert"]).rename(
        columns={"Indikatorwert": "pct_non_german"}
    )

    migbg = mig[mig["Jahr"] == year][["district_key", "Indikatorwert"]]
    migbg = ensure_one_row_per_district(to_numeric(migbg, ["Indikatorwert"]), ["Indikatorwert"]).rename(
        columns={"Indikatorwert": "pct_migration_background"}
    )

    density = dens[dens["Jahr"] == year][["district_key", "Indikatorwert"]]
    density = ensure_one_row_per_district(to_numeric(density, ["Indikatorwert"]), ["Indikatorwert"]).rename(
        columns={"Indikatorwert": "population_density"}
    )

    unempr = unemp[unemp["Jahr"] == year][["district_key", "Indikatorwert"]]
    unempr = ensure_one_row_per_district(to_numeric(unempr, ["Indikatorwert"]), ["Indikatorwert"]).rename(
        columns={"Indikatorwert": "unemployment_rate"}
    )

    singlehh_y = singlehh[singlehh["Jahr"] == year][["district_key", "Indikatorwert"]]
    singlehh_y = ensure_one_row_per_district(to_numeric(singlehh_y, ["Indikatorwert"]), ["Indikatorwert"]).rename(
        columns={"Indikatorwert": "pct_single_households"}
    )

    kids_y = kids[kids["Jahr"] == year][["district_key", "Indikatorwert"]]
    kids_y = ensure_one_row_per_district(to_numeric(kids_y, ["Indikatorwert"]), ["Indikatorwert"]).rename(
        columns={"Indikatorwert": "pct_households_with_kids"}
    )

    singleparent_y = singleparent[singleparent["Jahr"] == year][["district_key", "Indikatorwert"]]
    singleparent_y = ensure_one_row_per_district(to_numeric(singleparent_y, ["Indikatorwert"]), ["Indikatorwert"]).rename(
        columns={"Indikatorwert": "pct_single_parent_households"}
    )

    # --- Snapshot: avg household size (already one row per district) ---
    avg_hh = compute_avg_household_size(privhh)

    # --- Merge all ---
    df = district_display.merge(under18[["district_key", "pct_under18"]], on="district_key", how="inner")
    df = df.merge(plus65, on="district_key", how="inner")
    df = df.merge(nong, on="district_key", how="inner")
    df = df.merge(migbg, on="district_key", how="inner")
    df = df.merge(density, on="district_key", how="inner")
    df = df.merge(unempr, on="district_key", how="inner")
    df = df.merge(singlehh_y, on="district_key", how="inner")
    df = df.merge(kids_y, on="district_key", how="inner")
    df = df.merge(singleparent_y, on="district_key", how="inner")
    df = df.merge(avg_hh, on="district_key", how="inner")

    # remove city total if present
    df = df[df["district_display"] != "Stadt München"].copy()

    feature_cols = [
        "pct_under18",
        "pct_65plus",
        "pct_non_german",
        "pct_migration_background",
        "population_density",
        "unemployment_rate",
        "pct_single_households",
        "pct_households_with_kids",
        "pct_single_parent_households",
        "avg_household_size",
    ]
    df = df.dropna(subset=feature_cols).copy()

    print(f"\nUsing year: {year}")
    print("District rows after merge:", len(df))

    if len(df) < 2:
        raise ValueError("Not enough district rows after merge to cluster. Check data consistency.")

    # --- Cluster ---
    X = df[feature_cols].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    best_k, report = choose_k_with_min_cluster_size(Xs, K_MIN, K_MAX, MIN_CLUSTER_SIZE)

    km = KMeans(n_clusters=best_k, random_state=42, n_init=100)
    labels = km.fit_predict(Xs)

    counts = np.bincount(labels)
    print(f"\nChosen K: {best_k}")
    print(f"Cluster sizes (districts): {counts.tolist()} (min={int(counts.min())})")

    # Centroids back in original scale
    centroids = scaler.inverse_transform(km.cluster_centers_)
    centroids_df = pd.DataFrame(centroids, columns=feature_cols)
    centroids_df["cluster"] = range(best_k)

    df_out = df.copy()
    df_out["cluster"] = labels.astype(int)

    cluster_sizes = df_out["cluster"].value_counts().sort_index().to_dict()

    cluster_summaries = []
    for _, row in centroids_df.sort_values("cluster").iterrows():
        c = int(row["cluster"])
        cluster_summaries.append({
            "cluster": c,
            "size": int(cluster_sizes.get(c, 0)),
            "centroid_features": {col: float(row[col]) for col in feature_cols}
        })

    districts = []
    for _, row in df_out.sort_values(["cluster", "district_display"]).iterrows():
        districts.append({
            "district": row["district_display"],
            "district_key": row["district_key"],
            "year": int(year),
            "features": {col: float(row[col]) for col in feature_cols},
            "cluster": int(row["cluster"])
        })

    payload = {
        "metadata": {
            "source": "Munich Open Data Portal (Indikatorenatlas + privathaushalte snapshot)",
            "year": int(year),
            "kmeans": {
                "k_chosen": int(best_k),
                "min_cluster_size_constraint": int(MIN_CLUSTER_SIZE),
                "k_search_range": [int(K_MIN), int(K_MAX)],
                "fiveplus_bucket_value": float(FIVEPLUS_VALUE),
                "silhouette_sample_max": int(SIL_SAMPLE_MAX),
            }
        },
        "k_selection_report": report,
        "feature_columns": feature_cols,
        "cluster_summaries": cluster_summaries,
        "districts": districts
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for d in districts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n✅ Wrote: {OUT_JSON}")
    print(f"✅ Wrote: {OUT_JSONL}")


if __name__ == "__main__":
    main()
