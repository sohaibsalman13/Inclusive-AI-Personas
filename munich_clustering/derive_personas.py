import json
import numpy as np
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
INPUT_JSON = Path("outputs/munich_clusters_extended.json")
OUT_JSON = Path("outputs/ai_personas_from_clusters.json")
OUT_JSONL = Path("outputs/ai_personas_from_clusters.jsonl")

FEATURES = [
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

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    return json.loads(INPUT_JSON.read_text(encoding="utf-8"))

# -----------------------------
# QUANTILE THRESHOLDS
# -----------------------------
def compute_thresholds(values_by_feature):
    thresholds = {}
    for f, vals in values_by_feature.items():
        thresholds[f] = {
            "low": np.percentile(vals, 33),
            "high": np.percentile(vals, 66),
        }
    return thresholds

def bucket(value, t):
    if value <= t["low"]:
        return "low"
    elif value >= t["high"]:
        return "high"
    else:
        return "medium"

# -----------------------------
# FEATURE MATRIX
# -----------------------------
def build_feature_matrix(districts):
    values = {f: [] for f in FEATURES}
    for d in districts:
        for f in FEATURES:
            values[f].append(d["features"][f])
    return values

# -----------------------------
# PERSONA LABEL
# -----------------------------
def persona_label(tags):
    if tags["population_density"] == "high" and tags["avg_household_size"] == "low":
        return "Dense Urban Singles"
    if tags["pct_households_with_kids"] == "high":
        return "Family-Oriented Households"
    if tags["pct_65plus"] == "high":
        return "Ageing Residential Areas"
    if tags["pct_migration_background"] == "high":
        return "Diverse Migration Districts"
    return "Mixed Residential Profile"

# -----------------------------
# NEEDS / FRICTIONS / PREFERENCES
# -----------------------------
def derive_needs_and_frictions(tags):
    needs = []
    frictions = []
    preferences = []

    # Migration background
    if tags["pct_migration_background"] == "high":
        needs += ["language support", "clear administrative guidance"]
        frictions += ["complex terminology", "unclear processes"]
        preferences += ["simple language", "step-by-step explanations"]

    # Household structure
    if tags["pct_households_with_kids"] == "high":
        needs += ["family-related services", "education and childcare info"]
        frictions += ["time constraints"]
        preferences += ["clear summaries", "actionable checklists"]

    if tags["avg_household_size"] == "low":
        needs += ["individual-focused services"]
        preferences += ["fast access", "minimal steps"]

    # Density
    if tags["population_density"] == "high":
        needs += ["digital-first access", "efficient service handling"]
        frictions += ["overloaded services"]
        preferences += ["online self-service", "short response times"]

    # Unemployment
    if tags["unemployment_rate"] == "high":
        needs += ["clear eligibility information", "support navigation"]
        frictions += ["fear of mistakes", "uncertain requirements"]
        preferences += ["guided flows", "help tooltips"]

    # Clean duplicates
    needs = list(dict.fromkeys(needs))
    frictions = list(dict.fromkeys(frictions))
    preferences = list(dict.fromkeys(preferences))

    return needs, frictions, preferences

# -----------------------------
# BUILD PERSONAS
# -----------------------------
def build_personas(data):
    districts = data["districts"]
    clusters = data["cluster_summaries"]

    values = build_feature_matrix(districts)
    thresholds = compute_thresholds(values)

    personas = []

    for c in clusters:
        centroid = c["centroid_features"]

        tags = {
            f: bucket(centroid[f], thresholds[f])
            for f in FEATURES
        }

        needs, frictions, preferences = derive_needs_and_frictions(tags)

        persona = {
            "cluster": c["cluster"],
            "label": persona_label(tags),
            "cluster_size": c["size"],
            "numeric_profile": centroid,
            "tags": tags,
            "needs": needs,
            "frictions": frictions,
            "preferences": preferences,
            "system_prompt": (
                "You are an AI persona representing residents from this cluster. "
                "Respond based on the following characteristics:\n"
                f"- Area type: {tags['population_density']}\n"
                f"- Household structure: {tags['avg_household_size']}\n"
                f"- Migration context: {tags['pct_migration_background']}\n"
                "Be realistic, concise, and consistent."
            )
        }

        personas.append(persona)

    return personas

# -----------------------------
# MAIN
# -----------------------------
def main():
    data = load_data()
    personas = build_personas(data)

    OUT_JSON.write_text(
        json.dumps(personas, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for p in personas:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"✅ Wrote: {OUT_JSON}")
    print(f"✅ Wrote: {OUT_JSONL}")

if __name__ == "__main__":
    main()
