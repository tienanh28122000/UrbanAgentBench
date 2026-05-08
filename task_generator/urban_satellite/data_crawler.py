import os
import json
import time
import glob
from typing import Dict, List

import pandas as pd
import requests


# --- Configuration & Paths ---
CSV_PATH = "resources/London_img_indicators.csv"
EXISTING_OBJ_JSON = "resources/all_city_img_object_set.json"
OUTPUT_DB_PATH = "urban_satellite_db_London.json"
IMAGE_DIR = "satellite_imgs" # Create a symlink from domains/urban_satellite/data/satellite_imgs to this path.
USE_ALL_CSV_FOR_DENSITY = True

# OSM crawling params
OSM_OFFSET = 0.005
OSM_MAX_RETRIES = 4

# Override mode: skip OSM land_use query if True (useful for updating density/infra without re-crawling)
OVERRIDE_LAND_USE = True
OVERRIDE_LAND_USE_SOURCE = "data/urban_satellite_db_London.json"  # Load existing land_use from this file

# Override infra counts: when True, IGNORE existing_objs and crawl OSM directly for each site
# Use this when all_city_img_object_set.json has incorrect detections
OVERRIDE_INFRA_COUNTS = True

# Only keep these infrastructure keys in the final database
INFRA_FILTER_KEYS = [
    "Airport",
    "Stadium",
    "Golf Field",
    "Harbor",
    "Ground Track Field",
    "Soccer Ball Field",
    "Train Station",
]


def compute_population_density_rank_0_to_9(df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.Series:
    """Compute worldpop percentile rank in fractional bins [0.0..9.9], matching metrics.py format."""
    worldpop_ref = reference_df["worldpop"].rank(pct=True)
    worldpop_map = pd.DataFrame({
        "img_name": reference_df["img_name"].astype(str),
        "pop_rank_pct": worldpop_ref,
    })
    merged = df[["img_name"]].copy()
    merged["img_name"] = merged["img_name"].astype(str)
    merged = merged.merge(worldpop_map, on="img_name", how="left")

    # Convert percentile [0,1] to fractional bins: 0.0..9.9 (same as metrics.py)
    pop_rank_pct = merged["pop_rank_pct"].fillna(0.0)
    pop_rank = [int(r * 100.0) / 10.0 if r < 1.0 else 9.9 for r in pop_rank_pct]
    return pd.Series(pop_rank, index=merged.index)


def load_density_reference_df() -> pd.DataFrame:
    if not USE_ALL_CSV_FOR_DENSITY:
        return pd.read_csv(CSV_PATH)

    resources_dir = os.path.dirname(CSV_PATH) or "."
    csv_paths = sorted(glob.glob(os.path.join(resources_dir, "*_img_indicators.csv")))
    if not csv_paths:
        return pd.read_csv(CSV_PATH)

    all_df: List[pd.DataFrame] = []
    for path in csv_paths:
        try:
            all_df.append(pd.read_csv(path))
        except Exception:
            continue

    if not all_df:
        return pd.read_csv(CSV_PATH)
    return pd.concat(all_df, axis=0, ignore_index=True)


# OSM tag filters for each infrastructure type
_INFRA_OSM_QUERIES: Dict[str, str] = {
    "Airport":           '(node["aeroway"~"aerodrome|airport"]{bbox};way["aeroway"~"aerodrome|airport"]{bbox};relation["aeroway"~"aerodrome|airport"]{bbox};);',
    "Stadium":           '(node["leisure"="stadium"]{bbox};way["leisure"="stadium"]{bbox};node["building"="stadium"]{bbox};way["building"="stadium"]{bbox};);',
    "Golf Field":        '(node["leisure"="golf_course"]{bbox};way["leisure"="golf_course"]{bbox};relation["leisure"="golf_course"]{bbox};);',
    "Harbor":            '(node["harbour"="yes"]{bbox};way["harbour"="yes"]{bbox};node["landuse"="harbour"]{bbox};way["landuse"="harbour"]{bbox};);',
    "Ground Track Field":'(node["leisure"="track"]{bbox};way["leisure"="track"]{bbox};);',
    "Soccer Ball Field": '(node["leisure"="pitch"]["sport"~"soccer|football"]{bbox};way["leisure"="pitch"]["sport"~"soccer|football"]{bbox};);',
    "Train Station":     '(node["railway"="station"]{bbox};node["public_transport"="stop_position"]["train"="yes"]{bbox};way["railway"="station"]{bbox};);',
}


def fetch_osm_infrastructure_counts(
    lat: float,
    lon: float,
    offset: float = OSM_OFFSET,
    max_retries: int = OSM_MAX_RETRIES,
) -> Dict[str, int]:
    """
    Queries OSM Overpass for each infrastructure type and returns a 0/1 dict.
    Only queries keys listed in INFRA_FILTER_KEYS.
    """
    s, n = lat - offset, lat + offset
    w, e = lon - offset, lon + offset
    bbox = f"{s},{w},{n},{e}"

    headers = {"User-Agent": "UrbanSatelliteBenchmark/1.0 (Research Project)"}
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]

    result: Dict[str, int] = {}
    for key in INFRA_FILTER_KEYS:
        osm_filter = _INFRA_OSM_QUERIES.get(key, "").replace("{bbox}", f"({bbox})")
        if not osm_filter:
            result[key] = 0
            continue

        query = f"[out:json][timeout:40];{osm_filter}out count;"
        found = 0
        for attempt in range(max_retries):
            url = endpoints[attempt % len(endpoints)]
            try:
                resp = requests.get(url, params={"data": query}, headers=headers, timeout=45)
                if resp.status_code == 429:
                    time.sleep((attempt + 1) * 5)
                    continue
                if resp.status_code == 504:
                    time.sleep((attempt + 1) * 8)
                    continue
                resp.raise_for_status()
                data = resp.json()
                total = data.get("elements", [{}])[0].get("tags", {}).get("total", "0")
                found = 1 if int(total) > 0 else 0
                break
            except Exception:
                time.sleep((attempt + 1) * 3)
        result[key] = found
        time.sleep(1.0)  # be polite between per-key queries

    return result


# --- OSM land-use crawl for current image area ---
def fetch_osm_land_use_for_current_image(
    lat: float,
    lon: float,
    offset: float = OSM_OFFSET,
    max_retries: int = OSM_MAX_RETRIES,
) -> List[str]:
    """
    Queries OSM and returns list of land-use categories found.
    Example: ['landuse'], ['natural'], or [] if none found.
    """
    s, n = lat - offset, lat + offset
    w, e = lon - offset, lon + offset

    query = f"""
    [out:json][timeout:40];
    (
      way["landuse"]({s},{w},{n},{e});
      relation["landuse"]({s},{w},{n},{e});
      way["natural"]({s},{w},{n},{e});
      relation["natural"]({s},{w},{n},{e});
    );
    out tags;
    """

    headers = {"User-Agent": "UrbanSatelliteBenchmark/1.0 (Research Project)"}
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]

    grouped_counts: Dict[str, Dict[str, int]] = {
        "landuse": {},
        "natural": {},
    }

    for attempt in range(max_retries):
        url = endpoints[attempt % len(endpoints)]
        try:
            resp = requests.get(url, params={"data": query}, headers=headers, timeout=45)

            if resp.status_code == 429:
                time.sleep((attempt + 1) * 5)
                continue
            if resp.status_code == 504:
                time.sleep((attempt + 1) * 8)
                continue

            resp.raise_for_status()
            data = resp.json()

            for element in data.get("elements", []):
                tags = element.get("tags", {})

                if "landuse" in tags:
                    key = tags["landuse"]
                    grouped_counts["landuse"][key] = grouped_counts["landuse"].get(key, 0) + 1
                if "natural" in tags:
                    key = tags["natural"]
                    grouped_counts["natural"][key] = grouped_counts["natural"].get(key, 0) + 1

            # Build list of categories that have data (filter out None/empty)
            found_categories = []
            if grouped_counts["landuse"]:
                found_categories.append("landuse")
            if grouped_counts["natural"]:
                found_categories.append("natural")

            return found_categories
        except requests.exceptions.RequestException:
            time.sleep((attempt + 1) * 3)

    return []  # Return empty list if all retries failed


def build_database():
    global OVERRIDE_LAND_USE

    print("[*] Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    print("[*] Loading object ground truth...")
    with open(EXISTING_OBJ_JSON, "r", encoding="utf-8") as f:
        existing_objs = json.load(f)

    print("[*] Computing population_density rank (0-9)...")
    ref_df = load_density_reference_df()
    df["population_density"] = compute_population_density_rank_0_to_9(df, ref_df)

    db_data = {"sites": {}, "assessments": {}}
    skipped_no_land_use = 0

    # Load existing database if in OVERRIDE_LAND_USE mode
    existing_land_use_map = {}
    if OVERRIDE_LAND_USE:
        print(f"[*] OVERRIDE_LAND_USE=True. Loading existing land_use from {OVERRIDE_LAND_USE_SOURCE}...")
        try:
            with open(OVERRIDE_LAND_USE_SOURCE, "r", encoding="utf-8") as f:
                existing_db = json.load(f)
                for site_id, record in existing_db.get("sites", {}).items():
                    existing_land_use_map[site_id] = record["ground_truth"].get("land_use_categories", [])
            print(f"[*] Loaded land_use for {len(existing_land_use_map)} sites from existing database.")
        except FileNotFoundError:
            print(f"[!] OVERRIDE file not found: {OVERRIDE_LAND_USE_SOURCE}. Falling back to OSM queries.")
            OVERRIDE_LAND_USE = False

    print(f"[*] Processing {len(df)} sites...")
    for idx, row in df.iterrows():
        site_id = str(row["img_name"]).strip()
        lat = float(row["lat"])
        lon = float(row["lng"])

        curr_img_path = os.path.join(IMAGE_DIR, f"{site_id}.png")

        # Get land_use_categories: from override map or query OSM
        if OVERRIDE_LAND_USE and site_id in existing_land_use_map:
            land_use_categories = existing_land_use_map[site_id]
        else:
            land_use_categories = fetch_osm_land_use_for_current_image(lat, lon)
        
        if not land_use_categories:  # If list is empty, skip site
            skipped_no_land_use += 1
            continue

        if not OVERRIDE_LAND_USE:
            time.sleep(2.0)  # Sleep only if actually querying OSM

        # Resolve infrastructure counts
        if OVERRIDE_INFRA_COUNTS:
            # Crawl OSM directly — ignores existing_objs which may have wrong detections
            site_objects = fetch_osm_infrastructure_counts(lat, lon)
        else:
            # Filter existing_objs to only the configured infrastructure keys; default missing keys to 0
            site_objects = {k: int(existing_objs.get(site_id, {}).get(k, 0)) for k in INFRA_FILTER_KEYS}

        site_record = {
            "site_id": site_id,
            "lat": lat,
            "lon": lon,
            "ground_truth": {
                "population_density": float(row["population_density"]),  # Now 0.0-9.9
                "land_use_categories": land_use_categories,
                # "worldpop": float(row["worldpop"]),
                # "carbon": float(row["carbon"]),
                # "nightlight": float(row["nightlight"]),
                "infrastructure_counts": site_objects,
            },
            "meta": {
                "current_image": curr_img_path,
                "has_temporal": False,
                "past_image": None,
            },
        }

        db_data["sites"][site_id] = site_record

        if (idx + 1) % 100 == 0:
            print(f"  -> Processed {idx + 1}/{len(df)} sites. Cooling down...")
            time.sleep(10)
        elif (idx + 1) % 5 == 0:
            print(f"  -> Processed {idx + 1}/{len(df)} sites...")

    print("[*] Saving database...")
    with open(OUTPUT_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db_data, f, indent=4, ensure_ascii=False)

    print(f"[OK] Database built with {len(db_data['sites'])} sites.")
    print("[*] Skipped sites -> " f"no_land_use: {skipped_no_land_use}")
    if OVERRIDE_LAND_USE:
        print("[*] OVERRIDE_LAND_USE mode: Land-use categories loaded from existing database (no OSM queries)")


if __name__ == "__main__":
    build_database()
