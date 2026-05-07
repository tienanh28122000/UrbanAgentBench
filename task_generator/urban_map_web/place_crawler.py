"""
place_crawler.py — Run ONCE to pre-crawl all landmarks.
Results are saved to places_cache.json (raw Google API response).

data_crawler.py will read this file to avoid re-calling the API when re-running.

Usage:
    export GOOGLE_MAPS_API_KEY=...
    python place_crawler.py
"""

import os
import csv
import json
import requests

# ==========================================
# CONFIG
# ==========================================

API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

CITY = "newyork"  # [london, tokyo, paris, newyork, "melbourne"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKS_CSV_PATH = os.path.join(SCRIPT_DIR, "pois", f"{CITY}_pois.csv")
PLACES_CACHE_PATH = os.path.join(SCRIPT_DIR, f"places_cache_{CITY}.json")

# ==========================================
# HELPERS
# ==========================================

def load_landmarks() -> list:
    """Read all landmark names from the landmarks CSV."""
    landmarks = []
    with open(LANDMARKS_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Feature Name", "").strip()
            if name:
                landmarks.append(name)
    return landmarks


def fetch_text_search(query: str) -> str | None:
    """Call the Places Text Search API, returning the first place_id or None."""
    print(f"  [search] '{query}'...")
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "places.id",
    }
    payload = {"textQuery": query, "languageCode": "en", "maxResultCount": 1}
    resp = requests.post(url, json=payload, headers=headers, timeout=15)
    if resp.status_code == 200:
        places = resp.json().get("places", [])
        if places:
            return places[0]["id"]
    print(f"    [!] Error or not found: {resp.text[:120]}")
    return None


def fetch_place_details(place_id: str) -> dict:
    """Call the Place Details API, returning the raw response dict."""
    print(f"  [details] {place_id}...")
    url = f"https://places.googleapis.com/v1/places/{place_id}"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": (
            "id,displayName,formattedAddress,location,types,"
            "rating,userRatingCount,priceLevel,websiteUri,nationalPhoneNumber,"
            "regularOpeningHours.weekdayDescriptions,regularOpeningHours.openNow,businessStatus,"
            "accessibilityOptions,allowsDogs,goodForChildren,servesVegetarianFood,outdoorSeating,reviews"
        ),
    }
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code == 200:
        return resp.json()
    print(f"    [!] Place Details Error: {resp.text[:120]}")
    return {}


def load_existing_cache() -> dict:
    """Read the existing cache to resume if interrupted."""
    if os.path.exists(PLACES_CACHE_PATH):
        with open(PLACES_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"landmarks": {}, "by_id": {}}


def save_cache(cache: dict):
    with open(PLACES_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ==========================================
# MAIN
# ==========================================

def run():
    if not API_KEY:
        print("[!] Please set the GOOGLE_MAPS_API_KEY environment variable before running.")
        return

    all_landmarks = load_landmarks()
    cache = load_existing_cache()

    already_done = set(cache["landmarks"].keys())
    remaining = [lm for lm in all_landmarks if lm not in already_done]

    print(f"[Cache] Already have: {len(already_done)} / {len(all_landmarks)} landmarks")
    print(f"[Cache] Need to crawl: {len(remaining)} additional landmarks\n")

    for idx, name in enumerate(remaining, start=1):
        print(f"[{idx}/{len(remaining)}] {name}")

        place_id = fetch_text_search(name)
        if not place_id:
            # Mark as null to avoid useless retries
            cache["landmarks"][name] = None
            save_cache(cache)
            continue

        # If this place_id already has details (possible alias with same name), reuse it
        if place_id not in cache["by_id"]:
            details = fetch_place_details(place_id)
            if not details:
                cache["landmarks"][name] = None
                save_cache(cache)
                continue
            cache["by_id"][place_id] = details

        cache["landmarks"][name] = place_id
        # Save immediately after each landmark to prevent data loss if interrupted
        save_cache(cache)

    valid = sum(1 for v in cache["landmarks"].values() if v is not None)
    print(f"\n[OK] Completed! {valid}/{len(all_landmarks)} landmarks have been cached in {PLACES_CACHE_PATH}")


if __name__ == "__main__":
    run()
