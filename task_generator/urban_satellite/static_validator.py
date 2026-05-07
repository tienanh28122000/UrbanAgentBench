"""
static_validator.py
===================
Programmatic filter for generated tau2-bench urban_satellite scenarios.

Validates each sample in generated_all.json BEFORE any dynamic execution by
checking:
  1.  JSON structural integrity (required top-level fields, correct types)
  2.  Known tool names (no hallucinated tool names)
  3.  Required argument presence per tool signature
  4.  Argument type correctness (numeric/str where required)
  5.  get_satellite_tile: derived site_id must exist in db.sites
        (tool ALWAYS raises ValueError if key missing — hard crash)
  6.  get_past_satellite_tile: site must exist AND has_temporal must be True
        (tool ALWAYS raises ValueError on either miss — hard crash)
  7.  VLM tools (classify_land_use, analyze_urban_density, etc.):
        image_path filename must resolve to a known site_id in db.sites
        (past images are allowed — e.g. for temporal before/after comparisons)
  8.  compare_temporal_change: image_path_1 must be past image (_past.png),
        image_path_2 must be current image; both must map to the same site_id,
        and that site must have has_temporal=True
  9.  submit_site_assessment: site_id must exist in db.sites
  10. First action in every scenario must be get_satellite_tile
  11. nl_assertions must be a non-empty list
  12. No duplicate action_ids

Deliberately NOT checked (tool returns valid non-error result on unexpected input):
  - classify_land_use, analyze_urban_density, check_environmental_ratio,
    estimate_carbon_emission, detect_infrastructure, verify_path_connectivity,
    compare_temporal_change: VLM-backed; the tool does not raise ValueError for
    bad image CONTENT — the image_path format/site check in item 7 is sufficient.
  - measure_spatial_distance: pure Haversine math; never raises ValueError.
  - transfer_to_human_agents: always succeeds if args are present.

Usage
-----
    python static_validator.py [--input <path>] [--db <path>] [--output <path>]

Defaults:
    --input   scripts/urban_satellite/generated/generated_all.json
    --db      data/tau2/domains/urban_satellite/db.json
    --output  scripts/urban_satellite/generated/generated_all_static_validator.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants reflecting tools.py
# ---------------------------------------------------------------------------

VALID_TOOLS: set[str] = {
    "get_satellite_tile",
    "get_past_satellite_tile",
    "classify_land_use",
    "analyze_urban_density",
    "check_environmental_ratio",
    "estimate_carbon_emission",
    "detect_infrastructure",
    "verify_path_connectivity",
    "compare_temporal_change",
    "measure_spatial_distance",
    "submit_site_assessment",
    "transfer_to_human_agents",
}

# Required arguments per tool (no defaults in tools.py)
REQUIRED_ARGS: dict[str, list[str]] = {
    "get_satellite_tile":        ["lat", "lon"],
    "get_past_satellite_tile":   ["lat", "lon"],
    "classify_land_use":         ["image_path"],
    "analyze_urban_density":     ["image_path"],
    "check_environmental_ratio": ["image_path"],
    "estimate_carbon_emission":  ["image_path"],
    "detect_infrastructure":     ["image_path", "feature_query"],
    "verify_path_connectivity":  ["image_path", "start_coord", "end_coord"],
    "compare_temporal_change":   ["image_path_1", "image_path_2"],
    "measure_spatial_distance":  ["lat1", "lon1", "lat2", "lon2"],
    "submit_site_assessment":    ["site_id", "decision", "justification"],
    "transfer_to_human_agents":  ["summary"],
}

# Arguments that must be numeric (int or float, not bool)
NUMERIC_ARGS: dict[str, list[str]] = {
    "get_satellite_tile":       ["lat", "lon"],
    "get_past_satellite_tile":  ["lat", "lon"],
    "measure_spatial_distance": ["lat1", "lon1", "lat2", "lon2"],
}

# Arguments that must be strings
STR_ARGS: dict[str, list[str]] = {
    "classify_land_use":         ["image_path"],
    "analyze_urban_density":     ["image_path"],
    "check_environmental_ratio": ["image_path"],
    "estimate_carbon_emission":  ["image_path"],
    "detect_infrastructure":     ["image_path", "feature_query"],
    "verify_path_connectivity":  ["image_path", "start_coord", "end_coord"],
    "compare_temporal_change":   ["image_path_1", "image_path_2"],
    "submit_site_assessment":    ["site_id", "decision", "justification"],
    "transfer_to_human_agents":  ["summary"],
}

# VLM tools that take a single image_path (may be current or past — no restriction in tools.py)
_SINGLE_IMAGE_TOOLS: frozenset[str] = frozenset({
    "classify_land_use",
    "analyze_urban_density",
    "check_environmental_ratio",
    "estimate_carbon_emission",
    "detect_infrastructure",
    "verify_path_connectivity",
})

# Top-level fields every generated sample must have
REQUIRED_SAMPLE_FIELDS: list[str] = [
    "id", "description", "user_scenario", "evaluation_criteria",
]

# Regex: site_id filenames like "19652_30149.png" or "19652_30149_past.png"
_RE_SITE_FILENAME = re.compile(r"^(\d+_\d+)(?:_past)?\.png$")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_numeric(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _deg2num(lat_deg: float, lon_deg: float, zoom: int = 15) -> tuple[int, int]:
    """Convert lat/lon to OSM tile (x, y). Mirrors utils.py exactly (uses asinh)."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def _latlon_to_site_id(lat: float, lon: float, zoom: int = 15) -> str:
    """Derive the site_id from coordinates — matches tools.py: f'{y}_{x}'."""
    x, y = _deg2num(lat, lon, zoom)
    return f"{y}_{x}"


def _image_path_to_site_id(image_path: str) -> str | None:
    """
    Extract site_id from a filename like '19652_30149.png' or '19652_30149_past.png'.
    Returns None if the filename does not match the expected pattern.
    """
    fname = os.path.basename(image_path)
    m = _RE_SITE_FILENAME.match(fname)
    return m.group(1) if m else None


def _is_past_image(image_path: str) -> bool:
    return os.path.basename(image_path).endswith("_past.png")


# ---------------------------------------------------------------------------
# Per-action validator
# ---------------------------------------------------------------------------


def validate_action(action: dict, db: dict) -> list[str]:
    """Return a list of error strings for *action*. Empty list means valid."""
    errors: list[str] = []
    name = action.get("name")
    args = action.get("arguments")
    sites = db.get("sites", {})

    # ---- 1. Tool name ----
    if not isinstance(name, str) or name not in VALID_TOOLS:
        errors.append(f"Unknown tool name: {name!r}")
        return errors  # all further checks are meaningless

    # ---- 2. Arguments must be a dict ----
    if args is None:
        args = {}
    if not isinstance(args, dict):
        errors.append(f"[{name}] 'arguments' must be a dict, got {type(args).__name__}")
        return errors

    # ---- 3. Required arguments ----
    for req in REQUIRED_ARGS.get(name, []):
        if req not in args:
            errors.append(f"[{name}] Missing required argument: '{req}'")

    # ---- 4. Numeric type checks ----
    for k in NUMERIC_ARGS.get(name, []):
        if k in args and not _is_numeric(args[k]):
            errors.append(
                f"[{name}] Argument '{k}' must be numeric, "
                f"got {type(args[k]).__name__}: {args[k]!r}"
            )

    # ---- 5. String type checks ----
    for k in STR_ARGS.get(name, []):
        if k in args and not isinstance(args[k], str):
            errors.append(
                f"[{name}] Argument '{k}' must be str, "
                f"got {type(args[k]).__name__}: {args[k]!r}"
            )

    # ---- 6. Non-empty string checks ----
    if name == "detect_infrastructure" and isinstance(args.get("feature_query"), str):
        if not args["feature_query"].strip():
            errors.append(f"[{name}] 'feature_query' must not be an empty string")

    if name == "transfer_to_human_agents" and isinstance(args.get("summary"), str):
        if not args["summary"].strip():
            errors.append(f"[{name}] 'summary' must not be an empty string")

    # =========================================================
    # DB-level hard-crash checks
    # =========================================================

    # ---- get_satellite_tile / get_past_satellite_tile ----
    if name in ("get_satellite_tile", "get_past_satellite_tile"):
        lat = args.get("lat")
        lon = args.get("lon")
        if _is_numeric(lat) and _is_numeric(lon):
            site_id = _latlon_to_site_id(lat, lon)
            if site_id not in sites:
                errors.append(
                    f"[{name}] lat={lat}, lon={lon} maps to tile {site_id!r} "
                    f"which is NOT in db.sites — tool will raise ValueError (hard crash)"
                )
            elif name == "get_past_satellite_tile":
                meta = (sites[site_id].get("meta") or {})
                if not meta.get("has_temporal", False):
                    errors.append(
                        f"[get_past_satellite_tile] Site {site_id!r} exists but "
                        f"has_temporal=False — tool will raise ValueError (hard crash)"
                    )

    # ---- Single-image VLM tools ----
    elif name in _SINGLE_IMAGE_TOOLS:
        ip = args.get("image_path")
        if isinstance(ip, str):
            sid = _image_path_to_site_id(ip)
            if sid is None:
                errors.append(
                    f"[{name}] image_path basename {os.path.basename(ip)!r} "
                    f"does not match expected pattern '<y>_<x>.png'"
                )
            elif sid not in sites:
                errors.append(
                    f"[{name}] image_path references site_id {sid!r} "
                    f"which is not in db.sites"
                )

    # ---- compare_temporal_change ----
    elif name == "compare_temporal_change":
        ip1 = args.get("image_path_1")
        ip2 = args.get("image_path_2")

        if isinstance(ip1, str):
            sid1 = _image_path_to_site_id(ip1)
            # image_path_1 must be the PAST image
            if not _is_past_image(ip1):
                errors.append(
                    f"[{name}] 'image_path_1' should be the past image (ending in "
                    f"_past.png), got: {os.path.basename(ip1)!r}"
                )
            if sid1 is None:
                errors.append(
                    f"[{name}] image_path_1 basename {os.path.basename(ip1)!r} "
                    f"does not match expected pattern"
                )
            elif sid1 not in sites:
                errors.append(
                    f"[{name}] image_path_1 references unknown site_id {sid1!r}"
                )
            else:
                meta1 = (sites[sid1].get("meta") or {})
                if not meta1.get("has_temporal", False):
                    errors.append(
                        f"[{name}] Site {sid1!r} (image_path_1) has has_temporal=False; "
                        f"no past image should exist for this site"
                    )
        else:
            sid1 = None

        if isinstance(ip2, str):
            sid2 = _image_path_to_site_id(ip2)
            # image_path_2 must be the CURRENT image
            if _is_past_image(ip2):
                errors.append(
                    f"[{name}] 'image_path_2' should be the current image (not _past.png), "
                    f"got: {os.path.basename(ip2)!r}"
                )
            if sid2 is None:
                errors.append(
                    f"[{name}] image_path_2 basename {os.path.basename(ip2)!r} "
                    f"does not match expected pattern"
                )
            elif sid2 not in sites:
                errors.append(
                    f"[{name}] image_path_2 references unknown site_id {sid2!r}"
                )
        else:
            sid2 = None

        # Both images must reference the same site
        if sid1 and sid2 and sid1 != sid2:
            errors.append(
                f"[{name}] image_path_1 (site {sid1!r}) and image_path_2 (site {sid2!r}) "
                f"reference different sites — temporal comparison must use the same site"
            )

    # ---- submit_site_assessment ----
    elif name == "submit_site_assessment":
        site_id = args.get("site_id")
        if isinstance(site_id, str) and site_id not in sites:
            errors.append(
                f"[submit_site_assessment] site_id {site_id!r} not found in db.sites"
            )

    return errors


# ---------------------------------------------------------------------------
# Per-sample validator
# ---------------------------------------------------------------------------


def validate_sample(sample: dict, db: dict) -> list[str]:
    """Return all errors for *sample*. Empty list means fully valid."""
    errors: list[str] = []

    # ---- 1. Required top-level fields ----
    for field in REQUIRED_SAMPLE_FIELDS:
        if field not in sample:
            errors.append(f"Missing top-level field: '{field}'")

    # ---- 2. evaluation_criteria structure ----
    ec = sample.get("evaluation_criteria")
    if not isinstance(ec, dict):
        errors.append("'evaluation_criteria' must be a dict")
        return errors

    actions = ec.get("actions")
    if not isinstance(actions, list):
        errors.append("'evaluation_criteria.actions' must be a list")
        return errors

    if len(actions) == 0:
        errors.append("'evaluation_criteria.actions' must not be empty")

    # ---- 3. nl_assertions must not be empty ----
    nl_assertions = ec.get("nl_assertions")
    if not isinstance(nl_assertions, list) or len(nl_assertions) == 0:
        errors.append("'evaluation_criteria.nl_assertions' must be a non-empty list")

    # ---- 4. First action must be get_satellite_tile ----
    if actions and isinstance(actions[0], dict):
        first_name = actions[0].get("name")
        if first_name != "get_satellite_tile":
            errors.append(
                f"First action must be 'get_satellite_tile', got {first_name!r}"
            )

    # ---- 5. Per-action validation ----
    seen_action_ids: set[str] = set()
    for i, action in enumerate(actions):
        if not isinstance(action, dict):
            errors.append(f"action[{i}] is not a dict")
            continue

        aid = action.get("action_id")
        if not isinstance(aid, str) or not aid:
            errors.append(f"action[{i}] missing or invalid 'action_id'")
        elif aid in seen_action_ids:
            errors.append(f"action[{i}] duplicate action_id: {aid!r}")
        else:
            seen_action_ids.add(aid)

        action_errors = validate_action(action, db)
        for ae in action_errors:
            errors.append(f"action[{i}] ({aid!r}): {ae}")

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Static validator for generated urban_satellite benchmark scenarios."
    )
    script_dir = Path(__file__).resolve().parent
    repo_root  = script_dir.parent.parent  # scripts/urban_satellite/ -> repo root

    parser.add_argument(
        "--input",
        default=str(script_dir / "generated" / "generated_all.json"),
        help="Path to the generated scenarios JSON file",
    )
    parser.add_argument(
        "--db",
        default=str(repo_root / "data" / "tau2" / "domains" / "urban_satellite" / "db.json"),
        help="Path to the urban_satellite db.json",
    )
    parser.add_argument(
        "--output",
        default=str(script_dir / "generated" / "generated_all_static_validator.json"),
        help="Path to write the valid (cleaned) output JSON",
    )
    parser.add_argument(
        "--error-output",
        default=str(script_dir / "generated" / "generated_all_static_validator_error.json"),
        help="Path to write invalid samples with their errors",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    db_path     = Path(args.db)
    output_path = Path(args.output)
    error_path  = Path(args.error_output)

    print(f"Loading input  : {input_path}")
    print(f"Loading DB     : {db_path}")
    print(f"Output (valid) : {output_path}")
    print(f"Output (errors): {error_path}")
    print()

    try:
        with open(input_path, encoding="utf-8") as f:
            raw_data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"ERROR: Cannot load input file: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(db_path, encoding="utf-8") as f:
            db = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"ERROR: Cannot load DB file: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(raw_data, list):
        print("ERROR: Input JSON must be a list of samples.", file=sys.stderr)
        sys.exit(1)

    total            = len(raw_data)
    valid_samples:   list[dict] = []
    invalid_samples: list[dict] = []

    print(f"Validating {total} samples...\n{'=' * 70}")

    for sample in raw_data:
        if not isinstance(sample, dict):
            print("[SKIP] Non-dict entry")
            invalid_samples.append({"_raw": str(sample), "_validation_errors": ["Not a dict"]})
            continue

        sample_id = sample.get("id", "<unknown>")
        errors    = validate_sample(sample, db)

        if errors:
            print(f"[FAIL] {sample_id}")
            for err in errors:
                print(f"       - {err}")
            error_record = dict(sample)
            error_record["_validation_errors"] = errors
            invalid_samples.append(error_record)
        else:
            valid_samples.append(sample)
            print(f"[PASS] {sample_id}")

    print(f"{'=' * 70}")
    print(f"\nResults: {len(valid_samples)} valid / {len(invalid_samples)} invalid / {total} total")
    if total:
        print(f"Pass rate: {len(valid_samples) / total * 100:.1f}%\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(valid_samples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(valid_samples):>4} valid   samples → {output_path}")

    error_path.parent.mkdir(parents=True, exist_ok=True)
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(invalid_samples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(invalid_samples):>4} invalid samples → {error_path}")


if __name__ == "__main__":
    main()
