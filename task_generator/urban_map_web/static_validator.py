"""
static_validator.py
===================
Programmatic filter for generated tau2-bench urban_map_web scenarios.

Validates each sample in generated_all.json BEFORE any dynamic execution by
checking:
  1.  JSON structural integrity (required top-level fields present, correct types)
  2.  Known tool names (no hallucinated tool names)
  3.  Required argument presence per tool signature
  4.  Argument type correctness (numeric, int, str where required)
  5.  place_id references exist in UrbanDB.places
  6.  user_id references exist in UrbanDB.users
  7.  compute_routes: rounded key must exist in UrbanDB.routes
        (tool ALWAYS raises ValueError if key missing — hard crash)
  8.  get_transit_schedule: place_id must be a recognised transit stop
        (tool ALWAYS raises ValueError if not a transit stop — hard crash)
  9.  read_place_website: URL must be cached in UrbanDB.webpages
        (tool ALWAYS raises ValueError if URL not cached — hard crash)
  10. search_along_route: derived search key must exist in UrbanDB.search_along_routes
        (tool ALWAYS raises ValueError if key missing — hard crash)
  11. submit_council_report: issue_type must be one of the valid enum values
  12. datetime_str / date_str format validation (YYYY-MM-DD HH:MM / YYYY-MM-DD)
  13. party_size must be a positive integer

Deliberately NOT checked (tool returns a valid non-error response even on miss):
  - check_availability: unknown slot returns {"is_available": False} — scenario may
    intentionally test unavailability before trying an alternative slot.
  - book_place: slot/capacity — same reasoning; a failing book can be part of the flow.
  - search_venue_events: unknown place returns [] — empty result is valid behaviour.

Valid samples are written to generated_all_static_validator.json.
A summary of errors is printed per sample so failures are auditable.

Usage
-----
    python static_validator.py [--input <path>] [--db <path>] [--output <path>]

Defaults:
    --input   scripts/urban_map_web/generated/generated_all.json
    --db      scripts/urban_map_web/db.json
    --output  scripts/urban_map_web/generated/generated_all_static_validator.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants reflecting the tools.py implementation
# ---------------------------------------------------------------------------

# Valid tool names exposed by UrbanWebTools
VALID_TOOLS: set[str] = {
    "text_search",
    "place_details",
    "nearby_search",
    "compute_routes",
    "search_along_route",
    "read_place_website",
    "check_availability",
    "book_place",
    "get_transit_schedule",
    "search_venue_events",
    "submit_council_report",
    "transfer_to_human_agents",
}

# Required arguments per tool (positional/keyword without defaults)
REQUIRED_ARGS: dict[str, list[str]] = {
    "text_search":           ["query"],
    "place_details":         ["place_id"],
    "nearby_search":         ["lat", "lng"],
    "compute_routes":        ["origin_lat", "origin_lng", "dest_lat", "dest_lng"],
    "search_along_route":    ["polyline", "place_type"],
    "read_place_website":    ["place_id"],
    "check_availability":    ["place_id", "datetime_str", "party_size"],
    "book_place":            ["place_id", "user_id", "datetime_str", "party_size"],
    "get_transit_schedule":  ["place_id", "date_str"],
    "search_venue_events":   ["place_id", "date_str"],
    "submit_council_report": ["issue_type", "place_id", "description", "user_id"],
    "transfer_to_human_agents": ["summary"],
}

# Arguments that must be numeric (int or float)
NUMERIC_ARGS: dict[str, list[str]] = {
    "nearby_search":  ["lat", "lng"],
    "compute_routes": ["origin_lat", "origin_lng", "dest_lat", "dest_lng"],
}

# Arguments that must be strictly int
INT_ARGS: dict[str, list[str]] = {
    "nearby_search":     ["radius"],        # optional but int when present
    "check_availability": ["party_size"],
    "book_place":         ["party_size"],
}

# Arguments that must be strings
STR_ARGS: dict[str, list[str]] = {
    "text_search":           ["query"],
    "place_details":         ["place_id"],
    "nearby_search":         ["place_type"],  # optional but str when present
    "read_place_website":    ["place_id"],
    "check_availability":    ["place_id", "datetime_str"],
    "book_place":            ["place_id", "user_id", "datetime_str"],
    "get_transit_schedule":  ["place_id", "date_str"],
    "search_venue_events":   ["place_id", "date_str"],
    "submit_council_report": ["issue_type", "place_id", "description", "user_id"],
    "search_along_route":    ["polyline", "place_type"],
    "transfer_to_human_agents": ["summary"],
    "compute_routes":        ["travel_mode"],  # optional but str when present
}

# Valid issue types accepted by submit_council_report
VALID_ISSUE_TYPES: set[str] = {"pothole", "graffiti", "lighting", "waste", "other"}

# Top-level sample fields that must be present
REQUIRED_SAMPLE_FIELDS: list[str] = [
    "id", "sub_tasks", "description", "user_scenario", "evaluation_criteria"
]

# Regex patterns for date/time arguments
RE_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")
RE_DATE     = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_numeric(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _route_key(olat: float, olng: float, dlat: float, dlng: float, mode: str) -> str:
    """Build the route cache key exactly as tools.py does."""
    return f"{round(olat, 3)},{round(olng, 3)}|{round(dlat, 3)},{round(dlng, 3)}|{mode.upper()}"


def _sar_key(polyline: str, place_type: str) -> str:
    """Build the search-along-route cache key exactly as tools.py does."""
    return f"{polyline[:10]}_{place_type.lower()}"


_TASK_SUFFIX_RE = re.compile(r"_\d+$")


def _get_base_task_id(task_id: str) -> str:
    return _TASK_SUFFIX_RE.sub("", task_id)


def _build_passed_subtask_summary(all_samples: list[dict], valid_samples: list[dict]) -> list[dict[str, Any]]:
    passed_counts: Counter[str] = Counter()

    # Initialize all base task IDs found in input with 0 to ensure they appear even if none passed
    for sample in all_samples:
        if isinstance(sample, dict):
            sample_id = sample.get("id")
            if isinstance(sample_id, str) and sample_id:
                # Accessing it ensures it exists in the Counter keys
                passed_counts[_get_base_task_id(sample_id)] += 0

    # Count passed subtasks
    for sample in valid_samples:
        sample_id = sample.get("id")
        if isinstance(sample_id, str) and sample_id:
            passed_counts[_get_base_task_id(sample_id)] += 1

    return [
        {
            "task_id": task_id,
            "passed_subtasks": count,
        }
        for task_id, count in sorted(passed_counts.items())
    ]


# ---------------------------------------------------------------------------
# Per-action validator
# ---------------------------------------------------------------------------


def validate_action(action: dict, db: dict) -> list[str]:
    """Return a list of error strings for *action*. Empty list means valid."""
    errors: list[str] = []
    name = action.get("name")
    args = action.get("arguments")

    # ---- 1. tool name ----
    if not isinstance(name, str) or name not in VALID_TOOLS:
        errors.append(f"Unknown tool name: {name!r}")
        return errors  # further checks meaningless without a known tool

    # ---- 2. arguments field must be a dict ----
    if args is None:
        args = {}
    if not isinstance(args, dict):
        errors.append(f"[{name}] 'arguments' must be a dict, got {type(args).__name__}")
        return errors

    # ---- 3. required args present ----
    for req in REQUIRED_ARGS.get(name, []):
        if req not in args:
            errors.append(f"[{name}] Missing required argument: '{req}'")

    # ---- 4. numeric type checks ----
    for k in NUMERIC_ARGS.get(name, []):
        if k in args and not _is_numeric(args[k]):
            errors.append(f"[{name}] Argument '{k}' must be numeric, got {type(args[k]).__name__}: {args[k]!r}")

    # ---- 5. int type checks ----
    for k in INT_ARGS.get(name, []):
        if k in args and not isinstance(args[k], int):
            errors.append(f"[{name}] Argument '{k}' must be int, got {type(args[k]).__name__}: {args[k]!r}")

    # ---- 6. str type checks ----
    for k in STR_ARGS.get(name, []):
        if k in args and not isinstance(args[k], str):
            errors.append(f"[{name}] Argument '{k}' must be str, got {type(args[k]).__name__}: {args[k]!r}")

    # ---- 7. datetime_str / date_str format ----
    if "datetime_str" in args and isinstance(args["datetime_str"], str):
        if not RE_DATETIME.match(args["datetime_str"]):
            errors.append(f"[{name}] 'datetime_str' has invalid format (expected YYYY-MM-DD HH:MM): {args['datetime_str']!r}")
    if "date_str" in args and isinstance(args["date_str"], str):
        if not RE_DATE.match(args["date_str"]):
            errors.append(f"[{name}] 'date_str' has invalid format (expected YYYY-MM-DD): {args['date_str']!r}")

    # ---- 8. party_size must be positive ----
    if "party_size" in args and isinstance(args["party_size"], int):
        if args["party_size"] <= 0:
            errors.append(f"[{name}] 'party_size' must be a positive integer, got {args['party_size']}")

    # ---- DB-level checks (only if basic args are present and correct type) ----
    place_id = args.get("place_id")
    user_id  = args.get("user_id")

    # ---- 9. place_id must exist in DB ----
    if place_id is not None:
        if not isinstance(place_id, str) or place_id not in db["places"]:
            errors.append(f"[{name}] place_id {place_id!r} not found in UrbanDB.places")
            # Abort further place-dependent checks
            return errors

    # ---- 10. user_id must exist in DB ----
    if user_id is not None:
        if not isinstance(user_id, str) or user_id not in db["users"]:
            errors.append(f"[{name}] user_id {user_id!r} not found in UrbanDB.users")

    # ---- Tool-specific DB checks ----

    if name == "compute_routes":
        olat = args.get("origin_lat")
        olng = args.get("origin_lng")
        dlat = args.get("dest_lat")
        dlng = args.get("dest_lng")
        mode = args.get("travel_mode", "DRIVE")
        if all(_is_numeric(v) for v in [olat, olng, dlat, dlng]) and isinstance(mode, str):
            rkey = _route_key(olat, olng, dlat, dlng, mode)
            if rkey not in db["routes"]:
                errors.append(
                    f"[compute_routes] Route key {rkey!r} not cached in UrbanDB.routes"
                )

    elif name == "get_transit_schedule":
        if place_id and place_id in db["places"]:
            if place_id not in db.get("transit_schedules", {}):
                pname = db["places"][place_id].get("name", place_id)
                errors.append(
                    f"[get_transit_schedule] place_id {place_id!r} ('{pname}') "
                    f"is not a transit stop in UrbanDB.transit_schedules"
                )

    elif name == "read_place_website":
        if place_id and place_id in db["places"]:
            url = db["places"][place_id].get("website_url")
            if not url:
                pname = db["places"][place_id].get("name", place_id)
                errors.append(
                    f"[read_place_website] place_id {place_id!r} ('{pname}') "
                    f"has no website_url in UrbanDB.places"
                )
            elif url not in db.get("webpages", {}):
                errors.append(
                    f"[read_place_website] website URL {url!r} for place {place_id!r} "
                    f"is not cached in UrbanDB.webpages"
                )

    elif name == "search_along_route":
        polyline   = args.get("polyline", "")
        place_type = args.get("place_type", "")
        if isinstance(polyline, str) and isinstance(place_type, str):
            skey = _sar_key(polyline, place_type)
            if skey not in db.get("search_along_routes", {}):
                errors.append(
                    f"[search_along_route] cache key {skey!r} not found in "
                    f"UrbanDB.search_along_routes"
                )

    elif name == "submit_council_report":
        issue_type = args.get("issue_type", "")
        if isinstance(issue_type, str) and issue_type.lower() not in VALID_ISSUE_TYPES:
            errors.append(
                f"[submit_council_report] issue_type {issue_type!r} is not valid. "
                f"Must be one of: {sorted(VALID_ISSUE_TYPES)}"
            )

    return errors


# ---------------------------------------------------------------------------
# Per-sample validator
# ---------------------------------------------------------------------------


def validate_sample(sample: dict, db: dict) -> list[str]:
    """Return a list of all errors found in *sample*. Empty list means valid."""
    errors: list[str] = []

    # ---- 1. required top-level fields ----
    for field in REQUIRED_SAMPLE_FIELDS:
        if field not in sample:
            errors.append(f"Missing top-level field: '{field}'")

    # ---- 2. evaluation_criteria.actions must be a non-empty list ----
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

    # ---- 3. each action ----
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
        description="Static validator for generated urban_map_web benchmark scenarios."
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input",
        default=str(script_dir / "generated_newyork_1" / "generated_all.json"),
        help="Path to the generated scenarios JSON file",
    )
    parser.add_argument(
        "--db",
        default=str(script_dir / "db_newyork.json"),
        help="Path to the UrbanDB JSON file",
    )
    parser.add_argument(
        "--output",
        default=str(script_dir / "generated_newyork_1" / "generated_all_static_validator.json"),
        help="Path to write the cleaned (valid) output JSON file",
    )
    parser.add_argument(
        "--error-output",
        default=str(script_dir / "generated_newyork_1" / "generated_all_static_validator_error.json"),
        help="Path to write the invalid samples with their errors",
    )
    args = parser.parse_args()

    # --- load files ---
    input_path  = Path(args.input)
    db_path     = Path(args.db)
    output_path = Path(args.output)
    error_path  = Path(args.error_output)

    print(f"Loading input  : {input_path}")
    print(f"Loading DB     : {db_path}")
    print(f"Output path    : {output_path}")
    print(f"Error path     : {error_path}")
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

    total   = len(raw_data)
    valid_samples:   list[dict] = []
    invalid_samples: list[dict] = []
    invalid_count = 0

    print(f"Validating {total} samples...\n")
    print("=" * 70)

    for sample in raw_data:
        if not isinstance(sample, dict):
            print(f"[SKIP] Non-dict entry; skipping.")
            invalid_count += 1
            continue

        sample_id = sample.get("id", "<unknown>")
        errors    = validate_sample(sample, db)

        if errors:
            invalid_count += 1
            print(f"[FAIL] {sample_id}")
            for err in errors:
                print(f"       - {err}")
            # Store the original sample plus a _validation_errors field
            error_record = dict(sample)
            error_record["_validation_errors"] = errors
            invalid_samples.append(error_record)
        else:
            valid_samples.append(sample)
            print(f"[PASS] {sample_id}")

    print("=" * 70)
    print(f"\nResults: {len(valid_samples)} valid / {invalid_count} invalid / {total} total")
    print(f"Pass rate: {len(valid_samples)/total*100:.1f}%\n")

    # --- write valid output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(valid_samples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(valid_samples)} valid samples to  : {output_path}")

    # --- write error output ---
    error_path.parent.mkdir(parents=True, exist_ok=True)
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(invalid_samples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(invalid_samples)} invalid samples to: {error_path}")

    passed_summary = _build_passed_subtask_summary(raw_data, valid_samples)

    print("\nPassed sub-task summary by base task_id")
    print("-" * 72)
    print(f"{'Task ID':<56} {'Passed'}")
    print("-" * 72)
    for row in passed_summary:
        print(f"{row['task_id']:<56} {row['passed_subtasks']:>6}")
    print("-" * 72)


if __name__ == "__main__":
    main()
