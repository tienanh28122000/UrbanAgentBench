import math
import base64
import re
import json
from functools import lru_cache
from pathlib import Path
from litellm import completion
from tau2.utils.utils import DATA_DIR
from tau2.config import DEFAULT_LLM_VLM, DEFAULT_LLM_VLM_TEMPERATURE
import litellm

# Path Definitions
URBAN_SATELLITE_DATA_DIR = DATA_DIR / "tau2" / "domains" / "urban_satellite"
URBAN_SATELLITE_DB_PATH = URBAN_SATELLITE_DATA_DIR / "db.json"
URBAN_SATELLITE_POLICY_PATH = URBAN_SATELLITE_DATA_DIR / "policy.md"
URBAN_SATELLITE_TASK_SET_PATH = URBAN_SATELLITE_DATA_DIR / "tasks.json"
URBAN_SATELLITE_IMAGE_PATH = URBAN_SATELLITE_DATA_DIR / "satellite_imgs"
URBAN_SATELLITE_FEW_SHOT_PATH = URBAN_SATELLITE_DATA_DIR / "few_shot_examples.json"

# Feature flag for perception few-shot prompting. Keep False by default.
FEW_SHOT = False

def deg2num(lat_deg: float, lon_deg: float, zoom: int = 15) -> tuple[int, int]:
    """Converts decimal degrees to OpenStreetMap tile coordinates."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def _encode_image(image_path: str) -> str:
    # If the file doesn't exist during simulation (mock phase), return a dummy string
    raw_image_path = image_path
    if not Path(image_path).exists():
        image_path = f"{URBAN_SATELLITE_IMAGE_PATH}/{raw_image_path}"
    if not Path(image_path).exists():
        print(f"Warning: Image file {image_path} not found. Returning mock base64 string.")
        return "mock_base64_string"
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@lru_cache(maxsize=1)
def _load_few_shot_sites() -> dict:
    try:
        with open(URBAN_SATELLITE_FEW_SHOT_PATH, "r") as f:
            data = json.load(f)
        return data.get("sites", {})
    except Exception as e:
        print(f"Warning: Failed to load few-shot examples: {e}")
        return {}


def _resolve_example_image_path(raw_path: str) -> str:
    p = Path(raw_path)
    if p.exists():
        return str(p)
    candidate = URBAN_SATELLITE_DATA_DIR / raw_path
    if candidate.exists():
        return str(candidate)
    return raw_path


def _get_site_example(site_id: str) -> dict | None:
    return _load_few_shot_sites().get(site_id)


def _append_example_image_with_text(content: list, text: str, image_path: str) -> None:
    content.append({"type": "text", "text": text})
    encoded = _encode_image(image_path)
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})


def _first_n_site_ids(n: int) -> list:
    return list(_load_few_shot_sites().keys())[:n]


def _pick_density_examples() -> list[tuple[str, str]]:
    # One example per density bracket: low (0-3.3), medium (3.3-6.6), high (>6.6).
    brackets = {"low": None, "medium": None, "high": None}
    for sid, site in _load_few_shot_sites().items():
        score = site.get("ground_truth", {}).get("population_density")
        if score is None:
            continue
        if score < 3.3 and brackets["low"] is None:
            brackets["low"] = sid
        elif 3.3 <= score <= 6.6 and brackets["medium"] is None:
            brackets["medium"] = sid
        elif score > 6.6 and brackets["high"] is None:
            brackets["high"] = sid
    return [(label, sid) for label, sid in brackets.items() if sid is not None]


def _pick_land_use_examples() -> list[tuple[str, str]]:
    # One example per OSM land-use category combination.
    VALID_CATEGORIES = ["landuse", "natural"]
    seen_combos: set[str] = set()
    picks = []
    for sid, site in _load_few_shot_sites().items():
        cats = site.get("ground_truth", {}).get("land_use_categories", [])
        valid = [c for c in cats if c in VALID_CATEGORIES]
        combo_key = ",".join(sorted(valid))
        if combo_key and combo_key not in seen_combos:
            seen_combos.add(combo_key)
            picks.append((combo_key, sid))
    return picks


def _pick_infra_examples(limit: int = 3) -> list[tuple[str, str, bool, str, bool]]:
    """Pick few-shot examples for infrastructure presence detection (yes/no)."""
    EASY_INFRA = [
        "Airport", "Stadium", "Golf Field", "Harbor",
        "Ground Track Field", "Soccer Ball Field", "Train Station",
    ]
    picks = []
    ordered_ids = list(_load_few_shot_sites().keys())
    for sid in ordered_ids:
        counts = _load_few_shot_sites().get(sid, {}).get("ground_truth", {}).get("infrastructure_counts", {})
        if not isinstance(counts, dict):
            continue
        easy = {k: v for k, v in counts.items() if k in EASY_INFRA}
        absent = next(((k, v) for k, v in easy.items() if v == 0), None)
        present = next(((k, v) for k, v in easy.items() if isinstance(v, int) and v > 0), None)
        if absent and present:
            picks.append((sid, absent[0], False, present[0], True))
        if len(picks) >= limit:
            break
    return picks


def calculate_density(image_path: str, few_shot: bool = FEW_SHOT) -> float:
    encoded_string = _encode_image(image_path)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a population density estimation tool for satellite imagery. "
                "Please analyze the population density of the image shown comparing to all cities around the world. "
                "Rate the population density in a degree from 0.0 to 9.9, "
                "where higher rating represents higher population density. "
                "Output ONLY a single float number between 0.0 and 9.9. No text, no explanation."
            ),
        }
    ]

    if few_shot:
        content = []
        for label, sid in _pick_density_examples():
            site = _get_site_example(sid)
            if not site:
                continue
            gt = site.get("ground_truth", {})
            ex_path = site.get("meta", {}).get("current_image", "")
            score = float(gt.get("population_density", 0.0))
            ex_text = f"Example ({sid}): expected population density score={score:.1f}."
            _append_example_image_with_text(content, ex_text, ex_path)

        content.append({
            "type": "text",
            "text": "Now analyze the target image and output ONLY the population density score as a single float.",
        })
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}})
        messages.append({"role": "user", "content": content})
        temperature = DEFAULT_LLM_VLM_TEMPERATURE
    else:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Rate the population density of this satellite image from 0.0 to 9.9."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}},
            ],
        })
        temperature = 0.1

    response = completion(
        model=DEFAULT_LLM_VLM,
        messages=messages,
        temperature=temperature,
    )

    try:
        raw_output = response.choices[0].message.content.strip()
        # print(f"usage: {response.usage}")
        match = re.search(r"[-+]?\d*\.\d+|\d+", raw_output)
        score = float(match.group()) if match else 0.0
        return float(min(max(score, 0.0), 9.9))
    except Exception as e:
        print(f"Error parsing LLM response for density calculation. Exception: {e}. Raw output: {response.choices[0].message.content.strip()}")
        return 0.0




def classify_land_use_vlm(image_path: str, few_shot: bool = FEW_SHOT) -> list[str]:
    encoded_string = _encode_image(image_path)
    valid_categories = ["landuse", "natural"]

    try:
        messages = [{
            "role": "system",
            "content": (
                "You are a land-use classification tool for satellite imagery. "
                "Identify which of the following OpenStreetMap land-use categories are PRESENT in the image: "
                f"{', '.join(valid_categories)}. "
                "A site may have multiple categories. "
                "'landuse' covers built-up or managed land (residential, commercial, industrial, farmland). "
                "'natural' covers natural features (vegetation, water, bare ground, forests). "
                "Output ONLY a JSON object with key 'categories' containing a list of matching category strings. "
                "Example: {\"categories\": [\"landuse\", \"natural\"]}."
            ),
        }]

        if few_shot:
            content = []
            for combo_key, sid in _pick_land_use_examples():
                site = _get_site_example(sid)
                if not site:
                    continue
                ex_path = site.get("meta", {}).get("current_image", "")
                ex_text = f"Example ({sid}): present land-use categories are [{combo_key}]."
                _append_example_image_with_text(content, ex_text, ex_path)

            content.append({"type": "text", "text": "Now classify the target image. Output ONLY JSON with key 'categories'."})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": [
                {"type": "text", "text": "Which of [landuse, natural] categories are present in this satellite image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
            ]})

        response = completion(
            model=DEFAULT_LLM_VLM,
            response_format={"type": "json_object"},
            messages=messages,
            temperature=DEFAULT_LLM_VLM_TEMPERATURE
        )
        # print(f"usage: {response.usage}")
        raw_output = response.choices[0].message.content.strip()
        parsed = json.loads(raw_output)
        cats = parsed.get("categories", [])
        return [c for c in cats if c in valid_categories]
    except Exception as e:
        print(f"Error during land use classification. Exception: {e}")
        return []


def detect_infrastructure_vlm(image_path: str, feature_query: str, few_shot: bool = FEW_SHOT) -> bool:
    encoded_string = _encode_image(image_path)
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an infrastructure presence detector for satellite imagery. "
                    "Given a feature query, determine whether that infrastructure element is PRESENT or ABSENT in the image. "
                    "Output ONLY 'True' if the feature is visible, or 'False' if it is not. No explanation."
                ),
            }
        ]

        if few_shot:
            content = []
            for sid, absent_feat, absent_val, present_feat, present_val in _pick_infra_examples(limit=3):
                site = _get_site_example(sid)
                if not site:
                    continue
                ex_path = site.get("meta", {}).get("current_image", "")
                _append_example_image_with_text(
                    content,
                    f"Example ({sid}) A: query='{absent_feat}', expected=False (not present).",
                    ex_path,
                )
                _append_example_image_with_text(
                    content,
                    f"Example ({sid}) B: query='{present_feat}', expected=True (present).",
                    ex_path,
                )

            content.append({"type": "text", "text": f"Now detect feature '{feature_query}' in the target image. Output ONLY True or False."})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": [
                {"type": "text", "text": f"Is there a '{feature_query}' visible in this satellite image? Output ONLY True or False."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
            ]})

        response = completion(
            model=DEFAULT_LLM_VLM,
            messages=messages,
            temperature=DEFAULT_LLM_VLM_TEMPERATURE
        )
        # print(f"usage: {response.usage}")
    except Exception as e:
        print(f"Error during infrastructure detection. Exception: {e}")
        return False
    raw_output = response.choices[0].message.content.strip().lower()
    return "true" in raw_output