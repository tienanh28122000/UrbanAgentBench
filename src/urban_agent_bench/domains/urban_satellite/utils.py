import math
import base64
import re
import json
from functools import lru_cache
from pathlib import Path
from litellm import completion
from urban_agent_bench.utils.utils import DATA_DIR
from urban_agent_bench.config import DEFAULT_LLM_VLM, DEFAULT_LLM_VLM_TEMPERATURE
import litellm

# Path Definitions
URBAN_SATELLITE_DATA_DIR = DATA_DIR / "domains" / "urban_satellite"
URBAN_SATELLITE_DB_PATH = URBAN_SATELLITE_DATA_DIR / "db.json"
URBAN_SATELLITE_POLICY_PATH = URBAN_SATELLITE_DATA_DIR / "policy.md"
URBAN_SATELLITE_TASK_SET_PATH = URBAN_SATELLITE_DATA_DIR / "tasks.json"
URBAN_SATELLITE_IMAGE_PATH = URBAN_SATELLITE_DATA_DIR / "satellite_imgs"
URBAN_SATELLITE_IMAGE_ZOOM_17_PATH = URBAN_SATELLITE_DATA_DIR / "satellite_imgs_zoom17"
URBAN_SATELLITE_PAST_IMAGE_PATH = URBAN_SATELLITE_DATA_DIR / "satellite_imgs_past"
URBAN_SATELLITE_FEW_SHOT_PATH = URBAN_SATELLITE_DATA_DIR / "few_shot_examples.json"

# Feature flag for perception few-shot prompting. Keep False by default.
FEW_SHOT = True

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
        image_path = f"{URBAN_SATELLITE_PAST_IMAGE_PATH}/{raw_image_path}"
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
    # One example per category: Sparse, Moderate, Dense.
    target = {"Sparse": None, "Moderate": None, "Dense": None}
    for sid, site in _load_few_shot_sites().items():
        cat = site.get("ground_truth", {}).get("density_classification")
        if cat in target and target[cat] is None:
            target[cat] = sid
    return [(cat, sid) for cat, sid in target.items() if sid is not None]


def _pick_green_examples() -> list[tuple[str, str]]:
    # One example per category: Low, Medium, High.
    target = {"Low": None, "Medium": None, "High": None}
    for sid, site in _load_few_shot_sites().items():
        cat = site.get("ground_truth", {}).get("green_ratio_classification")
        if cat in target and target[cat] is None:
            target[cat] = sid
    return [(cat, sid) for cat, sid in target.items() if sid is not None]


def _pick_land_use_examples() -> list[tuple[str, str]]:
    # One example per land-use category.
    categories = [
        "High-Density Urban",
        "Suburban Residential",
        "Industrial/Commercial",
        "Rural/Agricultural",
        "Natural/Wildland",
    ]
    target = {c: None for c in categories}
    for sid, site in _load_few_shot_sites().items():
        cat = site.get("ground_truth", {}).get("land_use")
        if cat in target and target[cat] is None:
            target[cat] = sid
    return [(cat, sid) for cat, sid in target.items() if sid is not None]


def _pick_count_examples(limit: int = 3) -> list[tuple[str, str, int, str, int]]:
    # Prefer earliest examples while ensuring one zero and one non-zero feature per example.
    picks = []
    ordered_ids = list(_load_few_shot_sites().keys())
    for sid in ordered_ids:
        counts = _load_few_shot_sites().get(sid, {}).get("ground_truth", {}).get("infrastructure_counts", {})
        if not isinstance(counts, dict) or not counts:
            continue
        zero_item = next(((k, v) for k, v in counts.items() if v == 0), None)
        nonzero_item = next(((k, v) for k, v in counts.items() if isinstance(v, int) and v > 0), None)
        if zero_item and nonzero_item:
            picks.append((sid, zero_item[0], int(zero_item[1]), nonzero_item[0], int(nonzero_item[1])))
        if len(picks) >= limit:
            break
    return picks


def _pick_carbon_examples(limit: int = 3) -> list[str]:
    return _first_n_site_ids(limit)


def calculate_density(image_path: str, few_shot: bool = FEW_SHOT) -> dict:
    encoded_string = _encode_image(image_path)
    messages = [
        {
            "role": "system",
            "content": """You are a precise urban analysis tool.
Follow these steps:
1. Classify the building/population density into a 'category': Sparse (0-3), Moderate (3-7), or Dense (>7).
2. Assign a precise 'score' (float between 0.0 and 10.0) based on your classification.
Output exactly in JSON format: {\"category\": \"...\", \"score\": ...}""",
        }
    ]

    if few_shot:
        content = []
        for cat, sid in _pick_density_examples():
            site = _get_site_example(sid)
            if not site:
                continue
            gt = site.get("ground_truth", {})
            ex_path = site.get("meta", {}).get("current_image", "")
            ex_text = (
                f"Example ({sid}): expected category={cat}, "
                f"score={float(gt.get('density_score', 0.0)):.2f}."
            )
            _append_example_image_with_text(content, ex_text, ex_path)

        content.append({
            "type": "text",
            "text": "Now analyze the target image and return only JSON with keys category and score.",
        })
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}})
        messages.append({"role": "user", "content": content})
        temperature = DEFAULT_LLM_VLM_TEMPERATURE
    else:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this satellite image and return the density category and score."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}},
            ],
        })
        temperature = 0.1

    response = completion(
        model=DEFAULT_LLM_VLM,
        response_format={"type": "json_object"},
        messages=messages,
        temperature=temperature,
    )
    
    try:
        raw_output = response.choices[0].message.content.strip()
        result = json.loads(raw_output)
        print(f"usage: {response.usage}")
        return result
    except Exception as e:
        print(f"Error parsing LLM response for density calculation. Exception: {e}. Raw output: {response.choices[0].message.content.strip()}")
        return {"category": "Unknown", "score": 0.0}


def calculate_green_ratio(image_path: str, few_shot: bool = FEW_SHOT) -> dict:
    encoded_string = _encode_image(image_path)
    messages = [
        {
            "role": "system",
            "content": """You are an environmental analysis tool.
Follow these steps:
1. Classify the vegetation ratio into a 'category': Low (<0.3), Medium (0.3-0.7), or High (>0.7).
2. Assign a precise 'score' (float between 0.0 and 1.0) representing the vegetation ratio.
Output exactly in JSON format: {\"category\": \"...\", \"score\": ...}""",
        }
    ]

    if few_shot:
        content = []
        for cat, sid in _pick_green_examples():
            site = _get_site_example(sid)
            if not site:
                continue
            gt = site.get("ground_truth", {})
            ex_path = site.get("meta", {}).get("current_image", "")
            ex_text = (
                f"Example ({sid}): expected category={cat}, "
                f"score={float(gt.get('green_ratio', 0.0)):.2f}."
            )
            _append_example_image_with_text(content, ex_text, ex_path)

        content.append({
            "type": "text",
            "text": "Now analyze the target image and return only JSON with keys category and score.",
        })
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}})
        messages.append({"role": "user", "content": content})
        temperature = DEFAULT_LLM_VLM_TEMPERATURE
    else:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this satellite image and return the green ratio category and score."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}},
            ],
        })
        temperature = 0.1

    response = completion(
        model=DEFAULT_LLM_VLM,
        response_format={"type": "json_object"},
        messages=messages,
        temperature=temperature,
    )
    
    try:
        raw_output = response.choices[0].message.content.strip()
        result = json.loads(raw_output)
        print(f"usage: {response.usage}")
        return result
    except Exception as e:
        print(f"Error parsing LLM response for green ratio calculation. Exception: {e}. Raw output: {response.choices[0].message.content.strip()}")
        return {"category": "Unknown", "score": 0.0}


def classify_land_use_vlm(image_path: str, few_shot: bool = FEW_SHOT) -> str:
    encoded_string = _encode_image(image_path)
    valid_categories = ["High-Density Urban", "Suburban Residential", "Industrial/Commercial", "Rural/Agricultural", "Natural/Wildland"]
    
    try:
        messages = [{
            "role": "system",
            "content": f"Classify the image into exactly one of: {', '.join(valid_categories)}. Output ONLY the category name.",
        }]

        if few_shot:
            content = []
            for cat, sid in _pick_land_use_examples():
                site = _get_site_example(sid)
                if not site:
                    continue
                ex_path = site.get("meta", {}).get("current_image", "")
                ex_text = f"Example ({sid}): expected land-use category is {cat}."
                _append_example_image_with_text(content, ex_text, ex_path)

            content.append({"type": "text", "text": "Now classify the target image. Output ONLY one category name."})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": [
                {"type": "text", "text": "Categorize this site."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
            ]})

        response = completion(
            model=DEFAULT_LLM_VLM,
            messages=messages,
            temperature=DEFAULT_LLM_VLM_TEMPERATURE
        )
        print(f"usage: {response.usage}")
    except Exception as e:
        print(f"Error during land use classification. Exception: {e}")
        return "Unclassified"
    raw_output = response.choices[0].message.content.strip()
    for cat in valid_categories:
        if cat.lower() in raw_output.lower(): return cat
    return "Unclassified"


def detect_infrastructure_vlm(image_path: str, feature_query: str, few_shot: bool = FEW_SHOT) -> int:
    image_path = image_path.split("/")[-1]
    image_path = f"{URBAN_SATELLITE_IMAGE_ZOOM_17_PATH}/{image_path}"
    if Path(image_path).exists():
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    else:
        print(f"Warning: Image file {image_path} not found for detecting infrastructure. Returning mock base64 string.")
        encoded_string = "mock_base64_string"
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a feature counter. Output ONLY an integer representing the count of the requested feature.",
            }
        ]

        if few_shot:
            content = []
            for sid, zero_feat, zero_val, nonzero_feat, nonzero_val in _pick_count_examples(limit=3):
                site = _get_site_example(sid)
                if not site:
                    continue
                ex_path = site.get("meta", {}).get("current_image", "")
                ex_path = f"{URBAN_SATELLITE_IMAGE_ZOOM_17_PATH}/{ex_path}"
                if not Path(ex_path).exists():
                    print(f"Warning: Image file {ex_path} not found for detecting infrastructure. Returning mock base64 string.")

                _append_example_image_with_text(
                    content,
                    f"Example ({sid}) A: query='{zero_feat}', expected count={zero_val}.",
                    ex_path,
                )
                _append_example_image_with_text(
                    content,
                    f"Example ({sid}) B: query='{nonzero_feat}', expected count={nonzero_val}.",
                    ex_path,
                )

            content.append({"type": "text", "text": f"Now count feature '{feature_query}' in the target image. Output ONLY an integer."})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": [
                {"type": "text", "text": f"How many {feature_query} are visible in this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
            ]})

        response = completion(
            model=DEFAULT_LLM_VLM,
            messages=messages,
            temperature=DEFAULT_LLM_VLM_TEMPERATURE
        )
        print(f"usage: {response.usage}")    
    except Exception as e:
        print(f"Error during infrastructure detection. Exception: {e}")
        return 0
    raw_output = response.choices[0].message.content.strip()
    match = re.search(r"\d+", raw_output)
    return int(match.group()) if match else 0

def verify_path_vlm(image_path: str, start_coord: str, end_coord: str) -> bool:
    encoded_string = _encode_image(image_path)
    try:
        response = completion(
            model=DEFAULT_LLM_VLM,
            messages=[
                {"role": "system", "content": "You are a routing verifier. Output ONLY 'True' or 'False'."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Is there a continuous physical path between {start_coord} and {end_coord} in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
                ]}
            ],
            temperature=DEFAULT_LLM_VLM_TEMPERATURE
        )
        print(f"usage: {response.usage}")
    except Exception as e:
        print(f"Error during path verification. Exception: {e}")
        return False
    return "true" in response.choices[0].message.content.strip().lower()

def compare_temporal_vlm(image_path_1: str, image_path_2: str) -> str:
    enc1 = _encode_image(image_path_1)
    enc2 = _encode_image(image_path_2)
    response = completion(
        model=DEFAULT_LLM_VLM,
        messages=[
            {"role": "system", "content": "You are a temporal change analyst. Briefly describe the structural or environmental differences between the two images."},
            {"role": "user", "content": [
                {"type": "text", "text": "Compare these two images of the same location from different times. What has changed?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc1}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc2}"}}
            ]}
        ],
        temperature=DEFAULT_LLM_VLM_TEMPERATURE
    )
    print(f"usage: {response.usage}")
    return response.choices[0].message.content.strip()


def estimate_carbon_vlm(image_path: str, few_shot: bool = FEW_SHOT) -> float:
    encoded_string = _encode_image(image_path)
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a carbon emission estimation tool. Based on visible infrastructure, industrial activity, vegetation coverage, and urban density in satellite imagery, estimate the annual carbon emission index. Output ONLY a float value. Scale: 0.0 (pristine nature/dense forest) to 2000.0 (heavy industrial zone). No text.",
            }
        ]

        if few_shot:
            content = []
            for sid in _pick_carbon_examples(limit=3):
                site = _get_site_example(sid)
                if not site:
                    continue
                gt = site.get("ground_truth", {})
                ex_path = site.get("meta", {}).get("current_image", "")
                ex_text = f"Example ({sid}): expected carbon emission index={float(gt.get('carbon_emission', 0.0)):.3f}."
                _append_example_image_with_text(content, ex_text, ex_path)

            content.append({"type": "text", "text": "Now estimate the carbon emission index for the target image. Output ONLY a float."})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": [
                {"type": "text", "text": "Estimate the annual carbon emission index for this area."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
            ]})

        response = completion(
            model=DEFAULT_LLM_VLM,
            messages=messages,
            temperature=DEFAULT_LLM_VLM_TEMPERATURE
        )
        print(f"usage: {response.usage}")
    except Exception as e:
        print(f"Error during carbon emission estimation. Exception: {e}")
        return 0.0
    raw_output = response.choices[0].message.content.strip()
    match = re.search(r"[-+]?\d*\.\d+|\d+", raw_output)
    return float(match.group()) if match else 0.0