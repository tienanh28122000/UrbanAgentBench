import os
import csv
import json
import random
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# ==========================================
# CONFIG
# ==========================================
# Number of samples to generate per task category per run.
# Total scenarios produced = NUM_GEN × number_of_task_categories
NUM_GEN = 12

# Restrict generation to specific task names, or set to None to generate all.
# Example: SELECTED_TASKS = ["Detection", "Suitability"]
SELECTED_TASKS = None
# SELECTED_TASKS = ["Detection"]

# When True, each generation round (after the first) receives the previous
# round's scenario summary so the LLM actively avoids repeating it.
ANTI_REPETITION = True

# When True, the LLM is instructed to generate harder, multi-step scenarios.
INCREASE_COMPLEXITY = False

# Maximum number of NL assertions to generate per scenario.
MAX_NL_ASSERTIONS = 4

# Multi-turn linkage quality controls.
LINKAGE_SCORE_THRESHOLD = 0.55
STRICT_MULTI_TURN_MODE = True

# When True, each site is used at most once per task across all generation rounds.
# If the eligible pool is exhausted the full pool is used as fallback (no stall).
# Set to False to restore the original random-with-replacement behaviour.
DEDUP = True

# Per-difficulty thresholds for linkage scoring and validation.
# easy:   short, single-branch conversations (3-4 actions)
# medium: standard multi-turn with 2 conditional pivots (5-6 actions)
# hard:   deep multi-signal conversations (7-8+ actions)
_DIFFICULTY_TIERS: dict[str, dict] = {
    "easy":   {"linkage_threshold": 0.28, "min_cues": 1, "chain_divisor": 4.0},
    "medium": {"linkage_threshold": 0.50, "min_cues": 2, "chain_divisor": 6.0},
    "hard":   {"linkage_threshold": 0.65, "min_cues": 3, "chain_divisor": 8.0},
}

# Keep only core tasks after taxonomy refactor.
_ACTIVE_TASKS = {
    "Density Comparison",
    "Land Use Verification",
    "Infrastructure Detection",
    "Suitability",
    "Encroachment",
    "Urban Profile",
}

# Initialize OpenAI Client
load_dotenv()
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.environ.get("OPENROUTER_API_KEY"),
)
MODEL = "openai/gpt-5.4-mini"  # Use the mini variant for faster generation; switch to "gpt-4.1" for higher quality

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NAME_DATASET_PATH = os.path.join(
    os.path.dirname(SCRIPT_DIR), "urban_map_web", "name_dataset.csv"
)

# Short slugs used in generated scenario IDs
_TASK_SLUGS = {
    "Density Comparison":       "density_comparison",
    "Land Use Verification":    "land_use_verification",
    "Infrastructure Detection": "infra_detection",
    "Suitability":              "suitability",
    "Encroachment":             "encroachment",
    "Urban Profile":            "urban_profile",
}

# These rules are appended verbatim to every generated sample's task_instructions
# in post-processing, so the user simulator always has them regardless of what
# the LLM chose to include.
_SIMULATOR_CRITICAL_RULES = (
    "\nCRITICAL RULES FOR YOU:"
    "\n- You MUST follow the Steps strictly based on the Agent's LAST response."
    " Do NOT assume the agent succeeded — read what they actually said."
    "\n- IF the Agent gives wrong information or is unhelpful:"
    " Give them ONE chance to correct it by rephrasing your request or pointing out the issue naturally."
    " If they fail again on the retry, express dissatisfaction and end the conversation"
    " (e.g., 'Oh, that's too bad. I'll find another place. Bye.')."
    "\n- IF the Agent explicitly refuses to help (e.g. says it cannot do that):"
    " Do NOT retry. Express dissatisfaction naturally and end the conversation immediately."
    "\n- IF the Agent asks for clarification:"
    " Answer using ONLY your known_info."
    " If you don't know, say 'I don't have that information.'"
)

# ==========================================
# TOOL DEFINITIONS (for the LLM prompt)
# ==========================================
_TOOL_DEFINITIONS = [
    {
        "name": "get_satellite_tile",
        "type": "READ",
        "description": "Fetches the satellite image file path for specific coordinates.",
        "arguments": {"lat": "float", "lon": "float"},
        "returns": "str (file path to the satellite image)"
    },
    {
        "name": "classify_land_use",
        "type": "READ",
        "description": "Identifies which OpenStreetMap land-use categories are present at the site. A site may have multiple categories simultaneously.",
        "arguments": {"image_path": "str"},
        "returns": "list[str] — subset of ['landuse', 'natural', 'leisure']. 'landuse'=built-up/managed land; 'natural'=vegetation/water/bare ground; 'leisure'=parks/sports/recreation."
    },
    {
        "name": "analyze_urban_density",
        "type": "READ",
        "description": "Estimates the population density of the area from the satellite image, calibrated against cities worldwide.",
        "arguments": {"image_path": "str"},
        "returns": "float — population density score from 0.0 (uninhabited) to 9.9 (extremely dense). This is an approximate VLM estimate; do NOT assert exact values in nl_assertions."
    },
    {
        "name": "detect_infrastructure",
        "type": "READ",
        "description": "Determines whether a specific infrastructure element is present or absent in the satellite image.",
        "arguments": {"image_path": "str", "feature_query": "str"},
        "returns": "bool — True if the feature is visibly present, False if absent. Supported types: Bridge, Stadium, Train Station, Golf Field, Soccer Ball Field, Swimming Pool, Tennis Court, Roundabout, Basketball Court, Ground Track Field, Baseball Field, Overpass, Storage Tank, Windmill."
    },
    {
        "name": "submit_site_assessment",
        "type": "WRITE",
        "description": "Submits the agent's final assessment decision with justification. Requires explicit user confirmation (yes) before calling.",
        "arguments": {"site_id": "str", "decision": "str", "justification": "str"},
        "returns": "dict (confirmation receipt)"
    },
    {
        "name": "transfer_to_human_agents",
        "type": "GENERIC",
        "description": "Transfers the user to a human agent with a summary. Only use when the request is out of scope.",
        "arguments": {"summary": "str"},
        "returns": "str"
    }
]


# ==========================================
# VARIATION AXES
# ==========================================
# Each entry defines a distinct user angle (persona + constraint + request style).
# One entry is picked per generation round to ensure diverse scenarios.
_VARIATION_AXES: dict[str, list[dict]] = {
    "Density Comparison": [
        {"persona": "a city housing analyst selecting the most densely populated zone for urban renewal",
         "constraint": "needs to rank sites by population density score and identify the highest-density location",
         "style": "data-driven, asks for density on each site before drawing conclusions"},
        {"persona": "a disaster relief coordinator prioritising which area needs the most resources",
         "constraint": "wants to know which of three locations has the highest population concentration",
         "style": "urgent, asks density one site at a time, asks for a recommendation at the end"},
        {"persona": "a public transport planner deciding where to add a new bus route",
         "constraint": "needs the densest site among candidates to maximise ridership potential",
         "style": "methodical, checks density then land-use on the winner before committing"},
        {"persona": "a property developer comparing development feasibility across candidate plots",
         "constraint": "wants to avoid overdeveloped areas — needs the site with the lowest density score",
         "style": "business-focused, compares all sites before asking for a recommendation"},
        {"persona": "a school placement officer finding the most populated neighbourhood",
         "constraint": "must identify the highest-density location among three sites to build a new school",
         "style": "careful, cross-checks density with land-use before finalising"},
        {"persona": "a NGO field coordinator assessing where to deploy mobile health clinics",
         "constraint": "needs the highest-density site among four candidate areas",
         "style": "compassionate, asks density site by site, asks follow-up about land-use on the winner"},
        {"persona": "a telecom engineer planning cell tower placement for maximum coverage",
         "constraint": "wants the two densest sites from three candidates to co-locate towers",
         "style": "technical, ranks all three then verifies land-use on the top two"},
        {"persona": "a retail chain executive choosing between two candidate stores by footfall potential",
         "constraint": "higher density means higher footfall — needs to know which site is denser",
         "style": "fast-paced, asks density on both and requests a direct recommendation"},
        {"persona": "a green-space advocate comparing two sites to identify the more built-up location",
         "constraint": "needs the higher-density site to advocate for park creation there",
         "style": "socially motivated, checks density then asks about land-use categories"},
        {"persona": "a census analyst verifying population concentration across London tiles",
         "constraint": "needs to rank four sites by density score for a statistical report",
         "style": "systematic, checks density for each tile in turn, requests the ranking at the end"},
    ],
    "Land Use Verification": [
        {"persona": "a zoning officer verifying whether a site has active land-use built-up areas",
         "constraint": "needs to confirm 'landuse' is in the site's categories before approving a permit",
         "style": "regulatory, asks for the category list then checks if 'landuse' is present"},
        {"persona": "a conservationist checking that a protected zone still shows natural features",
         "constraint": "needs 'natural' to appear in the categories to confirm habitat integrity",
         "style": "concerned, asks for land-use first then follows up with a density check"},
        {"persona": "a parks department planner verifying recreational land is properly categorised",
         "constraint": "needs 'leisure' in the categories to confirm recreational use classification",
         "style": "friendly, asks for categories then asks about density to assess visitor load"},
        {"persona": "a developer checking site suitability before acquiring land",
         "constraint": "needs to know all categories present; if 'natural' dominates, acquisition is off",
         "style": "direct, asks for the full list then decides based on presence/absence of 'natural'"},
        {"persona": "a GIS analyst reconciling satellite imagery with official land registry records",
         "constraint": "needs to verify whether 'landuse' and 'leisure' are both present at the site",
         "style": "technical, asks for categories and cross-checks with density score"},
        {"persona": "a tourism board officer identifying sites with leisure and natural attributes",
         "constraint": "needs both 'natural' and 'leisure' to be present for a scenic-trail inclusion",
         "style": "enthusiastic, asks for categories on each site then compares the two"},
        {"persona": "a flood-risk assessor checking whether a site is predominantly natural ground",
         "constraint": "needs 'natural' in the categories as a proxy for permeable surfaces",
         "style": "serious, asks for categories first, then density if 'natural' is present"},
        {"persona": "a local authority officer comparing two candidate sites for a community centre",
         "constraint": "prefers the site with 'landuse' and 'leisure' but not 'natural' (avoid displacing nature)",
         "style": "measured, checks both sites categories before making a recommendation"},
        {"persona": "a real-estate lawyer performing due diligence on a mixed-use acquisition",
         "constraint": "needs to document all land-use categories present before signing contracts",
         "style": "formal, asks for all categories then asks density for each site"},
        {"persona": "an urban ecologist verifying that a corridor still contains natural land cover",
         "constraint": "needs 'natural' in the categories list; if absent, reports a corridor break",
         "style": "scientific, asks for categories and then density to contextualise the finding"},
    ],
    "Infrastructure Detection": [
        # ---- PRESENCE-FOCUSED (user expects / wants feature to be there) ----
        {"persona": "a sports event coordinator checking if a site has a stadium",
         "constraint": "needs to confirm stadium presence before booking the venue",
         "style": "direct, asks about stadium first then checks one more feature for context"},
        {"persona": "a cycling route planner checking for bridges along a candidate corridor",
         "constraint": "needs bridge presence confirmed before recommending the route",
         "style": "practical, asks bridge first then checks roundabout presence"},
        {"persona": "a transit authority officer verifying train station coverage",
         "constraint": "must confirm train station presence at a candidate stop location",
         "style": "formal, asks train station first then checks a second infrastructure item"},
        {"persona": "a golf resort developer scouting for existing golf facilities",
         "constraint": "needs to confirm whether a golf field already exists at the site",
         "style": "business-focused, asks golf field first then checks density if absent"},
        {"persona": "a school sports coordinator checking playing field availability",
         "constraint": "needs to know if a ground track field or soccer ball field is present",
         "style": "practical, asks about ground track field then soccer ball field in sequence"},
        {"persona": "a swimming club site assessor checking pool infrastructure",
         "constraint": "needs swimming pool confirmed before recommending the site to the club",
         "style": "specific, asks swimming pool first then a second related sport facility"},
        {"persona": "a tennis association officer auditing court availability",
         "constraint": "needs to confirm tennis court presence at the candidate site",
         "style": "systematic, asks tennis court first then checks another sport facility"},
        {"persona": "a logistics planner assessing roundabout availability for truck routing",
         "constraint": "needs roundabout presence confirmed for route planning",
         "style": "efficiency-focused, asks roundabout then train station for multi-modal check"},
        {"persona": "a wind-energy scout checking for existing turbines at a candidate site",
         "constraint": "needs windmill presence confirmed before commissioning a full wind survey",
         "style": "technical, asks windmill first then checks land-use to confirm rural setting"},
        {"persona": "a storage facility developer checking for competing infrastructure",
         "constraint": "needs to know if a storage tank is present before building a new facility",
         "style": "direct, asks storage tank first then a second infrastructure check"},
        {"persona": "a city overpass auditor verifying elevated road infrastructure",
         "constraint": "needs overpass presence confirmed at the inspection site",
         "style": "formal, asks overpass first then checks bridge presence for completeness"},
        # ---- ABSENCE-FOCUSED (user hopes / requires feature to be absent) ----
        {"persona": "a nature reserve manager verifying no stadium was built on protected land",
         "constraint": "site must have no Stadium present — absence is the expected and desired result",
         "style": "cautious, first confirms land-use is natural, then checks that no Stadium exists, then checks a second built feature to be thorough"},
        {"persona": "a green-belt conservation officer checking no overpass bisects the corridor",
         "constraint": "site must show Overpass as absent to remain eligible for green-belt status",
         "style": "formal, asks about Overpass absence first, then checks Bridge absence as a second verification"},
        {"persona": "a residential developer confirming the parcel is free of industrial storage",
         "constraint": "site must have no Storage Tank present before planning permission is submitted",
         "style": "business-focused, verifies Storage Tank is absent, then checks whether a Roundabout is present for access planning"},
        {"persona": "a heritage site assessor confirming no train station disturbs the historic area",
         "constraint": "needs Train Station confirmed as absent from the candidate heritage zone",
         "style": "methodical, checks land-use first, then confirms Train Station is absent, then notes density level"},
        {"persona": "an eco-park designer verifying no golf field occupies the proposed green space",
         "constraint": "site must show Golf Field as absent so the eco-park can proceed",
         "style": "practical, checks Golf Field absence first, then asks about a second sport facility to confirm no sport complex exists"},
        {"persona": "a flood-plain regulator confirming no windmill obstructs the drainage corridor",
         "constraint": "site must have no Windmill present to stay within aviation height limits",
         "style": "technical, verifies Windmill is absent, then checks that no Overpass blocks the drainage channel"},
        {"persona": "a school safety officer confirming no baseball field borders the campus",
         "constraint": "site must show Baseball Field as absent before a new school building can be approved",
         "style": "direct, asks Baseball Field absence first, then checks Soccer Ball Field absence as a follow-up"},
        {"persona": "a quiet-zone planner verifying no basketball court will generate noise near residential blocks",
         "constraint": "site must have no Basketball Court present for quiet-zone approval",
         "style": "systematic, confirms Basketball Court is absent, then checks Ground Track Field for completeness"},
    ],
    "Suitability": [
        {"persona": "a sports complex developer evaluating terrain for a new facility",
         "constraint": "site must have 'landuse' in land-use AND a Ground Track Field or Stadium present AND density below 6.0",
         "style": "checks land-use first, then infrastructure, then density before deciding"},
        {"persona": "an eco-lodge developer assessing a rural site",
         "constraint": "site must have 'natural' in land-use AND density below 3.0 AND no Stadium present",
         "style": "asks land-use first then density, then checks for incompatible infrastructure"},
        {"persona": "a public library planner evaluating a suburban candidate site",
         "constraint": "site must have 'landuse' in categories AND density between 3.0 and 7.0 AND no Airport nearby",
         "style": "systematic, checks land-use then density threshold then infrastructure"},
        {"persona": "a community garden advocate assessing urban land",
         "constraint": "site must have 'natural' in land-use AND density below 5.0 AND no Stadium or Train Station",
         "style": "asks land-use first then checks density, then verifies absence of incompatible features"},
        {"persona": "a humanitarian shelter planner evaluating a staging area",
         "constraint": "site must have 'natural' or 'landuse' in categories AND density below 2.0 AND no Stadium or Train Station",
         "style": "urgent, checks land-use, then density threshold, then infrastructure absence"},
        {"persona": "a school district officer checking a site for a new campus",
         "constraint": "site must have 'landuse' in categories AND density above 4.0 AND a Ground Track Field present",
         "style": "formal, checks land-use first then density level, then sport infrastructure"},
        {"persona": "a warehouse developer checking industrial land availability",
         "constraint": "site must have 'landuse' in categories AND density below 4.0 AND no Train Station present",
         "style": "business-focused, checks land-use then density then access infrastructure"},
        {"persona": "a wind farm developer screening a candidate plot",
         "constraint": "site must have 'natural' (non-urban) AND density below 2.0 AND no Stadium already present",
         "style": "checks land-use first, then density, then looks for incompatible built features"},
        {"persona": "a university campus expansion planner",
         "constraint": "site must have 'landuse' in categories AND density between 3.5 and 7.0 AND a Ground Track Field or Stadium",
         "style": "thorough, checks all three criteria in sequence and asks for final confirmation"},
        {"persona": "a data centre operator evaluating a low-density remote site",
         "constraint": "site must have 'landuse' in categories AND density below 2.5 AND no Train Station or Airport",
         "style": "technical, checks land-use first then density, then checks infrastructure absence"},
    ],
    "Encroachment": [
        {"persona": "an environmental regulator checking if built development has reached a natural zone",
         "constraint": "site has 'natural' in categories but may show high density or a Stadium indicating encroachment",
         "style": "authoritative, checks land-use first then density then one infrastructure check to confirm encroachment"},
        {"persona": "a green-belt authority officer monitoring boundary compliance",
         "constraint": "site should only have 'natural' categories but may contain 'landuse' indicating development",
         "style": "regulatory, checks all land-use categories first then density to assess severity"},
        {"persona": "a conservation officer checking whether a protected zone has been developed",
         "constraint": "site has 'natural' in categories but may show Stadium or Train Station presence indicating encroachment",
         "style": "systematic, checks land-use then infrastructure presence to identify conflict"},
        {"persona": "a farmer worried about industrial activity spreading onto agricultural land",
         "constraint": "site has 'landuse' in categories but user suspects high density and Airport or Harbor indicate industrial use",
         "style": "conversational, checks land-use first then density, then looks for Airport or Harbor"},
        {"persona": "a noise pollution analyst checking industrial proximity to residential areas",
         "constraint": "site has 'natural' categories but high density and Stadium or Train Station presence may indicate commercial encroachment",
         "style": "analytical, checks land-use categories then density score then infrastructure"},
        {"persona": "a heritage preservation officer checking site integrity",
         "constraint": "site should be dominated by 'natural' features but Train Station or Golf Field may have encroached",
         "style": "cautious, checks land-use then asks about Train Station and Golf Field presence"},
        {"persona": "a community activist concerned about overdevelopment in a natural zone",
         "constraint": "site should only have 'natural' but may also show 'landuse' and Stadium presence",
         "style": "passionate, checks land-use categories then density and one infrastructure check"},
        {"persona": "a wetland specialist checking if urban development impacts a natural area",
         "constraint": "site has 'natural' in categories but density and Train Station presence may signal urban pressure",
         "style": "scientific, checks land-use then density then Train Station presence"},
        {"persona": "a city planner assessing potential zoning violations",
         "constraint": "site has 'natural' land-use but may show high density and Train Station indicating misuse",
         "style": "formal, checks land-use categories then density, then infrastructure presence"},
        {"persona": "a coastal management officer checking infrastructure expansion into a natural area",
         "constraint": "site has 'natural' in categories but Harbor or Airport presence may indicate encroachment",
         "style": "focused, checks land-use first then looks for Harbor or Airport as encroachment signal"},
    ],
    "Urban Profile": [
        {"persona": "a policy analyst building a neighbourhood profile for a city report",
         "constraint": "needs land-use categories plus density score and at least one infrastructure check for a complete profile",
         "style": "systematic, collects all three data points before drafting the profile"},
        {"persona": "a journalist investigating urban development patterns in London",
         "constraint": "needs land-use categories and density to describe the character of the area",
         "style": "conversational, asks land-use first then density, asks follow-up about a specific infrastructure"},
        {"persona": "a university researcher collecting site-level data for an urban ecology study",
         "constraint": "needs land-use categories, density score, and presence of at least two infrastructure types",
         "style": "academic, asks each metric separately in sequence"},
        {"persona": "a city mayor reviewing neighbourhood dashboards before a planning meeting",
         "constraint": "needs a quick summary: land-use categories, density level, and one notable infrastructure",
         "style": "executive, wants concise answers, asks all three questions in quick succession"},
        {"persona": "a neighbourhood association chair preparing a development objection letter",
         "constraint": "needs land-use categories and density to argue against overdevelopment",
         "style": "engaged, checks land-use and density then cross-checks one infrastructure presence"},
        {"persona": "a real-estate analyst profiling two competing sites for a client report",
         "constraint": "needs land-use, density, and at least one infrastructure check for each site",
         "style": "professional, profiles each site in turn then compares them"},
        {"persona": "a transport authority officer building a corridor profile",
         "constraint": "needs land-use categories, density, and train station or roundabout presence for route planning",
         "style": "formal, asks land-use then density then infrastructure in each site"},
        {"persona": "a climate adaptation planner profiling areas at risk",
         "constraint": "needs land-use (is 'natural' present?), density, and bridge presence for flood risk assessment",
         "style": "methodical, checks natural cover first then density then bridge presence"},
        {"persona": "an insurance risk underwriter profiling a commercial site",
         "constraint": "needs land-use categories, density score, and stadium or storage tank presence for premium calculation",
         "style": "precise, asks each metric separately and requests a summary at the end"},
        {"persona": "a community development officer comparing two candidate sites for investment",
         "constraint": "needs land-use, density, and one infrastructure presence check for both sites before recommending",
         "style": "collaborative, profiles both sites then compares to make a recommendation"},
    ],
}



# ==========================================
# TASK-SPECIFIC GUIDANCE
# ==========================================
# Additional hand-written guidance injected per task beyond what the CSV
# columns can express (assertion style, edge-case notes).  Keyed by Task name.
_TASK_GUIDANCE_EXTRA: dict[str, str] = {
    "Density Comparison": (
        "The scenario MUST compare population density across at least two sites. "
        "COORDINATE REQUIREMENT: The user MUST provide the specific lat and lon for EACH site when first requesting its satellite image. "
        "Site A's coordinates go in the step that first asks about Site A; Site B's coordinates go in the step that first asks about Site B. "
        "The agent cannot know which tile to fetch without being given coordinates. "
        "After a site's image has been fetched, subsequent turns about that site use referential language ('this site', 'the first location'). "
        "nl_assertions MUST be directional ONLY — e.g., 'Site A has a higher density score than Site B'. "
        "NEVER assert an exact density float value. "
        "The density score is a VLM estimate and cannot be reproduced exactly."
    ),
    "Land Use Verification": (
        "nl_assertions MUST use the EXACT OSM tag strings from the valid set: 'landuse', 'natural'. "
        "If the ground_truth has ['landuse', 'natural'], the assertion must confirm both tags are present. "
        "If a tag is absent, assert its absence explicitly. "
        "Do NOT invent categories or use 'leisure' — it has been removed from the database."
    ),
    "Infrastructure Detection": (
        "Use ONLY infrastructure types from the ground_truth.infrastructure_counts when querying detect_infrastructure: "
        "Bridge, Stadium, Train Station, Golf Field, Soccer Ball Field, Swimming Pool, "
        "Tennis Court, Roundabout, Basketball Court, Ground Track Field, Baseball Field, "
        "Overpass, Storage Tank, Windmill. "
        "nl_assertions MUST use the ground_truth value to assert presence (1=True/present) or absence (0=False/absent). "
        "Use phrasing: 'The agent correctly identified [feature] as present/absent at the site.' "
        "IMPORTANT: nl_assertions MUST ONLY cover features explicitly queried via detect_infrastructure "
        "in the actions list — no assertions about unqueried features. "
        "BALANCE REQUIREMENT: Roughly half the scenarios should query features that are ABSENT (infrastructure_counts value = 0) "
        "and the other half should query features that are PRESENT (value = 1). "
        "When the persona constraint says 'must be absent' or 'no X present', pick a feature with value 0 in ground_truth.infrastructure_counts. "
        "When the persona constraint says 'needs X confirmed', pick a feature with value 1 in ground_truth.infrastructure_counts. "
        "MANDATORY NAMING RULE: Every reason_for_call step that asks the user to request infrastructure detection "
        "MUST state the EXACT feature name (e.g., 'ask whether a Stadium is present', 'ask the agent to check for a Train Station'). "
        "FORBIDDEN: vague phrases like 'check a specific infrastructure element', 'look for a built feature', "
        "'check for an indicator', or 'test a built feature as a fallback'. "
        "Pick feature names directly from the ground_truth.infrastructure_counts values present in the Database Context."
    ),
    "Suitability": (
        "Assert using exact OSM land_use_categories tags ('landuse', 'natural') — do NOT use 'leisure'. "
        "Do NOT assert density for single-site scenarios — density labels ('low', 'moderate', 'high') are not calibrated and unpredictable. "
        "Density may only appear in nl_assertions as a directional comparison between two sites. "
        "If a suitability criterion fails, the nl_assertion must state which criterion failed and why. "
        "MANDATORY NAMING RULE: Every reason_for_call step that asks the user to request infrastructure detection "
        "MUST state the EXACT feature name (e.g., 'ask whether a Swimming Pool is present', 'ask the agent to check for a Golf Field'). "
        "FORBIDDEN: vague phrases like 'check a specific infrastructure element' or 'look for a built feature'. "
        "Pick feature names directly from the ground_truth.infrastructure_counts values in the Database Context."
    ),
    "Encroachment": (
        "Assert the contradiction between land_use_categories and detected infrastructure. "
        "Example: 'The site has land_use_categories [\"natural\"] yet a Stadium is present, "
        "indicating encroachment.' "
        "Do NOT assert density for single-site scenarios — density labels ('low', 'moderate', 'high', 'elevated') are not calibrated and unpredictable. "
        "Density may only appear in nl_assertions as a directional comparison between two sites. "
        "MANDATORY NAMING RULE: Every reason_for_call step that asks the user to request infrastructure detection "
        "MUST state the EXACT feature name (e.g., 'ask whether a Stadium is present', 'ask the agent to check for a Train Station'). "
        "FORBIDDEN: vague phrases like 'check a specific infrastructure element', 'check for built features', "
        "'look for an indicator', or 'test a second built feature as a fallback'. "
        "Pick feature names directly from the ground_truth.infrastructure_counts values in the Database Context. "
        "Do NOT use 'leisure' in land-use constraints — valid categories are 'landuse' and 'natural' only."
    ),
    "Urban Profile": (
        "The scenario MUST collect at least two of three metrics before profiling: "
        "classify_land_use (returning OSM tags), analyze_urban_density, detect_infrastructure. "
        "nl_assertions must cover all metrics collected. "
        "Use directional density language only — never assert exact float. "
        "If profiling two sites: each site's lat and lon MUST be provided by the user in the first turn that requests that site's image. "
        "Assert each metric per site."
    ),
}



def build_task_guidance(task_row: dict) -> str:
    """Build task-specific guidance text from CSV metadata columns + extra notes."""
    task_name = task_row["Task"]
    required = task_row.get("Required_Tools", "")
    supplementary = task_row.get("Supplementary_Tools", "")
    min_actions = task_row.get("Min_Actions", "2")
    needs_temporal = task_row.get("Needs_Temporal", "False") == "True"
    needs_multi = task_row.get("Needs_Multi_Site", "False") == "True"

    req_list = [t.strip() for t in required.split(";") if t.strip()]
    supp_list = [t.strip() for t in supplementary.split(";") if t.strip()]

    parts = []

    # Required tool chain
    tool_chain = " + ".join(["get_satellite_tile"] + req_list)
    parts.append(
        f"The scenario must require the agent to call: {tool_chain}."
    )

    # Multi-site note
    if needs_multi:
        parts.append(
            "The agent must call get_satellite_tile for BOTH sites, then run analysis tools on each. "
            "CRITICAL: The user MUST provide the lat/lon of each site in the first conversational turn "
            "that introduces that site. The agent has no way to identify a location without coordinates."
        )

    # Temporal note (not used in current task set — all Needs_Temporal=False)
    if needs_temporal:
        parts.append(
            "IMPORTANT: This task requires temporal satellite data, which is not supported in the current dataset."
        )

    # Supplementary tools
    if supp_list:
        parts.append(
            f"Supplementary tools the user may also request: {', '.join(supp_list)}."
        )

    # Min actions
    parts.append(
        f"Minimum actions: {min_actions} (get_satellite_tile + required tools). "
        f"Recommended: {int(min_actions) + 1}-{int(min_actions) + 3}."
    )

    # Extra hand-written notes
    extra = _TASK_GUIDANCE_EXTRA.get(task_name, "")
    if extra:
        parts.append(extra)

    dep = _TASK_DEPENDENCY_HOOKS.get(task_name, "")
    if dep:
        parts.append(f"Multi-turn dependency requirement: {dep}")

    return " ".join(parts)



# ==========================================
# COMPLEXITY ESCALATION HINTS
# ==========================================
_COMPLEXITY_HINTS: dict[str, str] = {
    "Density Comparison":
        "Compare at least THREE sites on population density score. After ranking, have the user "
        "ask about land-use categories on the top two sites to cross-validate the ranking. "
        "Include at least one infrastructure check on the highest-density winner. "
        "nl_assertions must be purely directional — no exact floats. "
        "Require at least 8 steps in reason_for_call and at least 8 actions.",

    "Land Use Verification":
        "Verify land-use categories on TWO sites and compare their OSM tag profiles. "
        "After verifying, have the user ask a cross-question: if one site has 'natural' and the other has 'leisure', "
        "which better suits a recreational park? Include at least one infrastructure check per site. "
        "nl_assertions must use exact OSM tags. "
        "Require at least 6 steps in reason_for_call and at least 6 actions.",

    "Infrastructure Detection":
        "Check at least THREE different infrastructure types — at least TWO that are present (1) "
        "and at least ONE that is absent (0). After detection, ask the user to interpret the findings "
        "in the context of land-use categories. "
        "nl_assertions must only cover queried features, using 'present/absent' language. "
        "Require at least 6 steps in reason_for_call and at least 6 actions.",

    "Suitability":
        "Evaluate the site against at least THREE criteria (land-use category, density threshold, "
        "infrastructure presence). If any criterion fails, have the user re-evaluate with an alternative threshold. "
        "End with submit_site_assessment. "
        "nl_assertions must state each criterion and pass/fail verdict. "
        "Require at least 7 steps in reason_for_call and at least 7 actions.",

    "Encroachment":
        "Identify encroachment from MULTIPLE angles: classify land-use categories (OSM tags), "
        "check density directionally, and detect at least TWO infrastructure types that contradict the land-use. "
        "The user should probe the contradiction explicitly before concluding. "
        "End with submit_site_assessment. "
        "nl_assertions must state the specific contradiction found. "
        "Require at least 7 steps in reason_for_call and at least 7 actions.",

    "Urban Profile":
        "Build profiles for TWO sites, collecting land-use categories, density score, and "
        "at least TWO infrastructure checks per site. After profiling, have the user compare "
        "the two sites on at least two dimensions before making a recommendation. "
        "nl_assertions must cover all collected metrics per site. "
        "Require at least 9 steps in reason_for_call and at least 9 actions.",
}


_TASK_DEPENDENCY_HOOKS: dict[str, str] = {
    "Density Comparison": (
        "After getting density on both sites, the user MUST ask a trade-off follow-up that depends "
        "on the comparison result (e.g., 'Site A is denser — does it also have leisure categories?'). "
        "Include at least one explicit conditional branch."
    ),
    "Land Use Verification": (
        "The user MUST ask a follow-up that depends on the category result: "
        "if 'natural' is present, ask about density; if absent, check infrastructure. "
        "Include at least one explicit if/otherwise pivot."
    ),
    "Infrastructure Detection": (
        "After the first detection result, the user MUST ask a follow-up that depends on presence/absence: "
        "if present, ask about a related feature; if absent, ask about land-use. "
        "Include at least one explicit conditional branch."
    ),
    "Suitability": (
        "Use threshold-driven branching: if any required criterion fails, ask mitigation or re-evaluation; "
        "if all criteria pass, ask confirmation before final decision."
    ),
    "Encroachment": (
        "The user MUST perform contradiction probing: after getting land-use categories, "
        "if infrastructure contradicts them, ask the agent to reconcile the conflict before concluding."
    ),
    "Urban Profile": (
        "Profile each site in turn. After both profiles are complete, the user MUST ask at least one "
        "comparative follow-up question before concluding."
    ),
}



# ==========================================
# LOADERS
# ==========================================

def load_task_categories(csv_path: str) -> list[dict]:
    """Load all task rows from task_overall.csv."""
    tasks = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Task") in _ACTIVE_TASKS:
                tasks.append(row)
    return tasks


def load_db(db_path: str) -> dict:
    """Load raw database from db.json (includes ground_truth for scenario crafting)."""
    with open(db_path, encoding='utf-8') as f:
        return json.load(f)


def load_names(names_path: str) -> list[str]:
    """Load user names from name_dataset.csv."""
    names = []
    try:
        with open(names_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                names.append(row["name"])
    except FileNotFoundError:
        names = ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley",
                 "Quinn", "Avery", "Dakota", "Harper", "Hayden", "Jamie", "Kai"]
    return names


# ==========================================
# CONTEXT SAMPLER
# ==========================================

def sample_db_context(
    db: dict,
    task_name: str,
    rng: random.Random,
    user_name: str | None = None,
    task_row: dict | None = None,
    used_site_ids: set[str] | None = None,
) -> dict:
    """
    Sample a coherent context slice tailored to the given task type.

    Returns a dict with sampled site(s) including full ground_truth,
    so the LLM can craft correct assertions.
    """
    sites = db["sites"]
    site_list = list(sites.values())

    # ---- Easy infrastructure types: must match the keys actually present in the DB ----
    EASY_INFRA = {
        "Airport", "Stadium", "Golf Field", "Harbor",
        "Ground Track Field", "Soccer Ball Field", "Train Station"
    }

    def _has_at_least_one_easy_infra(site: dict) -> bool:
        counts = site.get("ground_truth", {}).get("infrastructure_counts", {})
        return any(counts.get(k, 0) == 1 for k in EASY_INFRA)

    def _task_uses_infra(name: str, row: dict | None) -> bool:
        if row:
            req = row.get("Required_Tools", "")
            supp = row.get("Supplementary_Tools", "")
            if "detect_infrastructure" in f"{req};{supp}".lower():
                return True
        return name in {"Infrastructure Detection", "Suitability", "Encroachment", "Urban Profile"}

    # For infra tasks, prefer sites with at least one easy infra present
    if _task_uses_infra(task_name, task_row):
        infra_eligible = [s for s in site_list if _has_at_least_one_easy_infra(s)]
        site_list = infra_eligible if infra_eligible else site_list

    if task_name == "Density Comparison":
        eligible = site_list  # all sites valid; multi-site sampler handles diversity

    elif task_name == "Land Use Verification":
        eligible = site_list  # all sites have land_use_categories in new DB

    elif task_name == "Infrastructure Detection":
        # Prefer sites with at least one present easy infra
        eligible = [s for s in site_list if _has_at_least_one_easy_infra(s)]
        if not eligible:
            eligible = site_list

    elif task_name == "Suitability":
        eligible = site_list

    elif task_name == "Encroachment":
        # Sites where infrastructure present (1) contrasts with a potentially natural/leisure land-use
        eligible = [
            s for s in site_list
            if _has_at_least_one_easy_infra(s)
        ]
        if not eligible:
            eligible = site_list

    elif task_name == "Urban Profile":
        eligible = site_list

    else:
        eligible = site_list

    # ---- Decide single-site vs multi-site from CSV metadata ----
    is_multi_site = (task_name in {"Density Comparison"})
    if task_row and task_row.get("Needs_Multi_Site", "False") == "True":
        is_multi_site = True

    # ---- Build context ----
    if is_multi_site:
        # Sample 2 sites with clearly different density scores for meaningful comparison.
        sites_sorted = sorted(eligible, key=lambda s: s["ground_truth"]["population_density"])
        n = len(sites_sorted)
        if n >= 4:
            # Pick from lower quarter and upper quarter to guarantee ≥1.0 density diff
            q1_end = n // 4
            q3_start = 3 * n // 4
            low_pool = sites_sorted[:q1_end + 1]
            high_pool = sites_sorted[q3_start:]
        else:
            low_pool = [sites_sorted[0]]
            high_pool = [sites_sorted[-1]]

        site1 = rng.choice(low_pool)
        site2 = rng.choice(high_pool)
        if site1["site_id"] == site2["site_id"]:
            # fallback: pick any two distinct sites
            two = rng.sample(eligible, min(2, len(eligible)))
            site1, site2 = two[0], two[-1]

        context_sites = {
            site1["site_id"]: site1,
            site2["site_id"]: site2,
        }
    else:
        # DEDUP: exclude already-used sites if possible, fall back to full pool
        if used_site_ids:
            fresh = [s for s in eligible if s["site_id"] not in used_site_ids]
            eligible = fresh if fresh else eligible
        site = rng.choice(eligible)
        context_sites = {site["site_id"]: site}

    return {
        "sites": context_sites,
        "user_name": user_name,
    }




# ==========================================
# SCENARIO GENERATOR
# ==========================================

def generate_urban_scenario(
    task_row: dict,
    db_context: dict,
    variation: dict | None = None,
    prev_instruction: str | None = None,
    increase_complexity: bool = False,
) -> dict | None:
    """
    Calls the OpenAI API to generate a structured scenario.

    Args:
        task_row:           Dict with 'Category', 'Task', 'Core Objective' from CSV.
        db_context:         Sampled context with site(s) and ground_truth.
        variation:          Optional persona/constraint/style dict.
        prev_instruction:   Previous round's reason_for_call (for anti-repetition).
        increase_complexity: If True, inject complexity escalation hints.

    Returns:
        Parsed JSON scenario dict, or None on error.
    """
    tools_description = json.dumps(_TOOL_DEFINITIONS, indent=2)
    user_name = db_context.get("user_name", "the user")

    # Count sites to determine if this is a comparison task
    num_sites = len(db_context["sites"])

    system_prompt = f"""You are an expert 'Scenario Architect' for UrbanConvBench, a benchmark evaluating tool-augmented AI agents that analyze satellite imagery.
Your objective is to generate a realistic, multi-turn conversational user task based strictly on the provided Task Row and Database Context.

CRITICAL RULES:
1. ZERO HALLUCINATION: Every coordinate, site_id, metric value, and infrastructure count in the evaluation_criteria MUST come directly from the ground_truth in the Database Context. Do not invent or approximate ANY values.
2. DEPENDENCY CHAIN: The `user_scenario.instructions.reason_for_call` MUST be multi-turn with explicit dependency between turns. Each step after Step 1 must depend on previous agent output (confirm, challenge, branch, or refine).
3. EXPLICIT TERMINATION: The final step in `reason_for_call` MUST explicitly instruct the user simulator to thank the agent and end the chat (e.g., "Step X: Thank the agent and end the chat.").
4. TOOL CHAINING: The `evaluation_criteria.actions` array must list the SINGLE DETERMINISTIC sequence of tool calls that the agent makes, based on the ground_truth values. It is a flat, ordered list with NO conditional branches, NO if/else, NO alternative paths. First action MUST be `get_satellite_tile`.
5. CORRECT TOOL NAMES: Use ONLY these exact names: get_satellite_tile, classify_land_use, analyze_urban_density, detect_infrastructure, submit_site_assessment, transfer_to_human_agents. NEVER use old or internal names like get_past_satellite_tile, check_environmental_ratio, estimate_carbon_emission, verify_path_connectivity, compare_temporal_change, measure_spatial_distance, calculate_density, calculate_green_ratio, classify_land_use_vlm.
6. PERSONA: Give the user a clear persona with a name. The user should speak naturally and conversationally (1-2 sentences per turn).
7. TASK ALIGNMENT: The generated scenario must directly exercise the Core Objective of the given task.
8. IMAGE PATH CONVENTION: In the `arguments` of actions, use the pattern "{{site_id}}.png" for current image paths (e.g., "10904_16379.png"). Do NOT use full directory paths.
9. GROUND TRUTH ASSERTIONS — follow these rules per metric type:
   - land_use_categories: use the EXACT OSM tag strings from ground_truth.land_use_categories (e.g., 'landuse', 'natural', 'leisure'). Assert all tags present and optionally absent ones.
   - population_density: For MULTI-SITE tasks ONLY — use directional comparisons like 'Site A has a higher density score than Site B'. NEVER assert density for single-site tasks — the VLM estimate is not calibrated to any label ('low', 'moderate', 'high', 'extremely dense'). NEVER assert the exact float value.
   - infrastructure_counts: assert presence (1) as True/present and absence (0) as False/absent. Use phrasing: 'The agent correctly identified [feature] as present/absent.' Only assert features queried in actions.
   - STRICT SCOPE RULE: nl_assertions MUST ONLY assert outcomes from tool calls listed in `evaluation_criteria.actions`. Every assertion must trace to a concrete tool call.
10. CONVERSATIONAL FLOW: The user should NOT ask all questions in one message. Each step in reason_for_call should be a separate conversational turn. The user should wait for the agent's response before asking the next question.
11. NL ASSERTIONS LIMIT: Generate AT MOST {MAX_NL_ASSERTIONS} items in `evaluation_criteria.nl_assertions`. Prioritize items that are QUANTITATIVE (e.g., exact counts, specific classification labels) over generic qualitative statements.
12. COORDINATE-BASED REQUESTS: The simulated user MUST NOT provide the `site_id` when requesting satellite imagery or analysis. Instead, the user MUST provide the specific `lat` and `lon` coordinates from the Database Context. The `site_id` is for internal evaluation grounding only (e.g., within `evaluation_criteria.actions` or `submit_site_assessment`).
13. COORDINATE PROVISION RULES:
   - SINGLE-SITE tasks: coordinates must be provided ONCE in the user's very first turn. All subsequent turns use referential language ("the image", "this site", "the location").
   - MULTI-SITE tasks (e.g. Density Comparison, Urban Profile with two sites): Each site's coordinates must be provided ONCE in the first conversational turn that requests analysis of THAT site. Site A's lat/lon goes in the step that asks about Site A; Site B's lat/lon goes in the step that first asks about Site B. NEVER omit coordinates when introducing a new site — the agent has no other way to identify it.
14. BRANCHING IN reason_for_call vs DETERMINISTIC actions — THIS IS THE MOST IMPORTANT DISTINCTION:
   - `reason_for_call` MAY and SHOULD have if/else branching to make the simulated user sound natural and reactive. Example: 'if the Bridge is present, ask about Roundabout; otherwise ask about land-use.' This branching is for the SIMULATOR only.
   - `evaluation_criteria.actions` MUST NOT have any branching. It is the SINGLE CORRECT execution path the agent will actually take, determined by the known ground_truth. You already know whether Bridge=1 or Bridge=0 from the Database Context. Use that to resolve ALL branches and write only the one path that will actually execute.
   - EXAMPLE: ground_truth has Bridge=1. reason_for_call says 'if Bridge present, check Roundabout; otherwise check land-use.' Since Bridge=1, actions includes detect_infrastructure(Bridge) then detect_infrastructure(Roundabout). The 'otherwise' branch (classify_land_use) does NOT appear in actions at all.
   - FORBIDDEN in actions: 'If X then Y else Z', conditional info fields, or listing both branches of an if/else.
15. INFRASTRUCTURE NAMING — MANDATORY: Whenever a reason_for_call step instructs the user to ask the agent to detect infrastructure, the step MUST name the EXACT feature (e.g., 'ask the agent whether a Stadium is present', 'ask whether a Train Station exists here'). NEVER use vague phrases like 'check a specific infrastructure element', 'look for a built feature', or 'check for an indicator'. The user must state the feature name so the agent knows what to pass as `feature_query` to detect_infrastructure. Choose feature names ONLY from the ground_truth.infrastructure_counts present in the Database Context.
16. ACTIONS ←→ CONVERSATION TRACEABILITY — MANDATORY: Every tool call in `evaluation_criteria.actions` MUST correspond to a user request in `reason_for_call` AND must be on the ground-truth-resolved execution path. Concretely:
   - `submit_site_assessment` MUST appear in actions ONLY if a reason_for_call step tells the user to ask the agent to formally submit or finalize the assessment. If no such step exists, do NOT include submit_site_assessment in actions.
   - `classify_land_use` MUST appear in actions ONLY if a reason_for_call step tells the user to ask for land-use category classification.
   - `analyze_urban_density` MUST appear in actions ONLY if a reason_for_call step tells the user to ask for a density estimate.
   - `detect_infrastructure` MUST appear in actions ONLY if a reason_for_call step tells the user to ask about a specific infrastructure feature.
   FORBIDDEN: adding tool calls to actions because they seem relevant, or because they are in the 'otherwise' branch of a conditional that the ground truth resolves to FALSE. Every action must be the direct result of the ONE branch the conversation actually takes.

AVAILABLE TOOLS:
{tools_description}

OUTPUT FORMAT:
Output a raw, valid JSON object (no markdown code blocks). The schema is:

{{
  "id": "urban_satellite_<slug>_<number>",
  "description": {{
    "purpose": "<1-2 sentence description of what this scenario tests>",
    "relevant_policies": null,
    "notes": "<task category> - <brief note about the specific focus>"
  }},
  "user_scenario": {{
    "persona": "<A concise persona description. E.g., 'A busy infrastructure auditor asking direct questions.'>",
    "instructions": {{
      "task_instructions": "<Role-play instructions for the user simulator. Include CRITICAL RULES about acting natural, keeping responses short, not echoing internal data.>",
      "domain": "urban_satellite",
    "reason_for_call": "<Step-by-step instructions for the user simulator. Each step is a conversational turn and MUST reference prior agent output. Include at least two conditional/branching cues.>",
      "known_info": "<What the user knows: coordinates (latitude/longitude), specific thresholds or criteria they care about. DO NOT use site IDs in the user's conversational turns.>",
      "unknown_info": "<What the user wants to find out from the agent.>"
    }}
  }},
  "initial_state": null,
  "evaluation_criteria": {{
    "actions": [
      {{
        "action_id": "<unique_id>",
        "name": "<exact tool name from AVAILABLE TOOLS>",
        "arguments": {{}},
        "info": "<what this action accomplishes>"
      }}
    ],
    "nl_assertions": [
      "<assertion about what the agent should report, using exact ground_truth values>"
    ]
  }}
}}
"""

    # Build variation section
    variation_section = ""
    if variation:
        variation_section = f"""
SCENARIO ANGLE — shape the scenario around this specific angle:
- User persona:    {variation['persona']}
- Key constraint:  {variation['constraint']}
- Request style:   {variation['style']}

Reflect this angle in the persona, task_instructions, and reason_for_call.
"""

    # Build anti-repetition section
    anti_repeat_section = ""
    if prev_instruction:
        anti_repeat_section = f"""
ANTI-REPETITION CONSTRAINT:
The previous generation produced this scenario:
\"\"\"{prev_instruction}\"\"\"

Generate a CLEARLY DISTINCT scenario: different goal, different metrics queried, different conversation flow.
"""

    # Build complexity section
    complexity_section = ""
    if increase_complexity:
        task_hint = _COMPLEXITY_HINTS.get(task_row["Task"], "")
        complexity_section = f"""
COMPLEXITY ESCALATION — generate a HARDER scenario:
1. LONGER TOOL CHAINS (aim for 5-8+ actions)
2. MULTI-METRIC GOALS (check 3+ different metrics)
3. STACKED CONSTRAINTS (multiple simultaneous requirements)
4. CONDITIONAL REASONING (agent must reconcile potentially conflicting data)

Task-specific guidance for "{task_row['Task']}":
{task_hint}
"""

    # Build task-specific guidance from CSV columns + extra notes
    task_guidance = build_task_guidance(task_row)

    user_prompt = f"""TASK TO GENERATE:
- Category:       {task_row['Category']}
- Task Name:      {task_row['Task']}
- Core Objective: {task_row['Core Objective']}

TASK-SPECIFIC RULES:
{task_guidance}

USER NAME: {user_name}
{variation_section}{anti_repeat_section}{complexity_section}
DATABASE CONTEXT (sites with ground_truth for crafting assertions):
{json.dumps(db_context["sites"], indent=2)}

Generate ONE complete scenario in JSON format.
Requirements:
1. The scenario must exercise the Core Objective above.
2. For nl_assertions: use directional language for density ('Site A has higher density than Site B') — NEVER assert exact float values. Use exact OSM tag strings for land_use_categories. Use present/absent for infrastructure.
3. The conversation must be non-decomposable: at least TWO steps must explicitly depend on previous answers (e.g., contradiction check, threshold fail pivot, or follow-up based on result).
4. The actions sequence must use ONLY tool names from AVAILABLE TOOLS: get_satellite_tile, classify_land_use, analyze_urban_density, detect_infrastructure, submit_site_assessment, transfer_to_human_agents.
5. Set domain to "urban_satellite".
6. For nl_assertions: generate AT MOST {MAX_NL_ASSERTIONS} items. Prioritize assertions that trace to a concrete tool call in the actions list.
7. USER REQUEST STYLE: The simulated user MUST use `lat` and `lon` from the Database Context for the initial image request. DO NOT mention `site_id` in the user's speech.
8. COORDINATE NON-REPETITION: Ensure coordinates are only provided in the first turn of `reason_for_call`. Subsequent turns should use referential language (e.g., "the site", "this area").
9. INFRASTRUCTURE NAMING — MANDATORY: Any step in reason_for_call that asks the user to request infrastructure detection MUST name the EXACT feature (e.g., 'ask whether a Stadium is present', 'ask the agent to check for a Train Station'). NEVER write vague instructions like 'check a specific infrastructure element', 'look for a built feature', 'check for an indicator', or 'test a second built feature as a fallback'. The feature name must match an entry in the ground_truth.infrastructure_counts from the Database Context.
10. BRANCHING vs DETERMINISTIC ACTIONS — CRITICAL: `reason_for_call` CAN have if/else branching for realism. But `evaluation_criteria.actions` must be a FLAT, ORDERED, UNCONDITIONAL list — the single execution path the agent actually takes, resolved against the ground_truth you already know. Before writing actions: look at each if/else branch in reason_for_call, check the ground_truth value, pick the branch that fires, and include ONLY those tool calls. The other branch's tool calls must NOT appear in actions at all. No conditional language ('If X', 'only if') is allowed in the action `info` field either.
"""

    diff_label = task_row.get("Difficulty", "medium")
    print(f"  Generating: [{task_row['Category']}] {task_row['Task']} [{diff_label}]...")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )

        raw_output = response.choices[0].message.content
        scenario_json = json.loads(raw_output)
        return scenario_json

    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def _get_queried_features(actions: list[dict]) -> set[str]:
    """Extract feature_query values from all detect_infrastructure actions."""
    return {
        a["arguments"].get("feature_query", "").lower()
        for a in actions
        if a.get("name") == "detect_infrastructure"
        and "feature_query" in a.get("arguments", {})
    }


# Known infrastructure feature keywords (lowercase) to detect in assertions.
# Must match the keys in infrastructure_counts in the DB (INFRA_FILTER_KEYS).
_INFRA_FEATURE_KEYWORDS = [
    "airport", "stadium", "golf field", "harbor",
    "ground track field", "soccer ball field", "train station",
]


def _assertion_references_infra_feature(assertion: str) -> str | None:
    """
    If the assertion references a specific infrastructure feature (present/absent),
    return the feature keyword (lowercase). Otherwise return None.
    """
    assertion_lower = assertion.lower()
    # Must contain presence/absence language to be an infra assertion
    if not any(kw in assertion_lower for kw in ("present", "absent", "correctly identified")):
        return None
    for feat in _INFRA_FEATURE_KEYWORDS:
        if feat in assertion_lower:
            return feat
    return None


# Maps assertion keyword hints → the tool that must be present in actions.
# Each entry: (keywords_any_match, tool_name)
_ASSERTION_TOOL_REQUIREMENTS = [
    # land-use assertions (OSM tags) require classify_land_use
    (("land_use_categories", "land use categories", "landuse", "'natural'", "'leisure'",
      "osm tag", "land-use category", "land use category"),
     "classify_land_use"),
    # density assertions require analyze_urban_density
    (("density score", "population density", "density appears", "density is higher",
      "density is lower", "higher density", "lower density", "denser"),
     "analyze_urban_density"),
]


def _is_qualitative_density_assertion(assertion: str) -> bool:
    """
    Returns True when an nl_assertion judges density using a qualitative label
    ('low', 'moderate', 'high', 'extremely dense', 'elevated', 'sparse', etc.)
    for a single-site scenario.  These labels are VLM-uncalibrated and cannot
    be reproduced reliably, so they must be removed.

    Directional multi-site comparisons ('Site A has higher density than Site B')
    are kept because the judge only checks direction, not the label.
    """
    al = assertion.lower()
    if not any(kw in al for kw in ("density", "population density")):
        return False
    # Keep directional multi-site comparisons
    directional = ("higher than", "lower than", "higher density", "lower density",
                   "denser than", "sparser than", "more dense", "less dense",
                   "site a", "site b", "first site", "second site",
                   "location a", "location b")
    if any(kw in al for kw in directional):
        return False
    # Flag qualitative labels on single-site density
    qualitative = ("appears low", "appears moderate", "appears high",
                   "is low", "is moderate", "is high",
                   "low to moderate", "moderate to high", "extremely dense",
                   "very dense", "fairly dense", "relatively low", "relatively high",
                   "elevated", "appears elevated", "seems low", "seems high",
                   "low density", "high density", "moderate density",
                   "density is low", "density is high", "density is moderate",
                   "sparse", "sparsely populated", "densely populated")
    return any(kw in al for kw in qualitative)


def _is_exact_density_float_assertion(assertion: str) -> bool:
    """
    Returns True when an nl_assertion pins density to an exact VLM-output float value,
    which is NOT reproducible across runs (e.g. "density score of 3.9", "density is 6.2").

    Returns False when the float is used as a user-defined directional threshold
    (e.g. "density appears above 4.0", "higher density than Site B") — these are
    safe because the judge evaluates direction, not the exact number.

    Rule:
      - assertion contains density/population language   AND
      - assertion contains a float literal              AND
      - no directional/threshold keyword is present
      => treat as exact-equality assertion → must be removed.
    """
    import re
    al = assertion.lower()
    if not any(kw in al for kw in ("density", "population density")):
        return False
    if not re.search(r"\b\d+\.\d+\b", al):
        return False
    # If a directional word exists, the float is a threshold — keep the assertion.
    directional = (
        "higher", "lower", "above", "below", "greater", "less",
        "exceeds", "exceed", "more than", "less than", " than ",
        "denser", "sparser", "higher than", "lower than",
    )
    if any(kw in al for kw in directional):
        return False
    # Float present + density language + no directional cue → exact equality → drop.
    return True


def filter_nl_assertions(scenario: dict) -> list[str]:
    """
    Remove nl_assertions whose content implies a tool that was NOT called in actions,
    or that assert an exact VLM-output float value (unreproducible).

    Covers:
      - Infrastructure presence/absence assertions for unqueried features
      - Land-use (OSM tag) assertions without classify_land_use in actions
      - Density assertions without analyze_urban_density in actions
      - Density assertions that pin an exact float (e.g. "density score of 3.9")
    Returns list of removed assertion strings (for logging).
    """
    removed = []
    try:
        actions = scenario["evaluation_criteria"]["actions"]
        assertions = scenario["evaluation_criteria"].get("nl_assertions", [])
        action_names = {a["name"] for a in actions}
        queried_features = _get_queried_features(actions)

        kept = []
        for assertion in assertions:
            assertion_lower = assertion.lower()
            drop = False

            # 1. Infrastructure count assertions: filter if feature not queried
            feat = _assertion_references_infra_feature(assertion)
            if feat is not None and feat not in queried_features:
                drop = True

            # 2. Metric assertions: filter if required tool not present
            if not drop:
                for keywords, required_tool in _ASSERTION_TOOL_REQUIREMENTS:
                    if any(kw in assertion_lower for kw in keywords):
                        if required_tool not in action_names:
                            drop = True
                            break

            # 3. Qualitative single-site density assertions are not reproducible.
            if not drop and _is_qualitative_density_assertion(assertion):
                drop = True

            # 4. Exact density float assertions are not reproducible from VLM output.
            if not drop and _is_exact_density_float_assertion(assertion):
                drop = True

            if drop:
                removed.append(assertion)
            else:
                kept.append(assertion)

        scenario["evaluation_criteria"]["nl_assertions"] = kept
    except (KeyError, TypeError):
        pass
    return removed


_TASK_MIN_CONDITIONAL_CUES = {
    "Density Comparison": 2,
    "Land Use Verification": 2,
    "Infrastructure Detection": 2,
    "Suitability": 2,
    "Encroachment": 2,
    "Urban Profile": 2,
}


def _count_conditional_cues(reason_for_call: str) -> int:
    """Count simple branch cues that indicate turn dependency."""
    text = reason_for_call.lower()
    cue_terms = [
        " if ",
        "if the",
        "if not",
        "otherwise",
        "instead",
        "in that case",
        "if the result",
        "if the agent",
        "if blocked",
        "if this fails",
    ]
    return sum(text.count(term) for term in cue_terms)


def score_turn_linkage(scenario: dict, difficulty: str = "medium") -> dict:
    """Compute a lightweight linkage score for multi-turn dependency quality."""
    tier = _DIFFICULTY_TIERS.get(difficulty, _DIFFICULTY_TIERS["medium"])
    try:
        reason = scenario["user_scenario"]["instructions"].get("reason_for_call", "")
        assertions = scenario["evaluation_criteria"].get("nl_assertions", [])
        actions = scenario["evaluation_criteria"].get("actions", [])
    except (KeyError, TypeError):
        return {
            "linkage_score": 0.0,
            "conditional_cues": 0,
            "step_count": 0,
            "cross_turn_mentions": 0,
            "action_count": 0,
        }

    reason_lower = reason.lower()
    step_count = reason.count("Step ")
    conditional_cues = _count_conditional_cues(reason)

    cross_turn_terms = [
        "after the agent",
        "based on",
        "if the result",
        "if the agent",
        "if that fails",
        "otherwise",
        "then ask",
    ]
    cross_turn_mentions = sum(reason_lower.count(term) for term in cross_turn_terms)

    bridging_terms = ["because", "therefore", "despite", "while", "however", "based on"]
    bridging_assertions = sum(
        1 for a in assertions if any(t in a.lower() for t in bridging_terms)
    )

    cond_score = min(conditional_cues / 3.0, 1.0)
    cross_turn_score = min(cross_turn_mentions / 3.0, 1.0)
    chain_score = min(len(actions) / tier["chain_divisor"], 1.0)
    assertion_score = (bridging_assertions / len(assertions)) if assertions else 0.0

    linkage_score = 0.4 * cond_score + 0.25 * cross_turn_score + 0.2 * chain_score + 0.15 * assertion_score
    return {
        "linkage_score": round(linkage_score, 3),
        "conditional_cues": conditional_cues,
        "step_count": step_count,
        "cross_turn_mentions": cross_turn_mentions,
        "action_count": len(actions),
        "bridging_assertions": bridging_assertions,
    }


def validate_reason_for_call(scenario: dict, task_name: str, difficulty: str = "medium") -> list[str]:
    """Validate that reason_for_call includes explicit turn dependency cues."""
    tier = _DIFFICULTY_TIERS.get(difficulty, _DIFFICULTY_TIERS["medium"])
    warnings = []
    try:
        reason = scenario["user_scenario"]["instructions"].get("reason_for_call", "")
    except (KeyError, TypeError):
        return ["Missing reason_for_call"]

    if not reason:
        return ["Empty reason_for_call"]

    cond_cues = _count_conditional_cues(reason)
    min_required = tier["min_cues"]
    if cond_cues < min_required:
        warnings.append(
            f"Weak dependency chain: found {cond_cues} conditional cues, need >= {min_required} for {task_name}"
        )

    reason_lower = reason.lower()
    if "after the agent" not in reason_lower and "based on" not in reason_lower:
        warnings.append("reason_for_call lacks explicit reference to prior agent output")

    return warnings


def validate_scenario(scenario: dict, db_context: dict) -> list[str]:
    """
    Post-generation validation checks. Returns a list of warning strings.
    """
    warnings = []

    # Check that all action names are valid tool names
    valid_tools = {t["name"] for t in _TOOL_DEFINITIONS}
    try:
        actions = scenario["evaluation_criteria"]["actions"]
        reason = scenario["user_scenario"]["instructions"].get("reason_for_call", "").lower()

        for action in actions:
            if action["name"] not in valid_tools:
                warnings.append(f"Invalid tool name: {action['name']}")

        # Check first action is get_satellite_tile
        if actions and actions[0]["name"] != "get_satellite_tile":
            warnings.append("First action is not get_satellite_tile")

        # Check that action info fields contain no conditional language (actions must be deterministic)
        conditional_phrases = ("if the", "if present", "if absent", "only if", "if confirmed", "if not")
        for action in actions:
            info = action.get("info", "").lower()
            if any(phrase in info for phrase in conditional_phrases):
                warnings.append(
                    f"Action '{action['name']}' has conditional language in its info field: \"{action.get('info', '')[:80]}\". "
                    "actions must be a flat deterministic list — resolve the branch using ground_truth and remove the other path."
                )

        # Check traceability: every action must be explicitly triggered by reason_for_call
        action_names = [a["name"] for a in actions]

        if "submit_site_assessment" in action_names:
            submit_cues = ("submit", "formal assessment", "file the assessment", "finalize")
            if not any(cue in reason for cue in submit_cues):
                warnings.append(
                    "submit_site_assessment is in actions but reason_for_call never asks the user "
                    "to submit/finalize the assessment. Either add a step or remove the action."
                )

        if "classify_land_use" in action_names:
            land_cues = ("land use", "land-use", "land_use", "categories", "classify")
            if not any(cue in reason for cue in land_cues):
                warnings.append(
                    "classify_land_use is in actions but reason_for_call never asks the user "
                    "to request land-use classification."
                )

        if "analyze_urban_density" in action_names:
            density_cues = ("density", "population", "density read", "density estimate")
            if not any(cue in reason for cue in density_cues):
                warnings.append(
                    "analyze_urban_density is in actions but reason_for_call never asks the user "
                    "to request a density estimate."
                )

        # (Metric-tool mismatches are filtered out by filter_nl_assertions; no extra warnings needed)
        assertions = scenario["evaluation_criteria"].get("nl_assertions", [])

        # Check that nl_assertions exist
        if not assertions:
            warnings.append("No nl_assertions found")

    except (KeyError, TypeError) as e:
        warnings.append(f"Malformed scenario structure: {e}")

    return warnings


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":

    # File paths
    csv_path = os.path.join(SCRIPT_DIR, "task_overall.csv")
    db_path = os.path.join(SCRIPT_DIR, "urban_satellite_db_Sydney.json")

    # Fallback to the deployed db.json if local copy doesn't exist
    if not os.path.exists(db_path):
        print(f"Local DB not found at {db_path}, falling back to deployed db.json")
        db_path = os.path.join(SCRIPT_DIR, "db.json")

    # Load shared resources
    task_categories = load_task_categories(csv_path)
    db = load_db(db_path)
    names = load_names(NAME_DATASET_PATH)

    print(f"DB: {db_path}")
    print(f"  sites: {len(db.get('sites', {}))}")
    print(f"  names: {len(names)}")
    print()

    # Optionally restrict to specific task categories
    if SELECTED_TASKS:
        task_categories = [t for t in task_categories if t["Task"] in SELECTED_TASKS]

    print(f"Tasks: {[t['Task'] for t in task_categories]}")
    print(f"NUM_GEN = {NUM_GEN}  →  {NUM_GEN * len(task_categories)} total scenarios\n")

    # Output directory
    output_dir = Path(SCRIPT_DIR) / "generated"
    output_dir.mkdir(exist_ok=True)

    all_generated: list[dict] = []

    # Tracks the `reason_for_call` of the most recently generated scenario
    # for each task, so the next round can actively avoid repeating it.
    prev_instructions: dict[str, str] = {}

    # Tracks site_ids already used per task to avoid image repetition (DEDUP mode).
    used_site_ids: dict[str, set[str]] = {}

    for gen_idx in range(NUM_GEN):
        print(f"{'='*60}")
        print(f"GENERATION ROUND {gen_idx + 1}/{NUM_GEN}")
        print(f"{'='*60}")

        rng = random.Random()

        for task_row in task_categories:
            # Determine output file path before calling the API so we can skip if it exists
            slug = _TASK_SLUGS.get(
                task_row["Task"],
                task_row["Task"].lower().replace(" ", "_")
            )
            diff_suffix = task_row.get("Difficulty", "medium")[:3]  # "eas" or "med"
            out_file = output_dir / f"{slug}_{diff_suffix}_gen{gen_idx + 1:02d}.json"
            if out_file.exists():
                print(f"  [SKIP] {out_file.name} already exists")
                # Still load the saved scenario to update prev_instructions for anti-repetition
                if ANTI_REPETITION:
                    try:
                        with open(out_file, encoding="utf-8") as _f:
                            _saved = json.load(_f)
                        reason = _saved["user_scenario"]["instructions"]["reason_for_call"]
                        prev_instructions[task_row["Task"]] = reason
                        all_generated.append(_saved)
                    except Exception:
                        pass
                continue

            # Pick a user name for this scenario
            user_name = rng.choice(names)

            # Sample a coherent context slice for this task
            db_context = sample_db_context(
                db, task_row["Task"], rng, user_name, task_row,
                used_site_ids=used_site_ids.get(task_row["Task"]) if DEDUP else None,
            )

            # Pick a variation angle — cycled to guarantee different personas across rounds
            axes = _VARIATION_AXES.get(task_row["Task"], [])
            variation = axes[gen_idx % len(axes)] if axes else None

            # Pass the previous round's instruction for anti-repetition
            prev_instr = prev_instructions.get(task_row["Task"]) if ANTI_REPETITION else None

            scenario = generate_urban_scenario(
                task_row, db_context, variation, prev_instr,
                increase_complexity=INCREASE_COMPLEXITY,
            )

            if scenario:
                # Assign a deterministic, human-readable ID
                scenario["id"] = f"urban_satellite_{slug}_{diff_suffix}_gen{gen_idx + 1:02d}"
                scenario["_gen_round"] = gen_idx + 1
                scenario["_difficulty"] = task_row.get("Difficulty", "medium")

                # Hard-inject critical simulator rules into task_instructions.
                # Done here in post-processing so they are always present,
                # regardless of whether the LLM included them.
                try:
                    instr = scenario["user_scenario"]["instructions"]
                    instr["task_instructions"] = (
                        instr["task_instructions"] + _SIMULATOR_CRITICAL_RULES
                    )
                except (KeyError, TypeError):
                    pass

                # Filter out nl_assertions that reference features not queried
                removed_assertions = filter_nl_assertions(scenario)
                if removed_assertions:
                    for ra in removed_assertions:
                        print(f"  [FILTERED assertion — feature not queried] {ra[:80]}")

                # Validate the generated scenario
                ctx_warnings = validate_scenario(scenario, db_context)
                if ctx_warnings:
                    for w in ctx_warnings:
                        print(f"  [WARN] {w}")

                # Validate dependency quality in reason_for_call
                difficulty = task_row.get("Difficulty", "medium")
                diff_tier = _DIFFICULTY_TIERS.get(difficulty, _DIFFICULTY_TIERS["medium"])
                linkage_warnings = validate_reason_for_call(scenario, task_row["Task"], difficulty)
                if linkage_warnings:
                    for w in linkage_warnings:
                        print(f"  [WARN - linkage] {w}")

                linkage_metrics = score_turn_linkage(scenario, difficulty)
                scenario["_linkage_metrics"] = linkage_metrics
                linkage_threshold = diff_tier["linkage_threshold"]
                if linkage_metrics["linkage_score"] < linkage_threshold:
                    print(
                        f"  [WARN - linkage score] {linkage_metrics['linkage_score']:.3f} "
                        f"< {linkage_threshold:.2f} ({difficulty})"
                    )

                if STRICT_MULTI_TURN_MODE and (
                    ctx_warnings
                    or linkage_warnings
                    or linkage_metrics["linkage_score"] < linkage_threshold
                ):
                    print("  [REJECT] scenario dropped due to structural or linkage issues")
                    continue

                # Remember this round's reason_for_call for anti-repetition
                try:
                    reason = scenario["user_scenario"]["instructions"]["reason_for_call"]
                    prev_instructions[task_row["Task"]] = reason
                except (KeyError, TypeError):
                    pass

                all_generated.append(scenario)

                # Track used sites for DEDUP
                if DEDUP:
                    task_key = task_row["Task"]
                    if task_key not in used_site_ids:
                        used_site_ids[task_key] = set()
                    for sid in db_context["sites"]:
                        used_site_ids[task_key].add(sid)

                # Save individual file
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(scenario, f, indent=2, ensure_ascii=False)
                print(f"  [OK] {out_file.name}")
            else:
                print(f"  [FAIL] {task_row['Task']}")

    # Save combined output
    combined_path = output_dir / "generated_all.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_generated, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Done. {len(all_generated)} scenarios saved to: {output_dir}")
    print(f"Combined file: {combined_path}")
