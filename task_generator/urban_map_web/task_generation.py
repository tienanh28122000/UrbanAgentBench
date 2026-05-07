import os
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# ==========================================
# CONFIG
# ==========================================
# Number of samples to generate per task category per run.
# Total scenarios produced = NUM_GEN × number_of_task_categories
NUM_GEN = 10

# Restrict generation to specific task names, or set to None to generate all.
# Example: SELECTED_TASKS = ["Temporal Event Coordination", "Civic Issue Reporting"]
SELECTED_TASKS = None
# SELECTED_TASKS = ["Temporal Event Coordination"]

# When True, each generation round (after the first) receives the previous
# round's scenario summary so the LLM actively avoids repeating it.
# Set to False (default) to generate each scenario independently.
ANTI_REPETITION = True

# Number of most-recent reason_for_call texts to inject into the anti-repetition
# prompt. Larger N = more diverse but longer prompts. 3 is a good default.
ANTI_REPETITION_TOP_N = 3

# When True, the LLM is instructed to generate a more complex, harder scenario:
#   - Longer tool-call chains (more reasoning steps)
#   - Multi-entity / multi-constraint goals
#   - Conditional replanning (e.g. first slot unavailable → find alternative)
#   - Cross-source reconciliation (db metadata vs. webpage may differ)
# Set to False (default) to generate standard-difficulty scenarios.
INCREASE_COMPLEXITY = True

# The probability (0.0 to 1.0) of forcing a negative outcome scenario (e.g. service
# fails or unavailable) to test agent failure handling. Evaluated independently.
NEGATIVE_OUTCOME_RATIO = 0.25

# Maximum number of NL assertions to generate per scenario.
MAX_NL_ASSERTIONS = 8

# Max number of search-along-route places included per context slice.
# Origin + dest are always kept; this caps the bonus SAR places.
MAX_SAR_PLACES = 2
# Max number of webpages included per context slice.
# Webpages are the largest items; keeping ≤2 avoids blowing the context window.
MAX_WEBPAGES = 4  # Increased from 2: en_route needs origin+dest+up-to-2 SAR place pages
# For Contextual Spatial Filtering, how many extra random places to add
# so the LLM has enough variety to build a realistic nearby_search scenario.
SPATIAL_EXTRA_PLACES = 3

# Initialize OpenAI Client (Make sure to set OPENAI_API_KEY in your environment variables)
load_dotenv()
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.environ.get("OPENROUTER_API_KEY"),
)
MODEL = "gpt-4.1-mini"  # Use the mini variant for faster generation; switch to "gpt-4.1" for higher quality

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Primary db: output from data_crawler.py (richer, larger dataset)
_CRAWLER_DB  = os.path.join(SCRIPT_DIR, "db_newyork.json")

# These rules are appended verbatim to every generated sample's task_instructions
# in post-processing, so the user simulator always has them regardless of what
# the LLM chose to include.
_SIMULATOR_CRITICAL_RULES = (
    "\nINTERACTION CRITICAL RULES FOR YOU:"
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

# Short slugs used in generated scenario IDs
_TASK_SLUGS = {
    "Place Discovery & Web Fact Verification":   "discovery",
    "Contextual Spatial Filtering":              "spatial_filter",
    "En-Route Logistics Optimization":           "en_route",
    "Transit Planning":                          "transit",
    "Temporal Event Coordination":               "event_transit",
    "Civic Issue Reporting":                     "civic",
    "Unstructured Web Evaluation & Reservation": "booking",
    "Multi-Constraint Itinerary Synthesis":      "itinerary",
}

# Variation axes for scenario diversity.
# Each entry defines a distinct user angle (persona + constraint + request style).
# One entry is picked per generation round so that NUM_GEN scenarios for the
# same task cover genuinely different user stories even on identical db data.
_VARIATION_AXES: dict[str, list[dict]] = {
    "Place Discovery & Web Fact Verification": [
        {"persona": "a first-time tourist unfamiliar with the area",
         "constraint": "wants to verify opening hours and accessibility before making the trip",
         "style": "asks broad questions and follows up with clarifying questions"},
        {"persona": "a local resident double-checking a favourite spot",
         "constraint": "needs to confirm one specific amenity (e.g. parking, wifi, outdoor seating)",
         "style": "direct and concise, already knows the place name"},
        {"persona": "a journalist writing a city travel guide",
         "constraint": "needs official, verifiable facts — no hearsay",
         "style": "asks for exact figures and authoritative sources"},
        {"persona": "a parent planning a family outing",
         "constraint": "must confirm the venue is child-friendly and physically accessible",
         "style": "conversational, lists multiple requirements across turns"},
        {"persona": "a business traveller with only 30 minutes between meetings",
         "constraint": "needs a quick fact-check — no browsing, just the answer",
         "style": "terse, impatient, single focused question per turn"},
        {"persona": "an event planner scouting venues for a private function",
         "constraint": "needs capacity, accessibility, and catering policy confirmed from the official website",
         "style": "professional tone, asks detailed follow-up questions per attribute"},
        {"persona": "a travel blogger comparing two competing cafes",
         "constraint": "wants rating, amenity list, and unique selling point for each place",
         "style": "comparative, asks side-by-side questions"},
        {"persona": "a student researching urban services for a university project",
         "constraint": "needs factual data: location, type, accessibility score, and operating hours",
         "style": "systematic, asks one attribute at a time"},
        {"persona": "a hotel concierge answering a guest's question about a nearby attraction",
         "constraint": "must give accurate, up-to-date information — no guessing",
         "style": "professional, verifies from official source before answering"},
        {"persona": "a person with mobility impairment planning their first visit",
         "constraint": "must confirm lift access, step-free entry, and accessible restroom before committing",
         "style": "specific and persistent, asks follow-up if answer is vague"},
        {"persona": "a real estate agent checking nearby amenity quality for a prospective buyer",
         "constraint": "needs verified amenity data and ratings to include in a property report",
         "style": "concise, wants data points rather than narrative descriptions"},
    ],
    "Contextual Spatial Filtering": [
        {"persona": "a tourist looking for highly-rated cafes nearby",
         "constraint": "must have a rating above 4.0 and offer wifi",
         "style": "exploratory, open to agent suggestions"},
        {"persona": "a wheelchair user searching for accessible venues",
         "constraint": "accessibility information is the primary filter",
         "style": "specific and firm about accessibility requirements"},
        {"persona": "a student on a tight budget",
         "constraint": "wants the closest affordable option within walking distance",
         "style": "asks for the cheapest option first, then refines"},
        {"persona": "a food blogger discovering local gems",
         "constraint": "wants places with unique character, not chain brands",
         "style": "asks for less-obvious suggestions with high ratings"},
        {"persona": "a remote worker needing a productive workspace",
         "constraint": "requires quiet atmosphere, strong wifi, and good coffee",
         "style": "lists multiple simultaneous requirements upfront"},
        {"persona": "a family with young children looking for a safe outdoor seating area",
         "constraint": "must have outdoor seating, be child-friendly, and have a rating above 4.2",
         "style": "asks about specific attributes in sequence across turns"},
        {"persona": "a night-shift worker wanting a late-night option",
         "constraint": "place must be open past 22:00 and within 500m",
         "style": "direct, asks about opening hours first, then filters by distance"},
        {"persona": "a fitness enthusiast wanting a healthy café post-workout",
         "constraint": "needs a place with healthy food options and high rating, close to current location",
         "style": "energetic and decisive, picks the top recommendation"},
        {"persona": "a vegan traveller looking for plant-based options",
         "constraint": "must verify amenity tags include vegan-friendly or plant-based menu",
         "style": "cautious, asks about amenity tags explicitly before deciding"},
        {"persona": "an architect doing a site visit wanting a nearby lunch spot",
         "constraint": "needs a place within 300m, with seating and a rating above 4.0",
         "style": "time-pressed, wants the nearest qualifying option immediately"},
        {"persona": "a group of colleagues choosing a post-meeting coffee spot",
         "constraint": "must seat at least 6 people, have high rating, and be within 10 minutes walk",
         "style": "collaborative, asks about capacity and rating before distance"},
    ],
    "En-Route Logistics Optimization": [
        {"persona": "a driver on a road trip who needs a coffee break",
         "constraint": "wants a stop no more than 5 minutes off the planned route",
         "style": "wants efficiency — one stop, in and out"},
        {"persona": "a delivery person looking for a quick lunch on a route",
         "constraint": "limited window — must be near the driving path, not a detour",
         "style": "practical and time-focused"},
        {"persona": "a couple on a road trip wanting a scenic detour",
         "constraint": "open to slightly longer detours if the place is highly rated",
         "style": "casual and exploratory, fine with browsing options"},
        {"persona": "a parent driving children and needing a family-friendly rest stop",
         "constraint": "needs a place suitable for children with seating area",
         "style": "asks about family-friendly features explicitly"},
        {"persona": "a business traveller driving between cities",
         "constraint": "must reach destination on time — stop must be minimal",
         "style": "precise, wants ETA impact of the stop stated clearly"},
        {"persona": "a long-haul truck driver needing a break within legal driving limits",
         "constraint": "must stop within the next 50km of the route — any qualifying place will do",
         "style": "no-frills, just the nearest option along the road"},
        {"persona": "a cyclist on a long-distance ride needing a refuel stop",
         "constraint": "must be reachable without significant elevation detour, high-energy food preferred",
         "style": "practical, mentions dietary energy needs"},
        {"persona": "a rideshare driver picking up a snack between fares",
         "constraint": "must be a drive-through or quick walk-in — maximum 5 minutes total stop",
         "style": "rushed, wants the single fastest option"},
        {"persona": "a tourist hiring a car for the first time, nervous about detours",
         "constraint": "wants a stop that is directly on the computed route with no navigation changes",
         "style": "anxious, asks for reassurance that the stop won't cause them to get lost"},
        {"persona": "a catering manager driving supplies who can stop for a brief break",
         "constraint": "needs a place with parking for a large vehicle near the route",
         "style": "practical, asks about parking capacity explicitly"},
        {"persona": "a road-tripper tracking fuel and time budget carefully",
         "constraint": "wants the stop with the best rating that adds less than 10 minutes to journey",
         "style": "data-driven, asks for options ranked by rating vs. detour time"},
    ],
    "Transit Planning": [
        {"persona": "a commuter planning their first week on a new route",
         "constraint": "needs reliable schedule with earliest and latest options",
         "style": "methodical, asks about multiple departure times"},
        {"persona": "a tourist who never uses public transit",
         "constraint": "needs clear, step-by-step guidance on which line to take",
         "style": "asks basic questions, needs reassurance"},
        {"persona": "an elderly person who moves slowly and needs extra buffer time",
         "constraint": "must catch a transit departure with at least 15 minutes to spare",
         "style": "cautious, asks about transfer times and complexity"},
        {"persona": "a student with a flexible schedule wanting the cheapest option",
         "constraint": "wants to compare transit vs walking cost and time",
         "style": "asks comparative questions across options"},
        {"persona": "a professional rushing to a meeting",
         "constraint": "must arrive by a hard deadline — first available departure",
         "style": "urgent, wants only the fastest option with no alternatives"},
        {"persona": "a parent travelling with a pushchair and young child",
         "constraint": "needs a step-free transit route with sufficient standing space",
         "style": "asks about accessibility and carriage type explicitly"},
        {"persona": "a night-shift nurse finishing work at an unusual hour",
         "constraint": "needs the last available departure or nearest 24h alternative",
         "style": "tired but precise, asks about last departures and frequency"},
        {"persona": "a conference attendee arriving in an unfamiliar city",
         "constraint": "needs to get from the airport-adjacent origin to the conference venue on time",
         "style": "asks about travel time, price, and frequency in one turn"},
        {"persona": "a teenager travelling alone for the first time",
         "constraint": "needs a simple, low-transfer route and reassurance about safety",
         "style": "asks follow-up questions about each step before proceeding"},
        {"persona": "a cyclist who missed their last bike connection",
         "constraint": "needs transit that allows bicycle boarding",
         "style": "asks specifically about bike-on-transit policy"},
        {"persona": "a freelancer working remotely who wants to ride during off-peak hours",
         "constraint": "wants a departure that avoids peak and arrives before a soft noon deadline",
         "style": "relaxed, asks about schedule spread across morning slots"},
    ],
    "Temporal Event Coordination": [
        {"persona": "a music fan trying to catch a live event after work",
         "constraint": "must leave work at a fixed time and arrive before event starts",
         "style": "asks about transit timing first, then event start time"},
        {"persona": "a tourist wanting to combine sightseeing with a special event",
         "constraint": "wants to attend the event and still catch a specific last train home",
         "style": "asks about event duration and last transit departure"},
        {"persona": "a couple planning a special evening out",
         "constraint": "wants to arrive early for a good seat, not just on time",
         "style": "asks for the transit option that arrives 30 minutes before event"},
        {"persona": "a student group organising a trip to an event",
         "constraint": "multiple people arriving from the same origin — needs group logistics",
         "style": "asks about transit frequency, not just first departure"},
        {"persona": "a parent taking children to a daytime event",
         "constraint": "must return home before a strict evening curfew",
         "style": "asks about event end time and available return transit"},
        {"persona": "a solo traveller trying to attend a one-night-only pop-up event",
         "constraint": "must confirm event is still scheduled before committing to transit",
         "style": "verifies event details before asking about transit"},
        {"persona": "an office worker surprising their partner with evening plans",
         "constraint": "needs to arrive at venue no later than event start without their partner knowing the destination",
         "style": "asks about transit options for a general area, avoids naming the venue initially"},
        {"persona": "a wheelchair user attending an accessible event",
         "constraint": "must confirm the event venue is accessible before planning transit",
         "style": "asks about venue accessibility first, then transit step-free options"},
        {"persona": "a photographer covering an event professionally",
         "constraint": "must arrive 45 minutes before event start for setup",
         "style": "precise, calculates backward from event start time"},
        {"persona": "a retiree attending a cultural event for the first time",
         "constraint": "needs simple transit with no transfers and plenty of departure time buffer",
         "style": "asks about the simplest possible route and confirms each step"},
        {"persona": "a student on a very tight budget trying to attend a free event",
         "constraint": "needs the cheapest transit option that still arrives before event starts",
         "style": "asks about price first, then confirms timing"},
    ],
    "Civic Issue Reporting": [
        {"persona": "a resident frustrated by a recurring infrastructure problem",
         "constraint": "wants to formally lodge a complaint on behalf of their street",
         "style": "detailed and assertive, provides precise location info"},
        {"persona": "a shop owner affected by a nearby issue hurting their business",
         "constraint": "wants a report submitted with urgency flag",
         "style": "professional tone, references business impact"},
        {"persona": "a newcomer to the neighbourhood unfamiliar with reporting channels",
         "constraint": "needs the agent to guide them through the entire process",
         "style": "asks step-by-step, needs confirmation at each stage"},
        {"persona": "a commuter reporting a hazard spotted during their daily route",
         "constraint": "wants to report quickly during their commute — minimal back-and-forth",
         "style": "brief, wants the agent to fill in details from context"},
        {"persona": "a local councillor's assistant documenting multiple issues",
         "constraint": "needs to file a formal report with category and description",
         "style": "precise, uses official language, asks for report ID confirmation"},
        {"persona": "a teacher reporting a safety hazard near a school",
         "constraint": "wants the report to mention proximity to a school and flag priority",
         "style": "concerned and urgent, asks whether the report will be escalated"},
        {"persona": "an elderly resident reporting poor street lighting",
         "constraint": "wants to describe the exact location clearly and get confirmation the report was filed",
         "style": "patient but persistent, asks for the report reference number"},
        {"persona": "a cyclist reporting a dangerous pothole on a bike lane",
         "constraint": "wants to categorise the hazard specifically as a road surface defect",
         "style": "specific about category, provides GPS-level location detail"},
        {"persona": "a parent reporting a broken playground fixture",
         "constraint": "wants to ensure the report is assigned to the parks maintenance team specifically",
         "style": "asks whether the right department will receive the report"},
        {"persona": "a business association representative reporting a street flooding issue",
         "constraint": "needs to submit on behalf of multiple affected parties, wants a group complaint filed",
         "style": "formal, references multiple complainants and asks for case tracking"},
        {"persona": "a delivery driver reporting an illegally blocked loading bay",
         "constraint": "wants an immediate response — reports it as an active obstruction",
         "style": "urgent and concise, asks if the report can trigger same-day action"},
    ],
    "Unstructured Web Evaluation & Reservation": [
        {"persona": "a food enthusiast wanting to try a specific dish before booking",
         "constraint": "must confirm the dish is on the menu from the website before reserving",
         "style": "asks about menu content first, then proceeds to booking"},
        {"persona": "a traveller with dietary restrictions who needs to verify the menu",
         "constraint": "must find a vegetarian/vegan option on the website before committing",
         "style": "cautious — won't book unless specific dietary need is confirmed"},
        {"persona": "a romantic partner planning a surprise dinner",
         "constraint": "wants to verify ambience and type of cuisine from the website",
         "style": "asks qualitative questions about the venue before booking"},
        {"persona": "an office manager booking lunch for a team meeting",
         "constraint": "needs to confirm group capacity and booking availability",
         "style": "professional, confirms group size and time slot"},
        {"persona": "a budget-conscious diner wanting to check prices before reserving",
         "constraint": "needs to read the webpage for pricing, then decides whether to book",
         "style": "price-sensitive, asks about value before committing to booking"},
        {"persona": "a tourist with a nut allergy needing allergen information from the menu",
         "constraint": "will not book without explicit allergen confirmation from the official website",
         "style": "safety-focused, asks specifically about allergen labelling"},
        {"persona": "a social media influencer scouting a venue for a sponsored post",
         "constraint": "needs to verify the venue's aesthetic and unique offerings from the website",
         "style": "asks about vibe, photography opportunities, and signature items"},
        {"persona": "a newly-engaged couple booking their first anniversary dinner",
         "constraint": "wants a private or semi-private dining option confirmed from the website",
         "style": "romantic and attentive to detail, asks multiple follow-up questions"},
        {"persona": "a nutritionist booking a client consultation venue",
         "constraint": "needs to confirm the venue offers healthy menu options per the website",
         "style": "professional, references specific nutritional criteria"},
        {"persona": "a solo diner visiting the city for one evening only",
         "constraint": "wants to confirm there is a bar-seating or walk-in option from the website before booking",
         "style": "flexible but wants written confirmation before committing"},
        {"persona": "a corporate assistant booking a client dinner for an executive",
         "constraint": "must confirm dress code and private room availability from the website, then book formally",
         "style": "formal and thorough, confirms every detail in sequence"},
    ],
    "Multi-Constraint Itinerary Synthesis": [
        {"persona": "a tourist planning a full-day itinerary in an unfamiliar city",
         "constraint": "must fit transit, a meal booking, and an event into one day",
         "style": "asks the agent to suggest an optimised order of activities"},
        {"persona": "a local planning a special occasion day out",
         "constraint": "must combine a route optimised stop-off, a booked dinner, and an evening event",
         "style": "has a vision, asks agent to fill in logistics gaps"},
        {"persona": "a business traveller with a free afternoon between flights",
         "constraint": "strict departure deadline — must book and attend within a 4-hour window",
         "style": "time-boxed, wants the agent to account for travel time at every step"},
        {"persona": "a couple celebrating an anniversary with a curated day",
         "constraint": "wants a scenic route, a romantic restaurant booking confirmed via website, and an evening concert",
         "style": "romantic and particular, verifies website details before committing"},
        {"persona": "a group of friends coordinating a spontaneous outing",
         "constraint": "need to agree on a meeting point, find a restaurant with availability, and get to an event in time",
         "style": "collaborative, asks for options and confirms choices step by step"},
        {"persona": "a parent planning a child's birthday outing",
         "constraint": "must find a child-friendly café stop en route, book a venue for the party, and reach a family event by a set time",
         "style": "warm and organised, confirms suitability at each step"},
        {"persona": "a solo traveller on a 48-hour city break maximising experiences",
         "constraint": "must combine the most efficient route, a must-visit restaurant reservation, and a cultural event",
         "style": "enthusiastic, asks agent to optimise the full day in one go"},
        {"persona": "a wellness blogger documenting a mindful city day",
         "constraint": "must include a healthy café stop en route, a booking at a wellness venue, and a sunset event",
         "style": "holistic, asks about each element's quality as well as logistics"},
        {"persona": "a corporate event planner scouting a city for an off-site day",
         "constraint": "must book a lunch venue, arrange transit for a group, and confirm an evening event capacity",
         "style": "methodical, asks about each leg in sequence with ETA confirmation"},
        {"persona": "an expat hosting visiting family for a day trip",
         "constraint": "must plan a scenic driving route with a café stop, confirm a dinner booking via website, and catch a local cultural event",
         "style": "considerate of guests' comfort, asks about accessibility and quality"},
        {"persona": "a travel journalist researching a city in a single tight day",
         "constraint": "needs to verify all venue details from official websites, make at least one booking, and attend a timed event",
         "style": "investigative, asks for source confirmation at each step"},
    ],
}


# ==========================================
# COMPLEXITY ESCALATION HINTS
# ==========================================
# Per-task instructions injected into the LLM prompt when INCREASE_COMPLEXITY=True.
# Each entry describes WHAT to add to make the scenario noticeably harder:
#   • more intermediate tool-call steps
#   • multi-entity / multi-constraint goals
#   • conditional replanning (first choice fails, agent must recover)
#   • cross-source reconciliation (db data vs. webpage may diverge)
_COMPLEXITY_HINTS: dict[str, str] = {
    "Place Discovery & Web Fact Verification":
        "Compare TWO different places that both appear in the database context. "
        "The user needs to verify at least THREE attributes per place (e.g. opening hours, "
        "accessibility, a specific service/feature). For whichever place has a webpage in "
        "the context, cross-reference its db metadata against the actual webpage content: "
        "the agent must explicitly confirm that what the db says matches what the website says "
        "(e.g. confirm opening hours shown in db are consistent with the website). "
        "Do NOT invent discrepancies — only use attributes that genuinely appear in both sources. "
        "Require at least 5 steps in reason_for_call and at least 5 actions.",

    "Contextual Spatial Filtering":
        "MANDATORY PREREQUISITE: The action sequence MUST start with text_search to resolve the "
        "user's reference landmark (e.g. 'Southbank Parklands') into a place_id, then place_details "
        "to extract the lat/lng coordinates of that landmark — ONLY then can nearby_search be called "
        "using those coordinates. Never start with nearby_search directly. "
        "Apply at least THREE simultaneous filters (e.g. rating threshold, specific place_type, "
        "AND an amenity filter), not just one or two. "
        "After the agent returns results, have the user add a new constraint "
        "(e.g. 'actually it must also be within 400 m') so the agent must re-run "
        "nearby_search with tightened parameters. "
        "Minimum action sequence: text_search → place_details → nearby_search → [place_details on result] → nearby_search (tightened). "
        "Require at least 5 steps in reason_for_call and at least 5 actions.",

    "En-Route Logistics Optimization":
        "The user needs TWO distinct stops along the route (e.g. a coffee shop AND a "
        "pharmacy/fuel station), each with its own constraint (rating, detour limit). "
        "They also have a hard arrival deadline, so the cumulative detour for both stops "
        "must stay within budget. "
        "COMPARISON & RECOMMENDATION REQUIRED: The agent must explicitly compare the available "
        "search-along-route results (listing at least two candidates with their ratings and "
        "estimated detour times) and recommend the BEST option for each stop, clearly "
        "justifying the choice (e.g. 'I recommend Stop A — it has the highest rating (4.7) "
        "among the on-route options and only adds 3 minutes to your journey'). "
        "If the top-rated option would exceed the budget, the agent must fall back to the "
        "next-best alternative and explain why. "
        "Require at least 6 steps in reason_for_call and at least 5 actions.",

    "Transit Planning":
        "The user has THREE simultaneous constraints for the single-origin journey: "
        "a strict arrival deadline, an accessibility requirement (step-free / wheelchair), "
        "AND a budget ceiling. The agent must look up the transit schedule, enumerate "
        "the available departures, and reason through which departure satisfies ALL three "
        "constraints — not just the earliest one. "
        "After choosing the outbound departure, the user also asks about the LAST return "
        "departure of the day (same route reversed) using the same transit schedule, "
        "so the agent must reason about end-of-day timing as well. "
        "NOTE: only transit_schedules[origin_id] is available — do NOT invent schedules "
        "for other stops or transfer points. "
        "Require at least 6 steps in reason_for_call and at least 6 actions.",

    "Temporal Event Coordination":
        "Focus on the SINGLE event at the destination place that exists in the database. "
        "Make the scenario complex by stacking multiple timing concerns around that one event: "
        "(1) the user must identify which transit departure from origin arrives with a comfortable "
        "buffer BEFORE the event start (not just on time); "
        "(2) they need to verify the event venue is accessible (check place_details for "
        "accessibility fields) before committing to transit; "
        "(3) they must confirm there is a return transit departure AFTER the event ends "
        "using only the origin's transit schedule, reasoning about the return leg timing; "
        "(4) the user also has a hard personal curfew, so the agent must confirm the last "
        "viable return departure satisfies that curfew. "
        "NOTE: only events[dest_id] is in context — do NOT invent a second event venue. "
        "Require at least 6 steps in reason_for_call and at least 6 actions.",

    "Civic Issue Reporting":
        "The user wants to report TWO or THREE distinct but related issues "
        "(e.g. broken street light + flooded footpath + damaged sign) at different "
        "nearby points on the same street. They use place_details to pin down the "
        "exact coordinates of each hazard before submitting each report, and they "
        "want confirmation (report ID or status) for every submission. "
        "Require at least 6 steps in reason_for_call and at least 5 actions. "
        "CRITICAL — `submit_council_report` `issue_type` MUST be one of these exact values: "
        "'pothole', 'graffiti', 'lighting', 'waste', 'other'. "
        "Do NOT invent other values (e.g. 'broken street light', 'flooded footpath', 'damaged sign' are INVALID). "
        "Map natural-language issues to the closest valid enum: broken/faulty lighting → 'lighting'; "
        "water/flooding → 'waste'; cracks/road surface → 'pothole'; vandalism/paint → 'graffiti'; "
        "anything else → 'other'.",

    "Unstructured Web Evaluation & Reservation":
        "The user must verify at least THREE attributes from the venue website "
        "(e.g. a specific menu item, allergen information, AND ambience/private-room policy) "
        "before they are willing to book. "
        "After reading the website, the user asks the agent to check availability for their "
        "FIRST preferred time slot. Whatever result comes back, the user then asks the agent "
        "to compare that slot with at least ONE alternative slot from the same place's "
        "available_slots data, and chooses the slot that best fits their schedule before booking. "
        "This forces the agent to enumerate and reason over multiple time slots rather than "
        "blindly booking the first one. "
        "Do NOT fabricate unavailability — use only the actual available_slots values in the "
        "database context. "
        "Require at least 6 steps in reason_for_call and at least 6 actions.",

    "Multi-Constraint Itinerary Synthesis":
        "Plan a full-day itinerary with AT LEAST FOUR distinct legs: "
        "(1) compute the driving route with an en-route stop, "
        "(2) verify a restaurant via its website (check 3+ attributes) and book it, "
        "(3) attend a timed event at the destination, and "
        "(4) plan return transit within a strict curfew. "
        "Mid-conversation, the user tightens one constraint (e.g. brings the return "
        "deadline forward by 30 minutes), forcing the agent to recheck the last leg. "
        "Require at least 8 steps in reason_for_call and at least 8 actions.",
}

# ==========================================
# KNOWN_INFO TEMPLATES
# ==========================================
# Task-specific required parameters that MUST be extracted into known_info
# by the LLM. These define the minimum set of contextual facts the user
# simulator needs to proceed successfully.
_KNOWN_INFO_TEMPLATES: dict[str, list[str]] = {
    "Place Discovery & Web Fact Verification": [
        "Verification attributes (e.g., opening hours, accessibility, parking)",
        "Specific venue name or landmark reference",
        "Search/verification deadline or context",
    ],
    "Contextual Spatial Filtering": [
        "Current location or reference point (lat/lng or landmark)",
        "Primary filter criteria (rating threshold, distance radius)",
        "Amenity or accessibility filters (wifi, parking, step-free, etc.)",
        "Place type or category preference",
        "Budget or price constraints (if any)",
    ],
    "En-Route Logistics Optimization": [
        "Origin and destination points",
        "Expected detour time budget or tolerance",
        "Stop type or service requirement (coffee, fuel, rest, etc.)",
        "Arrival deadline or time constraint",
        "Vehicle or accessibility type (if relevant)",
    ],
    "Transit Planning": [
        "Journey origin and destination",
        "Arrival deadline or time window",
        "Accessibility requirements (step-free, wheelchair, etc.)",
        "Passenger type or group size (if relevant)",
        "Budget or fare type constraint",
    ],
    "Temporal Event Coordination": [
        "Event location and name",
        "Event start time and date",
        "Event end time (for return leg planning)",
        "User's personal curfew or hard deadline",
        "Departure/arrival buffer preference (e.g., early arrival)",
    ],
    "Civic Issue Reporting": [
        "Issue type or hazard category",
        "Exact location (landmark or coordinates)",
        "Issue severity or urgency level",
        "Affected area or street name",
        "Number of related issues (if reporting multiple)",
    ],
    "Unstructured Web Evaluation & Reservation": [
        "Venue name and type (restaurant, cafe, etc.)",
        "Preferred reservation time and date",
        "Party size and composition",
        "Dietary restrictions or allergies",
        "Special requests or amenity requirements (private room, wifi, etc.)",
        "Budget or price sensitivity",
    ],
    "Multi-Constraint Itinerary Synthesis": [
        "Trip origin and final destination",
        "Overall time budget or schedule window",
        "Return deadline or curfew",
        "Key stop(s) or attraction(s)",
        "Meal booking time and party size",
        "Event location and timing",
        "Accessibility or group constraints",
    ],
}

# ==========================================
def load_task_categories(csv_path: str) -> list[dict]:
    """Load all task rows from task_overall.csv."""
    tasks = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(row)
    return tasks


def load_tools(tools_path: str) -> list[dict]:
    """Load tool definitions from tools.json."""
    with open(tools_path, encoding='utf-8') as f:
        return json.load(f)


def load_db(db_path: str) -> dict:
    """Load database context from db.json."""
    with open(db_path, encoding='utf-8') as f:
        return json.load(f)


# ==========================================
# COHERENT CONTEXT SAMPLER
# ==========================================
# The crawler (data_crawler.py) builds db.json in per-iteration clusters:
#
#   Each iteration picks (origin_landmark, dest_landmark) and produces:
#     routes[route_key]              — route between them, carries polyline
#     search_along_routes[key]       — place_ids along the polyline
#                                      key = polyline[:10] + "_" + place_type
#     transit_schedules[origin_id]   — bus/tram/train schedule at origin
#     events[dest_id]                — events happening at destination
#     places[*]                      — full details for origin, dest, SAR places
#     webpages[url]                  — markdown for places that have website_url
#     users[user_id]                 — a random user created each iteration
#
# Linkage rules:
#   route_key  = f"{round(lat,3)},{round(lng,3)}|{round(lat,3)},{round(lng,3)}|{mode}"
#   sar_key    = f"{polyline[:10]}_{place_type}"
#   webpages   = keyed by place.website_url




def _build_clusters(db: dict) -> list[dict]:
    """
    Reconstruct the coherent per-crawl-iteration clusters from db.json.

    Returns a list of cluster dicts, each containing:
      origin_id, dest_id, mode, route_key, sar_key, sar_place_ids, all_place_ids
    """
    # Build (rounded_lat, rounded_lng) → place_id lookup
    latlng_index: dict[tuple, str] = {}
    for pid, place in db.get("places", {}).items():
        lat = round(place["location"]["lat"], 3)
        lng = round(place["location"]["lng"], 3)
        latlng_index[(lat, lng)] = pid

    clusters: list[dict] = []
    for route_key, route in db.get("routes", {}).items():
        parts = route_key.split("|")
        if len(parts) != 3:
            continue
        try:
            o_lat, o_lng = map(float, parts[0].split(","))
            d_lat, d_lng = map(float, parts[1].split(","))
        except ValueError:
            continue

        origin_id = latlng_index.get((o_lat, o_lng))
        dest_id   = latlng_index.get((d_lat, d_lng))
        if not origin_id or not dest_id:
            continue

        polyline = route.get("polyline", "")

        # Match search_along_routes by polyline prefix (polyline[:10] + "_")
        sar_key: str | None = None
        sar_place_ids: list[str] = []
        if polyline:
            prefix = polyline[:10]
            for key, pids in db.get("search_along_routes", {}).items():
                if key.startswith(prefix):
                    sar_key = key
                    sar_place_ids = list(pids)
                    break

        # Cap SAR places to keep context size bounded
        sar_place_ids = sar_place_ids[:MAX_SAR_PLACES]
        all_place_ids = list({origin_id, dest_id} | set(sar_place_ids))
        clusters.append({
            "origin_id": origin_id,
            "dest_id":   dest_id,
            "mode":      parts[2],
            "route_key": route_key,
            "sar_key":   sar_key,
            "sar_place_ids": sar_place_ids,
            "all_place_ids": all_place_ids,
        })

    return clusters


def sample_db_context(db: dict, task_name: str, rng: random.Random | None = None) -> dict:
    """
    Sample a coherent db context slice tailored to the given task type.

    Strategy:
      1. Reconstruct all crawl clusters from db["routes"].
      2. Apply task-specific eligibility filters (each filter ensures the
         cluster contains EXACTLY the data needed to generate a valid scenario).
      3. Pick one eligible cluster at random.
      4. Assemble a context dict containing only the sections relevant to that
         task (no irrelevant data to mislead the LLM).
    """
    if rng is None:
        rng = random.Random()

    clusters = _build_clusters(db)

    # ------------------------------------------------------------------ #
    # Helper predicates (operate on a cluster dict + the full db)
    # ------------------------------------------------------------------ #

    def _has_webpage(c: dict) -> bool:
        """At least one cluster place has a webpage entry in db."""
        return any(
            db["places"].get(pid, {}).get("website_url") in db.get("webpages", {})
            for pid in c["all_place_ids"]
            if db["places"].get(pid, {}).get("website_url")
        )

    def _has_transit(c: dict) -> bool:
        """Origin place has a non-empty transit schedule."""
        sched = db.get("transit_schedules", {}).get(c["origin_id"], {})
        return bool(sched)

    def _has_events(c: dict) -> bool:
        """Destination place has at least one event."""
        return bool(db.get("events", {}).get(c["dest_id"]))

    def _has_sar(c: dict) -> bool:
        """Cluster has a search_along_route result with at least one place."""
        return bool(c.get("sar_key") and c.get("sar_place_ids"))

    def _has_amenity_data(c: dict) -> bool:
        """At least one place has amenity or accessibility fields (for spatial filtering)."""
        return any(
            db["places"].get(pid, {}).get("amenities") or
            db["places"].get(pid, {}).get("accessibility")
            for pid in c["all_place_ids"]
        )

    def _has_webpage_and_slots_same_place(c: dict) -> bool:
        """
        At least one place has BOTH a webpage (for read_place_website) AND
        non-zero available_slots (for check_availability / book_place).
        This must be the SAME place — both steps target the same entity.
        """
        for pid in c["all_place_ids"]:
            p = db["places"].get(pid, {})
            url = p.get("website_url")
            has_web = bool(url and url in db.get("webpages", {}))
            has_slots = any(v > 0 for v in p.get("available_slots", {}).values())
            if has_web and has_slots:
                return True
        return False

    def _has_user() -> bool:
        """DB contains at least one user (needed for write operations)."""
        return bool(db.get("users"))

    def _transit_events_dates_overlap(c: dict) -> bool:
        """
        At least one date in the origin's transit schedule also has an event
        at the destination (needed for temporal coordination scenario).
        """
        transit_dates = set(db.get("transit_schedules", {}).get(c["origin_id"], {}).keys())
        event_dates   = {ev.get("date") for ev in db.get("events", {}).get(c["dest_id"], [])}
        return bool(transit_dates & event_dates)

    # ------------------------------------------------------------------ #
    # Per-task eligibility filters
    #
    # Each filter ensures the cluster contains the minimum data required
    # to generate a valid, grounded conversation scenario for that task.
    # ------------------------------------------------------------------ #
    _FILTERS: dict[str, object] = {
        # Resolving entity → verify metadata or read official website.
        # Must have: place with webpage (for read_place_website path) and
        # accessibility/amenity data (for metadata verification path).
        "Place Discovery & Web Fact Verification":
            lambda c: _has_webpage(c) and _has_amenity_data(c),

        # nearby_search with lat/lng, type, min_rating, amenity filters.
        # Must have: ≥3 places total (including extras) with amenity data.
        "Contextual Spatial Filtering":
            lambda c: _has_amenity_data(c),

        # compute_routes → search_along_route.
        # Must have: route with polyline + SAR result with place details.
        "En-Route Logistics Optimization":
            lambda c: _has_sar(c),

        # text_search × 2 → place_details × 2 → compute_routes → get_transit_schedule.
        # Must have: transit schedule at origin + route between origin and dest.
        "Transit Planning":
            lambda c: _has_transit(c),

        # get_transit_schedule at origin + search_venue_events at dest,
        # with matching dates so the scenario can align arrival with event time.
        "Temporal Event Coordination":
            lambda c: _has_transit(c) and _has_events(c) and _transit_events_dates_overlap(c),

        # text_search → submit_council_report (+ optional place_details for accessibility).
        # Must have: at least one place with location + a user for the report.
        "Civic Issue Reporting":
            lambda c: _has_user(),

        # read_place_website → check_availability → book_place.
        # CRITICAL: webpage AND available_slots must be on THE SAME place.
        "Unstructured Web Evaluation & Reservation":
            lambda c: _has_webpage_and_slots_same_place(c) and _has_user(),

        # Full chain: routing + SAR + web eval + booking + events.
        # Requires all of the above combined.
        "Multi-Constraint Itinerary Synthesis":
            lambda c: (
                _has_sar(c) and
                _has_events(c) and
                _has_webpage_and_slots_same_place(c) and
                _has_user()
            ),
    }

    filter_fn = _FILTERS.get(task_name, lambda c: True)
    eligible = [c for c in clusters if filter_fn(c)]

    # Fallback 1: relax filters — any cluster
    if not eligible:
        eligible = clusters

    # Fallback 2: no routes at all — sample random places
    if not eligible:
        sampled_ids = set(rng.sample(list(db.get("places", {}).keys()),
                                     min(5, len(db.get("places", {})))))
        return _assemble_context(db, sampled_ids, None, None, None, None, task_name, rng)

    cluster = rng.choice(eligible)
    return _assemble_context(
        db,
        set(cluster["all_place_ids"]),
        cluster["route_key"],
        cluster.get("sar_key"),
        cluster["origin_id"],
        cluster["dest_id"],
        task_name,
        rng,
    )


# Which db sections each task actually needs.
# Only sections listed here are included in the assembled context —
# irrelevant sections are omitted to keep the LLM focused on the task.
_TASK_SECTIONS: dict[str, tuple] = {
    # text_search → place_details → read_place_website
    "Place Discovery & Web Fact Verification":
        ("places", "webpages", "users"),

    # text_search → place_details → nearby_search (lat/lng/type/rating/amenity)
    "Contextual Spatial Filtering":
        ("places", "users"),

    # text_search×2 → place_details×2 → compute_routes → search_along_route
    "En-Route Logistics Optimization":
        ("places", "webpages", "routes", "search_along_routes", "users"),

    # text_search×2 → place_details×2 → compute_routes → get_transit_schedule
    "Transit Planning":
        ("places", "routes", "transit_schedules", "users"),

    # text_search → get_transit_schedule + text_search → search_venue_events
    "Temporal Event Coordination":
        ("places", "routes", "transit_schedules", "events", "users"),

    # text_search → submit_council_report → (optional) place_details for accessibility
    "Civic Issue Reporting":
        ("places", "users"),

    # text_search → read_place_website → check_availability → book_place
    "Unstructured Web Evaluation & Reservation":
        ("places", "webpages", "users"),

    # Full multi-step: route + SAR + web + availability + booking + events
    "Multi-Constraint Itinerary Synthesis":
        ("places", "webpages", "routes", "search_along_routes", "events", "users"),
}


def _assemble_context(
    db: dict,
    place_ids: set,
    route_key: str | None,
    sar_key: str | None,
    origin_id: str | None,
    dest_id: str | None,
    task_name: str,
    rng: random.Random,
) -> dict:
    """
    Assemble a focused, coherent context dict for the given task.

    Only sections listed in _TASK_SECTIONS[task_name] are included.
    Within each section, items are ordered by relevance to the task
    (e.g. the bookable place's webpage is always prioritised first).
    """
    sections = _TASK_SECTIONS.get(task_name, tuple(_TASK_SECTIONS.keys()))

    # ---- PLACES --------------------------------------------------------
    # Contextual Spatial Filtering: add extra random places so the LLM has
    # enough variety (different types/ratings/amenities) for nearby_search.
    if task_name == "Contextual Spatial Filtering":
        all_ids = list(db.get("places", {}).keys())
        if len(all_ids) > len(place_ids):
            extra = set(rng.sample(all_ids, min(SPATIAL_EXTRA_PLACES, len(all_ids)))) - place_ids
            place_ids = place_ids | extra

    ctx_places = {
        pid: db["places"][pid]
        for pid in place_ids if pid in db.get("places", {})
    } if "places" in sections else {}

    # ---- WEBPAGES ------------------------------------------------------
    # Build a priority-ordered list of place_ids so the most task-relevant
    # webpages are always included first (before hitting MAX_WEBPAGES cap).
    ctx_webpages: dict = {}
    if "webpages" in sections:
        # For en_route: SAR places are the on-route stops the model will compare
        # and potentially call read_place_website on — they must get webpage slots
        # first, ahead of origin/dest.
        sar_place_ids_set: set[str] = set()
        if task_name in ("En-Route Logistics Optimization", "Multi-Constraint Itinerary Synthesis"):
            if sar_key and sar_key in db.get("search_along_routes", {}):
                sar_place_ids_set = set(db["search_along_routes"][sar_key])

        def _webpage_priority(pid: str) -> int:
            p = db.get("places", {}).get(pid, {})
            url = p.get("website_url")
            has_web   = bool(url and url in db.get("webpages", {}))
            has_slots = any(v > 0 for v in p.get("available_slots", {}).values())
            if task_name in ("Unstructured Web Evaluation & Reservation",
                             "Multi-Constraint Itinerary Synthesis"):
                # Highest priority: same place has both webpage AND bookable slots
                if has_web and has_slots: return 0
                if has_web:              return 1
                return 2
            if task_name == "En-Route Logistics Optimization":
                # SAR places (on-route stops) must get webpage slots first —
                # these are the places the model compares via read_place_website.
                if pid in sar_place_ids_set and has_web: return 0
                if has_web:                              return 1
                return 2
            # All other tasks: just prefer places that have a webpage
            return 0 if has_web else 1

        for pid in sorted(place_ids, key=_webpage_priority):
            if len(ctx_webpages) >= MAX_WEBPAGES:
                break
            url = db.get("places", {}).get(pid, {}).get("website_url")
            if url and url in db.get("webpages", {}):
                ctx_webpages[url] = db["webpages"][url]

    # ---- ROUTE ---------------------------------------------------------
    ctx_routes: dict = {}
    if "routes" in sections and route_key and route_key in db.get("routes", {}):
        ctx_routes[route_key] = db["routes"][route_key]

    # ---- SEARCH ALONG ROUTE --------------------------------------------
    ctx_sar: dict = {}
    if "search_along_routes" in sections and sar_key and sar_key in db.get("search_along_routes", {}):
        ctx_sar[sar_key] = db["search_along_routes"][sar_key]

    # ---- TRANSIT SCHEDULES (origin only) --------------------------------
    ctx_transit: dict = {}
    if "transit_schedules" in sections and origin_id and origin_id in db.get("transit_schedules", {}):
        ctx_transit[origin_id] = db["transit_schedules"][origin_id]

    # ---- EVENTS (destination only) -------------------------------------
    ctx_events: dict = {}
    if "events" in sections and dest_id and dest_id in db.get("events", {}):
        ctx_events[dest_id] = db["events"][dest_id]

    # ---- USERS (pick one at random) ------------------------------------
    ctx_users: dict = {}
    if "users" in sections:
        users = db.get("users", {})
        if users:
            uid = rng.choice(list(users.keys()))
            ctx_users[uid] = users[uid]

    return {
        "places":              ctx_places,
        "webpages":            ctx_webpages,
        "users":               ctx_users,
        "bookings":            {},
        "routes":              ctx_routes,
        "search_along_routes": ctx_sar,
        "transit_schedules":   ctx_transit,
        "events":              ctx_events,
        "council_reports":     {},
    }




# ==========================================
# PARAMETER EXTRACTION & VALIDATION
# ==========================================

def extract_missing_params_from_scenario(
    scenario: dict,
    task_name: str,
    variation: dict | None = None,
) -> dict:
    """
    Post-processing: parse the generated scenario and extract missing
    task-specific parameters from reason_for_call, variation, and actions
    into known_info to enrich the user simulator context.

    Args:
        scenario:   The generated scenario dict from LLM
        task_name:  The task name to determine required parameters
        variation:  Optional variation dict (persona, constraint, style)

    Returns:
        Enhanced scenario with richer known_info
    """
    if not scenario or "user_scenario" not in scenario:
        return scenario

    instructions = scenario.get("user_scenario", {}).get("instructions", {})
    known_info = instructions.get("known_info", "")
    reason_for_call = instructions.get("reason_for_call", "")

    # Extract parameters from variation if provided
    extracted_params = []
    if variation:
        constraint_text = variation.get("constraint", "")
        if constraint_text:
            extracted_params.append(f"Constraint: {constraint_text}")

    # Parse reason_for_call to extract mentioned parameters
    # Look for common keywords that indicate constraint/parameter values
    keywords_to_extract = {
        "party": "Party size",
        "people": "Group size",
        "accessibility": "Accessibility requirement",
        "wheelchair": "Wheelchair accessible",
        "dietary": "Dietary restriction",
        "vegetarian": "Vegetarian",
        "vegan": "Vegan",
        "allergy": "Allergy concerns",
        "nut": "Nut allergy",
        "budget": "Budget constraint",
        "price": "Price constraint",
        "rating": "Rating threshold",
        "wifi": "Wifi required",
        "parking": "Parking required",
        "deadline": "Arrival/departure deadline",
        "curfew": "Personal curfew",
        "distance": "Distance constraint",
        "walking": "Walking distance",
        "detour": "Detour tolerance",
        "available": "Availability checking",
        "comparison": "Place comparison",
        "private room": "Private seating",
        "outdoor seating": "Outdoor seating",
        "family-friendly": "Family-friendly",
        "child": "Child-friendly",
    }

    for keyword, param_name in keywords_to_extract.items():
        if keyword.lower() in reason_for_call.lower() and param_name not in known_info:
            # Try to extract the specific value from context
            extracted_params.append(param_name)

    # Parse actions to infer implied parameters
    actions = scenario.get("evaluation_criteria", {}).get("actions", [])
    for action in actions:
        args = action.get("arguments", {})
        # If there's a check_availability or book_place action, party size matters
        if action.get("name") in ["check_availability", "book_place"]:
            if "party_size" not in known_info.lower() and extracted_params:
                pass  # Already trying to extract from reason_for_call

    # Append extracted parameters to known_info if not already present
    if extracted_params:
        new_params = " | ".join([p for p in extracted_params if p not in known_info])
        if new_params:
            known_info = known_info.rstrip(".") + (" | " if known_info else "") + new_params

    # Update scenario
    scenario["user_scenario"]["instructions"]["known_info"] = known_info
    return scenario


def _ensure_reason_for_call_explicit_params(scenario: dict) -> dict:
    """
    Guarantee that user-provided argument values are explicitly stated in
    reason_for_call, per-tool whitelist only.

    Rules:
    - text_search.query          → must appear verbatim in the step that
                                   corresponds to that action (by action order).
    - All other whitelisted args → must appear anywhere in reason_for_call;
                                   if missing, injected into the first relevant
                                   step or before the closing step.
    - place_id, polyline, lat/lng and any non-whitelisted args → NOT checked.
    """
    if not scenario:
        return scenario

    try:
        instructions = scenario["user_scenario"]["instructions"]
        reason = instructions.get("reason_for_call", "")
        actions = scenario.get("evaluation_criteria", {}).get("actions", [])
    except (KeyError, TypeError):
        return scenario

    if not reason or not actions:
        return scenario

    # ------------------------------------------------------------------ #
    # Whitelist: tool_name -> list of arg keys the USER must state aloud  #
    # (arg keys not listed here are internal/technical and never needed   #
    # from the user, e.g. place_id, polyline, lat, lng)                  #
    # ------------------------------------------------------------------ #
    _REQUIRED_USER_ARGS: dict[str, list[str]] = {
        "text_search":           ["query"],
        "nearby_search":         ["place_type", "min_rating"],
        "compute_routes":        ["travel_mode"],
        "search_along_route":    ["place_type"],
        "check_availability":    ["datetime_str", "party_size"],
        "book_place":            ["user_id", "datetime_str", "party_size"],
        "get_transit_schedule":  ["date_str"],
        "search_venue_events":   ["date_str"],
        "submit_council_report": ["issue_type", "description", "user_id"],
        "transfer_to_human_agents": ["summary"],
    }

    lines = reason.splitlines()

    # Map "Step N" -> line index so we can inject into the right step.
    step_line: dict[int, int] = {}
    for idx, line in enumerate(lines):
        m = re.match(r"\s*Step\s+(\d+)\s*:", line, re.IGNORECASE)
        if m:
            step_line[int(m.group(1))] = idx

    # Find the last step number (used for appending a new step if needed).
    def _last_step_idx() -> int | None:
        """Return line index of the thank/end step, or None."""
        for idx, line in enumerate(lines):
            l = line.lower()
            if "thank the agent" in l or "end the chat" in l:
                return idx
        return None

    def _append_step(text: str) -> None:
        nums = sorted(step_line.keys())
        next_num = (nums[-1] + 1) if nums else 1
        new_line = f"Step {next_num}: {text}"
        end_idx = _last_step_idx()
        if end_idx is not None:
            lines.insert(end_idx, new_line)
            # Shift step_line indices that come after insertion.
            for k in list(step_line.keys()):
                if step_line[k] >= end_idx:
                    step_line[k] += 1
        else:
            lines.append(new_line)

    def _inject_into_line(line_idx: int, values: list[str]) -> None:
        suffix = " (use exact values: " + ", ".join(values) + ")"
        lines[line_idx] = lines[line_idx].rstrip() + suffix

    # ------------------------------------------------------------------ #
    # Process each action                                                  #
    # ------------------------------------------------------------------ #
    # For text_search, we try to match action order → step order.
    text_search_action_order = [
        i for i, a in enumerate(actions) if a.get("name") == "text_search"
    ]
    text_search_step_hits: list[int] = []  # line indices of "find/search" steps
    for idx, line in enumerate(lines):
        l = line.lower()
        if any(tok in l for tok in ("find", "search", "look up", "resolve", "identify")):
            text_search_step_hits.append(idx)

    # Global missing values for non-text_search tools (injected anywhere).
    global_missing: list[str] = []

    for action_idx, action in enumerate(actions):
        tool = action.get("name", "")
        args = action.get("arguments", {})
        required_keys = _REQUIRED_USER_ARGS.get(tool, [])
        if not required_keys:
            continue

        reason_lower = "\n".join(lines).lower()

        if tool == "text_search":
            query_val = str(args.get("query", "")).strip()
            if not query_val:
                continue
            if query_val.lower() in reason_lower:
                continue  # already present somewhere
            # Find the step line index that corresponds to this text_search.
            ts_rank = text_search_action_order.index(action_idx) if action_idx in text_search_action_order else 0
            if ts_rank < len(text_search_step_hits):
                target_line = text_search_step_hits[ts_rank]
                _inject_into_line(target_line, [query_val])
            else:
                # Fallback: inject at start (before end step).
                _append_step(f'Ask the agent to search for "{query_val}".')
        else:
            for key in required_keys:
                val = args.get(key)
                if val is None:
                    continue
                val_str = str(val).strip()
                if not val_str:
                    continue
                if val_str.lower() in reason_lower:
                    continue  # already stated
                # Format for readability.
                label = key.replace("_", " ")
                global_missing.append(f"{label}: {val_str}")

    # Inject all global_missing into the first relevant step.
    if global_missing:
        # Deduplicate.
        seen: set[str] = set()
        deduped = [v for v in global_missing if not (v in seen or seen.add(v))]  # type: ignore[func-returns-value]

        current_reason = "\n".join(lines).lower()
        still_missing = [v for v in deduped if v.split(": ", 1)[-1].lower() not in current_reason]

        if still_missing:
            keywords = (
                "availability", "book", "transit", "schedule", "event",
                "report", "submit", "slot", "compare", "filter", "nearby"
            )
            target_idx = None
            for idx, line in enumerate(lines):
                if any(kw in line.lower() for kw in keywords):
                    target_idx = idx
                    break

            if target_idx is not None:
                _inject_into_line(target_idx, still_missing)
            else:
                _append_step(
                    "Provide the following details to the agent: "
                    + ", ".join(still_missing) + "."
                )

    instructions["reason_for_call"] = "\n".join(lines)
    return scenario


def _enforce_nl_assertions_cap(scenario: dict) -> dict:
    """Clamp nl_assertions to MAX_NL_ASSERTIONS deterministically."""
    if not scenario:
        return scenario
    try:
        eval_criteria = scenario.get("evaluation_criteria", {})
        assertions = eval_criteria.get("nl_assertions", [])
        if isinstance(assertions, list) and len(assertions) > MAX_NL_ASSERTIONS:
            eval_criteria["nl_assertions"] = assertions[:MAX_NL_ASSERTIONS]
            scenario["evaluation_criteria"] = eval_criteria
    except (AttributeError, TypeError):
        return scenario
    return scenario


# ==========================================
# SCENARIO GENERATOR
# ==========================================

def generate_urban_scenario(
    task_row: dict,
    tools: list[dict],
    db_context: dict,
    variation: dict | None = None,
    prev_instruction: str | None = None,
    increase_complexity: bool = False,
    is_negative: bool = False,
) -> dict | None:
    """
    Calls the OpenAI API to generate a structured scenario based on a task row
    from task_overall.csv, the full tool definitions from tools.json, and a
    coherently sampled context slice from db.json.

    Args:
        task_row:         A dict with keys 'Category', 'Task', and 'Core Objective'
                          (one row from task_overall.csv).
        tools:            The full list of tool definitions loaded from tools.json.
        db_context:       A coherent, minimal db slice from sample_db_context().
        variation:        An optional dict with keys 'persona', 'constraint', and
                          'style' that shapes the user angle for this generation,
                          ensuring diverse scenarios across multiple runs on the
                          same db data.
        prev_instruction: The `reason_for_call` text from the immediately preceding
                          generation of this same task. Injected into the prompt so
                          the LLM actively avoids producing an identical scenario.
        increase_complexity: When True, injects a complexity-escalation section that
                          instructs the LLM to generate a harder, multi-step scenario
                          with more tool-call depth and richer constraints.

    Returns:
        A parsed JSON dict representing the generated scenario, or None on error.
    """

    # Build the tool reference section from the full tools.json definitions
    tools_description = json.dumps(tools, indent=2)

    # 1. Define the System Prompt
    system_prompt = f"""
You are an expert 'Scenario Architect' for UrbanConvBench, a benchmark evaluating tool-augmented AI agents.
Your objective is to generate a realistic, multi-turn conversational user task based strictly on the provided Task Row and Database Context.

CRITICAL RULES:
1. ZERO HALLUCINATION: Every entity name, place_id, datetime, price, polyline, or markdown content MUST be extracted directly from the provided Database Context. Do not invent any data.
2. STEP-BY-STEP INSTRUCTIONS: The `user_scenario.instructions.reason_for_call` MUST be formatted sequentially (e.g., Step 1: ..., Step 2: ...).
3. EXPLICIT TERMINATION: The final step in `reason_for_call` MUST explicitly instruct the user simulator to thank the agent and end the chat (e.g., "Step X: Thank the agent and end the chat.").
4. TOOL CHAINING & PREREQUISITE RESOLUTION — CRITICAL:
   The `evaluation_criteria.actions` array must represent the exact, logical sequence of tools the agent needs to call. Always use the exact tool `name` values from AVAILABLE TOOLS.
   MANDATORY PREREQUISITE CHAIN: ANY tool that requires a `place_id`, `lat`, or `lng` as an argument MUST be preceded by the tool calls that produce those values. The agent cannot know coordinates or place IDs in advance — they must come from prior tool results.
   Concretely:
   - `nearby_search(lat, lng, ...)` → the lat/lng MUST come from a preceding `place_details` call on a resolved place, which itself MUST be preceded by a `text_search` to resolve the landmark/location name to a place_id.
   - `compute_routes(origin_lat, origin_lng, dest_lat, dest_lng, ...)` → both origin and destination coords MUST each be resolved via `text_search` → `place_details` first.
   - `check_availability(place_id, ...)`, `book_place(place_id, ...)`, `read_place_website(place_id, ...)`, `get_transit_schedule(place_id, ...)`, `search_venue_events(place_id, ...)`, `submit_council_report(place_id, ...)` → the `place_id` MUST come from a preceding `text_search` call for that place.
   - `search_along_route(polyline, ...)` → the `polyline` MUST come from a preceding `compute_routes` call.
   NEVER start an action sequence with `nearby_search`, `compute_routes`, `check_availability`, `book_place`, or any place-id-dependent tool. ALWAYS begin by resolving place names through `text_search`.
5. PERSONA: Give the user a clear, concise persona reflecting their goal.
6. TASK ALIGNMENT: The generated scenario must directly exercise the Core Objective of the given task. The tool chain must reflect the capabilities described in the Core Objective.
7. SUB_TASKS FIELD: The `sub_tasks` field in the output JSON must exactly match the Task name from the provided task row.
8. CRITICAL — PARAMETER EXTRACTION:
   You MUST extract ALL task-relevant parameters from your generated scenario into 'known_info'.
   Do NOT leave parameters implicit or only mentioned in reason_for_call.
   For EVERY constraint or parameter mentioned in reason_for_call or actions, include it explicitly in known_info.
   Examples:
     - If reason_for_call mentions "party size 4", known_info MUST say "Party size: 4"
     - If reason_for_call mentions "wheelchair accessibility", known_info MUST say "Accessibility: wheelchair"
     - If reason_for_call mentions "vegetarian diet", known_info MUST say "Dietary: vegetarian"
     - If reason_for_call mentions "arrival by 18:00", known_info MUST say "Arrival deadline: 18:00"
     - If reason_for_call mentions "rating above 4.5", known_info MUST say "Minimum rating: 4.5"
   The user simulator MUST be able to answer ANY agent question about these parameters using ONLY known_info.
9. DATABASE GROUNDING FOR SPECIFIC TOOLS — these constraints are ABSOLUTE and override any other reasoning:
   a. `read_place_website`: ONLY include this action if the place's `website_url` field appears as a KEY in the `webpages` section of the Database Context. If a place has a `website_url` but it is NOT present in `webpages`, you MUST omit the `read_place_website` action for that place entirely — do NOT fabricate a call to it.
   b. `get_transit_schedule`: ONLY include this action with a `place_id` that appears as a KEY in the `transit_schedules` section of the Database Context. Never call it for destination places or any node not explicitly listed in `transit_schedules`. The transit schedule is available ONLY at the ORIGIN stop — do not call `get_transit_schedule` for the destination even for a "return trip" scenario.
   c. `search_along_route`: ONLY use `(polyline, place_type)` pairs whose cache key (formed as `polyline[:10] + "_" + place_type.lower()`) exists as a KEY in `search_along_routes`. The allowed `place_type` values are those already present in the `search_along_routes` keys — do NOT invent new place_type values not present there.
   d. `compute_routes`: ONLY generate route actions using (origin_lat, origin_lng, dest_lat, dest_lng, travel_mode) combinations whose rounded route key (`round(lat,3),round(lng,3)|round(lat,3),round(lng,3)|MODE`) exists as a KEY in the `routes` section. Verify the travel_mode matches exactly what is in the context — if only 'DRIVE' routes are present, do NOT use 'TRANSIT'.
10. CRITICAL — STEP/ACTION PARAMETER CONSISTENCY:
    The following action arguments MUST be stated explicitly in `reason_for_call` — these are values the user must say aloud; the agent cannot know them otherwise:
    - text_search         → query        (verbatim in the step where you ask for that search)
    - nearby_search       → place_type, min_rating
    - compute_routes      → travel_mode
    - search_along_route  → place_type
    - check_availability  → datetime_str, party_size
    - book_place          → user_id, datetime_str, party_size
    - get_transit_schedule → date_str
    - search_venue_events → date_str
    - submit_council_report → issue_type, description, user_id
    - transfer_to_human_agents → summary
    Do NOT enforce: place_id, lat, lng, polyline, radius, or any other internal/technical parameter.
    Example BAD step: "Ask the agent to check availability for a party."
    Example GOOD step: "Ask the agent to check availability for party size 2 on 2026-03-25 19:30."
11. NL ASSERTIONS — SEMANTIC & REASONING QUALITY: Generate AT MOST {MAX_NL_ASSERTIONS} items in `evaluation_criteria.nl_assertions`.
   These assertions are evaluated by an LLM-as-a-Judge against the final dialogue state. They must target SEMANTIC quality that CANNOT be verified by checking tool names or arguments alone.
   ALWAYS include at least one assertion of type (a). Only include (b) or (c) if the scenario actually involves comparing options or making a recommendation:
   a. SEMANTIC TRANSLATION (always applicable): Assert the agent correctly communicated a retrieved fact in natural language to the user.
      Good: "The agent correctly told the user that [Place X] offers wheelchair access and outdoor seating, based on the retrieved place details."
      Bad: "The agent called place_details for [Place X]." ← already covered by action verification
   b. COMPARATIVE REASONING (only if scenario retrieves 2+ options to compare): Assert the agent correctly identified and justified the BEST choice against the user's criteria.
      Good: "The agent correctly recommended [Stop A] over [Stop B] as the best en-route option because it has the highest rating (4.7) among results within the user's 5-minute detour budget."
      Bad: "The agent found a coffee shop along the route." ← vague, not grounded
      SKIP this category if the scenario only retrieves a single entity (e.g. one report, one schedule, one specific venue).
   c. RECOMMENDATION QUALITY (only if the scenario asks the agent to decide/recommend): Assert the agent's recommendation is correctly derived from the user's stated constraints.
      Good: "The agent correctly identified the 17:30 departure as the optimal option, reasoning that it arrives 20 minutes before the event start, satisfying the user's required arrival buffer."
      Bad: "The agent helped the user plan their trip." ← qualitative, unverifiable
      SKIP this category if the user did not ask the agent to choose or recommend among alternatives.
   DO NOT assert actions already verifiable by the tool-call check (e.g., 'the agent called text_search', 'the booking was confirmed with ID X').
   DO NOT write vague or qualitative assertions (e.g., 'the agent was helpful and clear').

AVAILABLE TOOLS (full definitions from tools.json):
{tools_description}

OUTPUT FORMAT:
You must output a raw, valid JSON object matching the exact schema below. Do NOT include markdown code blocks (```json) in your response, just the raw JSON.

EXAMPLE SCHEMA:
{{
  "id": "urban_map_web_booking_01",
  "sub_tasks": "Unstructured Web Evaluation & Reservation",
  "description": {{
    "purpose": "Evaluating unstructured web content to confirm a menu item, followed by checking capacity and securing a formal reservation.",
    "relevant_policies": null,
    "notes": "Unstructured Web Evaluation & Reservation - AMIRI CAFE"
  }},
  "user_scenario": {{
    "persona": "A user looking for a specific dessert and wanting to book a table. Conversational and direct.",
    "instructions": {{
      "task_instructions": "You are a Tim wanting to grab dessert at AMIRI CAFE on March 2nd, 2026. Your goal is to interact with the agent step-by-step.\\nCRITICAL RULES FOR YOU:\\n1. Act naturally. Keep your responses short (1-2 sentences max).\\n2. Do NOT echo or repeat internal tool data, markdown content, or system IDs.\\n3. Do NOT merge questions into one turn.",
      "domain": "urban_map_web",
      "reason_for_call": "Step 1: Ask the agent...\\nStep 2: Wait for confirmation, then ask...\\nStep 3: After confirmation, thank the agent and end the chat.",
      "known_info": "User name: Tim. User ID: user_001. Date and Time: 2026-03-02 19:30.",
      "unknown_info": "Whether the cafe has the cake, and the final booking ID."
    }}
  }},
  "initial_state": null,
  "evaluation_criteria": {{
    "actions": [
      {{"action_id": "step_1_action", "name": "text_search", "arguments": {{"query": "AMIRI CAFE"}}, "info": "Find the place ID for the cafe"}}
    ],
    "nl_assertions": ["The agent correctly communicated to the user that [Place X] has outdoor seating and is wheelchair-accessible, accurately reflecting the retrieved place details.", "The agent correctly recommended the [19:30 slot] over the [18:00 slot] because it better matched the user's stated preference, and confirmed the booking with the returned reservation ID."]
  }}
}}
"""

    # 2. Define the User Prompt
    variation_section = ""
    if variation:
        variation_section = f"""
SCENARIO ANGLE — you MUST shape the generated scenario around this specific angle:
- User persona:    {variation['persona']}
- Key constraint:  {variation['constraint']}
- Request style:   {variation['style']}

The persona, constraint, and request style above must be clearly reflected in:
  • The `user_scenario.persona` field
  • The `user_scenario.instructions.task_instructions` and `reason_for_call` text
  • The overall framing of the user's goal
"""

    anti_repeat_section = ""
    if prev_instruction:
        # prev_instruction is a list of the last N reason_for_call strings
        prev_list = prev_instruction if isinstance(prev_instruction, list) else [prev_instruction]
        prev_formatted = "\n\n".join(
            f"[Round -{len(prev_list)-i}]\n\"\"\"\n{p}\n\"\"\""
            for i, p in enumerate(prev_list)
        )
        anti_repeat_section = f"""
ANTI-REPETITION CONSTRAINT:
The {len(prev_list)} most recent generation(s) for this same task produced the following scenario summaries:
{prev_formatted}

You MUST generate a scenario that is clearly distinct from ALL of the above:
  • Different user goal or sub-objective (not just different wording)
  • Different place(s) or entity focus (if the database contains alternatives)
  • Different conversation flow / number of steps or step ordering
  • Different unknown_info the user is trying to resolve
"""

    complexity_section = ""
    if increase_complexity:
        task_hint = _COMPLEXITY_HINTS.get(task_row["Task"], "")
        complexity_section = f"""
COMPLEXITY ESCALATION — you MUST generate a HARDER, MORE COMPLEX scenario:

Target difficulty level: HARD
Core principles:
  1. LONGER TOOL CHAINS — the agent must call significantly more tools (aim for 5–8+ actions, not 2–3).
  2. MULTI-ENTITY GOALS — the user wants to compare, combine, or act on 2+ places/routes/events, not just one.
  3. STACKED CONSTRAINTS — the user has multiple simultaneous requirements; satisfying all of them at once is non-trivial.
  4. CONDITIONAL REPLANNING — at least one step should fail or return an unsatisfactory result (e.g. no availability, rating too low, slot full), forcing the agent to recover with an alternative.
  5. CROSS-SOURCE VERIFICATION — where possible, the user wants facts confirmed from BOTH the database AND the official webpage, and the agent must reconcile any discrepancies.

Task-specific escalation guidance for "{task_row['Task']}":
{task_hint}

FEW-SHOT CALIBRATION EXAMPLE (do NOT copy — use only as a difficulty reference):

  SIMPLE scenario (avoid this level):
    reason_for_call: |
      Step 1: Ask the agent to find AMIRI CAFE.
      Step 2: Ask if it has outdoor seating.
      Step 3: Thank the agent and end the chat.
    actions: [text_search, place_details]   ← only 2 tools

  COMPLEX scenario (aim for this level or harder):
    reason_for_call: |
      Step 1: Ask the agent to find both AMIRI CAFE and BLUE BEAN COFFEE and compare their ratings and accessibility.
      Step 2: Ask the agent to confirm the opening hours AND outdoor seating of the higher-rated venue from its official website.
      Step 3: Ask whether the website mentions a parking facility — reconcile with the db accessibility data if they differ.
      Step 4: Ask the agent to check table availability for 19:30 at the chosen venue.
      Step 5: If no slots are available at 19:30, ask the agent to find the next available slot and confirm it.
      Step 6: Confirm the booking at the available time and thank the agent.
    actions: [text_search, text_search, place_details, place_details, read_place_website, check_availability, book_place]   ← 7 tools

Ensure the final scenario's `reason_for_call` has AT LEAST as many steps as the task-specific guidance above specifies,
and the `actions` array has at least as many entries.
"""

    negative_section = ""
    if is_negative:
        negative_section = """
NEGATIVE OUTCOME SCENARIO — you MUST generate a scenario where the service FAILS or is UNAVAILABLE:
Target difficulty: The user asks for a valid service/query, but the underlying database implies failure.
Examples:
  • Booking is full / no available slots matching constraints → agent must clearly inform the user and the user ends the conversation or asks for simple alternatives.
  • Place is permanently or temporarily closed → agent communicates this and user ends the conversation disappointed.
  • nearby_search returns zero results matching all the user's strict criteria → agent informs, user relaxes one constraint or gives up.
  • The user's desired dish/amenity is NOT found on the website → agent confirms absence and conversation ends.

CRITICAL: In these negative cases, the `nl_assertions` array MUST assert that the AGENT CORRECTLY COMMUNICATES THE FAILURE (e.g. "The agent correctly informed the user that no tables were available matching the request"), NOT that the task succeeded. Do NOT fabricate unavailability if the db context shows availability—design a user constraint that is guaranteed to fail based on the provided db context (e.g. requesting a table at a time when 'available_slots' is 0, or requesting an amenity that is False).
"""

    user_prompt = f"""
TASK TO GENERATE:
- Category:       {task_row['Category']}
- Task Name:      {task_row['Task']}
- Core Objective: {task_row['Core Objective']}
{variation_section}{anti_repeat_section}{complexity_section}{negative_section}
DATABASE CONTEXT:
{json.dumps(db_context, indent=2)}

Based on the task above and the database context, generate ONE complete scenario in JSON format.
Requirements:
1. The scenario must directly exercise the Core Objective described above.
2. Every place_id, name, price, schedule entry, polyline, and markdown snippet used in the scenario must exist verbatim in the Database Context.
3. The `actions` sequence must logically and completely solve the user's objective using only the tools listed in AVAILABLE TOOLS.
4. Set the `sub_tasks` field to exactly: "{task_row['Task']}"
5. For nl_assertions: generate AT MOST {MAX_NL_ASSERTIONS} items targeting SEMANTIC quality only. Always include at least one assertion about whether the agent correctly communicated retrieved facts in natural language (type a). Only add comparative reasoning assertions (type b) if this scenario actually retrieved multiple options to compare, and only add recommendation-quality assertions (type c) if the user explicitly asked the agent to choose or recommend. Do NOT assert things already covered by the actions tool-call check.
6. In `reason_for_call`, explicitly include all user-provided concrete parameters needed by actions (date/time/slot/party_size/group size). Avoid vague wording like "check another slot"; name the exact slot values.
"""

    print(f"  Generating: [{task_row['Category']}] {task_row['Task']}...")

    # 3. Call the OpenAI API via OpenRouter
    try:
        # Use with_raw_response to access OpenRouter-specific headers like cost
        raw_completion = client.chat.completions.with_raw_response.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        # Parse the completion object
        response = raw_completion.parse()
        
        # 4. Extract and print cost from OpenRouter headers
        total_cost = raw_completion.http_response.headers.get("x-openrouter-cost", "0.0")
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        
        print(f"    [COST] {MODEL}: ${total_cost} (Tokens: {prompt_tokens} in, {completion_tokens} out)")

        # 5. Parse and return the result
        raw_output = response.choices[0].message.content
        scenario_json = json.loads(raw_output)
        
        # 6. Post-process: extract missing parameters and enhance known_info
        scenario_json = extract_missing_params_from_scenario(
            scenario_json, task_row["Task"], variation
        )

        # 7. Post-process: enforce explicit parameter mention in user steps
        scenario_json = _ensure_reason_for_call_explicit_params(scenario_json)

        # 8. Post-process: hard-cap nl_assertions length regardless of model drift
        scenario_json = _enforce_nl_assertions_cap(scenario_json)
        
        return scenario_json

    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":

    # File paths
    csv_path   = os.path.join(SCRIPT_DIR, "task_overall.csv")
    tools_path = os.path.join(SCRIPT_DIR, "tools.json")

    # Use crawler-generated db
    db_path = _CRAWLER_DB

    # Load shared resources
    task_categories = load_task_categories(csv_path)
    tools           = load_tools(tools_path)
    db              = load_db(db_path)

    print(f"DB: {db_path}")
    print(f"  places: {len(db.get('places', {}))} | "
          f"webpages: {len(db.get('webpages', {}))} | "
          f"routes: {len(db.get('routes', {}))} | "
          f"transit: {len(db.get('transit_schedules', {}))} | "
          f"events: {len(db.get('events', {}))}")
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
    prev_instructions: dict[str, list[str]] = defaultdict(list)

    for gen_idx in range(NUM_GEN):
        print(f"{'='*60}")
        print(f"GENERATION ROUND {gen_idx + 1}/{NUM_GEN}")
        print(f"{'='*60}")

        rng = random.Random()  # fresh RNG per round for sampling diversity

        for task_row in task_categories:
            # Sample a coherent, minimal context slice for this task + round
            db_context = sample_db_context(db, task_row["Task"], rng)

            # Pick a variation angle to ensure scenario diversity across rounds.
            # The axes are cycled (not purely random) so consecutive rounds
            # are guaranteed to use a different persona/constraint/style.
            axes = _VARIATION_AXES.get(task_row["Task"], [])
            if axes:
                # Use deterministic shuffling across cycles to guarantee all items are used
                # before repeating, but in a varied order across different NUM_GEN cycles.
                cycle_num = gen_idx // len(axes)
                seed_rng = random.Random(f"{task_row['Task']}_{cycle_num}")
                shuffled_axes = seed_rng.sample(axes, len(axes))
                variation = shuffled_axes[gen_idx % len(axes)]
            else:
                variation = None

            # Pass the previous round's instruction so the LLM avoids repeating it
            # (only when ANTI_REPETITION is enabled).
            # Pass the last ANTI_REPETITION_TOP_N summaries (most recent first)
            prev_instr = prev_instructions[task_row["Task"]][-ANTI_REPETITION_TOP_N:] if ANTI_REPETITION else None

            scenario = generate_urban_scenario(
                task_row, tools, db_context, variation, prev_instr,
                increase_complexity=INCREASE_COMPLEXITY,
                is_negative=(rng.random() < NEGATIVE_OUTCOME_RATIO),
            )

            if scenario:
                # Assign a deterministic, human-readable ID
                slug = _TASK_SLUGS.get(
                    task_row["Task"],
                    task_row["Task"].lower().replace(" ", "_")
                )
                scenario["id"]         = f"urban_map_web_{slug}_{gen_idx + 1:02d}"
                scenario["_gen_round"] = gen_idx + 1

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

                # Accumulate this round's reason_for_call for future anti-repetition
                try:
                    reason = (
                        scenario["user_scenario"]["instructions"]["reason_for_call"]
                    )
                    prev_instructions[task_row["Task"]].append(reason)
                except (KeyError, TypeError):
                    pass

                all_generated.append(scenario)

                # Save individual file
                out_file = output_dir / f"{slug}_gen{gen_idx + 1:02d}.json"
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