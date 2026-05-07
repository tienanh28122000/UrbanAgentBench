"""Crawl POI data from OpenStreetMap Nominatim and export to CSV.

CSV format matches:
	Theme,Sub Theme,Feature Name,Co-ordinates

Examples:
	python craw_pois.py
	python craw_pois.py --city paris --limit 80
	python craw_pois.py --city newyork --output ./newyork_pois.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from math import ceil
from typing import Any

import requests


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
REQUEST_HEADERS = {
	"User-Agent": "urbanbench-poi-crawler/1.0 (contact: local-script)",
	"Accept": "application/json",
	"Accept-Language": "en",
}

CITY_ALIASES = {
	"tokyo": "Tokyo",
	"paris": "Paris",
	"newyork": "New York",
	"london": "London",
	"melbourne": "Melbourne",
}

# Keep these domains aligned with the spirit of the sample dataset.
DOMAIN_QUERIES: list[tuple[str, str, str]] = [
	("Education Centre", "School", "school"),
	("Place Of Assembly", "Theatre Live", "theatre"),
	("Health Services", "Hospital", "hospital"),
	("Leisure/Recreation", "Major Sports & Recreation Facility", "stadium"),
	("Leisure/Recreation", "Informal Outdoor Facility (Park/Garden/Reserve)", "park"),
	("Place Of Assembly", "Art Gallery/Museum", "museum"),
	("Place of Worship", "Church", "church"),
	("Transport", "Railway Station", "railway station"),
	("Community Use", "Public Buildings", "city hall"),
	("Leisure/Recreation", "Outdoor Recreation Facility (Zoo, Golf Course)", "zoo"),
]


@dataclass
class PoiRow:
	theme: str
	sub_theme: str
	feature_name: str
	lat: float
	lon: float

	def to_csv_row(self) -> dict[str, str]:
		return {
			"Theme": self.theme,
			"Sub Theme": self.sub_theme,
			"Feature Name": self.feature_name,
			"Co-ordinates": f"{self.lat:.12f}, {self.lon:.12f}",
		}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Crawl POIs from OSM Nominatim and save as Melbourne-style CSV format."
	)
	parser.add_argument(
		"--city",
		type=str,
		default="tokyo",
		choices=sorted(CITY_ALIASES.keys()),
		help="Target city. Default: tokyo",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=300,
		help="Number of POIs to export. Default: 300",
	)
	parser.add_argument(
		"--output",
		type=str,
		default="",
		help="Output CSV path. Default: <script_dir>/<city>_pois.csv",
	)
	return parser.parse_args()

def fetch_nominatim(query: str, limit: int) -> list[dict[str, Any]]:
	params = {
		"q": query,
		"format": "jsonv2",
		"addressdetails": 1,
		"namedetails": 1,
		"limit": str(limit),
	}
	response = requests.get(
		NOMINATIM_URL,
		params=params,
		headers=REQUEST_HEADERS,
		timeout=60,
	)
	response.raise_for_status()
	data = response.json()
	if isinstance(data, list):
		return data
	return []


def select_feature_name(item: dict[str, Any]) -> str:
	name = (item.get("name") or "").strip()
	if name:
		return name

	namedetails = item.get("namedetails") or {}
	if isinstance(namedetails, dict):
		for key in ("name:en", "name"):
			value = (namedetails.get(key) or "").strip()
			if value:
				return value

	display_name = (item.get("display_name") or "").strip()
	if not display_name:
		return ""

	return display_name.split(",")[0].strip()


def fetch_osm_pois(city_name: str, limit: int) -> list[PoiRow]:
	rows: list[PoiRow] = []
	seen: set[tuple[str, float, float]] = set()

	# Pull slightly more per domain to handle dedupe and missing names.
	per_domain_limit = max(8, ceil(limit / len(DOMAIN_QUERIES)) + 4)

	for theme, sub_theme, keyword in DOMAIN_QUERIES:
		query = f"{keyword} {city_name}"
		try:
			items = fetch_nominatim(query=query, limit=per_domain_limit)
		except requests.RequestException:
			continue

		for item in items:
			feature_name = select_feature_name(item)
			if not feature_name:
				continue

			lat_str = item.get("lat")
			lon_str = item.get("lon")
			if lat_str is None or lon_str is None:
				continue

			try:
				lat = float(lat_str)
				lon = float(lon_str)
			except (TypeError, ValueError):
				continue

			key = (feature_name.lower(), round(lat, 6), round(lon, 6))
			if key in seen:
				continue
			seen.add(key)

			rows.append(
				PoiRow(
					theme=theme,
					sub_theme=sub_theme,
					feature_name=feature_name,
					lat=lat,
					lon=lon,
				)
			)

			if len(rows) >= limit:
				return rows

	return rows


def save_csv(rows: list[PoiRow], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=["Theme", "Sub Theme", "Feature Name", "Co-ordinates"],
		)
		writer.writeheader()
		for row in rows:
			writer.writerow(row.to_csv_row())


def main() -> None:
	args = parse_args()
	city_key = args.city.lower().strip()
	city_name = CITY_ALIASES[city_key]

	if args.limit <= 0:
		raise ValueError("--limit must be > 0")

	script_dir = Path(__file__).resolve().parent
	output_path = Path(args.output) if args.output else script_dir / f"{city_key}_pois.csv"

	rows = fetch_osm_pois(city_name=city_name, limit=args.limit)
	save_csv(rows=rows, output_path=output_path)

	print(f"[OK] City: {city_name}")
	print(f"[OK] Exported {len(rows)} POIs to: {output_path}")


if __name__ == "__main__":
	main()
