import math
from typing import Dict, Any
from urban_agent_bench.domains.urban_satellite.data_model import UrbanSatelliteDB, SiteAssessment
from urban_agent_bench.environment.toolkit import ToolKitBase, ToolType, is_tool
from urban_agent_bench.domains.urban_satellite.utils import (
    deg2num, 
    URBAN_SATELLITE_IMAGE_PATH,
    URBAN_SATELLITE_PAST_IMAGE_PATH,
    calculate_density,
    calculate_green_ratio,
    classify_land_use_vlm,
    detect_infrastructure_vlm,
    verify_path_vlm,
    compare_temporal_vlm,
    estimate_carbon_vlm
)
import os

class UrbanSatelliteTools(ToolKitBase):
    """Toolkit for the Urban Satellite domain using VLM-augmented atomic tools."""

    db: UrbanSatelliteDB

    def __init__(self, db: UrbanSatelliteDB) -> None:
        super().__init__(db)

    @is_tool(ToolType.READ)
    def get_satellite_tile(self, lat: float, lon: float) -> str:
        """Fetches the raw satellite imagery file path from the database for a specific coordinate and time.

        Args:
            lat: Latitude of the target location.
            lon: Longitude of the target location.

        Returns:
            str: The local file path to the saved satellite image.

        Raises:
            ValueError: If the tile for the specific coordinates and date is not found in the local environment.
        """
        zoom = 15  # Fixed zoom level for all tiles
        x, y = deg2num(lat, lon, zoom)
        site_id = f"{y}_{x}"
        if site_id not in self.db.sites:
            raise ValueError(
                f"No site found for coordinates ({lat}, {lon}). "
                f"Tile {site_id} is not in the database."
            )
        tile_name = f"{y}_{x}.png"
        image_path = f"{URBAN_SATELLITE_IMAGE_PATH}/{tile_name}"
        
        return image_path

    @is_tool(ToolType.READ)
    def get_past_satellite_tile(self, lat: float, lon: float) -> str:
        """Retrieves the historical satellite image path for a site, enabling temporal change analysis.

        Args:
            lat: Latitude of the target location.
            lon: Longitude of the target location.

        Returns:
            str: The local file path to the past satellite image.

        Raises:
            ValueError: If no site exists at the given coordinates or no historical imagery is available.
        """
        zoom = 15
        x, y = deg2num(lat, lon, zoom)
        site_id = f"{y}_{x}"
        if site_id not in self.db.sites:
            raise ValueError(
                f"No site found for coordinates ({lat}, {lon}). "
                f"Tile {site_id} is not in the database."
            )
        site = self.db.sites[site_id]
        if not site.meta or not site.meta.has_temporal:
            raise ValueError(
                f"No historical imagery available for site {site_id}. "
                f"Temporal analysis cannot be performed for this location."
            )
        tile_name = f"{y}_{x}_past.png"
        past_path = f"{URBAN_SATELLITE_PAST_IMAGE_PATH}/{tile_name}"
        return past_path

    @is_tool(ToolType.READ)
    def classify_land_use(self, image_path: str) -> str:
        """Prompts the backend VLM to strictly categorize the overall visual pattern of the site into a single land-use type.

        Args:
            image_path: The file path of the satellite image to analyze.

        Returns:
            str: The land-use category (e.g., 'Industrial', 'Residential', 'Agricultural').
        """
        return classify_land_use_vlm(image_path)

    @is_tool(ToolType.READ)
    def analyze_urban_density(self, image_path: str) -> dict:
        """Prompts the backend VLM to estimate the density of built structures.

        Args:
            image_path: The file path of the satellite image.

        Returns:
            dict: A dictionary containing:
                - 'category' (str): The density category ('Sparse', 'Moderate', or 'Dense').
                - 'score' (float): A numerical density score ranging from 0.0 (empty) to 10.0 (highly dense).
        """
        density_result = calculate_density(image_path)
        if not isinstance(density_result, dict):
             density_result = {"category": "Unknown", "score": float(density_result)}
        # Ensure score is within valid range
        density_result['score'] = float(min(max(density_result.get('score', 0.0), 0.0), 10.0))
        return density_result

    @is_tool(ToolType.READ)
    def check_environmental_ratio(self, image_path: str) -> dict:
        """Prompts the backend VLM to calculate the proportion of green spaces or water bodies.

        Args:
            image_path: The file path of the satellite image.

        Returns:
            dict: A dictionary containing:
                - 'category' (str): The vegetation category ('Low', 'Medium', or 'High').
                - 'score' (float): A ratio representing environmental coverage, from 0.0 to 1.0.
        """
        green_ratio_result = calculate_green_ratio(image_path)
        if not isinstance(green_ratio_result, dict):
             green_ratio_result = {"category": "Unknown", "score": float(green_ratio_result)}
        # Ensure score is within valid range
        green_ratio_result['score'] = float(min(max(green_ratio_result.get('score', 0.0), 0.0), 1.0))
        return green_ratio_result

    @is_tool(ToolType.READ)
    def estimate_carbon_emission(self, image_path: str) -> float:
        """Prompts the backend VLM to estimate the carbon emission index based on visible urban and environmental features.

        Args:
            image_path: The file path of the satellite image.

        Returns:
            float: An estimated annual carbon emission index (higher values indicate more emission-intensive areas).
        """
        return estimate_carbon_vlm(image_path)

    @is_tool(ToolType.READ)
    def detect_infrastructure(self, image_path: str, feature_query: str) -> int:
        """Prompts the backend VLM to count specific infrastructure elements.

        Args:
            image_path: The file path of the satellite image.
            feature_query: The specific object to detect and count (e.g., 'cooling towers', 'bridges').

        Returns:
            int: The total count of the requested features detected in the image.
        """
        return detect_infrastructure_vlm(image_path, feature_query)

    @is_tool(ToolType.READ)
    def verify_path_connectivity(self, image_path: str, start_coord: str, end_coord: str) -> bool:
        """Prompts the backend VLM to trace and verify if a continuous physical path exists between two specific points.

        Args:
            image_path: The file path of the satellite image.
            start_coord: A description or approximate location of the start point (e.g., 'top left corner').
            end_coord: A description or approximate location of the end point (e.g., 'river bank').

        Returns:
            bool: True if a continuous path exists, False otherwise.
        """
        return verify_path_vlm(image_path, start_coord, end_coord)

    @is_tool(ToolType.READ)
    def compare_temporal_change(self, image_path_1: str, image_path_2: str) -> str:
        """Feeds two multi-temporal images to the backend VLM to describe visual evidence of land modification over time.

        Args:
            image_path_1: The file path of the older satellite image.
            image_path_2: The file path of the newer satellite image.

        Returns:
            str: A detailed textual description of any structural or environmental changes between the two images.
        """
        return compare_temporal_vlm(image_path_1, image_path_2)

    @is_tool(ToolType.READ)
    def measure_spatial_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculates the exact ground distance between two coordinates using mathematical formulas.

        Args:
            lat1: Latitude of the first point.
            lon1: Longitude of the first point.
            lat2: Latitude of the second point.
            lon2: Longitude of the second point.

        Returns:
            float: The physical ground distance in meters.
        """
        R = 6371000  # Earth radius in meters
        l1, lo1 = map(math.radians, [lat1, lon1])
        l2, lo2 = map(math.radians, [lat2, lon2])
        
        dlat = l2 - l1
        dlon = lo2 - lo1
        
        a = math.sin(dlat / 2)**2 + math.cos(l1) * math.cos(l2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return float(R * c)

    @is_tool(ToolType.WRITE)
    def submit_site_assessment(self, site_id: str, decision: str, justification: str) -> Dict[str, Any]:
        """Submits the agent's final logical conclusion to the planning system, completing the multi-step reasoning task.

        Args:
            site_id: The unique identifier of the evaluated site.
            decision: The final categorical decision (e.g., 'Suitable', 'Not Suitable', 'Deforestation Detected').
            justification: A brief explanation of the logic leading to this decision based on tool outputs.

        Returns:
            Dict[str, Any]: A receipt confirming the assessment was saved successfully.
        """
        assessment = SiteAssessment(
            site_id=site_id,
            decision=decision,
            justification=justification
        )
        self.db.assessments[site_id] = assessment
        return {
            "status": "Success",
            "message": f"Assessment for site {site_id} successfully recorded."
        }

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """
        Transfer the user to a human agent, with a summary of the user's issue.
        Only transfer if
         -  the user explicitly asks for a human agent
         -  given the policy and the available tools, you cannot solve the user's issue.

        Args:
            summary: A summary of the user's issue.

        Returns:
            A message indicating the user has been transferred to a human agent.
        """
        return "Transfer successful"