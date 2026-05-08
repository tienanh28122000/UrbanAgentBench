from typing import Dict, Any
from tau2.domains.urban_satellite.data_model import UrbanSatelliteDB, SiteAssessment
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau2.domains.urban_satellite.utils import (
    deg2num,
    URBAN_SATELLITE_IMAGE_PATH,
    calculate_density,
    classify_land_use_vlm,
    detect_infrastructure_vlm,
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
    def classify_land_use(self, image_path: str) -> list[str]:
        """Identifies which OpenStreetMap land-use categories are present at the site based on the satellite image.

        Args:
            image_path: The file path of the satellite image to analyze.

        Returns:
            list[str]: A list of land-use category tags present at this site.
                Possible values: 'landuse' (built-up or managed land),
                'natural' (natural features such as vegetation or water).
                The list may contain one or both categories.
        """
        return classify_land_use_vlm(image_path)

    @is_tool(ToolType.READ)
    def analyze_urban_density(self, image_path: str) -> float:
        """Estimates the population density of the area visible in the satellite image.

        Args:
            image_path: The file path of the satellite image.

        Returns:
            float: A population density score from 0.0 (unpopulated) to 9.9 (extremely dense),
                calibrated against population densities of cities worldwide.
        """
        score = calculate_density(image_path)
        return float(min(max(score, 0.0), 9.9))

    @is_tool(ToolType.READ)
    def detect_infrastructure(self, image_path: str, feature_query: str) -> bool:
        """Determines whether a specific infrastructure element is present in the satellite image.

        Args:
            image_path: The file path of the satellite image.
            feature_query: The specific infrastructure element to check for
                (e.g., 'Bridge', 'Stadium', 'Train Station', 'Golf Field',
                'Soccer Ball Field', 'Swimming Pool', 'Tennis Court',
                'Roundabout', 'Basketball Court', 'Ground Track Field',
                'Baseball Field', 'Overpass', 'Storage Tank', 'Windmill').

        Returns:
            bool: True if the feature is visible in the image, False otherwise.
        """
        return detect_infrastructure_vlm(image_path, feature_query)

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