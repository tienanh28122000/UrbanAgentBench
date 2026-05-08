from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from tau2.domains.urban_satellite.utils import URBAN_SATELLITE_DB_PATH
from tau2.environment.db import DB


class SiteMeta(BaseModel):
    """Operational metadata for a site (image path)."""
    current_image: Optional[str] = Field(None, description="Relative path to the current satellite image")


class GroundTruth(BaseModel):
    """Ground-truth annotations for a site derived from OpenStreetMap and population data."""
    population_density: float = Field(description="Population density score from 0.0 to 9.9")
    land_use_categories: List[str] = Field(description="OSM land-use tags present at this site (e.g. landuse, natural, leisure)")
    infrastructure_counts: Dict[str, int] = Field(default_factory=dict, description="Presence counts for each infrastructure type (0 = absent, 1 = present)")


class Site(BaseModel):
    site_id: str = Field(description="Unique identifier for the site (e.g., '10904_16379')")
    lat: float = Field(description="Latitude of the site")
    lon: float = Field(description="Longitude of the site")
    description: Optional[str] = Field(None, description="General description of the location")
    meta: Optional[SiteMeta] = Field(None, description="Image metadata")
    ground_truth: Optional[GroundTruth] = Field(None, description="Ground-truth annotations used for evaluation")


class SiteAssessment(BaseModel):
    site_id: str
    decision: str
    justification: str


class UrbanSatelliteDB(DB):
    """Database containing infrastructure sites and final assessments."""
    sites: Dict[str, Site] = Field(default_factory=dict, description="Dictionary of known sites")
    assessments: Dict[str, SiteAssessment] = Field(default_factory=dict, description="Final submitted decisions by the agent")


def get_db():
    return UrbanSatelliteDB.load(URBAN_SATELLITE_DB_PATH)