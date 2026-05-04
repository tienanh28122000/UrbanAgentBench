from typing import Dict, Optional
from pydantic import BaseModel, Field
from urban_agent_bench.domains.urban_satellite.utils import URBAN_SATELLITE_DB_PATH
from urban_agent_bench.environment.db import DB


class SiteMeta(BaseModel):
    """Operational metadata for a site (image paths, temporal availability)."""
    current_image: str = Field(description="Relative path to the current satellite image")
    has_temporal: bool = Field(default=False, description="Whether historical imagery is available")
    past_image: Optional[str] = Field(None, description="Relative path to the past satellite image")


class Site(BaseModel):
    site_id: str = Field(description="Unique identifier for the site (e.g., '19652_30149')")
    lat: float = Field(description="Latitude of the site")
    lon: float = Field(description="Longitude of the site")
    description: Optional[str] = Field(None, description="General description of the location")
    meta: Optional[SiteMeta] = Field(None, description="Image metadata and temporal availability")

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