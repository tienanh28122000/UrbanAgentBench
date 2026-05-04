from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from urban_agent_bench.domains.urban_map_web.utils import URBAN_WEB_DB_PATH
from urban_agent_bench.environment.db import DB

class Location(BaseModel):
    lat: float
    lng: float

class Webpage(BaseModel):
    url: str
    title: str
    content_markdown: str

class Review(BaseModel):
    rating: float
    text: str
    time: str

# MỚI: Thêm model cho giờ mở cửa
class OpeningHours(BaseModel):
    open_now: bool = False
    weekday_text: List[str] = Field(default_factory=list)

class Place(BaseModel):
    place_id: str
    name: str
    location: Location
    address: str
    types: List[str]
    rating: float
    user_rating_count: int = 0  # MỚI: Số lượng review
    price_level: Optional[int] = None
    website_url: Optional[str] = None
    phone_number: Optional[str] = None  # MỚI: Số điện thoại
    
    # MỚI: Các trường thông tin mở rộng để tạo Edge Cases
    opening_hours: Optional[OpeningHours] = None
    business_status: str = "OPERATIONAL"
    accessibility: Dict[str, bool] = Field(default_factory=dict) # VD: wheelchairAccessible
    amenities: Dict[str, bool] = Field(default_factory=dict) # VD: servesVegetarianFood, goodForChildren

    reviews: List[Review] = Field(default_factory=list)
    available_slots: Dict[str, int] = Field(default_factory=dict)

class RouteCache(BaseModel):
    route_key: str = Field(description="Key format: originLat,originLng|destLat,destLng|mode")
    distance_meters: int
    duration_seconds: int
    polyline: str = Field(description="Encoded polyline from Google Maps")
    optimized_order: Optional[List[int]] = None

class User(BaseModel):
    user_id: str
    name: str
    phone: str

class Booking(BaseModel):
    booking_id: str
    place_id: str
    user_id: str
    datetime_str: str
    party_size: int
    status: Literal["confirmed", "cancelled"] = "confirmed"

class TransitSchedule(BaseModel):
    route_number: str
    destination: str
    departure_time: str

class VenueEvent(BaseModel):
    event_name: str
    date: str
    time: str
    ticket_price: float
    description: str

class CouncilReport(BaseModel):
    ticket_id: str
    issue_type: str
    location: Location
    reference_place: str = Field(description="The name of the nearest location so that the council can find it easily.")
    description: str
    user_id: str
    status: str

class UrbanWebDB(DB):
    places: Dict[str, Place] = Field(default_factory=dict)
    webpages: Dict[str, Webpage] = Field(default_factory=dict)
    users: Dict[str, User] = Field(default_factory=dict)
    bookings: Dict[str, Booking] = Field(default_factory=dict)
    routes: Dict[str, RouteCache] = Field(default_factory=dict)
    search_along_routes: Dict[str, List[str]] = Field(default_factory=dict)
    
    # MỚI: Kho chứa dữ liệu cho 3 Web Tools
    transit_schedules: Dict[str, Dict[str, List[TransitSchedule]]] = Field(default_factory=dict, description="Key is place_id")
    events: Dict[str, List[VenueEvent]] = Field(default_factory=dict, description="Key is place_id")
    council_reports: Dict[str, CouncilReport] = Field(default_factory=dict, description="Key is ticket_id")

    def get_statistics(self) -> dict[str, Any]:
        return {
            "num_places": len(self.places),
            "num_webpages": len(self.webpages),
            "num_routes": len(self.routes),
            "num_transit_stops": len(self.transit_schedules),
            "num_venues_with_events": len(self.events),
            "num_council_reports": len(self.council_reports)
        }

def get_db():
    return UrbanWebDB.load(URBAN_WEB_DB_PATH)