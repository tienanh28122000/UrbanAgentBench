import math
from typing import List, Dict, Any, Optional
from urban_agent_bench.domains.urban_map_web.data_model import UrbanWebDB, Place, Booking, User
from urban_agent_bench.domains.urban_map_web.utils import generate_booking_id
from urban_agent_bench.environment.toolkit import ToolKitBase, ToolType, is_tool

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two lat/lng coordinates in meters."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class UrbanWebTools(ToolKitBase):
    """Toolkit for the Urban Map & Web domain. Fully operates on a static local DB."""

    db: UrbanWebDB

    def __init__(self, db: UrbanWebDB) -> None:
        super().__init__(db)

    # =========================================================
    # SIMULATED MAP TOOLS (Static DB Lookup)
    # =========================================================
    
    @is_tool(ToolType.READ)
    def text_search(self, query: str) -> List[Dict[str, Any]]:
        """Resolves free-text names into structured IDs and coordinates.

        Args:
            query: The search query, such as 'Monash University' or 'Cafe near State Library'.

        Returns:
            List[Dict[str, Any]]: A list of up to 5 matching places with their IDs, names, addresses, and coordinates.
        """
        query_lower = query.lower().strip()

        def _to_result(place):
            return {
                "id": place.place_id,
                "displayName": place.name,
                "address": place.address,
                "location": {"lat": place.location.lat, "lng": place.location.lng}
            }

        # --- Pass 1: exact full-query match ---
        results = []
        for place in self.db.places.values():
            if (query_lower in place.name.lower() or
                    query_lower in place.address.lower() or
                    any(query_lower in t.lower() for t in place.types)):
                results.append(_to_result(place))
        if results:
            return results[:5]

        # --- Pass 2: token fallback (skip common filler words) ---
        _STOP_WORDS = {
            "a", "an", "the", "near", "in", "at", "on", "by", "of",
            "restaurant", "cafe", "shop", "store", "bar", "hotel",
        }
        tokens = [t for t in query_lower.split() if t not in _STOP_WORDS]
        if not tokens:
            return []

        scored = []
        for place in self.db.places.values():
            haystack = f"{place.name.lower()} {place.address.lower()} {' '.join(t.lower() for t in place.types)}"
            matched = sum(1 for tok in tokens if tok in haystack)
            if matched > 0:
                scored.append((matched, place))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [_to_result(p) for _, p in scored[:5]]

    @is_tool(ToolType.READ)
    def place_details(self, place_id: str) -> Dict[str, Any]:
        """Fetches granular metadata for a specific location, including amenities, accessibility, and opening hours.

        Args:
            place_id: The unique identifier of the place, such as 'ChIJXYZ123456789'.

        Returns:
            Dict[str, Any]: Detailed information about the place.

        Raises:
            ValueError: If the place ID is not found in the database.
        """
        if place_id not in self.db.places:
            raise ValueError(f"Place ID {place_id} not found in database.")
        
        # Trả về toàn bộ data dạng dict
        return self.db.places[place_id].model_dump()

    @is_tool(ToolType.READ)
    def nearby_search(
        self, lat: float, lng: float, radius: float = 1000, 
        place_type: Optional[str] = None, min_rating: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Discovers points of interest within a spatial radius from a specific coordinate.

        Args:
            lat: Latitude of the center point, such as -37.8098.
            lng: Longitude of the center point, such as 144.9652.
            radius: The search radius in meters. Default is 1000.
            place_type: Optional category to filter, such as 'cafe' or 'restaurant'.
            min_rating: Minimum rating required, such as 4.0.

        Returns:
            List[Dict[str, Any]]: A list of up to 5 matching places sorted by proximity.
        """
        results = []
        for place in self.db.places.values():
            if place.rating < min_rating:
                continue
            if place_type and place_type.lower() not in [t.lower() for t in place.types]:
                continue
            
            dist = haversine_distance(lat, lng, place.location.lat, place.location.lng)
            if dist <= radius:
                results.append({
                    "id": place.place_id,
                    "name": place.name,
                    "distance_meters": int(dist),
                    "rating": place.rating,
                    "location": {"lat": place.location.lat, "lng": place.location.lng}
                })
                
        results.sort(key=lambda x: x["distance_meters"])
        return results[:5]

    @is_tool(ToolType.READ)
    def compute_routes(self, origin_lat: float, origin_lng: float, dest_lat: float, dest_lng: float, travel_mode: str = "DRIVE") -> Dict[str, Any]:
        """Generates actual paths (polylines) and metrics by looking up cached benchmark routes.

        Args:
            origin_lat: Latitude of the starting point.
            origin_lng: Longitude of the starting point.
            dest_lat: Latitude of the destination.
            dest_lng: Longitude of the destination.
            travel_mode: The mode of transportation, such as 'DRIVE'.

        Returns:
            Dict[str, Any]: Route details including distance, duration, and the encoded polyline.

        Raises:
            ValueError: If the specific route is not cached in the environment.
        """
        orig_lat_r, orig_lng_r = round(origin_lat, 3), round(origin_lng, 3)
        dest_lat_r, dest_lng_r = round(dest_lat, 3), round(dest_lng, 3)
        
        route_key = f"{orig_lat_r},{orig_lng_r}|{dest_lat_r},{dest_lng_r}|{travel_mode.upper()}"
        
        if route_key in self.db.routes:
            cached_route = self.db.routes[route_key]
            return {
                "routes": [{
                    "distanceMeters": cached_route.distance_meters,
                    "durationSeconds": cached_route.duration_seconds,
                    "polyline": {"encodedPolyline": cached_route.polyline},
                    "travelMode": travel_mode.upper()
                }]
            }
            
        raise ValueError(f"Route from origin to destination not supported in this offline benchmark environment. Please check coordinates.")

    @is_tool(ToolType.READ)
    def search_along_route(self, polyline: str, place_type: str) -> List[Dict[str, Any]]:
        """Identifies specific services falling within a predefined travel path using an encoded polyline.

        Args:
            polyline: The encoded polyline string obtained from compute_routes.
            place_type: The category of place to search for, such as 'cafe'.

        Returns:
            List[Dict[str, Any]]: A list of places found along the route.

        Raises:
            ValueError: If no cached results match the polyline and place type.
        """
        search_key = f"{polyline[:10]}_{place_type.lower()}"
        
        if search_key in self.db.search_along_routes:
            place_ids = self.db.search_along_routes[search_key]
            
            results = []
            for pid in place_ids:
                if pid in self.db.places:
                    p = self.db.places[pid]
                    results.append({
                        "id": p.place_id,
                        "name": p.name,
                        "location": {"lat": p.location.lat, "lng": p.location.lng}
                    })
            return results
            
        raise ValueError("No cached results found for this specific route and place type in the benchmark DB.")

    # =========================================================
    # SIMULATED WEB TOOLS
    # =========================================================
    
    @is_tool(ToolType.READ)
    def read_place_website(self, place_id: str) -> str:
        """Extracts Markdown text content (e.g., menus, announcements, FAQs) from a place's official website.
        The Agent only needs to provide the place_id, and the tool will automatically resolve the URL.

        Args:
            place_id: The unique identifier of the place, such as 'ChIJXYZ123456789'.

        Returns:
            str: The markdown content of the website, or an informative message if no website exists.

        Raises:
            ValueError: If the place is not found, or its website content is offline/not cached.
        """
        if place_id not in self.db.places:
            raise ValueError(f"Place ID '{place_id}' not found in the database.")
            
        place = self.db.places[place_id]
        website_url = place.website_url
        
        if not website_url:
            return f"The place '{place.name}' does not have an official website listed on Google Maps."
            
        if website_url not in self.db.webpages:
            raise ValueError(f"The website for '{place.name}' ({website_url}) is currently offline or not cached in the environment.")
            
        return self.db.webpages[website_url].content_markdown

    # =========================================================
    # SIMULATED BOOKING TOOLS (State Mutation)
    # =========================================================
    
    @is_tool(ToolType.READ)
    def check_availability(self, place_id: str, datetime_str: str, party_size: int) -> Dict[str, Any]:
        """Checks if the location has enough capacity at a specific time.

        Args:
            place_id: The unique identifier of the place.
            datetime_str: The requested date and time in 'YYYY-MM-DD HH:MM' format.
            party_size: The number of people for the reservation.

        Returns:
            Dict[str, Any]: A dictionary containing a boolean 'is_available' and a status 'message'.

        Raises:
            ValueError: If the place ID is not found.
        """
        if place_id not in self.db.places:
            raise ValueError("Place ID not found.")
            
        place = self.db.places[place_id]
        available_seats = place.available_slots.get(datetime_str, 0)
        
        if available_seats >= party_size:
            return {"is_available": True, "message": f"Available. {available_seats} seats left."}
        return {"is_available": False, "message": f"Full. Only {available_seats} seats left."}

    @is_tool(ToolType.WRITE)
    def book_place(self, place_id: str, user_id: str, datetime_str: str, party_size: int) -> Booking:
        """Makes a reservation and deducts available slots from the environment state.

        Args:
            place_id: The unique identifier of the place.
            user_id: The ID of the user making the booking, such as 'user_001'.
            datetime_str: The requested date and time in 'YYYY-MM-DD HH:MM' format.
            party_size: The number of people for the reservation.

        Returns:
            Booking: The confirmed booking record.

        Raises:
            ValueError: If the user is not found, or if the booking fails due to lack of capacity.
        """
        if user_id not in self.db.users:
            raise ValueError("User not found in system.")
            
        avail = self.check_availability(place_id, datetime_str, party_size)
        if not avail["is_available"]:
            raise ValueError(f"Booking failed: {avail['message']}")
            
        self.db.places[place_id].available_slots[datetime_str] -= party_size
        
        booking = Booking(
            booking_id=generate_booking_id(place_id, user_id, datetime_str),
            place_id=place_id,
            user_id=user_id,
            datetime_str=datetime_str,
            party_size=party_size
        )
        self.db.bookings[booking.booking_id] = booking
        return booking

    @is_tool(ToolType.READ)
    def get_transit_schedule(self, place_id: str, date_str: str) -> List[Dict[str, Any]]:
        """Retrieves the departure schedule for a specific public transit stop from the city's transport website.

        Args:
            place_id: The unique identifier of the transit stop or station (obtained via Map tools).
            date_str: The date for the schedule in 'YYYY-MM-DD' format.

        Returns:
            List[Dict[str, Any]]: A list of upcoming departures, including route numbers, destinations, and times.

        Raises:
            ValueError: If the place ID is not found, or if it is not a recognized transit stop.
        """
        # 1. Kiểm tra tồn tại trong DB Places chung
        if place_id not in self.db.places:
            raise ValueError(f"Place ID '{place_id}' not found in the database.")
            
        # 2. Kiểm tra xem có lịch trình xe/tàu cho Place này không
        # Lưu ý: Crawler DB của chúng ta giờ sẽ dùng place_id làm Key cho bảng transit_schedules
        if place_id not in self.db.transit_schedules:
            place_name = self.db.places[place_id].name
            raise ValueError(f"No transit schedule found for '{place_name}'. This location might not be a transit stop.")
            
        schedules = self.db.transit_schedules[place_id].get(date_str, [])
        return schedules

    @is_tool(ToolType.READ)
    def search_venue_events(self, place_id: str, date_str: str) -> List[Dict[str, Any]]:
        """Searches the venue's official website for events, exhibitions, or shows happening on a specific date.

        Args:
            place_id: The unique identifier of the venue (obtained via Map tools).
            date_str: The specific date to check for events in 'YYYY-MM-DD' format.

        Returns:
            List[Dict[str, Any]]: A list of events, including event name, time, ticket price, and description.

        Raises:
            ValueError: If the place ID is not found in the database.
        """
        if place_id not in self.db.places:
            raise ValueError(f"Place ID '{place_id}' not found in the database.")
            
        if place_id not in self.db.events:
             return [] # Không có lỗi, trả về rỗng nghĩa là venue này hôm đó không có sự kiện
             
        venue_events = self.db.events[place_id]
        # Lọc sự kiện theo ngày
        events_on_date = [e.model_dump() for e in venue_events if e.date == date_str]
        return events_on_date

    @is_tool(ToolType.WRITE)
    def submit_council_report(
        self, 
        issue_type: str, 
        place_id: str, 
        description: str, 
        user_id: str
    ) -> Dict[str, Any]:
        """Submits a maintenance report (e.g., pothole, graffiti) near a specific location to the City Council web portal.
        The tool automatically extracts the exact coordinates from the provided place_id.

        Args:
            issue_type: The category of the issue (e.g., 'pothole', 'graffiti', 'lighting', 'waste').
            place_id: The unique identifier of the place nearest to the issue.
            description: A short summary of the problem reported by the user.
            user_id: The ID of the user submitting the report.

        Returns:
            Dict[str, Any]: A confirmation receipt containing the 'ticket_id' and 'status'.

        Raises:
            ValueError: If the issue type is invalid, the place ID is not found, or user_id is not recognized.
        """
        valid_issues = ["pothole", "graffiti", "lighting", "waste", "other"]
        if issue_type.lower() not in valid_issues:
            raise ValueError(f"Invalid issue type. Must be one of: {valid_issues}")
            
        if user_id not in self.db.users:
            raise ValueError("User not found in system. Cannot submit report.")
            
        if place_id not in self.db.places:
            raise ValueError(f"Place ID '{place_id}' not found in the database. Please find a valid place first.")

        # Automatically extract exact coordinates from DB
        target_place = self.db.places[place_id]
        exact_lat = target_place.location.lat
        exact_lng = target_place.location.lng

        # FIXED: Use deterministic hashing instead of random UUID
        # Exclude 'description' to prevent hash mismatch when LLM phrasing changes
        import hashlib
        raw_ticket = f"{issue_type.lower()}_{place_id}_{user_id}"
        hashed_ticket = hashlib.md5(raw_ticket.encode('utf-8')).hexdigest()[:8].upper()
        ticket_id = f"TICKET-{hashed_ticket}"
        
        report_record = {
            "ticket_id": ticket_id,
            "issue_type": issue_type.lower(),
            "location": {"lat": exact_lat, "lng": exact_lng, "reference_place": target_place.name},
            "description": description,
            "user_id": user_id,
            "status": "Submitted"
        }
        
        # Write to DB state
        self.db.council_reports[ticket_id] = report_record
        
        return {
            "success": True,
            "ticket_id": ticket_id,
            "message": f"Successfully submitted {issue_type} report near '{target_place.name}' to the local council."
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