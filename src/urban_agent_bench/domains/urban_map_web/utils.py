import hashlib
from urban_agent_bench.utils.utils import DATA_DIR

# Define paths for the static benchmark environment
URBAN_WEB_DATA_DIR = DATA_DIR / "domains" / "urban_map_web"
URBAN_WEB_DB_PATH = URBAN_WEB_DATA_DIR / "db.json"
URBAN_WEB_POLICY_PATH = URBAN_WEB_DATA_DIR / "policy.md"
URBAN_WEB_TASK_SET_PATH = URBAN_WEB_DATA_DIR / "tasks.json"

def generate_booking_id(place_id: str, user_id: str, datetime_str: str) -> str:
    """Generate a deterministic booking ID for reservations based on inputs."""
    # Gom các tham số lại thành 1 chuỗi duy nhất
    raw_string = f"{place_id}_{user_id}_{datetime_str}"
    # Băm chuỗi này ra, lấy 8 ký tự đầu và in hoa
    hashed = hashlib.md5(raw_string.encode()).hexdigest()[:8].upper()
    return f"BK-{hashed}"