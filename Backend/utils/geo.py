import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two GPS points in kilometers
    """
    R = 6371  # Earth radius in km

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2 +
        math.cos(phi1) * math.cos(phi2) *
        math.sin(dlambda / 2) ** 2
    )

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
def estimate_road_distance_km(lat1, lon1, lat2, lon2, road_factor=1.3):
    """
    Estimate road distance using haversine distance
    multiplied by a road factor.
    """
    straight_km = haversine_distance(lat1, lon1, lat2, lon2)
    return straight_km * road_factor
