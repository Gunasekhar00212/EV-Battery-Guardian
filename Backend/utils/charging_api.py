import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENCHARGEMAP_API_KEY = os.getenv("OPENCHARGEMAP_API_KEY")

if not OPENCHARGEMAP_API_KEY:
    raise RuntimeError("OPENCHARGEMAP_API_KEY not found in .env")


def fetch_nearby_charging_stations(
    lat: float,
    lon: float,
    radius_km: int = 20,
    max_results: int = 10
):
    """
    Fetch nearby charging stations from OpenChargeMap API
    """

    url = "https://api.openchargemap.io/v3/poi/"

    params = {
        "key": OPENCHARGEMAP_API_KEY,
        "latitude": lat,
        "longitude": lon,
        "distance": radius_km,
        "distanceunit": "KM",
        "maxresults": max_results,
        "compact": True,
        "verbose": False
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    return response.json()


def normalize_charging_stations(raw_stations):
    """
    Convert OpenChargeMap station data into Guardian-friendly format
    """

    normalized = []

    for station in raw_stations:
        address = station.get("AddressInfo", {})

        lat = address.get("Latitude")
        lon = address.get("Longitude")
        name = address.get("Title", "Unknown Station")

        # Skip invalid stations
        if lat is None or lon is None:
            continue

        normalized.append({
            "lat": float(lat),
            "lon": float(lon),
            "name": name
        })

    return normalized



