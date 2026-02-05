import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
import httpx
from fastapi import APIRouter, HTTPException, Query

# Setup logger
logger = logging.getLogger("fly_tlv.weather")

router = APIRouter(prefix="/api/weather", tags=["Weather"])

# Constants
CACHE_DIR = Path("cache")
CACHE_FILE = CACHE_DIR / "weather_cache.json"
CACHE_TTL_HOURS = 24

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def load_cache() -> dict:
    """Load the weather cache from disk."""
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load weather cache: {e}")
        return {}

def save_cache(data: dict):
    """Save the weather cache to disk."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save weather cache: {e}")

async def fetch_open_meteo(lat: float, lon: float) -> dict:
    """Fetch weather data from Open-Meteo API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": "true",
        "daily": "weathercode,temperature_2m_max,temperature_2m_min",
        "timezone": "auto",
        "forecast_days": 1
    }
    
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

@router.get("")
async def get_weather(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude")
):
    """
    Get weather for a specific location.
    Uses a 24-hour file-based cache.
    """
    cache = load_cache()
    key = f"{lat},{lon}"
    
    # Check cache
    if key in cache:
        entry = cache[key]
        timestamp = entry.get("timestamp")
        if timestamp:
            cached_time = datetime.fromisoformat(timestamp)
            if datetime.utcnow() - cached_time < timedelta(hours=CACHE_TTL_HOURS):
                logger.info(f"Using cached weather for {key}")
                return entry["data"]

    # Fetch new data
    try:
        data = await fetch_open_meteo(lat, lon)
    except httpx.HTTPError as e:
        logger.error(f"Open-Meteo API error: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch weather data")
    
    # Update cache
    cache[key] = {
        "timestamp": datetime.utcnow().isoformat(),
        "data": data
    }
    save_cache(cache)
    
    logger.debug(f"Fetched and cached new weather for {key}")
    return data

async def cleanup_weather_cache_task():
    """
    Scheduled Task: Cleans up old cache entries (> 24 hours).
    """
    logger.debug("Running weather cache cleanup task...")
    cache = load_cache()
    now = datetime.utcnow()
    keys_to_delete = []

    for key, entry in cache.items():
        timestamp = entry.get("timestamp")
        if not timestamp:
            keys_to_delete.append(key)
            continue
            
        cached_time = datetime.fromisoformat(timestamp)
        if now - cached_time > timedelta(hours=CACHE_TTL_HOURS):
            keys_to_delete.append(key)

    if keys_to_delete:
        for k in keys_to_delete:
            del cache[k]
        save_cache(cache)
        logger.debug(f"Cleaned up {len(keys_to_delete)} expired weather cache entries.")
    else:
        logger.debug("No expired weather cache entries found.")
