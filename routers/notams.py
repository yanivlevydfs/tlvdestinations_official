import json
import logging
import re
from pathlib import Path
from datetime import datetime, timedelta
import httpx
from fastapi import APIRouter, HTTPException, Query, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from core.templates import TEMPLATES
from helpers.helper import get_lang

# Setup logger
logger = logging.getLogger("fly_tlv.notams")

router = APIRouter(tags=["NOTAMs"])

# Constants
CACHE_DIR = Path("cache")
NOTAM_CACHE_FILE = CACHE_DIR / "notam_cache.json"
CACHE_TTL_HOURS = 5
AVIATION_EDGE_API_KEY = "da33f3-b78037" # Provided by user in request

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def extract_notam_description(condition: str) -> str:
    """
    Extracts the human-readable description part of a NOTAM. 
    NOTAMs usually have a field 'E)' containing the description.
    """
    if not condition:
        return ""
    
    # Use regex to find the content after 'E)' or 'e)' 
    match = re.search(r'[Ee]\)\s*(.*?)(?=\s+[FfGg]\)|$)', condition, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return condition

def load_cache() -> dict:
    """Load the NOTAM cache from disk."""
    if not NOTAM_CACHE_FILE.exists():
        return {}
    try:
        with open(NOTAM_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load NOTAM cache: {e}")
        return {}

def save_cache(data: dict):
    """Save the NOTAM cache to disk."""
    try:
        with open(NOTAM_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save NOTAM cache: {e}")

@router.get("/notams", response_class=HTMLResponse)
async def notams_page(request: Request, lang: str = Depends(get_lang)):
    """Render the NOTAM router UI."""
    from main import AIRPORTS_DB
    
    airport_list = []
    if AIRPORTS_DB:
        for iata, data in AIRPORTS_DB.items():
            if iata and data.get('name'):
                name = data.get('name', 'Unknown')
                city = data.get('city', '')
                label = f"{iata} - {name}"
                if city:
                    label += f" ({city})"
                airport_list.append({"iata": iata, "label": label})
    
    airport_list.sort(key=lambda x: x['iata'])

    return TEMPLATES.TemplateResponse("notams.html", {
        "request": request,
        "lang": lang,
        "airports": airport_list,
        "current_date": datetime.now().strftime("%Y-%m-%d")
    })

@router.get("/api/notams")
async def get_notams(
    iata: str = Query(..., description="Airport IATA code"),
    date: str = Query(None, description="Date in YYYY-MM-DD format")
):
    """
    Fetch NOTAMs for a specific airport.
    Uses a 5-hour file-based cache.
    """
    iata = iata.upper().strip()
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    cache = load_cache()
    cache_key = f"{iata}_{date}"
    
    # Check cache
    if cache_key in cache:
        entry = cache[cache_key]
        timestamp = entry.get("timestamp")
        if timestamp:
            cached_time = datetime.fromisoformat(timestamp)
            if datetime.now() - cached_time < timedelta(hours=CACHE_TTL_HOURS):
                return entry["data"]

    # Fetch new data
    url = f"https://aviation-edge.com/v2/public/notams?key={AVIATION_EDGE_API_KEY}&iata={iata}&date_from={date}"
    
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        logger.error(f"Aviation Edge API error for {iata}: {e}")
        if cache_key in cache:
            return cache[cache_key]["data"]
        raise HTTPException(status_code=502, detail="Failed to fetch NOTAM data")
    except Exception as e:
        logger.error(f"Unexpected error fetching NOTAMs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    # Process descriptions
    if isinstance(data, list):
        for notam in data:
            condition = notam.get("condition", "")
            notam["friendly_text"] = extract_notam_description(condition)

    # Update cache
    cache[cache_key] = {
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    save_cache(cache)
    
    return data
