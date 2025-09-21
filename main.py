# === Standard Library ===
import os
import sys
import time
import json
import asyncio
import logging
from html import escape
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Any, Dict, List
import threading
from math import radians, sin, cos, sqrt, atan2
import pycountry
import asyncio
import aiohttp
import re
import string
from pydantic import BaseModel
import google.generativeai as genai
from fastapi import FastAPI, Request, Response, Query, HTTPException, Depends, Body
import random
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from bs4 import BeautifulSoup
# === Third-Party Libraries ===
import pandas as pd
import requests
import folium
import airportsdata
from folium.plugins import MarkerCluster
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from logging_setup import setup_logging, get_app_logger
from fastapi import Depends
from fastapi.responses import FileResponse
# === Local Modules ===
from sitemap_tool import Url, build_sitemap

os.environ["PYTHONUTF8"] = "1"
try:
    enc = (sys.stdout.encoding or "").lower()
except Exception:
    enc = ""
if enc != "utf-8":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# --- Logging ---
scheduler = None
setup_logging(log_level="INFO", log_dir="logs", log_file_name="app.log")
logger = get_app_logger("flights_explorer")
logger.info("Server starting…")

# Set up Gemini API

genai.configure(api_key="AIzaSyBxOJwavtKVB9gJvA2OoAsKw90GogBNdZs")
chat_model = genai.GenerativeModel("gemini-2.5-flash")

class ChatQuery(BaseModel):
    question: str

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent
CACHE_DIR: Path = BASE_DIR / "cache"
STATIC_DIR: Path = BASE_DIR / "static"
TEMPLATES_DIR: Path = BASE_DIR / "templates"
DATA_DIR: Path = BASE_DIR / "data"

# Ensure dirs exist (won't error if already exist)
for d in (CACHE_DIR, TEMPLATES_DIR, STATIC_DIR, DATA_DIR):
    d.mkdir(exist_ok=True)

# Templates
TEMPLATES = Jinja2Templates(directory=str(TEMPLATES_DIR))
TEMPLATES.env.globals["now"] = datetime.utcnow

# Data files
AIRLINE_WEBSITES_FILE = DATA_DIR / "airline_websites.json"
ISRAEL_FLIGHTS_FILE   = CACHE_DIR / "israel_flights.json"
TRAVEL_WARNINGS_FILE = CACHE_DIR / "travel_warnings.json"

# Constants
TLV = {"IATA": "TLV", "Name": "Ben Gurion Airport", "lat": 32.0068, "lon": 34.8853}
ISRAEL_API = "https://data.gov.il/api/3/action/datastore_search"
RESOURCE_ID = "e83f763b-b7d7-479e-b172-ae981ddc6de5"
DEFAULT_LIMIT = 32000

TRAVEL_WARNINGS_API = "https://data.gov.il/api/3/action/datastore_search"
TRAVEL_WARNINGS_RESOURCE = "2a01d234-b2b0-4d46-baa0-cec05c401e7d"


# App
app = FastAPI(title="Flights Explorer (FastAPI)")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

CORS_ORIGINS = ["http://localhost:8000", "https://fly-tlv.com"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ───────────────────────────────────────────────
# Global in-memory dataset
# ───────────────────────────────────────────────
DATASET_DF: pd.DataFrame = pd.DataFrame()
DATASET_DATE: str = ""
AIRLINE_WEBSITES: dict = {}
scheduler: AsyncIOScheduler | None = None
# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def datetimeformat(value: str, fmt: str = "%d/%m/%Y %H:%M"):
    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime(fmt)
    except Exception:
        return value
TEMPLATES.env.filters["datetimeformat"] = datetimeformat


def normalize_case(value: str) -> str:
    """Capitalize each word properly, safe for missing/placeholder values."""
    if not value or value == "—":
        return value or "—"
    return string.capwords(value.strip())

def israel_flights_is_stale(max_age_hours: int = 24) -> bool:
    """Return True if the israel flights cache file is older than max_age_hours or missing."""
    if not ISRAEL_FLIGHTS_FILE.exists():
        return True
    age_hours = (datetime.now() - datetime.fromtimestamp(ISRAEL_FLIGHTS_FILE.stat().st_mtime)).total_seconds() / 3600
    return age_hours > max_age_hours

def travel_warnings_is_stale(max_age_hours: int = 24) -> bool:
    """Return True if the travel warnings cache file is older than max_age_hours or missing."""
    if not TRAVEL_WARNINGS_FILE.exists():
        return True
    age_hours = (datetime.now() - datetime.fromtimestamp(TRAVEL_WARNINGS_FILE.stat().st_mtime)).total_seconds() / 3600
    return age_hours > max_age_hours


def _dataset_file_date() -> str | None:
    """Return the DD-MM-YYYY date from 'updated' inside ISRAEL_FLIGHTS_FILE, or None."""
    if ISRAEL_FLIGHTS_FILE.exists():
        try:
            with open(ISRAEL_FLIGHTS_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
            updated = meta.get("updated")
            if updated:
                # Example: "2025-09-14T17:23:16.379406" → "14-09-2025"
                iso_date = updated.split("T")[0]  # "2025-09-14"
                return datetime.strptime(iso_date, "%Y-%m-%d").strftime("%d-%m-%Y")
        except Exception as e:
            logger.error(f"Failed to read israel flights file: {e}")
    return None


def _is_dataset_fresh_today() -> tuple[bool, str | None]:
    """Check if dataset is for today (checks in-memory first, then file)."""
    today = datetime.now().strftime("%d-%m-%Y")
    # in-memory
    if DATASET_DATE == today and not DATASET_DF.empty:
        return True, today
    # on disk
    file_date = _dataset_file_date()
    return (file_date == today), file_date

def get_lang(request: Request) -> str:
    return "he" if request.query_params.get("lang") == "he" else "en"

def get_all_countries(data) -> list[str]:
    """Return unique countries from data with 'All' first, rest alphabetically."""
    if hasattr(data, "columns"):  # DataFrame
        countries = {str(c).strip() for c in data["Country"].dropna()}
    elif isinstance(data, list):  # list of dicts
        countries = {str(item.get("Country", "")).strip() for item in data if item.get("Country")}
    else:
        return ["All"]

    countries = sorted(countries)
    return ["All"] + countries
    
def safe_js(text: str) -> str:
    """Escape backticks, quotes and newlines for safe JS embedding"""
    if text is None:
        return ""
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("`", "\\`")   # ✅ escape backticks
        .replace('"', '\\"')   # escape double quotes
        .replace("'", "\\'")   # escape single quotes
        .replace("\n", " ")
        .replace("\r", "")
    )
    
# ──────────────────────────────────────────────────────────────────────────────
# Load airports once for dataset builds
# ──────────────────────────────────────────────────────────────────────────────

def _clean_html(raw: str) -> str:
    if not raw:
        return ""
    return BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)


def _extract_first_href(raw: str) -> str:
    if not raw:
        return ""
    match = re.search(r'href="([^"]+)"', raw)
    return match.group(1) if match else ""


def _extract_first_img(raw: str) -> dict:
    if not raw:
        return {}
    match = re.search(r'<img src="([^"]+)" alt="([^"]+)"', raw)
    if match:
        return {"src": match.group(1), "alt": match.group(2)}
    return {}


def _extract_threat_level(text: str) -> str:
    """
    מזהה רמת איום מתוך ההמלצות
    מחזיר High / Medium / Low / Unknown
    """
    if not text:
        return "Unknown"
    t = text.strip()
    if "רמה 4" in t or "גבוה" in t:
        return "High"
    if "רמה 3" in t or "בינוני" in t:
        return "Medium"
    if "רמה 2" in t or "נמוך" in t:
        return "Low"
    return "Unknown"


def travel_warnings_is_stale(max_age_hours: int = 24) -> bool:
    """בדיקה אם הקובץ ישן"""
    if not TRAVEL_WARNINGS_FILE.exists():
        return True
    age_hours = (datetime.now() - datetime.fromtimestamp(TRAVEL_WARNINGS_FILE.stat().st_mtime)).total_seconds() / 3600
    return age_hours > max_age_hours


def fetch_travel_warnings(batch_size: int = 500) -> dict | None:
    """Fetch ALL travel warnings from gov.il (handles pagination) and cache them locally."""
    offset = 0
    all_records = []

    try:
        while True:
            params = {
                "resource_id": TRAVEL_WARNINGS_RESOURCE,
                "limit": batch_size,
                "offset": offset,
            }
            r = requests.get(TRAVEL_WARNINGS_API, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()

            records = data.get("result", {}).get("records", [])
            if not records:
                break

            for rec in records:
                raw_reco = rec.get("recommendations", "")
                raw_details = rec.get("details", "")

                all_records.append({
                    "id": rec.get("_id"),
                    "continent": rec.get("continent", "").strip(),
                    "country": rec.get("country", "").strip(),
                    # טקסט נקי בלבד
                    "recommendations": _clean_html(raw_reco),
                    # שליפת רמת איום מתוך הטקסט
                    "level": _extract_threat_level(raw_reco),
                    # שמירת לינק – קודם מ־recommendations, ואם אין אז מ־details
                    "details_url": (
                        _extract_first_href(raw_reco)
                        or _extract_first_href(raw_details)
                    ),
                    # לוגו
                    "logo": _extract_first_img(rec.get("logo", "")),
                    "date": rec.get("date"),
                    "office": rec.get("משרד", ""),
                })

            offset += batch_size
            total = data.get("result", {}).get("total")
            if total and offset >= total:
                break

        result = {
            "updated": datetime.now().isoformat(),
            "count": len(all_records),
            "warnings": all_records,
        }

        with open(TRAVEL_WARNINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"Travel warnings refreshed: {len(all_records)} total")
        return result

    except Exception as e:
        logger.error(f"Failed to fetch travel warnings: {e}")
        return None



def get_dataset_date():
    if ISRAEL_FLIGHTS_FILE.exists():
        with open(ISRAEL_FLIGHTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        updated = data.get("updated")
        if updated:
            try:
                iso_date = updated.split("T")[0]
                return datetime.strptime(iso_date, "%Y-%m-%d").strftime("%d-%m-%Y")
            except Exception:
                return iso_date
    return None

    
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return round(R * 2 * atan2(sqrt(a), sqrt(1 - a)), 1)

def load_airports_all() -> pd.DataFrame:
    iata_db = airportsdata.load("IATA")
    rows = []

    for iata, rec in iata_db.items():
        country = normalize_case(rec.get("country"))
        name = normalize_case(rec.get("name"))
        city = normalize_case(rec.get("city"))
        lat, lon = rec.get("lat"), rec.get("lon")

        dist_km = flight_time_hr = None
        if lat is not None and lon is not None:
            dist_km = round(haversine_km(TLV["lat"], TLV["lon"], lat, lon), 1)
            flight_time_hr = round(dist_km / 800, 2)  # assume 800 km/h

        rows.append({
            "IATA": iata.upper(),
            "Name": name,
            "City": city,
            "Country": country,
            "lat": lat,
            "lon": lon,
            "Distance_km": dist_km,
            "FlightTime_hr": flight_time_hr,
        })

    return pd.DataFrame(rows)

def load_airline_names() -> Dict[str, str]:
    cols = ["AirlineID","Name","Alias","IATA","ICAO","Callsign","Country","Active"]
    try:
        df = pd.read_csv(BASE_DIR / "data" / "airlines.dat", names=cols, header=None, encoding="utf-8")
        return df.set_index("IATA")["Name"].dropna().to_dict()
    except Exception as e:
        logger.warning(f"Airline names not loaded: {e}")
        return {}

AIRLINE_NAMES = load_airline_names()
AIRPORTS = load_airports_all()

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
 
def normalize_airline_list(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for name in items:
        if not name:
            continue
        key = str(name).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(key.title())
    return out
    
def fetch_israel_flights() -> dict | None:
    """Fetch gov.il flights (all countries) and cache to disk."""
    params = {
        "resource_id": RESOURCE_ID,
        "limit": DEFAULT_LIMIT
    }

    try:
        r = requests.get(ISRAEL_API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        flights = [
            {
                "airline": normalize_case(rec.get("CHOPERD", "—")),
                "iata": (rec.get("CHLOC1") or "—").upper(),  # IATA always uppercase
                "airport": normalize_case(rec.get("CHLOC1D", "—")),
                "city": normalize_case(rec.get("CHLOC1T", "—")),
                "country": normalize_case(rec.get("CHLOCCT", "—")),
                "scheduled": normalize_case(rec.get("CHSTOL", "—")),
                "actual": normalize_case(rec.get("CHPTOL", "—")),
                "direction": normalize_case(rec.get("CHAORD", "—")),
                "status": normalize_case(rec.get("CHRMINE", "—")),
            }
            for rec in data.get("result", {}).get("records", [])
        ]

        result = {
            "updated": datetime.now().isoformat(),
            "count": len(flights),
            "flights": flights
        }

        # Cache locally
        with open(ISRAEL_FLIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"gov.il flights refreshed: {len(flights)} total flights cached")
        return result

    except Exception as e:
        logger.error(f"Failed to fetch gov.il flights: {e}")
        return None

def _read_dataset_file() -> tuple[pd.DataFrame, str | None]:
    """Read israel_flights.json and convert flights → unique airports DataFrame."""
    if ISRAEL_FLIGHTS_FILE.exists():
        try:
            with open(ISRAEL_FLIGHTS_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)

            flights = meta.get("flights", [])
            iata_db = airportsdata.load("IATA")

            grouped = {}
            for f in flights:
                iata = f.get("iata")
                if not iata:
                    continue
                entry = grouped.setdefault(iata, {
                    "IATA": iata,
                    "Name": f.get("airport") or "—",
                    "City": f.get("city") or "—",
                    "Country": f.get("country") or "—",
                    "Airlines": set(),
                })
                airline = f.get("airline")
                if airline:
                    entry["Airlines"].add(airline)

            rows = []
            for iata, info in grouped.items():
                coords = iata_db.get(iata, {})
                lat, lon = coords.get("lat"), coords.get("lon")
                dist_km = haversine_km(TLV["lat"], TLV["lon"], lat, lon) if lat and lon else None
                flight_hr = round(dist_km / 800, 2) if dist_km else None
                rows.append({
                    **info,
                    "lat": lat,
                    "lon": lon,
                    "Distance_km": dist_km,
                    "FlightTime_hr": flight_hr,
                    "Airlines": sorted(info["Airlines"]),
                })

            df = pd.DataFrame(rows)

            # Get date from "updated"
            updated = meta.get("updated")
            file_date = None
            if updated:
                try:
                    iso_date = updated.split("T")[0]
                    file_date = datetime.strptime(iso_date, "%Y-%m-%d").strftime("%d-%m-%Y")
                except Exception:
                    file_date = iso_date

            return df, file_date

        except Exception as e:
            logger.error(f"Failed to read dataset file: {e}")

    # Empty fallback
    return pd.DataFrame(columns=["IATA", "Name", "City", "Country", "lat", "lon", "Airlines"]), None


def get_dataset(trigger_refresh: bool = False) -> pd.DataFrame:
    """
    Return the current dataset from memory or disk.
    If trigger_refresh=True and the dataset is stale, fetch fresh data immediately.
    """
    global DATASET_DF, DATASET_DATE

    fresh, file_date = _is_dataset_fresh_today()

    # ✅ Return in-memory if fresh
    if fresh and not DATASET_DF.empty:
        return DATASET_DF

    # ✅ Try loading from disk
    df, disk_date = _read_dataset_file()
    DATASET_DF, DATASET_DATE = df, disk_date or ""

    # ✅ If fresh on disk or refresh not requested
    if fresh or not trigger_refresh:
        return DATASET_DF

    # ✅ Refresh now (blocking, no background thread)
    logger.info("Dataset stale → refreshing now")
    fetch_israel_flights()

    # ✅ Reload dataset after refresh
    df, disk_date = _read_dataset_file()
    DATASET_DF, DATASET_DATE = df, disk_date or ""
    return DATASET_DF

def load_israel_flights_map():
    """Return dict {IATA: set(airlines)} from gov.il cache"""
    flights = []
    if ISRAEL_FLIGHTS_FILE.exists():
        try:
            with open(ISRAEL_FLIGHTS_FILE, "r", encoding="utf-8") as f:
                flights = json.load(f).get("flights", [])
        except Exception as e:
            logger.error(f"Failed to read gov.il flights: {e}")
            return {}

    # Build map
    mapping = {}
    for f in flights:
        iata = f.get("iata")
        airline = f.get("airline")
        if not iata or not airline:
            continue
        normalized = airline.strip().lower()
        if normalized:
            mapping.setdefault(iata, set()).add(normalized)
    return mapping

def start_refresh_thread():
    t = threading.Thread(target=fetch_israel_flights, daemon=True)
    t.start()
    
def load_airline_websites() -> dict:
    try:
        with open(AIRLINE_WEBSITES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load airline websites: {e}")
        return {}
# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    country: str = "All",
    query: str = "",    
    lang: str = Depends(get_lang), 
):
    # --- Step 1: Load dataset (Greek & Cyprus airports) ---
    df = get_dataset(trigger_refresh=False).copy()

    # Apply filters (local only)
    if country and country != "All":
        df = df[df["Country"].str.lower() == country.lower()]
    if query:
        q = query.strip().lower()
        if q:
            df = df[df.apply(lambda r: q in str(r["IATA"]).lower()
                                  or q in str(r["Name"]).lower()
                                  or q in str(r["City"]).lower()
                                  or q in str(r["Country"]).lower(),axis=1)]

    # --- Step 2: Merge AE airlines with gov.il flights ---
    govil_map = load_israel_flights_map()
    airports = df.to_dict(orient="records")

    for ap in airports:
        raw = ap.get("Airlines", [])

        # Normalize AE airlines
        if isinstance(raw, list):
            ae_airlines = set(a.strip().lower() for a in raw if a)
        elif isinstance(raw, str) and raw != "—":
            ae_airlines = set(a.strip().lower() for a in raw.split(",") if a.strip())
        else:
            ae_airlines = set()

        # Normalize gov.il airlines
        govil_airlines = set()
        for g in govil_map.get(ap["IATA"], set()):
            if isinstance(g, str):
                govil_airlines.add(g.strip().lower())

        merged = sorted(normalize_airline_list(list(ae_airlines.union(govil_airlines))))
        ap["Airlines"] = merged if merged else ["—"]
        
    # --- Step 3: Add missing gov.il-only destinations (group + merge) ---
    if ISRAEL_FLIGHTS_FILE.exists():
        try:
            with open(ISRAEL_FLIGHTS_FILE, "r", encoding="utf-8") as f:
                govil_json = json.load(f)
                govil_flights = govil_json.get("flights", [])
        except Exception as e:
            logger.error(f"Failed to read gov.il flights: {e}")
            govil_flights = []

        iata_db = airportsdata.load("IATA")
        existing_iatas = {ap["IATA"] for ap in airports}

        grouped: dict[str, dict] = {}
        for rec in govil_flights:
            iata = rec.get("iata")
            if not iata or iata in existing_iatas:
                continue

            if iata not in grouped:
                grouped[iata] = {
                    "Name": rec.get("airport") or "—",
                    "City": rec.get("city") or "—",
                    "Country": rec.get("country") or "—",
                    "Airlines": set(),
                }

            airline = rec.get("airline")
            if airline:
                grouped[iata]["Airlines"].add(airline)   
         
        for iata, meta in grouped.items():
            lat = lon = None
            dist_km = flight_time_hr = "—"

            if iata in iata_db:
                lat, lon = iata_db[iata]["lat"], iata_db[iata]["lon"]
                if lat is not None and lon is not None:
                    dist_km = round(haversine_km(TLV["lat"], TLV["lon"], lat, lon), 1)
                    flight_time_hr = round(dist_km / 800, 2)
            airports.append({
                "IATA": iata,
                "Name": meta.get("Name") or "—",
                "City": meta.get("City") or "—",
                "Country": meta.get("Country") or "—",
                "lat": lat,
                "lon": lon,
                "Airlines": normalize_airline_list(list(meta.get("Airlines", []))),
                "Distance_km": dist_km,
                "FlightTime_hr": flight_time_hr,
            })

    # --- Step 4: Render template ---
    countries = get_all_countries(airports)
    last_update = get_dataset_date()
    logger.info(f"GET /  country={country} query='{query}'  rows={len(airports)}")

    return TEMPLATES.TemplateResponse(
        "index.html",
        {
            "request": request,
            "last_update": last_update,
            "lang": lang,
            "now": datetime.now(), 
            "airports": airports,
            "countries": countries, 
            "country": country,
            "query": query,
            "AIRLINE_WEBSITES": AIRLINE_WEBSITES,
        },
    )
@app.get("/map", response_class=HTMLResponse)
def map_view(
    country: str = "All",
    query: str = "",
):
    # DO NOT trigger refresh here; just use cached data
    df = get_dataset(trigger_refresh=False).copy()
    airports = df.to_dict(orient="records")

    # --- Merge gov.il-only destinations ---
    if ISRAEL_FLIGHTS_FILE.exists():
        try:
            govil_json = json.load(open(ISRAEL_FLIGHTS_FILE, "r", encoding="utf-8"))
            govil_flights = govil_json.get("flights", [])
        except Exception as e:
            logger.error(f"Failed to read gov.il flights: {e}")
            govil_flights = []

        iata_db = airportsdata.load("IATA")
        existing_iatas = {ap["IATA"] for ap in airports}
        grouped: dict[str, dict] = {}

        for rec in govil_flights:
            iata = rec.get("iata")
            if not iata or iata in existing_iatas:
                continue

            if iata not in grouped:
                grouped[iata] = {
                    "Name": rec.get("airport") or "—",
                    "City": rec.get("city") or "—",
                    "Country": rec.get("country") or "—",
                    "Airlines": set(),
                }
            if rec.get("airline"):
                grouped[iata]["Airlines"].add(rec["airline"])

        for iata, meta in grouped.items():
            lat = lon = None
            dist_km = flight_time_hr = "—"
            if iata in iata_db:
                lat, lon = iata_db[iata]["lat"], iata_db[iata]["lon"]
                if lat and lon:
                    dist_km = round(haversine_km(TLV["lat"], TLV["lon"], lat, lon), 1)
                    flight_time_hr = round(dist_km / 800, 2)

            airports.append({
                "IATA": iata,
                "Name": meta["Name"].title(),
                "City": meta["City"].title(),
                "Country": meta["Country"].title(),
                "lat": lat,
                "lon": lon,
                "Airlines": sorted(meta["Airlines"]) if meta["Airlines"] else ["—"],
                "Distance_km": dist_km,
                "FlightTime_hr": flight_time_hr,
            })

    # ---- Apply filters ----
    if country and country != "All":
        airports = [ap for ap in airports if ap["Country"].lower() == country.lower()]
    if query:
        q = query.strip().lower()
        if q:
            airports = [
                ap for ap in airports
                if q in str(ap["IATA"]).lower()
                or q in str(ap["Name"]).lower()
                or q in str(ap["City"]).lower()
                or q in str(ap["Country"]).lower()
            ]

    # ---- Base map ----
    m = folium.Map(
        location=[35, 28.5],
        zoom_start=5,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
        zoom_control=True
    )
    cluster = MarkerCluster().add_to(m)

    # Disable scroll wheel zoom (prevents messy zoom inside modal)
    #m.options["scrollWheelZoom"] = False

    # Add TLV marker
    folium.Marker(
        [TLV["lat"], TLV["lon"]],
        tooltip="Tel Aviv (TLV)",
        icon=folium.Icon(color="blue", icon="plane", prefix="fa")
    ).add_to(m)

    bounds = [[TLV["lat"], TLV["lon"]]]

    for ap in airports:
        if not ap.get("lat") or not ap.get("lon"):
            logger.warning(f"Missing coords for {ap['IATA']} ({ap['City']}, {ap['Country']})")
            # fallback: skip marker but keep them in airports list
            continue

        # ✅ this must be OUTSIDE the "if missing" block
        #logger.info(f"Placing {len([ap for ap in airports if ap.get('lat') and ap.get('lon')])} markers on map")
        km = haversine_km(TLV["lat"], TLV["lon"], ap["lat"], ap["lon"])
        flight_time_hr = round(km / 800, 1) if km else "—"  # avg ~800 km/h

        flight_time_hr = round(km / 800, 1) if km else "—"
        flights_url    = f"https://www.google.com/travel/flights?q=flights%20from%20TLV%20to%20{ap['IATA']}"
        skyscanner_url = f"https://www.skyscanner.net/transport/flights/tlv/{ap['IATA'].lower()}/"
        gmaps_url      = f"https://maps.google.com/?q={ap['City'].replace(' ','+')},{ap['Country'].replace(' ','+')}"
        copy_js = f"navigator.clipboard && navigator.clipboard.writeText('{safe_js(ap['IATA'])}')"

        # ---- Airlines as clickable chips ----
        airlines_val = ap.get("Airlines", "—")

        if isinstance(airlines_val, (list, tuple)):
            airlines_val = ",".join(map(str, airlines_val))
        elif not isinstance(airlines_val, str):
            airlines_val = str(airlines_val)

        if airlines_val.strip() and airlines_val != "—" and airlines_val.lower() != "nan":
            chips = []
            for a in airlines_val.split(","):
                name = a.strip()
                if not name:
                    continue
                url = AIRLINE_WEBSITES.get(name)  # your dict
                style = (
                    "display:inline-block;"
                    "margin:2px 4px 2px 0;"
                    "padding:3px 8px;"
                    "font-size:12px;"
                    "border-radius:9999px;"
                    "background:#f3f4f6;"
                    "color:#111827;"
                    "text-decoration:none;"
                    "border:1px solid #d1d5db;"
                    "transition:all 0.25s ease-in-out;"
                )
                if url:
                    chips.append(f"<a href='{escape(url)}' target='_blank' class='chip' style='{style}'>{escape(name)}</a>")
                else:
                    chips.append(f"<span class='chip' style='{style}'>{escape(name)}</span>")

            # ✅ add CSS once
            chip_css = """
            <style>
            a.chip, span.chip {
              transition: all 0.25s ease-in-out;
            }
            a.chip:hover, span.chip:hover {
              background-color:#0d6efd !important;
              color:#fff !important;
              border-color:#0d6efd !important;
              transform:scale(1.08);
              box-shadow:0 3px 6px rgba(0,0,0,.2);
            }
            </style>
            """

            airline_html = (
                chip_css +
                "<div style='margin-top:6px;font-size:13px'>Airlines:<br>"
                + "".join(chips) +
                "</div>"
            )
        else:
            airline_html = "<div style='margin-top:6px;font-size:13px'>Airlines: <b>—</b></div>"

        # ---- Popup HTML ----
        popup_html = f"""
            <div style='font-family:system-ui;min-width:250px;max-width:300px'>
                <div style='font-weight:600;font-size:15px'>{escape(str(ap['Name']))} ({escape(str(ap['IATA']))})</div>
                <div style='color:#6b7280;font-size:12px;margin-top:2px'>{escape(str(ap['City']))} · {escape(str(ap['Country']))}</div>
                <div style='margin-top:6px;font-size:13px'>Distance: <b>{km} km</b></div>
                <div style='margin-top:2px;font-size:13px'>Flight time: <b>{flight_time_hr} hr</b></div>
                {airline_html}
                <div style='margin-top:8px;display:flex;gap:6px;flex-wrap:wrap'>
                    <a href='{escape(flights_url)}' target='_blank' style='text-decoration:none;background:#2563eb;color:white;padding:4px 8px;border-radius:20px;font-size:13px'>Google Flights</a>
                    <a href='{escape(skyscanner_url)}' target='_blank' style='text-decoration:none;background:#059669;color:white;padding:4px 8px;border-radius:20px;font-size:13px'>Skyscanner</a>
                    <a href='{escape(gmaps_url)}' target='_blank' style='text-decoration:none;background:#111827;color:white;padding:4px 8px;border-radius:20px;font-size:13px'>Google Maps</a>
                    <button onclick="{copy_js}" style='background:#6b7280;color:white;padding:4px 8px;border-radius:20px;border:none;cursor:pointer;font-size:13px'>Copy IATA</button>
                </div>
            </div>
        """

        folium.Marker(
            [ap["lat"], ap["lon"]],
            tooltip=f"{safe_js(ap['Name'])} ({safe_js(ap['IATA'])})",            
            popup=folium.Popup(popup_html, max_width=360),
            icon=folium.Icon(color="red", icon="plane", prefix="fa"),
        ).add_to(cluster)

        folium.PolyLine(
            [(TLV["lat"], TLV["lon"]), (ap["lat"], ap["lon"])],
            color="#9CA3AF", weight=0.8, opacity=0.9, dash_array="8,10"
        ).add_to(m)

        bounds.append([ap["lat"], ap["lon"]])

    # ---- Fit map to bounds ----
    if len(bounds) > 1:
        try:
            m.fit_bounds(bounds, padding=(30, 30))
        except Exception:
            pass

    logger.info(f"GET /map country={country} query='{query}' rows={len(airports)}")

    bounds_js = json.dumps(bounds)
    html = m.get_root().render()

    fix_script = f"""
    <script>
    window.addEventListener("message", function(e) {{
      if (e.data === "modal-shown") {{
        for (var key in window) {{
          if (key.startsWith("map_")) {{
            var map = window[key];
            setTimeout(function() {{
              map.invalidateSize();
              if (Array.isArray({bounds_js}) && {bounds_js}.length > 1) {{
                map.fitBounds({bounds_js}, {{padding:[30,30]}});
              }}
            }}, 200);
          }}
        }}
      }}
    }});
    </script>
    """

    html = html.replace("</body>", fix_script + "</body>")
    return HTMLResponse(content=html, media_type="text/html; charset=utf-8")
   

# ───────────────────────────────────────────────
# Startup handler
# ───────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    global scheduler, AIRLINE_WEBSITES, DATASET_DF, DATASET_DATE

    # 1) Load airline websites
    AIRLINE_WEBSITES = load_airline_websites()
    logger.info(f"Loaded {len(AIRLINE_WEBSITES)} airline websites")

    # 2) Load dataset from disk if exists
    if ISRAEL_FLIGHTS_FILE.exists():
        df, d = _read_dataset_file()
        DATASET_DF, DATASET_DATE = df, d or ""
        logger.info(f"Startup: dataset loaded (date={DATASET_DATE}, rows={len(DATASET_DF)})")
    else:
        logger.info("Startup: no dataset file on disk")

    # 3) If stale, refresh flights immediately
    if israel_flights_is_stale(max_age_hours=24):
        logger.info("Startup: gov.il flights stale → refreshing now")
        fetch_israel_flights()

    # 4) If stale, refresh travel warnings immediately
    if travel_warnings_is_stale(max_age_hours=24):
        logger.info("Startup: travel warnings stale → refreshing now")
        fetch_travel_warnings()

    # 5) Start scheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(fetch_israel_flights, "interval", hours=24,
                      id="govil_refresh", replace_existing=True)
    scheduler.add_job(fetch_travel_warnings, "interval", hours=24,
                      id="warnings_refresh", replace_existing=True)
    scheduler.start()
    logger.info("Scheduler started: gov.il refresh every 24h (flights + warnings)")


@app.post("/api/israel-flights/refresh")
async def refresh_israel_flights():
    """Fetch new flights from gov.il and update cache."""
    try:
        result = fetch_israel_flights()
        if not result:
            raise HTTPException(
                status_code=502,
                detail="gov.il API did not return data"
            )
        return {
            "status": "ok",
            "updated": result.get("updated"),
            "count": result.get("count", 0)
        }
    except HTTPException:
        raise  # re-raise cleanly
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/israel-flights", response_class=JSONResponse)
async def get_israel_flights():
    """Return cached gov.il flights."""
    if not ISRAEL_FLIGHTS_FILE.exists():
        raise HTTPException(status_code=404, detail="No cached flights available")
    try:
        with open(ISRAEL_FLIGHTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Cached flights file is corrupted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.on_event("shutdown")
async def shutdown_event():
    global scheduler
    if scheduler:
        try:
            scheduler.shutdown()
            logger.info("Scheduler stopped")
        except Exception as e:
            logger.error(f"Error shutting down scheduler: {e}")
    
@app.get("/about", response_class=HTMLResponse)
async def about(request: Request, lang: str = Depends(get_lang)):
    return TEMPLATES.TemplateResponse("about.html", {
        "request": request,
        "lang": lang,
        "now": datetime.now()
    })

@app.get("/privacy", response_class=HTMLResponse)
async def privacy(request: Request, lang: str = Depends(get_lang)):
    return TEMPLATES.TemplateResponse("privacy.html", {
        "request": request,
        "lang": lang,
        "now": datetime.now()
    })

@app.get("/contact", response_class=HTMLResponse)
async def contact(request: Request, lang: str = Depends(get_lang)):
    return TEMPLATES.TemplateResponse("contact.html", {
        "request": request,
        "lang": lang,
        "now": datetime.now()
    })

@app.get("/ads.txt", include_in_schema=False)
async def ads_txt():
    file_path = Path(__file__).parent / "ads.txt"
    return FileResponse(file_path, media_type="text/plain")
    
    
@app.get("/robots.txt", include_in_schema=False)
async def robots_txt():
    file_path = Path(__file__).parent / "robots.txt"
    return FileResponse(file_path, media_type="text/plain")
    
    
class Url:
    def __init__(self, loc: str, lastmod: date, changefreq: str, priority: float):
        self.loc = loc
        self.lastmod = lastmod.isoformat()
        self.changefreq = changefreq
        self.priority = priority

# Helper to generate XML
def build_sitemap(urls: List[Url]) -> str:
    sitemap = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    ]

    for url in urls:
        sitemap.append("  <url>")
        sitemap.append(f"    <loc>{url.loc}</loc>")
        sitemap.append(f"    <lastmod>{url.lastmod}</lastmod>")
        sitemap.append(f"    <changefreq>{url.changefreq}</changefreq>")
        sitemap.append(f"    <priority>{url.priority:.1f}</priority>")
        sitemap.append("  </url>")

    sitemap.append("</urlset>")
    return "\n".join(sitemap)

# Route to serve and write sitemap.xml
@app.get("/sitemap.xml", response_class=Response, include_in_schema=False)
def sitemap():
    base = "https://fly-tlv.com"
    today = date.today()

    urls = [
        Url(f"{base}/", today, "daily", 1.0),
        Url(f"{base}/about", today, "yearly", 0.6),
        Url(f"{base}/privacy", today, "yearly", 0.5),
        Url(f"{base}/contact", today, "yearly", 0.5),
        Url(f"{base}/map", today, "weekly", 0.7),
        Url(f"{base}/travel-warnings", today, "weekly", 0.7),
        Url(f"{base}/chat", today, "weekly", 0.8),

        Url(f"{base}/?lang=he", today, "daily", 1.0),
        Url(f"{base}/about?lang=he", today, "yearly", 0.6),
        Url(f"{base}/privacy?lang=he", today, "yearly", 0.5),
        Url(f"{base}/contact?lang=he", today, "yearly", 0.5),
        Url(f"{base}/map?lang=he", today, "weekly", 0.7),
        Url(f"{base}/travel-warnings?lang=he", today, "weekly", 0.7),
        Url(f"{base}/chat?lang=he", today, "weekly", 0.8),     
    ]
    xml = build_sitemap(urls)

    out_path = Path("static/sitemap.xml")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(xml, encoding="utf-8")

    return Response(content=xml, media_type="application/xml")
    
def generate_destination_questions(n: int = 10) -> list[str]:
    questions_en = [
        "Which airlines fly to New York?",
        "Show me all airports in Germany.",
        "What destinations are available in Italy?",
        "Which airline flies to Paris?",
        "Show all destinations in Cyprus."
    ]

    questions_he = [
        "איזה חברות תעופה טסות לניו יורק?",
        "הצג את כל שדות התעופה בגרמניה.",
        "אילו יעדים זמינים באיטליה?",
        "לאילו ערים ביוון אפשר לטוס?",
        "איזו חברת תעופה טסה לפריז?",
    ]

    # Mix both English + Hebrew, limit to n
    combined = questions_en + questions_he
    return combined[:n]


@app.post("/api/chat", response_class=JSONResponse)
async def chat_flight_ai(
    query: ChatQuery = Body(...),
    max_rows: int = Query(default=150, le=300),
    lang: str = Query(default="en")
):
    """
    Use Gemini to answer natural language questions about DATASET_DF.
    """
    question = query.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    if DATASET_DF.empty:
        raise HTTPException(status_code=503, detail="Flight dataset not loaded.")

    # Step 1: Format in-memory dataset into context
    context_rows = []
    for row in DATASET_DF.to_dict(orient="records")[:max_rows]:
        iata = row.get("IATA", "—")
        city = row.get("City", "—")
        country = row.get("Country", "—")
        airlines = ", ".join(sorted(set(row.get("Airlines", [])))) or "—"
        context_rows.append(f"{iata}, {city}, {country}, Airlines: {airlines}")
    context = "\n".join(context_rows)

    # Step 2: Build prompt
    prompt = f"""
You are an aviation expert helping users explore destinations from Ben Gurion Airport (TLV).

Here is structured destination data:

{context}

Now, based on the data above, answer this user question:

"{question}"

🚨 OUTPUT RULES — FOLLOW EXACTLY 🚨
1. Start with a bold section heading: **✈️ Flights to [Country/Region/City]**
2. Use a bullet list for airports.
   - Each airport must be formatted as: **City (IATA, Country)**
3. Under each airport, indent the airlines as sub-bullets.
   - One airline = one bullet.
   - No inline commas. No single-line lists.
4. Use only Markdown bullet lists ("- "). Never use paragraphs, tables, or bold for airlines.
5. Do not add extra commentary, explanations, or filler.
6. If no results exist: return exactly → `I couldn't find it in the data.`

✅ Correct example:
---
**✈️ Flights to United States**
- Newark (EWR, United States)  
  - Delta Airlines  
  - El Al Israel Airlines  
  - Jetblue Airways Corporation  
  - United Airlines  

- New York (JFK, United States)  
  - Aero Mexico  
  - Aerolineas Argentinas S.a.  
  - Arkia Israeli Airlines  
  - Delta Airlines  
  - Eastern Air Lines Inc.  
  - El Al Israel Airlines  
  - Jetblue Airways Corporation  
  - Virgin Atlantic Airways  
---
"""



    logger.debug("Gemini prompt built successfully (length=%d chars)", len(prompt))

    # Step 3: Ask Gemini
    try:
        result = await chat_model.generate_content_async(prompt)
        answer = getattr(result, "text", None) or "Currently i have no answer"
    except Exception as e:
        logger.error("Gemini API error: %s", str(e))
        raise HTTPException(status_code=500, detail="Gemini API error")

    # Step 4: Suggested questions from JSON schema
    field_names = [
        "airline", "iata", "airport", "city", "country",
        "scheduled", "actual", "direction", "status"
    ]

    if lang == "he":
        base = "הצג טיסות לפי"
        suggestions = [f"{base} {f}" for f in field_names]
    else:
        base = "Show me flights by"
        suggestions = [f"{base} {f}" for f in field_names]

    # Pick 3 random suggestions
    suggestions = random.sample(suggestions, k=3)

    # Step 5: Return answer + suggestions
    return {"answer": answer, "suggestions": suggestions}

        
        
@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    # Optional: List models that support `generateContent` (for debugging)
    #for m in genai.list_models():
    #    if "generateContent" in m.supported_generation_methods:
    #        print(m.name)

    # Render the chat UI
    return TEMPLATES.TemplateResponse("chat.html", {"request": request})
    
@app.get("/api/chat/suggestions", response_class=JSONResponse)
async def chat_suggestions():
    return {"questions": generate_destination_questions()}

# === API Routes ===
@app.get("/api/travel-warnings", response_class=JSONResponse)
async def api_travel_warnings():
    """Return cached travel warnings."""
    if not TRAVEL_WARNINGS_FILE.exists():
        raise HTTPException(status_code=404, detail="No cached travel warnings available")
    try:
        with open(TRAVEL_WARNINGS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Cached warnings file is corrupted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
@app.get("/travel-warnings", response_class=HTMLResponse)
async def travel_warnings_page(request: Request, lang: str = Depends(get_lang)):
    """Render travel warnings page with DataTable"""
    if not TRAVEL_WARNINGS_FILE.exists():
        return TEMPLATES.TemplateResponse("travel_warnings.html", {
            "request": request,
            "lang": lang,
            "warnings": [],
            "last_update": None,
            "continents": [],
            "countries": [],
            "levels": []
        })

    try:
        with open(TRAVEL_WARNINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        warnings = data.get("warnings", [])
        last_update = data.get("updated")

        # הפקת רשימות ייחודיות
        continents = sorted({w.get("continent", "") for w in warnings if w.get("continent")})
        countries = sorted({w.get("country", "") for w in warnings if w.get("country")})
        levels = ["גבוה", "בינוני", "נמוך", "לא ידוע"]

    except Exception as e:
        logger.error(f"Failed to load travel warnings file: {e}")
        warnings, last_update, continents, countries, levels = [], None, [], [], []

    return TEMPLATES.TemplateResponse("travel_warnings.html", {
        "request": request,
        "lang": lang,
        "warnings": warnings,
        "last_update": last_update,
        "continents": continents,
        "countries": countries,
        "levels": levels
    })
