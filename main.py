# === Standard Library ===
import os
import sys
import json
from html import escape
from pathlib import Path
from datetime import datetime, date
from typing import Any, Dict, List
from math import radians, sin, cos, sqrt, atan2
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
from airportsdata import load
from folium.plugins import MarkerCluster
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse,RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from logging_setup import setup_logging, get_app_logger,setup_feedback_logger
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi import status
import secrets
from starlette.exceptions import HTTPException as StarletteHTTPException
from collections import defaultdict
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
setup_logging(log_level="INFO", log_dir="logs", log_file_name="app.log")
feedback_logger = setup_feedback_logger()
logger = get_app_logger("flights_explorer")

logger.info("Server starting…")

# Set up Gemini API

genai.configure(api_key="AIzaSyBxOJwavtKVB9gJvA2OoAsKw90GogBNdZs")
chat_model = genai.GenerativeModel("gemini-2.5-flash")
security = HTTPBasic()

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
app = FastAPI(
    title="Flights Explorer (FastAPI)",
    docs_url=None,        # disable default /docs
    redoc_url=None,       # disable default /redoc
    openapi_url=None      # disable default /openapi.json
)
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
AIRPORTS_DB: dict = {}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_flight_time(dist_km):
    total_hours = dist_km / 800
    hours = int(total_hours)
    minutes = round((total_hours - hours) * 60)

    if minutes == 60:
        hours += 1
        minutes = 0

    return f"{hours}h {minutes}m"


def load_travel_warnings_df() -> pd.DataFrame:
    """Load travel warnings JSON from CACHE_DIR into a DataFrame."""
    if not TRAVEL_WARNINGS_FILE.exists():
        logger.warning(f"⚠️ Travel warnings file not found: {TRAVEL_WARNINGS_FILE}")
        return pd.DataFrame()
    try:
        with open(TRAVEL_WARNINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        warnings = data.get("warnings", [])
        df = pd.DataFrame(warnings)
        # Keep last_update in metadata
        df.attrs["last_update"] = data.get("updated")
        logger.info(f"✅ Loaded {len(df)} travel warnings from cache (updated {df.attrs['last_update']})")
        return df

    except Exception as e:
        logger.error(f"❌ Failed to load travel warnings: {e}")
        return pd.DataFrame()

def verify_docs_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = os.getenv("DOCS_USER", "admin")
    correct_password = os.getenv("DOCS_PASS", "secret123")
    is_user = secrets.compare_digest(credentials.username, correct_username)
    is_pass = secrets.compare_digest(credentials.password, correct_password)
    if not (is_user and is_pass):
        logger.warning(f"Unauthorized docs access attempt from {credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
    
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

def get_lang(request: Request) -> str:
    return "he" if request.query_params.get("lang") == "he" else "en"
    
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



def get_dataset_date() -> str | None:
    if not ISRAEL_FLIGHTS_FILE.exists():
        return None
    try:
        with open(ISRAEL_FLIGHTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        updated = data.get("updated")
        if updated:
            iso_date = updated.split("T")[0]
            try:
                return datetime.strptime(iso_date, "%Y-%m-%d").strftime("%d-%m-%Y")
            except Exception:
                logger.warning(f"Could not parse iso_date='{iso_date}'")
                return iso_date
    except Exception as e:
        logger.error(f"Failed to read dataset date: {e}")
    return None

    
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return round(R * 2 * atan2(sqrt(a), sqrt(1 - a)), 1)


def load_airline_names() -> Dict[str, str]:
    cols = ["AirlineID","Name","Alias","IATA","ICAO","Callsign","Country","Active"]
    try:
        df = pd.read_csv(BASE_DIR / "data" / "airlines.dat", names=cols, header=None, encoding="utf-8")
        return df.set_index("IATA")["Name"].dropna().to_dict()
    except Exception as e:
        logger.warning(f"Airline names not loaded: {e}")
        return {}

AIRLINE_NAMES = load_airline_names()

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

def _read_flights_file() -> tuple[pd.DataFrame, str | None]:
    """
    Extract raw 'flights' records from israel_flights.json without aggregation.
    Returns a flat flight events DataFrame and optional ISO date string.
    """
    try:
        import json
        with open(ISRAEL_FLIGHTS_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)

        flights = meta.get("flights", [])
        if not flights:
            logger.warning("No 'flights' found in israel_flights.json")
            return pd.DataFrame(), None

        df = pd.DataFrame(flights)

        # Extract update date if present
        updated = meta.get("updated")
        file_date = None
        if updated:
            try:
                file_date = updated.split("T")[0]
            except Exception:
                pass

        return df, file_date

    except Exception as e:
        logger.error(f"Failed to read flights dataset: {e}", exc_info=True)

    return pd.DataFrame(columns=[
        "airline", "iata", "airport", "city", "country",
        "scheduled", "actual", "direction", "status"
    ]), None


def _read_dataset_file() -> tuple[pd.DataFrame, str | None]:
    """
    Read israel_flights.json and convert flights → unique airports DataFrame.
    Uses global AIRPORTS_DB (loaded once on startup).
    """
    global AIRPORTS_DB

    if not ISRAEL_FLIGHTS_FILE.exists():
        return pd.DataFrame(columns=["IATA", "Name", "City", "Country", "lat", "lon", "Airlines"]), None

    try:
        with open(ISRAEL_FLIGHTS_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)

        flights = meta.get("flights", [])

        grouped: dict[str, dict] = {}
        for rec in flights:
            iata = rec.get("iata")
            if not iata:
                continue
            entry = grouped.setdefault(iata, {
                "IATA": iata,
                "Name": rec.get("airport") or "—",
                "City": rec.get("city") or "—",
                "Country": rec.get("country") or "—",
                "Airlines": set(),
            })
            airline = rec.get("airline")
            if airline:
                entry["Airlines"].add(airline)

        rows = []
        for iata, info in grouped.items():
            coords = AIRPORTS_DB.get(iata, {}) if AIRPORTS_DB else {}
            lat, lon = coords.get("lat"), coords.get("lon")
            dist_km = haversine_km(TLV["lat"], TLV["lon"], lat, lon) if lat and lon else None
            flight_hr = get_flight_time(dist_km) if dist_km else "—"
            rows.append({
                **info,
                "lat": lat,
                "lon": lon,
                "Distance_km": dist_km,
                "FlightTime_hr": flight_hr,
                "Airlines": sorted(info["Airlines"]),
            })

        df = pd.DataFrame(rows)

        # Extract updated date
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
        logger.error(f"Failed to read dataset file: {e}", exc_info=True)

    # Fallback
    return pd.DataFrame(columns=["IATA", "Name", "City", "Country", "lat", "lon", "Airlines"]), None

def get_dataset(trigger_refresh: bool = False) -> pd.DataFrame:
    """
    Return the current dataset from memory or disk.
    If trigger_refresh=True, fetch fresh data immediately.
    """
    global DATASET_DF, DATASET_DATE

    # ✅ Use in-memory if available
    if not DATASET_DF.empty:
        return DATASET_DF

    # ✅ Try loading from disk
    df, disk_date = _read_dataset_file()
    DATASET_DF, DATASET_DATE = df, disk_date or ""

    if not trigger_refresh:
        return DATASET_DF

    # ✅ Refresh now if explicitly requested
    logger.info("Dataset refresh requested → fetching gov.il flights")
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

def load_airline_websites() -> dict:
    try:
        with open(AIRLINE_WEBSITES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load airline websites: {e}")
        return {}
        
def format_time(dt_string):
    """Return (short, full, raw_iso) for datetime strings."""
    if not dt_string or dt_string.strip() in {"—", ""}:
        return "—", "—", ""

    dt_string = dt_string.strip()

    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(dt_string, fmt)
            formatted_short = dt.strftime("%b %d, %H:%M")
            formatted_full  = dt.strftime("%b %d, %H:%M")
            raw_iso         = dt.strftime("%Y-%m-%dT%H:%M:%S")
            return formatted_short, formatted_full, raw_iso
        except ValueError:
            continue

    return dt_string, dt_string, dt_string


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
    global AIRPORTS_DB, DATASET_DF, AIRLINE_WEBSITES

    # Defensive check
    if DATASET_DF is None or DATASET_DF.empty:
        return TEMPLATES.TemplateResponse("error.html", {
            "request": request,
            "message": "No data available.",
            "lang": lang
        })

    df = DATASET_DF.copy()

    # Filter by country
    if country != "All":
        df = df[df["Country"].str.lower() == country.lower()]

    # Text query (case-insensitive search on several fields)
    if query:
        q = query.lower()
        df = df[
            df["Name"].str.lower().str.contains(q) |
            df["City"].str.lower().str.contains(q) |
            df["Country"].str.lower().str.contains(q) |
            df["Airlines"].str.lower().str.contains(q)
        ]

    # Final airports list
    airports = df.to_dict(orient="records")

    # All unique countries for dropdown
    countries = ["All"] + sorted(DATASET_DF["Country"].dropna().unique().tolist())
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
    global DATASET_DF_FLIGHTS, AIRPORTS_DB

    # Main dataset
    df = get_dataset(trigger_refresh=False).copy()
    airports = df.to_dict(orient="records")
    existing_iatas = {ap["IATA"] for ap in airports}

    # Find additional flights in DATASET_DF_FLIGHTS not in main dataset
    grouped: dict[str, dict] = {}

    for rec in DATASET_DF_FLIGHTS.to_dict(orient="records"):
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

    # Enrich and append new airports
    for iata, meta in grouped.items():
        lat = lon = None
        dist_km = flight_time_hr = "—"
        if iata in AIRPORTS_DB:
            lat = AIRPORTS_DB[iata].get("lat")
            lon = AIRPORTS_DB[iata].get("lon")
            if lat and lon:
                dist_km = round(haversine_km(TLV["lat"], TLV["lon"], lat, lon), 1)
                flight_time_hr = get_flight_time(dist_km)

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

    # Filter by country
    if country and country != "All":
        airports = [
            ap for ap in airports
            if ap.get("Country", "").lower() == country.lower()
        ]

    # Filter by query
    if query:
        q = query.strip().lower()
        airports = [
            ap for ap in airports
            if q in str(ap.get("IATA", "")).lower()
            or q in str(ap.get("Name", "")).lower()
            or q in str(ap.get("City", "")).lower()
            or q in str(ap.get("Country", "")).lower()
        ]
    m = folium.Map(
        location=[35, 28.5],
        zoom_start=5,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
        zoom_control=True
    )
    cluster = MarkerCluster().add_to(m)
    folium.Marker(
        [TLV["lat"], TLV["lon"]],
        tooltip="Tel Aviv (TLV)",
        icon=folium.Icon(color="blue", icon="plane", prefix="fa")
    ).add_to(m)

    bounds = [[TLV["lat"], TLV["lon"]]]

    for ap in airports:
        if not ap.get("lat") or not ap.get("lon"):
            logger.warning(f"Missing coords for {ap['IATA']} ({ap['City']}, {ap['Country']})")
            continue

        km = haversine_km(TLV["lat"], TLV["lon"], ap["lat"], ap["lon"])
        flight_time_hr = get_flight_time(km) if km else "—"
        flights_url    = f"https://www.google.com/travel/flights?q=flights%20from%20TLV%20to%20{ap['IATA']}"
        skyscanner_url = f"https://www.skyscanner.net/transport/flights/tlv/{ap['IATA'].lower()}/"
        gmaps_url      = f"https://maps.google.com/?q={ap['City'].replace(' ','+')},{ap['Country'].replace(' ','+')}"
        copy_js = f"navigator.clipboard && navigator.clipboard.writeText('{safe_js(ap['IATA'])}')"
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
    global scheduler, AIRLINE_WEBSITES, DATASET_DF, DATASET_DATE, DATASET_DF_FLIGHTS, AIRPORTS_DB, TRAVEL_WARNINGS_DF

    logger.info("🚀 Application startup initiated")
    # 0) Load IATA DB once
    try:
        AIRPORTS_DB = load("IATA")
        logger.info(f"Loaded airportsdata IATA DB with {len(AIRPORTS_DB)} records")
    except Exception as e:
        logger.error(f"Failed to load airportsdata: {e}")
        AIRPORTS_DB = {}
        
    # 1) Init scheduler
    scheduler = AsyncIOScheduler()
    logger.debug("AsyncIOScheduler instance created")
    TRAVEL_WARNINGS_DF = load_travel_warnings_df()

    # 2) Load airline websites
    try:
        AIRLINE_WEBSITES = load_airline_websites()
        logger.info(f"Loaded {len(AIRLINE_WEBSITES)} airline websites")
    except Exception as e:
        logger.error(f"Failed to load airline websites: {e}", exc_info=True)
        AIRLINE_WEBSITES = {}

    # 3) Load datasets from disk if exists
    if ISRAEL_FLIGHTS_FILE.exists():
        try:
            # Load grouped (aggregated) dataset
            df, d = _read_dataset_file()
            DATASET_DF, DATASET_DATE = df, d or ""
            logger.info(
                f"Startup: dataset loaded from disk "
                f"(date={DATASET_DATE}, rows={len(DATASET_DF)})"
            )
            DATASET_DF_FLIGHTS, FLIGHTS_DATA_DATE = _read_flights_file()
            logger.info(f"Startup: loaded {len(DATASET_DF_FLIGHTS)} flight event records")

        except Exception as e:
            logger.error(f"Error reading dataset file: {e}", exc_info=True)
            DATASET_DF_FLIGHTS = pd.DataFrame()
    else:
        logger.warning("Startup: no dataset file on disk")
        DATASET_DF_FLIGHTS = pd.DataFrame()
    
    # 4) Schedule jobs (every 8h, run immediately at startup)
    try:
        scheduler.add_job(
            fetch_israel_flights,
            "interval",
            hours=8,
            id="govil_refresh",
            replace_existing=True,
            next_run_time=datetime.now()
        )
        logger.info("Scheduled job: gov.il flights refresh every 8h")

        scheduler.add_job(
            fetch_travel_warnings,
            "interval",
            hours=8,
            id="warnings_refresh",
            replace_existing=True,
            next_run_time=datetime.now()
        )
        logger.info("Scheduled job: travel warnings refresh every 8h")

        scheduler.start()
        logger.info("✅ Scheduler started successfully")
    except Exception as e:
        logger.critical(f"Failed to start scheduler: {e}", exc_info=True)

    logger.info("🎯 Application startup completed")



@app.on_event("shutdown")
async def shutdown_event():
    global scheduler
    logger.info("🛑 Application shutdown initiated")

    if scheduler:
        try:
            scheduler.shutdown(wait=False)
            logger.info("✅ Scheduler stopped successfully")
        except Exception as e:
            logger.error("Error shutting down scheduler", exc_info=True)
    else:
        logger.warning("No scheduler instance to shut down")

    logger.info("👋 Application shutdown completed")

    
@app.get("/about", response_class=HTMLResponse)
async def about(request: Request, lang: str = Depends(get_lang)):
    logger.info(f"GET /about | lang={lang} | client={request.client.host}")
    return TEMPLATES.TemplateResponse("about.html", {
        "request": request,
        "lang": lang,
        "now": datetime.now()
    })

@app.get("/privacy", response_class=HTMLResponse)
async def privacy(request: Request, lang: str = Depends(get_lang)):
    logger.info(f"GET /privacy | lang={lang} | client={request.client.host}")
    return TEMPLATES.TemplateResponse("privacy.html", {
        "request": request,
        "lang": lang,
        "now": datetime.now()
    })


@app.get("/contact", response_class=HTMLResponse)
async def contact(request: Request, lang: str = Depends(get_lang)):
    logger.info(f"GET /contact | lang={lang} | client={request.client.host}")
    return TEMPLATES.TemplateResponse("contact.html", {
        "request": request,
        "lang": lang,
        "now": datetime.now()
    })


@app.get("/ads.txt", include_in_schema=False)
async def ads_txt(request: Request):
    logger.info(f"GET /ads.txt | client={request.client.host}")
    file_path = Path(__file__).parent / "ads.txt"
    return FileResponse(file_path, media_type="text/plain")
   
    
@app.get("/robots.txt", include_in_schema=False)
async def robots_txt(request: Request):
    file_path = Path(__file__).parent / "robots.txt"
    if file_path.exists():
        logger.info(f"GET /robots.txt | client={request.client.host}")
        return FileResponse(file_path, media_type="text/plain")
    else:
        logger.error(f"robots.txt not found! | client={request.client.host}")
        raise HTTPException(status_code=404, detail="robots.txt not found")

    
    
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

@app.get("/sitemap.xml", response_class=Response, include_in_schema=False)
def sitemap():
    base = "https://fly-tlv.com"
    today = date.today()

    # Static base URLs
    urls = [
        Url(f"{base}/", today, "daily", 1.0),
        Url(f"{base}/stats", today, "daily", 1.0),
        Url(f"{base}/about", today, "yearly", 0.6),
        Url(f"{base}/direct-vs-nonstop", today, "yearly", 0.6),
        Url(f"{base}/privacy", today, "yearly", 0.5),
        Url(f"{base}/glossary", today, "yearly", 0.5),        
        Url(f"{base}/contact", today, "yearly", 0.5),
        Url(f"{base}/accessibility", today, "yearly", 0.5),
        Url(f"{base}/map", today, "weekly", 0.7),
        Url(f"{base}/flights", today, "weekly", 0.7),
        Url(f"{base}/travel-warnings", today, "weekly", 0.7),
        Url(f"{base}/chat", today, "weekly", 0.8),

        # Hebrew versions
        Url(f"{base}/?lang=he", today, "daily", 1.0),
        Url(f"{base}/stats?lang=he", today, "daily", 1.0),
        Url(f"{base}/about?lang=he", today, "yearly", 0.6),
        Url(f"{base}/direct-vs-nonstop?lang=he", today, "yearly", 0.6),
        Url(f"{base}/accessibility?lang=he", today, "yearly", 0.5),
        Url(f"{base}/privacy?lang=he", today, "yearly", 0.5),
        Url(f"{base}/contact?lang=he", today, "yearly", 0.5),
        Url(f"{base}/map?lang=he", today, "weekly", 0.7),
        Url(f"{base}/flights?lang=he", today, "weekly", 0.7),
        Url(f"{base}/travel-warnings?lang=he", today, "weekly", 0.7),
        Url(f"{base}/chat?lang=he", today, "weekly", 0.8),
        Url(f"{base}/glossary?lang=he", today, "yearly", 0.5),
    ]
    try:
        for iata in DATASET_DF["IATA"].dropna().unique():
            iata = str(iata).strip()
            if iata:
                urls.append(Url(f"{base}/destinations/{iata}", today, "weekly", 0.7))
                urls.append(Url(f"{base}/destinations/{iata}?lang=he", today, "weekly", 0.7))
    except Exception as e:
        logger.warning(f"Failed to load dynamic IATA links: {e}")

    # ✅ Build and save sitemap XML
    xml = build_sitemap(urls)
    out_path = STATIC_DIR / "sitemap.xml"

    try:
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(xml, encoding="utf-8")
        logger.info(f"Sitemap written to {out_path} with {len(urls)} URLs")
    except Exception as e:
        logger.error(f"Failed to write sitemap.xml: {e}")

    return Response(content=xml, media_type="application/xml")



def generate_questions_from_data(destinations: list[dict], n: int = 20) -> list[str]:
    cities = set()
    countries = set()
    airlines = set()

    for dest in destinations:
        city = dest.get("City") or dest.get("city")
        country = dest.get("Country") or dest.get("country")
        airline_list = dest.get("Airlines") or dest.get("airlines", [])

        if city and country:
            cities.add((city.strip(), country.strip()))
        if country:
            countries.add(country.strip())
        for airline in airline_list:
            if airline:
                airlines.add(airline.strip())

    cities = list(cities)
    countries = list(countries)
    airlines = list(airlines)

    questions = []

    # Build ENGLISH questions
    for country in random.sample(countries, min(5, len(countries))):
        questions.append(f"What cities in {country} can I fly to?")
        questions.append(f"Which airlines fly to {country}?")
    for city, country in random.sample(cities, min(5, len(cities))):
        questions.append(f"Which airlines fly to {city}?")
        questions.append(f"What country is {city} located in?")
    for airline in random.sample(airlines, min(5, len(airlines))):
        questions.append(f"Where does {airline} fly?")
        questions.append(f"What destinations are served by {airline}?")

    # Build HEBREW questions
    for country in random.sample(countries, min(4, len(countries))):
        questions.append(f"אילו ערים יש טיסות ל-{country}?")
        questions.append(f"אילו חברות טסות ל-{country}?")
    for city, country in random.sample(cities, min(4, len(cities))):
        questions.append(f"אילו חברות טסות ל-{city}?")
        questions.append(f"באיזו מדינה נמצאת {city}?")
    for airline in random.sample(airlines, min(4, len(airlines))):
        questions.append(f"לאן טסה חברת {airline}?")
        questions.append(f"אילו ערים משרתת חברת {airline}?")

    random.shuffle(questions)
    return questions[:n]





@app.post("/api/chat", response_class=JSONResponse)
async def chat_flight_ai(
    query: ChatQuery = Body(...),
    max_rows: int = Query(default=150, le=300),
    lang: str = Query(default="en")
):
    question = query.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    if DATASET_DF.empty:
        raise HTTPException(status_code=503, detail="Flight dataset not loaded.")

    context_rows = []
    for row in DATASET_DF.to_dict(orient="records")[:max_rows]:
        iata = row.get("IATA", "—")
        city = row.get("City", "—")
        country = row.get("Country", "—")
        airlines = ", ".join(sorted(set(row.get("Airlines", [])))) or "—"
        context_rows.append(f"{iata}, {city}, {country}, Airlines: {airlines}")
    context = "\n".join(context_rows)

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
    try:
        result = await chat_model.generate_content_async(prompt)
        answer = getattr(result, "text", None) or "Currently i have no answer"
    except Exception as e:
        logger.error("Gemini API error: %s", str(e))
        raise HTTPException(status_code=500, detail="Gemini API error")

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

    suggestions = random.sample(suggestions, k=3)

    return {"answer": answer, "suggestions": suggestions}

        
        
@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, lang: str = Depends(get_lang)):
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"GET /chat from {client_host} (lang={lang})")

    return TEMPLATES.TemplateResponse("chat.html", {
        "request": request,
        "lang": lang,
        "now": datetime.now()
    })

    
@app.get("/api/chat/suggestions", response_class=JSONResponse)
async def chat_suggestions(n: int = Query(default=10, le=20)):
    """Return up to n suggested chat questions (English + Hebrew)."""
    if DATASET_DF.empty:
        raise HTTPException(status_code=503, detail="Destination data not loaded.")       

    destinations = DATASET_DF.to_dict(orient="records")
    suggestions = generate_questions_from_data(destinations, n)
    logger.info(f"GET /api/chat/suggestions → {len(suggestions)} suggestions")
    return {"questions": suggestions}
    
@app.get("/travel-warnings", response_class=HTMLResponse)
async def travel_warnings_page(request: Request, lang: str = Depends(get_lang)):
    global TRAVEL_WARNINGS_DF

    client_host = request.client.host if request.client else "unknown"

    if TRAVEL_WARNINGS_DF.empty:
        logger.warning(f"GET /travel-warnings from {client_host} (lang={lang}) → no cached data")
        return TEMPLATES.TemplateResponse("travel_warnings.html", {
            "request": request,
            "lang": lang,
            "warnings": [],
            "last_update": None,
            "continents": [],
            "countries": [],
            "levels": []
        })

    warnings = TRAVEL_WARNINGS_DF.to_dict(orient="records")
    last_update = TRAVEL_WARNINGS_DF.attrs.get("last_update")

    continents = sorted(TRAVEL_WARNINGS_DF["continent"].dropna().unique())
    countries  = sorted(TRAVEL_WARNINGS_DF["country"].dropna().unique())
    levels     = ["גבוה", "בינוני", "נמוך", "לא ידוע"]

    logger.info(f"GET /travel-warnings from {client_host} (lang={lang}) → {len(warnings)} warnings")

    return TEMPLATES.TemplateResponse("travel_warnings.html", {
        "request": request,
        "lang": lang,
        "warnings": warnings,
        "last_update": last_update,
        "continents": continents,
        "countries": countries,
        "levels": levels
    })


@app.get("/openapi.json", include_in_schema=False)
async def custom_openapi(username: str = Depends(verify_docs_credentials)):
    logger.info(f"GET /openapi.json → served for user={username}")
    return get_openapi(title=app.title,version="1.0.0",routes=app.routes,)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui(username: str = Depends(verify_docs_credentials)):
    logger.info(f"GET /docs → Swagger UI served for user={username}")
    return get_swagger_ui_html(openapi_url="/openapi.json",title="API Docs")


@app.get("/redoc", include_in_schema=False)
async def custom_redoc(username: str = Depends(verify_docs_credentials)):
    logger.info(f"GET /redoc → ReDoc UI served for user={username}")
    return get_redoc_html(openapi_url="/openapi.json",title="API ReDoc")

@app.get("/flights", response_class=HTMLResponse)
async def flights_view(request: Request):
    global DATASET_DF_FLIGHTS

    if DATASET_DF_FLIGHTS is None or DATASET_DF_FLIGHTS.empty:
        return TEMPLATES.TemplateResponse("error.html", {
            "request": request,
            "message": "No live flight data available.",
            "lang": request.query_params.get("lang", "en")
        })

    flights = DATASET_DF_FLIGHTS.to_dict(orient="records")

    for f in flights:
        s_short, s_full, s_iso = format_time(f.get("scheduled", ""))
        a_short, a_full, a_iso = format_time(f.get("actual", ""))

        f["scheduled"] = s_short
        f["scheduled_full"] = s_full
        f["scheduled_iso"] = s_iso

        f["actual"] = a_short
        f["actual_full"] = a_full
        f["actual_iso"] = a_iso

    # ✅ Extract filters
    countries = sorted({
        f.get("country", "").strip()
        for f in flights if f.get("country")
    })

    actual_times = sorted({
        f["actual_iso"].split("T")[1][:5]
        for f in flights
        if f.get("actual_iso") and "T" in f["actual_iso"]
    })

    # ✅ Safely build actual_dates as list of (value, label)
    actual_dates_set = set()
    for f in flights:
        iso = f.get("actual_iso")
        if iso and "T" in iso:
            try:
                date_part = iso.split("T")[0]  # e.g. "2025-09-29"
                label = datetime.strptime(date_part, "%Y-%m-%d").strftime("%b %d")  # "Sep 29"
                actual_dates_set.add((date_part, label))
            except ValueError:
                continue

    actual_dates = sorted(actual_dates_set)

    return TEMPLATES.TemplateResponse("flights.html", {
        "request": request,
        "flights": flights,
        "countries": countries,
        "actual_dates": actual_dates,
        "actual_times": actual_times,
        "last_update": get_dataset_date(),
        "lang": request.query_params.get("lang", "en"),
        "now": datetime.now(),
        "AIRLINE_WEBSITES": AIRLINE_WEBSITES
    })

@app.get("/glossary", response_class=HTMLResponse)
async def glossary_view(request: Request):
    lang = request.query_params.get("lang", "en")

    try:
        return TEMPLATES.TemplateResponse("aviation_glossary.html", {
            "request": request,
            "lang": lang
        })
    except Exception as e:
        return TEMPLATES.TemplateResponse("error.html", {
            "request": request,
            "message": (
                "Glossary page could not be loaded."
                if lang != "he"
                else "לא ניתן לטעון את עמוד המונחים."
            ),
            "lang": lang
        })

@app.get("/destinations", include_in_schema=False)
async def redirect_to_home(request: Request):
    global DATASET_DF

    if DATASET_DF is None or DATASET_DF.empty:
        logger.warning("DATASET_DF is not loaded. Redirecting anyway to avoid SEO issues.")

    lang = request.query_params.get("lang")
    url = f"/?lang={lang}" if lang else "/"
    return RedirectResponse(url=url, status_code=301)

    
@app.get("/destinations/{iata}", response_class=HTMLResponse)
async def destination_detail(request: Request, iata: str):
    global DATASET_DF

    if DATASET_DF is None or DATASET_DF.empty:
        return TEMPLATES.TemplateResponse("error.html", {
            "request": request,
            "message": "No destination data available.",
            "lang": request.query_params.get("lang", "en")
        })

    iata = iata.upper()
    destination = DATASET_DF.loc[DATASET_DF["IATA"] == iata]

    if destination.empty:
        return TEMPLATES.TemplateResponse("error.html", {
            "request": request,
            "message": f"Destination {iata} not found.",
            "lang": request.query_params.get("lang", "en")
        })

    dest = destination.iloc[0].to_dict()

    return TEMPLATES.TemplateResponse("destination.html", {
        "request": request,
        "destination": dest,
        "lang": request.query_params.get("lang", "en")
    })

# Handle all HTTP errors (404, 403, etc.)
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return TEMPLATES.TemplateResponse("error.html", {
        "request": request,
        "status_code": exc.status_code,
        "message": exc.detail,
        "lang": request.query_params.get("lang", "en")
    }, status_code=exc.status_code)


# Handle validation errors (422 Unprocessable Entity, bad query/body params)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return TEMPLATES.TemplateResponse("error.html", {
        "request": request,
        "status_code": 422,
        "message": "Invalid request. Please check your input." if request.query_params.get("lang", "en") == "en" else "בקשה לא חוקית. בדוק את הנתונים שלך.",
        "lang": request.query_params.get("lang", "en")
    }, status_code=422)


# Handle all unexpected exceptions (500)
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    import traceback
    # Optional: log full stacktrace
    logger.error(f"Unhandled error: {exc}", exc_info=True)

    return TEMPLATES.TemplateResponse("error.html", {
        "request": request,
        "status_code": 500,
        "message": "Internal Server Error" if request.query_params.get("lang", "en") == "en" else "שגיאת שרת פנימית",
        "lang": request.query_params.get("lang", "en")
    }, status_code=500)

@app.get("/accessibility", response_class=HTMLResponse)
async def accessibility(request: Request, lang: str = "en"):
    return TEMPLATES.TemplateResponse(
        "accessibility.html",
        {
            "request": request,
            "lang": lang
        }
    )
@app.post("/api/chat/feedback")
async def receive_feedback(payload: dict):
    question = payload.get("question")
    score = payload.get("score")
    feedback_logger.info(f"{score} | {question}")
    return {"status": "ok"}
    
@app.get("/sw.js")
async def remove_sw():
    # Tell the browser "there’s no service worker here"
    js = """
    self.addEventListener("install", () => self.skipWaiting());
    self.addEventListener("activate", (event) => {
        event.waitUntil(
            caches.keys().then(keys => Promise.all(keys.map(k => caches.delete(k))))
        );
        self.registration.unregister();
    });
    """
    return Response(content=js, media_type="application/javascript")
    


@app.get("/stats", response_class=HTMLResponse)
async def flight_stats_view(request: Request, lang: str = Depends(get_lang)):
    global DATASET_DF_FLIGHTS

    if DATASET_DF_FLIGHTS is None or DATASET_DF_FLIGHTS.empty:
        return TEMPLATES.TemplateResponse("error.html", {
            "request": request,
            "message": "No live flight statistics available.",
            "lang": lang
        })

    # Copy & normalize dataframe
    df = DATASET_DF_FLIGHTS.copy()
    df["Direction"] = df["direction"].map({"D": "Departure", "A": "Arrival"})
    df = df.rename(columns={
        "airline": "Airline",
        "city": "City",
        "country": "Country"
    })

    # --- Helper: aggregate safely ---
    def build_stats(sub_df, group_field):
        stats = (
            sub_df.groupby([group_field, "Direction"])
                  .size()
                  .unstack(fill_value=0)
                  .reset_index()
        )
        stats = stats.rename(columns={"Departure": "Departures", "Arrival": "Arrivals"})
        for col in ["Departures", "Arrivals"]:  # ensure both always exist
            if col not in stats:
                stats[col] = 0
        stats["Total"] = stats["Departures"] + stats["Arrivals"]
        return stats.sort_values("Total", ascending=False).head(10).to_dict("records")

    # --- Global Stats ---
    top_countries = build_stats(df, "Country")
    top_cities = build_stats(df, "City")

    # --- Airlines Stats ---
    airlines = sorted(df["Airline"].dropna().unique())
    airlines_data = {"All": {"countries": top_countries, "cities": top_cities}}
    for airline in airlines:
        sub = df[df["Airline"] == airline]
        airlines_data[airline] = {
            "countries": build_stats(sub, "Country"),
            "cities": build_stats(sub, "City"),
        }

    # --- Render ---
    return TEMPLATES.TemplateResponse("stats.html", {
        "request": request,
        "lang": lang,
        "last_update": get_dataset_date(),
        "top_countries": top_countries,
        "top_cities": top_cities,
        "airlines": airlines,
        "airlines_data": airlines_data,
    })

@app.get("/direct-vs-nonstop", response_class=HTMLResponse)
async def direct_vs_nonstop(request: Request, lang: str = Depends(get_lang)):
    logger.info(f"GET /direct-vs-nonstop | lang={lang} | client={request.client.host}")
    return TEMPLATES.TemplateResponse("direct_vs_nonstop.html", {
        "request": request,
        "lang": lang,
        "now": datetime.now()
    })