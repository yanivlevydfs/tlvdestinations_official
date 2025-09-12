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

# === Third-Party Libraries ===
import pandas as pd
import requests
import folium
import airportsdata
from folium.plugins import MarkerCluster
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, Request, Response, Query, HTTPException
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

# Data files
AIRLINE_WEBSITES_FILE = DATA_DIR / "airline_websites.json"
AIRLINE_CACHE_FILE    = CACHE_DIR / "ae_routes.json"       # per-airport airlines
DATASET_FILE          = CACHE_DIR / "tlv_airports_dataset.json"  # full daily dataset
ISRAEL_FLIGHTS_FILE   = CACHE_DIR / "israel_flights.json"

# API
AE_API_KEY: str = "da33f3-b78037"  # keep your key here
AE_BASE = "https://aviation-edge.com/v2/public/routes"

# Constants
TLV = {"IATA": "TLV", "Name": "Ben Gurion Airport", "lat": 32.0068, "lon": 34.8853}
ISRAEL_API = "https://data.gov.il/api/3/action/datastore_search"
RESOURCE_ID = "e83f763b-b7d7-479e-b172-ae981ddc6de5"
DEFAULT_LIMIT = 32000

# App
app = FastAPI(title="Flights Explorer (FastAPI)")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def israel_flights_is_stale(max_age_hours: int = 24) -> bool:
    """Return True if the israel flights cache file is older than max_age_hours or missing."""
    if not ISRAEL_FLIGHTS_FILE.exists():
        return True
    age_hours = (datetime.now() - datetime.fromtimestamp(ISRAEL_FLIGHTS_FILE.stat().st_mtime)).total_seconds() / 3600
    return age_hours > max_age_hours
    
def dataset_is_stale(max_age_hours: int = 24) -> bool:
    """
    Returns True if dataset file is missing or older than `max_age_hours`.
    """
    if not DATASET_FILE.exists():
        logger.info("dataset_is_stale: file does not exist -> STALE")
        return True

    try:
        mtime = DATASET_FILE.stat().st_mtime
        file_time = datetime.fromtimestamp(mtime)
        age_hours = (datetime.now() - file_time).total_seconds() / 3600.0
        logger.info(f"dataset_is_stale: dataset age={age_hours:.2f}h (limit={max_age_hours}h)")
        return age_hours > max_age_hours
    except Exception as e:
        logger.error(f"Failed to check dataset file age: {e}")
        return True
        
def _dataset_file_date() -> str | None:
    """Return the date string inside DATASET_FILE, or None."""
    if DATASET_FILE.exists():
        try:
            meta = json.load(open(DATASET_FILE, "r", encoding="utf-8"))
            return meta.get("date")
        except Exception:
            return None
    return None

def _is_dataset_fresh_today() -> tuple[bool, str | None]:
    """Check if dataset is for today (checks in-memory first, then file)."""
    today = datetime.now().strftime("%Y-%m-%d")
    # in-memory
    if DATASET_DATE == today and not DATASET_DF.empty:
        return True, today
    # on disk
    file_date = _dataset_file_date()
    return (file_date == today), file_date

def get_lang(request: Request) -> str:
    return "he" if request.query_params.get("lang") == "he" else "en"

def get_all_countries() -> list[str]:
    """Return all countries with All, Greece, Cyprus first, rest alphabetically."""
    all_countries = sorted([c.name for c in pycountry.countries])
    
    # Ensure Greece and Cyprus appear only once
    rest = [c for c in all_countries if c not in {"Greece", "Cyprus"}]
    
    return ["All", "Greece", "Cyprus"] + rest
    
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
    
# ───────────────────────────────────────────────
# Async Aviation Edge fetchers (parallel with throttle)
# ───────────────────────────────────────────────

async def _fetch(session: aiohttp.ClientSession, params: dict, retries: int = 1) -> list:
    """Async GET with retries. Returns JSON list or [] on failure."""
    for attempt in range(retries + 1):
        try:
            async with session.get(AE_BASE, params=params, timeout=15) as r:
                if r.status == 200:
                    try:
                        return await r.json()
                    except Exception as e:
                        logger.error(f"[ERROR] JSON decode failed: {e}")
                        return []
                else:
                    text = await r.text()
                    logger.warning(f"[WARN] AE status {r.status} for {params} | {text[:100]}")
        except Exception as e:
            logger.error(f"[ERROR] AE exception for {params}: {e}")
        if attempt < retries:
            logger.info(f"[INFO] Retrying AE call ({attempt+1}/{retries}) for {params}")
            await asyncio.sleep(1.0)
    return []


async def fetch_airlines_one_dest(dest: str, api_key: str, session: aiohttp.ClientSession, sem: asyncio.Semaphore) -> set[str]:
    """Fetch TLV↔dest airlines with semaphore concurrency control."""
    airlines: set[str] = set()
    params_list = [
        {"departureIata": "TLV", "arrivalIata": dest, "key": api_key},
        {"departureIata": dest, "arrivalIata": "TLV", "key": api_key},
    ]

    async def _task(params):
        async with sem:
            data = await _fetch(session, params)
            if isinstance(data, list):
                for it in data:
                    if isinstance(it, dict) and (
                        it.get("departureIata") == "TLV" or it.get("arrivalIata") == "TLV"
                    ):
                        airlines.update(extract_airline_names(it))
            elif isinstance(data, dict) and data.get("error") == "No Record Found":
                logger.info(f"[INFO] AE {params['departureIata']}->{params['arrivalIata']}: no routes found")

    await asyncio.gather(*[_task(p) for p in params_list])
    return airlines


async def fetch_airlines_batch(dest_list: List[str], api_key: str, max_concurrent: int = 5) -> Dict[str, set[str]]:
    """
    Fetch airlines for multiple destinations in parallel.
    - max_concurrent limits in-flight requests (avoid rate limit).
    - Returns dict {IATA: set(airlines)}.
    """
    results: Dict[str, set[str]] = {}
    sem = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_airlines_one_dest(dest, api_key, session, sem) for dest in dest_list]
        results_list = await asyncio.gather(*tasks)

    for dest, airlines in zip(dest_list, results_list):
        results[dest] = airlines
    return results
# ──────────────────────────────────────────────────────────────────────────────
# Progress tracker for refresh
# ──────────────────────────────────────────────────────────────────────────────
PROGRESS: Dict[str, Any] = {"running": False, "total": 0, "done": 0, "date": None}
PROGRESS_LOCK = threading.Lock()

# ──────────────────────────────────────────────────────────────────────────────
# Load airports (GR & CY) once for dataset builds
# ──────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return round(R * 2 * atan2(sqrt(a), sqrt(1 - a)), 1)
    
def load_airports_gr_cy() -> pd.DataFrame:
    iata_db = airportsdata.load("IATA")
    rows = []
    for iata, rec in iata_db.items():
        if rec.get("country") not in {"GR", "CY"}:
            continue
        country = "Greece" if rec.get("country") == "GR" else "Cyprus"
        name, city = rec.get("name"), rec.get("city")
        lat, lon = rec.get("lat"), rec.get("lon")

        dist_km = None
        flight_time_hr = None
        if lat is not None and lon is not None:
            dist_km = round(haversine_km(TLV["lat"], TLV["lon"], lat, lon), 1)
            flight_time_hr = round(dist_km / 800, 2)  # assume 800 km/h

        rows.append({
            "IATA": iata,
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
AIRPORTS = load_airports_gr_cy()

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
# ── Disk cache helpers (per-airport airlines)
def read_airline_cache() -> dict:
    if AIRLINE_CACHE_FILE.exists():
        try:
            with open(AIRLINE_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read airline cache: {e}")
            return {}
    return {}

def write_airline_cache(cache: dict) -> None:
    try:
        with open(AIRLINE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to write airline cache: {e}")

def save_airlines_for(dest: str, airlines: set) -> None:
    cache = read_airline_cache()
    cache[dest] = sorted(list(airlines))
    write_airline_cache(cache)

def load_airlines_for(dest: str) -> set:
    cache = read_airline_cache()
    return set(cache.get(dest, []))
    
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
def extract_airline_names(route: dict) -> List[str]:
    """
    Extract airline names from AE response and normalize.
    Prefer IATA code → canonical name, fallback to raw names.
    Deduplicated case-insensitively and returned in Title Case.
    """

    # 1) Prefer canonical name via IATA code
    code = route.get("airlineIata") or route.get("airlineCode")
    if code:
        code = str(code).upper().strip()
        canonical = AIRLINE_NAMES.get(code)
        if canonical:
            return normalize_airline_list([canonical])

    # 2) Fallback: collect free-text names
    names = []
    for k in ("airlineName", "airline_name", "airline"):
        v = route.get(k)
        if v:
            names.append(str(v).strip())

    # Normalize & dedupe
    return normalize_airline_list(names)

    
    
def fetch_israel_flights() -> dict:
    """Fetch gov.il flights (all countries) and cache to disk."""    
    params = {
        "resource_id": RESOURCE_ID,
        "limit": DEFAULT_LIMIT
    }
    try:
        r = requests.get(ISRAEL_API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        flights = []
        for rec in data.get("result", {}).get("records", []):
            flights.append({
                "airline": rec.get("CHOPERD"),
                "iata": rec.get("CHLOC1"),
                "airport": rec.get("CHLOC1D"),
                "city": rec.get("CHLOC1T"),
                "country": rec.get("CHLOCCT"),
                "scheduled": rec.get("CHSTOL"),
                "actual": rec.get("CHPTOL"),
                "direction": rec.get("CHAORD"),
                "status": rec.get("CHRMINE")
            })

        result = {
            "updated": datetime.now().isoformat(),
            "count": len(flights),
            "flights": flights
        }

        with open(ISRAEL_FLIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"gov.il flights refreshed: {len(flights)} total flights cached")
        return result   # ✅ return dict instead of None

    except Exception as e:
        logger.error(f"Failed to fetch gov.il flights: {e}")
        return None     # so your endpoint can check


def fetch_airlines_from_ae(dest: str) -> set:
    """Call AE for TLV↔dest both directions. Returns set of airline names/codes."""
    if not AE_API_KEY:
        logger.warning("[WARN] No Aviation Edge API key set, skipping fetch.")
        return set()

    airlines = set()

    def fetch_with_retry(params: dict, retries: int = 1) -> list:
        """Try AE API call with retries. Returns the response data or empty list."""
        for attempt in range(retries + 1):
            time.sleep(1.05)  # Throttle: 1 request per second
            try:
                logger.debug(f"[DEBUG] AE request → {params}")
                r = requests.get(AE_BASE, params=params, timeout=15)
                if r.status_code == 200:
                    return r.json()
                else:
                    logger.warning(f"[WARN] AE status {r.status_code} for {params}")
            except Exception as e:
                logger.error(f"[ERROR] AE exception for {params}: {e}")
            if attempt < retries:
                logger.info(f"[INFO] Retrying AE call ({attempt+1}/{retries}) for {params}")
        return []

    for params in [
        {"departureIata": "TLV", "arrivalIata": dest, "key": AE_API_KEY},
        {"departureIata": dest, "arrivalIata": "TLV", "key": AE_API_KEY},
    ]:
        data = fetch_with_retry(params, retries=1)

        if isinstance(data, list):
            count_before = len(airlines)
            for it in data:
                if isinstance(it, dict) and (
                    it.get("departureIata") == "TLV" or it.get("arrivalIata") == "TLV"
                ):
                    airlines.update(extract_airline_names(it))
            added = len(airlines) - count_before
            logger.info(
                f"[INFO] AE {params['departureIata']}->{params['arrivalIata']} "
                f"added {added} airlines (total={len(airlines)})"
            )

        elif isinstance(data, dict):
            if data.get("error") == "No Record Found":
                logger.info(
                    f"[INFO] AE {params['departureIata']}->{params['arrivalIata']}: no routes found"
                )
            else:
                logger.warning(f"[WARN] Unexpected AE dict response for {dest}: {data}")

    logger.debug(f"[DEBUG] Final airlines for {dest}: {sorted(list(airlines))}")
    return airlines

def get_airlines_for(dest: str) -> set:
    """Disk-first, then AE, then persist."""
    cached = load_airlines_for(dest)
    if cached:
        return cached
    airlines = fetch_airlines_from_ae(dest)
    if airlines:
        save_airlines_for(dest, airlines)
    return airlines

# ──────────────────────────────────────────────────────────────────────────────
# In-memory dataset cache (so filters never touch AE)
# ──────────────────────────────────────────────────────────────────────────────
DATASET_DF: pd.DataFrame = pd.DataFrame()
DATASET_DATE: str = ""
DATASET_LOCK = threading.Lock()

def _read_dataset_file() -> pd.DataFrame:
    if DATASET_FILE.exists():
        try:
            meta = json.load(open(DATASET_FILE, "r", encoding="utf-8"))
            return pd.DataFrame(meta.get("airports", [])), meta.get("date")
        except Exception as e:
            logger.error(f"Failed to read dataset file: {e}")
    # empty fallback
    return pd.DataFrame(columns=["IATA","Name","City","Country","Region","lat","lon","Airlines"]), None

def get_dataset(trigger_refresh: bool = False) -> pd.DataFrame:
    """
    Return the current airport dataset, preferring in-memory or disk.
    Only trigger background AE API refresh if:
    - `trigger_refresh=True`
    - AND the dataset is not fresh
    - AND no refresh is already running
    """
    global DATASET_DF, DATASET_DATE

    today = datetime.now().strftime("%Y-%m-%d")
    fresh, file_date = _is_dataset_fresh_today()

    # ✅ In-memory fresh copy
    if fresh and not DATASET_DF.empty:
        logger.debug("Returning in-memory fresh dataset.")
        return DATASET_DF

    # ✅ Try loading from disk
    df, disk_date = _read_dataset_file()
    with DATASET_LOCK:
        DATASET_DF, DATASET_DATE = df, disk_date or ""

    # ✅ If fresh on disk or no refresh allowed
    if fresh or not trigger_refresh:
        logger.debug("Returning dataset from disk (no refresh needed).")
        return DATASET_DF

    # ✅ Stale data AND refresh is allowed AND not already running
    with PROGRESS_LOCK:
        if not PROGRESS["running"]:
            logger.info("Dataset is stale. Starting background refresh thread.")
            start_refresh_thread()
        else:
            logger.info("Refresh already running — not starting another.")

    return DATASET_DF
# ──────────────────────────────────────────────────────────────────────────────
# Dataset builder (the ONLY place that calls AE)
# ──────────────────────────────────────────────────────────────────────────────
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
    
def build_daily_dataset():
    logger.info("Building daily dataset (async batch)...")
    with PROGRESS_LOCK:
        if PROGRESS["running"]:
            logger.info("Build request ignored; already running.")
            return
        PROGRESS.update({
            "running": True,
            "total": len(AIRPORTS),
            "done": 0,
            "date": datetime.now().strftime("%Y-%m-%d"),
        })

    dest_list = AIRPORTS["IATA"].tolist()

    # Run asyncio batch fetcher
    try:
        airlines_map = asyncio.run(fetch_airlines_batch(dest_list, AE_API_KEY, max_concurrent=5))
    except Exception as e:
        logger.error(f"Async batch fetch failed, falling back to sequential: {e}")
        airlines_map = {iata: get_airlines_for(iata) for iata in dest_list}

    rows = []
    for _, r in AIRPORTS.iterrows():
        iata = r["IATA"]
        airlines = airlines_map.get(iata, set())
        with PROGRESS_LOCK:
            PROGRESS["done"] += 1

        save_airlines_for(iata, airlines)
        if not airlines:
            continue
        rows.append({
            "IATA": iata,
            "Name": r["Name"],
            "City": r["City"],
            "Country": r["Country"],
            "Region": r["Region"],
            "lat": r["lat"],
            "lon": r["lon"],
            "Airlines": normalize_airline_list(list(airlines)),
            "Distance_km": r.get("Distance_km"),
            "FlightTime_hr": r.get("FlightTime_hr"),
        })

    dataset = {"date": PROGRESS["date"], "airports": rows}
    try:
        with open(DATASET_FILE, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        logger.info(f"Dataset written: {DATASET_FILE}")
    except Exception as e:
        logger.error(f"Failed to write dataset: {e}")

    # put into memory
    with DATASET_LOCK:
        global DATASET_DF, DATASET_DATE
        DATASET_DF = pd.DataFrame(rows)
        DATASET_DATE = PROGRESS["date"]

    with PROGRESS_LOCK:
        PROGRESS["running"] = False
    logger.info("Dataset build completed (async batch).")


def start_refresh_thread():
    t = threading.Thread(target=build_daily_dataset, daemon=True)
    t.start()
    
# ───────────────────────────────────────────────
# Scheduler job
# ───────────────────────────────────────────────
def scheduled_refresh():
    """Job run by scheduler. Refresh only if stale (>24h)."""
    try:
        if not dataset_is_stale(max_age_hours=24):
            logger.info("Scheduled check: dataset fresh (<24h) -> skip refresh")
            return

        with PROGRESS_LOCK:
            if PROGRESS.get("running"):
                logger.info("Scheduled check: refresh already running -> skip")
                return

        url = "http://127.0.0.1:8000/admin/refresh?force=true"
        r = requests.post(url, timeout=60)
        safe_text = r.text.encode("ascii", "ignore").decode("ascii")  # avoid Unicode logging issues
        logger.info(f"Scheduled refresh -> {r.status_code}: {safe_text[:200]}...")

    except Exception as e:
        logger.error(f"Scheduled refresh failed: {e}")

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
                                  or q in str(r["Country"]).lower()
                                  or q in str(r["Region"]).lower(), axis=1)]

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

        def normalize_case(value: str) -> str:
            if not value or value == "—":
                return value or "—"
            return value.capitalize()            
         
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
                "Name": normalize_case(meta["Name"]),
                "City": normalize_case(meta["City"]),
                "Country": normalize_case(meta["Country"]),
                "lat": lat,
                "lon": lon,
                "Airlines": normalize_airline_list(list(meta["Airlines"])),
                "Distance_km": dist_km,
                "FlightTime_hr": flight_time_hr,
            })

    # --- Step 4: Render template ---
    countries = get_all_countries()
    logger.info(f"GET /  country={country} query='{query}'  rows={len(airports)}")

    return TEMPLATES.TemplateResponse(
        "index.html",
        {
            "request": request,
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
        logger.info(f"Placing {len([ap for ap in airports if ap.get('lat') and ap.get('lon')])} markers on map")
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

@app.get("/api/progress/stream")
async def api_progress_stream():
    async def eventgen():
        last = None
        try:
            while True:
                with PROGRESS_LOCK:
                    snapshot = {
                        "running": bool(PROGRESS.get("running")),
                        "total": int(PROGRESS.get("total", 0)),
                        "done": int(PROGRESS.get("done", 0)),
                        "date": PROGRESS.get("date"),
                    }
                payload = json.dumps(snapshot, ensure_ascii=False)

                # only send when changed, but send a heartbeat every ~30s
                if payload != last:
                    yield {"event": "progress", "data": payload}
                    last = payload
                else:
                    # heartbeat (some proxies need periodic bytes)
                    yield {"event": "ping", "data": "keep-alive"}

                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            # client disconnected; just exit quietly
            return
    return EventSourceResponse(eventgen())
    
@app.post("/admin/refresh", response_class=JSONResponse)
def admin_refresh(force: bool = Query(default=False)) -> JSONResponse:
    """
    Manually trigger dataset refresh. Will only run if:
    - force=true is passed explicitly
    - OR no dataset exists on disk
    """
    today = datetime.now().strftime("%Y-%m-%d")
    fresh, file_date = _is_dataset_fresh_today()

    if fresh and not force:
        logger.info("Refresh skipped: already fresh and no force=True.")
        return JSONResponse({"status": "fresh", "date": file_date}, status_code=200)

    with PROGRESS_LOCK:
        if PROGRESS["running"]:
            return JSONResponse({"status": "already-running"}, status_code=202)

    logger.info("Manual refresh triggered.")
    start_refresh_thread()
    return JSONResponse({
        "status": "started",
        "previous_date": file_date,
        "target_date": today
    }, status_code=202)


@app.on_event("startup")
def warm_up_airlines():
    global AIRLINE_WEBSITES
    AIRLINE_WEBSITES = load_airline_websites()
    logger.info(f"Loaded {len(AIRLINE_WEBSITES)} airline websites")


# ───────────────────────────────────────────────
# Startup handler
# ───────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    global scheduler

    # 1) Load dataset if present
    if DATASET_FILE.exists():
        df, d = _read_dataset_file()
        with DATASET_LOCK:
            global DATASET_DF, DATASET_DATE
            DATASET_DF, DATASET_DATE = df, d or ""
        logger.info(f"Startup: dataset loaded (date={DATASET_DATE}, rows={len(DATASET_DF)})")
    else:
        logger.info("Startup: no dataset file on disk")

    # 2) Refresh AE dataset if stale (>24h)
    try:
        if dataset_is_stale(max_age_hours=24):
            logger.info("Startup: AE dataset stale -> refreshing now")
            get_dataset(trigger_refresh=True)
    except Exception as e:
        logger.error(f"Startup AE refresh failed: {e}")

    # 3) Refresh gov.il flights if stale (>24h)
    try:
        if israel_flights_is_stale(max_age_hours=24):
            logger.info("Startup: gov.il flights stale -> refreshing now")
            fetch_israel_flights()
    except Exception as e:
        logger.error(f"Startup gov.il refresh failed: {e}")

    # 4) Start scheduler (every 24h)
    scheduler = BackgroundScheduler()
    scheduler.add_job(scheduled_refresh, "interval", hours=24, id="ae_refresh", replace_existing=True)
    scheduler.add_job(fetch_israel_flights, "interval", hours=24, id="govil_refresh", replace_existing=True)
    scheduler.start()
    logger.info("Scheduler started: AE + gov.il refresh every 24h")

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
            logger.error(f"Error while shutting down scheduler: {e}")
    else:
        logger.info("Shutdown event: scheduler was not running")
    
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
        Url(f"{base}/about", today, "monthly", 0.6),
        Url(f"{base}/privacy", today, "monthly", 0.5),
        Url(f"{base}/contact", today, "monthly", 0.5),
        Url(f"{base}/map", today, "weekly", 0.6),
        Url(f"{base}/flights", today, "weekly", 0.8),  # 👈 Optional: Flights page
        Url(f"{base}/?lang=he", today, "daily", 0.9),   # 👈 Optional: Hebrew version
    ]

    xml = build_sitemap(urls)

    out_path = Path("static/sitemap.xml")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(xml, encoding="utf-8")

    return Response(content=xml, media_type="application/xml")