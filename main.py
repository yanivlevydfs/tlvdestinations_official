# === Standard Library ===
import app_state
import os
import sys
import json
from html import escape
from pathlib import Path
from datetime import datetime, date
import time
from typing import Any, Dict, List
from google import genai
from fastapi import FastAPI, Request, Response, Query, HTTPException, Depends, Body
import random
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pandas as pd
import requests
import folium
from airportsdata import load
from folium.plugins import MarkerCluster
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse,RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
import logging
from logging.handlers import RotatingFileHandler
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi import status
import secrets
from starlette.exceptions import HTTPException as StarletteHTTPException
from collections import defaultdict
import html
import ast
from helpers.percent23_redirect import Percent23RedirectMiddleware
from helpers.securitymiddleware import SecurityMiddleware
from json_repair import repair_json
from json.decoder import JSONDecodeError
import httpx
from routers.infra_docs import router as infra_docs
from routers.error_handlers import (
    http_exception_handler,
    validation_exception_handler,
    generic_exception_handler
)
from helpers.helper import (
    _clean_html,
    _extract_first_href,
    _extract_first_img,
    get_git_version,
    normalize_case,
    get_lang,
    safe_js,
    get_flight_time,
    datetimeformat,
    _extract_threat_level,
    haversine_km,
    format_time,
    build_country_name_to_iso_map,
    normalize_airline_list,
    cleanup_local_cache,    
)
from helpers.sitemap_utils import Url, build_sitemap
from core.templates import TEMPLATES
import re
from helpers.chat_query import ChatQuery
from routers.middleware_redirect import redirect_and_log_404
from routers.attractions import router as attractions_router
from requests.exceptions import RequestException, ReadTimeout, ConnectTimeout
from routers.analytics import router as analytics_router
from helpers.proxies import get_random_proxy
from routers.generic_routes import router as generic_routes
from helpers.attraction_filters import load_filters
from config_paths import (
    BASE_DIR, CACHE_DIR, STATIC_DIR, TEMPLATES_DIR, DATA_DIR,
    AIRLINE_WEBSITES_FILE, ISRAEL_FLIGHTS_FILE, TRAVEL_WARNINGS_FILE,
    COUNTRY_TRANSLATIONS, CITY_TRANSLATIONS_FILE, CITY_NAME_CORRECTIONS_FILE)
from routers.sitemap_routes import router as sitemap_routes
from routers.sitemap_routes import sitemap
from routers.destination_diff_routes import router as destination_diff_routes
from helpers.destination_diff import ensure_previous_snapshot, generate_destination_diff
from routers.airlines_tlv import router as airlines_router

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
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        RotatingFileHandler(
            LOG_DIR / "app.log",
            maxBytes=10_000_000,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
# Silence httpx noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
# Silence uvicorn noise
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
# App loggers
logger = logging.getLogger("fly_tlv.flights_explorer")

feedback_logger = logging.getLogger("fly_tlv.feedback")
feedback_logger.setLevel(logging.INFO)

logger.debug("Server starting…")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set up Gemini API (NEW SDK)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash-lite"
security = HTTPBasic()

# Templates
TEMPLATES.env.globals["now"] = datetime.utcnow
TEMPLATES.env.globals['time'] = time

# Constants
TLV = {"IATA": "TLV", "Name": "Ben Gurion Airport", "lat": 32.0068, "lon": 34.8853}
ISRAEL_API = "https://data.gov.il/api/3/action/datastore_search"
RESOURCE_ID = "e83f763b-b7d7-479e-b172-ae981ddc6de5"
DEFAULT_LIMIT = 32000

TRAVEL_WARNINGS_API = "https://data.gov.il/api/3/action/datastore_search"
TRAVEL_WARNINGS_RESOURCE = "2a01d234-b2b0-4d46-baa0-cec05c401e7d"
WIKI_API_BASE = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{city}"


# App
app = FastAPI(
    title="Flights Explorer (FastAPI)",
    docs_url=None,        # disable default /docs
    redoc_url=None,       # disable default /redoc
    openapi_url=None      # disable default /openapi.json
)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.mount("/.well-known",StaticFiles(directory=STATIC_DIR / ".well-known"),name="well-known")

CORS_ORIGINS = ["http://localhost:8000", "https://fly-tlv.com"]
app.add_middleware(Percent23RedirectMiddleware)
app.add_middleware(SecurityMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(infra_docs)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)
app.middleware("http")(redirect_and_log_404)
app.include_router(attractions_router)
app.include_router(analytics_router)
app.include_router(generic_routes)
app.include_router(sitemap_routes)
app.include_router(destination_diff_routes)
app.include_router(airlines_router)

# ───────────────────────────────────────────────
# Global in-memory dataset
# ───────────────────────────────────────────────
DATASET_DF: pd.DataFrame = pd.DataFrame()
DATASET_DATE: str = ""
AIRLINE_WEBSITES: dict = {}
scheduler: AsyncIOScheduler | None = None
AIRPORTS_DB: dict = {}
COUNTRY_NAME_TO_ISO: dict[str, str] = {}
EN_TO_HE_COUNTRY = {}
CITY_TRANSLATIONS = {}
TRAVEL_WARNINGS_DF: pd.DataFrame = pd.DataFrame()

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_city_info(city_en: str, return_type: str = "both"):
    """
    Get Hebrew city and country names by English city.
    :param return_type: "both", "city", or "country"
    """
    if not CITY_TRANSLATIONS:
        logger.error("⚠️ CITY_TRANSLATIONS not loaded. Call load_city_translations() first.")
        raise RuntimeError("CITY_TRANSLATIONS not loaded. Run load_city_translations() first.")

    city_key = next((k for k in CITY_TRANSLATIONS if k.lower() == city_en.lower()), None)

    if city_key is None:
        logger.warning(f"⚠️ City '{city_en}' not found in translations.")
        return None

    entry = CITY_TRANSLATIONS[city_key]
    logger.debug(f"Found translation for '{city_en}': {entry['he']} ({entry['country_he']})")

    if return_type == "city":
        return entry["he"]
    elif return_type == "country":
        return entry["country_he"]
    else:
        return {"city_he": entry["he"], "country_he": entry["country_he"]}

def load_city_translations(file_path: Path = CITY_TRANSLATIONS_FILE):
    """
    Load city translation JSON into global dictionary.
    :param file_path: Path to the JSON file (default: DATA_DIR/city_translations.json)
    """
    global CITY_TRANSLATIONS

    try:
        if not file_path.exists():
            logger.error(f"❌ File not found: {file_path}")
            raise FileNotFoundError(f"City translations file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            CITY_TRANSLATIONS = json.load(f)

        logger.debug(f"✅ Loaded {len(CITY_TRANSLATIONS)} city entries from {file_path.name}.")

    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse {file_path.name}: {e}")
        raise

    except Exception as e:
        logger.error(f"❌ Unexpected error loading {file_path.name}: {e}")
        raise

def load_city_name_corrections() -> dict:
    
    global CITY_NAME_CORRECTIONS
    """
    Loads city_name_corrections.json into memory.
    Returns a dictionary mapping wrong city names -> correct city names.
    """
    try:
        with open(CITY_NAME_CORRECTIONS_FILE, "r", encoding="utf-8") as f:
            CITY_NAME_CORRECTIONS = json.load(f)
            return CITY_NAME_CORRECTIONS
    except FileNotFoundError:
        logger.error(f"⚠️ WARNING: {CITY_NAME_CORRECTIONS_FILE} not found!")
        CITY_NAME_CORRECTIONS = {}
    except Exception as e:
        logger.error(f"❌ Failed to load city_name_corrections.json: {e}")
        CITY_NAME_CORRECTIONS = {}

    
def load_country_translations():
    global EN_TO_HE_COUNTRY
    try:
        with open(COUNTRY_TRANSLATIONS, encoding="utf-8") as f:
            EN_TO_HE_COUNTRY = json.load(f)
            logger.debug(f"Loaded {len(EN_TO_HE_COUNTRY)} country translations from {COUNTRY_TRANSLATIONS.name}")
    except FileNotFoundError:
        logger.error(f"Translation file not found: {COUNTRY_TRANSLATIONS}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON in {COUNTRY_TRANSLATIONS}: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error loading country translations: {e}")
        
    
APP_VERSION = get_git_version()
TEMPLATES.env.globals["app_version"] = APP_VERSION

def update_travel_warnings():
    """
    Try to refresh the cached travel warnings.
    Since API access is blocked on Render, we only reload the local file.
    """
    try:
        # Try online fetch first (disabled but kept for future)
        result = fetch_travel_warnings()

        if result and result.get("count"):
            # Saved and cached → reload global
            reload_travel_warnings_globals()
            logger.debug(f"✅ Travel warnings updated via API and reloaded ({result['count']} records)")
            return

        # If API fails or returns nothing → fallback to local file
        reload_travel_warnings_globals()
        logger.debug("⚠️ API unavailable → Reloaded travel warnings from local cache only")

    except Exception:
        logger.exception("❌ Scheduled travel warnings update failed (fallback to local cache)")
        reload_travel_warnings_globals()

def reload_travel_warnings_globals():
    global TRAVEL_WARNINGS_DF
    try:
        df = load_travel_warnings_df()
        TRAVEL_WARNINGS_DF = df
        logger.debug(f"🧠 TRAVEL_WARNINGS_DF global updated (rows={len(df)})")
    except Exception as e:
        logger.exception("❌ Failed to reload TRAVEL_WARNINGS_DF from cache")
        TRAVEL_WARNINGS_DF = pd.DataFrame()

def update_flights():
    try:
        ensure_previous_snapshot()
        fetch_israel_flights()
        reload_israel_flights_globals()
        generate_destination_diff()
        sitemap()
        logger.debug("✅ Scheduled flight update + diff completed.")
    except Exception as e:
        logger.exception("❌ Scheduled flight update failed.")



def fix_city_name(name: str) -> str:
    """
    Returns corrected city name based ONLY on city_name_corrections.json.
    No hardcoded rules. No magic.
    """
    if not name:
        return name

    cleaned = name.strip()

    # direct correction from JSON
    return CITY_NAME_CORRECTIONS.get(cleaned, cleaned)
        
def reload_israel_flights_globals():
    global DATASET_DF, DATASET_DATE, DATASET_DF_FLIGHTS

    df, d = _read_dataset_file()
    DATASET_DF, DATASET_DATE = df, d or ""

    df_flights, _ = _read_flights_file()
    DATASET_DF_FLIGHTS = df_flights
    app_state.DATASET_DF = DATASET_DF
    app_state.DATASET_DF_FLIGHTS = DATASET_DF_FLIGHTS
    logger.debug(f"🔁 Globals reloaded: {len(DATASET_DF)} dataset rows, {len(DATASET_DF_FLIGHTS)} flights")

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
        logger.debug(f"✅ Loaded {len(df)} travel warnings from cache (updated {df.attrs['last_update']})")
        return df

    except Exception as e:
        logger.error(f"❌ Failed to load travel warnings: {e}")
        return pd.DataFrame()
    
TEMPLATES.env.filters["datetimeformat"] = datetimeformat
    
# ──────────────────────────────────────────────────────────────────────────────
# Load airports once for dataset builds
# ──────────────────────────────────────────────────────────────────────────────

def fetch_travel_warnings(batch_size: int = 500) -> dict | None:
    """
    Fetch all travel warnings from data.gov.il.
    Try direct connection first, fall back to proxy rotation only if blocked.
    Includes JSON repair, paging, and DataFrame updates.
    """
    HEADERS = {
        "User-Agent": (
            "datagov-external-client; Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.gov.il/",
    }

    offset = 0
    all_records = []
    global TRAVEL_WARNINGS_DF

    while True:
        params = {
            "resource_id": TRAVEL_WARNINGS_RESOURCE,
            "limit": batch_size,
            "offset": offset,
        }

        # --- 1️⃣ Direct fetch first ---
        try:
            logger.debug(f"🌐 Fetching TW offset={offset} direct …")
            r = requests.get(
                TRAVEL_WARNINGS_API,
                params=params,
                headers=HEADERS,
                timeout=40,
            )
            r.raise_for_status()
        except Exception as e:
            logger.warning(f"⚠️ Direct fetch failed ({e}) → switching to proxy.")
            # --- 2️⃣ Proxy fallback ---
            proxy = get_random_proxy()
            proxy_url = f"http://{proxy['user']}:{proxy['pass']}@{proxy['host']}:{proxy['port']}"
            proxies = {"http": proxy_url, "https": proxy_url}
            try:
                r = requests.get(
                    TRAVEL_WARNINGS_API,
                    params=params,
                    headers=HEADERS,
                    proxies=proxies,
                    timeout=40,
                )
                r.raise_for_status()
            except Exception as e2:
                logger.error(f"❌ Proxy {proxy['host']} failed: {e2}")
                continue

        # --- 3️⃣ Parse / repair JSON ---
        try:
            data = r.json()
        except JSONDecodeError:
            try:
                fixed_json = repair_json(r.text)
                data = json.loads(fixed_json)
                logger.debug("🔧 JSON repaired successfully")
            except Exception as parse_err:
                logger.error(f"❌ JSON repair failed: {parse_err}")
                return None


        # ------------------ PARSE JSON ------------------
        try:
            data = r.json()
        except JSONDecodeError:
            try:
                fixed_json = repair_json(r.text)
                data = json.loads(fixed_json)
                logger.debug("🔧 JSON repaired")
            except Exception as parse_err:
                logger.error(f"❌ JSON repair failed: {parse_err}")
                return None

        records = data.get("result", {}).get("records", [])
        total = data.get("result", {}).get("total", 0)

        if not records:
            logger.warning("⚠️ No more records — stopping.")
            break

        # ------------------ PROCESS RECORDS ------------------
        for rec in records:
            raw_reco = rec.get("recommendations", "")
            raw_details = rec.get("details", "")

            all_records.append({
                "id": rec.get("_id"),
                "continent": rec.get("continent", "").strip(),
                "country": rec.get("country", "").strip(),
                "recommendations": _clean_html(raw_reco),
                "level": _extract_threat_level(raw_reco),
                "details_url": (
                    _extract_first_href(raw_reco)
                    or _extract_first_href(raw_details)
                ),
                "logo": _extract_first_img(rec.get("logo", "")),
                "date": rec.get("date"),
                "office": rec.get("משרד", ""),
            })

        offset += batch_size
        logger.debug(f"📦 Loaded {len(all_records)}/{total}")

        # stop if last batch
        if len(records) < batch_size or offset >= total:
            break

    # ------------------ SAVE RESULT ------------------
    if not all_records:
        logger.error("❌ No travel warnings fetched at all")
        return None

    result = {
        "updated": datetime.now().isoformat(),
        "count": len(all_records),
        "warnings": all_records,
    }

    try:
        with open(TRAVEL_WARNINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.debug("💾 Travel warnings cached to disk")
    except Exception as e:
        logger.error(f"❌ Failed saving travel warnings: {e}")
        return None

    # ------------------ UPDATE GLOBAL DF ------------------
    TRAVEL_WARNINGS_DF = pd.DataFrame(all_records)
    TRAVEL_WARNINGS_DF.attrs["last_update"] = result["updated"]

    logger.debug(f"🧠 TRAVEL_WARNINGS_DF updated ({len(TRAVEL_WARNINGS_DF)} rows)")

    return result

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

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
   
def fetch_israel_flights(batch_size: int = 500) -> dict | None:
    """
    Fetch all Israel Airports Authority flight records from data.gov.il.
    First tries a direct HTTPS request; if blocked (403, timeout, etc.),
    falls back to rotating Webshare proxies. Handles JSON repair,
    DataFrame normalization, and disk caching.
    """
    HEADERS = {
        "User-Agent": (
            "datagov-external-client; Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.gov.il/",
    }

    params = {
        "resource_id": RESOURCE_ID,
        "limit": DEFAULT_LIMIT,
    }

    logger.debug("🌐 Requesting Israel flight data (direct + proxy fallback)…")

    # ----------------------------
    # 1️⃣ TRY DIRECT REQUEST
    # ----------------------------
    try:
        r = requests.get(ISRAEL_API, params=params, headers=HEADERS, timeout=40)
        r.raise_for_status()
    except Exception as e:
        logger.warning(f"⚠️ Direct fetch failed ({e}) → switching to proxy.")

        # ----------------------------
        # 2️⃣ FALLBACK: ROTATING PROXY
        # ----------------------------
        while True:
            proxy = get_random_proxy()
            proxy_url = f"http://{proxy['user']}:{proxy['pass']}@{proxy['host']}:{proxy['port']}"
            proxies = {"http": proxy_url, "https": proxy_url}
            logger.debug(f"🌍 Trying proxy {proxy['host']} …")

            try:
                r = requests.get(
                    ISRAEL_API,
                    params=params,
                    headers=HEADERS,
                    proxies=proxies,
                    timeout=40,
                )
                r.raise_for_status()
                break  # ✅ success
            except Exception as e2:
                logger.error(f"❌ Proxy {proxy['host']} failed → retrying: {e2}")
                continue

    # ----------------------------
    # 3️⃣ PARSE + JSON REPAIR
    # ----------------------------
    try:
        data = r.json()
    except JSONDecodeError:
        try:
            fixed_json = repair_json(r.text)
            data = json.loads(fixed_json)
            logger.debug("🔧 JSON repaired successfully (flights)")
        except Exception as parse_err:
            logger.error(f"❌ Cannot parse JSON from gov.il flights: {parse_err}")
            return None

    # ✅ Everything below should always run after JSON is parsed
    records = data.get("result", {}).get("records", [])
    logger.debug(f"✈ Received {len(records)} raw flight rows")

    if not records:
        logger.warning("⚠ Gov.il returned 0 records for flights.")
        return None

    # ----------------------------
    # PARSE & NORMALIZE RECORDS
    # ----------------------------
    flights = []
    for rec in records:
        iata = (rec.get("CHLOC1") or "").strip().upper()
        direction = normalize_case(rec.get("CHAORD", ""))

        if not iata or not direction:
            continue

        raw_city = normalize_case(rec.get("CHLOC1T", ""))
        corrected_city = fix_city_name(raw_city)

        flights.append({
            "airline": normalize_case(rec.get("CHOPERD", "")),
            "iata": iata,
            "airport": normalize_case(rec.get("CHLOC1D", "")),
            "city": corrected_city,
            "country": normalize_case(rec.get("CHLOCCT", "")),
            "scheduled": normalize_case(rec.get("CHSTOL", "")),
            "actual": normalize_case(rec.get("CHPTOL", "")),
            "direction": direction,
            "status": normalize_case(rec.get("CHRMINE", "")),
        })

    if not flights:
        logger.warning("⚠ No valid flight rows after filtering.")
        return None

    df = pd.DataFrame(flights)
    df["iata"] = (
        df["iata"].astype(str).str.strip().str.upper().replace("NAN", None)
    )
    flights = df.to_dict(orient="records")

    result = {
        "updated": datetime.now().isoformat(),
        "count": len(flights),
        "flights": flights,
    }

    # ----------------------------
    # WRITE TO CACHE FILE
    # ----------------------------
    try:
        with open(ISRAEL_FLIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.debug(f"💾 Cached {len(flights)} flights to disk")
    except Exception as e:
        logger.error(f"❌ Failed writing flights cache: {e}")
        return None

    logger.debug(f"✈ Flight data refreshed ({len(flights)} records)")    
    return result

def _read_flights_file() -> tuple[pd.DataFrame, str | None]:
    """
    Extract raw 'flights' records from israel_flights.json without aggregation.
    Returns a flat flight events DataFrame and optional ISO date string.
    """
    try:
        with open(ISRAEL_FLIGHTS_FILE, "r", encoding="utf-8") as f:
            try:
                meta = json.load(f)
            except JSONDecodeError as e:
                logger.warning(f"⚠️ Corrupted israel_flights.json detected: {e}")
                f.seek(0)
                broken_text = f.read()

                try:
                    fixed_text = repair_json(broken_text)
                    meta = json.loads(fixed_text)
                    logger.warning("✅ israel_flights.json repaired successfully using json-repair")

                    # Optional: rewrite the repaired version to disk
                    with open(ISRAEL_FLIGHTS_FILE, "w", encoding="utf-8") as fw:
                        json.dump(meta, fw, ensure_ascii=False, indent=2)
                        logger.warning("💾 Repaired israel_flights.json saved to disk")

                except Exception as repair_err:
                    logger.error(f"❌ JSON repair failed: {repair_err}", exc_info=True)
                    return pd.DataFrame(columns=[
                        "airline", "iata", "airport", "city", "country",
                        "scheduled", "actual", "direction", "status"
                    ]), None

        flights = meta.get("flights", [])
        if not flights:
            logger.warning("⚠️ No 'flights' found in israel_flights.json")
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

    # Return empty fallback DataFrame
    return pd.DataFrame(columns=[
        "airline", "iata", "airport", "city", "country",
        "scheduled", "actual", "direction", "status"
    ]), None

def _read_dataset_file() -> tuple[pd.DataFrame, str | None]:
    global AIRPORTS_DB

    if not ISRAEL_FLIGHTS_FILE.exists():
        return pd.DataFrame(columns=[
            "IATA", "Name", "City", "Country", "lat", "lon", "Airlines", "Direction"
        ]), None

    try:
        # ✅ Attempt to load JSON normally
        with open(ISRAEL_FLIGHTS_FILE, "r", encoding="utf-8") as f:
            try:
                meta = json.load(f)
            except JSONDecodeError as e:
                logger.warning(f"⚠️ Corrupted israel_flights.json detected: {e}")
                f.seek(0)
                broken_text = f.read()
                try:
                    fixed_text = repair_json(broken_text)
                    meta = json.loads(fixed_text)
                    logger.debug("✅ israel_flights.json repaired successfully using json-repair")

                    # Optional: persist repaired version
                    with open(ISRAEL_FLIGHTS_FILE, "w", encoding="utf-8") as fw:
                        json.dump(meta, fw, ensure_ascii=False, indent=2)
                        logger.debug("💾 Repaired israel_flights.json saved to disk")
                except Exception as repair_err:
                    logger.error(f"❌ JSON repair failed: {repair_err}", exc_info=True)
                    return pd.DataFrame(columns=[
                        "IATA", "Name", "City", "Country", "lat", "lon", "Airlines", "Direction"
                    ]), None

        flights = meta.get("flights", [])
        grouped: dict[tuple[str, str], dict] = {}

        for rec in flights:
            iata = rec.get("iata")
            direction = rec.get("direction")
            if not iata or not direction:
                continue

            key = (iata, direction)
            entry = grouped.setdefault(key, {
                "IATA": iata,
                "Direction": direction,
                "Name": rec.get("airport") or "—",
                "City": rec.get("city") or "—",
                "Country": rec.get("country") or "—",
                "Airlines": set(),
            })
            airline = rec.get("airline")
            if airline:
                entry["Airlines"].add(airline)

        rows = []
        for (iata, direction), info in grouped.items():
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

        # Extract last update date if present
        file_date = None
        updated = meta.get("updated")
        if updated:
            try:
                iso_date = updated.split("T")[0]
                file_date = datetime.strptime(iso_date, "%Y-%m-%d").strftime("%d-%m-%Y")
            except Exception:
                file_date = iso_date

        return df, file_date

    except Exception as e:
        logger.error(f"Failed to read dataset file: {e}", exc_info=True)
        return pd.DataFrame(columns=[
            "IATA", "Name", "City", "Country", "lat", "lon", "Airlines", "Direction"
        ]), None

def load_israel_flights_map():
    """
    Build a mapping {IATA: set(airlines)} directly from DATASET_DF.
    No file access, no reloading—DATASET_DF already contains all current flights.
    """
    global DATASET_DF

    if DATASET_DF is None or DATASET_DF.empty:
        logger.warning("DATASET_DF is empty — cannot build flights map.")
        return {}

    mapping = {}

    for _, row in DATASET_DF.iterrows():
        iata = row.get("IATA") or row.get("iata")
        airline = row.get("AIRLINE") or row.get("airline")

        if not iata or not airline:
            continue

        normalized = str(airline).strip().lower()
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

def schedule_immediate_refresh():
    global scheduler

    if scheduler is None:
        logger.error("Scheduler not initialized — cannot schedule refresh")
        return

    job_id = "manual_refresh_once"

    # Prevent duplicates
    if scheduler.get_job(job_id):
        logger.debug("Manual refresh already scheduled — skipping")
        return

    logger.warning("🕓 Dataset empty — scheduling immediate refresh job")

    scheduler.add_job(
        refresh_data_webhook,   # sync function
        trigger="date",
        run_date=datetime.now(),
        id=job_id,
        replace_existing=True,
        misfire_grace_time=300,
        max_instances=1,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.head("/")
async def read_root_head():
    return Response(status_code=200)

@app.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    country: str = "All",
    query: str = "",    
    lang: str = Depends(get_lang),
):
    global AIRPORTS_DB, DATASET_DF, AIRLINE_WEBSITES

    if DATASET_DF is None or DATASET_DF.empty:
        logger.warning("⚠️ DATASET_DF empty — scheduling background refresh")
        schedule_immediate_refresh()

        return TEMPLATES.TemplateResponse(
            "error.html",
            {
                "request": request,
                "message": "Data is loading. Please refresh in a moment.",
                "lang": lang,
            },
            status_code=503,
        )
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
    logger.debug(f"GET /  country={country} query='{query}'  rows={len(airports)}")
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
            "version": APP_VERSION,
        },
    )

@app.get("/map", response_class=HTMLResponse)
def map_view(country: str = "All", query: str = ""):
    global DATASET_DF_FLIGHTS, AIRPORTS_DB

    if DATASET_DF_FLIGHTS is None or DATASET_DF_FLIGHTS.empty:
        logger.error("⚠️ No flight data available for map")
        return TEMPLATES.TemplateResponse("error.html", {
            "request": request,
            "message": "No flight data available to show on the map.",
        })

    df = DATASET_DF_FLIGHTS.copy()
    airports = df.to_dict(orient="records")
    existing_iatas = {
        str(ap.get("iata", "")).upper()
        for ap in airports
        if isinstance(ap, dict)
        and ap.get("iata")
        and isinstance(ap.get("iata"), (str, int, float))
    }


    # ✅ Group flights by IATA, ignore direction (D/A)
    grouped: dict[str, dict] = {}

    for rec in DATASET_DF_FLIGHTS.to_dict(orient="records"):
        iata = rec.get("iata")
        if not iata:
            continue

        # Initialize entry if not exists
        if iata not in grouped:
            grouped[iata] = {
                "Name": rec.get("airport") or "—",
                "City": rec.get("city") or "—",
                "Country": rec.get("country") or "—",
                "Airlines": set(),
            }

        # Merge airlines across both directions
        if rec.get("airline"):
            for a in rec["airline"].split(","):
                grouped[iata]["Airlines"].add(a.strip())

    # ✅ Enrich data from AIRPORTS_DB and append
    merged_airports = {}

    for iata, meta in grouped.items():
        if iata in merged_airports:
            continue  # skip duplicates

        lat = lon = None
        dist_km = flight_time_hr = "—"

        if iata in AIRPORTS_DB:
            lat = AIRPORTS_DB[iata].get("lat")
            lon = AIRPORTS_DB[iata].get("lon")

        if lat and lon:
            dist_km = round(haversine_km(TLV["lat"], TLV["lon"], lat, lon), 1)
            flight_time_hr = get_flight_time(dist_km)

        merged_airports[iata] = {
            "IATA": iata,
            "Name": meta["Name"].title(),
            "City": meta["City"].title(),
            "Country": meta["Country"].title(),
            "lat": lat,
            "lon": lon,
            "Airlines": sorted(meta["Airlines"]) if meta["Airlines"] else ["—"],
            "Distance_km": dist_km,
            "FlightTime_hr": flight_time_hr,
        }

    airports = list(merged_airports.values())

    # ✅ Filter by country
    if country and country != "All":
        airports = [
            ap for ap in airports
            if ap.get("Country", "").lower() == country.lower()
        ]

    # ✅ Filter by query (city, name, code, country)
    if query:
        q = query.strip().lower()
        airports = [
            ap for ap in airports
            if q in str(ap.get("IATA", "")).lower()
            or q in str(ap.get("Name", "")).lower()
            or q in str(ap.get("City", "")).lower()
            or q in str(ap.get("Country", "")).lower()
        ]

    # ✅ Create Folium map
    m = folium.Map(
        location=[35, 28.5],
        zoom_start=5,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
        zoom_control=True
    )
    cluster = MarkerCluster().add_to(m)

    # Add TLV marker
    folium.Marker(
        [TLV["lat"], TLV["lon"]],
        tooltip="Tel Aviv (TLV)",
        icon=folium.Icon(color="blue", icon="plane", prefix="fa")
    ).add_to(m)

    bounds = [[TLV["lat"], TLV["lon"]]]

    # ✅ Add destination markers
    for ap in airports:
        if not ap.get("lat") or not ap.get("lon"):
            continue

        km = ap["Distance_km"]
        flight_time_hr = ap["FlightTime_hr"]
        flights_url = f"https://www.google.com/travel/flights?q=flights%20from%20TLV%20to%20{ap['IATA']}"
        skyscanner_url = f"https://www.skyscanner.net/transport/flights/tlv/{ap['IATA'].lower()}/"
        gmaps_url = f"https://maps.google.com/?q={ap['City'].replace(' ','+')},{ap['Country'].replace(' ','+')}"
        copy_js = f"navigator.clipboard && navigator.clipboard.writeText('{safe_js(ap['IATA'])}')"

        airlines_val = ap.get("Airlines", [])
        if isinstance(airlines_val, str):
            airlines_val = [airlines_val]
        chips = []

        for a in airlines_val:
            name = a.strip()
            if not name:
                continue
            url = AIRLINE_WEBSITES.get(name)
            style = (
                "display:inline-block;"
                "margin:2px 4px 2px 0;"
                "padding:4px 10px;"
                "font-size:12px;"
                "border-radius:9999px;"
                "background:#6f42c1;"
                "color:white;"
                "text-decoration:none;"
                "border:none;"
                "font-weight:500;"
                "box-shadow:0 0 0 2px rgba(111,66,193,0.2);"
                "transition:all 0.2s ease-in-out;"
            )
            if url:
                chips.append(f"<a href='{escape(url)}' target='_blank' class='chip' style='{style}'>{escape(name)}</a>")
            else:
                chips.append(f"<span class='chip' style='{style}'>{escape(name)}</span>")

        airline_html = (
            "<div style='margin-top:6px;font-size:13px'>Airlines:<br>"
            + "".join(chips) +
            "</div>"
        ) if chips else "<div style='margin-top:6px;font-size:13px'>Airlines: <b>—</b></div>"

        # ---- Popup HTML ----
        popup_html = f"""
            <div style='font-family:system-ui;min-width:250px;max-width:300px'>
                <div style='font-weight:600;font-size:15px'>{escape(ap['Name'])} ({escape(ap['IATA'])})</div>
                <div style='color:#6b7280;font-size:12px;margin-top:2px'>{escape(ap['City'])} · {escape(ap['Country'])}</div>
                <div style='margin-top:6px;font-size:13px'>Distance: <b>{km} km</b></div>
                <div style='margin-top:2px;font-size:13px'>Flight time: <b>{flight_time_hr}</b></div>
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

    logger.debug(f"✅ /map rendered: {len(airports)} unique airports (merged by IATA)")

    # ✅ Ensure map reflows inside modal if needed
    bounds_js = json.dumps(bounds)
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

    html = m.get_root().render().replace("</body>", fix_script + "</body>")
    return HTMLResponse(content=html, media_type="text/html; charset=utf-8")


# ───────────────────────────────────────────────
# Startup handler
# ───────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    global scheduler, AIRLINE_WEBSITES, AIRPORTS_DB
    global TRAVEL_WARNINGS_DF, COUNTRY_NAME_TO_ISO
    global DATASET_DF, DATASET_DATE, DATASET_DF_FLIGHTS
    global APP_VERSION
    global EN_TO_HE_COUNTRY
    global CITY_NAME_CORRECTIONS
    
    logger.debug("🚀 Application startup initiated")
    # 🎯 0) Set Git version
    logger.debug(f"🔖 App Version: {APP_VERSION}")
    cleanup_local_cache()
    load_city_translations()
    load_country_translations()
    load_city_name_corrections()
    
    # 0) Load IATA DB once
    try:
        AIRPORTS_DB = load("IATA")
        logger.debug(f"Loaded airportsdata IATA DB with {len(AIRPORTS_DB)} records")
    except Exception as e:
        logger.error("Failed to load airportsdata", exc_info=True)
        AIRPORTS_DB = {}

    # 1) Init scheduler
    scheduler = AsyncIOScheduler()
    logger.debug("AsyncIOScheduler instance created")

    # 2) Load travel warnings
    try:
        update_travel_warnings()        
    except Exception as e:
        logger.error("Failed to load travel warnings", exc_info=True)
        TRAVEL_WARNINGS_DF = pd.DataFrame()

    # 3) Load airline websites
    try:
        AIRLINE_WEBSITES = load_airline_websites()
        logger.debug(f"Loaded {len(AIRLINE_WEBSITES)} airline websites")
    except Exception as e:
        logger.error("Failed to load airline websites", exc_info=True)
        AIRLINE_WEBSITES = {}

    # 4) Load datasets or fetch from API
    try:
        update_flights()
    except Exception as e:
        logger.error("Error loading or fetching datasets", exc_info=True)
        DATASET_DF = pd.DataFrame()
        DATASET_DF_FLIGHTS = pd.DataFrame()

    app_state.DATASET_DF = DATASET_DF
    app_state.TRAVEL_WARNINGS_DF = TRAVEL_WARNINGS_DF

    # 5) Load country → ISO code mapping
    try:
        COUNTRY_NAME_TO_ISO = build_country_name_to_iso_map()
        logger.debug(f"Loaded {len(COUNTRY_NAME_TO_ISO)} country → ISO mappings")
    except Exception as e:
        logger.error("Failed to build ISO mapping", exc_info=True)
        COUNTRY_NAME_TO_ISO = {}

    # 6) Schedule background jobs
    try:
        scheduler.add_job(
            update_flights,
            "interval",
            hours=3,
            id="govil_refresh",
            replace_existing=True,
            next_run_time=datetime.now()
        )
        scheduler.add_job(
            update_travel_warnings,
            "interval",
            hours=24,
            id="warnings_refresh",
            replace_existing=True,
            next_run_time=datetime.now())           
        scheduler.start()
        logger.debug("✅ Scheduler started")
    except Exception as e:
        logger.error("Failed to start scheduler", exc_info=True)

    logger.debug("🎯 Application startup completed")


@app.on_event("shutdown")
async def shutdown_event():
    global scheduler
    logger.debug("🛑 Application shutdown initiated")

    if scheduler:
        try:
            scheduler.shutdown(wait=False)
            logger.debug("✅ Scheduler stopped successfully")
        except Exception as e:
            logger.error("Error shutting down scheduler", exc_info=True)
    else:
        logger.warning("No scheduler instance to shut down")

    logger.debug("👋 Application shutdown completed")


def generate_questions_from_data(destinations: list[dict], n: int = 20) -> list[str]:
    """
    Generate bilingual (EN + HE) flight-related questions from destination data.
    - Auto-uses Hebrew translations via get_city_info().
    - Handles invalid / float / NaN fields gracefully.
    """
    n = max(1, min(n, 50))

    cities, countries, airlines = set(), set(), set()

    for dest in destinations:
        try:
            city = str(dest.get("city", "") or "").strip()
            country = str(dest.get("country", "") or "").strip()
            airline_list = dest.get("airline") or []

            # Normalize airline list
            if isinstance(airline_list, str):
                airline_list = [a.strip() for a in airline_list.split(",") if a.strip()]
            elif isinstance(airline_list, list):
                airline_list = [str(a).strip() for a in airline_list if a]
            else:
                airline_list = []

            # Skip invalid data
            if not city or not country or city.lower() == "nan" or country.lower() == "nan":
                continue

            # Add to collections
            cities.add((city, country))
            countries.add(country)
            for airline in airline_list:
                if airline and airline.lower() != "nan":
                    airlines.add(airline)

        except Exception as e:
            logger.warning(f"⚠️ Skipping malformed record: {e}")
            continue

    cities = list(cities)
    countries = list(countries)
    airlines = list(airlines)

    if not (cities or countries or airlines):
        logger.warning("⚠️ No valid data for question generation.")
        return ["No valid data available for question generation."]

    questions = []

    # --- 🇬🇧 ENGLISH ---
    if countries:
        for country in random.sample(countries, min(5, len(countries))):
            questions.append(f"What cities can I fly to in {country}?")
            questions.append(f"Are there direct flights to {country}?")
            questions.append(f"Which destinations are available in {country}?")
            questions.append(f"Can I fly to {country} from TLV?")
            questions.append(f"What airports are available in {country}?")

    if cities:
        for city, country in random.sample(cities, min(5, len(cities))):
            questions.append(f"Which airlines fly directly to {city}?")
            questions.append(f"Is there a direct flight to {city}?")
            questions.append(f"What country does {city} belong to?")
            questions.append(f"Can I fly to {city} from Ben Gurion Airport?")
            questions.append(f"Are there flights from TLV to {city}?")


    if airlines:
        for airline in random.sample(airlines, min(5, len(airlines))):
            questions.append(f"What are the destinations for {airline}?")
            questions.append(f"Where does {airline} operate flights?")
            questions.append(f"What cities are served by {airline}?")
            questions.append(f"What countries does {airline} fly to?")
            questions.append(f"Which routes are available with {airline}?")
            url = AIRLINE_WEBSITES.get(airline)
            if url:
                questions.append(f"What is the website of {airline}?")
                questions.append(f"Show me the website for {airline}")
                questions.append(f"Where can I find the website of {airline}?")
                questions.append(f"Does {airline} have a website?")
                questions.append(f"Can you give me the link to {airline}'s website?")
                questions.append(f"What's the official site of {airline}?")
                questions.append(f"Where do I find the official website of {airline}?")
                questions.append(f"Is there an official website for {airline}?")
                questions.append(f"Give me the website for {airline}.")
                questions.append(f"What's the homepage URL of {airline}?")

    # --- 🇮🇱 HEBREW (with translation lookup) ---
    if countries:
        for country in random.sample(countries, min(4, len(countries))):
            try:
                # Try to get Hebrew version of the country (via any city in it)
                city_example = next((c for c, cn in cities if cn == country), None)
                he_country = None
                if city_example:
                    info = get_city_info(city_example, return_type="both")
                    he_country = info.get("country_he") if info else None

                if not he_country:
                    he_country = country  # fallback to English
                questions.append(f"לאילו ערים אפשר לטוס בתוך {he_country}?")
                questions.append(f"איזה יעדים יש במדינת {he_country}?")
                questions.append(f"אילו טיסות יוצאות ל-{he_country}?")
                questions.append(f"אילו ערים ב-{he_country} זמינות בטיסות ישירות?")
                questions.append(f"לאן אפשר להגיע ב-{he_country} בטיסה?")
            except Exception as e:
                logger.error(f"⚠️ Country translation fallback: {e}")
                continue

    if cities:
        for city, country in random.sample(cities, min(4, len(cities))):
            try:
                info = get_city_info(city, return_type="both")
                city_he = info.get("city_he") if info else city
                country_he = info.get("country_he") if info else country
                questions.append(f"עם אילו חברות תעופה אפשר להגיע ל-{city_he}?")
                questions.append(f"אילו טיסות מגיעות ל-{city_he}?")
                questions.append(f"יש טיסות ישירות ל-{city_he}?")
                questions.append(f"אילו חברות מציעות טיסות ל-{city_he}?")
                questions.append(f"אילו חברות טסות ישירות ל-{city_he}?")
            except Exception as e:
                logger.error(f"⚠️ City translation fallback: {e}")
                continue

    if airlines:
        for airline in random.sample(airlines, min(4, len(airlines))):
            questions.append(f"לאן אפשר לטוס עם חברת {airline}?")
            questions.append(f"באילו מסלולים פועלת חברת {airline}?")
            questions.append(f"אילו מדינות נמצאות ברשימת היעדים של חברת {airline}?")
            questions.append(f"מהם היעדים שחברת {airline} מגיעה אליהם?")
            questions.append(f"באילו נמלי תעופה נוחתת חברת {airline}?")
            questions.append(f"האם חברת {airline} טסה ליעדים בינלאומיים?")
            questions.append(f"יש טיסות של חברת {airline} לאירופה?")
            questions.append(f"אילו יעדים פופולריים יש עם חברת {airline}?")
            questions.append(f"באילו מדינות פועלת חברת {airline}?")
            questions.append(f"מהם היעדים העיקריים של חברת {airline}?")
            url = AIRLINE_WEBSITES.get(airline)
            if url:
                questions.append(f"מה האתר של חברת {airline}?")
                questions.append(f"תראה לי את האתר של חברת {airline}")
                questions.append(f"איפה אפשר למצוא את האתר של חברת {airline}?")
                questions.append(f"האם יש אתר לחברת {airline}?")
                questions.append(f"אתה יכול לשלוח לי קישור לאתר של חברת {airline}?")
                questions.append(f"מהו האתר הרשמי של חברת {airline}?")

    random.shuffle(questions)
    logger.debug(f"✅ Generated {len(questions[:n])} bilingual question suggestions (EN+HE).")
    return questions[:n]

def build_flight_context(df) -> str:
    
    global AIRLINE_WEBSITES
    
    """
    Build a rich Markdown-formatted flight context from the dataset.
    Supports:
      - City → Country lookups
      - Country → Cities lists
      - Airline → Destinations
      - Airline → Countries
      - Country → Destinations
      - Airline → Website Directory
    """

    # === Data Structures ===
    grouped = defaultdict(set)              # (iata, city, country) -> airlines
    city_country_pairs = set()              # (city, country)
    country_to_cities = defaultdict(set)    # country -> cities
    airline_routes = defaultdict(set)       # airline -> (iata, city, country)
    airline_to_countries = defaultdict(set) # airline -> countries

    # === Normalize and aggregate ===
    for row in df.to_dict(orient="records"):
        iata = str(row.get("iata", "—")).strip()
        city = str(row.get("city", "—")).strip()
        country = str(row.get("country", "—")).strip()
        airlines = row.get("airline", [])

        # Skip incomplete or invalid rows
        if not iata or not city or not country or "—" in (iata, city, country):
            continue

        # Normalize airline field
        if isinstance(airlines, str):
            airlines = [a.strip() for a in airlines.split(",") if a.strip()]
        elif isinstance(airlines, list):
            airlines = [str(a).strip() for a in airlines if str(a).strip()]
        else:
            airlines = []

        # Update relationships
        key = (iata, city, country)
        for airline in airlines:
            grouped[key].add(airline)
            airline_routes[airline].add(key)
            airline_to_countries[airline].add(country)

        city_country_pairs.add((city, country))
        country_to_cities[country].add(city)

    # === Section 1: City → Country ===
    city_country_section = "\n".join(
        f"- {city} is in {country}" for city, country in sorted(city_country_pairs)
    )

    # === Section 2: Country → Cities ===
    country_city_section = "\n".join(
        f"- **{country}** includes cities: {', '.join(sorted(cities))}"
        for country, cities in sorted(country_to_cities.items())
    )

    # === Section 3: Flights by Destination (Grouped by Country) ===
    country_dest_map = defaultdict(list)
    for (iata, city, country), airlines in grouped.items():
        if airlines:
            country_dest_map[country].append((iata, city, sorted(airlines)))

    destination_section = []
    for country, destinations in sorted(country_dest_map.items()):
        destination_section.append(f"✈️ **Flights to {country}**\n")
        
        for iata, city, airline_list in sorted(destinations, key=lambda x: x[1]):
            destination_section.append(f"**{city} ({iata}, {country})**")
            destination_section.extend([f"- {airline}" for airline in airline_list])
            destination_section.append("")  # newline after each city
        
        destination_section.append("---")  # divider between countries (optional)
        destination_section.append("")     # extra newline

    # === Section 4: Airline → Destinations (Beautiful Markdown) ===
    airline_dest_section = []
    for airline, destinations in sorted(airline_routes.items()):
        airline_dest_section.append(f"🛫 **{airline}**")
        for iata, city, country in sorted(destinations, key=lambda x: (x[2], x[1])):
            airline_dest_section.append(f"- {city} ({iata}, {country})")
        airline_dest_section.append("")

    # === Section 5: Airline → Countries ===
    airline_country_section = "\n".join(
        f"- **{airline}** operates in: {', '.join(sorted(countries))}"
        for airline, countries in sorted(airline_to_countries.items())
    )
    
    # === Section 6: Airline → Website Directory ===
    airline_website_section = "\n".join(
        f"- {airline}: {url}"
        for airline, url in sorted(AIRLINE_WEBSITES.items())
        if url
    )
    # === Section 7: Real-Time Flight Status Data ===
    if {"actual", "direction", "status"}.issubset(df.columns):

        # Separate arrivals & departures
        arrivals = []
        departures = []

        for row in df.to_dict(orient="records"):
            airline = str(row.get("airline", "")).strip()
            airport = str(row.get("airport", "")).strip()
            time = str(row.get("actual", "")).strip()
            status = str(row.get("status", "")).strip()
            direction = str(row.get("direction", "")).strip().upper()

            # Skip incomplete or invalid rows
            if not all([airline, airport, time, status, direction]):
                continue

            if direction == "A":
                arrivals.append(f"- **{airline}** — Arrival — {time} — {status} — {airport}")
            elif direction == "D":
                departures.append(f"- **{airline}** — Departure — {time} — {status} — {airport}")

        # Build final text output
        parts = ["🕓 **Flight Status Schedule**"]

        if arrivals:
            parts.append("\n### 🛬 Arrivals\n" + "\n".join(arrivals))

        if departures:
            parts.append("\n### 🛫 Departures\n" + "\n".join(departures))

        flight_status_text = "\n".join(parts)

    else:
        flight_status_text = ""

        
    # === Final Context Assembly ===
    context = (
        "📌 **City-to-Country Mapping**\n"
        + city_country_section
        + "\n\n🌍 **Country-to-Cities**\n"
        + country_city_section
        + "\n\n🛬 **Flights by Destination**\n"
        + "\n".join(destination_section)
        + "\n\n🛫 **Airline-to-Destinations**\n"
        + "\n".join(airline_dest_section)
        + "\n\n🌐 **Airline-to-Countries**\n"
        + airline_country_section
        + "\n\n🌐 **Airline Website Directory**\n"
        + airline_website_section
    )

    if flight_status_text:
        context += "\n\n" + flight_status_text

    return context


@app.post("/api/chat", response_class=JSONResponse)
async def chat_flight_ai(
    query: ChatQuery = Body(...),
    lang: str = Query(default="en")
):
    global DATASET_DF_FLIGHTS

    question = query.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    logger.debug(f"📥 User question: {question}")
    if DATASET_DF_FLIGHTS.empty:
        raise HTTPException(status_code=503, detail="Flight dataset is empty or not loaded.")

    # Build structured context    
    context = build_flight_context(DATASET_DF_FLIGHTS)

    # Construct the AI prompt
    prompt = f"""
    You are a highly accurate aviation assistant helping users explore direct flights departing from Ben Gurion Airport (TLV).

You are given structured aviation data including:
- ✅ Verified airport destinations
- ✅ City-to-country mappings
- ✅ Airline-to-destination mappings
- ✅ Real-time flight schedule data (arrival/departure/status)

📊 DATA START:
{context}
📊 DATA END.

Now, using ONLY the data above, answer the user's question below:

"{question}"

🚨 STRICT RESPONSE RULES — FOLLOW EXACTLY 🚨

✅ SUPPORTED QUESTION TYPES:
- Which cities are in [Country]?
- What country is [City] located in?
- What destinations does [Airline] serve?
- What airlines fly to [City] or [Country]?
- What is the website of [Airline]?
- What flights are arriving/departing to/from [City or Country]?
- What is the status of flights to/from [City or Airport]?
- When is the next flight to/from [City or Country]?
- שאלות בעברית? → Translate to English first, then answer in English.

🧾 OUTPUT FORMAT (REQUIRED):

1. If the question is about flight destinations:
   - Start with a bold heading: **✈️ Flights to [Country/Region/City]**
   - Use Markdown bullet points:
     - Format each destination as: **City (IATA, Country)** — [https://fly-tlv.com/destinations/IATA](https://fly-tlv.com/destinations/IATA)
       - Replace `IATA` in the URL with the actual IATA code of the destination (e.g., GYD)
     - Under each city, indent each airline as a separate sub-bullet:
       - Format: `- Airline Name — [https://example.com](https://example.com)`
       - You MUST include the link if it is available in the data
       - Only omit the link if no URL is available for that airline
       - One airline per line
       - No commas, no inline lists, no slashes

2. If the question is about an airline website:
   - Start with a bold heading: **🌐 Website for [Airline]**
   - On the next line, provide the URL as a Markdown link:
     [https://example.com](https://example.com)
   - Do not include summaries, explanations, or extra text.

3. If the question is about flight times or statuses:
   - Start with a bold heading: **🕓 Flight Details for [City or Airport]**
   - Use a Markdown bullet list for each matching flight:
     - Format: **Airline Name** — Direction — Time — Status
       - `Direction`: use "Arrival" if `A`, "Departure" if `D`
       - `Time`: use the `actual` field exactly as shown (e.g., 2025-11-15T18:06:00)
       - `Status`: must match exactly from the data (e.g., On Time, Landed)
   - Sort by time ascending
   - Show up to 10 matching flights

🟢 CORRECT EXAMPLE:
---
**✈️ Flights to United States**
- Newark (EWR, United States) — [https://fly-tlv.com/destinations/EWR](https://fly-tlv.com/destinations/EWR)  
  - Delta Airlines — [https://delta.com](https://delta.com)  
  - El Al Israel Airlines — [https://www.elal.com](https://www.elal.com)  
  - Jetblue Airways Corporation — [https://jetblue.com](https://jetblue.com)  
  - United Airlines — [https://united.com](https://united.com)  

**🕓 Flight Details for Munich**
- Lufthansa — Departure — 2025-11-15T18:08:00 — Departed  
- Brussels Airlines — Departure — 2025-11-15T18:08:00 — Departed  
---

🚫 NEVER DO THIS:
- No commas in airline lists
- No paragraphs, summaries, or prose
- No tables or YAML
- No Hebrew in the output
- Do NOT say “Sure”, “Here’s the answer”, etc.
- Do NOT show airline name without a link if one exists
- Do NOT reword or reformat time/status/direction values

❌ IF THERE’S NO MATCH:
Reply ONLY with this exact line (no formatting):
I couldn't find it in our current destination catalog, please check the main table.
"""
    
    logger.debug("Gemini prompt built successfully (length=%d chars)", len(prompt))

    try:
        response = await gemini_client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        answer = response.text or "Currently I have no answer"
    except Exception as e:
        logger.error("Gemini API error: %s", str(e).split('\n')[0])
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
    
@app.get("/api/chat/suggestions", response_class=JSONResponse)
async def chat_suggestions(n: int = Query(default=10, le=20)):
    """
    Return up to `n` suggested chat questions (in English and Hebrew).
    """
    global DATASET_DF_FLIGHTS

    if DATASET_DF_FLIGHTS.empty:
        raise HTTPException(status_code=503, detail="Destination data not loaded.")

    destinations = DATASET_DF_FLIGHTS.to_dict(orient="records")

    try:
        suggestions = generate_questions_from_data(destinations, n)
    except Exception as e:
        logger.error("Suggestion generation failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Could not generate suggestions.")

    if not suggestions:
        return {"questions": ["No suggestions available at the moment."]}

    logger.debug(f"GET /api/chat/suggestions → {len(suggestions)} suggestions")
    return {"questions": suggestions}

    
@app.get("/travel-warnings", response_class=HTMLResponse)
async def travel_warnings_page(request: Request, lang: str = Depends(get_lang)):
    global TRAVEL_WARNINGS_DF

    client_host = request.client.host if request.client else "unknown"

    if TRAVEL_WARNINGS_DF is None or TRAVEL_WARNINGS_DF.empty:
        logger.error(f"GET /travel-warnings from {client_host} (lang={lang}) → no cached data")
        return TEMPLATES.TemplateResponse("error.html", {
            "request": request,
            "lang": lang,
            "message": "No travel warnings available at this time."
        }, status_code=503)

    warnings = TRAVEL_WARNINGS_DF[TRAVEL_WARNINGS_DF["office"] == 'מל"ל'].to_dict(orient="records")

    # Safely get and parse last update timestamp
    last_update_str = TRAVEL_WARNINGS_DF.attrs.get("last_update")
    try:
        dtlu = datetime.fromisoformat(last_update_str)
        last_update = dtlu.strftime("%d-%m-%Y")
    except Exception as e:
        logger.error(f"Failed to parse last_update: {e}")
        last_update = None

    continents = sorted(TRAVEL_WARNINGS_DF["continent"].dropna().unique())
    countries  = sorted(TRAVEL_WARNINGS_DF["country"].dropna().unique())
    levels     = ["גבוה", "בינוני", "נמוך", "לא ידוע"]

    logger.debug(f"GET /travel-warnings from {client_host} (lang={lang}) → {len(warnings)} warnings")

    return TEMPLATES.TemplateResponse("travel_warnings.html", {
        "request": request,
        "lang": lang,
        "warnings": warnings,
        "last_update": last_update,
        "continents": continents,
        "countries": countries,
        "levels": levels
    })

@app.get("/flights", response_class=HTMLResponse)
async def flights_view(request: Request):
    global DATASET_DF_FLIGHTS

    # ✅ Handle missing data
    if DATASET_DF_FLIGHTS is None or DATASET_DF_FLIGHTS.empty:
        return TEMPLATES.TemplateResponse("error.html", {
            "request": request,
            "message": "No live flight data available.",
            "lang": request.query_params.get("lang", "en")
        })

    flights = DATASET_DF_FLIGHTS.to_dict(orient="records")
    processed_flights = []

    # ✅ Process times only for display, no filtering
    for f in flights:
        s_short, s_full, s_iso = format_time(f.get("scheduled", ""))
        a_short, a_full, a_iso = format_time(f.get("actual", ""))

        f["scheduled"] = s_short
        f["scheduled_full"] = s_full
        f["scheduled_iso"] = s_iso
        f["actual"] = a_short
        f["actual_full"] = a_full
        f["actual_iso"] = a_iso

        processed_flights.append(f)

    # ✅ Extract dropdown filters (all flights included)
    countries = sorted({
        f.get("country", "").strip()
        for f in processed_flights if f.get("country")
    })

    actual_times = sorted({
        f["actual_iso"].split("T")[1][:5]
        for f in processed_flights
        if f.get("actual_iso") and "T" in f["actual_iso"]
    })

    # ✅ Build actual_dates dropdown list (optional)
    actual_dates_set = set()
    for f in processed_flights:
        iso = f.get("actual_iso")
        if iso and "T" in iso:
            try:
                date_part = iso.split("T")[0]
                label = datetime.strptime(date_part, "%Y-%m-%d").strftime("%b %d")
                actual_dates_set.add((date_part, label))
            except ValueError:
                continue
    actual_dates = sorted(actual_dates_set)

    # ✅ Simple logs
    logger.debug(f"✅ Loaded total flights: {len(processed_flights)} (no filtering applied)")

    return TEMPLATES.TemplateResponse("flights.html", {
        "request": request,
        "flights": processed_flights,
        "countries": countries,
        "actual_dates": actual_dates,
        "actual_times": actual_times,
        "last_update": get_dataset_date(),  # just show dataset date
        "lang": request.query_params.get("lang", "en"),
        "AIRLINE_WEBSITES": AIRLINE_WEBSITES
    })

@app.get("/destinations", include_in_schema=False)
async def redirect_to_home(request: Request):
    global DATASET_DF

    if DATASET_DF is None or DATASET_DF.empty:
        logger.error("DATASET_DF is not loaded. Redirecting anyway to avoid SEO issues.")

    lang = request.query_params.get("lang")
    url = f"/?lang={lang}" if lang else "/"
    return RedirectResponse(url=url, status_code=301)
  
@app.get("/destinations/{iata}", response_class=HTMLResponse)
async def destination_detail(request: Request, iata: str):
    """Render destination page and export it to static HTML for SEO"""
    global DATASET_DF, COUNTRY_NAME_TO_ISO, AIRLINE_WEBSITES, STATIC_DIR

    lang = request.query_params.get("lang", "en").lower()
    iata = iata.upper().strip()

    # ✅ Redirect if TLV itself is requested
    if iata in {"TLV", "LLBG", "BEN-GURION"}:
        logger.debug(f"🔁 Redirected attempt to access TLV itself → homepage")
        target = "/" if lang == "en" else "/?lang=he"
        return RedirectResponse(url=target, status_code=302)


    # ✅ 1. Validate dataset
    if DATASET_DF is None or DATASET_DF.empty:
        return TEMPLATES.TemplateResponse(
            "error.html",
            {
                "request": request,
                "message": "No destination data available.",
                "status_code": 410,
                "lang": lang
            },
            status_code=410
        )

    iata = iata.upper()
    destination = DATASET_DF.loc[DATASET_DF["IATA"] == iata]

    if destination.empty:
        return TEMPLATES.TemplateResponse(
            "error.html",
            {
                "request": request,
                "message": f"Destination {iata} not found from Tel-Aviv (TLV).",
                "status_code": 410,
                "lang": lang
            },
            status_code=410
        )

    dest = destination.iloc[0].to_dict()

    # ✅ 2. Normalize and find country ISO
    country_name = str(dest.get("Country", "")).strip()
    iso_code = COUNTRY_NAME_TO_ISO.get(country_name) or next(
        (v for k, v in COUNTRY_NAME_TO_ISO.items() if k.lower() == country_name.lower()),
        ""
    )
    dest["CountryISO"] = iso_code or ""

    # ✅ 3. Prepare static export path
    output_dir = STATIC_DIR / "destinations" / lang
    if iso_code:
        output_dir = output_dir / iso_code.upper()

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"⚠️ Could not create folder {output_dir}: {e}")

    output_file = output_dir / f"{iata.lower()}.html"

    # ✅ 4. Serve cached version if still fresh
    MAX_AGE = 432000  # 5 days
    if output_file.exists():
        mtime = output_file.stat().st_mtime
        if time.time() - mtime < MAX_AGE:
            logger.debug(f"🗂️ Using cached static page: {output_file}")
            return HTMLResponse(content=output_file.read_text(encoding='utf-8'))
        else:
            logger.debug(f"♻️ Rebuilding expired page for {iata}")

    # ✅ 5. Fetch travel info internally (from your existing route)
    travel_info_data = None
    wiki_summary_data = None
    city_name = dest.get("City", "").strip()
    lat = dest.get("lat")
    lon = dest.get("lon")

    if city_name:
        try:
            travel_info_data = await get_travel_info(city_name, lang=lang)
            logger.debug(f"✅ Travel info loaded for coordinates {lat}, {lon}")
        except Exception as e:
            logger.error(f"⚠️ Travel info unavailable for ({lat},{lon}): {e}", exc_info=True)
            travel_info_data = None
    else:
        logger.warning(f"⚠️ No coordinates available for destination {city_name!r}")
        travel_info_data = None
    # Fetch Wikipedia summary
    try:
        wiki_summary_data = await fetch_wikipedia_summary(city_name, lang)
        if wiki_summary_data:
            logger.debug(f"✅ Wikipedia summary fetched for {city_name}")
        else:
            logger.error(f"⚠️ No Wikipedia summary for {city_name}")
    except Exception as e:
        logger.error(f"⚠️ Wikipedia summary unavailable for {city_name}: {e}")
        wiki_summary_data = None

    # ✅ 6. Render and cache
    rendered_html = TEMPLATES.get_template("destination.html").render({
        "request": request,
        "destination": dest,
        "lang": lang,
        "AIRLINE_WEBSITES": AIRLINE_WEBSITES,
        "travel_info": travel_info_data,
        "wiki_summary": wiki_summary_data
    })

    try:
        with open(output_file, "w", encoding="utf-8", buffering=1024*64) as f:
            f.write(rendered_html)
        logger.debug(f"🌍 Static page generated: {output_file}")
    except Exception as e:
        logger.error(f"⚠️ Failed to write static HTML for {iata}: {e}")


    return HTMLResponse(content=rendered_html)

@app.post("/api/chat/feedback")
async def receive_feedback(payload: dict):
    question = payload.get("question")
    score = payload.get("score")
    feedback_logger.debug(f"{score} | {question}")
    return {"status": "ok"}
    
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

@app.get("/api/refresh-data", response_class=JSONResponse)
def refresh_data_webhook():
    global DATASET_DF, DATASET_DATE, DATASET_DF_FLIGHTS
    ensure_previous_snapshot()
    logger.warning("🔁 Incoming request: /api/refresh-data")

    response = {
        "fetch_israel_flights": None,
        "fetch_travel_warnings": None
    }

    # Run each fetch with logging
    try:
        res1 = fetch_israel_flights()
        if res1:
            logger.debug("✅ fetch_israel_flights completed successfully")
            reload_israel_flights_globals()
            response["fetch_israel_flights"] = "Success"
        else:
            logger.error("❌ fetch_israel_flights returned None")
            response["fetch_israel_flights"] = "Failed: returned None"
    except Exception as e:
        logger.exception("❌ Exception in fetch_israel_flights")
        response["fetch_israel_flights"] = f"Exception: {str(e)}"

    try:
        res2 = fetch_travel_warnings()
        if res2:
            logger.debug("✅ fetch_travel_warnings completed successfully")
            response["fetch_travel_warnings"] = "Success"
        else:
            logger.error("❌ fetch_travel_warnings returned None")
            response["fetch_travel_warnings"] = "Failed: returned None"
    except Exception as e:
        logger.exception("❌ Exception in fetch_travel_warnings")
        response["fetch_travel_warnings"] = f"Exception: {str(e)}"

    logger.debug("🔁 Refresh summary: %s", json.dumps(response, indent=2, ensure_ascii=False))
    return response
      
@app.get("/feed.xml", response_class=Response)
def flight_feed():
    now = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000")
    items = []

    for _, row in DATASET_DF[DATASET_DF["Direction"].str.upper() == "D"].iterrows():
        direction_label = "Arrival" if row["Direction"] == "A" else "Departure"
        title = f"{direction_label} | {row['City']} ({row['IATA']}) - {row['Country']}"
        link = f"https://fly-tlv.com/destinations/{row['IATA']}"
        
        try:
            airlines_list = ast.literal_eval(row['Airlines']) if isinstance(row['Airlines'], str) else row['Airlines']
        except Exception:
            airlines_list = [row['Airlines']]
        
        airlines = ', '.join(a.strip() for a in airlines_list)

        description = (
            f"{row['Name']} ({row['IATA']}) in {row['City']}, {row['Country']}<br>"
            f"Airlines: {airlines}<br>"
            f"Distance: {row['Distance_km']} km<br>"
            f"Flight Time: {row['FlightTime_hr']}"
        )

        item = f"""<item>
<title>{html.escape(title)}</title>
<link>{html.escape(link)}</link>
<guid isPermaLink="true">{html.escape(link)}</guid>
<pubDate>{now}</pubDate>
<description><![CDATA[{description}]]></description>
</item>"""

        items.append(item)

    rss = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
<title>TLV Flight Destinations Feed</title>
<link>https://fly-tlv.com/feed.xml</link>
<description>All destinations served from TLV</description>
<language>en-us</language>
<lastBuildDate>{now}</lastBuildDate>
{''.join(items)}
</channel>
</rss>"""

    return Response(content=rss, media_type="application/rss+xml")
    
@app.get("/travel-questionnaire", include_in_schema=False)
async def travel_questionnaire(request: Request, lang: str = "en"):
    countries = sorted(DATASET_DF["Country"].dropna().unique())
    return TEMPLATES.TemplateResponse("questionnaire.html", {
        "request": request,
        "countries": countries,
        "lang": lang,
        "last_update": get_dataset_date()
    })

@app.get("/api/cities")
async def get_cities(country: str):
    df = DATASET_DF
    df.columns = df.columns.str.strip()  # Clean any extra spaces
    cities = sorted(df[df["Country"] == country]["City"].dropna().unique())
    return JSONResponse(content={"cities": cities})

@app.get("/api/airports")
async def get_airports(country: str, city: str):
    df = DATASET_DF.copy()
    df.columns = df.columns.str.strip()

    # Filter by country + city
    filtered = df[(df["Country"] == country) & (df["City"] == city)]

    # Normalize Airlines column
    filtered.loc[:, "Airlines"] = filtered["Airlines"].astype(str)

    # Drop duplicate airport entries
    unique_airports = filtered[["IATA", "Name", "Airlines"]].drop_duplicates(subset=["IATA", "Name"])

    airports = []
    for _, row in unique_airports.iterrows():
        airlines = []

        val = row["Airlines"]
        if pd.notna(val) and val.strip().lower() != "nan":
            # Try to parse list-like strings safely
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    airlines = [str(a).strip() for a in parsed if str(a).strip()]
                else:
                    airlines = [str(parsed).strip()]
            except Exception:
                # fallback for comma-separated
                airlines = [a.strip() for a in val.replace(";", ",").split(",") if a.strip()]

        airports.append({
            "iata": row.IATA,
            "name": row.Name,
            "airlines": airlines
        })

    return JSONResponse(content={"airports": airports})

@app.get("/api/warnings")
async def get_warnings(country: str):
    df = TRAVEL_WARNINGS_DF.copy()
    df.columns = df.columns.str.strip().str.lower()

    he_country = EN_TO_HE_COUNTRY.get(country)
    if not he_country:
        return JSONResponse(status_code=400, content={"error": f"No Hebrew mapping found for '{country}'"})

    warnings = df[(df["country"] == he_country) & (df["office"] == 'מל"ל')]

    if not warnings.empty:
        # Keep only the required columns
        warnings = warnings[["recommendations", "details_url", "date"]]
        warnings = warnings.rename(columns={"details_url": "link"})
        return JSONResponse(content={"warnings": warnings.to_dict(orient="records")})

    return JSONResponse(content={"warnings": []})

@app.get("/api/wiki-summary")
async def fetch_wikipedia_summary(
    city: str = Query(..., description="City name in English (or Hebrew if lang=he)"),
    lang: str = Query("en", pattern="^(en|he)$", description="Language: en or he")
):

    """
    🔹 Fetch summarized Wikipedia info for a city (supports English & Hebrew)
    🔹 Hebrew mode:
        1. Translate city → Hebrew (if exists)
        2. Try fetching Hebrew Wikipedia
        3. If missing → fallback to English Wikipedia
    """
    try:
        city_original = city.strip().title()
        city = city_original

        # ============================================================
        # 🈁 HEBREW MODE — ATTEMPT TRANSLATION + FALLBACK TO ENGLISH
        # ============================================================
        if lang == "he":

            # Try Hebrew database translation
            translated_city = get_city_info(city_original, return_type="city")

            if translated_city:
                logger.debug(f"🌐 Translating '{city_original}' → '{translated_city}' for Hebrew Wikipedia lookup.")
                city_he = translated_city
            else:
                logger.warning(f"⚠️ No Hebrew translation for '{city_original}', using English name.")
                city_he = city_original

            # → First try Hebrew Wikipedia
            url_he = WIKI_API_BASE.format(lang="he", city=city_he.replace(" ", "_"))
            logger.debug(f"📘 Trying Hebrew Wikipedia for '{city_he}' → {url_he}")

            async with httpx.AsyncClient(timeout=15) as client:
                response_he = await client.get(url_he, headers={
                    "User-Agent": "Fly-TLV (https://fly-tlv.com; contact@fly-tlv.com)",
                    "Accept": "application/json"
                })

            # If found → return Hebrew page
            if response_he.status_code == 200:
                data = response_he.json()
                return {
                    "title": data.get("title"),
                    "description": data.get("description"),
                    "extract": data.get("extract"),
                    "thumbnail": data.get("thumbnail", {}).get("source"),
                    "lang": "he",
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page"),
                    "requested_city": city_he
                }

            # If not → fallback to English
            logger.warning(f"⚠️ Hebrew Wikipedia NOT found for '{city_he}'. Falling back to English.")

            lang = "en"     # Force English lookup
            city = city_original   # Use English city name

        # ============================================================
        # 🌍 ENGLISH MODE (default or fallback)
        # ============================================================

        city_en = city.replace(" ", "_")
        url = WIKI_API_BASE.format(lang="en", city=city_en)
        logger.debug(f"🌍 Fetching English Wikipedia for '{city}' → {url}")

        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url, headers={
                "User-Agent": "Fly-TLV (https://fly-tlv.com; contact@fly-tlv.com)",
                "Accept": "application/json"
            })

        if response.status_code == 404:
            raise HTTPException(404, f"No Wikipedia article for {city} (en)")

        if response.status_code != 200:
            logger.error(f"❌ Wikipedia API returned {response.status_code} for '{city}'")
            raise HTTPException(response.status_code, "Wikipedia API error")

        data = response.json()

        result = {
            "title": data.get("title"),
            "description": data.get("description"),
            "extract": data.get("extract"),
            "thumbnail": data.get("thumbnail", {}).get("source"),
            "lang": "en",
            "url": data.get("content_urls", {}).get("desktop", {}).get("page"),
            "requested_city": city
        }

        if not result["extract"]:
            raise HTTPException(204, "No summary available")

        return result

    except httpx.TimeoutException:
        raise HTTPException(504, "Wikipedia request timed out")

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"💥 Unexpected error fetching Wikipedia for '{city_original}' ({lang}): {e}")
        raise HTTPException(500, str(e))

async def get_travel_info(city: str, lang: str = "en") -> Dict[str, Any]:
    """
    PRODUCTION-GRADE tourist attraction search by CITY NAME.
    Uses Wikipedia GeoSearch + robust filtering.
    Supports Hebrew + English.
    Extremely strict in keeping only real attractions.
    """

    # ============================================================
    # 0️⃣ VALIDATE INPUT
    # ============================================================
    if not city or not isinstance(city, str) or not city.strip():
        return {"city": city, "pois": [], "tips": [], "error": "Invalid city"}

    city_clean = city.strip()
    logger.debug(f"🌍 get_travel_info(city={city_clean}, lang={lang})")

    # ============================================================
    # 1️⃣ TRANSLATE CITY FOR HEBREW LOOKUP
    # ============================================================
    if lang == "he":
        try:
            translated = get_city_info(city_clean, return_type="city")
            city_lookup = translated or city_clean
            if translated:
                logger.debug(f"✨ Hebrew lookup: {city_lookup}")
        except Exception:
            city_lookup = city_clean
    else:
        city_lookup = city_clean

    # ============================================================
    # 2️⃣ CHOOSE API BASE
    # ============================================================
    if lang == "he":
        WIKI_API = "https://he.wikipedia.org/w/api.php"
        FALLBACK_API = "https://en.wikipedia.org/w/api.php"
    else:
        WIKI_API = "https://en.wikipedia.org/w/api.php"
        FALLBACK_API = None

    # ============================================================
    # 3️⃣ HTTP CLIENT
    # ============================================================
    HEADERS = {"User-Agent": "FlyTLV/1.0 (contact@fly-tlv.com)"}
    timeout = httpx.Timeout(connect=5, read=10, write=5, pool=5)

    try:
        async with httpx.AsyncClient(timeout=timeout, headers=HEADERS) as client:

            # ============================================================
            # 4️⃣ GET CITY COORDINATES (HE → fallback EN → resolve ambiguity)
            # ============================================================
            async def fetch_coords(api, title):
                try:
                    return await client.get(
                        api,
                        params={
                            "action": "query",
                            "format": "json",
                            "prop": "coordinates|categories",
                            "redirects": 1,
                            "titles": title,
                            "cllimit": 50,
                            "clshow": "!hidden",
                        },
                    )
                except Exception:
                    return None

            # First attempt
            geo_resp = await fetch_coords(WIKI_API, city_lookup)

            # Fallback to EN
            if (
                not geo_resp
                or not geo_resp.json().get("query", {}).get("pages")
            ) and FALLBACK_API:
                logger.warning("⚠️ Hebrew page missing → English fallback")
                geo_resp = await fetch_coords(FALLBACK_API, city_clean)

            # Extract page
            data = geo_resp.json()
            pages = data.get("query", {}).get("pages", {})
            page = next(iter(pages.values()), {})

            # ------------------------------------------------------------
            # Detect disambiguation and resolve it
            # ------------------------------------------------------------
            categories = [c.get("title", "").lower() 
                          for c in page.get("categories", [])]

            is_disambig = any("disambiguation" in c for c in categories)

            if is_disambig:
                logger.debug(f"⚠️ '{city_lookup}' is a disambiguation page → resolving best match…")

                # Search for real pages
                search_resp = await client.get(
                    WIKI_API,
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": city_lookup,
                        "format": "json",
                    },
                )

                search_list = search_resp.json().get("query", {}).get("search", [])
                if search_list:

                    resolved_page = None
                    filters = load_filters()
                    CITY_KEYWORDS_EN = filters.get("CITY_KEYWORDS_EN", [])
                    CITY_KEYWORDS_HE = filters.get("CITY_KEYWORDS_HE", [])

                    for item in search_list:
                        candidate = item["title"]

                        geo_resp2 = await fetch_coords(WIKI_API, candidate)
                        if not geo_resp2:
                            continue

                        data2 = geo_resp2.json()
                        page2 = next(iter(data2.get("query", {}).get("pages", {}).values()), {})

                        # ---------------------------------------------------
                        # Skip disambiguation pages
                        # ---------------------------------------------------
                        cats = [c.get("title", "").lower() for c in page2.get("categories", [])]
                        if any("disambiguation" in c for c in cats):
                            continue

                        # ---------------------------------------------------
                        # Must have coordinates to be a real place
                        # ---------------------------------------------------
                        if not page2.get("coordinates"):
                            continue

                        # ---------------------------------------------------
                        # CITY DETECTOR from categories
                        # ---------------------------------------------------
                        cat_text = " ".join(cats)

                        is_city = (
                            any(k in cat_text for k in CITY_KEYWORDS_EN) or
                            any(k in cat_text for k in CITY_KEYWORDS_HE)
                        )

                        if not is_city:
                            # Example: New York (state) → skip
                            #          New York County → skip
                            continue

                        # FIRST valid city wins → stop
                        logger.debug(f"➡️ Resolved ambiguous '{city_lookup}' → city page: '{candidate}'")
                        resolved_page = page2
                        break

                    if not resolved_page:
                        logger.debug("❌ No valid city pages found in disambiguation search")
                        return {"city": city_lookup, "pois": [], "tips": []}

                    page = resolved_page

                else:
                    logger.debug("❌ No usable results in disambiguation search")
                    return {"city": city_lookup, "pois": [], "tips": []}



            # ------------------------------------------------------------
            # Extract coordinates (with safe fallback to English canonical page)
            # ------------------------------------------------------------
            # ------------------------------------------------------------
            # Extract coordinates (Wikipedia → fallback to English → GEOPY)
            # ------------------------------------------------------------
            coords = page.get("coordinates")
            lat = None
            lon = None

            # -----------------------------
            # 1) Wikipedia coords (primary)
            # -----------------------------
            if coords:
                try:
                    lat = coords[0].get("lat")
                    lon = coords[0].get("lon")
                except:
                    lat = None
                    lon = None

            # ------------------------------------------------------------
            # 2) If still missing → try English canonical variations
            # ------------------------------------------------------------
            if (lat is None or lon is None) and FALLBACK_API:
                logger.warning(f"⚠️ No coords for '{city_clean}' → trying English canonical variants")

                candidates = [
                    city_clean,
                    f"{city_clean}, Spain",
                    f"{city_clean} (city)",
                    f"{city_clean} (Spain)",
                ]

                for cand in candidates:
                    try:
                        geo_fb_resp = await fetch_coords(FALLBACK_API, cand)
                        if not geo_fb_resp:
                            continue

                        data_fb = geo_fb_resp.json()
                        page_fb = next(iter(data_fb.get("query", {}).get("pages", {}).values()), {})
                        coords_fb = page_fb.get("coordinates")

                        # Skip disambiguation
                        cats = [c.get("title", "").lower() for c in page_fb.get("categories", [])]
                        if any("disambiguation" in c for c in cats):
                            continue

                        if coords_fb:
                            lat = coords_fb[0].get("lat")
                            lon = coords_fb[0].get("lon")
                            logger.debug(f"✔️ Canonical resolved '{cand}' → {lat},{lon}")
                            break
                    except:
                        continue

            # ------------------------------------------------------------
            # 3) FINAL FALLBACK → PHOTON (RELIABLE SERVER-SAFE)
            # ------------------------------------------------------------
            if lat is None or lon is None:
                logger.debug(f"⚠️ Wikipedia failed → PHOTON fallback for '{city_clean}'")

                try:
                    photon_url = (
                        "https://photon.komoot.io/api/"
                        f"?q={city_clean}&limit=1"
                    )

                    resp = await client.get(photon_url, timeout=10)
                    data_photon = resp.json()

                    features = data_photon.get("features")
                    if features:
                        coords_photon = features[0]["geometry"]["coordinates"]  # [lon, lat]
                        lon = coords_photon[0]
                        lat = coords_photon[1]

                        logger.debug(f"✔️ Photon resolved '{city_clean}' → {lat}, {lon}")
                    else:
                        logger.error(f"❌ Photon could not resolve '{city_clean}'")
                        return {"city": city_lookup, "pois": [], "tips": []}

                except Exception as ex:
                    logger.error(f"❌ Photon error for '{city_clean}': {ex}")
                    return {"city": city_lookup, "pois": [], "tips": []}


            # ------------------------------------------------------------
            # SAFETY CHECK
            # ------------------------------------------------------------
            if lat is None or lon is None:
                logger.error(f"❌ FINAL coord failure for '{city_clean}'")
                return {"city": city_lookup, "pois": [], "tips": []}

            logger.debug(f"📍 Coordinates for {city_clean}: {lat},{lon}")

            # ============================================================
            # 5️⃣ GEOSEARCH — find POIs near the city center
            # ============================================================
            geo_resp = await client.get(
                WIKI_API,
                params={
                    "action": "query",
                    "list": "geosearch",
                    "gscoord": f"{lat}|{lon}",
                    "gsradius": 10000,
                    "gslimit": 50,
                    "gsprop": "type|name",
                    "format": "json",
                },
            )

            geolist = geo_resp.json().get("query", {}).get("geosearch", [])
            logger.debug(f"🔍 Found {len(geolist)} raw POIs for {city_lookup}")

            if not geolist:
                return {"city": city_lookup, "pois": [], "tips": []}

            # ============================================================
            # 6️⃣ DETAIL FETCH + FILTERING
            # ============================================================
            pois = []
            seen = set()
            # -------------- FILTER CONSTANTS -----------------
            filters = load_filters()
            BLOCK = filters["BLOCK"]
            ALLOW = filters["ALLOW"]
            DESC_KEYS = filters["DESC_KEYS"]
            TITLE_KEYS = filters["TITLE_KEYS"]

            # -------------------------------------------------
            for item in geolist:   # NO LIMIT HERE
                pageid = item.get("pageid")
                title = item.get("title")

                if not pageid:
                    continue

                # ----- BATCH DETAIL -----
                try:
                    dresp = await client.get(
                        WIKI_API,
                        params={
                            "action": "query",
                            "pageids": pageid,
                            "prop": "extracts|pageimages|langlinks|categories",
                            "redirects": 1,
                            "explaintext": 1,
                            "exintro": 1,
                            "exchars": 700,
                            "exsectionformat": "plain",
                            "exlang": lang,
                            "pithumbsize": 640,
                            "lllimit": 1,
                            "lllang": "he" if lang == "he" else "en",
                            "cllimit": 100,
                            "clshow": "!hidden",
                            "format": "json",
                        },
                    )
                except Exception:
                    continue

                #dpage = next(iter(dresp.json().get("query", {}).get("pages", {}).values()), {})
                # -------- SAFE JSON PARSING (MANDATORY) --------
                if dresp.status_code != 200 or not dresp.content:
                    logger.warning(
                        f"⚠️ POI detail empty/non-200 "
                        f"(status={dresp.status_code}) pageid={pageid}"
                    )
                    continue

                try:
                    ddata = dresp.json()
                except ValueError:
                    logger.warning(
                        f"⚠️ POI detail non-JSON pageid={pageid}: "
                        f"{dresp.text[:120]}"
                    )
                    continue

                dpage = next(
                    iter(ddata.get("query", {}).get("pages", {}).values()),
                    {}
                )


                # ===== normalize =====
                rawcats = [c.get("title", "") for c in dpage.get("categories", [])]

                def norm(x):
                    x = x.lower()
                    x = x.replace("category:", "").replace("קטגוריה:", "")
                    return x.strip()

                cat_titles = [norm(x) for x in rawcats]
                desc = (dpage.get("extract") or "").lower()
                title_low = title.lower()

                # ====================================================
                # 🎯 FILTER LAYERS
                # ====================================================

                # 1️⃣ BLOCK
                if any(bad in cat for bad in BLOCK for cat in cat_titles):
                    continue

                # 2️⃣ ALLOW
                allow_hit = any(good in cat for good in ALLOW for cat in cat_titles)

                # 3️⃣ DESC
                desc_hit = any(k in desc for k in DESC_KEYS)

                # 4️⃣ TITLE
                title_hit = any(k in title_low for k in TITLE_KEYS)

                if not (allow_hit or desc_hit or title_hit):
                    continue

                # ====================================================
                # TITLE in Hebrew
                # ====================================================
                if lang == "he" and "langlinks" in dpage:
                    for link in dpage["langlinks"]:
                        if link.get("lang") == "he":
                            title = link.get("*") or title

                # ====================================================
                # CREATE POI
                # ====================================================
                thumb = dpage.get("thumbnail", {}).get("source")
                poi_lat = item.get("lat")
                poi_lon = item.get("lon")

                # dedup
                key = (title.strip().lower(), round(poi_lat or 0, 5), round(poi_lon or 0, 5))
                if key in seen:
                    continue
                seen.add(key)

                pois.append({
                    "name": title,
                    "description": dpage.get("extract", ""),
                    "image": thumb,
                    "lat": poi_lat,
                    "lon": poi_lon,
                    "gmap_url": f"https://www.google.com/maps/search/?api=1&query={poi_lat},{poi_lon}"
                })

            random.shuffle(pois)
            return {"city": city_lookup, "pois": pois, "tips": []}

    # ============================================================
    # 7️⃣ GLOBAL EXCEPTIONS
    # ============================================================
    except httpx.TimeoutException:
        return {"city": city_lookup, "pois": [], "tips": [], "error": "Timeout contacting Wikipedia"}

    except httpx.NetworkError:
        return {"city": city_lookup, "pois": [], "tips": [], "error": "Network error"}

    except Exception as e:
        logger.error("🔥 Unexpected error in get_travel_info()", exc_info=True)
        return {"city": city_lookup, "pois": [], "tips": [], "error": "Internal error"}

