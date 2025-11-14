# === Standard Library ===
import os
import sys
import json
from html import escape
from pathlib import Path
from datetime import datetime, date
import time
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
import pycountry
from geopy.distance import geodesic
import subprocess
import html
import ast
from percent23_redirect import Percent23RedirectMiddleware
from securitymiddleware import SecurityMiddleware
from json_repair import repair_json
from json.decoder import JSONDecodeError
import httpx

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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Set up Gemini API

genai.configure(api_key=GEMINI_API_KEY)
chat_model = genai.GenerativeModel("gemini-2.5-flash-lite")
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
TEMPLATES.env.globals['time'] = time

# Data files
AIRLINE_WEBSITES_FILE = DATA_DIR / "airline_websites.json"
ISRAEL_FLIGHTS_FILE   = CACHE_DIR / "israel_flights.json"
TRAVEL_WARNINGS_FILE = CACHE_DIR / "travel_warnings.json"
COUNTRY_TRANSLATIONS = DATA_DIR / "country_translations.json"
CITY_TRANSLATIONS_FILE = DATA_DIR / "city_translations.json"

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
    logger.info(f"Found translation for '{city_en}': {entry['he']} ({entry['country_he']})")

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

        logger.info(f"✅ Loaded {len(CITY_TRANSLATIONS)} city entries from {file_path.name}.")

    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse {file_path.name}: {e}")
        raise

    except Exception as e:
        logger.error(f"❌ Unexpected error loading {file_path.name}: {e}")
        raise
    
def load_country_translations():
    global EN_TO_HE_COUNTRY
    try:
        with open(COUNTRY_TRANSLATIONS, encoding="utf-8") as f:
            EN_TO_HE_COUNTRY = json.load(f)
            logger.info(f"Loaded {len(EN_TO_HE_COUNTRY)} country translations from {COUNTRY_TRANSLATIONS.name}")
    except FileNotFoundError:
        logger.error(f"Translation file not found: {COUNTRY_TRANSLATIONS}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON in {COUNTRY_TRANSLATIONS}: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error loading country translations: {e}")
        
def get_git_version():
    """Return the project version based on Git commit and date, or 'dev' if unavailable."""
    root = os.path.dirname(os.path.abspath(__file__))

    def run_git_command(args):
        return subprocess.check_output(["git"] + args, cwd=root).decode().strip()

    try:
        commit = run_git_command(["rev-parse", "--short", "HEAD"])
        date = run_git_command(["log", "-1", "--format=%cd", "--date=short"])
        return f"{date.replace('-', '.')}–{commit}"

    except FileNotFoundError:
        logger.warning("[version] Git not found. Is it installed and in PATH? Falling back to 'dev'.")
    except subprocess.CalledProcessError as e:
        logger.warning(f"[version] Git command failed: {e}. Falling back to 'dev'.")
    except Exception as e:
        logger.warning(f"[version] Unexpected error: {e}. Falling back to 'dev'.")

    return "dev"
    
APP_VERSION = get_git_version()
TEMPLATES.env.globals["app_version"] = APP_VERSION

def update_travel_warnings():
    try:
        result = fetch_travel_warnings()
        if result:
            reload_travel_warnings_globals()
            logger.info(f"✅ Travel warnings updated and reloaded ({result['count']} records)")
        else:
            logger.warning("⚠️ fetch_travel_warnings returned None")
    except Exception:
        logger.exception("❌ Scheduled travel warnings update failed")


def reload_travel_warnings_globals():
    global TRAVEL_WARNINGS_DF
    try:
        TRAVEL_WARNINGS_DF = load_travel_warnings_df()
        logger.info(f"🧠 TRAVEL_WARNINGS_DF global updated (rows={len(TRAVEL_WARNINGS_DF)})")
    except Exception as e:
        logger.exception("❌ Failed to reload TRAVEL_WARNINGS_DF from cache")
        TRAVEL_WARNINGS_DF = pd.DataFrame()


def update_flights():
    try:
        fetch_israel_flights()
        reload_israel_flights_globals()
        logger.info("✅ Scheduled flight update completed.")
    except Exception as e:
        logger.exception("❌ Scheduled flight update failed.")
        
def reload_israel_flights_globals():
    global DATASET_DF, DATASET_DATE, DATASET_DF_FLIGHTS

    df, d = _read_dataset_file()
    DATASET_DF, DATASET_DATE = df, d or ""

    df_flights, _ = _read_flights_file()
    DATASET_DF_FLIGHTS = df_flights

    logger.info(f"🔁 Globals reloaded: {len(DATASET_DF)} dataset rows, {len(DATASET_DF_FLIGHTS)} flights")


def get_flight_time(dist_km: float | None) -> str:
    if not dist_km or dist_km <= 0:
        return "—"

    # Adjust speed based on flight range
    if dist_km < 500:
        cruise_speed_kmh = 700
        buffer = 0.4  # 24 mins
    elif dist_km < 2000:
        cruise_speed_kmh = 800
        buffer = 0.5
    else:
        cruise_speed_kmh = 850
        buffer = 0.6  # long-haul

    estimated_time_hr = dist_km / cruise_speed_kmh + buffer

    hours = int(estimated_time_hr)
    minutes = int(round((estimated_time_hr - hours) * 60))

    if minutes == 60:
        hours += 1
        minutes = 0

    return f"{hours}h" if minutes == 0 else f"{hours}h {minutes}m"


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

def normalize_case(value) -> str:
    """Capitalize each word safely, handling None, numbers, and placeholders."""
    if not value or str(value).strip() in {"", "—", "None", "nan"}:
        return "—"
    return string.capwords(str(value).strip())

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
    global TRAVEL_WARNINGS_DF

    try:
        while True:
            params = {
                "resource_id": TRAVEL_WARNINGS_RESOURCE,
                "limit": batch_size,
                "offset": offset,
            }
            logger.info(f"🌐 Fetching travel warnings batch offset={offset} ...")
            r = requests.get(TRAVEL_WARNINGS_API, params=params, timeout=30)
            r.raise_for_status()

            # ✅ Try normal JSON parsing, fallback to json-repair if malformed
            try:
                data = r.json()
            except JSONDecodeError as e:
                logger.warning(f"⚠️ Malformed JSON from gov.il travel warnings API: {e}")
                try:
                    fixed_json = repair_json(r.text)
                    data = json.loads(fixed_json)
                    logger.info("✅ JSON repaired successfully using json-repair")
                except Exception as repair_err:
                    logger.error(f"❌ JSON repair failed: {repair_err}", exc_info=True)
                    break  # exit pagination loop safely

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

        # Cache to disk
        try:
            with open(TRAVEL_WARNINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Cached {len(all_records)} travel warnings to disk")
        except Exception as e:
            logger.error(f"❌ Failed to write travel warnings to disk: {e}", exc_info=True)
            return None

        # Update DataFrame
        if all_records:
            TRAVEL_WARNINGS_DF = pd.DataFrame(all_records)
            TRAVEL_WARNINGS_DF.attrs["last_update"] = result["updated"]
            logger.info(f"🧠 TRAVEL_WARNINGS_DF updated with {len(TRAVEL_WARNINGS_DF)} rows")
        else:
            logger.warning("⚠️ No records fetched. Global TRAVEL_WARNINGS_DF was not updated.")

        logger.info(f"Travel warnings refreshed: {len(all_records)} total")
        return result

    except RequestException as e:
        logger.error(f"🚨 Request error fetching travel warnings: {e}", exc_info=True)
        return None
    except JSONDecodeError as e:
        logger.error(f"🚨 Failed to decode travel warnings JSON: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"❌ Failed to fetch travel warnings: {e}", exc_info=True)
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
    return round(geodesic((lat1, lon1), (lat2, lon2)).kilometers, 1)


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
def build_country_name_to_iso_map() -> dict[str, str]:
    """
    Build a robust mapping from country name variants to ISO alpha-2 codes.
    Includes official, common, short, and overridden names.
    """
    mapping = {}

    for country in pycountry.countries:
        names = {
            country.name.strip().lower(): country.alpha_2,
            country.alpha_2.strip().upper(): country.alpha_2
        }

        if hasattr(country, "official_name"):
            names[country.official_name.strip().lower()] = country.alpha_2

        # Add all name variants to mapping
        for k, v in names.items():
            mapping[k] = v

    # Add overrides and aliases (manually curated)
    overrides = {
        "usa": "US",
        "united states": "US",
        "united states of america": "US",
        "south korea": "KR",
        "north korea": "KP",
        "russia": "RU",
        "vietnam": "VN",
        "syria": "SY",
        "palestine": "PS",
        "iran": "IR",
        "uk": "GB",
        "united kingdom": "GB",
        "bolivia": "BO",
        "venezuela": "VE",
        "tanzania": "TZ",
        "moldova": "MD",
        "czech republic": "CZ",
        "ivory coast": "CI",
        "côte d’ivoire": "CI",
        "cote d'ivoire": "CI",
        "brunei": "BN",
        "laos": "LA",
        "myanmar": "MM",
        "macedonia": "MK",
        "north macedonia": "MK",
        "são tomé and príncipe": "ST",
        "sao tome and principe": "ST"
    }

    mapping.update(overrides)

    return mapping



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
    """
    Fetch gov.il flights (all countries) and cache to disk.
    Returns the data dict if successful, or None if failed or empty.
    """
    params = {
        "resource_id": RESOURCE_ID,
        "limit": DEFAULT_LIMIT
    }

    try:
        logger.info("🌐 Requesting flight data from gov.il API...")
        r = requests.get(ISRAEL_API, params=params, timeout=30)
        r.raise_for_status()

        # ✅ Try normal JSON first, fallback to json-repair if broken
        try:
            data = r.json()
        except JSONDecodeError as e:
            logger.warning(f"⚠️ Malformed JSON from gov.il API: {e}")
            try:
                fixed_json = repair_json(r.text)
                data = json.loads(fixed_json)
                logger.info("✅ JSON repaired successfully using json-repair")
            except Exception as repair_err:
                logger.error(f"❌ JSON repair failed: {repair_err}", exc_info=True)
                return None

        records = data.get("result", {}).get("records", [])
        logger.debug(f"🔍 API returned {len(records)} raw records")

        # Parse and normalize flight records
        flights = []
        for rec in records:
            iata = (rec.get("CHLOC1") or "—").strip().upper()
            direction = normalize_case(rec.get("CHAORD", "—"))

            if not iata or iata == "—" or not direction or direction == "—":
                continue  # Skip incomplete records
    
            flights.append({
                "airline": normalize_case(rec.get("CHOPERD", "—")),
                "iata": iata,
                "airport": normalize_case(rec.get("CHLOC1D", "—")),
                "city": normalize_case(rec.get("CHLOC1T", "—")),
                "country": normalize_case(rec.get("CHLOCCT", "—")),
                "scheduled": normalize_case(rec.get("CHSTOL", "—")),
                "actual": normalize_case(rec.get("CHPTOL", "—")),
                "direction": direction,
                "status": normalize_case(rec.get("CHRMINE", "—")),
            })

        if not flights:
            logger.warning("❌ No usable flight records received — skipping cache write.")
            return None

        df = pd.DataFrame(flights)
        if "iata" not in df.columns:
            logger.error("❌ 'iata' column missing in DataFrame — invalid records.")
            return None

        df["iata"] = (
            df["iata"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace("NAN", None)
        )
        flights = df.to_dict(orient="records")
    
        result = {
            "updated": datetime.now().isoformat(),
            "count": len(flights),
            "flights": flights
        }

        # Cache to disk
        try:
            with open(ISRAEL_FLIGHTS_FILE, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Cached {len(flights)} flight records to disk")
        except Exception as e:
            logger.error(f"❌ Failed to write flight data to disk: {e}", exc_info=True)
            return None

        return result

    except RequestException as e:
        logger.error(f"🚨 Request error fetching gov.il flights: {e}", exc_info=True)
        return None
    except JSONDecodeError as e:
        logger.error(f"🚨 Failed to decode gov.il API response: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error fetching gov.il flights: {e}", exc_info=True)
        return None

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
                    logger.info("✅ israel_flights.json repaired successfully using json-repair")

                    # Optional: rewrite the repaired version to disk
                    with open(ISRAEL_FLIGHTS_FILE, "w", encoding="utf-8") as fw:
                        json.dump(meta, fw, ensure_ascii=False, indent=2)
                        logger.info("💾 Repaired israel_flights.json saved to disk")

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
                    logger.info("✅ israel_flights.json repaired successfully using json-repair")

                    # Optional: persist repaired version
                    with open(ISRAEL_FLIGHTS_FILE, "w", encoding="utf-8") as fw:
                        json.dump(meta, fw, ensure_ascii=False, indent=2)
                        logger.info("💾 Repaired israel_flights.json saved to disk")
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

    logger.info(f"✅ /map rendered: {len(airports)} unique airports (merged by IATA)")

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
    
    logger.info("🚀 Application startup initiated")
    # 🎯 0) Set Git version
    logger.info(f"🔖 App Version: {APP_VERSION}")
    load_city_translations()
    load_country_translations()
    
    # 0) Load IATA DB once
    try:
        AIRPORTS_DB = load("IATA")
        logger.info(f"Loaded airportsdata IATA DB with {len(AIRPORTS_DB)} records")
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
        logger.info(f"Loaded {len(AIRLINE_WEBSITES)} airline websites")
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

    # 5) Load country → ISO code mapping
    try:
        COUNTRY_NAME_TO_ISO = build_country_name_to_iso_map()
        logger.info(f"Loaded {len(COUNTRY_NAME_TO_ISO)} country → ISO mappings")
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
            hours=8,
            id="warnings_refresh",
            replace_existing=True,
            next_run_time=datetime.now()
        )
        scheduler.add_job(
            sitemap,
            "interval",
            hours=24,
            id="static_regen",
            replace_existing=True,
            next_run_time=datetime.now())        
        scheduler.start()
        logger.info("✅ Scheduler started")
    except Exception as e:
        logger.critical("Failed to start scheduler", exc_info=True)

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

@app.get("/.well-known/traffic-advice", include_in_schema=False)
async def traffic_advice(request: Request):
    """Responds to Google's Traffic Advice probe requests."""
    ua = request.headers.get("user-agent", "").lower()

    # 🕵️‍♂️ If not Googlebot — silently ignore (prevents noise)
    if "googlebot" not in ua:
        return Response(status_code=204)  # No Content

    # 🟢 Log Googlebot probe once
    logger.info(f"✅ Googlebot traffic-advice request from {request.client.host}")

    # 🔵 Recommended JSON format (official spec)
    # Docs: https://developers.google.com/search/docs/crawling-indexing/traffic-advice
    return JSONResponse(
        content={
            "crawling": {"state": "allowed"}  # or "disallowed" if you want to throttle bots
        }
    )
    
class Url:
    def __init__(self, loc: str, lastmod: date, changefreq: str, priority: float):
        self.loc = loc
        self.lastmod = lastmod.isoformat()
        self.changefreq = changefreq
        self.priority = priority

# Helper to generate XML
def build_sitemap(urls: List[Url]) -> str:
    """Build a valid, deduplicated sitemap.xml from Url objects."""
    if not urls:
        return ""

    seen = set()
    unique_urls = []

    for u in urls:
        if u.loc not in seen:
            seen.add(u.loc)
            unique_urls.append(u)

    # Sort URLs for deterministic output (by loc)
    unique_urls.sort(key=lambda u: u.loc)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    ]

    for u in unique_urls:
        # Normalize lastmod → ISO 8601
        if isinstance(u.lastmod, (datetime, date)):
            lastmod_str = u.lastmod.strftime("%Y-%m-%d")
        else:
            lastmod_str = str(u.lastmod)

        # Escape any unsafe characters in loc
        loc_escaped = html.escape(u.loc, quote=True)

        lines.append("  <url>")
        lines.append(f"    <loc>{loc_escaped}</loc>")
        lines.append(f"    <lastmod>{lastmod_str}</lastmod>")
        lines.append(f"    <changefreq>{u.changefreq}</changefreq>")
        lines.append(f"    <priority>{float(u.priority):.1f}</priority>")
        lines.append("  </url>")

    lines.append("</urlset>")
    return "\n".join(lines)


@app.get("/sitemap.xml", response_class=Response, include_in_schema=False)
def sitemap():
    """Generate sitemap.xml including static pages (all on disk) and dynamic endpoints."""
    global STATIC_DIR, DATASET_DF
    base = "https://fly-tlv.com"
    today = date.today()

    # --- 1. Static base URLs ---
    urls = [
        Url(f"{base}/", today, "daily", 1.0),
        Url(f"{base}/stats", today, "daily", 1.0),
        Url(f"{base}/about", today, "yearly", 0.6),
        Url(f"{base}/direct-vs-nonstop", today, "yearly", 0.6),
        Url(f"{base}/privacy", today, "yearly", 0.5),
        Url(f"{base}/glossary", today, "yearly", 0.5),
        Url(f"{base}/contact", today, "yearly", 0.5),
        Url(f"{base}/accessibility", today, "yearly", 0.5),
        Url(f"{base}/terms", today, "yearly", 0.5),
        Url(f"{base}/map", today, "weekly", 0.7),
        Url(f"{base}/flights", today, "weekly", 0.7),
        Url(f"{base}/travel-warnings", today, "weekly", 0.7),
        Url(f"{base}/chat", today, "weekly", 0.8),
        # Hebrew
        Url(f"{base}/?lang=he", today, "daily", 1.0),
        Url(f"{base}/stats?lang=he", today, "daily", 1.0),
        Url(f"{base}/about?lang=he", today, "yearly", 0.6),
        Url(f"{base}/direct-vs-nonstop?lang=he", today, "yearly", 0.6),
        Url(f"{base}/accessibility?lang=he", today, "yearly", 0.5),
        Url(f"{base}/terms?lang=he", today, "yearly", 0.5),
        Url(f"{base}/privacy?lang=he", today, "yearly", 0.5),
        Url(f"{base}/contact?lang=he", today, "yearly", 0.5),
        Url(f"{base}/map?lang=he", today, "weekly", 0.7),
        Url(f"{base}/flights?lang=he", today, "weekly", 0.7),
        Url(f"{base}/travel-warnings?lang=he", today, "weekly", 0.7),
        Url(f"{base}/chat?lang=he", today, "weekly", 0.8),
        Url(f"{base}/glossary?lang=he", today, "yearly", 0.5),
    ]

    # --- 2. Add dynamic FastAPI destinations (live routes) ---
    try:
        for iata in DATASET_DF["IATA"].dropna().unique():
            iata = str(iata).strip()
            if iata:
                urls.append(Url(f"{base}/destinations/{iata}", today, "weekly", 0.7))
                urls.append(Url(f"{base}/destinations/{iata}?lang=he", today, "weekly", 0.7))
        logger.info(f"🧭 Added {len(DATASET_DF['IATA'].dropna().unique())} dynamic destinations.")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load dynamic IATA links: {e}")

    # --- 3. Include *all* static-generated HTML pages physically saved on disk ---
    static_dest_dir = STATIC_DIR / "destinations"
    if static_dest_dir.exists():
        logger.info(f"🗺️ Scanning static HTML destinations recursively from {static_dest_dir} ...")
        total_files = 0

        for lang_dir in static_dest_dir.iterdir():
            if not lang_dir.is_dir():
                continue

            lang_code = lang_dir.name  # e.g. "en" or "he"
            for html_file in lang_dir.rglob("*.html"):
                try:
                    iata = html_file.stem.upper()
                    lastmod = date.fromtimestamp(html_file.stat().st_mtime)

                    # Build the URL
                    if lang_code == "he":
                        url = f"{base}/destinations/{iata}?lang=he"
                    else:
                        url = f"{base}/destinations/{iata}"

                    urls.append(Url(url, lastmod, "weekly", 0.7))
                    total_files += 1

                    # 🪵 Detailed log line for each file
                    logger.info(f"📄 Added static file: {html_file.relative_to(STATIC_DIR)} → {url}")

                except Exception as e:
                    logger.warning(f"⚠️ Skipped file {html_file}: {e}")

        logger.info(f"✅ Added {total_files:,} static HTML destination files from {static_dest_dir}")
    else:
        logger.warning(f"⚠️ Static destinations folder not found: {static_dest_dir}")

    # --- 4. Build and write sitemap.xml ---
    xml = build_sitemap(urls)
    out_path = STATIC_DIR / "sitemap.xml"

    try:
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(xml, encoding="utf-8")
        logger.info(f"✅ Sitemap written to {out_path} with {len(urls)} URLs total")
    except Exception as e:
        logger.error(f"❌ Failed to write sitemap.xml: {e}")

    return Response(content=xml, media_type="application/xml")

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
            questions.append(f"What cities in {country} can I fly to?")
            questions.append(f"Which airlines fly to {country}?")

    if cities:
        for city, country in random.sample(cities, min(5, len(cities))):
            questions.append(f"Which airlines fly to {city}?")
            questions.append(f"What country is {city} located in?")

    if airlines:
        for airline in random.sample(airlines, min(5, len(airlines))):
            questions.append(f"Where does {airline} fly?")
            questions.append(f"What destinations are served by {airline}?")

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

                questions.append(f"אילו ערים יש טיסות ל-{he_country}?")
                questions.append(f"אילו חברות טסות ל-{he_country}?")
            except Exception as e:
                logger.debug(f"⚠️ Country translation fallback: {e}")
                continue

    if cities:
        for city, country in random.sample(cities, min(4, len(cities))):
            try:
                info = get_city_info(city, return_type="both")
                city_he = info.get("city_he") if info else city
                country_he = info.get("country_he") if info else country
                questions.append(f"אילו חברות טסות ל-{city_he}?")
                questions.append(f"באיזו מדינה נמצאת {city_he}? ({country_he})")
            except Exception as e:
                logger.debug(f"⚠️ City translation fallback: {e}")
                continue

    if airlines:
        for airline in random.sample(airlines, min(4, len(airlines))):
            questions.append(f"לאן טסה חברת {airline}?")
            questions.append(f"אילו ערים משרתת חברת {airline}?")

    random.shuffle(questions)
    logger.info(f"✅ Generated {len(questions[:n])} bilingual question suggestions (EN+HE).")
    return questions[:n]

def build_flight_context(df) -> str:
    """
    Build a rich Markdown-formatted flight context from the dataset.
    Supports:
      - City → Country lookups
      - Country → Cities lists
      - Airline → Destinations
      - Airline → Countries
      - Country → Destinations
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
    )

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

    logger.info(f"📥 User question: {question}")
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
- שאלות בעברית? → Translate to English first, then answer in English.

🧾 OUTPUT FORMAT (REQUIRED):
1. Start with a bold heading: **✈️ Flights to [Country/Region/City]**
2. Use Markdown bullet points:
   - Format each destination as: **City (IATA, Country)**
   - Under each, indent each airline as a separate sub-bullet:
     - One airline per line  
     - No commas, no inline lists, no slashes

🟢 CORRECT EXAMPLE:
---
**✈️ Flights to United States**
- Newark (EWR, United States)  
  - Delta Airlines  
  - El Al Israel Airlines  
  - Jetblue Airways Corporation  
  - United Airlines  

- New York (JFK, United States)  
  - Aero Mexico  
  - Arkia Israeli Airlines  
  - El Al Israel Airlines  
  - Jetblue Airways Corporation  
  - Virgin Atlantic Airways  
---

🚫 NEVER DO THIS:
- No commas in airline lists
- No paragraphs, summaries, or prose
- No tables or YAML
- No Hebrew in the output
- Do NOT say “Sure”, “Here’s the answer”, etc.

❌ IF THERE’S NO MATCH:
Reply ONLY with this exact line (no formatting):
I couldn't find it in our current destination catalog, please check the main table.
"""
    
    logger.debug("Gemini prompt built successfully (length=%d chars)", len(prompt))

    try:
        result = await chat_model.generate_content_async(prompt)
        answer = getattr(result, "text", None) or "Currently I have no answer"
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

    logger.info(f"GET /api/chat/suggestions → {len(suggestions)} suggestions")
    return {"questions": suggestions}

    
@app.get("/travel-warnings", response_class=HTMLResponse)
async def travel_warnings_page(request: Request, lang: str = Depends(get_lang)):
    global TRAVEL_WARNINGS_DF

    client_host = request.client.host if request.client else "unknown"

    if TRAVEL_WARNINGS_DF is None or TRAVEL_WARNINGS_DF.empty:
        logger.warning(f"GET /travel-warnings from {client_host} (lang={lang}) → no cached data")
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
    logger.info(f"✅ Loaded total flights: {len(processed_flights)} (no filtering applied)")

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
@app.get("/glossary", response_class=HTMLResponse)
async def glossary_view(request: Request):
    lang = request.query_params.get("lang", "en").lower()

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
    """Render destination page and export it to static HTML for SEO"""
    global DATASET_DF, COUNTRY_NAME_TO_ISO, AIRLINE_WEBSITES, STATIC_DIR

    lang = request.query_params.get("lang", "en").lower()
    iata = iata.upper().strip()

    # ✅ Redirect if TLV itself is requested
    if iata in {"TLV", "LLBG", "BEN-GURION"}:
        logger.info(f"🔁 Redirected attempt to access TLV itself → homepage")
        target = "/" if lang == "en" else "/?lang=he"
        return RedirectResponse(url=target, status_code=302)


    # ✅ 1. Validate dataset
    if DATASET_DF is None or DATASET_DF.empty:
        return TEMPLATES.TemplateResponse("error.html", {
            "request": request,
            "message": "No destination data available.",
            "lang": lang
        })

    iata = iata.upper()
    destination = DATASET_DF.loc[DATASET_DF["IATA"] == iata]

    if destination.empty:
        return TEMPLATES.TemplateResponse("error.html", {
            "request": request,
            "message": f"Destination {iata} not found from Tel-Aviv (TLV).",
            "lang": lang
        })

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
    MAX_AGE = 86400  # 1 day
    if output_file.exists():
        mtime = output_file.stat().st_mtime
        if time.time() - mtime < MAX_AGE:
            logger.info(f"🗂️ Using cached static page: {output_file}")
            return HTMLResponse(content=output_file.read_text(encoding='utf-8'))
        else:
            logger.info(f"♻️ Rebuilding expired page for {iata}")

    # ✅ 5. Fetch travel info internally (from your existing route)
   # ✅ fetch travel info
    travel_info_data = None
    wiki_summary_data = None
    city_name = dest.get("City", "").strip()

    if city_name:
        # Fetch travel info
        try:
            travel_info_data = await get_travel_info(city_name, lang=lang)
            logger.info(f"✅ Travel info loaded for {city_name}")
        except Exception as e:
            logger.warning(f"⚠️ Travel info unavailable for {city_name}: {e}")
            travel_info_data = None

        # Fetch Wikipedia summary
        try:
            wiki_summary_data = await fetch_wikipedia_summary(city_name, lang)
            if wiki_summary_data:
                logger.info(f"✅ Wikipedia summary fetched for {city_name}")
            else:
                logger.warning(f"⚠️ No Wikipedia summary for {city_name}")
        except Exception as e:
            logger.warning(f"⚠️ Wikipedia summary unavailable for {city_name}: {e}")
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
        output_file.write_text(rendered_html, encoding="utf-8")
        logger.info(f"🌍 Static page generated: {output_file}")
    except Exception as e:
        logger.error(f"⚠️ Failed to write static HTML for {iata}: {e}")

    return HTMLResponse(content=rendered_html)



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
    
@app.get("/sw.js", include_in_schema=False)
async def service_worker():
    js = """
    self.addEventListener("install", (event) => {
        console.log("Service Worker installed");
        self.skipWaiting();
    });

    self.addEventListener("activate", (event) => {
        console.log("Service Worker activated");
        event.waitUntil(clients.claim());
    });

    // Minimal fetch handler (passthrough to network)
    self.addEventListener("fetch", (event) => {
        event.respondWith(fetch(event.request));
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
    client = request.client.host
    try:
        logger.info(f"GET /direct-vs-nonstop | lang={lang} | client={client}")
        return TEMPLATES.TemplateResponse("direct_vs_nonstop.html", {
            "request": request,
            "lang": lang,
            "now": datetime.now()
        })
    except Exception as e:
        logger.error(f"❌ Failed to render direct_vs_nonstop.html | {e} | client={client}")
        return TEMPLATES.TemplateResponse("error.html", {
            "request": request,
            "lang": lang,
            "message": (
                "The Direct vs Nonstop page could not be loaded. Please try again later."
                if lang != "he"
                else "לא ניתן לטעון את עמוד 'ישיר מול ללא-עצירות'. נסה שוב מאוחר יותר."
            )
        }, status_code=500)
        
@app.get("/manifest.json", include_in_schema=False)
async def manifest(request: Request):
    lang = request.query_params.get("lang", "en").lower()
    if lang not in ("en", "he"):
        lang = "en"

    client = request.client.host if request.client else "unknown"
    base_path = Path(__file__).parent
    manifest_en = base_path / "manifest.json"
    manifest_he = base_path / "manifest.he.json"

    selected_manifest = manifest_he if lang == "he" and manifest_he.exists() else manifest_en

    if selected_manifest.exists():
        logger.info(f"📄 GET /manifest.json | lang={lang} | client={client} | file={selected_manifest.name}")
        return FileResponse(selected_manifest, media_type="application/manifest+json")
    else:
        logger.error(f"❌ Manifest not found | lang={lang} | path={selected_manifest}")
        return JSONResponse(
            {"error": "Manifest not found", "lang": lang},
            status_code=404
        )
   
@app.middleware("http")
async def redirect_and_log_404(request: Request, call_next):
    host_header = request.headers.get("host", "").lower()
    hostname = (request.url.hostname or "").lower()
    client_host = (request.client.host or "").lower()
    path = request.url.path

    # 🚫 0. Block obvious malicious paths (before anything else)
    suspicious_patterns = (".env", ".git", "phpinfo", "config", "composer.json", "wp-admin", "shell", "eval(")
    if any(p in path.lower() for p in suspicious_patterns):
        logger.warning(f"🚫 Blocked suspicious request from {client_host} → {path}")
        return JSONResponse({"detail": "Forbidden"}, status_code=403)

    # 🚫 Skip redirects for localhost or internal testing
    if any(kw in host_header for kw in ("localhost", "127.0.0.1", "::1")) \
       or hostname in ("localhost", "127.0.0.1", "::1") \
       or client_host in ("localhost", "127.0.0.1", "::1"):
        response = await call_next(request)
        if response.status_code == 404 and not path.startswith("/%23"):
            logger.error(f"⚠️ 404 (dev) from {client_host} → {path}")
        return response
    
    # 🚫 Skip redirect logic for known static endpoints
    if path in ("/favicon.ico", "/favicon.svg", "/robots.txt", "/sitemap.xml"):
        return await call_next(request)
    # 🌍 Production: clean & normalize URLs
    url = str(request.url)
    redirect_url = url

    # ✅ 1. Enforce HTTPS
    if url.startswith("http://"):
        redirect_url = redirect_url.replace("http://", "https://", 1)

    # ✅ 2. Remove 'www.'
    if "://www." in redirect_url:
        redirect_url = redirect_url.replace("://www.", "://", 1)

    # ✅ 3. Handle malformed encoded fragments (e.g. /%23c)
    if "/%23" in url or path.startswith("/#") or "%23" in path:
        clean_base = url.split("/%23")[0]
        logger.error(f"🧹 Cleaning malformed anchor → redirecting {path} → {clean_base}")
        return RedirectResponse(url=clean_base, status_code=301)

    # ✅ 4. Trailing slash normalization (SEO-friendly)
    # Don't strip for known dynamic paths like /destinations/{iata}
    if (
        path.endswith("/") 
        and len(path) > 1
        and not (
            path.startswith("/static") or 
            path.startswith("/.well-known") or 
            re.match(r"^/destinations/[A-Z]{3}/?$", path, re.IGNORECASE)
        )
    ):
        redirect_url = redirect_url.rstrip("/")


    # Redirect only if changed
    if redirect_url != url:
        logger.info(f"🔁 Redirecting {url} → {redirect_url}")
        return RedirectResponse(url=redirect_url, status_code=301)

    # 🧩 Continue normally
    response = await call_next(request)

    # ⚠️ Log real 404s only (ignore bots hitting /%23 junk)
    if response.status_code == 404 and not path.startswith("/%23"):
        logger.info(f"⚠️ 404 from {client_host} → {path}")

    return response

    
@app.get("/api/refresh-data", response_class=JSONResponse)
async def refresh_data_webhook():
    global DATASET_DF, DATASET_DATE, DATASET_DF_FLIGHTS

    logger.info("🔁 Incoming request: /api/refresh-data")

    response = {
        "fetch_israel_flights": None,
        "fetch_travel_warnings": None
    }

    # Run each fetch with logging
    try:
        res1 = fetch_israel_flights()
        if res1:
            logger.info("✅ fetch_israel_flights completed successfully")
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
            logger.info("✅ fetch_travel_warnings completed successfully")
            response["fetch_travel_warnings"] = "Success"
        else:
            logger.error("❌ fetch_travel_warnings returned None")
            response["fetch_travel_warnings"] = "Failed: returned None"
    except Exception as e:
        logger.exception("❌ Exception in fetch_travel_warnings")
        response["fetch_travel_warnings"] = f"Exception: {str(e)}"

    logger.info("🔁 Refresh summary: %s", json.dumps(response, indent=2, ensure_ascii=False))
    return response


@app.get("/terms", response_class=HTMLResponse)
async def terms_view(request: Request):
    lang = request.query_params.get("lang", "en").lower()

    try:
        return TEMPLATES.TemplateResponse("terms.html", {
            "request": request,
            "lang": lang
        })
    except Exception as e:
        return TEMPLATES.TemplateResponse("error.html", {
            "request": request,
            "message": (
                "Terms & Conditions page could not be loaded."
                if lang != "he"
                else "לא ניתן לטעון את עמוד התנאים וההגבלות."
            ),
            "lang": lang
        })
        
@app.get("/feed.xml", response_class=Response)
def flight_feed():
    now = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000")
    items = []

    for _, row in DATASET_DF.iterrows():
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
            f"Direction: {'Arrival' if row['Direction'] == 'A' else 'Departure'}<br>"
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
        # Optionally rename for frontend expectations
        warnings = warnings.rename(columns={"details_url": "link"})
        return JSONResponse(content={"warnings": warnings.to_dict(orient="records")})

    return JSONResponse(content={"warnings": []})

@app.get("/api/wiki-summary")
async def fetch_wikipedia_summary(
    city: str = Query(..., description="City name in English (or Hebrew if lang=he)"),
    lang: str = Query("en", pattern="^(en|he)$", description="Language: en or he")
):
    """
    🔹 Fetch summarized Wikipedia info for a city (supports English & Hebrew).
    - If lang=he, it will try to translate the city name using get_city_info().
    - Returns title, description, extract, thumbnail, and page URL.
    """
    try:
        city = city.strip().title()

        # 🈁 Hebrew translation (if requested)
        if lang == "he":
            translated_city = get_city_info(city, return_type="city")
            if translated_city:
                logger.info(f"🌐 Translating '{city}' → '{translated_city}' for Hebrew Wikipedia lookup.")
                city = translated_city
            else:
                logger.warning(f"⚠️ No Hebrew translation found for '{city}', using English name.")

        url = WIKI_API_BASE.format(lang=lang, city=city.replace(" ", "_"))
        logger.info(f"🌍 Fetching Wikipedia summary for '{city}' ({lang}) — {url}")

        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url, headers={
                "User-Agent": "Fly-TLV (https://fly-tlv.com; contact@fly-tlv.com)",
                "Accept": "application/json"
            })

        # 🧭 Handle API responses
        if response.status_code == 404:
            logger.warning(f"⚠️ Wikipedia article not found for '{city}' ({lang})")
            raise HTTPException(status_code=404, detail=f"No Wikipedia article for {city} ({lang})")
        elif response.status_code != 200:
            logger.error(f"❌ Wikipedia API returned {response.status_code} for '{city}'")
            raise HTTPException(status_code=response.status_code, detail="Wikipedia API error")

        data = response.json()

        result = {
            "title": data.get("title"),
            "description": data.get("description"),
            "extract": data.get("extract"),
            "thumbnail": data.get("thumbnail", {}).get("source"),
            "lang": lang,
            "url": data.get("content_urls", {}).get(lang, {}).get("page"),
            "requested_city": city
        }

        if not result["extract"]:
            logger.warning(f"⚠️ Wikipedia summary missing text for '{city}' ({lang})")
            raise HTTPException(status_code=204, detail="No summary available")
        
        return result

    except httpx.TimeoutException:
        logger.error(f"⏱️ Timeout fetching Wikipedia for '{city}' ({lang})")
        raise HTTPException(status_code=504, detail="Wikipedia request timed out")

    except HTTPException:
        raise  # Already logged

    except Exception as e:
        logger.exception(f"💥 Unexpected error fetching Wikipedia for '{city}' ({lang}): {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/log-click")
async def log_click(request: Request):
    data = await request.json()
    click_type = data.get("type")

    if click_type == "airline":
        airline = data.get("airline")
        logger.info(f"🛫 Airline chip clicked: {airline}")

    elif click_type == "destination":
        iata = data.get("iata")
        airport = data.get("airport")
        logger.info(f"🌍 Destination link clicked: {airport} ({iata})")

    else:
        logger.warning(f"⚠️ Unknown click log type: {data}")

    return {"status": "ok"}

SPARQL_URL = "https://query.wikidata.org/sparql"

ALLOWED_ROOTS = [
    "wd:Q4989906",   # tourist attraction
    "wd:Q33506",     # attraction
    "wd:Q839954",    # landmark
    "wd:Q5084",      # archaeological site
    "wd:Q9134",      # historic site
    "wd:Q35509",     # monument
    "wd:Q570116",    # museum
    "wd:Q12973014",  # art gallery
    "wd:Q22698",     # park
    "wd:Q123705",    # square
    "wd:Q16970",     # cathedral
    "wd:Q23413",     # church
    "wd:Q2065736",   # temple
    "wd:Q875157",    # statue
    "wd:Q811979",    # viewpoint
]

ROOT_VALUES = " ".join(ALLOWED_ROOTS)


async def get_travel_info(city: str, lang="en"):
    HEADERS = {
    "User-Agent": "FlyTLV/1.0 (contact@fly-tlv.com)",
    "Accept": "application/json"}
    
    city = city.strip().title()

    async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:

        # --------------------------------------------------------
        # 1) SEARCH CITY → GET QID
        # --------------------------------------------------------
        r = await client.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": city,
                "language": "en",
                "type": "item",
                "limit": 1,
                "format": "json",
            }
        )

        data = r.json()
        if not data.get("search"):
            logger.warning(f"❌ No QID found for {city}")
            return {"city": city, "pois": [], "tips": []}

        city_qid = data["search"][0]["id"]
        logger.info(f"🔎 City QID = {city_qid}")

        # --------------------------------------------------------
        # 2) SPARQL → Extended + DISTINCT + image + desc
        # --------------------------------------------------------
        query = f"""
        SELECT DISTINCT ?item ?itemLabel ?coord ?image ?desc ?sitelinks WHERE {{
          VALUES ?root {{ {ROOT_VALUES} }}

          ?item wdt:P131 wd:{city_qid} .
          ?item wdt:P31/wdt:P279* ?root .

          OPTIONAL {{ ?item wdt:P625 ?coord }}
          OPTIONAL {{ ?item wdt:P18 ?image }}
          OPTIONAL {{ ?item schema:description ?desc FILTER(LANG(?desc)="{lang}") }}
          OPTIONAL {{ ?item wikibase:sitelinks ?sitelinks }}

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang},en". }}
        }}
        ORDER BY DESC(?sitelinks)
        LIMIT 100
        """

        r = await client.post(
            SPARQL_URL,
            params={"format": "json"},
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
        )

        ctype = r.headers.get("content-type", "").lower()
        if "json" not in ctype:
            logger.error("❌ SPARQL returned NON-JSON content!")
            logger.error(r.text[:600])
            return {"city": city, "pois": [], "tips": []}

        results = r.json().get("results", {}).get("bindings", [])
        pois = []
        seen_keys = set()  # (name, lat, lon) deduplication

        # --------------------------------------------------------
        # 3) PARSE + FILTER + DEDUPLICATE
        # --------------------------------------------------------
        for row in results:
            name = row.get("itemLabel", {}).get("value")
            desc = row.get("desc", {}).get("value")
            img = row.get("image", {}).get("value")

            if not name or not desc or not img:
                continue  # skip if missing any required field

            coord_raw = row.get("coord", {}).get("value", "")
            lat = lon = None
            if coord_raw.startswith("Point("):
                lon, lat = coord_raw[6:-1].split()
                lat, lon = float(lat), float(lon)
            else:
                continue  # skip if no coordinates

            # Deduplicate by (name, lat, lon)
            key = (name.lower(), round(lat, 5), round(lon, 5))
            if key in seen_keys:
                continue
            seen_keys.add(key)

            pois.append({
                "name": name,
                "description": desc,
                "image": img,
                "lat": lat,
                "lon": lon,
                "gmap_url": f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
            })

        logger.info(f"📌 Filtered down to {len(pois)} top tourist attractions for {city}")
        random.shuffle(pois)
        pois = pois[:20]
        return {
            "city": city,
            "pois": pois,
            "tips": []
        }


