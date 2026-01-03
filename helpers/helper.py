# helper.py
"""
HTML parsing & extraction helpers — Production grade.
"""
from fastapi import Request
import re
from bs4 import BeautifulSoup
import subprocess
import os
import string
from datetime import datetime
from geopy.distance import geodesic
import pycountry
from typing import List, Dict
import shutil
from config_paths import CACHE_DIR
import logging

_COUNTRY_CACHE: dict[str, str] | None = None

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
    
def get_git_version():
    sha = os.getenv("RAILWAY_GIT_COMMIT_SHA")
    return sha[:7] if sha else "dev"

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
    
def datetimeformat(value: str, fmt: str = "%d/%m/%Y %H:%M"):
    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime(fmt)
    except Exception:
        return value
        

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
    
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return round(geodesic((lat1, lon1), (lat2, lon2)).kilometers, 1)
    
    
def format_time(dt_string):
    """Return (short, full, raw_iso) for datetime strings."""

    # ---- FIX: Safely handle float/None/NaN ----
    if dt_string is None or isinstance(dt_string, float):
        return "—", "—", ""

    dt_string = str(dt_string).strip()

    if dt_string in {"—", "", "nan", "None"}:
        return "—", "—", ""

    # -------------------------------------------

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
    
def build_country_name_to_iso_map() -> dict[str, str]:
    """
    Cached: Build a mapping from country name variants to ISO alpha-2 codes.
    Loaded once and reused.
    """
    global _COUNTRY_CACHE

    if _COUNTRY_CACHE is not None:
        return _COUNTRY_CACHE   # ⚡ instant return, no processing

    mapping = {}

    for country in pycountry.countries:
        try:
            names = {
                country.name.strip().lower(): country.alpha_2,
                country.alpha_2.strip().upper(): country.alpha_2,
            }

            if hasattr(country, "official_name") and country.official_name:
                names[country.official_name.strip().lower()] = country.alpha_2

            for k, v in names.items():
                mapping[k] = v

        except Exception:
            continue

    # Manual overrides
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
        "sao tome and principe": "ST",
    }

    mapping.update(overrides)

    _COUNTRY_CACHE = mapping  # 💾 save in cache

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
    
def cleanup_local_cache():
    logger = logging.getLogger("cache_cleanup")

    # ============================================================
    # 🚫 NEVER clean cache on cloud platforms
    # ============================================================
    if (
        os.getenv("RENDER")
        or os.getenv("RAILWAY_ENVIRONMENT")
        or os.getenv("RAILWAY_PROJECT_ID")
        or os.getenv("FLY_ENV") == "prod"
    ):
        logger.debug("Cache cleanup skipped (cloud environment detected)")
        return

    # ============================================================
    # ✅ Localhost only
    # ============================================================
    if not CACHE_DIR.exists():
        logger.debug("Cache directory does not exist — nothing to clean")
        return

    logger.warning("🧹 Local development detected — clearing CACHE directory")

    for item in CACHE_DIR.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

            logger.warning(f"🗑️ Removed cache item: {item.name}")

        except Exception:
            logger.error(
                f"❌ Failed to remove cache item {item}",
                exc_info=True
            )