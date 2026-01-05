# config_paths.py
# -----------------------------------------------------
# Shared filesystem paths used across the application.
# Safe to import from helpers (no circular imports).
# -----------------------------------------------------

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Base directories
# ------------------------------------------------------------------

BASE_DIR: Path = Path(__file__).resolve().parent
CACHE_DIR: Path = BASE_DIR / "cache"
STATIC_DIR: Path = BASE_DIR / "static"
TEMPLATES_DIR: Path = BASE_DIR / "templates"
DATA_DIR: Path = BASE_DIR / "data"

# ------------------------------------------------------------------
# Static sub-structure (used by sitemap & pre-rendering)
# ------------------------------------------------------------------
STATIC_DESTINATIONS_DIR = STATIC_DIR / "destinations"
STATIC_DESTINATIONS_EN = STATIC_DESTINATIONS_DIR / "en"
STATIC_DESTINATIONS_HE = STATIC_DESTINATIONS_DIR / "he"

# ------------------------------------------------------------------
# Ensure directories exist (deployment-safe)
# ------------------------------------------------------------------
for d in (
    CACHE_DIR,
    STATIC_DIR,
    STATIC_DESTINATIONS_DIR,
    STATIC_DESTINATIONS_EN,
    STATIC_DESTINATIONS_HE,
    TEMPLATES_DIR,
    DATA_DIR,
):
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)
        logger.info("Created directory: %s", d)

# Data files
AIRLINE_WEBSITES_FILE = DATA_DIR / "airline_websites.json"
ISRAEL_FLIGHTS_FILE   = CACHE_DIR / "israel_flights.json"
TRAVEL_WARNINGS_FILE  = CACHE_DIR / "travel_warnings.json"
COUNTRY_TRANSLATIONS  = DATA_DIR / "country_translations.json"
CITY_TRANSLATIONS_FILE = DATA_DIR / "city_translations.json"
CITY_NAME_CORRECTIONS_FILE = DATA_DIR / "city_name_corrections.json"

