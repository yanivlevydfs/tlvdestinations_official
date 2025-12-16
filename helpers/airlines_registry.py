import json
import httpx
import logging
import re
from config_paths import ISRAEL_FLIGHTS_FILE
from config_paths import DATA_DIR
import app_state
import time
from pathlib import Path

logger = logging.getLogger("airlines_registry")

AIRLINES_URL = "https://api.travelpayouts.com/data/en/airlines.json"
AIRLINES_FILE = DATA_DIR / "airlines_all.json"

IATA_RE = re.compile(r"^[A-Z0-9]{2}$")

AIRLINE_BRANDS = {
    "wizzair": {
        "wizz air",
        "wizz air uk",
        "wizz air malta",
        "wizz air abu dhabi",
        "wizz",
    },
    "ryanair": {
        "ryanair",
        "ryanair uk",
        "buzz",
        "malta air",
    },
    "easyjet": {
        "easyjet",
        "easyjet europe",
        "easyjet switzerland",
    },
    "transavia": {
        "transavia",
        "transavia france",
    },
    "lufthansa": {
        "lufthansa",
        "lufthansa cityline",
    },
    "elal": {
        "el al",
        "elal",
        "el al israel airlines",
    },
}


STOP_WORDS = {
    "air", "airlines", "airways", "airline",
    "international", "intl",
    "ltd", "inc", "llc", "plc", "sa", "ag",
    "company", "group", "holding", "holdings",
    "dba",
    "uk", "eu", "europe", "malta", "france", "germany",
    "israel", "switzerland", "portugal"
}

def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()
    
def normalize_airline_brand(name: str) -> str:
    if not name:
        return ""

    s = name.lower()
    s = re.sub(r"[^\w\s]", " ", s)   # remove punctuation
    tokens = [
        t for t in s.split()
        if t not in STOP_WORDS and len(t) > 1
    ]

    return " ".join(tokens)

def resolve_brand(name: str) -> str:
    n = normalize_name(name)

    for brand, variants in AIRLINE_BRANDS.items():
        for v in variants:
            if v in n:
                return brand

    # fallback: first token (safe default)
    return n.split()[0] if n else ""

def _is_valid_iata(code: str) -> bool:
    return bool(code and IATA_RE.match(code))


# ⏱️ Cache TTL: 7 days
AIRLINES_CACHE_TTL = 7 * 24 * 60 * 60  # seconds


def _is_cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < AIRLINES_CACHE_TTL


def download_airlines_list(force: bool = False) -> list[dict]:
    """
    Downloads and caches worldwide airline registry (Travelpayouts).
    File-based cache with TTL.
    """

    # ✅ Serve from cache
    if not force and _is_cache_valid(AIRLINES_FILE):
        try:
            with open(AIRLINES_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Airlines cache corrupted, reloading: {e}")

    logger.info("⬇️ Downloading worldwide airlines list")

    with httpx.Client(timeout=30) as client:
        r = client.get(AIRLINES_URL)
        r.raise_for_status()
        raw = r.json()  # LIST

    airlines: list[dict] = []

    for a in raw:
        code = str(a.get("code", "")).upper()

        # 🚫 Filter invalid / non-IATA / unicode garbage
        if not _is_valid_iata(code):
            continue

        airlines.append({
            "iata": code,
            "name": a.get("name"),
            "name_en": a.get("name_translations", {}).get("en"),
            "is_lowcost": bool(a.get("is_lowcost")),
        })

    # 💾 Write cache atomically
    tmp_file = AIRLINES_FILE.with_suffix(".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(airlines, f, ensure_ascii=False, indent=2)
    tmp_file.replace(AIRLINES_FILE)

    logger.info(f"✔ Cached {len(airlines)} valid airlines")

    return airlines


def get_active_tlv_airlines() -> list[dict]:
    if app_state.DATASET_DF_FLIGHTS.empty:
        logger.warning("DATASET_DF_FLIGHTS empty — no TLV airlines")
        return []

    df = app_state.DATASET_DF_FLIGHTS

    # 1️⃣ Extract TLV airline BRANDS from flight data
    tlv_brands: set[str] = set()

    if "airline" in df.columns:
        for raw_name in df["airline"].dropna().astype(str):
            brand = resolve_brand(raw_name)
            if brand:
                tlv_brands.add(brand)

    if not tlv_brands:
        logger.warning("No airline brands extracted from TLV flights")
        return []

    logger.info(f"Detected TLV airline brands: {sorted(tlv_brands)}")

    # 2️⃣ Load worldwide airline registry
    airlines_all = download_airlines_list()

    active: list[dict] = []
    seen_brands: set[str] = set()

    # 3️⃣ Match registry airlines → TLV brands
    for a in airlines_all:
        name = a.get("name") or a.get("name_en")
        if not name:
            continue

        brand = resolve_brand(name)

        # ✅ brand match + deduplicate
        if brand in tlv_brands and brand not in seen_brands:
            a = dict(a)               # do not mutate cache
            a["brand"] = brand
            active.append(a)
            seen_brands.add(brand)

    return sorted(active, key=lambda x: x["brand"])



