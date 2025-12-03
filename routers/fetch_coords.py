from fastapi import APIRouter, Query
import httpx
import logging
from functools import lru_cache

router = APIRouter()
logger = logging.getLogger("fetch_coords")

# ============================================================
# CACHE
# ============================================================
@lru_cache(maxsize=3000)
def cache_key(city: str, lang: str):
    return f"{city.lower()}::{lang}"

CACHE = {}


def set_cache(city, lang, data):
    CACHE[cache_key(city, lang)] = data


def get_cache(city, lang):
    return CACHE.get(cache_key(city, lang))


# ============================================================
# MAIN ROUTE
# ============================================================
@router.get("/api/fetch_coords")
async def api_travel_info(
    city: str = Query(...),
    lang: str = Query("en")
):
    """
    Returns:
      - city coordinates from Wikipedia
      - supports Hebrew lookup & EN fallback
    """

    city_clean = city.strip()
    logger.info(f"🌍 /api/fetch_coords city={city_clean}, lang={lang}")

    # ===== 0️⃣ check cache =====
    cached = get_cache(city_clean, lang)
    if cached:
        logger.info(f"⚡ cache hit for {city_clean}/{lang}")
        return cached

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

    async def fetch_coords(api, lookup):
        try:
            return await client.get(
                api,
                params={
                    "action": "query",
                    "format": "json",
                    "prop": "coordinates",
                    "redirects": 1,
                    "titles": lookup,
                },
            )
        except Exception as e:
            logger.error(f"❌ fetch_coords failed: {e}")
            return None

    # ===== RUN HTTP =====
    async with httpx.AsyncClient(timeout=timeout, headers=HEADERS) as client:

        geo_resp = await fetch_coords(WIKI_API, city_lookup)

        # ===== 4️⃣ fallback to English =====
        if (
            not geo_resp
            or not geo_resp.json().get("query", {}).get("pages")
        ) and FALLBACK_API:
            logger.warning("⚠️ Hebrew page missing → English fallback")
            geo_resp = await fetch_coords(FALLBACK_API, city_clean)

    # ============================================================
    # 5️⃣ PARSE RESULT
    # ============================================================
    coords = None
    pages = geo_resp.json().get("query", {}).get("pages", {})
    for _, page in pages.items():
        if "coordinates" in page:
            c = page["coordinates"][0]
            coords = {"lat": c["lat"], "lon": c["lon"]}
            break

    if not coords:
        logger.error(f"❌ No coordinates found for: {city_lookup}")
        result = {"city": city_clean, "ok": False, "coords": None}
        set_cache(city_clean, lang, result)
        return result

    result = {
        "city": city_clean,
        "lookup": city_lookup,
        "coords": coords,
        "ok": True
    }

    # save cache
    set_cache(city_clean, lang, result)

    return result
