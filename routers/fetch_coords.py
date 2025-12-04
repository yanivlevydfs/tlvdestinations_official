from fastapi import APIRouter, Query, HTTPException
import httpx
import logging
import time

router = APIRouter()
logger = logging.getLogger("fetch_coords")

# ============================================================
# ⚡ TTL CACHE (FAST, SAFE, PRODUCTION)
# ============================================================

CACHE = {}
CACHE_TTL = 5 * 24 * 3600   # 5 days


def cache_get(city: str, lang: str):
    key = f"{city.lower()}::{lang}"
    item = CACHE.get(key)
    if not item:
        return None
    value, ts = item
    if time.time() - ts < CACHE_TTL:
        return value
    del CACHE[key]
    return None


def cache_set(city: str, lang: str, data):
    key = f"{city.lower()}::{lang}"
    CACHE[key] = (data, time.time())


# ============================================================
# 🌍 WIKIPEDIA & WIKIDATA HELPERS
# ============================================================

WIKI_API = "https://{lang}.wikipedia.org/w/api.php"
WIKI_REST = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{page}"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"


async def fetch_wikipedia_coords(client, lang: str, title: str):
    """Return coordinates from Wikipedia page, or None."""
    try:
        r = await client.get(
            WIKI_API.format(lang=lang),
            params={
                "action": "query",
                "format": "json",
                "prop": "coordinates|pageprops",
                "redirects": 1,
                "titles": title,
            }
        )
        data = r.json()
        pages = data.get("query", {}).get("pages", {})

        for _, p in pages.items():
            # direct coordinates
            coords = p.get("coordinates")
            if coords:
                c = coords[0]
                return {"lat": c["lat"], "lon": c["lon"]}

            # maybe Wikidata ID is available
            wikidata_id = p.get("pageprops", {}).get("wikibase_item")
            if wikidata_id:
                coords = await fetch_wikidata_coords(client, wikidata_id)
                if coords:
                    return coords

    except Exception as e:
        logger.error(f"❌ Wikipedia coordinate fetch failed for {title}: {e}")

    return None


async def fetch_wikidata_coords(client, entity_id: str):
    """Fetch coordinates from Wikidata (MUCH more accurate & reliable)."""
    try:
        r = await client.get(
            WIKIDATA_API,
            params={
                "action": "wbgetentities",
                "format": "json",
                "ids": entity_id,
                "props": "claims",
            }
        )
        data = r.json()
        entity = data.get("entities", {}).get(entity_id, {})
        claims = entity.get("claims", {})

        # P625 = geocoordinates
        if "P625" in claims:
            val = claims["P625"][0]["mainsnak"]["datavalue"]["value"]
            return {"lat": val["latitude"], "lon": val["longitude"]}

    except Exception as e:
        logger.error(f"❌ Wikidata fetch failed ({entity_id}): {e}")

    return None


# ============================================================
# 🚀 MAIN ROUTE
# ============================================================

@router.get("/api/fetch_coords")
async def api_fetch_coords(
    city: str = Query(..., description="City name in English"),
    lang: str = Query("en", pattern="^(en|he)$")
):
    """
    Production-grade coordinate lookup:
    - Hebrew → EN fallback
    - Wikipedia + Wikidata fallback
    - Caching for 5 days
    """

    city_clean = city.strip().title()
    logger.info(f"🌍 /api/fetch_coords city={city_clean}, lang={lang}")

    # ===== 0️⃣ CHECK CACHE =====
    cached = cache_get(city_clean, lang)
    if cached:
        logger.debug(f"⚡ Cache hit for {city_clean}/{lang}")
        return cached

    # ===== 1️⃣ PREPARE LOOKUP =====
    if lang == "he":
        # optional: translation
        try:
            from helpers.helper import get_city_info
            translated = get_city_info(city_clean, return_type="city")
            lookup_title = translated or city_clean
        except Exception:
            lookup_title = city_clean
    else:
        lookup_title = city_clean

    timeout = httpx.Timeout(connect=5, read=10, write=5, pool=5)
    HEADERS = {"User-Agent": "FlyTLV/2.0 (contact@fly-tlv.com)"}

    async with httpx.AsyncClient(timeout=timeout, headers=HEADERS) as client:

        # ===== 2️⃣ FIRST TRY: primary language (he or en) =====
        coords = await fetch_wikipedia_coords(client, lang, lookup_title)

        # ===== 3️⃣ FALLBACK TO ENGLISH =====
        if not coords and lang == "he":
            logger.warning(f"⚠️ Hebrew coords not found for {lookup_title} → fallback to EN")
            coords = await fetch_wikipedia_coords(client, "en", city_clean)

        # ===== 4️⃣ If still none → no result =====
        if not coords:
            logger.error(f"❌ No coordinates found for {city_clean}")
            result = {"ok": False, "city": city_clean, "coords": None}
            cache_set(city_clean, lang, result)
            return result

    # ===== 5️⃣ SUCCESS =====
    result = {
        "ok": True,
        "city": city_clean,
        "lookup": lookup_title,
        "coords": coords,
    }

    cache_set(city_clean, lang, result)
    return result
