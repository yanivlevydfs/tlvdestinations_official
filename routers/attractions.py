# routers/attractions.py
from fastapi import APIRouter, Query, HTTPException
import httpx
import urllib.parse
import logging
from typing import List, Dict, Any

router = APIRouter(prefix="/api", tags=["Attractions"])
logger = logging.getLogger("attractions")

WIKI_API = "https://{lang}.wikipedia.org/w/api.php"  # lang = en | he
HEADERS = {
    "User-Agent": "Fly-TLV/1.0 (https://fly-tlv.com; contact@fly-tlv.com)",
    "Accept": "application/json",
}

CATEGORY_KEYWORDS = {

    "museums": [
        "museum", "museums",
        "gallery", "galleries",
        "exhibition", "art center", "heritage center",
        "history museum", "science museum", "archaeological museum",
        "open air museum", "cultural center", "visitor center"
    ],

    "parks": [
        "park", "garden", "botanical", "lake", "forest",
        "national park", "nature reserve", "grove",
        "public garden", "city park", "urban park",
        "arboretum", "wetland", "woods"
    ],

    "landmarks": [
        "landmark", "square", "plaza", "tower", "palace", "bridge",
        "monument", "statue", "clock", "fountain",
        "historic site", "memorial", "citadel",
        "old town", "historic center", "city walls",
        "archaeological site", "ruins", "arch", "gate",
        "obelisk", "boulevard", "avenue", "main street"
    ],

    "religious": [
        "church", "cathedral", "mosque", "temple", "synagogue",
        "basilica", "monastery", "convent", "shrine",
        "chapel", "holy site", "religious site", "minaret"
    ],

    "beaches": [
        "beach", "bay", "cove", "coast", "promenade", "harbor",
        "marina", "seaside", "shore", "waterfront",
        "pier", "boardwalk"
    ],

    "castles": [
        "castle", "fort", "fortress", "citadel", "acropolis",
        "stronghold", "bastion", "keep", "tower house",
        "walled city"
    ],

    "zoo": [
        "zoo", "wildlife park", "animal park",
        "safari park", "animal reserve"
    ],

    "aquarium": [
        "aquarium", "marine park", "oceanarium",
        "sealife center"
    ],

    "viewpoints": [
        "mount", "mountain", "hill", "viewpoint", "lookout",
        "observation deck", "panorama", "skywalk",
        "summit", "peak", "cliff", "scenic overlook"
    ],
}



def classify(title: str) -> str:
    t = title.lower().replace("_", " ")

    # PRIORITY: religious > museums > landmarks > parks ...
    for category, keywords in CATEGORY_KEYWORDS.items():
        for k in keywords:
            if k in t:
                return category

    return "other"



async def fetch_category_pages(lang: str, city: str) -> List[Dict[str, Any]]:
    """
    Fetch members of Category:Tourist_attractions_in_<City>.
    If missing, gracefully fall back to alternative categories.
    """

    async def fetch(cat_title: str) -> List[Dict[str, Any]]:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": cat_title,
            "cmlimit": "max",
            "cmtype": "page",
        }

        items = []
        async with httpx.AsyncClient(headers=HEADERS, timeout=20) as client:
            while True:
                r = await client.get(WIKI_API.format(lang=lang), params=params)
                if r.status_code != 200:
                    return []
                data = r.json()
                items.extend(data.get("query", {}).get("categorymembers", []))
                cont = data.get("continue")
                if not cont:
                    break
                params.update(cont)
        return items

    city_clean = city.replace(" ", "_")

    # Primary category
    primary = f"Category:Tourist_attractions_in_{city_clean}"
    members = await fetch(primary)

    if members:
        return members

    logger.debug(f"No primary category for {city} — trying fallbacks")

    # ⭐ DYNAMIC fallback list (no hardcoding per country)
    fallback_titles = [
        f"Category:{city_clean}",
        f"Category:Buildings_and_structures_in_{city_clean}",
        f"Category:Landmarks_in_{city_clean}",
        f"Category:Museums_in_{city_clean}",
        f"Category:Parks_in_{city_clean}",
    ]

    for fb in fallback_titles:
        if not fb:
            continue
        members = await fetch(fb)
        if members:
            logger.debug(f"Using fallback category: {fb}")
            return members

    # No results at all
    logger.error(f"No categories found for {city}")
    return []

    items: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(headers=HEADERS, timeout=20) as client:
        while True:
            r = await client.get(WIKI_API.format(lang=lang), params=params)
            if r.status_code != 200:
                raise HTTPException(503, f"Wikipedia error {r.status_code}")
            data = r.json()
            items.extend(data.get("query", {}).get("categorymembers", []))
            cont = data.get("continue", {})
            if not cont:
                break
            params.update(cont)
    return items

async def fetch_page_details(lang: str, titles: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Batch-fetch details (extract, thumbnail, coords, url, langlinks) for up to 50 titles at a time.
    Returns dict keyed by normalized title.
    Adds Hebrew title fallback when available.
    """
    if not titles:
        return {}

    # Wikipedia allows many titles via pipe-separated list
    title_param = "|".join(titles)

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|pageimages|coordinates|info|langlinks",
        "exintro": 1,
        "explaintext": 1,
        "pithumbsize": 400,
        "inprop": "url",
        "lllimit": 1,
        "lllang": "he",        # ⭐ request Hebrew page title
        "titles": title_param,
    }

    async with httpx.AsyncClient(headers=HEADERS, timeout=20) as client:
        r = await client.get(WIKI_API.format(lang=lang), params=params)
        if r.status_code != 200:
            raise HTTPException(503, f"Wikipedia detail error {r.status_code}")
        data = r.json()

    pages = data.get("query", {}).get("pages", {})
    out: Dict[str, Dict[str, Any]] = {}

    for _, p in pages.items():
        title = p.get("title")
        if not title:
            continue

        # Coordinates
        coords = p.get("coordinates", [])
        lat = lon = None
        if coords:
            c0 = coords[0]
            lat = c0.get("lat")
            lon = c0.get("lon")

        # ⭐ Extract Hebrew title if available
        hebrew_title = None
        for ll in p.get("langlinks", []):
            if ll.get("lang") == "he":
                hebrew_title = ll.get("*")
                break

        out[title.lower()] = {
            "title": title,
            "title_he": hebrew_title,     # ⭐ added Hebrew variant
            "extract": p.get("extract"),
            "thumbnail": p.get("thumbnail", {}).get("source"),
            "url": p.get("fullurl"),
            "lat": lat,
            "lon": lon,
        }

    return out


def chunked(seq: List[str], n: int) -> List[List[str]]:
    return [seq[i:i+n] for i in range(0, len(seq), n)]


@router.get("/attractions")
async def attractions(
    city: str = Query(..., description="City name in English (e.g., Tirana)"),
    category: str | None = Query(None, description="Filter: museums, parks, landmarks, religious, beaches, castles, zoo, aquarium, viewpoints"),
    lang: str = Query("en", pattern="^(en|he)$", description="Wikipedia language (en/he)"),
    limit: int = Query(50, ge=1, le=200, description="Max results to return"),
    include_other: bool = Query(False, description="Include 'other' category items if True"),
):
    """
    Returns curated attractions from Wikipedia based on:
      1) Primary category: Tourist_attractions_in_<City>
      2) Fallback categories (landmarks, museums, parks, structures)
      3) Text-search fallback
      4) GeoSearch fallback

    Output format remains identical.
    """
    city_clean = city.strip().title()
    logger.debug(f"🌍 Fetching attractions for: {city_clean} ({lang})")

    try:
        # ---------------------------------------------------------
        # 1) Primary: Tourist_attractions_in_<City>
        # ---------------------------------------------------------
        members = await fetch_category_pages(lang, city_clean)

        # ---------------------------------------------------------
        # FALLBACK SECTION
        # ---------------------------------------------------------
        if not members:
            logger.debug(f"No primary category for {city_clean} → trying fallback categories")

            fallback_categories = [
                f"Category:Landmarks_in_{city_clean.replace(' ', '_')}",
                f"Category:Buildings_and_structures_in_{city_clean.replace(' ', '_')}",
                f"Category:Museums_in_{city_clean.replace(' ', '_')}",
                f"Category:Parks_in_{city_clean.replace(' ', '_')}",
            ]

            async with httpx.AsyncClient(headers=HEADERS, timeout=20) as client:
                for fc in fallback_categories:
                    try:
                        r = await client.get(WIKI_API.format(lang=lang), params={
                            "action": "query",
                            "format": "json",
                            "list": "categorymembers",
                            "cmtitle": fc,
                            "cmlimit": "max",
                            "cmtype": "page",
                        })
                        cm = r.json().get("query", {}).get("categorymembers", [])
                        if cm:
                            logger.debug(f"Using fallback category: {fc}")
                            members = cm
                            break
                    except Exception as e:
                        logger.warning(f"Fallback category {fc} failed: {e}")

        # ---------------------------------------------------------
        # 2) Text search fallback
        # ---------------------------------------------------------
        if not members:
            logger.debug(f"No fallback categories worked → text search for {city_clean}")

            async with httpx.AsyncClient(headers=HEADERS, timeout=20) as client:
                r = await client.get(WIKI_API.format(lang=lang), params={
                    "action": "query",
                    "format": "json",
                    "list": "search",
                    "srlimit": 20,
                    "srsearch": f"{city_clean} attractions OR landmarks OR points of interest",
                })
                search_items = r.json().get("query", {}).get("search", [])

            if search_items:
                members = [{"title": s["title"]} for s in search_items]
                logger.debug(f"Using text-search fallback for {city_clean}")

        # ---------------------------------------------------------
        # 3) GeoSearch fallback
        # ---------------------------------------------------------
        if not members:
            logger.debug(f"No text search results → GeoSearch fallback for {city_clean}")

            # Get city coordinates
            async with httpx.AsyncClient(headers=HEADERS, timeout=20) as client:
                coords_resp = await client.get(WIKI_API.format(lang=lang), params={
                    "action": "query",
                    "format": "json",
                    "prop": "coordinates",
                    "titles": city_clean,
                })

            pages = coords_resp.json().get("query", {}).get("pages", {})
            page = next(iter(pages.values()), {})
            coords = page.get("coordinates", [])

            if coords:
                lat, lon = coords[0]["lat"], coords[0]["lon"]

                async with httpx.AsyncClient(headers=HEADERS, timeout=20) as client:
                    r = await client.get(WIKI_API.format(lang=lang), params={
                        "action": "query",
                        "format": "json",
                        "list": "geosearch",
                        "gscoord": f"{lat}|{lon}",
                        "gsradius": 8000,
                        "gslimit": 20,
                    })
                geolist = r.json().get("query", {}).get("geosearch", [])
                if geolist:
                    members = [{"title": g["title"]} for g in geolist]
                    logger.debug(f"Using GeoSearch fallback for {city_clean}")

        # ---------------------------------------------------------
        # 4) FINAL failure → return empty structured response
        # ---------------------------------------------------------
        if not members:
            logger.error(f"🚫 All fallback methods failed for {city_clean}")
            return {
                "city": city_clean,
                "requested_city": city,
                "lang": lang,
                "category": category,
                "count": 0,
                "results": [],
            }

        # ---------------------------------------------------------
        # 5) Enrich details (existing functionality)
        # ---------------------------------------------------------
        titles = [m.get("title") for m in members if m.get("title")]
        details: Dict[str, Dict[str, Any]] = {}

        for batch in chunked(titles, 50):
            enriched = await fetch_page_details(lang, batch)
            if enriched:
                details.update(enriched)

        # ---------------------------------------------------------
        # 6) Build results using your classify()
        # ---------------------------------------------------------
        results = []
        seen = set()

        for t in titles:
            d = details.get(t.lower())
            if not d:
                continue

            name = d["title"].lower()
            if name in seen:
                continue
            seen.add(name)

            detected = classify(d["title"])

            if category and detected != category:
                continue
            if not include_other and detected == "other":
                continue

            lat, lon = d.get("lat"), d.get("lon")

            results.append({
                "title": d["title"],
                "category": detected,
                "url": d["url"] or f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(d['title'].replace(' ', '_'))}",
                "thumbnail": d.get("thumbnail"),
                "extract": d.get("extract"),
                "lat": lat,
                "lon": lon,
                "maps": f"https://www.google.com/maps?q={lat},{lon}" if lat and lon else None,
                "waze": f"https://waze.com/ul?ll={lat}%2C{lon}&navigate=yes" if lat and lon else None,
                "osm": f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=17/{lat}/{lon}" if lat and lon else None,
            })

        # ---------------------------------------------------------
        # 7) Sorting (same logic)
        # ---------------------------------------------------------
        pref_order = ["landmarks", "museums", "parks", "castles", "religious",
                      "viewpoints", "beaches", "zoo", "aquarium", "other"]

        results.sort(
            key=lambda x: (
                pref_order.index(x["category"]) if x["category"] in pref_order else 999,
                x["title"],
            )
        )

        return {
            "city": city_clean,
            "requested_city": city,
            "lang": lang,
            "category": category,
            "count": min(len(results), limit),
            "results": results[:limit],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Attractions API failed: %s", e)
        raise HTTPException(500, "Internal attractions error")

