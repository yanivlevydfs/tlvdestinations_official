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
        "exhibition", "art center", "heritage center"
    ],

    "parks": [
        "park", "garden", "botanical", "lake", "forest",
        "national park", "nature reserve", "green", "grove"
    ],

    "landmarks": [
        "landmark", "square", "plaza", "tower", "palace", "bridge",
        "monument", "statue", "clock", "pyramid", "fountain",
        "historic site", "memorial", "citadel"
    ],

    "religious": [
        "church", "cathedral", "mosque", "temple", "synagogue",
        "basilica", "monastery", "convent", "shrine"
    ],

    "beaches": [
        "beach", "bay", "cove", "coast", "promenade", "harbor"
    ],

    "castles": [
        "castle", "fort", "fortress", "citadel", "acropolis"
    ],

    "zoo": [
        "zoo", "wildlife park", "animal park"
    ],

    "aquarium": [
        "aquarium", "marine park", "oceanarium"
    ],

    "viewpoints": [
        "mount", "mountain", "hill", "viewpoint", "lookout",
        "observation deck", "panorama", "skywalk"
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
    Fetch members of Category:Tourist_attractions_in_<City> (pages only, not subcategories/files).
    Will follow pagination until all are fetched.
    """
    category_title = f"Category:Tourist_attractions_in_{city.replace(' ', '_')}"
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": category_title,
        "cmlimit": "max",
        "cmtype": "page",  # critical: exclude Category/File
    }

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
    Batch-fetch details (extract, thumbnail, coords, url) for up to 50 titles at a time.
    Returns dict keyed by normalized title.
    """
    if not titles:
        return {}

    # Wikipedia allows many titles via pipe-separated list
    title_param = "|".join(titles)
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|pageimages|coordinates|info",
        "exintro": 1,
        "explaintext": 1,
        "pithumbsize": 400,
        "inprop": "url",
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
        coords = p.get("coordinates", [])
        lat = lon = None
        if coords and isinstance(coords, list):
            c0 = coords[0]
            lat = c0.get("lat")
            lon = c0.get("lon")
        out[title.lower()] = {
            "title": title,
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
    Returns curated attractions from Wikipedia's category:
      'Category:Tourist_attractions_in_<City>'
    Enriched with summary, thumbnail, coordinates, and filtered by category if requested.
    """
    city_clean = city.strip().title()
    try:
        # 1) Pull category members (pages only)
        members = await fetch_category_pages(lang, city_clean)
        if not members:
            raise HTTPException(404, f"No attractions category found for {city_clean} ({lang})")

        # 2) Enrich with details in batches
        titles = [m["title"] for m in members if "title" in m]
        # fetch in batches of 50 (API constraint)
        details: Dict[str, Dict[str, Any]] = {}
        for batch in chunked(titles, 50):
            d = await fetch_page_details(lang, batch)
            details.update(d)

        # 3) Build results, classify + filter
        results = []
        seen = set()
        for t in titles:
            d = details.get(t.lower())
            if not d:
                continue
            key = d["title"].lower()
            if key in seen:
                continue
            seen.add(key)

            detected = classify(d["title"])
            if category and detected != category:
                continue
            if not include_other and detected == "other":
                continue
                
            lat = d["lat"]
            lon = d["lon"]
            results.append({
                "title": d["title"],
                "category": detected,
                "url": d["url"] or f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(d['title'].replace(' ', '_'))}",
                "thumbnail": d["thumbnail"],
                "extract": d["extract"],
                "lat": lat,
                "lon": lon,
                "maps": f"https://www.google.com/maps?q={lat},{lon}" if lat and lon else None,
                "waze":  f"https://waze.com/ul?ll={lat}%2C{lon}&navigate=yes" if lat and lon else None,
                "osm":   f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=17/{lat}/{lon}" if lat and lon else None,
            })

        # 4) Sort: preferred categories first, then alpha
        pref_order = ["landmarks", "museums", "parks", "castles", "religious", "viewpoints", "beaches", "zoo", "aquarium", "other"]
        results.sort(key=lambda x: (pref_order.index(x["category"]) if x["category"] in pref_order else 999, x["title"]))

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
