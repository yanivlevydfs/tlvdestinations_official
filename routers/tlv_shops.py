import json
import httpx
import logging
from pathlib import Path
from fastapi import Request, APIRouter, Depends
from fastapi.responses import HTMLResponse
from json_repair import repair_json
from helpers.helper import get_lang

from core.templates import TEMPLATES

logger = logging.getLogger("fly_tlv.tlv_shops")

DATA_GOV_URL = (
    "https://data.gov.il/api/3/action/datastore_search"
    "?resource_id=9cc1be06-79e0-4f9b-b7a6-ad9af1293cc0"
)

# ───────────────────────────────────────────────
# GLOBAL + CACHE
# ───────────────────────────────────────────────
TLV_SHOPS_DATA: dict | None = None
CACHE_FILE = Path("cache/tlv_shops_cache.json")

router = APIRouter()


# ───────────────────────────────────────────────
# Loader (API → cache → global)
# ───────────────────────────────────────────────
async def load_tlv_shops() -> dict:
    global TLV_SHOPS_DATA

    # 1️⃣ GLOBAL
    if TLV_SHOPS_DATA is not None:
        return TLV_SHOPS_DATA

    # 2️⃣ DISK CACHE
    if CACHE_FILE.exists():
        try:
            raw = CACHE_FILE.read_text(encoding="utf-8")
            TLV_SHOPS_DATA = json.loads(repair_json(raw))
            return TLV_SHOPS_DATA
        except Exception as e:
            logger.error("Failed reading TLV shops cache: %s", e)

    # 3️⃣ API (FIRST TIME / FORCED)
    logger.debug("Fetching TLV shops from data.gov.il (first time)")
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(DATA_GOV_URL)
        r.raise_for_status()
        data = r.json()

    records = data.get("result", {}).get("records", [])

    shops = []
    categories = set()
    locations = set()
    duty_free_values = set()

    for r in records:
        shop = {
            "name": (r.get("חנות") or "").strip(),
            "category": r.get("קטיגוריה") or "—",
            "duty_free": r.get("חנות ללא מכס") or "—",
            "location": r.get("מיקום") or "—",
        }

        if shop["category"] != "—":
            categories.add(shop["category"])
        if shop["location"] != "—":
            locations.add(shop["location"])
        if shop["duty_free"] != "—":
            duty_free_values.add(shop["duty_free"])

        shops.append(shop)

    TLV_SHOPS_DATA = {
        "shops": shops,
        "categories": sorted(categories),
        "locations": sorted(locations),
        "duty_free_values": sorted(duty_free_values),
        "updated": data.get("result", {}).get("records_updated"),
    }

    # Persist cache
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(
            json.dumps(TLV_SHOPS_DATA, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.debug("TLV shops cached to disk")
    except Exception as e:
        logger.error("⚠️ Failed writing TLV shops cache: %s", e)

    return TLV_SHOPS_DATA


# ───────────────────────────────────────────────
# Route
# ───────────────────────────────────────────────
@router.get("/tlv-shops", response_class=HTMLResponse)
async def tlv_shops_view(
    request: Request,
    lang: str = Depends(get_lang),

):
    data = await load_tlv_shops()

    return TEMPLATES.TemplateResponse(
        "tlv_shops.html",
        {
            "request": request,
            "lang": lang,
            "shops": data["shops"],
            "categories": data["categories"],
            "locations": data["locations"],
            "duty_free_values": data["duty_free_values"],
            "last_update": data.get("updated"),
        },
    )

