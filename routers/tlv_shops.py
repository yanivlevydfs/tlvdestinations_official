import json
import httpx
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Request, APIRouter, Depends
from fastapi.responses import HTMLResponse
from json_repair import repair_json

from helpers.helper import get_lang
from core.templates import TEMPLATES

logger = logging.getLogger("fly_tlv.tlv_shops")

BASE_URL = "https://data.gov.il/api/3/action/datastore_search"
RESOURCE_ID = "9cc1be06-79e0-4f9b-b7a6-ad9af1293cc0"
LIMIT = 32000

TLV_SHOPS_DATA: dict | None = None
CACHE_FILE = Path("cache/tlv_shops_cache.json")

router = APIRouter()


def _clean(val: object) -> str:
    """Normalize CKAN text fields (None → '', trim whitespace)."""
    if val is None:
        return ""
    s = str(val).strip()
    return s


def _is_junk_row(category: str, name: str, duty_free: str, location: str) -> bool:
    """
    Filters rows that are clearly not real data, based on your schema sample:
    - rows with all empty fields
    - 'header' rows where values equal the field names (e.g. category == 'קטיגוריה')
    """
    if not (category or name or duty_free or location):
        return True

    # header-like junk row observed in sample (_id=45)
    if category == "קטיגוריה" and location == "מיקום":
        return True

    # another common junk pattern: category exists but name/location empty (e.g. _id=43 in sample)
    # keep this strict to avoid deleting legitimate rows; but your sample shows it's junk.
    if category and not name and not location:
        return True

    return False


async def _fetch_all_records(client: httpx.AsyncClient) -> list[dict]:
    """
    CKAN datastore_search supports offset paging.
    We fetch until:
      - returned page has 0 records, or
      - we fetched >= total (if total provided), or
      - no 'next' link (optional)
    """
    offset = 0
    all_records: list[dict] = []
    total: int | None = None

    while True:
        params = {"resource_id": RESOURCE_ID, "limit": LIMIT, "offset": offset}
        r = await client.get(BASE_URL, params=params)
        r.raise_for_status()
        data = r.json()

        if not data.get("success"):
            raise RuntimeError(f"CKAN request failed: success=false, payload keys={list(data.keys())}")

        result = data.get("result") or {}
        records = result.get("records") or []
        total = result.get("total") if isinstance(result.get("total"), int) else total

        if not records:
            break

        all_records.extend(records)

        offset += len(records)

        # stop conditions
        if total is not None and offset >= total:
            break

        # If server returns less than limit, we likely reached end.
        if len(records) < LIMIT:
            break

    return all_records


async def load_tlv_shops() -> dict:
    global TLV_SHOPS_DATA

    # 1) GLOBAL
    if TLV_SHOPS_DATA is not None:
        return TLV_SHOPS_DATA

    # 2) DISK CACHE
    if CACHE_FILE.exists():
        try:
            raw = CACHE_FILE.read_text(encoding="utf-8")
            TLV_SHOPS_DATA = json.loads(repair_json(raw))
            return TLV_SHOPS_DATA
        except Exception as e:
            logger.error("Failed reading TLV shops cache: %s", e)

    # 3) API
    logger.debug("Fetching TLV shops from data.gov.il (cache miss)")
    fetched_at = datetime.now(timezone.utc).isoformat()

    async with httpx.AsyncClient(timeout=30) as client:
        records = await _fetch_all_records(client)

    shops: list[dict] = []
    categories: set[str] = set()
    locations: set[str] = set()
    duty_free_values: set[str] = set()

    for rec in records:
        category = _clean(rec.get("קטיגוריה"))
        name = _clean(rec.get("חנות"))
        duty_free = _clean(rec.get("חנות ללא מכס"))
        location = _clean(rec.get("מיקום"))

        if _is_junk_row(category, name, duty_free, location):
            continue

        # Keep your UI contract: use "—" placeholders
        shop = {
            "name": name or "—",
            "category": category or "—",
            "duty_free": duty_free or "—",
            "location": location or "—",
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
        # Schema-accurate: we compute our own update timestamp
        "updated": fetched_at,
        # Optional: useful for debugging / sanity checks
        "total": len(shops),
    }

    # Persist cache
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(TLV_SHOPS_DATA, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("TLV shops cached to disk")
    except Exception as e:
        logger.error("⚠️ Failed writing TLV shops cache: %s", e)

    return TLV_SHOPS_DATA


@router.get("/tlv-shops", response_class=HTMLResponse)
async def tlv_shops_view(request: Request, lang: str = Depends(get_lang)):
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
