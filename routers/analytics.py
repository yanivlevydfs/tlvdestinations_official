import aiosqlite
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from datetime import datetime, timedelta
import asyncio

router = APIRouter(tags=["Analytics"])

from helpers.analytics_db import analytics_db
from core.templates import TEMPLATES

# simple in-memory cache for /top
TOP_CACHE: list | None = None
TOP_CACHE_EXP: datetime | None = None
TOP_TTL_SEC = 10

# ------------------------------------------------------
# LIFECYCLE
# ------------------------------------------------------
@router.on_event("startup")
async def init_analytics():
    await analytics_db.init_db()


# ------------------------------------------------------
# MODELS
# ------------------------------------------------------
class ClickEvent(BaseModel):
    iata: str
    city: str
    country: str


# ------------------------------------------------------
# TRACK CLICK (async, serialized writes)
# ------------------------------------------------------
@router.post("/api/data/log_visit")
async def track_click(event: ClickEvent):
    iata = (event.iata or "").upper().strip()
    city = (event.city or "").strip()
    country = (event.country or "").strip()

    if len(iata) != 3:
        raise HTTPException(status_code=400, detail="Invalid IATA code")

    # If you prefer to reject bad payloads instead of preserving old values, uncomment:
    # if not city or not country:
    #     raise HTTPException(status_code=400, detail="City and country are required")

    today = datetime.now().strftime("%Y-%m-%d")
    
    async with analytics_db.get_connection() as conn:
        await conn.execute(
            """
            INSERT INTO destination_clicks (iata, city, country, total_clicks, last_clicked)
            VALUES (?, NULLIF(?, ''), NULLIF(?, ''), 1, CURRENT_TIMESTAMP)
            ON CONFLICT(iata) DO UPDATE SET
                total_clicks = destination_clicks.total_clicks + 1,
                city = COALESCE(excluded.city, destination_clicks.city),
                country = COALESCE(excluded.country, destination_clicks.country),
                last_clicked = CURRENT_TIMESTAMP
            """,
            (iata, city, country),
        )

        await conn.execute(
            """
            INSERT INTO analytics_daily (date, iata, clicks)
            VALUES (?, ?, 1)
            ON CONFLICT(date, iata) DO UPDATE SET
                clicks = analytics_daily.clicks + 1
            """,
            (today, iata),
        )

        await conn.commit()

    # invalidate cache
    global TOP_CACHE_EXP
    TOP_CACHE_EXP = None

    return {"status": "ok", "iata": iata}

# ------------------------------------------------------
# TOP DESTINATIONS (async + cached)
# ------------------------------------------------------
@router.get("/api/analytics/top")
async def get_top(limit: int = 20):
    global TOP_CACHE, TOP_CACHE_EXP

    now = datetime.now()
    if TOP_CACHE and TOP_CACHE_EXP and now < TOP_CACHE_EXP:
        return TOP_CACHE[:limit]

    async with analytics_db.get_connection() as conn:
        cur = await conn.execute(
            """
            SELECT iata, city, country, total_clicks, last_clicked
            FROM destination_clicks
            ORDER BY total_clicks DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cur.fetchall()

    result = [
        {
            "iata": r[0],
            "city": r[1],
            "country": r[2],
            "total_clicks": r[3],
            "last_clicked": r[4],
        }
        for r in rows
    ]

    TOP_CACHE = result
    TOP_CACHE_EXP = now + timedelta(seconds=TOP_TTL_SEC)
    return result


# ------------------------------------------------------
# TRENDING DESTINATIONS (today vs yesterday)
# ------------------------------------------------------
@router.get("/api/analytics/trending")
async def trending_destinations(limit: int = 50):
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    async with analytics_db.get_connection() as conn:
        cur = await conn.execute(
            "SELECT iata, clicks FROM analytics_daily WHERE date = ?", (today,)
        )
        today_rows = {r[0]: r[1] for r in await cur.fetchall()}

        cur = await conn.execute(
            "SELECT iata, clicks FROM analytics_daily WHERE date = ?", (yesterday,)
        )
        yesterday_rows = {r[0]: r[1] for r in await cur.fetchall()}

        cur = await conn.execute("SELECT iata, city, country FROM destination_clicks")
        meta = {r[0]: (r[1], r[2]) for r in await cur.fetchall()}

    trending = []
    for iata, today_clicks in today_rows.items():
        y_clicks = yesterday_rows.get(iata, 0)
        change = today_clicks - y_clicks
        city, country = meta.get(iata, ("Unknown", "Unknown"))
        trending.append(
            {
                "iata": iata,
                "city": city,
                "country": country,
                "today": today_clicks,
                "yesterday": y_clicks,
                "change": change,
                "percent": round((change / y_clicks * 100), 1) if y_clicks else None,
            }
        )

    trending.sort(key=lambda x: x["change"], reverse=True)
    return trending[:limit]


@router.get("/admin/analytics", response_class=HTMLResponse)
async def analytics_dashboard(request: Request):
    return TEMPLATES.TemplateResponse(
        "admin_dashboard.html",
        {
            "request": request,
            "lang": "en",  # Admin usually in English, or you can use get_lang
            "title": "Analytics Dashboard — Fly TLV",
        }
    )


# ------------------------------------------------------
# RESET DB (careful)
# ------------------------------------------------------
@router.get("/analytics/reset")
async def reset_stats():
    async with analytics_db.get_connection() as conn:
        await conn.execute("DELETE FROM destination_clicks")
        await conn.execute("DELETE FROM analytics_daily")
        await conn.commit()
    # also drop cache
    global TOP_CACHE, TOP_CACHE_EXP
    TOP_CACHE, TOP_CACHE_EXP = None, None
    return {"status": "reset"}


# Optional: quick health check
@router.get("/analytics/health")
async def analytics_health():
    try:
        async with analytics_db.get_connection() as conn:
            await conn.execute("SELECT 1")
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
