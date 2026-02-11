import aiosqlite
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
import sqlite3

router = APIRouter(tags=["Analytics"])

# ------------------------------------------------------
# PATHS
# ------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
ANALYTICS_DIR = BASE_DIR / "analytics"
DB_PATH = ANALYTICS_DIR / "destinations.db"
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------
# GLOBALS
# ------------------------------------------------------
DB_CONN: aiosqlite.Connection | None = None
DB_LOCK = asyncio.Lock()   # serialize writes to avoid collisions

# simple in-memory cache for /top
TOP_CACHE: list | None = None
TOP_CACHE_EXP: datetime | None = None
TOP_TTL_SEC = 10


# ------------------------------------------------------
# LIFECYCLE
# ------------------------------------------------------
@router.on_event("startup")
async def init_db_and_conn():
    """
    Render-safe startup:
    1) Open a short-lived connection to set PRAGMAs and create tables.
    2) Open one global async connection for the app lifetime (with busy_timeout).
    """
    # 1) One-time PRAGMAs + schema (short-lived)
    async with aiosqlite.connect(DB_PATH) as db:
        # Do NOT loop this per-connection; WAL is a DB-level switch.
        try:
            await db.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.OperationalError:
            # if another worker flipped it milliseconds earlier
            pass

        # Reasonable durability/perf balance for web analytics
        await db.execute("PRAGMA synchronous=NORMAL;")
        await db.execute("PRAGMA temp_store=MEMORY;")
        await db.execute("PRAGMA busy_timeout=5000;")

        # Schema
        await db.execute("""
            CREATE TABLE IF NOT EXISTS destination_clicks (
                iata TEXT PRIMARY KEY,
                city TEXT,
                country TEXT,
                total_clicks INTEGER NOT NULL DEFAULT 0,
                last_clicked TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS analytics_daily (
                date   TEXT,
                iata   TEXT,
                clicks INTEGER,
                PRIMARY KEY (date, iata)
            )
        """)
        await db.commit()

    # 2) Long-lived connection (reuse in handlers)
    global DB_CONN
    DB_CONN = await aiosqlite.connect(DB_PATH)
    # Busy timeout on the long-lived handle as well
    await DB_CONN.execute("PRAGMA busy_timeout=5000;")
    await DB_CONN.execute("PRAGMA temp_store=MEMORY;")
    # No WAL here; already set above
    await DB_CONN.commit()


@router.on_event("shutdown")
async def close_db_conn():
    global DB_CONN
    if DB_CONN is not None:
        await DB_CONN.close()
        DB_CONN = None


async def get_conn() -> aiosqlite.Connection:
    """Return the global connection; fallback if something reopened."""
    global DB_CONN
    if DB_CONN is None:
        # Very rare path (e.g., hot-reload); keep PRAGMAs light here.
        DB_CONN = await aiosqlite.connect(DB_PATH)
        await DB_CONN.execute("PRAGMA busy_timeout=5000;")
        await DB_CONN.execute("PRAGMA temp_store=MEMORY;")
        await DB_CONN.commit()
    return DB_CONN


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
    conn = await get_conn()

    async with DB_LOCK:
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

    conn = await get_conn()
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

    conn = await get_conn()

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


# ------------------------------------------------------
# ADMIN DASHBOARD
# ------------------------------------------------------
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Analytics Dashboard — Fly TLV</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body { padding: 30px; }
h1 { margin-bottom: 20px; }
.card { margin-bottom: 25px; }
</style>
</head>
<body>
<h1>Fly TLV — Analytics Dashboard</h1>
<div class="card p-3 shadow">
  <h4>Top Destinations (Clicks)</h4>
  <canvas id="topDestChart" height="80"></canvas>
</div>
<div class="card p-3 shadow">
  <h4>Raw Table</h4>
  <table class="table table-striped" id="rawTable">
      <thead>
          <tr>
              <th>IATA</th>
              <th>City</th>
              <th>Country</th>
              <th>Total Clicks</th>
              <th>Last Clicked</th>
          </tr>
      </thead>
      <tbody></tbody>
  </table>
</div>
<script>
// Fetch analytics
fetch('/api/analytics/top?limit=5')
  .then(r => r.json())
  .then(data => {
      // Table
      const tbody = document.querySelector('#rawTable tbody');
      data.forEach(row => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
              <td>${row.iata}</td>
              <td>${row.city}</td>
              <td>${row.country}</td>
              <td>${row.total_clicks}</td>
              <td>${row.last_clicked}</td>
          `;
          tbody.appendChild(tr);
      });
      // Chart
      const ctx = document.getElementById('topDestChart').getContext('2d');
      new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(r => r.iata),
            datasets: [{
                label: 'Clicks',
                data: data.map(r => r.total_clicks),
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            scales: { y: { beginAtZero: true } }
        }
      });
  });
</script>
</body>
</html>
"""

@router.get("/admin/analytics", response_class=HTMLResponse)
async def analytics_dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


# ------------------------------------------------------
# RESET DB (careful)
# ------------------------------------------------------
@router.get("/analytics/reset")
async def reset_stats():
    conn = await get_conn()
    async with DB_LOCK:
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
        conn = await get_conn()
        await conn.execute("SELECT 1")
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
