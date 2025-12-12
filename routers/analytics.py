import aiosqlite
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime, timedelta
import asyncio

router = APIRouter(tags=["Analytics"])

# ------------------------------------------------------
# PATHS
# ------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
ANALYTICS_DIR = BASE_DIR / "analytics"
DB_PATH = ANALYTICS_DIR / "destinations.db"
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------
# GLOBAL ASYNC CONNECTION (kept open)
# ------------------------------------------------------
DB_CONN: aiosqlite.Connection | None = None
DB_LOCK = asyncio.Lock()   # Avoid concurrency write collisions


async def get_conn() -> aiosqlite.Connection:
    global DB_CONN
    if DB_CONN is None:
        DB_CONN = await aiosqlite.connect(DB_PATH)
        await DB_CONN.execute("PRAGMA journal_mode=WAL;")
        await DB_CONN.execute("PRAGMA synchronous=NORMAL;")
        await DB_CONN.execute("PRAGMA temp_store=MEMORY;")
    return DB_CONN


# ------------------------------------------------------
# DATABASE INIT (async)
# ------------------------------------------------------
@router.on_event("startup")
async def init_tables():
    conn = await get_conn()

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS destination_clicks (
            iata TEXT PRIMARY KEY,
            city TEXT,
            country TEXT,
            total_clicks INTEGER NOT NULL DEFAULT 0,
            last_clicked TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS analytics_daily (
            date   TEXT,
            iata   TEXT,
            clicks INTEGER,
            PRIMARY KEY (date, iata)
        )
    """)

    await conn.commit()


# ------------------------------------------------------
# CACHING LAYER FOR READS
# ------------------------------------------------------
TOP_CACHE = None
TOP_CACHE_EXP = None
TOP_TTL_SEC = 10   # refresh every 10 sec


# ------------------------------------------------------
# MODELS
# ------------------------------------------------------
class ClickEvent(BaseModel):
    iata: str
    city: str
    country: str


# ------------------------------------------------------
# TRACK CLICK  (async, non-blocking)
# ------------------------------------------------------
@router.post("/api/analytics/click")
async def track_click(event: ClickEvent):
    iata = event.iata.upper().strip()
    city = event.city.strip()
    country = event.country.strip()

    if len(iata) != 3:
        raise HTTPException(status_code=400, detail="Invalid IATA code")

    today = datetime.now().strftime("%Y-%m-%d")
    conn = await get_conn()

    async with DB_LOCK:  # prevent write collisions
        await conn.execute("""
            INSERT INTO destination_clicks (iata, city, country, total_clicks, last_clicked)
            VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
            ON CONFLICT(iata)
            DO UPDATE SET
                total_clicks = total_clicks + 1,
                city = excluded.city,
                country = excluded.country,
                last_clicked = CURRENT_TIMESTAMP
        """, (iata, city, country))

        await conn.execute("""
            INSERT INTO analytics_daily (date, iata, clicks)
            VALUES (?, ?, 1)
            ON CONFLICT(date, iata)
            DO UPDATE SET clicks = clicks + 1
        """, (today, iata))

        await conn.commit()

    # invalidate cache
    global TOP_CACHE_EXP
    TOP_CACHE_EXP = None

    return {"status": "ok", "iata": iata}


# ------------------------------------------------------
# GET TOP DESTINATIONS (async + cached)
# ------------------------------------------------------
@router.get("/api/analytics/top")
async def get_top(limit: int = 20):

    global TOP_CACHE, TOP_CACHE_EXP

    now = datetime.now()

    if TOP_CACHE and TOP_CACHE_EXP and now < TOP_CACHE_EXP:
        return TOP_CACHE[:limit]  # return already-sliced

    conn = await get_conn()

    cursor = await conn.execute("""
        SELECT iata, city, country, total_clicks, last_clicked
        FROM destination_clicks
        ORDER BY total_clicks DESC
        LIMIT ?
    """, (limit,))

    rows = await cursor.fetchall()

    result = [{
        "iata": r[0],
        "city": r[1],
        "country": r[2],
        "total_clicks": r[3],
        "last_clicked": r[4],
    } for r in rows]

    # update cache
    TOP_CACHE = result
    TOP_CACHE_EXP = now + timedelta(seconds=TOP_TTL_SEC)

    return result


# ------------------------------------------------------
# TRENDING DESTINATIONS (async)
# ------------------------------------------------------
@router.get("/api/analytics/trending")
async def trending_destinations(limit: int = 50):
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    conn = await get_conn()

    # LOAD TODAY
    cur = await conn.execute(
        "SELECT iata, clicks FROM analytics_daily WHERE date = ?", (today,)
    )
    today_rows = {r[0]: r[1] for r in await cur.fetchall()}

    # LOAD YESTERDAY
    cur = await conn.execute(
        "SELECT iata, clicks FROM analytics_daily WHERE date = ?", (yesterday,)
    )
    yesterday_rows = {r[0]: r[1] for r in await cur.fetchall()}

    # LOAD META (city/country)
    cur = await conn.execute(
        "SELECT iata, city, country FROM destination_clicks"
    )
    meta = {r[0]: (r[1], r[2]) for r in await cur.fetchall()}

    trending = []

    for iata, today_clicks in today_rows.items():
        y_clicks = yesterday_rows.get(iata, 0)
        change = today_clicks - y_clicks
        city, country = meta.get(iata, ("Unknown", "Unknown"))

        trending.append({
            "iata": iata,
            "city": city,
            "country": country,
            "today": today_clicks,
            "yesterday": y_clicks,
            "change": change,
            "percent": round((change / y_clicks * 100), 1) if y_clicks else None
        })

    trending.sort(key=lambda x: x["change"], reverse=True)
    return trending[:limit]


# ------------------------------------------------------
# ADMIN DASHBOARD
# ------------------------------------------------------
@router.get("/admin/analytics", response_class=HTMLResponse)
async def analytics_dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


# ------------------------------------------------------
# RESET DB
# ------------------------------------------------------
@router.get("/analytics/reset")
async def reset_stats():
    conn = await get_conn()
    async with DB_LOCK:
        await conn.execute("DELETE FROM destination_clicks")
        await conn.execute("DELETE FROM analytics_daily")
        await conn.commit()
    return {"status": "reset"}





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
      // ---------- TABLE FILL ----------
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

      // ---------- CHART ----------
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
def analytics_dashboard():
    html = DASHBOARD_HTML
    return HTMLResponse(content=html)
    
@router.get("/api/analytics/trending")
def trending_destinations(limit: int = 50):
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    with get_conn() as conn:
        today_rows = {
            r[0]: r[1] for r in conn.execute(
                "SELECT iata, clicks FROM analytics_daily WHERE date = ?", (today,)
        )}

        yesterday_rows = {
            r[0]: r[1] for r in conn.execute(
                "SELECT iata, clicks FROM analytics_daily WHERE date = ?", (yesterday,)
        )}

        meta = {
            r[0]: (r[1], r[2])  # city, country
            for r in conn.execute("SELECT iata, city, country FROM destination_clicks")
        }

    trending = []

    for iata, today_clicks in today_rows.items():
        y_clicks = yesterday_rows.get(iata, 0)
        change = today_clicks - y_clicks
        city, country = meta.get(iata, ("Unknown", "Unknown"))

        trending.append({
            "iata": iata,
            "city": city,
            "country": country,
            "today": today_clicks,
            "yesterday": y_clicks,
            "change": change,
            "percent": round((change / y_clicks * 100), 1) if y_clicks else None
        })

    trending.sort(key=lambda x: x["change"], reverse=True)
    return trending[:limit]

