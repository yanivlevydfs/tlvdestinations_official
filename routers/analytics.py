from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import sqlite3
import json
from fastapi.responses import HTMLResponse
from datetime import datetime, timedelta

router = APIRouter(tags=["Analytics"])

# ------------------------------------------------------
# PATHS
# ------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
ANALYTICS_DIR = BASE_DIR / "analytics"
DB_PATH = ANALYTICS_DIR / "destinations.db"
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------
# DB CONNECTION
# ------------------------------------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


# ------------------------------------------------------
# CREATE TABLE (NOW DIRECTORY EXISTS!)
# ------------------------------------------------------
with get_conn() as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS destination_clicks (
            iata TEXT PRIMARY KEY,
            city TEXT,
            country TEXT,
            total_clicks INTEGER NOT NULL DEFAULT 0,
            last_clicked TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

with get_conn() as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analytics_daily (
            date   TEXT,
            iata   TEXT,
            clicks INTEGER,
            PRIMARY KEY (date, iata)
        )
    """)
    conn.commit()
# ------------------------------------------------------
# Pydantic Model
# ------------------------------------------------------
class ClickEvent(BaseModel):
    iata: str
    city: str
    country: str


def save_daily_snapshot():
    """Store the current total_clicks into analytics_daily for today."""
    today = datetime.now().strftime("%Y-%m-%d")

    with get_conn() as conn:
        rows = conn.execute("""
            SELECT iata, total_clicks
            FROM destination_clicks
        """).fetchall()

        for iata, clicks in rows:
            conn.execute("""
                INSERT OR REPLACE INTO analytics_daily (date, iata, clicks)
                VALUES (?, ?, ?)
            """, (today, iata, clicks))  # correct order: date, iata, clicks

        conn.commit()

# ------------------------------------------------------
# API: Track click
# ------------------------------------------------------
@router.post("/api/analytics/click")
def track_click(event: ClickEvent):
    iata = event.iata.upper().strip()
    city = event.city.strip()
    country = event.country.strip()

    if len(iata) != 3:
        raise HTTPException(status_code=400, detail="Invalid IATA code")

    today = datetime.now().strftime("%Y-%m-%d")

    with get_conn() as conn:

        # lifetime
        conn.execute("""
            INSERT INTO destination_clicks (iata, city, country, total_clicks, last_clicked)
            VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
            ON CONFLICT(iata)
            DO UPDATE SET
                total_clicks = total_clicks + 1,
                city = excluded.city,
                country = excluded.country,
                last_clicked = CURRENT_TIMESTAMP
        """, (iata, city, country))

        # DAILY realtime trending counter
        conn.execute("""
            INSERT INTO analytics_daily (date, iata, clicks)
            VALUES (?, ?, 1)
            ON CONFLICT(date, iata)
            DO UPDATE SET clicks = clicks + 1
        """, (today, iata))

        conn.commit()

    return {"status": "ok", "iata": iata}



# ------------------------------------------------------
# API: Get Top Destinations
# ------------------------------------------------------
@router.get("/api/analytics/top")
def get_top(limit: int = 20):
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT iata, city, country, total_clicks, last_clicked
            FROM destination_clicks
            ORDER BY total_clicks DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return [
        {
            "iata": r[0],
            "city": r[1],
            "country": r[2],
            "total_clicks": r[3],
            "last_clicked": r[4]
        }
        for r in rows
    ]


# ------------------------------------------------------
# API: Reset Stats (optional)
# ------------------------------------------------------
@router.get("/analytics/reset")
def reset_stats():
    with get_conn() as conn:
        conn.execute("DELETE FROM destination_clicks")
        conn.execute("DELETE FROM analytics_daily")
        conn.commit()
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

