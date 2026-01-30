import os
import json
import requests
from io import BytesIO
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
API_KEY = "da33f3-b78037"
API_URL = f"https://aviation-edge.com/v2/public/airlineDatabase?key={API_KEY}"

BASE_LOGO_URL = "https://uds.xplorer.com/img/airlines_logos/"
OUTPUT_DIR = "airline_logos"
CACHE_JSON = "airlines.json"

MAX_WORKERS = 20
WEBP_QUALITY = 85
WEBP_METHOD = 6  # best compression

# ───────────────────────────────────────────────
# INIT
# ───────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
file_lock = Lock()

# ───────────────────────────────────────────────
# DOWNLOAD + CONVERT (THREAD SAFE)
# ───────────────────────────────────────────────
def download_logo(airline: dict) -> str:
    iata = airline.get("codeIataAirline", "").strip().lower()

    if not iata:
        return "⏭ Skipped: missing IATA"

    filename = f"{iata}.webp"
    path = os.path.join(OUTPUT_DIR, filename)

    # First fast existence check
    with file_lock:
        if os.path.exists(path):
            return f"✅ Already exists: {filename}"

    logo_url = f"{BASE_LOGO_URL}{iata}_small.png"

    try:
        resp = requests.get(logo_url, timeout=10)
        if resp.status_code != 200 or not resp.content:
            return f"⚠ Not found: {iata.upper()}"

        # Convert PNG → WebP fully in memory
        image = Image.open(BytesIO(resp.content)).convert("RGBA")

        # Second guarded write
        with file_lock:
            if not os.path.exists(path):
                image.save(
                    path,
                    format="WEBP",
                    quality=WEBP_QUALITY,
                    method=WEBP_METHOD,
                    lossless=False
                )

        return f"✔ Saved: {filename}"

    except Exception as e:
        return f"❌ Error for {iata.upper()}: {e}"

# ───────────────────────────────────────────────
# LOAD AIRLINES (CACHE → API)
# ───────────────────────────────────────────────
if os.path.exists(CACHE_JSON):
    try:
        with open(CACHE_JSON, "r", encoding="utf-8") as f:
            airlines = json.load(f)
        print(f"📂 Loaded {len(airlines)} airlines from cache.\n")
    except Exception as e:
        print(f"❌ Failed to load cache: {e}")
        raise SystemExit(1)
else:
    try:
        resp = requests.get(API_URL, timeout=15)
        resp.raise_for_status()
        airlines = resp.json()

        with open(CACHE_JSON, "w", encoding="utf-8") as f:
            json.dump(airlines, f, ensure_ascii=False, indent=2)

        print(f"🌐 Fetched {len(airlines)} airlines from API.\n")
    except Exception as e:
        print(f"❌ Error fetching airline data: {e}")
        raise SystemExit(1)

# ───────────────────────────────────────────────
# FILTER: ONLY MISSING LOGOS
# ───────────────────────────────────────────────
airlines_to_download: list[dict] = []

for airline in airlines:
    iata = airline.get("codeIataAirline", "").strip().lower()
    if not iata:
        continue

    path = os.path.join(OUTPUT_DIR, f"{iata}.webp")
    if not os.path.exists(path):
        airlines_to_download.append(airline)

print(f"🔁 Starting downloads for {len(airlines_to_download)} new logos...\n")

# ───────────────────────────────────────────────
# PARALLEL EXECUTION
# ───────────────────────────────────────────────
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(download_logo, airline) for airline in airlines_to_download]
    for future in as_completed(futures):
        print(future.result())

print("\n✅ All done!")
