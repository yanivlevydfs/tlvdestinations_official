import os
import re
import json
import requests
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

API_KEY = "da33f3-b78037"
api_url = f"https://aviation-edge.com/v2/public/airlineDatabase?key={API_KEY}"

output_dir = "airline_logos"
os.makedirs(output_dir, exist_ok=True)

base_logo_url = "https://uds.xplorer.com/img/airlines_logos/"
file_lock = Lock()

json_path = "airlines.json"

# ✅ Clean airline name for safe filenames
def clean_name(name: str) -> str:
    name = name.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r'[\\/:*?"<>|]', "", name)
    name = re.sub(r"\s{2,}", " ", name)
    return name.strip()

# ✅ Download logo if not already saved
def download_logo(airline):
    iata = airline.get("codeIataAirline", "").strip().upper()
    name = airline.get("nameAirline", "").strip()

    if not iata or not name:
        return "⏭ Skipped: missing IATA or name"

    filename = f"{clean_name(name)}.png"
    path = os.path.join(output_dir, filename)

    with file_lock:
        if os.path.exists(path):
            return f"✅ Already exists: {filename}"

    logo_url = f"{base_logo_url}{iata.lower()}_small.png"

    try:
        resp = requests.get(logo_url, timeout=10)
        if resp.status_code == 200 and resp.content:
            with file_lock:
                if not os.path.exists(path):  # double-check in case saved during wait
                    with open(path, "wb") as f:
                        f.write(resp.content)
            return f"✔ Saved: {filename}"
        else:
            return f"⚠ Not found: {name} ({iata})"
    except requests.RequestException as e:
        return f"❌ Error for {name}: {e}"

# 🔁 Load from cache if possible, else fetch from API
if os.path.exists(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            airlines = json.load(f)
        print(f"📂 Loaded {len(airlines)} airlines from cache.\n")
    except Exception as e:
        print(f"❌ Failed to load cache: {e}")
        exit(1)
else:
    try:
        resp = requests.get(api_url, timeout=15)
        resp.raise_for_status()
        airlines = resp.json()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(airlines, f, ensure_ascii=False, indent=2)
        print(f"🌐 Fetched {len(airlines)} airlines from API.\n")
    except Exception as e:
        print("❌ Error fetching airline data:", e)
        exit(1)

# 🚫 Filter out airlines that already have a logo
airlines_to_download = []
for airline in airlines:
    name = airline.get("nameAirline", "").strip()
    if not name:
        continue
    filename = f"{clean_name(name)}.png"
    path = os.path.join(output_dir, filename)
    if not os.path.exists(path):
        airlines_to_download.append(airline)

print(f"🔁 Starting downloads for {len(airlines_to_download)} new logos...\n")

# 🚀 Download logos in parallel
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(download_logo, airline) for airline in airlines_to_download]
    for future in as_completed(futures):
        print(future.result())

print("\n✅ All done!")
