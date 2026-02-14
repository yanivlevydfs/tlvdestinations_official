import csv
import json
import requests

OPENFLIGHTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat"

# Your curated dataset (has is_lowcost)
LOCAL_AIRLINES_ALL = "airlines_all.json"

# Common LCC brand keywords (heuristic fallback)
LCC_KEYWORDS = [
    "ryanair", "easyjet", "wizz", "scoot", "peach", "airasia", "volotea",
    "vueling", "flynas", "flyadeal", "jetstar", "spirit", "frontier",
    "allegiant", "indigo", "gol", "lion air", "cebu pacific", "tigerair",
    "norwegian", "eurowings", "transavia", "smartwings"
]

# Some high-confidence LCC IATA codes (heuristic fallback)
LCC_IATA = {
    "FR","U2","W6","W4","TR","MM","AK","D7","VY","V7","PC","FZ","G9","JQ","GK","3K","NK","F9","G4","B6","OD","XQ","EW"
}

def load_overrides():
    try:
        with open(LOCAL_AIRLINES_ALL, "r", encoding="utf-8") as f:
            data = json.load(f)
        # map by IATA code
        return {a.get("iata"): a.get("is_lowcost") for a in data if a.get("iata")}
    except FileNotFoundError:
        return {}

def fetch_openflights():
    resp = requests.get(OPENFLIGHTS_URL, timeout=30)
    resp.raise_for_status()

    airlines = []
    for row in csv.reader(resp.text.splitlines()):
        if len(row) < 6:
            continue

        name = row[1].strip()
        iata = row[3] if row[3] != "\\N" else None
        icao = row[4] if row[4] != "\\N" else None

        # Skip junk: no name, no codes
        if not name or (iata is None and icao is None):
            continue

        airlines.append({
            "iata": iata,
            "icao": icao,
            "name": name,
            "is_lowcost": None,          # will be enriched
            "operator_type": "unknown",  # keep unknown (don’t lie)
            "charter_capable": None      # keep unknown (don’t lie)
        })
    return airlines

def dedupe(airlines):
    seen = {}
    for a in airlines:
        key = a["iata"] or a["icao"] or a["name"]
        seen[key] = a
    return list(seen.values())

def enrich_lowcost(airlines, overrides):
    for a in airlines:
        iata = a.get("iata")
        name_l = (a.get("name") or "").lower()

        # 1) Your authoritative override
        if iata and iata in overrides and overrides[iata] is not None:
            a["is_lowcost"] = bool(overrides[iata])
            continue

        # 2) Heuristic fallback
        if iata in LCC_IATA:
            a["is_lowcost"] = True
            continue

        if any(k in name_l for k in LCC_KEYWORDS):
            a["is_lowcost"] = True
            continue

        # 3) Unknown stays None (better than wrong false)
        a["is_lowcost"] = None

def main():
    overrides = load_overrides()
    airlines = fetch_openflights()
    airlines = dedupe(airlines)
    enrich_lowcost(airlines, overrides)

    with open("airlines_master.json", "w", encoding="utf-8") as f:
        json.dump(airlines, f, indent=2, ensure_ascii=False)

    known = sum(a["is_lowcost"] is not None for a in airlines)
    print(f"OK → generated {len(airlines)} airlines. Low-cost known for {known}.")

if __name__ == "__main__":
    main()
