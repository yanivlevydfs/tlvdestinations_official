import json
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Dict, Tuple

from config_paths import CACHE_DIR, ISRAEL_FLIGHTS_FILE

DEST_PREVIOUS = CACHE_DIR / "destinations_previous.json"
DEST_DIFF     = CACHE_DIR / "destinations_diff.json"


# ===============================================================
# SAFE JSON LIST LOADER
# ===============================================================
def _load_list(path: Path) -> List[Dict]:
    """Returns a list. If invalid/missing, returns empty list."""
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        # If someone accidentally replaced file with {} or string
        return []
    except Exception:
        return []


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ===============================================================
# ROUTE KEY (airline + IATA)
# ===============================================================
def _route_key(rec: Dict) -> Tuple[str, str]:
    airline = (rec.get("airline") or "").strip().lower()
    iata    = (rec.get("iata") or "").strip().upper()
    return airline, iata


# ===============================================================
# EXTRACT ROUTES FROM israel_flights.json
# ===============================================================
def extract_routes(path: Path) -> List[Dict]:
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return []

    flights = raw.get("flights", [])
    routes = {}

    for rec in flights:
        key = _route_key(rec)
        airline, iata = key

        if not airline or not iata:
            continue

        # Initialize route bucket ONCE
        if key not in routes:
            routes[key] = {
                "airline": airline.title(),  # keep as you requested
                "iata": iata,
                "airport": rec.get("airport", ""),
                "city": rec.get("city", ""),
                "country": rec.get("country", ""),
                "direction": None,
                "status": None,
                "scheduled": None,
                "actual": None,
                "_has_departure": False
            }

        # Mark route as valid if ANY departure exists
        if rec.get("direction") == "D":
            routes[key]["_has_departure"] = True

            # Use the FIRST departure record as canonical metadata
            if routes[key]["direction"] is None:
                routes[key].update({
                    "direction": "D",
                    "status": rec.get("status", ""),
                    "scheduled": rec.get("scheduled", ""),
                    "actual": rec.get("actual", "")
                })

    # Emit only routes that actually have departures
    result = []
    for r in routes.values():
        if r["_has_departure"]:
            r.pop("_has_departure", None)
            result.append(r)

    return sorted(
        result,
        key=lambda r: (r["airline"].lower(), r["iata"])
    )



# ===============================================================
# SNAPSHOT INITIALIZATION (VERY IMPORTANT)
# ===============================================================
def ensure_previous_snapshot():
    """
    On first-ever run OR when snapshot accidentally empty,
    the current israel_flights.json is used as baseline.
    """
    prev = _load_list(DEST_PREVIOUS)
    if prev:
        return  # snapshot already exists and is valid

    curr = extract_routes(ISRAEL_FLIGHTS_FILE)
    _save_json(DEST_PREVIOUS, curr)
    #print("FIX APPLIED: Auto-created destinations_previous.json")



# ===============================================================
# DIFF ENGINE
# ===============================================================
def compute_diff(prev: List[Dict], curr: List[Dict]) -> Dict:
    prev_map = {_route_key(r): r for r in prev}
    curr_map = {_route_key(r): r for r in curr}

    prev_keys = set(prev_map.keys())
    curr_keys = set(curr_map.keys())

    added_keys   = sorted(curr_keys - prev_keys)
    removed_keys = sorted(prev_keys - curr_keys)

    added   = [curr_map[k] for k in added_keys]
    removed = [prev_map[k] for k in removed_keys]

    # Detect cancelled flights from current dataset
    cancelled = []
    for r in curr:
        status = (r.get("status") or "").strip().lower()
        if "cancel" in status or "בוטל" in status:
            cancelled.append(r)

    # Sort cancelled flights by airline and city
    cancelled = sorted(cancelled, key=lambda r: (r.get("airline", "").lower(), r.get("city", "")))

    return {
        "generated": datetime.now(UTC).isoformat(),
        "counts": {
            "previous": len(prev),
            "current": len(curr),
            "added": len(added),
            "removed": len(removed),
            "cancelled": len(cancelled),
        },
        "added": added,
        "removed": removed,
        "cancelled": cancelled,
    }


# ===============================================================
# MAIN FUNCTION
# ===============================================================
def generate_destination_diff() -> Dict:
    ensure_previous_snapshot()

    prev = _load_list(DEST_PREVIOUS)
    curr = extract_routes(ISRAEL_FLIGHTS_FILE)

    diff = compute_diff(prev, curr)

    _save_json(DEST_DIFF, diff)

    return diff


# ===============================================================
# DEBUGGING
# ===============================================================
def debug_print_routes():
    prev = _load_list(DEST_PREVIOUS)
    curr = extract_routes(ISRAEL_FLIGHTS_FILE)

    print("\n===== ROUTE DEBUG =====")
    print(f"Prev routes: {len(prev)}")
    print(f"Curr routes: {len(curr)}")

    print("\nPREV KEYS:")
    for r in prev:
        print("   ", _route_key(r))

    print("\nCURR KEYS:")
    for r in curr:
        print("   ", _route_key(r))


if __name__ == "__main__":
    print("RUNNING DIFF…")
    print(json.dumps(generate_destination_diff(), indent=2, ensure_ascii=False))
