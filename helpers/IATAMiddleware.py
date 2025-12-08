from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import json
from pathlib import Path
import ijson

# Path to your flights JSON
ISRAEL_FLIGHTS_FILE = Path("cache") / "israel_flights.json"

# Load once into memory
def load_valid_iatas_fast(file_path):
    valid = set()

    try:
        with open(file_path, 'rb') as f:
            # Stream only the "flights.item.iata" entries
            for iata in ijson.items(f, "flights.item.iata"):
                if iata:
                    valid.add(iata.upper())

    except Exception as e:
        print("IATA scan failed:", e)

    return valid

VALID_IATAS = load_valid_iatas_fast(ISRAEL_FLIGHTS_FILE)


class IATAMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.url.path

        # Only handle /destinations/<iata>
        if path.startswith("/destinations/"):
            parts = path.split("/")
            if len(parts) >= 3:
                iata = parts[2].upper().strip()

                # Skip TLV aliases
                if iata in {"TLV", "LLBG", "BEN-GURION"}:
                    return await call_next(request)

                # If IATA not in our cached list → auto 404
                if iata not in VALID_IATAS:
                    return Response(
                        content=f"Invalid IATA code: {iata}",
                        status_code=404,
                        media_type="text/plain"
                    )

        return await call_next(request)
