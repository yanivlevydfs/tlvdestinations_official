from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
from pathlib import Path

# Correct imports – do NOT import ISRAEL_FLIGHTS_FILE from destination_diff
from config_paths import ISRAEL_FLIGHTS_FILE

from helpers.destination_diff import (
    generate_destination_diff,
    debug_print_routes,
    extract_routes,
    _save_json,
    DEST_DIFF,
    DEST_PREVIOUS,
)

logger = logging.getLogger("destination_diff")

router = APIRouter(tags=["Destination Diff"])


# ================================================================
# Pydantic Models (clean Swagger docs)
# ================================================================
class RouteRecord(BaseModel):
    airline: str
    iata: str
    airport: str
    city: str
    country: str
    direction: str
    status: str
    scheduled: str
    actual: str


class DiffCounts(BaseModel):
    previous: int
    current: int
    added: int
    removed: int
    cancelled: int


class DiffResponse(BaseModel):
    generated: str
    counts: DiffCounts
    added: list[RouteRecord]
    removed: list[RouteRecord]
    cancelled: list[RouteRecord]


# ================================================================
# API ROUTE
# ================================================================
@router.get(
    "/api/destinations/diff",
    response_model=DiffResponse,
    summary="Return added/removed airline–destination routes",
    description=(
        "**Detects route changes between snapshots:**\n"
        "- Newly added airline routes\n"
        "- Removed airline routes\n\n"
        "**Optional parameters:**\n"
        "`?debug=1` – Print debug info into logs\n"
        "`?refresh_snapshot=1` – REBUILD baseline snapshot BEFORE diff (danger!)"
    ),
)
def api_destination_diff(
    debug: int = Query(default=0, ge=0, le=1),
    refresh_snapshot: int = Query(default=0, ge=0, le=1),
):
    try:
        # ---------------------------------------------------------
        # Debug Mode
        # ---------------------------------------------------------
        if debug == 1:
            logger.warning("DEBUG MODE → Printing route tables")
            debug_print_routes()

        # ---------------------------------------------------------
        # Snapshot Refresh (manual override)
        # ---------------------------------------------------------
        if refresh_snapshot == 1:
            logger.warning("FORCED SNAPSHOT REFRESH REQUESTED")

            if not Path(ISRAEL_FLIGHTS_FILE).exists():
                raise HTTPException(
                    status_code=500,
                    detail="Cannot refresh snapshot: israel_flights.json missing.",
                )

            curr = extract_routes(Path(ISRAEL_FLIGHTS_FILE))
            _save_json(DEST_PREVIOUS, curr)

            logger.debug(
                f"Snapshot rebuilt successfully → {len(curr)} routes stored."
            )

        # ---------------------------------------------------------
        # Compute Diff
        # ---------------------------------------------------------
        diff_raw = generate_destination_diff()

        logger.debug(
            f"DIFF GENERATED → added={diff_raw['counts']['added']} "
            f"removed={diff_raw['counts']['removed']}"
        )

        # ---------------------------------------------------------
        # Convert raw dict to Pydantic format for FastAPI
        # ---------------------------------------------------------
        diff_response = DiffResponse(
            generated=diff_raw["generated"],
            counts=DiffCounts(**diff_raw["counts"]),
            added=[RouteRecord(**rec) for rec in diff_raw["added"]],
            removed=[RouteRecord(**rec) for rec in diff_raw["removed"]],
            cancelled=[RouteRecord(**rec) for rec in diff_raw["cancelled"]],
        )

        return diff_response

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Fatal failure in diff generation")
        raise HTTPException(status_code=500, detail=str(e))
