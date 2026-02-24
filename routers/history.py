from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import sqlite3
import logging
from helpers.history_db import DB_PATH

logger = logging.getLogger("fly_tlv.history_router")
router = APIRouter()

@router.get("/api/history", include_in_schema=False)
async def get_history_list(limit: int = 50):
    """Get a list of recent fetch requests."""
    try:
        from helpers.history_db import db_manager
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, timestamp, count FROM fetch_requests ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching history list: {e}")
        return JSONResponse({"error": "Failed to fetch history"}, status_code=500)

@router.get("/api/history/{request_id}", include_in_schema=False)
async def get_history_detail(request_id: int):
    """Get all flight records for a specific fetch request."""
    try:
        from helpers.history_db import db_manager
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if request exists - Explicit columns
            cursor.execute("SELECT id, timestamp, count FROM fetch_requests WHERE id = ?", (request_id,))
            request_info = cursor.fetchone()
            
            if not request_info:
                raise HTTPException(status_code=404, detail="History request not found")
                
            # Explicit columns for flights_history
            cursor.execute("""
                SELECT 
                    id, request_id, timestamp, airline_code, flight_number, airline_name, 
                    scheduled_time, actual_time, direction, airport_iata, airport_name, 
                    airport_name_he, city_name, city_code, country_name, terminal, 
                    checkin_counter, checkin_zone, status_en, status_he 
            FROM flights_history 
            WHERE request_id = ?
            """, (request_id,))
            rows = cursor.fetchall()
            
            return {
                "request": dict(request_info),
                "flights": [dict(row) for row in rows]
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching history detail for {request_id}: {e}")
        return JSONResponse({"error": "Failed to fetch history detail"}, status_code=500)
