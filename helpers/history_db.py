import sqlite3
import logging
import threading
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from config_paths import DATA_DIR, HISTORY_DB_PATH

logger = logging.getLogger("fly_tlv.history_db")

DB_PATH = HISTORY_DB_PATH

class DatabaseManager:
    """Production-grade SQLite Connection Manager with WAL mode and performance optimizations."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatabaseManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.init_db()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections with performance pragmas."""
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            # Performance & Concurrency Tuning
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            conn.execute("PRAGMA temp_store=MEMORY")
            yield conn
        finally:
            conn.close()

    def init_db(self):
        """Initialize the SQLite database and create tables if they don't exist."""
        DATA_DIR.mkdir(exist_ok=True)
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Table for tracking individual fetch requests
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fetch_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    count INTEGER
                )
            """)

            # Table for storing individual flight records
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS flights_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id INTEGER,
                    timestamp DATETIME,
                    airline_code TEXT,
                    flight_number TEXT,
                    airline_name TEXT,
                    scheduled_time TEXT,
                    actual_time TEXT,
                    direction TEXT,
                    airport_iata TEXT,
                    airport_name TEXT,
                    airport_name_he TEXT,
                    city_name TEXT,
                    city_code TEXT,
                    country_name TEXT,
                    terminal INTEGER,
                    checkin_counter TEXT,
                    checkin_zone TEXT,
                    status_en TEXT,
                    status_he TEXT,
                    FOREIGN KEY (request_id) REFERENCES fetch_requests(id)
                )
            """)

            # Indexes for high performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON fetch_requests(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_flights_request_id ON flights_history(request_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_flights_timestamp ON flights_history(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_flights_iata ON flights_history(airport_iata)")

            conn.commit()
        logger.info(f"💾 Production-grade Flight History DB initialized at {DB_PATH}")

# Global singleton instance
db_manager = DatabaseManager()

def save_flight_snapshot(records: list):
    """Save a snapshot of raw JSON records into the database, skipping if identical to latest."""
    if not records:
        return

    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # --- De-duplication Check ---
            cursor.execute("SELECT id FROM fetch_requests ORDER BY id DESC LIMIT 1")
            last_req = cursor.fetchone()
            
            if last_req:
                last_id = last_req["id"]
                cursor.execute("""
                    SELECT 
                        airline_code, flight_number, airline_name, scheduled_time, 
                        actual_time, direction, airport_iata, airport_name, 
                        airport_name_he, city_name, city_code, country_name, 
                        terminal, checkin_counter, checkin_zone, status_en, status_he
                    FROM flights_history 
                    WHERE request_id = ?
                    ORDER BY id ASC
                """, (last_id,))
                db_records = [dict(r) for r in cursor.fetchall()]

                new_mapped = [
                    {
                        "airline_code": rec.get("CHOPER"),
                        "flight_number": rec.get("CHFLTN"),
                        "airline_name": rec.get("CHOPERD"),
                        "scheduled_time": rec.get("CHSTOL"),
                        "actual_time": rec.get("CHPTOL"),
                        "direction": rec.get("CHAORD"),
                        "airport_iata": rec.get("CHLOC1"),
                        "airport_name": rec.get("CHLOC1D"),
                        "airport_name_he": rec.get("CHLOC1TH"),
                        "city_name": rec.get("CHLOC1T"),
                        "city_code": rec.get("CHLOC1CH"),
                        "country_name": rec.get("CHLOCCT"),
                        "terminal": rec.get("CHTERM"),
                        "checkin_counter": rec.get("CHCINT"),
                        "checkin_zone": rec.get("CHCKZN"),
                        "status_en": rec.get("CHRMINE"),
                        "status_he": rec.get("CHRMINH")
                    }
                    for rec in records
                ]

                def sort_key(x): return (x.get("flight_number") or "", x.get("scheduled_time") or "")
                if sorted(db_records, key=sort_key) == sorted(new_mapped, key=sort_key):
                    logger.info("⏭️  Skipping update: Data is identical to snapshot.")
                    return last_id

            # --- Proceed with Insertion ---
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                "INSERT INTO fetch_requests (timestamp, count) VALUES (?, ?)",
                (now_str, len(records))
            )
            request_id = cursor.lastrowid

            insert_query = """
                INSERT INTO flights_history (
                    request_id, timestamp, airline_code, flight_number, airline_name, 
                    scheduled_time, actual_time, direction, airport_iata, airport_name, 
                    airport_name_he, city_name, city_code, country_name, terminal, 
                    checkin_counter, checkin_zone, status_en, status_he
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            data_to_insert = [
                (
                    request_id, now_str,
                    rec.get("CHOPER"), rec.get("CHFLTN"), rec.get("CHOPERD"),
                    rec.get("CHSTOL"), rec.get("CHPTOL"), rec.get("CHAORD"),
                    rec.get("CHLOC1"), rec.get("CHLOC1D"), rec.get("CHLOC1TH"),
                    rec.get("CHLOC1T"), rec.get("CHLOC1CH"), rec.get("CHLOCCT"),
                    rec.get("CHTERM"), rec.get("CHCINT"), rec.get("CHCKZN"),
                    rec.get("CHRMINE"), rec.get("CHRMINH")
                )
                for rec in records
            ]

            cursor.executemany(insert_query, data_to_insert)
            conn.commit()
            logger.info(f"✅ Saved snapshot (ID: {request_id}, Records: {len(records)})")
            return request_id

    except Exception as e:
        logger.error(f"❌ DB storage failed: {e}", exc_info=True)
        return None

def get_available_snapshots(limit: int = 100):
    """Get a list of available fetch request snapshots for history filtering."""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, timestamp, count FROM fetch_requests ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error fetching available snapshots: {e}", exc_info=True)
        return []

def get_flight_stats_from_db(request_id: int = None):
    """Fetch flight statistics for a specific snapshot or the latest one."""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            if request_id:
                cursor.execute("SELECT id, timestamp FROM fetch_requests WHERE id = ?", (request_id,))
            else:
                cursor.execute("SELECT id, timestamp FROM fetch_requests ORDER BY timestamp DESC LIMIT 1")
            
            row = cursor.fetchone()
            
            if not row:
                return None, None
                
            active_request_id = row["id"]
            timestamp = row["timestamp"]
            
            cursor.execute("""
                SELECT airline_name, city_name, country_name, direction, airline_code 
                FROM flights_history 
                WHERE request_id = ?
            """, (active_request_id,))
            
            flights = [dict(r) for r in cursor.fetchall()]
            return flights, timestamp
    except Exception as e:
        logger.error(f"Error fetching stats: {e}", exc_info=True)
        return None, None
