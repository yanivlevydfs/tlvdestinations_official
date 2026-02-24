import aiosqlite
import logging
import asyncio
from contextlib import asynccontextmanager
from config_paths import ANALYTICS_DB_PATH

logger = logging.getLogger("fly_tlv.analytics_db")

class AnalyticsDatabaseManager:
    """Production-grade Async SQLite Connection Manager for Analytics."""
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AnalyticsDatabaseManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        # We don't init_db here because it's async

    @asynccontextmanager
    async def get_connection(self) -> aiosqlite.Connection:
        """Async context manager that returns a connection and ensures it is closed."""
        conn = await aiosqlite.connect(ANALYTICS_DB_PATH, timeout=30)
        try:
            # Performance & Concurrency Tuning
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA cache_size=-32000")  # 32MB cache
            await conn.execute("PRAGMA temp_store=MEMORY")
            await conn.execute("PRAGMA busy_timeout=5000")
            yield conn
        finally:
            await conn.close()

    async def init_db(self):
        """Initialize the analytics database schema."""
        async with self.get_connection() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS destination_clicks (
                    iata TEXT PRIMARY KEY,
                    city TEXT,
                    country TEXT,
                    total_clicks INTEGER NOT NULL DEFAULT 0,
                    last_clicked TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS analytics_daily (
                    date   TEXT,
                    iata   TEXT,
                    clicks INTEGER,
                    PRIMARY KEY (date, iata)
                )
            """)
            await db.commit()
            logger.info("Analytics database initialized at %s", ANALYTICS_DB_PATH)

analytics_db = AnalyticsDatabaseManager()
