# logging_setup.py
from __future__ import annotations
import logging
import logging.config
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable, List, Optional

# נתיבים שנסנן מהלוג של uvicorn.access (אפשר להרחיב/לשנות ב-setup_logging)
DEFAULT_IGNORED_ACCESS_PATHS: List[str] = [
    ".well-known/appspecific/com.chrome.devtools.json",  # DevTools
    "/api/progress/stream",                              # SSE / health
    "/favicon.ico",
    "/static/",                                          # סטטיקה מרובה
]

class IgnorePathsFilter(logging.Filter):
    """מסנן החוצה שורות לוג שמכילות אחד מהנתיבים הנתונים."""
    def __init__(self, ignored_paths: Optional[Iterable[str]] = None):
        super().__init__()
        self.ignored_paths = list(ignored_paths or [])

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in self.ignored_paths)

def setup_logging(
    log_level: str = "INFO",
    log_dir: Path | str = "logs",
    log_file_name: str = "app.log",
    ignored_access_paths: Optional[Iterable[str]] = None,
    file_max_bytes: int = 5 * 1024 * 1024,  # 5MB
    file_backup_count: int = 3,
) -> None:
    """
    מפעיל קונפיגורציית לוגים ל-FastAPI/Uvicorn + אפליקציה:
    - קונסול עם צבעים (אם colorlog מותקן)
    - קובץ לוג עם רוטציה
    - סינון נתיבי גישה מציקים (uvicorn.access)
    - השקטת מודולים רועשים
    """
    # יצירת תיקיית לוגים
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(log_dir / log_file_name)

    # ניסוי טעינת colorlog (אופציונלי)
    try:
        import colorlog  # type: ignore
        use_colorlog = True
    except Exception:
        use_colorlog = False

    fmt_plain = "%(levelname)s | %(asctime)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Formatter-ים
    formatters = {
        "plain": {
            "format": fmt_plain,
            "datefmt": datefmt,
        }
    }
    if use_colorlog:
        formatters["color"] = {
            "()": "colorlog.ColoredFormatter",
            "format": "%(log_color)s%(levelname)-8s%(reset)s | %(asctime)s | %(name)s | %(message)s",
            "datefmt": datefmt,
            "log_colors": {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        }

    # Handlers
    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "level": log_level,
            "filters": ["ignore_paths"],
            "formatter": "color" if use_colorlog else "plain",
        },
        "file": {
            "()": RotatingFileHandler,
            "level": "INFO",
            "filename": log_file,
            "maxBytes": file_max_bytes,
            "backupCount": file_backup_count,
            "encoding": "utf-8",
            "formatter": "plain",
        },
    }

    # Filters
    ignored_paths = list(ignored_access_paths or DEFAULT_IGNORED_ACCESS_PATHS)
    filters = {
        "ignore_paths": {
            "()": IgnorePathsFilter,
            "ignored_paths": ignored_paths,
        }
    }

    # הגדרות לוגרים
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "filters": filters,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": {
            # לוג של בקשות HTTP (GET /path 200 OK)
            "uvicorn.access": {
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            # לוג פנימי של uvicorn/שרת
            "uvicorn.error": {
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            # לוגר האפליקציה שלך: logging.getLogger("app")
            "app": {
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            # השקטת ספריות רועשות (אופציונלי)
            "asyncio": {"level": "WARNING"},
            "httpx": {"level": "WARNING"},
            "urllib3": {"level": "WARNING"},
            "watchfiles": {"level": "WARNING"},
        },
        "root": {  # כל השאר
            "level": log_level,
            "handlers": ["console", "file"],
        },
    })

    logging.getLogger("app").info("Logging configured ✅")

def get_app_logger(name: str = "app") -> logging.Logger:
    """מחזיר לוגר לשימוש באפליקציה."""
    return logging.getLogger(name)
