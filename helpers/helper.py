# helper_html.py
"""
HTML parsing & extraction helpers — Production grade.
"""
from fastapi import Request
import re
from bs4 import BeautifulSoup
import subprocess
import os
import string
from datetime import datetime
from geopy.distance import geodesic

def _clean_html(raw: str) -> str:
    if not raw:
        return ""
    return BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)


def _extract_first_href(raw: str) -> str:
    if not raw:
        return ""
    match = re.search(r'href="([^"]+)"', raw)
    return match.group(1) if match else ""


def _extract_first_img(raw: str) -> dict:
    if not raw:
        return {}
    match = re.search(r'<img src="([^"]+)" alt="([^"]+)"', raw)
    if match:
        return {"src": match.group(1), "alt": match.group(2)}
    return {}
    
def get_git_version():
    """Return the project version based on Git commit and date, or 'dev' if unavailable."""
    root = os.path.dirname(os.path.abspath(__file__))

    def run_git_command(args):
        return subprocess.check_output(["git"] + args, cwd=root).decode().strip()

    try:
        commit = run_git_command(["rev-parse", "--short", "HEAD"])
        date = run_git_command(["log", "-1", "--format=%cd", "--date=short"])
        return f"{date.replace('-', '.')}–{commit}"

    except FileNotFoundError:
        logger.error("[version] Git not found. Is it installed and in PATH? Falling back to 'dev'.")
    except subprocess.CalledProcessError as e:
        logger.error(f"[version] Git command failed: {e}. Falling back to 'dev'.")
    except Exception as e:
        logger.error(f"[version] Unexpected error: {e}. Falling back to 'dev'.")

    return "dev"

def normalize_case(value) -> str:
    """Capitalize each word safely, handling None, numbers, and placeholders."""
    if not value or str(value).strip() in {"", "—", "None", "nan"}:
        return "—"
    return string.capwords(str(value).strip())

def get_lang(request: Request) -> str:
    return "he" if request.query_params.get("lang") == "he" else "en"
    
def safe_js(text: str) -> str:
    """Escape backticks, quotes and newlines for safe JS embedding"""
    if text is None:
        return ""
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("`", "\\`")   # ✅ escape backticks
        .replace('"', '\\"')   # escape double quotes
        .replace("'", "\\'")   # escape single quotes
        .replace("\n", " ")
        .replace("\r", "")
    )
def get_flight_time(dist_km: float | None) -> str:
    if not dist_km or dist_km <= 0:
        return "—"

    # Adjust speed based on flight range
    if dist_km < 500:
        cruise_speed_kmh = 700
        buffer = 0.4  # 24 mins
    elif dist_km < 2000:
        cruise_speed_kmh = 800
        buffer = 0.5
    else:
        cruise_speed_kmh = 850
        buffer = 0.6  # long-haul

    estimated_time_hr = dist_km / cruise_speed_kmh + buffer

    hours = int(estimated_time_hr)
    minutes = int(round((estimated_time_hr - hours) * 60))

    if minutes == 60:
        hours += 1
        minutes = 0

    return f"{hours}h" if minutes == 0 else f"{hours}h {minutes}m"
    
def datetimeformat(value: str, fmt: str = "%d/%m/%Y %H:%M"):
    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime(fmt)
    except Exception:
        return value
        

def _extract_threat_level(text: str) -> str:
    """
    מזהה רמת איום מתוך ההמלצות
    מחזיר High / Medium / Low / Unknown
    """
    if not text:
        return "Unknown"
    t = text.strip()
    if "רמה 4" in t or "גבוה" in t:
        return "High"
    if "רמה 3" in t or "בינוני" in t:
        return "Medium"
    if "רמה 2" in t or "נמוך" in t:
        return "Low"
    return "Unknown"
    
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return round(geodesic((lat1, lon1), (lat2, lon2)).kilometers, 1)