from fastapi import FastAPI, Request, Response, Query, HTTPException, Depends, Body
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

security = HTTPBasic()
# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def verify_docs_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = os.getenv("DOCS_USER", "admin")
    correct_password = os.getenv("DOCS_PASS", "secret123")
    is_user = secrets.compare_digest(credentials.username, correct_username)
    is_pass = secrets.compare_digest(credentials.password, correct_password)
    if not (is_user and is_pass):
        logger.warning(f"Unauthorized docs access attempt from {credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
    
def datetimeformat(value: str, fmt: str = "%d/%m/%Y %H:%M"):
    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime(fmt)
    except Exception:
        return value

def normalize_case(value: str) -> str:
    """Capitalize each word properly, safe for missing/placeholder values."""
    if not value or value == "—":
        return value or "—"
    return string.capwords(value.strip())

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