from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.status import HTTP_403_FORBIDDEN
import re

SAFE_PATHS = (
    "/favicon.ico", "/favicon.svg", "/robots.txt", "/sitemap.xml",
    "/feed.xml", "/.well-known/traffic-advice"
)
SAFE_PATH_PREFIXES = (
    "/static/", "/assets/", "/css/", "/js/", "/fonts/", "/images/",
    "/icons/", "/og/", "/logos/", "/.well-known/"
)

SUSPICIOUS_PATTERNS = [
    r"(^|/)phpinfo(\.php)?$",
    r"(^|/)(index|config|env|setup)\.php$",
    r"\.php$",
    r"wp-(admin|login|config|includes)",
    r"(wlwmanifest\.xml|xmlrpc\.php)",
    r"\.(env|git|svn|bak|old|tmp|log|sql|db)$",
    r"(token|secret|key|credentials)[_.-]?(id|key)?",
    r"\.(zip|tar|gz|7z|rar)$",
]
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SUSPICIOUS_PATTERNS]

BAD_USER_AGENTS = [
    "curl", "wget", "python-requests", "nikto", "fimap", "sqlmap", "nmap",
    "scanner", "spider", "bot", "fetch", "httpx", "libwww", "scrapy",
]

GOOD_BOTS = [
    "googlebot", "bingbot", "yandex", "baiduspider", "duckduckbot",
    "facebookexternalhit", "twitterbot", "applebot", "chatgpt",
    "openai", "msnbot", "slurp"
]

class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path.lower()
        user_agent = request.headers.get("user-agent", "").lower()
        client_ip = request.client.host

        if request.method in ("HEAD", "OPTIONS"):
            return await call_next(request)

        if path in SAFE_PATHS or any(path.startswith(p) for p in SAFE_PATH_PREFIXES):
            return await call_next(request)

        # ✅ Allow if from a known good bot
        if any(bot in user_agent for bot in GOOD_BOTS):
            return await call_next(request)

        for pattern in COMPILED_PATTERNS:
            if pattern.search(path):                
                return JSONResponse({"detail": "Forbidden"}, status_code=HTTP_403_FORBIDDEN)

        for bad_ua in BAD_USER_AGENTS:
            if bad_ua in user_agent:
                return JSONResponse({"detail": "Forbidden"}, status_code=HTTP_403_FORBIDDEN)

        return await call_next(request)
