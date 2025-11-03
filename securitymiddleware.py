from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.status import HTTP_403_FORBIDDEN
import re

# ✅ Safe paths (don’t block static resources)
SAFE_PATHS = ("/favicon.ico", "/favicon.svg", "/robots.txt", "/sitemap.xml")
SAFE_PATH_PREFIXES = (
    "/static/", "/assets/", "/css/", "/js/", "/fonts/", "/images/",
    "/icons/", "/og/", "/logos/", "/.well-known/"
)

# 🛡️ Suspicious path patterns (incl. WordPress scanner defense)
SUSPICIOUS_PATTERNS = [
    r"(^|/)phpinfo(\.php)?$",                    # phpinfo
    r"(^|/)(index|config|env|setup)\.php$",      # common PHP files
    r"\.php$",                                   # any .php
    r"wp-(admin|login|config|includes)",         # WordPress core paths
    r"(wlwmanifest\.xml|xmlrpc\.php)",           # WP API endpoints
    r"\.(env|git|svn|bak|old|tmp|log|sql|db)$",  # sensitive dotfiles
    r"(token|secret|key|credentials)[_.-]?(id|key)?",  # credentials
    r"\.(zip|tar|gz|7z|rar)$",                   # archives
]
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SUSPICIOUS_PATTERNS]

# 🤖 Known bad bots/user-agents
BAD_USER_AGENTS = [
    "curl", "wget", "python-requests", "nikto", "fimap", "sqlmap", "nmap",
    "scanner", "spider", "bot", "fetch", "httpx", "libwww", "scrapy",
]

class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path.lower()
        user_agent = request.headers.get("user-agent", "").lower()
        client_ip = request.client.host

        # ⛔ HEAD/OPTIONS are safe
        if request.method in ("HEAD", "OPTIONS"):
            return await call_next(request)

        # ✅ Allow listed static & public files
        if path in SAFE_PATHS or any(path.startswith(p) for p in SAFE_PATH_PREFIXES):
            return await call_next(request)

        # 🛡️ Block bad paths
        for pattern in COMPILED_PATTERNS:
            if pattern.search(path):                
                return JSONResponse({"detail": "Forbidden"}, status_code=HTTP_403_FORBIDDEN)

        # 🤖 Block bad bots
        for bad_ua in BAD_USER_AGENTS:
            if bad_ua in user_agent:
                return JSONResponse({"detail": "Forbidden"}, status_code=HTTP_403_FORBIDDEN)

        # 👍 All good — proceed
        return await call_next(request)
