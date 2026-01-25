from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.status import HTTP_403_FORBIDDEN
import re


# -----------------------------------------
# SAFE PATHS & SAFE PREFIXES
# -----------------------------------------

SAFE_PATHS = (
    "/",
    "/about",
    "/accessibility",
    "/chat",
    "/contact",
    "/direct-vs-nonstop",
    "/glossary",
    "/map",
    "/privacy",
    "/stats",
    "/terms",
    "/travel-warnings",
    "/tlv-shops", 

    "/favicon.ico", "/favicon.svg",
    "/robots.txt", "/sitemap.xml",
    "/feed.xml",
    "/manifest.json",
    "/manifest.en.json",
    "/manifest.he.json",
    "/sw.js",
    "/.well-known/traffic-advice",
    "/.well-known/assetlinks.json",
)

SAFE_PATH_PREFIXES = (
    "/static/", "/assets/", "/css/", "/js/", "/fonts/", "/images/",
    "/icons/", "/og/", "/logos/", "/.well-known/",
    "/stats","/flights", "/destinations", "/travel-questionnaire",
)


# -----------------------------------------
# SUSPICIOUS FILE PATTERNS (security)
# -----------------------------------------

SUSPICIOUS_PATTERNS = [
    # ---- PHP & generic ----
    r"\.php$",

    # ---- WordPress core ----
    r"(^|/)wp-admin(/|$)",
    r"(^|/)wp-login\.php$",
    r"(^|/)wp-config\.php$",
    r"(^|/)wp-content(/|$)",
    r"(^|/)wp-includes(/|$)",
    r"(^|/)wp-json(/|$)",
    r"(^|/)xmlrpc\.php$",
    r"(^|/)wlwmanifest\.xml$",

    # ---- WordPress backups / exploits ----
    r"wp-content/(uploads|plugins|themes).*(\.php|\.zip|\.bak|\.old)?$",

    # ---- REST abuse ----
    r"rest_route=/wp/",

    # ---- Secrets / configs ----
    r"\.(env|git|svn|bak|old|tmp|log|sql|db)$",
    r"(token|secret|key|credentials)[_.-]?(id|key)?",
    r"(^|/)(config|settings|secret|env)\.(json|yaml|yml)$",

    # ---- Archives ----
    r"\.(zip|tar|gz|7z|rar)$",
]


COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SUSPICIOUS_PATTERNS]


# -----------------------------------------
# BOT WHITELIST + BLACKLIST
# -----------------------------------------

BAD_USER_AGENTS = [
    "curl", "wget", "python-requests", "nikto", "fimap", "sqlmap", "nmap",
    "scanner", "fetch", "httpx", "libwww", "scrapy",
]

GOOD_BOTS = [
    "googlebot", "bingbot", "yandex", "baiduspider", "duckduckbot",
    "facebookexternalhit", "twitterbot", "applebot", "chatgpt",
    "openai", "msnbot", "slurp"
]


class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        original_path = request.url.path  
        path_lower = original_path.lower() 
        user_agent = request.headers.get("user-agent", "").lower()
        
        if original_path.startswith("//"):
            return Response(status_code=404)

        # -----------------------------------------
        # 1️⃣ Normalize only trailing slash — do NOT lowercase
        # -----------------------------------------
        if original_path != "/" and original_path.endswith("/"):
            clean_path = original_path.rstrip("/")
            query = request.url.query
            redirect_target = clean_path + (f"?{query}" if query else "")
            return RedirectResponse(redirect_target, status_code=301)

        # -----------------------------------------
        # 2️⃣ Methods allowed without checks
        # -----------------------------------------
        if request.method in ("HEAD", "OPTIONS"):
            return await call_next(request)

        # -----------------------------------------
        # 3️⃣ Safe paths and prefixes
        # -----------------------------------------
        if (
            path_lower in SAFE_PATHS
            or any(path_lower.startswith(p) for p in SAFE_PATH_PREFIXES)
        ):
            return await call_next(request)
        # -----------------------------------------
        # 3.5️⃣ Quietly drop WordPress noise
        # -----------------------------------------
        if path_lower.startswith((
            "/wp-",
            "/wp/",
            "/wp-content",
            "/wp-includes",
        )):
            return Response(status_code=404)
            
        if path_lower.endswith((".env", ".env.bak", ".env.example")):
            return Response(status_code=404)

        # -----------------------------------------
        # 4️⃣ Allow good bots
        # -----------------------------------------
        if any(bot in user_agent for bot in GOOD_BOTS):            
            return await call_next(request)

        # -----------------------------------------
        # 5️⃣ Allow real browsers (Mozilla etc.)
        # -----------------------------------------
        if "mozilla" in user_agent or user_agent.strip() == "":
            return await call_next(request)

        # -----------------------------------------
        # 6️⃣ Block suspicious file patterns (.php, .env, etc)
        # -----------------------------------------
        for pattern in COMPILED_PATTERNS:
            if pattern.search(path_lower):
                return JSONResponse({"detail": "Forbidden"}, status_code=HTTP_403_FORBIDDEN)

        # -----------------------------------------
        # 7️⃣ Block known bad scanning tools
        # -----------------------------------------
        for bad in BAD_USER_AGENTS:
            if bad in user_agent:
                return JSONResponse({"detail": "Forbidden"}, status_code=HTTP_403_FORBIDDEN)
        
        if path_lower.endswith("config.json"):
            return JSONResponse({"detail": "Forbidden"}, status_code=HTTP_403_FORBIDDEN)

        # -----------------------------------------
        # 8️⃣ Default → allow
        # -----------------------------------------
        return await call_next(request)
