# routers/middleware_redirect.py
import re
import logging
from fastapi import Request
from fastapi.responses import RedirectResponse, JSONResponse

logger = logging.getLogger("middleware_redirect")

async def redirect_and_log_404(request: Request, call_next):
    host_header = request.headers.get("host", "").lower()
    hostname = (request.url.hostname or "").lower()
    client_host = (request.client.host or "").lower()
    path = request.url.path

    # 🚫 0. Block obvious malicious paths (before anything else)
    suspicious_patterns = (".env", ".git", "phpinfo", "config", "composer.json", "wp-admin", "shell", "eval(")
    if any(p in path.lower() for p in suspicious_patterns):
        logger.debug(f"🚫 Blocked suspicious request from {client_host} → {path}")
        return JSONResponse({"detail": "Forbidden"}, status_code=403)

    # 🚫 Skip redirects for localhost or internal testing
    if any(kw in host_header for kw in ("localhost", "127.0.0.1", "::1")) \
       or hostname in ("localhost", "127.0.0.1", "::1") \
       or client_host in ("localhost", "127.0.0.1", "::1"):
        response = await call_next(request)
        if response.status_code == 404 and not path.startswith("/%23"):
            # Suppress logs for missing airline logos (frontend fallback handles this)
            if not path.startswith("/static/airlines_logo/airline_logos/"):
                logger.error(f"⚠️ 404 (dev) from {client_host} → {path}")
        return response
    
    # 🚫 Skip redirect logic for known static endpoints
    if path in ("/favicon.ico", "/favicon.svg", "/robots.txt", "/sitemap.xml"):
        return await call_next(request)
    # 🌍 Production: clean & normalize URLs
    url = str(request.url)
    redirect_url = url

    # ✅ 1. Enforce HTTPS (proxy-aware, Railway-safe)
    proto = request.headers.get("x-forwarded-proto", "").lower()
    
    # [Lines 40-74 omitted as they are unchanged]

    if proto == "http" and not os.getenv("RAILWAY_ENVIRONMENT"):
        redirect_url = redirect_url.replace("http://", "https://", 1)

    # ✅ 2. Remove 'www.'
    if "://www." in redirect_url:
        redirect_url = redirect_url.replace("://www.", "://", 1)

    # ✅ 3. Handle malformed encoded fragments (e.g. /%23c)
    if "/%23" in url or path.startswith("/#") or "%23" in path:
        clean_base = url.split("/%23")[0]
        logger.debug(f"🧹 Cleaning malformed anchor → redirecting {path} → {clean_base}")
        return RedirectResponse(url=clean_base, status_code=301)

    # ✅ 4. Trailing slash normalization (SEO-friendly)
    # Don't strip for known dynamic paths like /destinations/{iata}
    if (
        path.endswith("/") 
        and len(path) > 1
        and not (
            path.startswith("/static") or 
            path.startswith("/.well-known") or 
            re.match(r"^/destinations/[A-Z]{3}/?$", path, re.IGNORECASE)
        )
    ):
        redirect_url = redirect_url.rstrip("/")


    # Redirect only if changed
    if redirect_url != url:
        logger.debug(f"🔁 Redirecting {url} → {redirect_url}")
        return RedirectResponse(url=redirect_url, status_code=301)

    # 🧩 Continue normally
    response = await call_next(request)

    # ⚠️ Log real 404s only (ignore bots hitting /%23 junk)
    if response.status_code == 404 and not path.startswith("/%23"):
        if not path.startswith("/static/airlines_logo/airline_logos/"):
            logger.debug(f"⚠️ 404 from {client_host} → {path}")


    return response