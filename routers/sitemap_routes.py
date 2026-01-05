# routers/sitemap_routes.py
# ---------------------------------------------------------------
# Sitemap generator (externalized from main.py)
# Uses config_paths for paths and project-wide global DATASET_DF.
# ---------------------------------------------------------------

from fastapi import APIRouter, Response
from pathlib import Path
from datetime import date
import logging
from helpers.sitemap_utils import Url, build_sitemap
from config_paths import STATIC_DIR
import app_state


logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------
# Main Sitemap Route
# ---------------------------------------------------------------
@router.get("/sitemap.xml", response_class=Response, include_in_schema=False)
def sitemap():
    """Generate sitemap.xml including static pages (all on disk) and dynamic endpoints."""
    global STATIC_DIR, DATASET_DF
    base = "https://fly-tlv.com"
    today = date.today()

    # --- 1. Static base URLs ---
    urls = [
        Url(f"{base}/", today, "daily", 1.0),
        Url(f"{base}/stats", today, "daily", 1.0),
        Url(f"{base}/about", today, "yearly", 0.6),
        Url(f"{base}/direct-vs-nonstop", today, "yearly", 0.6),
        Url(f"{base}/privacy", today, "yearly", 0.5),
        Url(f"{base}/glossary", today, "yearly", 0.5),
        Url(f"{base}/contact", today, "yearly", 0.5),
        Url(f"{base}/accessibility", today, "yearly", 0.5),
        Url(f"{base}/terms", today, "yearly", 0.5),
        Url(f"{base}/map", today, "weekly", 0.7),
        Url(f"{base}/flights", today, "weekly", 0.7),
        Url(f"{base}/travel-warnings", today, "weekly", 0.7),
        Url(f"{base}/chat", today, "weekly", 0.8),
        # Hebrew
        Url(f"{base}/?lang=he", today, "daily", 1.0),
        Url(f"{base}/stats?lang=he", today, "daily", 1.0),
        Url(f"{base}/about?lang=he", today, "yearly", 0.6),
        Url(f"{base}/direct-vs-nonstop?lang=he", today, "yearly", 0.6),
        Url(f"{base}/accessibility?lang=he", today, "yearly", 0.5),
        Url(f"{base}/terms?lang=he", today, "yearly", 0.5),
        Url(f"{base}/privacy?lang=he", today, "yearly", 0.5),
        Url(f"{base}/contact?lang=he", today, "yearly", 0.5),
        Url(f"{base}/map?lang=he", today, "weekly", 0.7),
        Url(f"{base}/flights?lang=he", today, "weekly", 0.7),
        Url(f"{base}/travel-warnings?lang=he", today, "weekly", 0.7),
        Url(f"{base}/chat?lang=he", today, "weekly", 0.8),
        Url(f"{base}/glossary?lang=he", today, "yearly", 0.5),
    ]

    # --- 2. Add dynamic FastAPI destinations (live routes) ---
    try:
        df = app_state.DATASET_DF

        if df is None or df.empty:
            logger.info("DATASET_DF is empty — skipping dynamic IATA links")
        elif "IATA" not in df.columns:
            logger.info("DATASET_DF missing 'IATA' column — skipping dynamic links")
        else:
            iatas = df["IATA"].dropna().unique()
            for iata in iatas:
                code = str(iata).strip()
                if not code:
                    continue

                urls.append(Url(f"{base}/destinations/{code}", today, "weekly", 0.7))
                urls.append(Url(f"{base}/destinations/{code}?lang=he", today, "weekly", 0.7))

            logger.debug(f"🧭 Added {len(iatas)} dynamic destinations.")

    except Exception as e:
        logger.info("Failed to load dynamic IATA links (continuing without them): %s", e)


    # --- 3. Include *all* static-generated HTML pages physically saved on disk ---
    static_dest_dir = STATIC_DIR / "destinations"
    if static_dest_dir.exists():
        logger.debug(f"🗺️ Scanning static HTML destinations recursively from {static_dest_dir} ...")
        total_files = 0

        for lang_dir in static_dest_dir.iterdir():
            if not lang_dir.is_dir():
                continue

            lang_code = lang_dir.name  # e.g. "en" or "he"
            for html_file in lang_dir.rglob("*.html"):
                try:
                    iata = html_file.stem.upper()
                    lastmod = date.fromtimestamp(html_file.stat().st_mtime)

                    # Build the URL
                    if lang_code == "he":
                        url = f"{base}/destinations/{iata}?lang=he"
                    else:
                        url = f"{base}/destinations/{iata}"

                    urls.append(Url(url, lastmod, "weekly", 0.7))
                    total_files += 1

                    # 🪵 Detailed log line for each file
                    logger.debug(f"📄 Added static file: {html_file.relative_to(STATIC_DIR)} → {url}")

                except Exception as e:
                    logger.debug("Skipped file %s: %s", html_file, e)

        logger.debug("Added %d static HTML destination files from %s",total_files,static_dest_dir,)
    else:
        logger.info("Static destinations folder not found (optional): %s",static_dest_dir,)


    # --- 4. Build and write sitemap.xml ---
    xml = build_sitemap(urls)
    out_path = STATIC_DIR / "sitemap.xml"

    try:
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(xml, encoding="utf-8")
        logger.debug("Sitemap written to %s with %d URLs total",out_path,len(urls),)
    except Exception as e:
        logger.error("Failed to write sitemap.xml",exc_info=True,)

    return Response(content=xml, media_type="application/xml")
