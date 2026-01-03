from fastapi import APIRouter, Request, Depends, Response
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pathlib import Path
import logging
from helpers.helper import get_lang
from main import TEMPLATES
from datetime import datetime
from config_paths import BASE_DIR 
logger = logging.getLogger("generic_router")

router = APIRouter()

# ----------------------------------------------------------
# Google Traffic Advice (/.well-known/traffic-advice)
# ----------------------------------------------------------
@router.get("/.well-known/traffic-advice", include_in_schema=False)
async def traffic_advice(request: Request):
    """
    Responds to Google's Traffic Advice probe requests.
    Docs:
    https://developers.google.com/search/docs/crawling-indexing/traffic-advice
    """
    ua = request.headers.get("user-agent", "").lower() if request.headers else ""

    # Silent ignore for non-Googlebot
    if "googlebot" not in ua:
        return Response(status_code=204)

    client_ip = request.client.host if request.client else "unknown"
    logger.debug(f"Googlebot traffic-advice request from {client_ip}")

    return JSONResponse(
        content={"crawling": {"state": "allowed"}},
        headers={"Cache-Control": "public, max-age=86400"}
    )

@router.api_route("/robots.txt", methods=["GET", "HEAD"])
async def robots_txt(request: Request):
    file_path = BASE_DIR / "robots.txt" 

    if file_path.exists():
        logger.debug(f"GET /robots.txt | client={request.client.host}")
        return FileResponse(file_path, media_type="text/plain")

    logger.error(f"robots.txt not found! | client={request.client.host}")
    raise HTTPException(status_code=404, detail="robots.txt not found")

@router.get("/ads.txt", include_in_schema=False)
async def ads_txt(request: Request):
    logger.debug(f"GET /ads.txt | client={request.client.host}")

    # ads.txt must live in project root or static folder
    file_path = Path(__file__).resolve().parent.parent / "ads.txt"

    return FileResponse(file_path, media_type="text/plain")

@router.get("/manifest.json", include_in_schema=False)
async def manifest(request: Request):
    lang = request.query_params.get("lang", "en").lower()
    if lang not in ("en", "he"):
        lang = "en"

    client = request.client.host if request.client else "unknown"
    base_path = Path(__file__).resolve().parent.parent

    manifest_en = base_path / "manifest.en.json"
    manifest_he = base_path / "manifest.he.json"

    # Choose manifest based on ?lang
    selected_manifest = (
        manifest_he if lang == "he" and manifest_he.exists()
        else manifest_en
    )

    if selected_manifest.exists():
        logger.debug(
            f"📄 GET /manifest.json | lang={lang} | client={client} | file={selected_manifest.name}"
        )
        return FileResponse(
            selected_manifest, 
            media_type="application/manifest+json"
        )

    logger.error(f"❌ Manifest not found | lang={lang} | path={selected_manifest}")
    return JSONResponse(
        {"error": "Manifest not found", "lang": lang},
        status_code=404
    )

@router.get("/about", response_class=HTMLResponse)
async def about(request: Request, lang: str = Depends(get_lang)):
    logger.debug(f"GET /about | lang={lang} | client={request.client.host}")
    return TEMPLATES.TemplateResponse("about.html", {
        "request": request,
        "lang": lang,
        "now": datetime.now()
    })
    
@router.get("/privacy", response_class=HTMLResponse)
async def privacy(request: Request, lang: str = Depends(get_lang)):
    logger.debug(f"GET /privacy | lang={lang} | client={request.client.host}")
    return TEMPLATES.TemplateResponse("privacy.html", {
        "request": request,
        "lang": lang,
        "now": datetime.now()
    })

@router.get("/contact", response_class=HTMLResponse)
async def contact(request: Request, lang: str = Depends(get_lang)):
    logger.debug(f"GET /contact | lang={lang} | client={request.client.host}")
    return TEMPLATES.TemplateResponse("contact.html", {
        "request": request,
        "lang": lang,
        "now": datetime.now()
    })

@router.get("/glossary", response_class=HTMLResponse)
async def glossary_view(request: Request, lang: str = Depends(get_lang)):
    try:
        return TEMPLATES.TemplateResponse("aviation_glossary.html", {
            "request": request,
            "lang": lang,
            "now": datetime.now()
        })
    except Exception:
        # graceful fallback
        msg = (
            "Glossary page could not be loaded."
            if lang != "he"
            else "לא ניתן לטעון את עמוד המונחים."
        )
        return TEMPLATES.TemplateResponse("error.html", {
            "request": request,
            "lang": lang,
            "message": msg,
            "now": datetime.now()
        })
        
@router.get("/accessibility", response_class=HTMLResponse)
async def accessibility(request: Request, lang: str = Depends(get_lang)):
    logger.debug(f"GET /accessibility | lang={lang} | client={request.client.host}")

    return TEMPLATES.TemplateResponse(
        "accessibility.html",
        {
            "request": request,
            "lang": lang,
            "now": datetime.now()
        }
    )
    
@router.get("/direct-vs-nonstop", response_class=HTMLResponse)
async def direct_vs_nonstop(request: Request, lang: str = Depends(get_lang)):
    client = request.client.host

    try:
        logger.debug(f"GET /direct-vs-nonstop | lang={lang} | client={client}")

        return TEMPLATES.TemplateResponse(
            "direct_vs_nonstop.html",
            {
                "request": request,
                "lang": lang,
                "now": datetime.now()
            }
        )

    except Exception as e:
        logger.error(
            f"❌ Failed to render direct_vs_nonstop.html | error={e} | client={client}"
        )

        return TEMPLATES.TemplateResponse(
            "error.html",
            {
                "request": request,
                "lang": lang,
                "message": (
                    "The Direct vs Nonstop page could not be loaded. Please try again later."
                    if lang != "he"
                    else "לא ניתן לטעון את עמוד 'ישיר מול ללא-עצירות'. נסה שוב מאוחר יותר."
                )
            },
            status_code=500
        )
        
@router.get("/terms", response_class=HTMLResponse)
async def terms_view(request: Request, lang: str = Depends(get_lang)):
    client = request.client.host

    try:
        logger.debug(f"GET /terms | lang={lang} | client={client}")

        return TEMPLATES.TemplateResponse(
            "terms.html",
            {
                "request": request,
                "lang": lang,
                "now": datetime.now()
            }
        )

    except Exception as e:
        logger.error(f"❌ Failed to render terms.html | {e} | client={client}")

        return TEMPLATES.TemplateResponse(
            "error.html",
            {
                "request": request,
                "lang": lang,
                "message": (
                    "Terms & Conditions page could not be loaded."
                    if lang != "he"
                    else "לא ניתן לטעון את עמוד התנאים וההגבלות."
                )
            },
            status_code=500
        )
        
@router.get("/sw.js", include_in_schema=False)
async def sw_root():
    return FileResponse(
        "static/js/sw.js",
        media_type="application/javascript",
        headers={
            "Cache-Control": "no-store",
            "Service-Worker-Allowed": "/",
        },
    )
    
@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, lang: str = Depends(get_lang)):
    client = request.client.host if request.client else "unknown"

    try:
        logger.debug(f"GET /chat | lang={lang} | client={client}")

        return TEMPLATES.TemplateResponse(
            "chat.html",
            {
                "request": request,
                "lang": lang,
                "now": datetime.now()
            }
        )

    except Exception as e:
        logger.error(f"❌ Failed to render chat.html | {e} | client={client}")

        return TEMPLATES.TemplateResponse(
            "error.html",
            {
                "request": request,
                "lang": lang,
                "message": (
                    "The chat page could not be loaded."
                    if lang != "he"
                    else "לא ניתן לטעון את עמוד הצ'אט."
                )
            },
            status_code=500
        )