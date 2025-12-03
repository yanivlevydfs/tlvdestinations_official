# routers/error_handlers.py
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse,RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from logging_setup import setup_logging, get_app_logger,setup_feedback_logger
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi import status
import logging

TEMPLATES = Jinja2Templates(directory="templates")

logger = logging.getLogger("errors")

router = APIRouter(include_in_schema=False)


# ---------------------------------------------------------
# 404 / 403 / כל StarletteHTTPException
# ---------------------------------------------------------
@router.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    lang = request.query_params.get("lang", "en")

    return TEMPLATES.TemplateResponse(
        "error.html",
        {
            "request": request,
            "status_code": exc.status_code,
            "message": exc.detail,
            "lang": lang,
        },
        status_code=exc.status_code,
    )


# ---------------------------------------------------------
# Validation Errors (422)
# ---------------------------------------------------------
@router.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    lang = request.query_params.get("lang", "en")

    message = (
        "Invalid request. Please check your input."
        if lang == "en"
        else "בקשה לא חוקית. בדוק את הנתונים שלך."
    )

    return TEMPLATES.TemplateResponse(
        "error.html",
        {
            "request": request,
            "status_code": 422,
            "message": message,
            "lang": lang,
        },
        status_code=422,
    )


# ---------------------------------------------------------
# Generic Exception Handler (500)
# ---------------------------------------------------------
@router.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    lang = request.query_params.get("lang", "en")

    # Log full traceback
    logger.error(f"⚠️ Unhandled Server Error: {exc}", exc_info=True)

    message = (
        "Internal Server Error"
        if lang == "en"
        else "שגיאת שרת פנימית"
    )

    return TEMPLATES.TemplateResponse(
        "error.html",
        {
            "request": request,
            "status_code": 500,
            "message": message,
            "lang": lang,
        },
        status_code=500,
    )
