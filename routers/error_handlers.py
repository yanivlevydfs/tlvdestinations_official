# routers/error_handlers.py
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from core.templates import TEMPLATES

# Handle HTTP errors (404, 403, etc.)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return TEMPLATES.TemplateResponse("error.html", {
        "request": request,
        "status_code": exc.status_code,
        "message": exc.detail,
        "lang": request.query_params.get("lang", "en")
    }, status_code=exc.status_code)


# Handle validation errors (422)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return TEMPLATES.TemplateResponse("error.html", {
        "request": request,
        "status_code": 422,
        "message": (
            "Invalid request. Please check your input."
            if request.query_params.get("lang", "en") == "en"
            else "בקשה לא חוקית. בדוק את הנתונים שלך."
        ),
        "lang": request.query_params.get("lang", "en")
    }, status_code=422)


# Handle unexpected errors (500)
async def generic_exception_handler(request: Request, exc: Exception):
    return TEMPLATES.TemplateResponse("error.html", {
        "request": request,
        "status_code": 500,
        "message": (
            "Internal Server Error"
            if request.query_params.get("lang", "en") == "en"
            else "שגיאת שרת פנימית"
        ),
        "lang": request.query_params.get("lang", "en")
    }, status_code=500)
