# routers/docs_router.py
import os
import secrets
import logging

from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.security import HTTPBasic, HTTPBasicCredentials

logger = logging.getLogger("docs")

router = APIRouter(include_in_schema=False)

# ---------------------------------------------------------
#  SECURITY OBJECT
# ---------------------------------------------------------
security = HTTPBasic()


# ---------------------------------------------------------
#  VERIFY CREDENTIALS
# ---------------------------------------------------------
def verify_docs_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = os.getenv("DOCS_USER", "admin")
    correct_password = os.getenv("DOCS_PASS", "secret123")

    is_user = secrets.compare_digest(credentials.username, correct_username)
    is_pass = secrets.compare_digest(credentials.password, correct_password)

    if not (is_user and is_pass):
        logger.warning(f"❌ Unauthorized docs access attempt: {credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username


# ---------------------------------------------------------
#  /openapi.json  (Protected)
# ---------------------------------------------------------
@router.get("/openapi.json")
async def custom_openapi(request: Request, username: str = Depends(verify_docs_credentials)):
    logger.debug(f"GET /openapi.json → served for user={username}")

    schema = get_openapi(
        title=request.app.title,
        version="1.0.0",
        routes=request.app.routes,
    )

    return schema


# ---------------------------------------------------------
#  /docs (Protected Swagger)
# ---------------------------------------------------------
@router.get("/docs")
async def custom_swagger_ui(username: str = Depends(verify_docs_credentials)):
    logger.debug(f"GET /docs → Swagger served to {username}")

    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="API Documentation",
    )


# ---------------------------------------------------------
#  /redoc (Protected ReDoc)
# ---------------------------------------------------------
@router.get("/redoc")
async def custom_redoc(username: str = Depends(verify_docs_credentials)):
    logger.debug(f"GET /redoc → ReDoc served to {username}")

    return get_redoc_html(
        openapi_url="/openapi.json",
        title="API ReDoc",
    )
