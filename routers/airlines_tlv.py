from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from helpers.airlines_registry import (
    download_airlines_list,
    get_active_tlv_airlines,
)
from core.templates import TEMPLATES

router = APIRouter(prefix="/airlines", tags=["Airlines"])


@router.get("/tlv", response_class=HTMLResponse)
async def airlines_tlv_page(request: Request):
    # Load once
    airlines_all = download_airlines_list()
    active_airlines = get_active_tlv_airlines()

    return TEMPLATES.TemplateResponse(
        "airlines_tlv.html",
        {
            "request": request,
            "total_worldwide": len(airlines_all),
            "total_active_tlv": len(active_airlines),
            "active_airlines": active_airlines,
        },
    )


@router.get("/tlv.json", response_class=JSONResponse)
async def airlines_tlv_api():
    airlines_all = download_airlines_list()
    active_airlines = get_active_tlv_airlines()

    return {
        "total_worldwide": len(airlines_all),
        "total_active_tlv": len(active_airlines),
        "airlines": active_airlines,
    }

