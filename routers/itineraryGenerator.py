import logging
from datetime import datetime, timedelta
import httpx
import asyncio
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import json
from typing import List
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
# Setup logger
logger = logging.getLogger("fly_tlv.itineraryGeneration")

router = APIRouter(tags=["itineraryGeneration"])

# ------------------------------------------------------
# GEMINI SETUP
# ------------------------------------------------------
import os
from google import genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash-lite"


# ------------------------------------------------------
# ITINERARY GENERATOR (Gemini)
# ------------------------------------------------------
class ItineraryRequest(BaseModel):
    city: str
    country: str
    interests: List[str] = []

@router.post("/api/generate-itinerary", response_class=JSONResponse)
async def generate_itinerary(payload: ItineraryRequest):
    """
    Generate a 3-day itinerary for a specific city using Gemini.
    """
    city = payload.city.strip()
    country = payload.country.strip()
    interests = ", ".join(payload.interests) if payload.interests else "general sightseeing, food, and culture"

    logger.debug(f"🤖 Generating itinerary for {city}, {country} (Interests: {interests})")

    prompt = f"""
    You are an expert travel guide. Create a personalized 3-day travel itinerary for **{city}, {country}**.
    
    User Interests: {interests}

    🚨 **STRICT OUTPUT FORMAT (JSON ONLY)** 🚨
    Return a raw JSON object with this exact structure (no markdown, no backticks):
    {{
        "city": "{city}",
        "days": [
            {{
                "day": 1,
                "theme": "Day Theme (e.g., Historic Center)",
                "morning": "Activity description...",
                "afternoon": "Activity description...",
                "evening": "Activity description..."
            }},
            {{
                "day": 2,
                "theme": "Day Theme",
                "morning": "Activity description...",
                "afternoon": "Activity description...",
                "evening": "Activity description..."
            }},
            {{
                "day": 3,
                "theme": "Day Theme",
                "morning": "Activity description...",
                "afternoon": "Activity description...",
                "evening": "Activity description..."
            }}
        ]
    }}

    Keep descriptions concise (1-2 sentences). Focus on top-rated, real attractions.
    """

    try:
        response = await gemini_client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        
        # Clean response (sometimes Gemini adds markdown backticks)
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:-3].strip()
            
        return JSONResponse(content=json.loads(raw_text))

    except json.JSONDecodeError:
        logger.error("Failed to parse Gemini JSON response")
        return JSONResponse(status_code=500, content={"error": "AI returned invalid format. Please try again."})
    except Exception as e:
        logger.error(f"Gemini Itinerary Error: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to generate itinerary."})