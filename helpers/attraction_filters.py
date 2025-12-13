import json
from functools import lru_cache
from pathlib import Path
import logging

logger = logging.getLogger("attractions")

FILTERS_FILE = Path(__file__).parent.parent / "data" / "attractions_filters.json"

@lru_cache(maxsize=1)
def load_filters():
    """
    Load all attraction filter lists from external JSON.
    Returned dict is cached (loaded only once per process).
    All items are normalized to lowercase.
    """
    try:
        with open(FILTERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Normalize everything to lowercase for consistent matching
        for key, values in data.items():
            if isinstance(values, list):
                data[key] = [v.lower() for v in values]

        logger.debug(f"Loaded attraction filters from {FILTERS_FILE}")
        return data

    except Exception as e:
        logger.error(f"❌ Failed to load attraction filters: {e}")

        # Safe fallback — return ALL keys your code expects
        return {
            "BLOCK": [],
            "ALLOW": [],
            "DESC_KEYS": [],
            "TITLE_KEYS": [],
            "CITY_KEYWORDS_EN": [],
            "CITY_KEYWORDS_HE": [],
        }
