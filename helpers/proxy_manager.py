import random
import logging
import requests

logger = logging.getLogger("proxy_manager")

# ───────────────────────────────────────────────
#  PROXY LIST (Webshare.io)
# ───────────────────────────────────────────────
WEBPROXY_LIST = [
    "142.111.48.253:7030:njqtgfze:ni4z09oz96km",
    "31.59.20.176:6754:njqtgfze:ni4z09oz96km",
    "23.95.150.145:6114:njqtgfze:ni4z09oz96km",
    "198.23.239.134:6540:njqtgfze:ni4z09oz96km",
    "107.172.163.27:6543:njqtgfze:ni4z09oz96km",
    "198.105.121.200:6462:njqtgfze:ni4z09oz96km",
    "64.137.96.74:6641:njqtgfze:ni4z09oz96km",
    "84.247.60.125:6095:njqtgfze:ni4z09oz96km",
    "216.10.27.159:6837:njqtgfze:ni4z09oz96km",
    "142.111.67.146:5611:njqtgfze:ni4z09oz96km",
]

# ───────────────────────────────────────────────
#  HEADERS REQUIRED BY data.gov.il
# ───────────────────────────────────────────────
GOVIL_HEADERS = {
    "User-Agent": (
        "datagov-external-client; "
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.gov.il/"
}


# ───────────────────────────────────────────────
#  Build a random proxy dict for requests
# ───────────────────────────────────────────────
def get_random_proxy() -> dict:
    """Return a proxy dict compatible with requests.get()"""
    line = random.choice(WEBPROXY_LIST)
    ip, port, user, password = line.split(":")

    proxy_url = f"http://{user}:{password}@{ip}:{port}"

    return {
        "http": proxy_url,
        "https": proxy_url,
    }


# ───────────────────────────────────────────────
#  Generic proxy-enabled GET fetcher with retries
# ───────────────────────────────────────────────
def fetch_with_proxy(url: str, params: dict | None = None, retries: int = 3) -> dict | None:
    """
    Perform a GET request to data.gov.il using rotating Webshare proxies.
    Returns JSON dict or None if all retries fail.
    """
    for attempt in range(1, retries + 1):
        proxy = get_random_proxy()
        try:
            logger.debug(
                f"🌐 Attempt {attempt}/{retries} → Proxy={proxy['http']} | URL={url}"
            )

            response = requests.get(
                url,
                params=params,
                headers=GOVIL_HEADERS,
                proxies=proxy,
                timeout=25,
                verify=True
            )
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.warning(
                f"⚠️ Proxy failed ({proxy['http']}) on attempt {attempt}/{retries} — {e}"
            )

    logger.error("❌ All proxies failed for URL: %s", url)
    return None
