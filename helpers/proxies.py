# proxies.py
import random

# Your Webshare proxy list
RAW_PROXIES = [
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

# Parse proxies into dicts
PROXY_LIST = []

for line in RAW_PROXIES:
    try:
        host, port, user, pwd = line.strip().split(":")
        PROXY_LIST.append({
            "host": host,
            "port": port,
            "user": user,
            "pass": pwd,
        })
    except ValueError:
        continue


def get_random_proxy() -> dict:
    """Return one random proxy in dict format."""
    return random.choice(PROXY_LIST)
