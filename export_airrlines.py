#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a global Airline -> Website JSON map.

Sources:
- OpenFlights airlines.dat (airline list, codes, active flag)
- Wikidata (official website P856) matched by IATA (P229) / ICAO (P230)

Output:
- airline_websites.json  (Name -> Website or null)

Notes:
- Respects Wikidata user-agent policy.
- Handles code collisions and duplicates deterministically.
"""

import csv
import io
import json
import re
import time
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional, Iterable

import requests

OPENFLIGHTS_AIRLINES_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat"
WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# --- Tunables ---
ACTIVE_ONLY = True          # keep only airlines marked Active in OpenFlights
WIKIDATA_CHUNK = 150        # codes per SPARQL batch (keep modest to be polite)
SLEEP_BETWEEN_QUERIES = 0.5 # seconds between SPARQL requests (politeness)

# --- Helpers ---

def http_get(url: str, **kwargs) -> requests.Response:
    headers = kwargs.pop("headers", {})
    headers.setdefault(
        "User-Agent",
        "TLV-Airlines-Sync/1.0 (contact: admin@fly-tlv.com)"
    )
    return requests.get(url, headers=headers, timeout=30, **kwargs)

def normalize_name(name: str) -> str:
    # Trim, collapse spaces, title-case selected patterns sensibly (but keep acronyms)
    s = re.sub(r"\s+", " ", name or "").strip()
    # Keep common acronyms upper-case
    acronyms = {"JAL","ANA","KLM","SAS","TAP","TUI","LOT","MEA","PIA","CSA"}
    parts = []
    for p in s.split(" "):
        if p.upper() in acronyms:
            parts.append(p.upper())
        else:
            parts.append(p.capitalize())
    return " ".join(parts)

def clean_url(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    u = u.strip()
    # Remove surrounding quotes that may creep in
    u = u.strip("'\"")
    # Add scheme if missing
    if not re.match(r"^https?://", u, flags=re.I):
        u = "https://" + u
    # Drop trailing slash noise
    u = re.sub(r"/+$", "", u)
    return u

# --- OpenFlights parsing ---

def fetch_openflights_airlines() -> List[dict]:
    """
    OpenFlights airlines.dat columns (quoted CSV):
    0 AirlineID, 1 Name, 2 Alias, 3 IATA, 4 ICAO, 5 Callsign, 6 Country, 7 Active (Y/N)
    """
    resp = http_get(OPENFLIGHTS_AIRLINES_URL)
    resp.raise_for_status()
    text = resp.text

    rows = []
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        # Normalize row length
        if len(row) < 8:
            # pad to length 8
            row = row + [""] * (8 - len(row))
        airline_id, name, alias, iata, icao, callsign, country, active = row[:8]
        # OpenFlights uses \N for nulls
        def nz(s): return None if s == r"\N" or s == "" else s
        rows.append({
            "AirlineID": nz(airline_id),
            "Name": nz(name),
            "Alias": nz(alias),
            "IATA": nz(iata),
            "ICAO": nz(icao),
            "Callsign": nz(callsign),
            "Country": nz(country),
            "Active": (nz(active) or "").upper() == "Y"
        })
    return rows

# --- Wikidata enrichment ---

def chunked(it: Iterable[str], size: int) -> Iterable[List[str]]:
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf

def sparql_query(query: str) -> dict:
    resp = http_get(
        WIKIDATA_SPARQL_ENDPOINT,
        params={"query": query, "format": "json"}
    )
    resp.raise_for_status()
    return resp.json()

def build_values(var: str, values: List[str]) -> str:
    # SPARQL VALUES clause
    safe_vals = " ".join(f"\"{v}\"" for v in values if v)
    return f"VALUES ?{var} {{ {safe_vals} }}\n"

def query_wikidata_websites_by_iata(iata_batch: List[str]) -> Dict[str, Optional[str]]:
    """
    P229 = IATA airline code
    P856 = official website
    """
    values = build_values("iata", iata_batch)
    q = f"""
    SELECT ?iata ?name ?website WHERE {{
      ?airline wdt:P31/wdt:P279* wd:Q46970 .
      ?airline wdt:P229 ?iata .
      {values}
      OPTIONAL {{ ?airline wdt:P856 ?website . }}
      OPTIONAL {{ ?airline rdfs:label ?name FILTER (LANG(?name)="en") }}
    }}
    """
    data = sparql_query(q)
    out = {}
    for b in data.get("results", {}).get("bindings", []):
        iata = b.get("iata", {}).get("value")
        website = clean_url(b.get("website", {}).get("value"))
        # prefer https cleaned site; if multiple rows, first non-null wins
        if iata and (iata not in out or (out[iata] is None and website)):
            out[iata] = website
    return out

def query_wikidata_websites_by_icao(icao_batch: List[str]) -> Dict[str, Optional[str]]:
    """
    P230 = ICAO airline code
    P856 = official website
    """
    values = build_values("icao", icao_batch)
    q = f"""
    SELECT ?icao ?name ?website WHERE {{
      ?airline wdt:P31/wdt:P279* wd:Q46970 .
      ?airline wdt:P230 ?icao .
      {values}
      OPTIONAL {{ ?airline wdt:P856 ?website . }}
      OPTIONAL {{ ?airline rdfs:label ?name FILTER (LANG(?name)="en") }}
    }}
    """
    data = sparql_query(q)
    out = {}
    for b in data.get("results", {}).get("bindings", []):
        icao = b.get("icao", {}).get("value")
        website = clean_url(b.get("website", {}).get("value"))
        if icao and (icao not in out or (out[icao] is None and website)):
            out[icao] = website
    return out

def enrich_with_wikidata(airlines: List[dict]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns: (iata->website, icao->website)
    """
    iata_codes = sorted({a["IATA"] for a in airlines if a.get("IATA")})
    icao_codes = sorted({a["ICAO"] for a in airlines if a.get("ICAO")})

    iata2site: Dict[str, Optional[str]] = {}
    icao2site: Dict[str, Optional[str]] = {}

    # Query IATA chunks
    for batch in chunked(iata_codes, WIKIDATA_CHUNK):
        iata2site.update(query_wikidata_websites_by_iata(batch))
        time.sleep(SLEEP_BETWEEN_QUERIES)

    # Query ICAO chunks
    for batch in chunked(icao_codes, WIKIDATA_CHUNK):
        icao2site.update(query_wikidata_websites_by_icao(batch))
        time.sleep(SLEEP_BETWEEN_QUERIES)

    # Cast None -> explicit None and then filter only when present
    return (
        {k: v for k, v in iata2site.items() if v is not None},
        {k: v for k, v in icao2site.items() if v is not None},
    )

# --- Assembly ---

def build_airline_website_map() -> OrderedDict:
    airlines = fetch_openflights_airlines()

    if ACTIVE_ONLY:
        airlines = [a for a in airlines if a["Active"]]

    # Enrich via Wikidata
    iata2site, icao2site = enrich_with_wikidata(airlines)

    # Build final Name -> Website map with deterministic choice:
    # 1) website by IATA
    # 2) else website by ICAO
    # 3) else None
    # Deduplicate by (IATA, ICAO, Name) with preference for IATA uniqueness
    by_key = OrderedDict()

    def key_for(a: dict) -> Tuple[Optional[str], Optional[str], str]:
        return (a.get("IATA"), a.get("ICAO"), normalize_name(a.get("Name") or ""))

    # Use OrderedDict to keep stable ordering: IATA present first, then ICAO, then name-only
    airlines.sort(key=lambda a: (
        0 if a.get("IATA") else (1 if a.get("ICAO") else 2),
        (a.get("IATA") or "ZZZ"),
        (a.get("ICAO") or "ZZZZ"),
        normalize_name(a.get("Name") or "")
    ))

    for a in airlines:
        nm = normalize_name(a.get("Name") or "")
        iata = a.get("IATA")
        icao = a.get("ICAO")
        site = None
        if iata and iata in iata2site:
            site = iata2site[iata]
        elif icao and icao in icao2site:
            site = icao2site[icao]
        # allow only one entry per normalized name; prefer first with a site
        if nm not in by_key:
            by_key[nm] = site
        else:
            # if we previously had None and now we have a site, upgrade
            if by_key[nm] is None and site:
                by_key[nm] = site

    # Sort by name for readability in JSON
    ordered = OrderedDict(sorted(by_key.items(), key=lambda kv: kv[0]))
    return ordered

def main():
    data = build_airline_website_map()
    with open("airline_websites.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved airline_websites.json with {sum(1 for _ in data)} airlines")

if __name__ == "__main__":
    main()
