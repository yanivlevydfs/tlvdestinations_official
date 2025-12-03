# utils/sitemap_utils.py

from __future__ import annotations
from datetime import date, datetime
from typing import List
import html


class Url:
    def __init__(self, loc: str, lastmod: date, changefreq: str, priority: float):
        self.loc = loc
        self.lastmod = lastmod.isoformat()
        self.changefreq = changefreq
        self.priority = priority


def build_sitemap(urls: List[Url]) -> str:
    """Build a valid, deduplicated sitemap.xml from Url objects."""
    if not urls:
        return ""

    seen = set()
    unique_urls: List[Url] = []

    for u in urls:
        if u.loc not in seen:
            seen.add(u.loc)
            unique_urls.append(u)

    # Sort URLs for deterministic output (by loc)
    unique_urls.sort(key=lambda u: u.loc)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    ]

    for u in unique_urls:
        # Normalize lastmod → ISO 8601
        if isinstance(u.lastmod, (datetime, date)):
            lastmod_str = u.lastmod.strftime("%Y-%m-%d")
        else:
            lastmod_str = str(u.lastmod)

        # Escape any unsafe characters in loc
        loc_escaped = html.escape(u.loc, quote=True)

        lines.append("  <url>")
        lines.append(f"    <loc>{loc_escaped}</loc>")
        lines.append(f"    <lastmod>{lastmod_str}</lastmod>")
        lines.append(f"    <changefreq>{u.changefreq}</changefreq>")
        lines.append(f"    <priority>{float(u.priority):.1f}</priority>")
        lines.append("  </url>")

    lines.append("</urlset>")
    return "\n".join(lines)
