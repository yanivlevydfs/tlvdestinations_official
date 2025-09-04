# sitemap_tool.py
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional
from xml.sax.saxutils import escape

@dataclass
class Url:
    loc: str
    lastmod: Optional[str] = None     # ISO date string or datetime/date
    changefreq: Optional[str] = None  # always, hourly, daily, weekly, monthly, yearly, never
    priority: Optional[float] = None  # 0.0 - 1.0

def _iso(x):
    return x.isoformat() if hasattr(x, "isoformat") else (x or None)

def build_sitemap(urls: Iterable[Url]) -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    ]
    for u in urls:
        parts.append("  <url>")
        parts.append(f"    <loc>{escape(u.loc)}</loc>")
        if u.lastmod:    parts.append(f"    <lastmod>{_iso(u.lastmod)}</lastmod>")
        if u.changefreq: parts.append(f"    <changefreq>{u.changefreq}</changefreq>")
        if u.priority is not None: parts.append(f"    <priority>{u.priority:.1f}</priority>")
        parts.append("  </url>")
    parts.append("</urlset>")
    return "\n".join(parts)

# Optional: simple CLI
if __name__ == "__main__":
    import argparse, pathlib
    p = argparse.ArgumentParser(description="Generate sitemap.xml")
    p.add_argument("--base", required=True, help="e.g. https://fly-tlv.com")
    p.add_argument("--out", default="sitemap.xml")
    p.add_argument("paths", nargs="+", help="e.g. / /about /privacy")
    args = p.parse_args()

    today = date.today()
    urls = [Url(args.base.rstrip("/") + path, today, "weekly", 0.5) for path in args.paths]
    xml = build_sitemap(urls)
    pathlib.Path(args.out).write_text(xml, encoding="utf-8")
    print(f"Wrote {args.out}")
