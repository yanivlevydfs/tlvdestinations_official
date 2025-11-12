import base64
from pathlib import Path

FONT_FILES = [
    "DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf",
    "DejaVuSans-Oblique.ttf",
    "DejaVuSans-BoldOblique.ttf"
]

FONT_DIR = Path("fonts")  # your folder with the .ttf files
OUTPUT_FILE = Path("static/js/vfs_fonts_custom.js")


def encode_fonts():
    entries = []
    for name in FONT_FILES:
        path = FONT_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Font not found: {path}")
        with path.open("rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            entries.append(f'"{name}": "{b64}"')
    return entries


def main():
    entries = encode_fonts()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        out.write("// Auto-generated pdfMake VFS\n")
        out.write("pdfMake = pdfMake || {};\n")
        out.write("pdfMake.vfs = {\n")
        out.write(",\n".join(entries))
        out.write("\n};\n")
    print(f"✅ Done: Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
