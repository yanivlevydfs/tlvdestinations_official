# core/templates.py
from pathlib import Path
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"

TEMPLATES = Jinja2Templates(directory=str(TEMPLATES_DIR))
