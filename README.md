# Flights Explorer (FastAPI)

## 1) Install
python -m venv .venv
. .venv/Scripts/activate   # Windows
# or: source .venv/bin/activate
pip install -r requirements.txt

## 2) Set Environment Variables (optional)
# PowerShell:
setx GEMINI_API_KEY "YOUR_KEY"
# Bash/macOS:
export GEMINI_API_KEY="YOUR_KEY"

## 3) Run (Development)
You can use the provided `run.bat` script on Windows:
```cmd
run.bat
```
Or run Uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
# Open http://127.0.0.1:8000/

## Notes
- **Architecture**: The application runs on FastAPI (migrated from Flask). It serves Jinja2 templates and manages global data with Pandas.
- **Data Gathering**: Flight and travel warnings data is fetched from Israeli government APIs. It automatically handles errors, fallback proxy rotations, and repairs JSON on-the-fly.
- **Caching**: Data is cached in `./cache/` (`israel_flights.json`, `travel_warnings.json`).
- **Subsequent loads**: Application startup is instant using cached data. The dataset auto-refreshes seamlessly via a background scheduler.
- **More Info**: For an in-depth code and structural overview, see `ARCHITECTURE.md`.
