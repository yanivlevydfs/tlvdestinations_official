# Flights Explorer (Flask)

## 1) Install
python -m venv .venv
. .venv/Scripts/activate   # Windows
# or: source .venv/bin/activate
pip install -r requirements.txt

## 2) Set API key (optional, else app runs using disk cache only)
# PowerShell:
setx AVIATION_EDGE_KEY "YOUR_KEY"
# Bash/macOS:
export AVIATION_EDGE_KEY="YOUR_KEY"

## 3) Run
python app.py
# open http://127.0.0.1:5000/

## Notes
- First-time data build: Click "♻ Refresh routes (daily)" — progress bar shows.
- Data is cached in ./cache/ (per-airport + full dataset).
- Subsequent loads are instant; dataset auto-refreshes daily on first visit.
- Map opens in a modal; URL remains on "/".
