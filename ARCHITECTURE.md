# Flights Explorer Architecture & Codebase Overview

This document provides a comprehensive overview of the `tlvdestinations_official` repository structure, architecture, and core data flows. It is intended to serve as a reference for future development.

## 1. Core Technology Stack
- **Backend Framework**: FastAPI (migrated from an older Flask architecture, as seen in `README.md`).
- **Template Engine**: Jinja2 (`fastapi.templating.Jinja2Templates`).
- **Data Processing**: Pandas (for normalizing and filtering flight data and travel warnings).
- **Frontend**: Vanilla JavaScript (`static/js/`) and CSS (`static/css/app.css`). HTML templates utilize Jinja2 for server-side rendering.

## 2. Directory Structure

```text
tlvdestinations_official/
├── main.py                     # Application entry point, core logic, data fetching/repair
├── app_state.py                # Global in-memory state (Pandas DataFrames)
├── config_paths.py             # Centralized file path definitions
├── README.md                   # Legacy instructions (mentions Flask but uses FastAPI)
├── core/
│   └── templates.py            # Jinja2 template initialization
├── routers/                    # Modular FastAPI routers
│   ├── generic_routes.py       # Static-like pages (about, privacy, terms, questionnaire)
│   ├── weather.py              # Weather data routes
│   ├── airlines_registry.py    # Airline information routes
│   ├── destination_diff_routes.py # Difference calculation routes
│   └── tlv_shops.py            # Shops information routes
├── data/                       # Static reference JSONs
│   ├── airlines_all.json       # General airline data
│   ├── city_translations.json  # Translations for city names
│   ├── country_translations.json # Translations for country names
│   └── city_name_corrections.json # Hardcoded city name fixes
├── cache/                      # Dynamic data (fetched & cached)
│   ├── israel_flights.json     # Fetched flight data from gov.il
│   └── travel_warnings.json    # Fetched travel warnings from gov.il
├── static/                     # Frontend Assets
│   ├── js/                     # Client-side JavaScript (app.js, lang.js, nagishli, etc.)
│   ├── css/                    # Stylesheets (app.css)
│   ├── icons/ & logos/         # Image assets
│   └── airlines_logo/          # Airline specific logos
├── templates/                  # Jinja2 HTML templates
│   ├── base.html               # Main layout wrapper
│   ├── index.html              # Homepage template
│   ├── map.html                # Interactive map view
│   ├── stats.html              # Statistics and charts view
│   ├── chat.html               # AI Chat interface
│   └── destination.html        # Individual destination view
└── helpers/                    # Utilities and middleware
    ├── helper.py               # Language and versioning helpers
    ├── proxies.py & proxy_manager.py # IP rotation for bypassing API blocks
    └── securitymiddleware.py   # App security policies
```

## 3. Core Architecture & Data Flow

### 3.1. In-Memory State (`app_state.py`)
To ensure fast access across different routers without circular import issues, the application loads massive datasets into memory upon startup:
- `DATASET_DF`: Main dataset containing grouped and processed destination information.
- `DATASET_DF_FLIGHTS`: Raw flight records.
- `TRAVEL_WARNINGS_DF`: Travel warnings data.

### 3.2. Data Fetching & Healing (`main.py`)
The application relies heavily on data provided by **data.gov.il** (Israel Airports Authority flights and Travel Warnings).
- **Fetching Strategy**: The app tries direct HTTPS requests to the Israeli government APIs.
- **Proxy Fallback**: If the direct request times out or is blocked (e.g., 403 Forbidden), the app gracefully falls back to using rotating proxies via `helpers/proxy_manager.py`.
- **JSON Repair**: The government APIs often return malformed JSON. The codebase implements robust error handling using the `json_repair` library to fix and continuously parse broken payloads.
- **Normalization**: After successfully fetching and repairing the data, the app normalizes the JSON arrays into structured Pandas DataFrames. It handles missing values (NaN/NaT replacement with `None` to remain JSON compliant) and standardizes formatting.
- **Caching**: The prepared DataFrames are saved to disk in the `cache/` directory (`israel_flights.json`, `travel_warnings.json`) to prevent hitting the APIs unnecessarily and to serve as a fast startup source.

### 3.3. Job Scheduling
`main.py` configures an **APScheduler** background task (`scheduler.add_job`) to run `update_flights()` automatically. This function orchestrates fetching new flights, calculating changes (`generate_destination_diff`), and building the site map.

### 3.4. Routing & Templating
- The root route (`/`) and a few core endpoints are directly in `main.py`.
- They inject the global `DATASET_DF` into the templates context.
- The `templates/base.html` defines the shell, while specific views like `index.html` (the data table) or `map.html` (the visualization) extend it.

### 3.5. Frontend Interactivity
- `static/js/app.js`: Connects UI elements, handles client-side filtering (e.g., DataTables), and manages map initialization.
- `static/js/lang.js`: Handles Hebrew/English localization logic on the client-side.
- Filters and selections typically trigger DOM manipulations without necessarily reloading the entire page, providing a snappy experience.

## 4. Notable Implementation Details
- **City Name Corrections**: The app strictly uses `data/city_name_corrections.json` to fix typos and mismatched English names from the government flight data without hardcoding rules in Python.
- **Distance & Time**: Uses Haversine formulas to calculate flight distances from TLV and estimates flight time automatically.
- **Low-Cost Detection**: The `load_airline_websites()` and related functions map known airline codes to their "Low-Cost" classification for frontend filtering.
