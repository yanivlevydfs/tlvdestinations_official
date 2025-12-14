# app_state.py
# --------------------------------------------------
# Shared global application state for all routers.
# SAFE to import from anywhere (no circular imports).
# --------------------------------------------------

import pandas as pd

DATASET_DF: pd.DataFrame = pd.DataFrame()   # default empty
TRAVEL_WARNINGS_DF: pd.DataFrame = pd.DataFrame()

