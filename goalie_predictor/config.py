"""Project-wide settings: scoring rules, paths, seasons, cache lifetimes."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

# --- Your league's goaltender scoring -------------------------------------
SCORING = {
    "W": 3.0,     # win
    "GA": -1.5,   # goal against
    "SV": 0.2,    # save
    "SO": 4.0,    # shutout
    "OTL": 1.0,   # overtime / shootout loss
}

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "goalie_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
DOCS_DIR = ROOT / "docs"
SITE_DATA_PATH = DOCS_DIR / "data" / "predictions.json"
MY_GOALIES_PATH = ROOT / "my_goalies.json"

# --- Data window -------------------------------------------------------------
# Completed seasons used to train the model (on top of the current season).
# Career head-to-head history always uses every NHL season a goalie has played.
TRAINING_SEASONS_BACK = 4

# --- Cache lifetimes (seconds) ---------------------------------------------
TTL_CURRENT = 3 * 3600      # current-season game logs, rosters, schedules
TTL_FOREVER = None          # completed seasons never change

# --- Model knobs -------------------------------------------------------------
RECENT_N = 10          # "recent form" window, in starts
TEAM_WINDOW = 15       # team/opponent form window, in games
H2H_PRIOR_STARTS = 5   # shrink head-to-head rates toward career with this weight
H2H_PRIOR_SHOTS = 150  # same idea for head-to-head save %

REQUEST_PAUSE = 0.15   # be polite to the NHL API


def season_for(date: dt.date) -> int:
    """NHL season id (e.g. 20262027) that a calendar date belongs to."""
    start = date.year if date.month >= 7 else date.year - 1
    return start * 10000 + start + 1


def current_season(today: dt.date | None = None) -> int:
    return season_for(today or dt.date.today())


def previous_seasons(season: int, n: int) -> list[int]:
    start = season // 10000
    return [(start - i) * 10000 + (start - i + 1) for i in range(1, n + 1)]
