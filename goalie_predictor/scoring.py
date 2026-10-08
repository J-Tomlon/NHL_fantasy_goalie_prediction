"""Fantasy scoring for a single goalie game."""
from __future__ import annotations

from .config import SCORING


def fantasy_points(decision: str | None, shots_against: int, goals_against: int,
                   shutout: int) -> float:
    saves = max(int(shots_against) - int(goals_against), 0)
    pts = SCORING["GA"] * goals_against + SCORING["SV"] * saves
    if decision == "W":
        pts += SCORING["W"]
    elif decision in ("O", "OT", "OTL", "SO"):
        pts += SCORING["OTL"]
    if shutout:
        pts += SCORING["SO"]
    return round(pts, 2)


def breakeven_saves(goals_against: int, decision: str | None) -> float:
    """Saves needed to finish above zero for a given GA / decision (for display)."""
    bonus = SCORING["W"] if decision == "W" else SCORING["OTL"] if decision == "O" else 0.0
    return max((-SCORING["GA"] * goals_against - bonus) / SCORING["SV"], 0.0)
