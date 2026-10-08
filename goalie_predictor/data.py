"""Pull goalie game logs from the NHL API and turn them into tidy tables."""
from __future__ import annotations

import datetime as dt
from typing import Callable, Iterable

import pandas as pd

from .config import TRAINING_SEASONS_BACK, current_season, previous_seasons
from .nhl_api import NHLClient, ttl_for
from .scoring import fantasy_points

GAME_COLUMNS = ["player_id", "name", "season", "game_id", "date", "team", "opp", "home",
                "started", "decision", "sa", "ga", "sv", "so", "toi_min", "fp", "positive"]


def _toi_minutes(toi: str | None) -> float:
    if not toi or ":" not in toi:
        return 0.0
    m, s = toi.split(":")[:2]
    return int(m) + int(s) / 60


def parse_game_log(player_id: int, name: str, season: int, log: dict) -> list[dict]:
    rows = []
    for g in log.get("gameLog", []) or []:
        sa = int(g.get("shotsAgainst") or 0)
        ga = int(g.get("goalsAgainst") or 0)
        so = int(g.get("shutouts") or 0)
        decision = g.get("decision") or ""  # absent when another goalie got the decision
        fp = fantasy_points(decision, sa, ga, so)
        rows.append({
            "player_id": player_id, "name": name, "season": season,
            "game_id": g["gameId"], "date": g["gameDate"],
            "team": g.get("teamAbbrev"), "opp": g.get("opponentAbbrev"),
            "home": 1 if g.get("homeRoadFlag") == "H" else 0,
            "started": int(g.get("gamesStarted") or 0), "decision": decision,
            "sa": sa, "ga": ga, "sv": max(sa - ga, 0), "so": so,
            "toi_min": round(_toi_minutes(g.get("toi")), 2),
            "fp": fp, "positive": int(fp > 0),
        })
    return rows


def fetch_career(client: NHLClient, player_id: int, name: str, anchor_season: int,
                 current: int) -> list[dict]:
    """All regular-season NHL games a goalie has played, every season of his career."""
    seasons = set(client.career_seasons(player_id)) | {anchor_season}
    rows: list[dict] = []
    for season in sorted(seasons):
        log = client.game_log(player_id, season, ttl_for(season, current))
        rows += parse_game_log(player_id, name, season, log)
    return rows


def current_goalies(client: NHLClient) -> list[dict]:
    """Every goalie on a current NHL roster (what the site lets you pick from)."""
    goalies: dict[int, dict] = {}
    for team in client.team_abbrevs():
        for g in client.roster_goalies(team):
            goalies[g["id"]] = g
    return sorted(goalies.values(), key=lambda g: g["name"])


def build_dataset(client: NHLClient, today: dt.date | None = None,
                  seasons_back: int = TRAINING_SEASONS_BACK,
                  extra_goalies: Iterable[dict] = (),
                  progress: Callable[[str], None] = lambda _m: None) -> tuple[pd.DataFrame, list[int]]:
    """Return (games table for every relevant goalie's whole career, training seasons)."""
    current = current_season(today)
    train_seasons = sorted(previous_seasons(current, seasons_back) + [current])

    # goalie id -> (name, most recent season we know he played)
    known: dict[int, tuple[str, int]] = {}
    for season in train_seasons:
        for g in client.season_goalies(season, ttl_for(season, current)):
            known[g["id"]] = (g["name"], season)
    for g in list(extra_goalies):
        known[g["id"]] = (g["name"], current)

    rows: list[dict] = []
    for i, (pid, (name, anchor)) in enumerate(sorted(known.items()), 1):
        if i % 20 == 0 or i == len(known):
            progress(f"  game logs: {i}/{len(known)} goalies")
        rows += fetch_career(client, pid, name, anchor, current)

    games = pd.DataFrame(rows, columns=GAME_COLUMNS)
    if games.empty:
        return games, train_seasons
    games["date"] = pd.to_datetime(games["date"])
    games = (games.drop_duplicates(["player_id", "game_id"])
                  .sort_values(["date", "game_id", "player_id"]).reset_index(drop=True))
    return games, train_seasons


def team_games(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: goals/shots for & against and result.

    Built from goalie logs, so it covers every game in the dataset window.
    """
    g = games.groupby(["game_id", "date", "season", "team", "opp"], as_index=False).agg(
        ga=("ga", "sum"), sa=("sa", "sum"),
        win=("decision", lambda d: int((d == "W").any())),
        starter=("player_id", lambda p: p.iloc[0]),
    )
    # starter = goalie with gamesStarted == 1 for that team-game
    starters = games[games["started"] == 1].drop_duplicates(["game_id", "team"])
    g = g.drop(columns="starter").merge(
        starters[["game_id", "team", "player_id"]].rename(columns={"player_id": "starter"}),
        on=["game_id", "team"], how="left")
    opp_side = g[["game_id", "team", "ga", "sa"]].rename(
        columns={"team": "opp", "ga": "gf", "sa": "sf"})
    g = g.merge(opp_side, on=["game_id", "opp"], how="left")
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)