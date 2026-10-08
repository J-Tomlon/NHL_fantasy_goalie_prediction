"""Caching wrapper around nhl-api-py (https://github.com/coreyjs/nhl-api-py).

nhl-api-py handles the HTTP calls to the NHL's public APIs; this module adds a
disk cache (completed seasons are cached forever, current data for a few hours),
polite pacing, retries, and returns data in the shapes the rest of the project
expects.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from .config import CACHE_DIR, REQUEST_PAUSE, TTL_CURRENT, TTL_FOREVER


def _default_api():
    try:
        from nhlpy import NHLClient as _NHLPy
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("nhl-api-py is not installed. Run:  pip install -r requirements.txt") from exc
    return _NHLPy(timeout=20)


def _error_kind(exc: Exception) -> str:
    """Classify nhl-api-py / httpx errors without importing them at module load."""
    name = type(exc).__name__
    if name == "ResourceNotFoundException" or getattr(exc, "status_code", None) == 404:
        return "missing"
    if name in ("RateLimitExceededException", "ServerErrorException") or \
            name.endswith(("TimeoutException", "ConnectError", "ReadTimeout", "ConnectTimeout",
                           "RemoteProtocolError", "ReadError")):
        return "retry"
    return "fatal"


class NHLClient:
    def __init__(self, cache_dir: Path = CACHE_DIR, offline: bool = False, api=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self._api = api
        self._last_call = 0.0

    @property
    def api(self):
        if self._api is None:
            self._api = _default_api()
        return self._api

    # ------------------------------------------------------------------ core
    def _cache_path(self, key: str) -> Path:
        slug = "".join(c if c.isalnum() else "_" for c in key)[:80]
        return self.cache_dir / f"{slug}_{hashlib.sha1(key.encode()).hexdigest()[:10]}.json"

    def cached(self, key: str, fetch: Callable[[], Any], ttl: float | None = TTL_CURRENT,
               missing: Any = None) -> Any:
        path = self._cache_path(key)
        if path.exists():
            age = time.time() - path.stat().st_mtime
            if ttl is None or age < ttl or self.offline:
                return json.loads(path.read_text())
        if self.offline:
            raise RuntimeError(f"offline mode and nothing cached for {key}")

        for attempt in range(5):
            wait = REQUEST_PAUSE - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            try:
                data = fetch()
                break
            except Exception as exc:
                kind = _error_kind(exc)
                if kind == "missing":
                    data = missing
                    break
                if kind == "retry" and attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                if path.exists():  # serve stale cache rather than fail
                    return json.loads(path.read_text())
                raise RuntimeError(f"NHL API call failed for {key}: {exc}") from exc

        path.write_text(json.dumps(data))
        return data

    # ------------------------------------------------------------- endpoints
    def team_abbrevs(self) -> list[str]:
        teams = self.cached("teams", lambda: self.api.teams.teams())
        return sorted({t["abbr"] for t in teams if t.get("abbr")})

    def roster_goalies(self, team: str) -> list[dict]:
        data = self.cached(f"roster_{team}_current",
                           lambda: self.api.teams.team_roster(team_abbr=team, season="current"),
                           missing={}) or {}
        return [{"id": g["id"],
                 "name": f'{g["firstName"]["default"]} {g["lastName"]["default"]}',
                 "team": team, "headshot": g.get("headshot")}
                for g in data.get("goalies", [])]

    def season_goalies(self, season: int, ttl: float | None) -> list[dict]:
        """Every goalie who appeared in a regular-season game that season."""
        rows = self.cached(
            f"goalie_summary_{season}",
            lambda: self.api.stats.goalie_stats_summary(
                start_season=str(season), end_season=str(season), game_type_id=2, limit=-1),
            ttl=ttl, missing=[]) or []
        return [{"id": r["playerId"], "name": r["goalieFullName"],
                 "team": (r.get("teamAbbrevs") or "").split(",")[-1].strip()} for r in rows]

    def game_log(self, player_id: int, season: int, ttl: float | None) -> dict:
        games = self.cached(
            f"gamelog_{player_id}_{season}",
            lambda: self.api.stats.player_game_log(
                player_id=str(player_id), season_id=str(season), game_type=2),
            ttl=ttl, missing=[]) or []
        return {"gameLog": games}

    def landing(self, player_id: int) -> dict:
        return self.cached(f"landing_{player_id}",
                           lambda: self.api.stats.player_career_stats(player_id=str(player_id)),
                           ttl=24 * 3600, missing={}) or {}

    def career_seasons(self, player_id: int) -> list[int]:
        """Regular seasons the player appeared in the NHL (from his career totals)."""
        totals = self.landing(player_id).get("seasonTotals", []) or []
        return sorted({int(s["season"]) for s in totals
                       if s.get("leagueAbbrev") == "NHL" and s.get("gameTypeId") == 2})

    def team_schedule(self, team: str) -> dict:
        return self.cached(f"schedule_{team}_now",
                           lambda: self.api.schedule.team_season_schedule(team_abbr=team, season="now"),
                           missing={}) or {}


def ttl_for(season: int, current: int) -> float | None:
    return TTL_CURRENT if season >= current else TTL_FOREVER
