"""Stand-in for nhlpy.NHLClient with the same method names and return shapes,
backed by FakeNHL data. Lets the real caching wrapper run offline."""
from __future__ import annotations


class ResourceNotFoundException(Exception):
    status_code = 404


class RateLimitExceededException(Exception):
    status_code = 429


class _NS:
    pass


class StubNHLPy:
    def __init__(self, fake, flaky_every: int = 0):
        self.fake, self.calls, self.flaky_every = fake, 0, flaky_every
        self.teams, self.stats, self.schedule = _NS(), _NS(), _NS()
        self.teams.teams = self._teams
        self.teams.team_roster = self._roster
        self.stats.goalie_stats_summary = self._summary
        self.stats.player_game_log = self._game_log
        self.stats.player_career_stats = self._landing
        self.schedule.team_season_schedule = self._schedule

    def _tick(self):
        self.calls += 1
        if self.flaky_every and self.calls % self.flaky_every == 0:
            raise RateLimitExceededException("slow down")

    # same signatures / shapes as nhl-api-py 3.x
    def _teams(self, date="now"):
        self._tick()
        return [{"name": t, "common_name": t, "abbr": t} for t in self.fake.team_abbrevs()]

    def _roster(self, team_abbr, season):
        self._tick()
        gs = self.fake.roster_goalies(team_abbr)
        if not gs:
            raise ResourceNotFoundException("no roster")
        return {"forwards": [], "defensemen": [], "goalies": [
            {"id": g["id"], "firstName": {"default": g["name"].split()[0]},
             "lastName": {"default": g["name"].split()[1]}, "headshot": None} for g in gs]}

    def _summary(self, start_season, end_season=None, game_type_id=2, limit=25, **_):
        self._tick()
        rows = self.fake.season_goalies(int(start_season), None)
        return [{"playerId": r["id"], "goalieFullName": r["name"], "teamAbbrevs": r["team"],
                 "seasonId": int(start_season)} for r in rows][: None if limit == -1 else limit]

    def _game_log(self, player_id, season_id, game_type):
        self._tick()
        return self.fake.game_log(int(player_id), int(season_id), None)["gameLog"]

    def _landing(self, player_id):
        self._tick()
        land = self.fake.landing(int(player_id))
        land["seasonTotals"] = (
            [{"season": s, "gameTypeId": 2, "leagueAbbrev": "NHL"}
             for s in self.fake.career_seasons(int(player_id))]
            + [{"season": 20182019, "gameTypeId": 2, "leagueAbbrev": "AHL"}])  # must be ignored
        return land

    def _schedule(self, team_abbr, season):
        self._tick()
        return self.fake.team_schedule(team_abbr)
