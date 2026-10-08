"""Synthetic NHL data served in the same JSON shape as the real API (for offline tests)."""
from __future__ import annotations

import datetime as dt
import random

TEAMS = ["WPG", "NYI", "BOS", "TOR", "EDM", "VGK", "COL", "DAL"]


class FakeNHL:
    def __init__(self, today: dt.date, seasons=(20212022, 20222023, 20232024, 20242025,
                                                 20252026, 20262027), seed: int = 7):
        rng = random.Random(seed)
        self.today = today
        self.goalies = {}
        pid = 8470000
        for t in TEAMS:
            for role in ("starter", "backup"):
                pid += 1
                self.goalies[pid] = {"id": pid, "first": f"Goalie{pid % 1000}",
                                     "last": f"{t.title()}{role.title()}", "team": t,
                                     "skill": rng.gauss(0.905 if role == "starter" else 0.895, 0.008),
                                     "role": role}
        offense = {t: rng.gauss(3.0, 0.35) for t in TEAMS}
        self.logs = {pid: {} for pid in self.goalies}  # pid -> season -> [games]
        self.future = {t: [] for t in TEAMS}
        gid = 0
        for season in seasons:
            y = season // 10000
            day = dt.date(y, 10, 8)
            for rnd in range(70):
                order = TEAMS[:]
                rng.shuffle(order)
                for a, h in zip(order[::2], order[1::2]):
                    gid += 1
                    game_id = int(f"{y}02{gid % 10000:04d}")
                    if day >= today:
                        for t, home in ((h, True), (a, False)):
                            self.future[t].append({"id": game_id, "date": day, "home": home,
                                                   "opp": a if home else h})
                        continue
                    self._play(rng, season, game_id, day, h, a, offense)
                day += dt.timedelta(days=2)
                if day.month == 4 and day.day > 15:
                    break

    def _starter(self, rng, team):
        gs = [g for g in self.goalies.values() if g["team"] == team]
        return gs[0] if rng.random() < 0.7 else gs[1]

    def _play(self, rng, season, game_id, day, home, away, offense):
        res = {}
        for team, opp, is_home in ((home, away, True), (away, home, False)):
            g = self._starter(rng, team)
            shots = max(int(rng.gauss(29 + (offense[opp] - 3) * 3, 5)), 12)
            ga = sum(rng.random() > g["skill"] for _ in range(shots))
            res[team] = (g, shots, ga, is_home, opp)
        gh, ga_h = res[home][2], res[away][2]  # goals allowed by each team
        ot = gh == ga_h
        winner = home if (ga_h > gh or (ot and rng.random() < 0.55)) else away
        for team, (g, shots, ga, is_home, opp) in res.items():
            dec = "W" if team == winner else ("O" if ot else "L")
            self.logs[g["id"]].setdefault(season, []).append({
                "gameId": game_id, "teamAbbrev": team, "homeRoadFlag": "H" if is_home else "R",
                "gameDate": day.isoformat(), "goals": 0, "assists": 0, "gamesStarted": 1,
                "decision": dec, "shotsAgainst": shots, "goalsAgainst": ga,
                "savePctg": (shots - ga) / shots, "shutouts": int(ga == 0 and dec == "W"),
                "pim": 0, "toi": "60:00", "opponentAbbrev": opp})

    # ---- same interface as NHLClient ---------------------------------------
    def team_abbrevs(self):
        return sorted(TEAMS)

    def roster_goalies(self, team):
        return [{"id": g["id"], "name": f'{g["first"]} {g["last"]}', "team": team,
                 "headshot": None} for g in self.goalies.values() if g["team"] == team]

    def season_goalies(self, season, ttl):
        return [{"id": g["id"], "name": f'{g["first"]} {g["last"]}', "team": g["team"]}
                for g in self.goalies.values() if season in self.logs[g["id"]]]

    def game_log(self, pid, season, ttl):
        seasons = sorted(self.logs[pid], reverse=True)
        return {"seasonId": season, "gameTypeId": 2,
                "playerStatsSeasons": [{"season": s, "gameTypes": [2]} for s in seasons],
                "gameLog": list(reversed(self.logs[pid].get(season, [])))}

    def career_seasons(self, pid):
        return sorted(self.logs[pid])

    def landing(self, pid):
        g = self.goalies[pid]
        return {"firstName": {"default": g["first"]}, "lastName": {"default": g["last"]},
                "currentTeamAbbrev": g["team"], "headshot": None}

    def team_schedule(self, team):
        games = [{"id": f["id"], "gameType": 2, "gameDate": f["date"].isoformat(),
                  "gameState": "FUT", "startTimeUTC": f'{f["date"].isoformat()}T23:00:00Z',
                  "homeTeam": {"abbrev": team if f["home"] else f["opp"]},
                  "awayTeam": {"abbrev": f["opp"] if f["home"] else team}}
                 for f in self.future[team]]
        return {"games": games}
