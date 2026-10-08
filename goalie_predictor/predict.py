"""Turn a trained model + data into a next-game prediction for chosen goalies."""
from __future__ import annotations

import datetime as dt
import unicodedata

import pandas as pd

from .config import current_season, season_for
from .data import fetch_career
from .features import Priors, goalie_state, next_game_features
from .model import predict_one
from .nhl_api import NHLClient

LIVE_OR_FUTURE = {"FUT", "PRE", "LIVE", "CRIT"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def resolve_goalies(queries: list[str], roster: list[dict]) -> tuple[list[dict], list[str]]:
    """Match names (or numeric ids) against rostered goalies. Returns (found, problems)."""
    found, problems = [], []
    for q in queries:
        q = str(q).strip()
        if q.isdigit():
            hit = [g for g in roster if g["id"] == int(q)] or [{"id": int(q), "name": q, "team": None}]
        else:
            nq = _norm(q)
            hit = [g for g in roster if _norm(g["name"]) == nq] or \
                  [g for g in roster if nq in _norm(g["name"])]
        if len(hit) == 1:
            found.append(hit[0])
        elif not hit:
            problems.append(f'No rostered goalie matches "{q}".')
        else:
            names = ", ".join(f'{g["name"]} ({g["team"]})' for g in hit)
            problems.append(f'"{q}" matches several goalies: {names}. Be more specific.')
    return found, problems


def next_game(client: NHLClient, team: str, today: dt.date) -> dict | None:
    sched = client.team_schedule(team)
    for g in sched.get("games", []):
        if g.get("gameType") != 2 or g.get("gameState") not in LIVE_OR_FUTURE:
            continue
        if dt.date.fromisoformat(g["gameDate"]) < today:
            continue
        home = g["homeTeam"]["abbrev"] == team
        return {
            "game_id": g["id"], "date": g["gameDate"], "start_utc": g.get("startTimeUTC"),
            "home": int(home),
            "opp": g["awayTeam"]["abbrev"] if home else g["homeTeam"]["abbrev"],
            "state": g.get("gameState"),
        }
    return None


def _record(t) -> str:
    return f"{t.w}-{t.l}-{t.o}"


def _tally_summary(t) -> dict:
    return {"starts": t.n, "record": _record(t), "shutouts": t.so,
            "pos_rate": round(t.pos / t.n, 3) if t.n else None,
            "avg_points": round(t.fp / t.n, 2) if t.n else None,
            "save_pct": round(t.sv / t.sa, 3) if t.sa else None}


def _game_row(r) -> dict:
    return {"date": pd.Timestamp(r.date).date().isoformat(), "opp": r.opp, "home": int(r.home),
            "decision": r.decision or "-", "saves": int(r.sv), "ga": int(r.ga),
            "so": int(r.so), "points": float(r.fp)}


def signals(f: dict, pr: Priors, opp: str, h2h_starts: int) -> list[dict]:
    """Plain-language context for the prediction (good / bad / neutral)."""
    out = []

    def add(label, detail, diff, good_if_positive=True, deadband=0.0):
        if abs(diff) <= deadband:
            tone = "neutral"
        else:
            tone = "good" if (diff > 0) == good_if_positive else "bad"
        out.append({"label": label, "detail": detail, "tone": tone})

    d = f["recent_pos"] - f["career_pos"]
    add("Recent form", f"{f['recent_pos']:.0%} positive lately vs {f['career_pos']:.0%} career "
        f"(save % {f['recent_svpct']:.3f})", d, deadband=0.04)
    if h2h_starts:
        add(f"History vs {opp}", f"{h2h_starts} career starts; adjusted positive rate "
            f"{f['h2h_pos']:.0%} vs {f['career_pos']:.0%} overall", f["h2h_pos"] - f["career_pos"],
            deadband=0.03)
    else:
        out.append({"label": f"History vs {opp}", "detail": "No career starts against them yet",
                    "tone": "neutral"})
    add("Opponent offense", f"{opp} scoring {f['opp_gf_pg']:.2f} goals/game "
        f"(league {pr.gf_pg:.2f})", f["opp_gf_pg"] - pr.gf_pg, good_if_positive=False, deadband=0.15)
    add("Team support", f"His team has won {f['team_win']:.0%} of its last 15",
        f["team_win"] - 0.5, deadband=0.07)
    rest = "back-to-back" if f["b2b"] else f"{int(f['rest_days'])} days rest"
    out.append({"label": "Venue & rest", "detail": f"{'Home' if f['home'] else 'Road'} game, {rest}",
                "tone": "bad" if f["b2b"] else ("good" if f["home"] else "neutral")})
    return out


def verdict(p: float) -> str:
    if p >= 0.65:
        return "Likely positive"
    if p >= 0.55:
        return "Lean positive"
    if p >= 0.45:
        return "Toss-up"
    return "Risky"


def predict_goalie(client: NHLClient, bundle, games: pd.DataFrame, snapshot: dict,
                   goalie: dict, today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    season = current_season(today)
    landing = client.landing(goalie["id"])
    name = (f'{landing["firstName"]["default"]} {landing["lastName"]["default"]}'
            if landing.get("firstName") else goalie["name"])
    team = landing.get("currentTeamAbbrev") or goalie.get("team")
    out = {"id": goalie["id"], "name": name, "team": team,
           "headshot": landing.get("headshot") or goalie.get("headshot")}
    if not team:
        return {**out, "error": "Not currently on an NHL team."}

    g_games = games[games["player_id"] == goalie["id"]]
    if g_games.empty:  # e.g. queried by id but not part of the league pull
        rows = fetch_career(client, goalie["id"], name, season, season)
        g_games = pd.DataFrame(rows)
        if not g_games.empty:
            g_games["date"] = pd.to_datetime(g_games["date"])

    priors = Priors(**bundle["priors"])
    state = goalie_state(g_games, priors) if not g_games.empty else goalie_state(
        pd.DataFrame(columns=games.columns), priors)

    nxt = next_game(client, team, today)
    if not nxt:
        return {**out, "error": "No upcoming regular-season game found."}

    gdate = dt.date.fromisoformat(nxt["date"])
    feats = next_game_features(state, nxt["opp"], nxt["home"], gdate, season_for(gdate),
                               snapshot.get(team), snapshot.get(nxt["opp"]))
    pred = predict_one(bundle, feats)

    starters = (snapshot.get(team) or {}).get("recent_starters", [])
    share = (sum(1 for s in starters if s == goalie["id"]) / len(starters)) if starters else None

    starts = g_games[g_games["started"] == 1].sort_values("date") if not g_games.empty else g_games
    vs = starts[starts["opp"] == nxt["opp"]] if not starts.empty else starts
    this_season = state.by_season.get(season_for(gdate))

    return {
        **out,
        "next_game": nxt,
        **pred,
        "verdict": verdict(pred["prob_positive"]),
        "signals": signals(feats, priors, nxt["opp"], state.by_opp.get(nxt["opp"], type(state.career)()).n),
        "starter_share_last10": None if share is None else round(share, 2),
        "career": _tally_summary(state.career),
        "season": _tally_summary(this_season) if this_season else None,
        "vs_opponent": {**_tally_summary(state.by_opp.get(nxt["opp"], type(state.career)())),
                        "games": [_game_row(r) for r in vs.tail(5).iloc[::-1].itertuples(index=False)]},
        "last5": [_game_row(r) for r in starts.tail(5).iloc[::-1].itertuples(index=False)],
    }

