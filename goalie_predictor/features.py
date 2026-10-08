"""Leak-free features. Every feature for a game uses only games played before it.

The same GoalieState object builds training rows and the live prediction, so the
model never sees features computed differently at prediction time.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import H2H_PRIOR_SHOTS, H2H_PRIOR_STARTS, RECENT_N, TEAM_WINDOW
from .data import team_games

FEATURES = [
    "career_pos", "career_fp", "career_svpct", "experience",
    "season_pos", "recent_pos", "recent_fp", "recent_svpct", "recent_win",
    "h2h_pos", "h2h_fp", "h2h_svpct", "h2h_n",
    "home", "rest_days", "b2b",
    "team_win", "team_sa_pg", "team_ga_pg",
    "opp_gf_pg", "opp_sf_pg", "opp_win",
]

LABELS = {
    "career_pos": "Career rate of positive games",
    "career_fp": "Career avg fantasy points",
    "career_svpct": "Career save %",
    "experience": "NHL experience (starts)",
    "season_pos": "This season's positive rate",
    "recent_pos": f"Positive rate, last {RECENT_N} starts",
    "recent_fp": f"Avg fantasy points, last {RECENT_N} starts",
    "recent_svpct": f"Save %, last {RECENT_N} starts",
    "recent_win": f"Win rate, last {RECENT_N} starts",
    "h2h_pos": "Positive rate vs this opponent",
    "h2h_fp": "Avg fantasy points vs this opponent",
    "h2h_svpct": "Save % vs this opponent",
    "h2h_n": "Games of history vs this opponent",
    "home": "Home ice",
    "rest_days": "Days of rest",
    "b2b": "Back-to-back",
    "team_win": "His team's recent win rate",
    "team_sa_pg": "Shots his team allows per game",
    "team_ga_pg": "Goals his team allows per game",
    "opp_gf_pg": "Opponent goals per game",
    "opp_sf_pg": "Opponent shots per game",
    "opp_win": "Opponent recent win rate",
}


@dataclass
class Priors:
    pos: float = 0.62
    fp: float = 2.0
    svpct: float = 0.903
    win: float = 0.5
    gf_pg: float = 3.0
    sf_pg: float = 29.0
    win_rate: float = 0.5

    @classmethod
    def from_games(cls, games: pd.DataFrame) -> "Priors":
        s = games[games["started"] == 1]
        if s.empty:
            return cls()
        return cls(pos=float(s["positive"].mean()), fp=float(s["fp"].mean()),
                   svpct=float(s["sv"].sum() / max(s["sa"].sum(), 1)),
                   win=float((s["decision"] == "W").mean()),
                   gf_pg=float(s["ga"].mean()), sf_pg=float(s["sa"].mean()))


def _shrink(num: float, n: float, prior: float, k: float) -> float:
    return (num + k * prior) / (n + k)


@dataclass
class _Tally:
    n: int = 0
    pos: int = 0
    fp: float = 0.0
    sv: int = 0
    sa: int = 0
    w: int = 0
    l: int = 0
    o: int = 0
    so: int = 0

    def add(self, r) -> None:
        self.n += 1
        self.pos += int(r.positive)
        self.fp += float(r.fp)
        self.sv += int(r.sv)
        self.sa += int(r.sa)
        self.w += int(r.decision == "W")
        self.l += int(r.decision == "L")
        self.o += int(r.decision == "O")
        self.so += int(r.so)


@dataclass
class GoalieState:
    priors: Priors
    career: _Tally = field(default_factory=_Tally)
    by_season: dict = field(default_factory=lambda: defaultdict(_Tally))
    by_opp: dict = field(default_factory=lambda: defaultdict(_Tally))
    opp_games: dict = field(default_factory=lambda: defaultdict(list))
    recent: deque = field(default_factory=lambda: deque(maxlen=RECENT_N))
    last_date: pd.Timestamp | None = None

    def update(self, r) -> None:
        self.last_date = pd.Timestamp(r.date)
        if not r.started:
            return  # relief appearances only count for rest
        self.career.add(r)
        self.by_season[r.season].add(r)
        self.by_opp[r.opp].add(r)
        self.opp_games[r.opp].append(r)
        self.recent.append(r)

    def features(self, opp: str, home: int, date, season: int) -> dict:
        p, c = self.priors, self.career
        career_pos = _shrink(c.pos, c.n, p.pos, 10)
        career_fp = _shrink(c.fp, c.n, p.fp, 10)
        career_sv = _shrink(c.sv, c.sa, p.svpct, 300)
        career_win = _shrink(c.w, c.n, p.win, 10)

        s = self.by_season.get(season, _Tally())
        rec = list(self.recent)
        rn = len(rec)
        r_pos = sum(x.positive for x in rec)
        r_fp = sum(x.fp for x in rec)
        r_sv = sum(x.sv for x in rec)
        r_sa = sum(x.sa for x in rec)
        r_w = sum(x.decision == "W" for x in rec)

        h = self.by_opp.get(opp, _Tally())
        if self.last_date is None:
            rest = 4
        else:
            rest = max(min((pd.Timestamp(date) - self.last_date).days, 7), 0)

        return {
            "career_pos": career_pos,
            "career_fp": career_fp,
            "career_svpct": career_sv,
            "experience": math.log1p(c.n),
            "season_pos": _shrink(s.pos, s.n, career_pos, 8),
            "recent_pos": _shrink(r_pos, rn, career_pos, 3),
            "recent_fp": _shrink(r_fp, rn, career_fp, 3),
            "recent_svpct": _shrink(r_sv, r_sa, career_sv, 100),
            "recent_win": _shrink(r_w, rn, career_win, 3),
            "h2h_pos": _shrink(h.pos, h.n, career_pos, H2H_PRIOR_STARTS),
            "h2h_fp": _shrink(h.fp, h.n, career_fp, H2H_PRIOR_STARTS),
            "h2h_svpct": _shrink(h.sv, h.sa, career_sv, H2H_PRIOR_SHOTS),
            "h2h_n": math.log1p(h.n),
            "home": int(home),
            "rest_days": rest,
            "b2b": int(rest == 1),
        }


# ------------------------------------------------------------------ team form
def team_form(games: pd.DataFrame, seasons: list[int]) -> tuple[pd.DataFrame, dict]:
    """Pre-game rolling team form keyed by (game_id, team), plus latest snapshot."""
    tg = team_games(games[games["season"].isin(seasons)])
    tg = tg.dropna(subset=["gf"])  # need both teams' goalies to know goals for
    if tg.empty:
        return pd.DataFrame(columns=["game_id", "team"]), {}
    cols = {"win": "win", "sa": "sa_pg", "ga": "ga_pg", "gf": "gf_pg", "sf": "sf_pg"}
    roll = tg.groupby("team")[list(cols)].transform(
        lambda x: x.rolling(TEAM_WINDOW, min_periods=1).mean().shift(1))
    pre = pd.concat([tg[["game_id", "team"]], roll.rename(columns=cols)], axis=1)

    snapshot = {}
    for team, grp in tg.groupby("team"):
        last = grp.tail(TEAM_WINDOW)
        snapshot[team] = {cols[k]: float(last[k].mean()) for k in cols}
        snapshot[team]["recent_starters"] = grp["starter"].tail(10).tolist()
    return pre, snapshot


def team_feats(team_ctx: dict | None, opp_ctx: dict | None, priors: Priors) -> dict:
    t, o = team_ctx or {}, opp_ctx or {}

    def val(d, k, default):
        v = d.get(k)
        return default if v is None or (isinstance(v, float) and np.isnan(v)) else v

    return {
        "team_win": val(t, "win", priors.win_rate),
        "team_sa_pg": val(t, "sa_pg", priors.sf_pg),
        "team_ga_pg": val(t, "ga_pg", priors.gf_pg),
        "opp_gf_pg": val(o, "gf_pg", priors.gf_pg),
        "opp_sf_pg": val(o, "sf_pg", priors.sf_pg),
        "opp_win": val(o, "win", priors.win_rate),
    }


# ------------------------------------------------------------- training frame
def build_training_frame(games: pd.DataFrame, train_seasons: list[int]):
    """Return (X, y_positive, y_points, meta, priors)."""
    window = games[games["season"].isin(train_seasons)]
    priors = Priors.from_games(window)
    pre, _ = team_form(games, train_seasons)
    pre_idx = {(r.game_id, r.team): r._asdict() for r in pre.itertuples(index=False)}

    states: dict[int, GoalieState] = {}
    rows, y, yfp, meta = [], [], [], []
    for r in games.itertuples(index=False):
        st = states.setdefault(r.player_id, GoalieState(priors))
        if r.started and r.season in train_seasons:
            f = st.features(r.opp, r.home, r.date, r.season)
            f.update(team_feats(pre_idx.get((r.game_id, r.team)),
                                pre_idx.get((r.game_id, r.opp)), priors))
            rows.append(f)
            y.append(int(r.positive))
            yfp.append(float(r.fp))
            meta.append({"player_id": r.player_id, "season": r.season, "date": r.date})
        st.update(r)

    X = pd.DataFrame(rows, columns=FEATURES).astype(float)
    return X, np.array(y), np.array(yfp), pd.DataFrame(meta), priors


def goalie_state(goalie_games: pd.DataFrame, priors: Priors) -> GoalieState:
    st = GoalieState(priors)
    for r in goalie_games.sort_values("date").itertuples(index=False):
        st.update(r)
    return st


def next_game_features(state: GoalieState, opp: str, home: int, date: dt.date, season: int,
                       team_ctx: dict | None, opp_ctx: dict | None) -> dict:
    f = state.features(opp, home, pd.Timestamp(date), season)
    f.update(team_feats(team_ctx, opp_ctx, state.priors))
    return f
