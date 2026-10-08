"""Offline end-to-end test:  python -m pytest tests  (or python tests/test_pipeline.py)."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_nhl import FakeNHL  # noqa: E402

from goalie_predictor import config  # noqa: E402
from goalie_predictor.features import build_training_frame  # noqa: E402
from goalie_predictor.scoring import fantasy_points  # noqa: E402

TODAY = dt.date(2026, 10, 25)


def test_scoring():
    assert fantasy_points("W", 30, 2, 0) == round(3 - 3 + 5.6, 2)          # 5.6
    assert fantasy_points("W", 25, 0, 1) == round(3 + 5.0 + 4, 2)          # 12.0
    assert fantasy_points("O", 30, 3, 0) == round(1 - 4.5 + 5.4, 2)        # 1.9
    assert fantasy_points("L", 20, 5, 0) == round(-7.5 + 3.0, 2)           # -4.5


def test_no_leakage():
    fake = FakeNHL(TODAY)
    from goalie_predictor.data import build_dataset
    games, seasons = build_dataset(fake, TODAY, 5)  # window = whole fake history
    X, y, yfp, meta, _ = build_training_frame(games, seasons)
    # first start of every goalie must have zero head-to-head history
    first = meta.groupby("player_id").head(1).index
    assert (X.loc[first, "h2h_n"] == 0).all() and (X.loc[first, "experience"] == 0).all()


def test_end_to_end(tmp_path=None, monkeypatch=None):
    tmp = Path(tmp_path or (Path(__file__).parent / "_tmp"))
    tmp.mkdir(exist_ok=True)
    config.MODEL_PATH = tmp / "m.joblib"
    import goalie_predictor.model as model_mod
    import goalie_predictor.pipeline as pipe
    model_mod.MODEL_PATH, model_mod.METRICS_PATH, model_mod.MODEL_DIR = tmp / "m.joblib", tmp / "m.json", tmp
    pipe.SITE_DATA_PATH = tmp / "docs" / "data" / "predictions.json"

    ws = pipe.Workspace(client=FakeNHL(TODAY), today=TODAY, seasons_back=3)
    m = ws.fit()
    assert m["model"]["auc"] > 0.55, m
    found, problems = ws.resolve(["wpgstarter", "Goalie99999"])
    assert len(found) == 1 and len(problems) == 1
    preds = ws.predict(ws.roster)
    ok = [p for p in preds if "error" not in p]
    assert len(ok) == len(ws.roster)
    p = next(p for p in ok if p["id"] == found[0]["id"])
    assert 0 < p["prob_positive"] < 1 and p["next_game"]["opp"]
    assert p["vs_opponent"]["starts"] > 0
    pipe.write_site(ws, preds)
    return m, p


def test_nhlpy_wrapper():
    """Run the real caching wrapper against a stand-in with nhl-api-py's interface."""
    import shutil
    import tempfile
    from stub_nhlpy import StubNHLPy
    import goalie_predictor.nhl_api as api_mod
    from goalie_predictor.data import build_dataset, current_goalies
    from goalie_predictor.nhl_api import NHLClient

    api_mod.REQUEST_PAUSE = 0
    real_sleep = api_mod.time.sleep
    cache = Path(tempfile.mkdtemp())
    fake = FakeNHL(TODAY)
    try:
        stub = StubNHLPy(fake)
        client = NHLClient(cache_dir=cache, api=stub)
        roster = current_goalies(client)
        assert len(roster) == 16
        direct, _ = build_dataset(fake, TODAY, 3, extra_goalies=roster)
        wrapped, _ = build_dataset(client, TODAY, 3, extra_goalies=roster)
        assert len(wrapped) == len(direct) and wrapped["fp"].sum() == direct["fp"].sum()
        assert client.career_seasons(roster[0]["id"]) == fake.career_seasons(roster[0]["id"])
        first_calls = stub.calls
        build_dataset(client, TODAY, 3, extra_goalies=roster)       # all from cache now
        assert stub.calls == first_calls, (stub.calls, first_calls)
        assert client.team_schedule("WPG")["games"]

        # rate-limit errors are retried; a 404 becomes an empty result
        api_mod.time.sleep = lambda s: None
        flaky = NHLClient(cache_dir=Path(tempfile.mkdtemp()), api=StubNHLPy(fake, flaky_every=3))
        assert len(current_goalies(flaky)) == 16
        assert flaky.roster_goalies("XXX") == []
    finally:
        api_mod.time.sleep = real_sleep
        shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    test_nhlpy_wrapper()
    test_scoring()
    test_no_leakage()
    metrics, pred = test_end_to_end()
    import json
    print(json.dumps({k: v for k, v in metrics.items() if k != "coefficients"}, indent=2))
    from goalie_predictor.cli import print_prediction
    print_prediction(pred)
    print("\nALL TESTS PASSED")
