"""High-level steps used by the CLI and the GitHub Actions job."""
from __future__ import annotations

import datetime as dt
import json
import math

from .config import MY_GOALIES_PATH, SCORING, SITE_DATA_PATH, TRAINING_SEASONS_BACK
from .data import build_dataset, current_goalies
from .features import build_training_frame, team_form
from .model import load, train
from .nhl_api import NHLClient
from .predict import predict_goalie, resolve_goalies


def log(msg: str) -> None:
    print(msg, flush=True)


def my_goalies() -> list[str]:
    if not MY_GOALIES_PATH.exists():
        return []
    return json.loads(MY_GOALIES_PATH.read_text()).get("goalies", [])


class Workspace:
    """Loads (and caches) everything a prediction needs."""

    def __init__(self, client: NHLClient | None = None, today: dt.date | None = None,
                 seasons_back: int = TRAINING_SEASONS_BACK):
        self.client = client or NHLClient()
        self.today = today or dt.date.today()
        self.seasons_back = seasons_back
        log("Loading current NHL rosters...")
        self.roster = current_goalies(self.client)
        log(f"  {len(self.roster)} rostered goalies")
        log("Loading goalie game logs (cached after the first run)...")
        self.games, self.train_seasons = build_dataset(
            self.client, self.today, seasons_back, extra_goalies=self.roster, progress=log)
        log(f"  {len(self.games):,} goalie games loaded")
        _, self.snapshot = team_form(self.games, self.train_seasons)
        self.bundle = None

    def fit(self) -> dict:
        log("Building features and training the model...")
        X, y, yfp, meta, priors = build_training_frame(self.games, self.train_seasons)
        self.bundle = train(X, y, yfp, meta, priors, self.train_seasons)
        return self.bundle["metrics"]

    def ensure_model(self, retrain: bool = False) -> None:
        self.bundle = None if retrain else load()
        if self.bundle is None:
            self.fit()

    def predict(self, goalies: list[dict]) -> list[dict]:
        out = []
        for g in goalies:
            try:
                out.append(predict_goalie(self.client, self.bundle, self.games,
                                          self.snapshot, g, self.today))
            except Exception as exc:  # one bad goalie shouldn't sink the batch
                out.append({"id": g["id"], "name": g["name"], "team": g.get("team"),
                            "error": f"{type(exc).__name__}: {exc}"})
        return out

    def resolve(self, queries: list[str]):
        return resolve_goalies(queries, self.roster)


def _json_safe(obj):
    """Replace NaN/inf (which browsers reject in JSON) with null, recursively."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalars
        return _json_safe(obj.item())
    return obj


def write_site(ws: Workspace, predictions: list[dict]) -> None:
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sample": False,
        "scoring": SCORING,
        "metrics": {k: v for k, v in ws.bundle["metrics"].items() if k != "coefficients"},
        "default_goalies": [g["id"] for g in ws.resolve(my_goalies())[0]],
        "goalies": sorted(predictions, key=lambda p: (p.get("error") is not None, p["name"])),
    }
    SITE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_safe(payload), indent=1, default=str, allow_nan=False)
    SITE_DATA_PATH.write_text(text)
    log(f"Wrote {SITE_DATA_PATH.relative_to(SITE_DATA_PATH.parents[2])} "
        f"({len(predictions)} goalies)")