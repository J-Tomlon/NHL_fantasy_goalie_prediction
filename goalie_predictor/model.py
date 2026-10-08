"""Train, evaluate, save and load the goalie models."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import METRICS_PATH, MODEL_DIR, MODEL_PATH
from .features import FEATURES, Priors


def _classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))


def _regressor():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def _scores(y, p) -> dict:
    out = {"n": int(len(y)), "brier": float(brier_score_loss(y, p)),
           "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
           "accuracy": float(accuracy_score(y, p >= 0.5))}
    out["auc"] = float(roc_auc_score(y, p)) if len(set(y)) > 1 else None
    return out


def evaluate(X, y, yfp, meta) -> dict:
    """Walk-forward check: train on earlier seasons, test on the latest full one."""
    seasons = sorted(meta["season"].unique())
    counts = meta["season"].value_counts()
    full = [s for s in seasons if counts[s] >= 500]
    if len(full) < 2:
        return {"note": "not enough seasons to run a holdout test"}
    test_season = full[-1]
    tr = (meta["season"] < test_season).values
    te = (meta["season"] == test_season).values

    clf = _classifier().fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    reg = _regressor().fit(X[tr], yfp[tr])
    mae = float(np.mean(np.abs(reg.predict(X[te]) - yfp[te])))

    base_rate = float(y[tr].mean())
    naive = X.loc[te, "career_pos"].values  # "just use his career positive %"
    return {
        "test_season": int(test_season),
        "model": _scores(y[te], p),
        "baseline_league_rate": _scores(y[te], np.full(te.sum(), base_rate)),
        "baseline_career_rate": _scores(y[te], naive),
        "points_mae": mae,
        "points_mae_baseline": float(np.mean(np.abs(yfp[tr].mean() - yfp[te]))),
        "positive_rate": float(y[te].mean()),
    }


def train(X, y, yfp, meta, priors: Priors, train_seasons: list[int]) -> dict:
    metrics = evaluate(X, y, yfp, meta)
    clf = _classifier().fit(X, y)
    reg = _regressor().fit(X, yfp)
    coefs = clf[-1].coef_[0]
    metrics["coefficients"] = {f: round(float(c), 4) for f, c in zip(FEATURES, coefs)}
    metrics["training_rows"] = int(len(y))
    metrics["train_seasons"] = [int(s) for s in train_seasons]
    metrics["trained_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    bundle = {"clf": clf, "reg": reg, "features": FEATURES,
              "priors": dataclasses.asdict(priors), "metrics": metrics}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return bundle


def load():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def predict_one(bundle, feats: dict) -> dict:
    import pandas as pd
    x = pd.DataFrame([feats], columns=bundle["features"]).astype(float)
    prob = float(bundle["clf"].predict_proba(x)[0, 1])
    exp_fp = float(bundle["reg"].predict(x)[0])

    return {"prob_positive": round(prob, 4), "expected_points": round(exp_fp, 2)}
