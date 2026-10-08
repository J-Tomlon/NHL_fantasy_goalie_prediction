"""Command line:  python -m goalie_predictor <command>

  predict [NAMES...]   Predict next game for goalies (default: my_goalies.json)
  search NAME          Find a goalie's exact name / id
  train                Re-train the model on fresh data
  update               Train + predict every rostered goalie + write the website data
  serve                Open the website locally (http://localhost:8000)
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import sys
import webbrowser

from .config import DOCS_DIR, METRICS_PATH, SCORING
from .data import current_goalies
from .nhl_api import NHLClient
from .pipeline import Workspace, log, my_goalies, write_site
from .predict import _norm


def _pct(x) -> str:
    return "  -  " if x is None else f"{x * 100:4.0f}%"


def print_prediction(p: dict) -> None:
    print()
    head = f"{p['name']} ({p.get('team') or '?'})"
    if p.get("error"):
        print(f"{head}: {p['error']}")
        return
    g = p["next_game"]
    where = "vs" if g["home"] else "@"
    print("=" * 64)
    print(f"{head}   next: {g['date']} {where} {g['opp']}")
    print("=" * 64)
    print(f"  Chance of a positive fantasy game: {p['prob_positive'] * 100:.0f}%  -> {p['verdict']}")
    print(f"  Expected fantasy points:          {p['expected_points']:+.1f}")
    if p.get("starter_share_last10") is not None:
        print(f"  Started {p['starter_share_last10'] * 100:.0f}% of his team's last 10 games "
              "(prediction assumes he starts)")
    rows = [("Career", p["career"]), ("This season", p.get("season")),
            (f"vs {g['opp']}", p["vs_opponent"])]
    print(f"\n  {'':12} {'GS':>4} {'W-L-OTL':>9} {'SV%':>6} {'Pos%':>6} {'Avg pts':>8}")
    for label, s in rows:
        if not s or not s["starts"]:
            print(f"  {label:12} {'0':>4}")
            continue
        print(f"  {label:12} {s['starts']:>4} {s['record']:>9} "
              f"{(s['save_pct'] or 0):>6.3f} {_pct(s['pos_rate']):>6} {s['avg_points']:>8.2f}")
    if p["vs_opponent"]["games"]:
        print(f"\n  Recent games vs {g['opp']}:")
        for r in p["vs_opponent"]["games"]:
            print(f"    {r['date']}  {r['decision']:>2}  {r['saves']:>2} SV {r['ga']} GA  {r['points']:+.1f} pts")
    print("\n  Signals:")
    for sgl in p["signals"]:
        mark = {"good": "+", "bad": "-", "neutral": "="}[sgl["tone"]]
        print(f"    {mark} {sgl['label']}: {sgl['detail']}")


def cmd_predict(args) -> int:
    queries = args.goalies or my_goalies()
    if not queries:
        print('No goalies given. Pass names, e.g.  predict "Hellebuyck" "Sorokin",\n'
              "or list them in my_goalies.json.")
        return 2
    ws = Workspace()
    found, problems = ws.resolve(queries)
    for msg in problems:
        print("!", msg)
    if not found:
        return 1
    ws.ensure_model(retrain=args.retrain)
    preds = ws.predict(found)
    if args.json:
        print(json.dumps(preds, indent=2, default=str))
    else:
        for p in preds:
            print_prediction(p)
        print()
    return 0


def cmd_search(args) -> int:
    # rosters only; skips the heavy game-log load
    roster = current_goalies(NHLClient())
    q = _norm(" ".join(args.name))
    hits = [g for g in roster if q in _norm(g["name"])] if q else roster
    for g in hits:
        print(f"{g['id']:>8}  {g['team']:<4} {g['name']}")
    if not hits:
        print("No match.")
    return 0


def cmd_train(args) -> int:
    ws = Workspace(seasons_back=args.seasons)
    m = ws.fit()
    print(json.dumps({k: v for k, v in m.items() if k != "coefficients"}, indent=2))
    print(f"\nSaved model; full metrics in {METRICS_PATH}")
    return 0


def cmd_update(args) -> int:
    ws = Workspace(seasons_back=args.seasons)
    ws.fit()
    log(f"Predicting next game for {len(ws.roster)} goalies...")
    preds = ws.predict(ws.roster)
    write_site(ws, preds)
    return 0


def cmd_serve(args) -> int:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DOCS_DIR))
    url = f"http://localhost:{args.port}"
    print(f"Serving the site at {url}  (Ctrl+C to stop)")
    webbrowser.open(url)
    try:
        http.server.ThreadingHTTPServer(("", args.port), handler).serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="goalie_predictor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="predict next game for goalies")
    p.add_argument("goalies", nargs="*", help="names or NHL player ids")
    p.add_argument("--retrain", action="store_true", help="retrain before predicting")
    p.add_argument("--json", action="store_true", help="print raw JSON")
    p.set_defaults(func=cmd_predict)

    s = sub.add_parser("search", help="look up a goalie")
    s.add_argument("name", nargs="*")
    s.set_defaults(func=cmd_search)

    for name, fn in (("train", cmd_train), ("update", cmd_update)):
        t = sub.add_parser(name)
        t.add_argument("--seasons", type=int, default=4, help="completed seasons to train on")
        t.set_defaults(func=fn)

    v = sub.add_parser("serve", help="view the website locally")
    v.add_argument("--port", type=int, default=8000)
    v.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    print(f"Scoring: W {SCORING['W']:+g}  GA {SCORING['GA']:+g}  SV {SCORING['SV']:+g}  "
          f"SO {SCORING['SO']:+g}  OTL {SCORING['OTL']:+g}")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
