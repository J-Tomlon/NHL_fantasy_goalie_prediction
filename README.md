# Crease Check — NHL fantasy goalie predictor

Predicts whether a goalie will **go positive** (score > 0 fantasy points) in his
next game, using his full career game history, his career history against that
night's opponent, recent form, and both teams' recent form.

**League scoring** (edit in `goalie_predictor/config.py`):

| Win | Goal against | Save | Shutout | OT loss |
|----:|----:|----:|----:|----:|
| +3 | −1.5 | +0.2 | +4 | +1 |

Two ways to use it:

- **Website (GitHub Pages)** — pick your goalies from a search box; they're
  remembered in your browser. A table shows every goalie's next game, sorted by
  chance of going positive (useful for streaming pickups). A GitHub Action
  refreshes the data twice a day.
- **Command line** — `python -m goalie_predictor predict 

---

## Setup on any machine (.venv)

Requires Python 3.10+.

```bash
git clone https://github.com/<you>/nhl-goalie-predictor.git
cd nhl-goalie-predictor
bash scripts/setup.sh          # macOS / Linux
# .\scripts\setup.ps1          # Windows PowerShell
source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
```

The `.venv` folder is git-ignored, so each machine builds its own from
`requirements.txt`. To do it by hand: `python -m venv .venv` then
`.venv/bin/pip install -r requirements.txt`.

## Command line

```bash
python -m goalie_predictor predict                  # goalies listed in my_goalies.json
python -m goalie_predictor predict "Shesterkin"     # any name (partial is fine)
python -m goalie_predictor predict 8476945          # or an NHL player id
python -m goalie_predictor search sor               # find exact names / ids
python -m goalie_predictor update                   # retrain + rebuild website data
python -m goalie_predictor serve                    # view the site at localhost:8000
python -m goalie_predictor predict --retrain        # force a fresh model
python -m goalie_predictor predict --json           # raw output
```

**Changing your goalies:** edit `my_goalies.json` (CLI default and the site's
starting picks), or just add/remove them on the website.

The **first run takes a few minutes** — it downloads every goalie's career game
logs. Responses are cached in `data/cache/`: completed seasons forever, the
current season for 3 hours, so later runs are quick.

Example output:

```
================================================================
Connor Hellebuyck (WPG)   next: 2026-10-10 @ DAL
================================================================
  Chance of a positive fantasy game: 70%  -> Likely positive
  Expected fantasy points:          +2.8
  Started 80% of his team's last 10 games (prediction assumes he starts)

                 GS   W-L-OTL    SV%   Pos%  Avg pts
  Career        ...
  This season   ...
  vs DAL        ...

  Signals:
    + Recent form: 80% positive lately vs 70% career (save % .918)
    - History vs DAL: 34 career starts; adjusted positive rate 66% vs 70% overall
    ...
```

## GitHub Pages setup

1. Create a GitHub repo and push this folder (`git init`, `git add .`,
   `git commit`, `git push`).
2. **Settings → Pages →** Source: *Deploy from a branch*, Branch: `main`,
   Folder: `/docs`. Save.
3. **Settings → Actions → General → Workflow permissions →** *Read and write
   permissions*. Save.
4. **Actions tab → Update predictions → Run workflow** to load real data now.
   After that it runs automatically at 6:15 AM and 1:15 PM Arizona time.

Your site will be at `https://<you>.github.io/<repo>/`. Until the first
workflow run it shows clearly labelled sample data.

You can also run `python -m goalie_predictor update` locally and push
`docs/data/predictions.json` yourself.

## How the model works

- **Data** (public NHL API, via [nhl-api-py](https://github.com/coreyjs/nhl-api-py)): every goalie's regular-season game logs for his
  whole NHL career, current rosters, and team schedules.
- **Target:** fantasy points > 0 under your scoring, computed per start.
- **Features** — all computed only from games *before* the one being predicted,
  so there's no peeking:
  - Career: positive rate, avg points, save %, experience
  - Form: this season's positive rate; last-10-starts positive rate, points,
    save %, win rate
  - **Vs this opponent:** career positive rate, avg points, save %, number of
    starts — shrunk toward his overall numbers when the sample is small (3
    starts vs a team shouldn't outweigh 300 overall)
  - Context: home/road, days of rest, back-to-back
  - His team's last 15 games: win rate, shots and goals allowed
  - Opponent's last 15 games: goals and shots per game, win rate
- **Models:** logistic regression for the probability, ridge regression for
  expected points (both scikit-learn, standardized features). Trained on the last
  4 completed seasons plus the current one.
- **Honest check:** every training run back-tests on the most recent full
  season using a model trained only on earlier seasons, and compares against
  "just use his career positive %". Results are in `models/metrics.json` and the
  site footer.

**Expect modest accuracy.** One goalie game is noisy — a 30-save win and a
5-goal loss can come from the same goalie a night apart. The value here is
ranking which goalie, on which night, has better odds, not certainty.

**Starts aren't known in advance.** Predictions assume he starts. The site
flags goalies who have started under 60% of their team's last 10 games.

## Project layout

```
goalie_predictor/
  config.py      scoring, paths, windows
  nhl_api.py     caching wrapper around nhl-api-py
  data.py        game logs → tidy tables
  features.py    leak-free feature building (shared by training & prediction)
  model.py       train / back-test / save
  predict.py     next game lookup, goalie matching, prediction summary
  pipeline.py    glue used by CLI and the GitHub Action
  cli.py         command line
docs/            GitHub Pages site (index.html, app.js, style.css, data/)
tests/           offline tests on synthetic data (incl. an nhl-api-py stand-in)
.github/workflows/update-predictions.yml
my_goalies.json  your goalies
```

Run the tests with `python tests/test_pipeline.py` (no internet needed).
