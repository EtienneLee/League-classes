# Do Riot's champion classes match how champions play?

Baseline checkpoint code for a course project. Each champion's average in-game stats are used to predict Riot's primary class (first tag) with kNN and ridge logistic regression, scored by repeated stratified 5-fold cross-validation with bootstrap intervals over champions.

## Data (not included)

The data is not redistributed here. Download it yourself and put it in a `data/` folder:

- Match data: Kaggle, "League of Legends Ranked Match Data (Season 15, EUN)", CC BY-SA 4.0. https://www.kaggle.com/datasets/jakubkrasuski/league-of-legends-ranked-match-data-season-15 (save the CSV as `data/match.csv`)
- Class labels: Riot Data Dragon, patch 15.3.1, used only as a label lookup under Riot's developer terms. https://ddragon.leagueoflegends.com/cdn/15.3.1/data/en_US/champion.json (save as `data/champion.json`)

## Run

```
pip install -r requirements.txt

# all 23 stats (10 play stats + 13 final_* stats)
python3 baseline.py data/match.csv data/champion.json out

# 10 play stats only
python3 baseline.py data/match.csv data/champion.json out_play nofinal

# figure
python3 make_fig.py out/results.json out_play/results.json baseline.png
```

Each `baseline.py` run takes about two minutes. Seeds: 0 to 9 for the outer CV folds, 12345 for the bootstrap, 777 and up for the label shuffles. Results are written to `results.json` and `per_champion.csv` in the output folder.

## Files

- `baseline.py`: builds per-champion stats, joins Riot's labels by numeric champion id, runs the nested CV, the largest-class floor, the shuffled-label reference and the bootstrap.
- `make_fig.py`: draws the baseline figure from the two `results.json` files.
