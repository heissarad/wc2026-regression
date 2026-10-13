"""Compute 3 pre-match variables for every WC 2026 match from the full international results file:
  goals_pg_last10_diff, rest_days_diff, matches_12m_diff   (all = Team A - Team B)
Only matches played BEFORE each match's date are used (no data leakage).
Run from the project root:  python src/02_form_features.py
"""
import pandas as pd

URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
all_games = pd.read_csv(URL, parse_dates=["date"])
matches = pd.read_csv("data/processed/matches.csv", parse_dates=["date"])

# Long format: one row per team per game
home = all_games.rename(columns={"home_team": "team", "home_score": "gf", "away_score": "ga"})[["date", "team", "gf", "ga"]]
away = all_games.rename(columns={"away_team": "team", "away_score": "gf", "home_score": "ga"})[["date", "team", "gf", "ga"]]
long = pd.concat([home, away]).sort_values("date", kind="stable").reset_index(drop=True)

def team_features(team, date):
    past = long[(long.team == team) & (long.date < date)]       # strictly before the match
    last10 = past.tail(10)
    return pd.Series({
        "goals_pg_last10": last10.gf.mean(),
        "rest_days": (date - past.date.iloc[-1]).days,
        "matches_12m": (past.date >= date - pd.Timedelta(days=365)).sum(),
        "n_past": len(last10),
    })

rows = []
for _, m in matches.iterrows():
    a = team_features(m.team_a, m.date)
    b = team_features(m.team_b, m.date)
    rows.append({
        "match_id": m.match_id,
        "goals_pg_last10_diff": a.goals_pg_last10 - b.goals_pg_last10,
        "rest_days_diff": a.rest_days - b.rest_days,
        "matches_12m_diff": a.matches_12m - b.matches_12m,
        "min_games_found": min(a.n_past, b.n_past),
    })
out = pd.DataFrame(rows)
assert out.min_games_found.min() == 10, "some team has fewer than 10 past games"
out.drop(columns="min_games_found").to_csv("data/processed/form_features.csv", index=False)
print(out.describe().round(2).T[["mean", "std", "min", "max"]])
print("Missing values:", int(out.isna().sum().sum()))
