"""Build the match table for Model 2.1 (one row per World Cup 2026 match).
Run from the project root:  python src/01_matches.py
Source: github.com/martj42/international_results  (results.csv)
"""
import pandas as pd

URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

df = pd.read_csv(URL, parse_dates=["date"])
wc = df[(df.tournament == "FIFA World Cup") & (df.date.dt.year == 2026)].copy()
wc = wc.sort_values("date", kind="stable").reset_index(drop=True)
assert len(wc) == 104, f"Expected 104 matches, found {len(wc)}"

# Team A = first-listed (home_team) team, Team B = second-listed team
m = pd.DataFrame({
    "match_id": range(1, 105),
    "date": wc.date.dt.date,
    "team_a": wc.home_team,
    "team_b": wc.away_team,
    "goals_a": wc.home_score,
    "goals_b": wc.away_score,
    "venue_country": wc.country,
})
m["goal_diff"] = m.goals_a - m.goals_b          # RESPONSE VARIABLE

# host_diff: +1 if A plays in its own country, -1 if B does, otherwise 0
m["host_diff"] = (m.team_a == m.venue_country).astype(int) - (m.team_b == m.venue_country).astype(int)

m.to_csv("data/processed/matches.csv", index=False)
print(m.head(), "\n")
print("Rows:", len(m))
print("Teams:", len(set(m.team_a) | set(m.team_b)))
print("goal_diff summary:\n", m.goal_diff.describe().round(2))
print("host_diff counts:\n", m.host_diff.value_counts())
