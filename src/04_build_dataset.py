"""Build the final Model 2.1 dataset: 104 rows, 8 explanatory variables (Team A - Team B) + goal_diff.
Run from the project root AFTER scripts 01, 02, 03 and 05 and after teams_manual.csv is complete:
    python src/04_build_dataset.py
"""
import pandas as pd

matches = pd.read_csv("data/processed/matches.csv")
form = pd.read_csv("data/processed/form_features.csv")
confed = pd.read_csv("data/processed/confed_strength.csv").rename(columns={"ppm": "confed_ppm"})
teams = pd.read_csv("data/raw/teams_manual.csv", encoding="utf-8-sig")
caps = pd.read_csv("data/raw/team_caps.csv")[["team", "avg_caps"]].rename(columns={"avg_caps": "caps"})

# Team-level lookup table (48 rows): FIFA points, market value, caps, confederation strength
teams = teams[["team", "confederation", "fifa_points", "market_value_eur_m"]]
teams = teams.merge(caps, on="team", how="left")
teams = teams.merge(confed[["confederation", "confed_ppm"]], on="confederation", how="left")
missing = teams[teams.isna().any(axis=1)]
assert missing.empty, f"Fill these team values first:\n{missing}"

def side(letter):
    t = teams.add_suffix(f"_{letter}").rename(columns={f"team_{letter}": f"team_{letter}"})
    return t

df = matches.merge(side("a"), on="team_a", how="left").merge(side("b"), on="team_b", how="left")
df = df.merge(form, on="match_id", how="left")

out = pd.DataFrame({
    "match_id": df.match_id,
    "fifa_pts_diff": df.fifa_points_a - df.fifa_points_b,
    "market_value_diff": df.market_value_eur_m_a - df.market_value_eur_m_b,
    "goals_pg_last10_diff": df.goals_pg_last10_diff,
    "rest_days_diff": df.rest_days_diff,
    "confed_strength_diff": df.confed_ppm_a - df.confed_ppm_b,
    "matches_12m_diff": df.matches_12m_diff,
    "squad_caps_diff": df.caps_a - df.caps_b,
    "host_diff": df.host_diff,
    "goal_diff": df.goal_diff,
})
assert len(out) == 104 and out.isna().sum().sum() == 0, "Dataset has missing values or wrong size"
out.to_csv("data/processed/model_2_1_dataset.csv", index=False)
print(out.describe().round(2).T[["mean", "std", "min", "max"]])
print("\nSaved data/processed/model_2_1_dataset.csv with", len(out), "rows and", out.shape[1] - 2, "explanatory variables")
