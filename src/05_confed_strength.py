"""Confederation strength = points per match against OTHER confederations
in FIFA World Cups 1998-2022. Source: github.com/martj42/international_results
Run from the project root:  python src/05_confed_strength.py
"""
import pandas as pd

URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

UEFA = ["Austria","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia","Czech Republic",
        "Denmark","England","France","Germany","Greece","Iceland","Italy","Netherlands",
        "Norway","Poland","Portugal","Republic of Ireland","Romania","Russia","Scotland",
        "Serbia","Slovakia","Slovenia","Spain","Sweden","Switzerland","Turkey","Ukraine","Wales"]
CAF = ["Algeria","Angola","Cameroon","Egypt","Ghana","Ivory Coast","Morocco","Nigeria",
       "Senegal","South Africa","Togo","Tunisia"]
AFC = ["Australia","China","Iran","Japan","North Korea","Qatar","Saudi Arabia","South Korea"]
CONCACAF = ["Canada","Costa Rica","Honduras","Jamaica","Mexico","Panama",
            "Trinidad and Tobago","United States"]
CONMEBOL = ["Argentina","Brazil","Chile","Colombia","Ecuador","Paraguay","Peru","Uruguay"]
OFC = ["New Zealand"]   # Australia was an AFC member at every World Cup it played in 1998-2022
CONFED = {t: c for c, ts in [("UEFA",UEFA),("CAF",CAF),("AFC",AFC),("CONCACAF",CONCACAF),
          ("CONMEBOL",CONMEBOL),("OFC",OFC)] for t in ts}

df = pd.read_csv(URL, parse_dates=["date"])
wc = df[(df.tournament == "FIFA World Cup") & df.date.dt.year.between(1998, 2022)].copy()
unmapped = (set(wc.home_team) | set(wc.away_team)) - set(CONFED)
assert not unmapped, f"Unmapped teams: {unmapped}"

wc["conf_h"] = wc.home_team.map(CONFED)
wc["conf_a"] = wc.away_team.map(CONFED)
inter = wc[wc.conf_h != wc.conf_a]

rows = []
for _, r in inter.iterrows():
    hp = 3 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 0)
    ap = 3 if r.away_score > r.home_score else (1 if r.home_score == r.away_score else 0)
    rows += [(r.conf_h, hp), (r.conf_a, ap)]
long = pd.DataFrame(rows, columns=["confederation", "points"])
out = long.groupby("confederation").points.agg(matches="count", ppm="mean").round(3)
print(out)
out.to_csv("data/processed/confed_strength.csv")
