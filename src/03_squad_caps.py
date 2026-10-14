"""Average international caps per World Cup 2026 squad (26 players each).
Source: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads
Wikipedia states that caps do not include matches played after the tournament started,
so this variable is known BEFORE every match (no data leakage).
Run from the project root:  python src/03_squad_caps.py
Needs:  python -m pip install requests beautifulsoup4 lxml pandas
"""
import re
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"


def parse_squads(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    team, rows = None, []
    for el in soup.find_all(["h3", "table"]):
        if el.name == "h3":
            team = re.sub(r"\[.*?\]", "", el.get_text(strip=True)).strip()
        elif team and "Caps" in el.get_text():
            df = pd.read_html(StringIO(str(el)))[0]
            if "Caps" not in df.columns:
                continue
            caps = pd.to_numeric(df["Caps"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
            rows.append({"team": team, "n_players": int(caps.notna().sum()),
                         "avg_caps": round(float(caps.mean()), 2)})
    return pd.DataFrame(rows).drop_duplicates("team")


if __name__ == "__main__":
    resp = requests.get(URL, headers={"User-Agent": "Mozilla/5.0 (student project)"}, timeout=30)
    resp.raise_for_status()
    out = parse_squads(resp.text).sort_values("team")
    print(out.to_string(index=False))
    print("\nTeams found:", len(out), "(expected 48)")
    print("Squads not 26 players:", out[out.n_players != 26].team.tolist())
    out.to_csv("data/raw/team_caps.csv", index=False)
