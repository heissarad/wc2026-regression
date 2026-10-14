# HIT140 – Linear Regression 2.1: Goal Difference per Match (FIFA World Cup 2026)

Charles Darwin University · HIT140 Foundations of Data Science · Assessment 3 (Objective 2) · Darwin Group 87

| Regression | Team members | Repository |
|---|---|---|
| **2.1 Goal difference** (this repo) | Sarad Gyawali (S404821), Suprim Thapa (S399407) | https://github.com/heissarad/wc2026-regression |
| 2.2 Goals per team per match | Bijaya Subedi (S397826), Sushant Malla (S396205) | https://github.com/Cremsoline/HIT140-WorldCup2026-Goals |

## Aim
Predict the goal difference (Team A goals − Team B goals) in each of the 104 matches of the FIFA World Cup 2026, using only information available **before** each match.

## Dataset
- **104 rows**, one per match (`data/processed/model_2_1_dataset.csv`)
- **Response:** `goal_diff`. Team A is the first-listed team; the score is after extra time, excluding penalty shootouts
- **8 explanatory variables**: seven Team A − Team B differences and one host indicator

| # | Variable | Definition | Source |
|---|---|---|---|
| 1 | `fifa_pts_diff` | FIFA ranking points, 11 June 2026 release | FIFA/Coca-Cola Men's World Ranking |
| 2 | `market_value_diff` | Squad market value (€ m) | Transfermarkt (Wayback Machine copy, 13 July 2026) |
| 3 | `goals_pg_last10_diff` | Goals per game in the last 10 matches before kick-off | martj42 international results |
| 4 | `rest_days_diff` | Days since the previous match | martj42 international results |
| 5 | `confed_strength_diff` | Confederation points per match against other confederations, World Cups 1998–2022 | martj42 international results |
| 6 | `matches_12m_diff` | International matches in the 365 days before kick-off | martj42 international results |
| 7 | `squad_caps_diff` | Mean caps of the 26-player squad at the start of the tournament | Wikipedia, 2026 FIFA World Cup squads |
| 8 | `host_diff` | +1 if Team A plays at home, −1 if Team B does, otherwise 0 | Venue country (martj42) |

Form variables use only matches played strictly before each match, so there is no data leakage. Full source details, retrieval dates and reasons are in `data/SOURCES.csv`.

## Repository structure

| Path | Contents |
|---|---|
| `src/01_matches.py` | Builds the 104-match table and the response `goal_diff` |
| `src/02_form_features.py` | Goals per game (last 10), rest days and matches in the last 12 months |
| `src/03_squad_caps.py` | Average squad caps from Wikipedia |
| `src/05_confed_strength.py` | Confederation strength from World Cups 1998–2022 |
| `src/04_build_dataset.py` | Joins everything into the final 8-variable dataset |
| `src/06_eda.py` | EDA: summary statistics, correlations, scatterplots, VIF |
| `src/07_model.py` | Baseline, optimisation experiments, competing algorithms, hold-out test |
| `src/08_report_figures.py` | The two figures used in the report |
| `data/raw/` | Manually collected team values (`teams_manual.csv`) and squad caps |
| `data/processed/` | Intermediate tables and the final `model_2_1_dataset.csv` |
| `figures/` | All EDA and model figures |
| `reports/` | EDA tables, OLS summaries, cross-validation and hold-out results |

## Method
1. EDA: Pearson and Spearman correlations, scatterplots and variance inflation factors. These exposed one rest-days outlier (match 16, −70 days).
2. Baseline: predict the training mean. The model is evaluated with 5-fold cross-validation repeated 10 times (RMSE, MAE, R²) and checked on an 80/20 hold-out against the actual 2026 results.
3. Optimisation: capping rest days at ±10, signed-log transformation and scaling, Yeo-Johnson, a VIF filter, Ridge, Elastic Net, Huber regression and top-k feature selection (selected inside each fold).
4. Competing algorithms: SVR, Random Forest, Gradient Boosting and k-nearest neighbours.

## Results (cross-validated RMSE in goals; lower is better)

| Model | RMSE | R² |
|---|---|---|
| Baseline (mean) | 2.01 | −0.08 |
| Linear, all 8 variables | 2.02 | −0.37 |
| Linear, all 8, rest days capped | 1.63 | 0.28 |
| **Linear, top 3 by F-test (recommended)** | **1.56** | **0.34** |
| SVR (RBF) | 1.61 | 0.30 |
| Random Forest | 1.63 | 0.27 |
| Gradient Boosting | 1.64 | 0.27 |
| k-nearest neighbours | 1.74 | 0.19 |

Recommended model: linear regression on FIFA points, market value and recent goals per game. It is 22% better than the baseline, as accurate as the non-linear models and easy to interpret.

## How to run
```bash
pip install -r requirements.txt
pip install requests beautifulsoup4 lxml        # only needed for 03_squad_caps.py
python src/01_matches.py
python src/02_form_features.py
python src/03_squad_caps.py
python src/05_confed_strength.py
python src/04_build_dataset.py
python src/06_eda.py
python src/07_model.py
python src/08_report_figures.py
```
Run every script from the repository root. The final dataset is already included, so to reproduce only the analysis, start at `06_eda.py`.

## Team collaboration and contributions
The group worked in two pairs: Sarad Gyawali and Suprim Thapa completed Regression 2.1, and Bijaya Subedi and Sushant Malla completed Regression 2.2. Each pair worked together on its regression, covering data collection, coding, analysis and writing. To avoid Git merge conflicts, each pair pushed its work from one account, so the commit history for this repository shows mainly Sarad Gyawali. This reflects how commits were made, not who did the work: Suprim Thapa contributed fully to Regression 2.1. The two sections were then combined into one group report.
