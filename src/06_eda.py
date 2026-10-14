"""Exploratory data analysis for Model 2.1. Run from project root:  python src/06_eda.py
Outputs: figures/eda_*.png and reports/eda_*.csv
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

os.makedirs("figures", exist_ok=True); os.makedirs("reports", exist_ok=True)
df = pd.read_csv("data/processed/model_2_1_dataset.csv")
X_COLS = ["fifa_pts_diff", "market_value_diff", "goals_pg_last10_diff", "rest_days_diff",
          "confed_strength_diff", "matches_12m_diff", "squad_caps_diff", "host_diff"]
y = "goal_diff"
sns.set_theme(style="whitegrid", context="notebook")

# 1. Summary statistics + skewness
summary = df[X_COLS + [y]].describe().T
summary["skew"] = df[X_COLS + [y]].skew()
print("SUMMARY\n", summary.round(2)); summary.round(3).to_csv("reports/eda_summary.csv")

# 2. Correlation with the response (Pearson r and p-value) and between predictors
rows = []
for c in X_COLS:
    r, p = stats.pearsonr(df[c], df[y]); rho, _ = stats.spearmanr(df[c], df[y])
    rows.append({"variable": c, "pearson_r": r, "p_value": p, "spearman_rho": rho})
corr_y = pd.DataFrame(rows).sort_values("pearson_r", key=abs, ascending=False)
print("\nCORRELATION WITH goal_diff\n", corr_y.round(3)); corr_y.round(4).to_csv("reports/eda_corr_with_target.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(df[X_COLS + [y]].corr(), annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Pearson r"})
ax.set_title("Correlation matrix: explanatory variables and goal difference")
plt.xticks(rotation=40, ha="right"); plt.tight_layout(); plt.savefig("figures/eda_corr_heatmap.png", dpi=200); plt.close()

# 3. Scatterplots of each explanatory variable vs the response (with fitted line)
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, c in zip(axes.ravel(), X_COLS):
    sns.regplot(x=df[c], y=df[y], ax=ax, scatter_kws={"alpha": .6, "s": 22}, line_kws={"color": "crimson"})
    ax.set_title(f"{c}  (r = {df[c].corr(df[y]):.2f})", fontsize=10); ax.set_ylabel("goal_diff")
plt.tight_layout(); plt.savefig("figures/eda_scatter_grid.png", dpi=200); plt.close()

# 4. Distributions (check normality / skew -> need for transformation)
fig, axes = plt.subplots(3, 3, figsize=(13, 9))
for ax, c in zip(axes.ravel(), X_COLS + [y]):
    sns.histplot(df[c], kde=True, ax=ax, bins=20); ax.set_title(f"{c} (skew {df[c].skew():.2f})", fontsize=10)
plt.tight_layout(); plt.savefig("figures/eda_distributions.png", dpi=200); plt.close()

# 5. Multicollinearity: Variance Inflation Factor (VIF > 5 is a warning, > 10 is serious)
Xc = add_constant(df[X_COLS])
vif = pd.DataFrame({"variable": X_COLS, "VIF": [variance_inflation_factor(Xc.values, i + 1) for i in range(len(X_COLS))]})
vif = vif.sort_values("VIF", ascending=False)
print("\nVIF\n", vif.round(2)); vif.round(3).to_csv("reports/eda_vif.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(data=vif, y="variable", x="VIF", color="steelblue", ax=ax)
ax.axvline(5, color="orange", ls="--", label="VIF = 5"); ax.axvline(10, color="red", ls="--", label="VIF = 10")
ax.set_title("Multicollinearity check (VIF)"); ax.legend(); plt.tight_layout(); plt.savefig("figures/eda_vif.png", dpi=200); plt.close()

# 6. Outliers: boxplots (IQR rule) - e.g. rest_days_diff
out_counts = {}
for c in X_COLS:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    out_counts[c] = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
print("\nOUTLIERS (IQR rule)\n", out_counts)
print("\nEDA finished. Figures saved in figures/")
