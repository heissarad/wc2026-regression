"""Compact figures for the 4-page report. Run after 06_eda.py and 07_model.py:  python src/08_report_figures.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
df = pd.read_csv("data/processed/model_2_1_dataset.csv")
X = ["fifa_pts_diff", "market_value_diff", "goals_pg_last10_diff", "rest_days_diff",
     "confed_strength_diff", "matches_12m_diff", "squad_caps_diff", "host_diff"]

# ---- Figure 1: correlation matrix + four scatterplots (the last shows the rest-days outlier) ----
fig = plt.figure(figsize=(7.2, 3.0))
gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], wspace=0.42, hspace=0.62)
ax = fig.add_subplot(gs[:, 0])
sns.heatmap(df[X + ["goal_diff"]].corr(), annot=True, fmt=".2f", annot_kws={"size": 5.2}, cmap="vlag", center=0, vmin=-1, vmax=1,
            cbar=False, ax=ax, linewidths=.3)
ax.set_title("(a) Correlation matrix"); ax.tick_params(axis="x", rotation=60); plt.setp(ax.get_xticklabels(), ha="right")
for a, c, lab in zip([fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])],
                     ["fifa_pts_diff", "market_value_diff", "goals_pg_last10_diff", "rest_days_diff"], "bcde"):
    sns.regplot(x=df[c], y=df["goal_diff"], ax=a, scatter_kws={"alpha": .55, "s": 8}, line_kws={"color": "crimson", "lw": 1})
    a.set_title(f"({lab}) {c}, r = {df[c].corr(df['goal_diff']):.2f}"); a.set_ylabel("goal_diff"); a.set_xlabel("")
out = df.loc[df.rest_days_diff.abs() > 10]
a.scatter(out.rest_days_diff, out.goal_diff, s=22, facecolors="none", edgecolors="black", lw=.8)
a.annotate("1 match: -70 days", (out.rest_days_diff.iloc[0], out.goal_diff.iloc[0]), xytext=(-58, -12), textcoords="data", fontsize=6, arrowprops=dict(arrowstyle="->", lw=.6))
plt.savefig("figures/report_fig1.png", dpi=300, bbox_inches="tight"); plt.close()

# ---- Figure 2: model comparison (CV RMSE) + hold-out predicted vs actual ----
r = pd.read_csv("reports/model_cv_results.csv").set_index("model")
pick = {"Baseline: predict the mean": "E0 Baseline (predict training mean)",
        "Linear, all 8 variables": "E1 Linear regression, all 8 variables",
        "Linear, all 8, rest days capped": "C1 All 8, rest days capped at +-10",
        "Capped + signed-log + scaling": "C3 Capped + signed-log + scaling",
        "Linear, top 3 by F-test": "E6 Top-3 by F-test (inside CV)",
        "SVR (RBF)": "SVR, RBF kernel (capped data)",
        "Random Forest": "Random Forest (capped data)",
        "Gradient Boosting": "Gradient Boosting (capped data)",
        "k-Nearest Neighbours": "k-Nearest Neighbours, k=10 (capped data)"}
sel = pd.DataFrame({"label": list(pick), "rmse": [r.loc[v, "CV_RMSE"] for v in pick.values()], "sd": [r.loc[v, "RMSE_sd"] for v in pick.values()]})
cols = ["#7f8c8d", "#2980b9", "#27ae60", "#27ae60", "#27ae60", "#c0392b", "#c0392b", "#c0392b", "#c0392b"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.55})
ax = axes[0]; ax.barh(sel.label, sel.rmse, xerr=sel.sd, color=cols, error_kw={"lw": .7}); ax.invert_yaxis()
for i, v in enumerate(sel.rmse): ax.text(0.04, i, f"{v:.2f}", va="center", color="white", fontsize=6.5, fontweight="bold")
ax.set_xlim(0, 3.1); ax.set_xlabel("Cross-validated RMSE in goals (lower is better; bars = sd across folds)"); ax.set_title("(a) Model comparison, 5-fold x 10 CV")
h = pd.read_csv("reports/model_holdout_predictions.csv"); a = axes[1]
a.scatter(h.actual, h.predicted, s=14, alpha=.75); lim = [-4.2, 5.5]; a.plot(lim, lim, "r--", lw=.8)
a.set_xlabel("Actual goal difference"); a.set_ylabel("Predicted goal difference"); a.set_title("(b) Hold-out: 21 matches, top-3 linear model")
plt.savefig("figures/report_fig2.png", dpi=300, bbox_inches="tight"); plt.close()
print("saved figures/report_fig1.png and figures/report_fig2.png")
