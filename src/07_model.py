"""Linear Regression 2.1: baseline, optimisation experiments, competing algorithms, final evaluation.
Run from project root:  python src/07_model.py
Outputs: reports/model_*.csv, reports/ols_summary_*.txt, figures/model_*.png

Experiment groups
  E*  : raw variables
  C*  : same ideas after capping rest_days_diff at +-10 days (one match has -70, a high-leverage outlier)
"""
import os, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler
from sklearn.svm import SVR
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True); os.makedirs("reports", exist_ok=True)
SEED, CAP = 42, 10
df = pd.read_csv("data/processed/model_2_1_dataset.csv")
X_COLS = ["fifa_pts_diff", "market_value_diff", "goals_pg_last10_diff", "rest_days_diff",
          "confed_strength_diff", "matches_12m_diff", "squad_caps_diff", "host_diff"]
y = df["goal_diff"]
DATA = {"raw": df[X_COLS].copy(), "cap": df[X_COLS].copy()}
DATA["cap"]["rest_days_diff"] = DATA["cap"]["rest_days_diff"].clip(-CAP, CAP)   # outlier treatment
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)                # 50 train/validation splits

def vif_reduce(X, thresh=5.0):
    cols = list(X.columns)
    while len(cols) > 2:
        v = [variance_inflation_factor(add_constant(X[cols]).values, i + 1) for i in range(len(cols))]
        if max(v) <= thresh: break
        cols.pop(int(np.argmax(v)))
    return cols
low_vif = {k: vif_reduce(DATA[k]) for k in DATA}
print("Kept after VIF<=5 filter:", low_vif["raw"])

def signed_log(a): return np.sign(a) * np.log1p(np.abs(a))
def slog_pipe(): return make_pipeline(ColumnTransformer([("s", FunctionTransformer(signed_log), ["market_value_diff", "fifa_pts_diff"])],
                                                        remainder="passthrough"), StandardScaler(), LinearRegression())
def topk(k, reg=None): return make_pipeline(StandardScaler(), SelectKBest(f_regression, k=k), reg or LinearRegression())
def ridge(): return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30)))
def enet(): return make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.2, .5, .8, 1], cv=5, random_state=SEED))
def yj(): return make_pipeline(PowerTransformer(), LinearRegression())

# (name, model factory, data version, columns or None, group, is_linear)
EXPS = [
 ("E0 Baseline (predict training mean)", DummyRegressor, "raw", None, "baseline", False),
 ("E1 Linear regression, all 8 variables", LinearRegression, "raw", None, "raw", True),
 ("E2 + Yeo-Johnson transformation", yj, "raw", None, "raw", True),
 ("E3 + signed-log of skewed variables, scaling", slog_pipe, "raw", None, "raw", True),
 ("E4 Remove collinear variables (VIF<=5)", LinearRegression, "raw", low_vif["raw"], "raw", True),
 ("E5 VIF filter + Yeo-Johnson", yj, "raw", low_vif["raw"], "raw", True),
 ("E6 Top-3 by F-test (inside CV)", lambda: topk(3), "raw", None, "raw", True),
 ("E6 Top-4 by F-test (inside CV)", lambda: topk(4), "raw", None, "raw", True),
 ("E6 Top-5 by F-test (inside CV)", lambda: topk(5), "raw", None, "raw", True),
 ("E6 Top-6 by F-test (inside CV)", lambda: topk(6), "raw", None, "raw", True),
 ("E7 Ridge (alpha by CV)", ridge, "raw", None, "raw", True),
 ("E8 Elastic Net (alpha, l1 by CV)", enet, "raw", None, "raw", True),
 ("C1 All 8, rest days capped at +-10", LinearRegression, "cap", None, "capped", True),
 ("C2 Capped + Yeo-Johnson", yj, "cap", None, "capped", True),
 ("C3 Capped + signed-log + scaling", slog_pipe, "cap", None, "capped", True),
 ("C4 Capped + VIF filter (VIF<=5)", LinearRegression, "cap", low_vif["cap"], "capped", True),
 ("C5 Capped + Huber robust regression", lambda: make_pipeline(StandardScaler(), HuberRegressor()), "cap", None, "capped", True),
 ("C6 Capped + Top-3 by F-test", lambda: topk(3), "cap", None, "capped", True),
 ("C6 Capped + Top-4 by F-test", lambda: topk(4), "cap", None, "capped", True),
 ("C7 Capped + Ridge", ridge, "cap", None, "capped", True),
 ("Random Forest (capped data)", lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED), "cap", None, "competing", False),
 ("Gradient Boosting (capped data)", lambda: GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=.05, subsample=.8, random_state=SEED), "cap", None, "competing", False),
 ("SVR, RBF kernel (capped data)", lambda: make_pipeline(StandardScaler(), SVR(C=2.0, epsilon=.3)), "cap", None, "competing", False),
 ("k-Nearest Neighbours, k=10 (capped data)", lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(10)), "cap", None, "competing", False),
]

def get_X(ver, cols): return DATA[ver] if cols is None else DATA[ver][cols]
rows = []
for name, mk, ver, cols, grp, lin in EXPS:
    s = cross_validate(mk(), get_X(ver, cols), y, cv=cv, n_jobs=-1,
                       scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"})
    rows.append({"group": grp, "model": name, "CV_RMSE": -s["test_rmse"].mean(), "RMSE_sd": s["test_rmse"].std(),
                 "CV_MAE": -s["test_mae"].mean(), "CV_R2": s["test_r2"].mean()})
res = pd.DataFrame(rows); res.round(4).to_csv("reports/model_cv_results.csv", index=False)
base = res.loc[0, "CV_RMSE"]; full = res.loc[1, "CV_RMSE"]
res["vs_baseline_%"] = (1 - res.CV_RMSE / base) * 100
print("\nALL EXPERIMENTS (5-fold x 10 CV)\n", res.round(3).to_string(index=False))

# ---------- best linear model -> 80/20 hold-out vs actual WC 2026 results ----------
lin_names = [e[0] for e in EXPS if e[5]]
best = res[res.model.isin(lin_names)].sort_values("CV_RMSE").iloc[0].model
name, mk, ver, cols, grp, lin = [e for e in EXPS if e[0] == best][0]
Xs = get_X(ver, cols)
Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=SEED)
mdl = mk().fit(Xtr, ytr); pred = mdl.predict(Xte); bpred = DummyRegressor().fit(Xtr, ytr).predict(Xte)
hold = pd.DataFrame({"model": [best, "Baseline (mean)"],
  "test_RMSE": [mean_squared_error(yte, pred) ** .5, mean_squared_error(yte, bpred) ** .5],
  "test_MAE": [mean_absolute_error(yte, pred), mean_absolute_error(yte, bpred)],
  "test_R2": [r2_score(yte, pred), r2_score(yte, bpred)], "n_test_matches": len(yte)})
print("\nBest linear model:", best); print(hold.round(3).to_string(index=False))
hold.round(4).to_csv("reports/model_holdout.csv", index=False)
pd.DataFrame({"actual": yte.values, "predicted": pred}).to_csv("reports/model_holdout_predictions.csv", index=False)

# ---------- statsmodels summaries (standardised variables): raw and capped ----------
for key in ("raw", "cap"):
    Xz = (DATA[key] - DATA[key].mean()) / DATA[key].std()
    ols = sm.OLS(y, sm.add_constant(Xz)).fit()
    open(f"reports/ols_summary_{key}.txt", "w").write(str(ols.summary()))
    if key == "cap": print("\nOLS (capped, standardised)\n", ols.summary())
    if key == "cap": OLS_CAP = ols

# ---------- diagnostic figure ----------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
axes[0].scatter(yte, pred, alpha=.7); lim = [min(yte.min(), pred.min()) - .5, max(yte.max(), pred.max()) + .5]
axes[0].plot(lim, lim, "r--", label="perfect prediction"); axes[0].set_xlabel("Actual goal difference"); axes[0].set_ylabel("Predicted goal difference")
axes[0].set_title("Hold-out: predicted vs actual"); axes[0].legend()
axes[1].scatter(pred, yte - pred, alpha=.7); axes[1].axhline(0, color="r", ls="--")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual"); axes[1].set_title("Residuals vs predicted")
coef = OLS_CAP.params.drop("const").sort_values(); ci = OLS_CAP.conf_int().drop("const").loc[coef.index]
axes[2].barh(coef.index, coef.values, xerr=[coef.values - ci[0].values, ci[1].values - coef.values], color="steelblue")
axes[2].axvline(0, color="k", lw=.8); axes[2].set_title("Standardised coefficients (95% CI), capped data")
plt.tight_layout(); plt.savefig("figures/model_diagnostics.png", dpi=200); plt.close()

allres = res.sort_values("CV_RMSE"); colmap = {"baseline": "#7f8c8d", "raw": "#2980b9", "capped": "#27ae60", "competing": "#c0392b"}
fig, ax = plt.subplots(figsize=(10, 9))
ax.barh(allres.model, allres.CV_RMSE, xerr=allres.RMSE_sd, color=[colmap[g] for g in allres.group])
ax.set_xlabel("Cross-validated RMSE (lower is better)"); ax.invert_yaxis()
ax.set_title("Model comparison (grey baseline, blue raw, green capped, red competing algorithms)")
plt.tight_layout(); plt.savefig("figures/model_comparison.png", dpi=200); plt.close()
print("\nModelling finished.")
