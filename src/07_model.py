"""Linear Regression 2.1: baseline, optimisation experiments, competing algorithms, final evaluation.
Run from project root:  python src/07_model.py
Outputs: reports/model_*.csv, figures/model_*.png
"""
import os, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler
from sklearn.svm import SVR
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True); os.makedirs("reports", exist_ok=True)
SEED = 42
df = pd.read_csv("data/processed/model_2_1_dataset.csv")
X_COLS = ["fifa_pts_diff", "market_value_diff", "goals_pg_last10_diff", "rest_days_diff",
          "confed_strength_diff", "matches_12m_diff", "squad_caps_diff", "host_diff"]
X, y = df[X_COLS], df["goal_diff"]
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)   # 50 train/validation splits

def evaluate(name, model, cols=None, group="experiment"):
    Xs = X if cols is None else X[cols]
    s = cross_validate(model, Xs, y, cv=cv, n_jobs=-1,
                       scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"})
    return {"group": group, "model": name, "CV_RMSE": -s["test_rmse"].mean(), "RMSE_sd": s["test_rmse"].std(),
            "CV_MAE": -s["test_mae"].mean(), "CV_R2": s["test_r2"].mean()}

# ---------- helper: iterative VIF removal (uses explanatory variables only, no target) ----------
def vif_reduce(cols, thresh=5.0):
    cols = list(cols)
    while len(cols) > 2:
        Xc = add_constant(X[cols]); v = [variance_inflation_factor(Xc.values, i + 1) for i in range(len(cols))]
        if max(v) <= thresh: break
        cols.pop(int(np.argmax(v)))
    return cols
low_vif_cols = vif_reduce(X_COLS)
print("Columns kept after VIF filtering (VIF<=5):", low_vif_cols)

def signed_log(a): return np.sign(a) * np.log1p(np.abs(a))
skewed = [c for c in ["market_value_diff", "fifa_pts_diff"] if c in X_COLS]
slog = ColumnTransformer([("slog", FunctionTransformer(signed_log), skewed)], remainder="passthrough")

# ---------- 1. Baseline + linear regression optimisation experiments ----------
res = [
    evaluate("E0 Baseline (predict training mean)", DummyRegressor(strategy="mean")),
    evaluate("E1 Linear regression, all 8 variables", LinearRegression()),
    evaluate("E2 + Yeo-Johnson transformation", make_pipeline(PowerTransformer(), LinearRegression())),
    evaluate("E3 + signed-log of skewed variables, then scaling", make_pipeline(slog, StandardScaler(), LinearRegression())),
    evaluate("E4 Remove collinear variables (VIF<=5)", LinearRegression(), cols=low_vif_cols),
    evaluate("E5 VIF filter + Yeo-Johnson", make_pipeline(PowerTransformer(), LinearRegression()), cols=low_vif_cols),
]
for k in (3, 4, 5, 6):
    res.append(evaluate(f"E6 Top-{k} variables by F-test (inside CV)",
                        make_pipeline(StandardScaler(), SelectKBest(f_regression, k=k), LinearRegression())))
res.append(evaluate("E7 Ridge regression (alpha by CV)", make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30)))))
res.append(evaluate("E8 Elastic Net (alpha, l1 by CV)", make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.2, .5, .8, 1], cv=5, random_state=SEED))))
exp = pd.DataFrame(res); print("\nOPTIMISATION EXPERIMENTS (5-fold x10 CV)\n", exp.round(3).to_string(index=False))

# ---------- 2. Competing algorithms (HD criterion) ----------
comp = [
    evaluate("Linear regression (baseline linear model)", LinearRegression(), group="competing"),
    evaluate("Random Forest", RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED), group="competing"),
    evaluate("Gradient Boosting", GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=.05, subsample=.8, random_state=SEED), group="competing"),
    evaluate("SVR (RBF kernel)", make_pipeline(StandardScaler(), SVR(C=2.0, epsilon=.3)), group="competing"),
    evaluate("k-Nearest Neighbours (k=10)", make_pipeline(StandardScaler(), KNeighborsRegressor(10)), group="competing"),
]
compdf = pd.DataFrame(comp); print("\nCOMPETING ALGORITHMS\n", compdf.round(3).to_string(index=False))
pd.concat([exp, compdf]).round(4).to_csv("reports/model_cv_results.csv", index=False)

# ---------- 3. Pick best linear model and evaluate on an 80/20 hold-out vs actual results ----------
lin = exp[exp.model.str.startswith(("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"))].sort_values("CV_RMSE")
best_name = lin.iloc[0].model; print("\nBest linear experiment:", best_name)
best_models = {
    "E1": (LinearRegression(), None), "E2": (make_pipeline(PowerTransformer(), LinearRegression()), None),
    "E3": (make_pipeline(slog, StandardScaler(), LinearRegression()), None),
    "E4": (LinearRegression(), low_vif_cols), "E5": (make_pipeline(PowerTransformer(), LinearRegression()), low_vif_cols),
    "E7": (make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30))), None),
    "E8": (make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.2, .5, .8, 1], cv=5, random_state=SEED)), None),
}
key = best_name.split()[0]
if key == "E6":
    k = int(best_name.split("Top-")[1].split()[0])
    mdl, cols = make_pipeline(StandardScaler(), SelectKBest(f_regression, k=k), LinearRegression()), None
else:
    mdl, cols = best_models[key]
Xs = X if cols is None else X[cols]
Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=SEED)
mdl.fit(Xtr, ytr); pred = mdl.predict(Xte)
base = DummyRegressor().fit(Xtr, ytr).predict(Xte)
hold = pd.DataFrame({"model": [best_name, "Baseline (mean)"],
    "test_RMSE": [mean_squared_error(yte, pred) ** .5, mean_squared_error(yte, base) ** .5],
    "test_MAE": [mean_absolute_error(yte, pred), mean_absolute_error(yte, base)],
    "test_R2": [r2_score(yte, pred), r2_score(yte, base)]})
print("\nHOLD-OUT (20% of matches, compared with actual WC 2026 results)\n", hold.round(3).to_string(index=False))
hold.round(4).to_csv("reports/model_holdout.csv", index=False)

# ---------- 4. statsmodels summary of the full linear model (coefficients, p-values, diagnostics) ----------
Xz = (X - X.mean()) / X.std()
ols = sm.OLS(y, sm.add_constant(Xz)).fit()
print("\nOLS SUMMARY (standardised explanatory variables)\n", ols.summary())
open("reports/ols_summary.txt", "w").write(str(ols.summary()))

# ---------- 5. Figures ----------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
axes[0].scatter(yte, pred, alpha=.7); lim = [min(yte.min(), pred.min()) - .5, max(yte.max(), pred.max()) + .5]
axes[0].plot(lim, lim, "r--", label="perfect prediction"); axes[0].set_xlabel("Actual goal difference"); axes[0].set_ylabel("Predicted goal difference")
axes[0].set_title("Hold-out: predicted vs actual"); axes[0].legend()
res_ = yte - pred; axes[1].scatter(pred, res_, alpha=.7); axes[1].axhline(0, color="r", ls="--")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual"); axes[1].set_title("Residuals vs predicted")
coef = ols.params.drop("const").sort_values(); ci = ols.conf_int().drop("const").loc[coef.index]
axes[2].barh(coef.index, coef.values, xerr=[coef.values - ci[0].values, ci[1].values - coef.values], color="steelblue")
axes[2].axvline(0, color="k", lw=.8); axes[2].set_title("Standardised coefficients (95% CI)")
plt.tight_layout(); plt.savefig("figures/model_diagnostics.png", dpi=200); plt.close()

allres = pd.concat([exp, compdf]).sort_values("CV_RMSE")
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#c0392b" if g == "competing" else "#2980b9" for g in allres.group]
ax.barh(allres.model, allres.CV_RMSE, xerr=allres.RMSE_sd, color=colors)
ax.set_xlabel("Cross-validated RMSE (lower is better)"); ax.invert_yaxis()
ax.set_title("Model comparison (blue = linear experiments, red = competing algorithms)")
plt.tight_layout(); plt.savefig("figures/model_comparison.png", dpi=200); plt.close()
print("\nModelling finished. See reports/ and figures/")
