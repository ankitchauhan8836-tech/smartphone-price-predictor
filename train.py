"""Train, compare, tune and save the best smartphone price model (real Smartprix data)."""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from data_cleaning import load_and_clean

warnings.filterwarnings("ignore", category=FutureWarning)

TARGET = "price_inr"
CAT_OHE = ["brand", "os_name", "notch_type", "variant"]
CAT_TE = ["processor"]            # ~200 chipsets -> target-encoded inside CV (no leakage)
NUM = ["ram_gb", "storage_gb", "cpu_cores", "clock_ghz", "battery_mah", "fast_charge_w",
       "screen_inches", "pixels_mp", "refresh_rate_hz", "rear_camera_mp", "n_rear_cameras",
       "front_camera_mp", "has_5g", "has_nfc", "has_ir_blaster", "expandable_storage",
       "is_foldable", "os_version"]
FEATURES = CAT_OHE + CAT_TE + NUM

df = load_and_clean()
df.to_csv("data/smartphones_clean.csv", index=False)
print(f"Clean dataset: {df.shape[0]} phones, {len(FEATURES)} features")

X, y = df[FEATURES], df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)


def make(model):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CAT_OHE),
        ("te", TargetEncoder(target_type="continuous", random_state=42), CAT_TE),
    ])
    # log target: prices are heavily right-skewed (Rs 2k - Rs 6.5L), errors are proportional
    return TransformedTargetRegressor(Pipeline([("pre", pre), ("model", model)]),
                                      func=np.log1p, inverse_func=np.expm1)


candidates = {
    "Baseline (median)": TransformedTargetRegressor(DummyRegressor(), func=np.log1p, inverse_func=np.expm1),
    "Ridge": make(Ridge(alpha=1.0)),
    "RandomForest": make(RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    "GradientBoosting": make(GradientBoostingRegressor(random_state=42)),
}

kf = KFold(5, shuffle=True, random_state=42)
# select models on log-price R2: a Rs 1.5L flagship shouldn't dominate the score over hundreds of Rs 10-30k phones
log_r2 = make_scorer(lambda a, b: r2_score(np.log(a), np.log(b)))
results = {}
for name, est in candidates.items():
    cv = cross_val_score(est, X_tr, y_tr, cv=kf, scoring=log_r2)
    results[name] = cv.mean()
    print(f"{name:18s} CV R2(log) = {cv.mean():.4f} (+/- {cv.std():.4f})")

best_name = max((k for k in results if "Baseline" not in k), key=results.get)
print(f"\nBest baseline model: {best_name} -> tuning...")
grid = {
    "RandomForest": {"regressor__model__max_depth": [None, 15], "regressor__model__min_samples_leaf": [1, 3]},
    "GradientBoosting": {"regressor__model__n_estimators": [200, 400],
                         "regressor__model__learning_rate": [0.03, 0.05, 0.1],
                         "regressor__model__max_depth": [2, 3, 4],
                         "regressor__model__subsample": [0.8, 1.0]},
    "Ridge": {"regressor__model__alpha": [0.1, 1, 10, 30]},
}[best_name]
search = GridSearchCV(candidates[best_name], grid, cv=kf, scoring=log_r2, n_jobs=-1).fit(X_tr, y_tr)
model = search.best_estimator_
print("Best params:", search.best_params_)

pred = model.predict(X_te)
rel_err = np.abs(pred - y_te) / y_te
metrics = {
    "n_phones": int(len(df)),
    "best_model": best_name,
    "best_params": {k.split("__")[-1]: v for k, v in search.best_params_.items()},
    "cv_r2_log_price": {k: round(v, 4) for k, v in results.items()},
    "test_r2_log_price": round(r2_score(np.log(y_te), np.log(pred)), 4),
    "test_r2": round(r2_score(y_te, pred), 4),
    "test_mae_inr": round(mean_absolute_error(y_te, pred)),
    "test_median_ae_inr": round(float(np.median(np.abs(pred - y_te)))),
    "test_mape_pct": round(100 * mean_absolute_percentage_error(y_te, pred), 2),
    "within_15pct": round(100 * float((rel_err <= 0.15).mean()), 1),
}
print(json.dumps(metrics, indent=2))

# refit on all data for the deployed model
final = clone(search.best_estimator_).fit(X, y)
joblib.dump({"model": final, "features": FEATURES, "cat_ohe": CAT_OHE, "num": NUM}, "models/price_model.joblib")
json.dump(metrics, open("models/metrics.json", "w"), indent=2)

# ---- plots ----
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(y_te, pred, s=10, alpha=0.5)
lim = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
ax.plot(lim, lim, "r--"); ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Actual price (INR)"); ax.set_ylabel("Predicted price (INR)")
ax.set_title(f"Actual vs Predicted - {best_name} (log-R²={metrics['test_r2_log_price']})")
fig.tight_layout(); fig.savefig("assets/actual_vs_pred.png", dpi=130); plt.close(fig)

fitted = search.best_estimator_  # GridSearchCV refit on the training split only (before the final all-data refit)
imp = permutation_importance(fitted, X_te, y_te, n_repeats=8, random_state=42, n_jobs=-1,
                             scoring="neg_mean_absolute_percentage_error")
imp_s = pd.Series(imp.importances_mean, index=X_te.columns).sort_values()
imp_s.to_csv("models/feature_importance.csv", header=["importance"])
fig, ax = plt.subplots(figsize=(6.5, 6))
imp_s.tail(15).plot.barh(ax=ax); ax.set_title("Top features (permutation importance)")
ax.set_xlabel("Increase in error when shuffled")
fig.tight_layout(); fig.savefig("assets/feature_importance.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
df.groupby("brand").price_inr.median().sort_values().tail(15).plot.barh(ax=ax)
ax.set_title("Median price by brand (top 15)"); ax.set_xlabel("INR")
fig.tight_layout(); fig.savefig("assets/brand_prices.png", dpi=130); plt.close(fig)
print("Saved model, metrics and plots.")
