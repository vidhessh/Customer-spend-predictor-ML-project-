import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import numpy as np
import pickle

data = pd.read_csv("ecom.csv")

le = LabelEncoder()
data["Gender"]           = le.fit_transform(data["Gender"])
data["Membership Type"]  = le.fit_transform(data["Membership Type"])
data["Discount Applied"] = le.fit_transform(data["Discount Applied"])

X = data[[
    "Age", "Gender", "Membership Type",
    "Items Purchased", "Average Rating",
    "Discount Applied", "Days Since Last Purchase"
]]
Y = data["Total Spend"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ── Train models ─────────────────────────────
rf_model  = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
ridge_model = Ridge(alpha=1.0)

rf_model.fit(X_train, Y_train)
xgb_model.fit(X_train, Y_train)
ridge_model.fit(X_train, Y_train)

# ── KMeans clustering ─────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train_scaled)

# ── Save models ───────────────────────────────
pickle.dump(rf_model,    open("rf_model.pkl",    "wb"))
pickle.dump(xgb_model,   open("xgb_model.pkl",   "wb"))
pickle.dump(ridge_model, open("ridge_model.pkl",  "wb"))
pickle.dump(scaler,      open("scaler.pkl",       "wb"))
pickle.dump(kmeans,      open("kmeans.pkl",       "wb"))

# ── Evaluate ──────────────────────────────────
models = {
    "Random Forest"   : rf_model,
    "XGBoost"         : xgb_model,
    "Ridge Regression": ridge_model,
}

print("\n" + "=" * 52)
print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R² %':>8}")
print("=" * 52)

best_name, best_r2 = "", -999

for name, m in models.items():
    preds = m.predict(X_test)
    mae   = mean_absolute_error(Y_test, preds)
    rmse  = np.sqrt(mean_squared_error(Y_test, preds))
    r2    = r2_score(Y_test, preds) * 100          # as percentage
    flag  = ""
    if r2 > best_r2:
        best_r2, best_name = r2, name
    print(f"{name:<20} ${mae:>7.2f} ${rmse:>7.2f} {r2:>7.2f}%")

print("=" * 52)
print(f"Best model : {best_name}  (R² = {best_r2:.2f}%)")
print("=" * 52)
print("\nWhat the metrics mean:")
print("  MAE  — average dollar error per prediction (lower = better)")
print("  RMSE — penalises large errors more (lower = better)")
print("  R²   — % of spend variation explained by the model (higher = better)")
print("\nAll models saved successfully.")