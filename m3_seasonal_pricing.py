# ============================================
# SEASONAL PRICING PREDICTION (MONTH-WISE)
# WITH MODEL COMPARISON + CONFUSION MATRICES
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------
df = pd.read_csv("hotel_bookings.csv")
print("Dataset Shape:", df.shape)

# --------------------------------------------
# 2. PRE‑PROCESSING (UNCHANGED)
# --------------------------------------------
df = df[df["is_canceled"] == 0]

df["children"] = df["children"].fillna(0)

df["total_guests"] = (
    df["adults"] + df["children"] + df["babies"]
)

df["total_stay"] = (
    df["stays_in_weekend_nights"] +
    df["stays_in_week_nights"]
)

df = df[df["total_stay"] > 0]

# --------------------------------------------
# 3. MONTH‑WISE AVERAGE PRICE
# --------------------------------------------
monthly_price = (
    df.groupby("arrival_date_month")["adr"]
    .mean()
    .reset_index()
)

month_order = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

monthly_price["arrival_date_month"] = pd.Categorical(
    monthly_price["arrival_date_month"],
    categories=month_order,
    ordered=True
)

monthly_price = monthly_price.sort_values("arrival_date_month")

# --------------------------------------------
# 4. PRICE LEVELS
# --------------------------------------------
low_th = monthly_price["adr"].quantile(0.33)
high_th = monthly_price["adr"].quantile(0.66)

def price_level(x):
    if x <= low_th:
        return "Low Price"
    elif x <= high_th:
        return "Medium Price"
    else:
        return "High Price"

monthly_price["Price_Level"] = monthly_price["adr"].apply(price_level)

print("\nMonthly Seasonal Pricing Table:")
print(monthly_price)

# --------------------------------------------
# 5. VISUALIZATION (UNCHANGED)
# --------------------------------------------
plt.figure(figsize=(12,6))

sns.barplot(
    data=monthly_price,
    x="arrival_date_month",
    y="adr",
    hue="Price_Level",
    palette={
        "Low Price": "green",
        "Medium Price": "orange",
        "High Price": "red"
    }
)

plt.title("Seasonal Pricing Prediction")
plt.xlabel("Month")
plt.ylabel("Average Room Price")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("seasonal_pricing_graph.png")
plt.show()

# --------------------------------------------
# 6. MODEL BUILDING (SAME FEATURES)
# --------------------------------------------
df["arrival_date_month"] = pd.Categorical(
    df["arrival_date_month"],
    categories=month_order,
    ordered=True
).codes

X = df[[
    "arrival_date_month",
    "arrival_date_week_number",
    "stays_in_week_nights",
    "adults",
    "children"
]]

y = df["adr"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# MODEL 1 — LINEAR REGRESSION
# ==========================================================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_r2  = r2_score(y_test, lr_pred)
lr_rmse = mean_squared_error(y_test, lr_pred) ** 0.5

print("\nLinear Regression R2:", lr_r2)

# Residual Plot (instead of confusion matrix for regression)
plt.figure(figsize=(5,4))
sns.scatterplot(x=y_test, y=lr_pred)
plt.title("Linear Regression Prediction Plot")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

# ==========================================================
# MODEL 2 — DECISION TREE
# ==========================================================
dt_model = DecisionTreeRegressor(max_depth=5)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_r2  = r2_score(y_test, dt_pred)
dt_rmse = mean_squared_error(y_test, dt_pred) ** 0.5

print("\nDecision Tree R2:", dt_r2)

plt.figure(figsize=(5,4))
sns.scatterplot(x=y_test, y=dt_pred, color="green")
plt.title("Decision Tree Prediction Plot")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

# ==========================================================
# MODEL 3 — RANDOM FOREST
# ==========================================================
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_r2  = r2_score(y_test, rf_pred)
rf_rmse = mean_squared_error(y_test, rf_pred) ** 0.5

print("\nRandom Forest R2:", rf_r2)

plt.figure(figsize=(5,4))
sns.scatterplot(x=y_test, y=rf_pred, color="orange")
plt.title("Random Forest Prediction Plot")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

# ==========================================================
# MODEL PERFORMANCE COMPARISON
# ==========================================================
models = [
    "Linear Regression",
    "Decision Tree",
    "Random Forest"
]

r2_scores = [
    lr_r2,
    dt_r2,
    rf_r2
]

plt.figure(figsize=(7,4))
sns.barplot(x=models, y=r2_scores, palette="Set2")

plt.title("Seasonal Pricing Model Comparison")
plt.ylabel("R2 Score")
plt.ylim(0,1)

for i, v in enumerate(r2_scores):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")

plt.show()


# ==========================================================
# FIND BEST MODEL AUTOMATICALLY
# ==========================================================
model_dict = {
    "Linear Regression": (lr_model, lr_r2),
    "Decision Tree": (dt_model, dt_r2),
    "Random Forest": (rf_model, rf_r2)
}

best_model_name = max(model_dict, key=lambda x: model_dict[x][1])
best_model, best_score = model_dict[best_model_name]

print("\n🏆 Best Model:", best_model_name)
print("Best R2 Score:", best_score)


# ==========================================================
# SAVE BEST MODEL AS PKL
# ==========================================================
import joblib

joblib.dump(best_model, "m3_price_model.pkl")

print(f"✅ Best model ({best_model_name}) saved as m3_price_model.pkl")