# ============================================
# SEASONAL PRICING PREDICTION
# INR + CONFUSION MATRICES + COMPARISON GRAPH
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix

# 1. LOAD DATASET
df = pd.read_csv("hotel_bookings.csv")
print("Dataset Shape:", df.shape)

# 2. PREPROCESSING
df = df[df["is_canceled"] == 0]
df["children"] = df["children"].fillna(0)

df["total_guests"] = df["adults"] + df["children"] + df["babies"]
df["total_stay"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]

df = df[df["total_stay"] > 0]
df = df[df["adr"] > 0]

# 3. EURO TO INR
EURO_TO_INR = 90
df["adr_inr"] = df["adr"] * EURO_TO_INR

# 4. MONTH ORDER
month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

df["arrival_date_month"] = pd.Categorical(
    df["arrival_date_month"],
    categories=month_order,
    ordered=True
).codes + 1

# 5. FEATURES AND TARGET
X = df[[
    "arrival_date_month",
    "arrival_date_week_number",
    "stays_in_week_nights",
    "adults",
    "children"
]]

y = df["adr_inr"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 6. PRICE LEVEL FUNCTION
low_th = y.quantile(0.33)
high_th = y.quantile(0.66)

def price_level(price):
    if price <= low_th:
        return "Low Price"
    elif price <= high_th:
        return "Medium Price"
    else:
        return "High Price"

def show_confusion_matrix(model_name, y_actual, y_pred):
    actual_level = y_actual.apply(price_level)
    predicted_level = pd.Series(y_pred).apply(price_level)

    cm = confusion_matrix(
        actual_level,
        predicted_level,
        labels=["Low Price", "Medium Price", "High Price"]
    )

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low Price", "Medium Price", "High Price"],
        yticklabels=["Low Price", "Medium Price", "High Price"]
    )

    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Price Level")
    plt.ylabel("Actual Price Level")
    plt.tight_layout()
    plt.show()

# 7. MODELS
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

r2_scores = {}
rmse_scores = {}
trained_models = {}

# 8. TRAIN + CONFUSION MATRIX
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5

    r2_scores[name] = r2
    rmse_scores[name] = rmse
    trained_models[name] = model

    print(f"\n{name}")
    print("R2 Score:", r2)
    print("RMSE:", rmse)

    show_confusion_matrix(name, y_test, pred)

# 9. MODEL COMPARISON GRAPH
plt.figure(figsize=(8, 5))

sns.barplot(
    x=list(r2_scores.keys()),
    y=list(r2_scores.values()),
    hue=list(r2_scores.keys()),
    palette="Set2",
    legend=False
)

plt.title("Seasonal Pricing Model Comparison")
plt.xlabel("Model")
plt.ylabel("R2 Score")
plt.ylim(0, 1)

for i, score in enumerate(r2_scores.values()):
    plt.text(i, score + 0.02, f"{score:.2f}", ha="center")

plt.tight_layout()
plt.savefig("pricing_model_comparison.png")
plt.show()

# 10. MONTH-WISE SEASONAL PRICING GRAPH
monthly_price = (
    df.groupby("arrival_date_month")["adr_inr"]
    .mean()
    .reset_index()
)

monthly_price["Month"] = monthly_price["arrival_date_month"].apply(
    lambda x: month_order[x - 1]
)

monthly_price["Price_Level"] = monthly_price["adr_inr"].apply(price_level)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=monthly_price,
    x="Month",
    y="adr_inr",
    hue="Price_Level",
    palette={
        "Low Price": "green",
        "Medium Price": "orange",
        "High Price": "red"
    }
)

plt.title("Seasonal Pricing Prediction in Indian Rupees")
plt.xlabel("Month")
plt.ylabel("Average Room Price per Night (INR)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("seasonal_pricing_graph.png")
plt.show()

# 11. BEST MODEL SAVE
best_model_name = max(r2_scores, key=r2_scores.get)
best_model = trained_models[best_model_name]

print("\n🏆 Best Model:", best_model_name)
print("Best R2 Score:", r2_scores[best_model_name])

joblib.dump(best_model, "m3_price_model.pkl")

print("✅ Best model saved as m3_price_model.pkl")
print("✅ Seasonal pricing graph saved as seasonal_pricing_graph.png")
print("✅ Model comparison graph saved as pricing_model_comparison.png")