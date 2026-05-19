# ==========================================================
# DEMAND‑BASED ROOM PREDICTION
# UPDATED WITH ML COMPARISON + GRAPHS
# LOGIC NOT CHANGED
# ==========================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# ----------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------
df = pd.read_csv("hotel_bookings.csv")

# ----------------------------------------------------------
# PREPROCESSING (UNCHANGED)
# ----------------------------------------------------------
df["children"] = df["children"].fillna(0)

df["total_guests"] = (
    df["adults"] +
    df["children"] +
    df["babies"]
)

df["total_stay"] = (
    df["stays_in_week_nights"] +
    df["stays_in_weekend_nights"]
)

df = df[df["total_stay"] > 0]

# ----------------------------------------------------------
# MONTH ENCODING (UNCHANGED)
# ----------------------------------------------------------
month_order = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

df["arrival_date_month"] = pd.Categorical(
    df["arrival_date_month"],
    categories=month_order,
    ordered=True
).codes + 1

# ----------------------------------------------------------
# ROOM TYPE PROXY (UNCHANGED)
# ----------------------------------------------------------
def room_type_proxy(x):
    if x < 80:
        return 1
    elif x < 150:
        return 2
    else:
        return 3

df["room_num"] = df["adr"].apply(room_type_proxy)

# ----------------------------------------------------------
# DEMAND LEVEL CREATION (UNCHANGED)
# ----------------------------------------------------------
monthly = (
    df.groupby("arrival_date_month")["total_guests"]
    .sum()
)

low  = monthly.quantile(0.33)
high = monthly.quantile(0.66)

def demand_label(x):
    if x <= low:
        return 0
    elif x <= high:
        return 1
    else:
        return 2

df["Demand_Level"] = df["arrival_date_month"].map(
    monthly.apply(demand_label)
)

# ----------------------------------------------------------
# FEATURES (UNCHANGED)
# ----------------------------------------------------------
X = df[[
    "arrival_date_month",
    "total_guests",
    "total_stay",
    "room_num",
    "lead_time"
]]

y = df["Demand_Level"]

# ----------------------------------------------------------
# TRAIN TEST SPLIT (UNCHANGED)
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# MODELS TRAINING
# ==========================================================

# 1️⃣ Random Forest (YOUR ORIGINAL MODEL)
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
rf_model.fit(X_train, y_train)

# 2️⃣ Decision Tree
dt_model = DecisionTreeClassifier(
    random_state=42
)
dt_model.fit(X_train, y_train)

# 3️⃣ Logistic Regression
lr_model = LogisticRegression(
    max_iter=1000
)
lr_model.fit(X_train, y_train)

# ----------------------------------------------------------
# PREDICTIONS
# ----------------------------------------------------------
rf_pred = rf_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# ----------------------------------------------------------
# ACCURACY SCORES
# ----------------------------------------------------------
rf_acc = accuracy_score(y_test, rf_pred)
dt_acc = accuracy_score(y_test, dt_pred)
lr_acc = accuracy_score(y_test, lr_pred)

print("Random Forest Accuracy :", rf_acc)
print("Decision Tree Accuracy :", dt_acc)
print("Logistic Regression Accuracy :", lr_acc)

# ==========================================================
# CONFUSION MATRICES
# ==========================================================

models_preds = [
    ("Decision Tree", dt_pred),
    ("Random Forest", rf_pred),
    ("Logistic Regression", lr_pred)
]

for name, pred in models_preds:

    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low","Medium","High"],
        yticklabels=["Low","Medium","High"]
    )

    plt.title(f"{name} – Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ==========================================================
# PERFORMANCE COMPARISON BAR CHART
# ==========================================================

model_names = [
    "Decision Tree",
    "Random Forest",
    "Logistic Regression"
]

accuracies = [
    dt_acc,
    rf_acc,
    lr_acc
]

plt.figure(figsize=(8,5))

bars = plt.bar(
    model_names,
    accuracies,
    color=["orange","green","blue"]
)

plt.title("Model Performance Comparison")
plt.ylabel("Accuracy")
plt.ylim(0,1)

for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + 0.25,
        yval + 0.01,
        round(yval,3)
    )

plt.show()

# ==========================================================
# MONTH vs ROOMS GRAPH
# (Guests + Stay proxy as you requested)
# ==========================================================

df["rooms_proxy"] = (
    df["total_guests"] *
    df["total_stay"]
)

monthly_rooms = (
    df.groupby("arrival_date_month")["rooms_proxy"]
    .sum()
)

plt.figure(figsize=(10,5))

plt.plot(
    monthly_rooms.index,
    monthly_rooms.values,
    marker="o",
    linewidth=3
)

plt.title("Month vs Number of Rooms Demand")
plt.xlabel("Month")
plt.ylabel("Total Rooms (Proxy)")
plt.grid(True)
plt.show()

# ==========================================================
# SAVE ORIGINAL MODEL (UNCHANGED)
# ==========================================================
joblib.dump(rf_model, "m2_demand_model.pkl")

print("✅ Model saved as m2_demand_model.pkl")