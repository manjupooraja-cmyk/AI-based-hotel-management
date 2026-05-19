import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------------------------------------
# 1. LOAD DATASET
# ----------------------------------------------------------
df = pd.read_csv("tripadvisor_hotel_reviews.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ----------------------------------------------------------
# 2. PRE‑PROCESSING
# ----------------------------------------------------------
df.columns = df.columns.str.strip().str.lower()

df = df[["review", "rating"]]
df.dropna(inplace=True)

# ----------------------------------------------------------
# 3. CREATE SENTIMENT LABEL FROM RATING
# ----------------------------------------------------------
def get_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df["Sentiment"] = df["rating"].apply(get_sentiment)

print("\nSentiment Count:\n")
print(df["Sentiment"].value_counts())

# ----------------------------------------------------------
# 4. TEXT VECTORIZATION (TF‑IDF)
# ----------------------------------------------------------
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)

X = tfidf.fit_transform(df["review"])
y = df["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================================
# 5. MODEL 1 — LOGISTIC REGRESSION
# ==========================================================
lr_model = LogisticRegression(max_iter=300)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_acc  = accuracy_score(y_test, lr_pred)

print("\nLOGISTIC REGRESSION ACCURACY:", lr_acc)
print(classification_report(y_test, lr_pred))

cm_lr = confusion_matrix(y_test, lr_pred)

plt.figure(figsize=(4,3))
sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Neg","Neu","Pos"],
    yticklabels=["Neg","Neu","Pos"]
)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================================================
# 6. MODEL 2 — NAIVE BAYES
# ==========================================================
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)
nb_acc  = accuracy_score(y_test, nb_pred)

print("\nNAIVE BAYES ACCURACY:", nb_acc)
print(classification_report(y_test, nb_pred))

cm_nb = confusion_matrix(y_test, nb_pred)

plt.figure(figsize=(4,3))
sns.heatmap(
    cm_nb,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Neg","Neu","Pos"],
    yticklabels=["Neg","Neu","Pos"]
)
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================================================
# 7. MODEL PERFORMANCE COMPARISON GRAPH
# ==========================================================
models = ["Logistic Regression", "Naive Bayes"]
accuracy = [lr_acc, nb_acc]

plt.figure(figsize=(6,4))
sns.barplot(x=models, y=accuracy, hue=models, palette="Set2", legend=False)

plt.title("Model Performance Comparison")
plt.ylabel("Accuracy")
plt.ylim(0,1)

for i, v in enumerate(accuracy):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")

plt.tight_layout()
plt.show()

# ==========================================================
# 8. FINAL SENTIMENT GRAPH
# ==========================================================
sentiment_count = df["Sentiment"].value_counts().reset_index()
sentiment_count.columns = ["Sentiment", "Count"]

plt.figure(figsize=(6,5))

sns.barplot(
    data=sentiment_count,
    x="Sentiment",
    y="Count",
    hue="Sentiment",
    palette={
        "Positive":"green",
        "Neutral":"orange",
        "Negative":"red"
    },
    legend=False
)

plt.title("Final Sentiment Analysis Graph")
plt.xlabel("Sentiment Type")
plt.ylabel("Number of Reviews")

plt.tight_layout()
plt.savefig("Final_Sentiment_Graph.png")
plt.show()

# ==========================================================
# 9. BEST MODEL SELECTION
# ==========================================================
models_dict = {
    "Logistic Regression": (lr_model, lr_acc),
    "Naive Bayes": (nb_model, nb_acc)
}

best_model_name = max(models_dict, key=lambda x: models_dict[x][1])
best_model, best_accuracy = models_dict[best_model_name]

print("\n🏆 Best Model:", best_model_name)
print("Best Accuracy:", best_accuracy)

# ==========================================================
# 10. SAVE BEST MODEL + VECTORIZER
# ==========================================================
joblib.dump(best_model, "m4_sentiment_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print(f"\n✅ Best model ({best_model_name}) saved as m4_sentiment_model.pkl")
print("✅ TF‑IDF Vectorizer saved as tfidf_vectorizer.pkl")