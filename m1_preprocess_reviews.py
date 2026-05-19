import pandas as pd
import re

# ===============================
# STEP 1: Load Review Dataset
# ===============================
df = pd.read_csv("tripadvisor_hotel_reviews.csv")

print("Review dataset loaded")
print(df.shape)
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

# ===============================
# STEP 2: Rename Columns (if needed)
# Kaggle dataset usually has: Review, Rating
# ===============================
df = df.rename(columns={
    'Review': 'review_text',
    'Rating': 'rating'
})

# Keep only required columns
reviews_df = df[['review_text', 'rating']].copy()

# ===============================
# STEP 3: Handle Missing Values
# ===============================
reviews_df['review_text'] = reviews_df['review_text'].fillna("")
reviews_df['rating'] = reviews_df['rating'].fillna(reviews_df['rating'].median())

print("\nMissing values handled")

# ===============================
# STEP 4: Clean Review Text
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)   # remove numbers & symbols
    text = re.sub(r'\s+', ' ', text)       # remove extra spaces
    return text

reviews_df['clean_review'] = reviews_df['review_text'].apply(clean_text)

print("\nText cleaning done")

# ===============================
# STEP 5: Create Sentiment Label
# (For ML Classification)
# ===============================
# Rating >= 4 → Positive
# Rating <= 2 → Negative
# Rating == 3 → Neutral

def label_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating <= 2:
        return "Negative"
    else:
        return "Neutral"

reviews_df['sentiment'] = reviews_df['rating'].apply(label_sentiment)

print("\nSentiment labels created")

# ===============================
# STEP 6: Final Dataset Check
# ===============================
print("\nFinal preprocessed review dataset:")
print(reviews_df.head())
print(reviews_df['sentiment'].value_counts())

# ===============================
# STEP 7: Save Preprocessed Dataset
# ===============================
reviews_df.to_csv("reviews_preprocessed.csv", index=False)
print("\nPreprocessed review dataset saved as reviews_preprocessed.csv")