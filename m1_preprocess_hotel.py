import pandas as pd
import numpy as np

# ===============================
# STEP 1: Load Dataset
# ===============================
df = pd.read_csv("hotel_bookings.csv")
 
print("Hotel dataset loaded")
print(df.shape)
print(df.columns)

# ===============================
# STEP 2: Select Required Columns
# (useful for demand + pricing)
# ===============================
hotel_df = df[[
    'hotel',
    'is_canceled',
    'lead_time',
    'arrival_date_year',
    'arrival_date_month',
    'arrival_date_week_number',
    'arrival_date_day_of_month',
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'adults',
    'children',
    'babies',
    'meal',
    'country',
    'market_segment',
    'distribution_channel',
    'is_repeated_guest',
    'previous_cancellations',
    'previous_bookings_not_canceled',
    'reserved_room_type',
    'assigned_room_type',
    'booking_changes',
    'deposit_type',
    'customer_type',
    'adr',
    'required_car_parking_spaces',
    'total_of_special_requests'
]].copy()

print("\nSelected columns:")
print(hotel_df.head())

# ===============================
# STEP 3: Handle Missing Values
# ===============================
hotel_df['children'] = hotel_df['children'].fillna(0)
hotel_df['babies'] = hotel_df['babies'].fillna(0)
hotel_df['country'] = hotel_df['country'].fillna("Unknown")
hotel_df['adr'] = hotel_df['adr'].fillna(hotel_df['adr'].median())

print("\nMissing values handled")

# ===============================
# STEP 4: Feature Engineering
# ===============================
# Total nights stayed
hotel_df['total_nights'] = (
    hotel_df['stays_in_weekend_nights'] +
    hotel_df['stays_in_week_nights']
)

# ===============================
# STEP 5: Encode Categorical Columns
# ===============================
categorical_cols = [
    'hotel',
    'arrival_date_month',
    'meal',
    'market_segment',
    'distribution_channel',
    'reserved_room_type',
    'assigned_room_type',
    'deposit_type',
    'customer_type',
    'country'
]

for col in categorical_cols:
    hotel_df[col] = pd.factorize(hotel_df[col])[0]

print("\nCategorical encoding done")

# ===============================
# STEP 6: Final Dataset Check
# ===============================
print("\nFinal preprocessed dataset:")
print(hotel_df.head())
print(hotel_df.shape)

# ===============================
# STEP 7: Save Preprocessed Data
# ===============================
hotel_df.to_csv("hotel_booking_preprocessed.csv", index=False)
print("\nPreprocessed hotel dataset saved as hotel_booking_preprocessed.csv")