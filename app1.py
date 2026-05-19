# ==========================================================
# HOTEL ML PROJECT – STREAMLIT APP (FINAL UPDATED)
# DEMAND + PRICING + SENTIMENT
# ==========================================================

import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Hotel Prediction System",
    page_icon="🏨",
    layout="wide"
)

# ----------------------------------------------------------
# TITLE
# ----------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;color:#2c3e50;'>🏨 Hotel ML Prediction Dashboard</h1>",
    unsafe_allow_html=True
)

# ----------------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------------
demand_model = joblib.load("m2_demand_model.pkl")
price_model = joblib.load("m3_price_model.pkl")
sentiment_model = joblib.load("m4_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
menu = st.sidebar.radio(
    "Select Module",
    [
        "Room Demand Prediction",
        "Seasonal Pricing Prediction",
        "Sentiment Analysis"
    ]
)

# ==========================================================
# 1️⃣ ROOM DEMAND
# ==========================================================
if menu == "Room Demand Prediction":

    st.subheader("📊 Room Demand Prediction")

    month = st.selectbox(
        "Month",
        [
            "January","February","March","April","May","June",
            "July","August","September","October","November","December"
        ]
    )

    guests = st.number_input("Guests", 1, 10, 2)
    stay_nights = st.slider("Stay Nights", 1, 14, 2)

    room_type = st.selectbox(
        "Room Type",
        ["Standard","Deluxe","Premium"]
    )

    lead_time = st.slider("Lead Time", 0, 365, 30)

    month_num = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ].index(month) + 1

    room_map = {
        "Standard":1,
        "Deluxe":2,
        "Premium":3
    }

    room_num = room_map[room_type]

    input_data = np.array([[
        month_num,
        guests,
        stay_nights,
        room_num,
        lead_time
    ]])

    if st.button("Predict Demand"):

        demand_level = int(
            demand_model.predict(input_data)[0]
        )

        base_rooms = guests * stay_nights

        if demand_level == 0:
            rooms = int(base_rooms * 2 + 20)
            label = "Low Demand"
            color = "red"

        elif demand_level == 1:
            rooms = int(base_rooms * 3 + 40)
            label = "Medium Demand"
            color = "orange"

        else:
            rooms = int(base_rooms * 5 + 80)
            label = "High Demand"
            color = "green"

        st.markdown(
            f"""
            <div style="
            background:white;
            padding:15px;
            border-radius:10px;
            font-size:22px;
            font-weight:bold;
            color:{color};">
            Demand Level: {label} <br>
            Predicted Rooms Needed: {rooms}
            </div>
            """,
            unsafe_allow_html=True
        )

# ==========================================================
# 2️⃣ PRICING
# ==========================================================
elif menu == "Seasonal Pricing Prediction":

    st.subheader("💰 Seasonal Pricing")

    month = st.selectbox("Month", [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ])

    guests = st.number_input("Guests",1,10,2)
    stay_nights = st.slider("Stay Nights",1,14,2)

    room_type = st.selectbox(
        "Room Type",
        ["Standard","Deluxe","Premium"]
    )

    lead_time = st.slider("Lead Time",0,365,30)

    month_num = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ].index(month) + 1

    room_map = {"Standard":1,"Deluxe":2,"Premium":3}
    room_num = room_map[room_type]

    input_data = np.array([[
        month_num,
        stay_nights,
        guests,
        lead_time,
        room_num
    ]])

    if st.button("Predict Price"):

        price = price_model.predict(input_data)[0]

        # If model trained in EURO → convert
        euro_to_inr = 90
        price_inr = price * euro_to_inr

        st.success(
            f"Predicted Room Price: ₹ {round(price_inr,2)}"
        )

# ==========================================================
# 3️⃣ SENTIMENT
# ==========================================================
else:

    st.subheader("🧠 Sentiment Analysis")

    df = pd.read_csv("tripadvisor_hotel_reviews.csv")

    def label(r):
        if r >= 4:
            return "Positive"
        elif r == 3:
            return "Neutral"
        else:
            return "Negative"

    df["Sentiment"] = df["Rating"].apply(label)

    review = st.text_area("Enter Review")

    if st.button("Analyze"):

        vec = vectorizer.transform([review])
        pred = sentiment_model.predict(vec)[0]

        if pred == "Positive":
            st.success("Positive 😊")
        elif pred == "Neutral":
            st.warning("Neutral 😐")
        else:
            st.error("Negative 😠")

    st.markdown("---")

    option = st.radio(
        "Filter Dataset",
        ["All","Positive","Neutral","Negative"]
    )

    if option == "All":
        data = df
    else:
        data = df[df["Sentiment"] == option]

    st.write("Review Count:", data.shape[0])
    st.dataframe(data[["Review","Sentiment"]].head(50))