import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# Load trained model
model_path = BASE_DIR / "car_price_model.pkl"

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)

except FileNotFoundError:
    st.error("Model file not found!")
    st.warning("Please run train_model.py first.")
    st.stop()

# Application title
st.title("🚗 Car Price Prediction")

st.write(
    "### Used Car Price Prediction System"
)

st.write(
    "Enter your car details to predict its estimated selling price."
)

st.divider()

# Car details
st.subheader("Enter Car Details")

col1, col2 = st.columns(2)

with col1:

    year = st.number_input(
        "Manufacturing Year",
        min_value=2000,
        max_value=2026,
        value=2018
    )

    present_price = st.number_input(
        "Present Price (Lakhs)",
        min_value=0.1,
        max_value=100.0,
        value=5.0,
        step=0.1
    )

    kms = st.number_input(
        "Kilometers Driven",
        min_value=0,
        max_value=1000000,
        value=30000
    )

    fuel = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "CNG"]
    )

with col2:

    seller = st.selectbox(
        "Seller Type",
        ["Dealer", "Individual"]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )

    owner = st.selectbox(
        "Previous Owners",
        [0, 1, 2, 3]
    )

st.divider()

# Prediction button
if st.button("Predict Car Price", use_container_width=True):

    # Prepare input data
    input_data = pd.DataFrame([{
        'Year': year,
        'Present_Price': present_price,
        'Kms_Driven': kms,
        'Fuel_Type': fuel,
        'Seller_Type': seller,
        'Transmission': transmission,
        'Owner': owner
    }])

    # Predict price
    prediction = model.predict(input_data)[0]

    # Display result
    st.success("Car Price Predicted Successfully!")

    st.metric(
        label="Estimated Car Selling Price",
        value=f"₹ {prediction:.2f} Lakhs"
    )

    st.info(
        "This is an estimated price based on the trained "
        "Machine Learning model."
    )

st.divider()

st.caption(
    "Car Price Prediction Using Machine Learning | "
    "Developed using Python and Random Forest Regression"
)