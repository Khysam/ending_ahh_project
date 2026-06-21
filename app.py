import streamlit as st
import pandas as pd
import joblib

# ==========================
# CONFIG
# ==========================
st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================
model = joblib.load("model.pkl")



# ==========================
# HOME PAGE
# ==========================
if menu == "Home":

    st.title("💻 Laptop Price Prediction System")

    st.markdown("---")

    st.write("""
    ### Welcome

    This application predicts laptop prices using Machine Learning Regression.

    ### Features
    - Laptop Price Prediction
    - Automated Preprocessing
    - Machine Learning Model
    - Interactive User Interface

    ### Instructions
    1. Open **ML Prediction App** from the sidebar.
    2. Enter laptop specifications.
    3. Click **Predict Price**.
    4. View prediction results.

    ### Dataset Features
    - Company
    - TypeName
    - Inches
    - RAM
    - Weight
    - CPU Brand
    - GPU Brand
    - SSD
    - HDD
    - PPI
    - Touchscreen
    - IPS
    - Operating System
    """)

    st.info(
        "Navigate to 'ML Prediction App' from the sidebar to start predicting."
    )

