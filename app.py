import streamlit as st
from ml_app.py import show_prediction_page

st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="wide"
)

# =====================================
# SIDEBAR MENU
# =====================================

st.sidebar.title("💻 Navigation")

menu = st.sidebar.radio(
    "Choose Menu",
    [
        "Home",
        "Prediction"
    ]
)

# =====================================
# HOME PAGE
# =====================================

if menu == "Home":

    st.title("💻 Laptop Price Prediction System")

    st.markdown("---")

    st.write("""
    ### Welcome

    This application predicts laptop prices using Machine Learning Regression.

    ### Features
    - Laptop Price Prediction
    - Multiple Regression Models
    - Interactive Dashboard
    - Streamlit Deployment

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
    - IPS Display
    - Operating System
    """)

# =====================================
# PREDICTION PAGE
# =====================================

elif menu == "Prediction":
    show_prediction_page()
