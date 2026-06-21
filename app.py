import streamlit as st
from ml_app import show_prediction_page

st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="wide"
)

# Sidebar
st.sidebar.title("💻 Navigation")
menu = st.sidebar.radio("Choose Menu", ["Home", "Prediction"])

# Home Page
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
    """)

# Prediction Page
elif menu == "Prediction":
    show_prediction_page()
