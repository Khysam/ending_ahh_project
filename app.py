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
# SIDEBAR MENU
# ==========================
menu = st.sidebar.radio(
    "📌 Navigation",
    ["Home", "ML Prediction App"]
)

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

# ==========================
# PREDICTION PAGE
# ==========================
elif menu == "ML Prediction App":

    st.title("💻 Laptop Price Prediction")

    st.markdown("---")

    company = st.selectbox(
        "Company",
        ["Dell", "HP", "Lenovo", "Apple", "Asus", "Acer", "MSI", "Toshiba"]
    )

    typename = st.selectbox(
        "Type Name",
        ["Notebook", "Ultrabook", "Gaming", "Workstation", "Netbook"]
    )

    inches = st.number_input(
        "Screen Size (Inches)",
        min_value=10.0,
        max_value=20.0,
        value=15.6
    )

    ram = st.number_input(
        "RAM (GB)",
        min_value=2,
        max_value=64,
        value=8
    )

    weight = st.number_input(
        "Weight (Kg)",
        min_value=0.5,
        max_value=5.0,
        value=2.0
    )

    cpu_brand = st.selectbox(
        "CPU Brand",
        ["Intel Core i3", "Intel Core i5", "Intel Core i7", "AMD", "Other"]
    )

    gpu_brand = st.selectbox(
        "GPU Brand",
        ["Intel", "Nvidia", "AMD"]
    )

    ssd = st.number_input(
        "SSD Capacity (GB)",
        min_value=0,
        max_value=2048,
        value=256
    )

    hdd = st.number_input(
        "HDD Capacity (GB)",
        min_value=0,
        max_value=4000,
        value=0
    )

    ppi = st.number_input(
        "PPI",
        min_value=50.0,
        max_value=500.0,
        value=141.0
    )

    touchscreen = st.selectbox(
        "Touchscreen",
        [0, 1]
    )

    ips = st.selectbox(
        "IPS Display",
        [0, 1]
    )

    os = st.selectbox(
        "Operating System",
        ["Windows", "Mac", "Linux", "No OS", "Other"]
    )

    if st.button("🔮 Predict Price"):

        input_data = pd.DataFrame({
            'Company': [company],
            'TypeName': [typename],
            'Inches': [inches],
            'Ram': [ram],
            'Weight': [weight],
            'Cpu brand': [cpu_brand],
            'Gpu brand': [gpu_brand],
            'SSD': [ssd],
            'HDD': [hdd],
            'PPI': [ppi],
            'Touchscreen': [touchscreen],
            'Ips': [ips],
            'os': [os]
        })

        prediction = model.predict(input_data)

        st.success(
            f"💰 Predicted Laptop Price: Rp {prediction[0]:,.0f}"
        )

        st.balloons()
