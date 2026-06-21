import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="ML Prediction App",
    page_icon="🤖",
    layout="wide"
)

# ==================================================
# FILE PATH
# ==================================================

PREPROCESSOR_PATH = "preprocessor.pkl"
MODEL_PATH = "model.pkl"

# ==================================================
# LOAD OBJECTS
# ==================================================

@st.cache_resource
def load_objects():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessor, model

# ==================================================
# PREDICTION PAGE FUNCTION
# ==================================================

def show_prediction_page():
    # cek file
    missing_files = []
    if not os.path.exists(PREPROCESSOR_PATH):
        missing_files.append(PREPROCESSOR_PATH)
    if not os.path.exists(MODEL_PATH):
        missing_files.append(MODEL_PATH)

    if len(missing_files) > 0:
        st.error(f"Missing Files: {', '.join(missing_files)}")
        st.stop()

    preprocessor, model = load_objects()

    # judul
    st.title("🤖 Laptop Price Prediction")
    st.markdown("---")

    # form input
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            company = st.selectbox("Company", ["Apple","Dell","HP","Lenovo","Asus","Acer","MSI","Toshiba","Samsung","Huawei","Xiaomi","Microsoft","Razer","LG","Chuwi"])
            typename = st.selectbox("TypeName", ["Notebook","Ultrabook","Gaming","2 in 1 Convertible","Workstation","Netbook"])
            inches = st.number_input("Screen Size", min_value=10.0, max_value=20.0, value=15.6)
            ram = st.selectbox("RAM (GB)", [2,4,8,16,32,64])
            weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0)
            cpu_brand = st.selectbox("CPU Brand", ["Intel Core i3","Intel Core i5","Intel Core i7","Intel Core i9","AMD Ryzen 3","AMD Ryzen 5","AMD Ryzen 7","AMD Ryzen 9"])

        with col2:
            gpu_brand = st.selectbox("GPU Brand", ["Intel","AMD","Nvidia"])
            ssd = st.selectbox("SSD (GB)", [0,128,256,512,1024,2048])
            hdd = st.selectbox("HDD (GB)", [0,500,1000,2000])
            ppi = st.number_input("PPI", min_value=50, max_value=500, value=141)
            touchscreen = st.selectbox("Touchscreen", [0,1])
            ips = st.selectbox("IPS Display", [0,1])
            os_name = st.selectbox("Operating System", ["Windows","Mac","Linux","Other"])

        submit = st.form_submit_button("🚀 Predict Price")

    # prediction
    if submit:
        try:
            input_df = pd.DataFrame({
                'Company':[company],
                'TypeName':[typename],
                'Inches':[inches],
                'Ram':[ram],
                'Weight':[weight],
                'Cpu Brand':[cpu_brand],
                'GPU Brand':[gpu_brand],
                'SSD':[ssd],
                'HDD':[hdd],
                'ppi':[ppi],
                'Touchscreen':[touchscreen],
                'IPS':[ips],
                'OS':[os_name]
            })

            X_processed = preprocessor.transform(input_df)
            prediction = model.predict(X_processed)[0]
            prediction = np.expm1(prediction)

            euro_price = prediction
            idr_price = prediction * 19000

            col1, col2 = st.columns(2)
            with col1:
                st.metric("💶 Price (Euro)", f"€ {euro_price:,.2f}")
            with col2:
                st.metric("🇮🇩 Price (IDR)", f"Rp {idr_price:,.0f}")

            st.markdown("---")
            st.subheader("Input Specification")
            st.dataframe(input_df, use_container_width=True)

            st.subheader("Model Information")
            st.write(f"Model : {type(model).__name__}")

        except Exception as e:
            st.error("Prediction Failed")
            st.exception(e)

# ==================================================
# SIDEBAR MENU
# ==================================================

st.sidebar.title("💻 Navigation")
menu = st.sidebar.radio("Choose Menu", ["Home","Prediction"])

# ==================================================
# HOME PAGE
# ==================================================

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

elif menu == "Prediction":
    show_prediction_page()
