import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==================================================
# IMPORT MODEL LIBRARIES
# ==================================================

# WAJIB untuk model pickle
try:
    from xgboost import XGBRegressor
except:
    pass

try:
    from lightgbm import LGBMRegressor
except:
    pass

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="wide"
)

# ==================================================
# SIDEBAR INFO
# ==================================================

st.sidebar.title("System Information")

# ==================================================
# CHECK FILE EXIST
# ==================================================

MODEL_PATH = "best_laptop_price_model.pkl"

if not os.path.exists(MODEL_PATH):

    st.error(
        f"""
        Model file not found
        
        Expected:
        {MODEL_PATH}
        """
    )

    st.stop()

# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():

    try:

        model = joblib.load(MODEL_PATH)

        return model

    except Exception as e:

        st.error("Failed loading model")

        st.exception(e)

        return None

model = load_model()

if model is None:
    st.stop()

# ==================================================
# HEADER
# ==================================================

st.title("💻 Laptop Price Prediction")

st.markdown("---")

st.write(
    """
    Predict laptop prices using Machine Learning.
    """
)

# ==================================================
# INPUT FORM
# ==================================================

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        company = st.selectbox(
            "Company",
            [
                "Apple",
                "Dell",
                "HP",
                "Lenovo",
                "Asus",
                "Acer",
                "MSI",
                "Toshiba",
                "Samsung",
                "Huawei",
                "Xiaomi",
                "Microsoft",
                "Razer",
                "LG",
                "Chuwi"
            ]
        )

        typename = st.selectbox(
            "Laptop Type",
            [
                "Notebook",
                "Ultrabook",
                "Gaming",
                "2 in 1 Convertible",
                "Workstation",
                "Netbook"
            ]
        )

        inches = st.number_input(
            "Screen Size",
            min_value=10.0,
            max_value=20.0,
            value=15.6
        )

        ram = st.selectbox(
            "RAM (GB)",
            [2,4,8,16,32,64]
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=0.5,
            max_value=5.0,
            value=2.0
        )

        cpu_brand = st.selectbox(
            "CPU Brand",
            [
                "Intel Core i3",
                "Intel Core i5",
                "Intel Core i7",
                "Intel Core i9",
                "AMD Ryzen 3",
                "AMD Ryzen 5",
                "AMD Ryzen 7",
                "AMD Ryzen 9"
            ]
        )

    with col2:

        gpu_brand = st.selectbox(
            "GPU Brand",
            [
                "Intel",
                "AMD",
                "Nvidia"
            ]
        )

        ssd = st.selectbox(
            "SSD (GB)",
            [0,128,256,512,1024,2048]
        )

        hdd = st.selectbox(
            "HDD (GB)",
            [0,500,1000,2000]
        )

        ppi = st.number_input(
            "PPI",
            min_value=50,
            max_value=500,
            value=141
        )

        touchscreen = st.selectbox(
            "Touchscreen",
            [0,1]
        )

        ips = st.selectbox(
            "IPS Display",
            [0,1]
        )

        os_name = st.selectbox(
            "Operating System",
            [
                "Windows",
                "Mac",
                "Linux",
                "Other"
            ]
        )

    submit = st.form_submit_button(
        "Predict Price"
    )

# ==================================================
# PREDICTION
# ==================================================

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

        prediction = model.predict(input_df)[0]

        try:
            prediction = np.expm1(prediction)
        except:
            pass

        st.success("Prediction Success")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Estimated Price (€)",
                f"{prediction:,.2f}"
            )

        with col2:

            st.metric(
                "Estimated Price (IDR)",
                f"Rp {prediction*19000:,.0f}"
            )

        st.subheader("Input Data")

        st.dataframe(input_df)

    except Exception as e:

        st.error("Prediction Failed")

        st.exception(e)
