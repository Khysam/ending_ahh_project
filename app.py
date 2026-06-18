import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻"
)

@st.cache_resource
def load_objects():

    preprocessor = joblib.load(
        "preprocessor.pkl"
    )

    model = joblib.load(
        "model.pkl"
    )

    return preprocessor, model

preprocessor, model = load_objects()

st.title(
    "💻 Laptop Price Prediction"
)

company = st.selectbox(
    "Company",
    [
        "Apple","Dell","HP",
        "Lenovo","Asus","Acer"
    ]
)

typename = st.selectbox(
    "TypeName",
    [
        "Notebook",
        "Ultrabook",
        "Gaming"
    ]
)

inches = st.number_input(
    "Screen Size",
    value=15.6
)

ram = st.selectbox(
    "RAM",
    [4,8,16,32,64]
)

weight = st.number_input(
    "Weight",
    value=2.0
)

cpu_brand = st.text_input(
    "CPU Brand"
)

gpu_brand = st.selectbox(
    "GPU Brand",
    ["Intel","AMD","Nvidia"]
)

ssd = st.number_input(
    "SSD"
)

hdd = st.number_input(
    "HDD"
)

ppi = st.number_input(
    "PPI",
    value=141
)

touchscreen = st.selectbox(
    "Touchscreen",
    [0,1]
)

ips = st.selectbox(
    "IPS",
    [0,1]
)

os_name = st.selectbox(
    "OS",
    [
        "Windows",
        "Mac",
        "Linux",
        "Other"
    ]
)

if st.button("Predict"):

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

    X_processed = preprocessor.transform(
        input_df
    )

    prediction = model.predict(
        X_processed
    )[0]

    prediction = np.expm1(
        prediction
    )

    st.success(
        f"Estimated Price: € {prediction:,.2f}"
    )
