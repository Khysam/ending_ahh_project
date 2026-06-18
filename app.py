import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ==========================================
# CONFIG
# ==========================================

st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("best_laptop_price_model.pkl")

model = load_model()

# ==========================================
# TITLE
# ==========================================

st.title("💻 Laptop Price Prediction System")
st.markdown("---")

st.write("""
Predict laptop prices using Machine Learning Regression Models.
Enter laptop specifications in the sidebar and click **Predict Price**.
""")

# ==========================================
# SIDEBAR INPUT
# ==========================================

st.sidebar.header("Laptop Specification")

company = st.sidebar.selectbox(
    "Company",
    [
        "Apple","Dell","HP","Lenovo",
        "Asus","Acer","MSI","Toshiba",
        "Samsung","Huawei","Xiaomi",
        "Microsoft","Razer","LG","Chuwi"
    ]
)

typename = st.sidebar.selectbox(
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

inches = st.sidebar.slider(
    "Screen Size (Inches)",
    10.0,
    20.0,
    15.6
)

ram = st.sidebar.selectbox(
    "RAM (GB)",
    [2,4,8,16,32,64]
)

weight = st.sidebar.slider(
    "Weight (kg)",
    0.8,
    5.0,
    2.0
)

cpu_brand = st.sidebar.selectbox(
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

gpu_brand = st.sidebar.selectbox(
    "GPU Brand",
    [
        "Intel",
        "Nvidia",
        "AMD"
    ]
)

ssd = st.sidebar.selectbox(
    "SSD Capacity (GB)",
    [0,128,256,512,1024,2048]
)

hdd = st.sidebar.selectbox(
    "HDD Capacity (GB)",
    [0,500,1000,2000]
)

ppi = st.sidebar.slider(
    "PPI",
    90,
    400,
    141
)

touchscreen = st.sidebar.selectbox(
    "Touchscreen",
    [0,1]
)

ips = st.sidebar.selectbox(
    "IPS Display",
    [0,1]
)

os = st.sidebar.selectbox(
    "Operating System",
    [
        "Windows",
        "Mac",
        "Linux",
        "Other"
    ]
)

# ==========================================
# CREATE INPUT DATAFRAME
# ==========================================

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
    'OS':[os]
})

# ==========================================
# MAIN CONTENT
# ==========================================

col1, col2 = st.columns([1,1])

with col1:

    st.subheader("📋 Laptop Specification")

    st.dataframe(
        input_df,
        use_container_width=True
    )

with col2:

    st.subheader("⚙️ Configuration Summary")

    st.write(f"**Company:** {company}")
    st.write(f"**Type:** {typename}")
    st.write(f"**RAM:** {ram} GB")
    st.write(f"**CPU:** {cpu_brand}")
    st.write(f"**GPU:** {gpu_brand}")
    st.write(f"**SSD:** {ssd} GB")
    st.write(f"**HDD:** {hdd} GB")

st.markdown("---")

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("🚀 Predict Laptop Price"):

    progress = st.progress(0)

    for i in range(100):
        progress.progress(i+1)

    try:

        pred_log = model.predict(input_df)[0]

        predicted_euro = np.expm1(pred_log)

        kurs_eur_idr = 19000

        predicted_idr = predicted_euro * kurs_eur_idr

        st.success("Prediction Completed!")

        # ==============================
        # METRICS
        # ==============================

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "💶 Estimated Price (Euro)",
                f"€ {predicted_euro:,.2f}"
            )

        with c2:
            st.metric(
                "🇮🇩 Estimated Price (IDR)",
                f"Rp {predicted_idr:,.0f}"
            )

        st.markdown("---")

        # ==============================
        # GAUGE CHART
        # ==============================

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_euro,
            title={'text': "Predicted Price (€)"},
            gauge={
                'axis': {'range': [0, 5000]}
            }
        ))

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ==============================
        # CATEGORY
        # ==============================

        st.subheader("🏷️ Price Category")

        if predicted_euro < 500:
            st.info("Budget Laptop")

        elif predicted_euro < 1000:
            st.success("Mid Range Laptop")

        elif predicted_euro < 2000:
            st.warning("Premium Laptop")

        else:
            st.error("High-End Workstation / Gaming Laptop")

    except Exception as e:

        st.error("Prediction Error")
        st.exception(e)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Machine Learning Laptop Price Prediction | Streamlit Deployment"
)