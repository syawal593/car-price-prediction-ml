import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="BMW Price Predictor Pro",
    page_icon="🚗",
    layout="wide"
)

@st.cache_resource 
def load_assets():
    model = joblib.load('model_bmw.pkl')
    scaler = joblib.load('scaler_bmw.pkl')
    model_columns = joblib.load('columns_bmw.pkl')
    return model, scaler, model_columns

model, scaler, model_columns = load_assets()

st.markdown("""
    <style>
    /* Mengatur latar belakang halaman utama */
    .stApp {
        background-color: #f5f7f9;
    }

    /* Mengatur gaya tombol */
    .stButton>button {
        width: 100%;
        background-color: #0066b1;
        color: white;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #004a80;
        color: white;
    }

    /* Mengatur kotak hasil estimasi (Metric) */
    [data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }

    /* Memaksa warna teks Label dan Nilai agar berwarna gelap (Terlihat) */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #1f1f1f !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/44/BMW.svg", width=80)
st.sidebar.header("🛠️ Konfigurasi Kendaraan")

model_name = st.sidebar.selectbox("Pilih Model BMW", sorted(['5 Series', '6 Series', '1 Series', '7 Series', '2 Series', '4 Series', '3 Series', 'X5', 'X3', 'X4', 'i3', 'X2', 'Z4', 'X1', 'M4', 'X6', 'X7', 'i8', 'M2', 'M3', 'M5', 'M6', 'Z3']))
year = st.sidebar.slider("Tahun Perakitan", 2000, 2020, 2018)
transmission = st.sidebar.radio("Sistem Transmisi", ['Manual', 'Semi-Auto', 'Automatic'], horizontal=True)
fuel_type = st.sidebar.selectbox("Jenis Bahan Bakar", ['Diesel', 'Petrol', 'Hybrid', 'Other'])

st.title("🚗 BMW AI Price Estimator")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    mileage = st.number_input("Jarak Tempuh (Mileage)", 0, 200000, 30000, help="Total jarak yang sudah ditempuh dalam mil")
with col2:
    engine_size = st.number_input("Kapasitas Mesin (Litre)", 0.0, 6.6, 2.0, step=0.1)
with col3:
    tax = st.number_input("Pajak Tahunan (£)", 0, 500, 145)

col4, col5 = st.columns(2)
with col4:
    mpg = st.slider("Efisiensi Bahan Bakar (MPG)", 10.0, 100.0, 55.0)

if st.button("💰 HITUNG ESTIMASI HARGA"):
    year_old = 2020 - year
    mileage_log = np.log1p(mileage)
    
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0
    input_data['tax'] = tax
    input_data['mpg'] = mpg
    input_data['engineSize'] = engine_size
    input_data['year_old'] = year_old
    input_data['mileage_log'] = mileage_log
    
    for cat in ['model_' + model_name, 'transmission_' + transmission, 'fuelType_' + fuel_type]:
        if cat in input_data.columns:
            input_data[cat] = 1

    input_sc = scaler.transform(input_data)
    pred_log = model.predict(input_sc)
    prediction = np.expm1(pred_log)[0]

    st.markdown("### 📊 Hasil Estimasi")
    m1, m2, m3 = st.columns(3)
    
    m1.metric("Harga Estimasi", f"£ {prediction:,.2f}")
    m2.metric("Model Terpilih", f"BMW {model_name}")
    m3.metric("Kondisi", "Premium" if year > 2015 else "Standard")
    
    st.info(f"Berdasarkan data pasar BMW Inggris, unit {model_name} tahun {year} dengan mesin {engine_size}L memiliki nilai jual yang kompetitif di angka tersebut.")

st.markdown("---")
st.caption("© 2024 BMW Analytics Dashboard - Dibuat dengan XGBoost Predictive Model")
