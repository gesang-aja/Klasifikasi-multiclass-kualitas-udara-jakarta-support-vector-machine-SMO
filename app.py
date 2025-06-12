import streamlit as st
import pandas as pd
import joblib
import numpy as np
from svm_scrath import SVM_SMO, OneVsOneSVM


# ⛳️ WAJIB paling atas sebelum Streamlit command lainnya
st.set_page_config(page_title="Prediksi Kategori ISPU", layout="centered")

# Load dict berisi model, scaler, encoder, imputer
@st.cache_resource
def load():
    return joblib.load('models/svm_ovo_smo.joblib')

artifacts = load()
model = artifacts['model']
scaler = artifacts['scaler']
encoder = artifacts['encoder']

st.title("Form Input Kualitas Udara")
# Input text
pm10 = st.text_input("PM10 (μg/m³)", placeholder="Contoh: 120.5")
so2 = st.text_input("SO2 (μg/m³)", placeholder="Contoh: 30.2")
co = st.text_input("CO (mg/m³)", placeholder="Contoh: 2.1")
o3 = st.text_input("O3 (μg/m³)", placeholder="Contoh: 90.0")
no2 = st.text_input("NO2 (μg/m³)", placeholder="Contoh: 45.3")

# Tombol submit
if st.button("Kirim"):
    try:
        # Langsung buat array input
        input_array = np.array([[float(pm10), float(so2), float(co), float(o3), float(no2)]], dtype=float)
        print(input_array)
        # Scaling
        input_scaled = scaler.transform(input_array)

        # Prediksi
        pred = model.predict(input_scaled)

        # Inverse transform label
        pred_label = encoder.inverse_transform(pred)

        st.success("Prediksi berhasil!")
        st.write("Kategori ISPU:", f"**{pred_label[0]}**")

    except ValueError:
        st.error("Mohon isi semua kolom dengan angka yang valid.")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
