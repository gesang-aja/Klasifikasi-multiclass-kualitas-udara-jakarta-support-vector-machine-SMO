import streamlit as st
import pandas as pd
import joblib
import numpy as np
from svm_scrath import SVM_SMO, OneVsOneSVM

st.set_page_config(page_title="Prediksi Kategori ISPU", layout="centered", page_icon="🌫️")

# Load model & scaler
@st.cache_resource
def load():
    return joblib.load(r'models/svm_ovo_smo.joblib')

artifacts = load()
model = artifacts['model']
scaler = artifacts['scaler']
encoder = artifacts['encoder']

# --- HEADER
st.markdown("""
    <h1 style='text-align: center;'>🌫️ Prediksi Kategori ISPU</h1>
    <p style='text-align: center; font-size: 16px; color: gray;'>
        Masukkan nilai parameter kualitas udara untuk memprediksi kategori ISPU.
    </p>
    <hr>
""", unsafe_allow_html=True)

# --- FORM INPUT
with st.form("form_ispu", clear_on_submit=False):
    st.subheader("📥 Input Data Polutan")
    st.markdown("Masukkan konsentrasi tiap polutan dalam satuan standar:")

    pm10 = st.text_input("• PM10 (μg/m³)", placeholder="Contoh: 120.5")
    so2 = st.text_input("• SO2 (μg/m³)", placeholder="Contoh: 30.2")
    co = st.text_input("• CO (mg/m³)", placeholder="Contoh: 2.1")
    o3 = st.text_input("• O3 (μg/m³)", placeholder="Contoh: 90.0")
    no2 = st.text_input("• NO2 (μg/m³)", placeholder="Contoh: 45.3")

    submitted = st.form_submit_button("🔍 Prediksi Sekarang")

# --- HASIL PREDIKSI
if submitted:
    try:
        inputs = [pm10, so2, co, o3, no2]
        if any(x.strip() == "" for x in inputs):
            st.warning("⚠️ Mohon isi semua kolom sebelum submit.")
        else:
            input_array = np.array([[float(pm10), float(so2), float(co), float(o3), float(no2)]], dtype=float)
            input_scaled = scaler.transform(input_array)

            pred = model.predict(input_scaled)
            pred_label = encoder.inverse_transform(pred)[0]

            st.success("✅ Prediksi berhasil!")
            st.markdown("### Hasil Prediksi:")
            st.markdown(f"**Kategori ISPU:** {pred_label}")

    except ValueError:
        st.error("🚫 Pastikan semua kolom diisi dengan angka yang valid (gunakan titik untuk desimal).")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: `{e}`")
