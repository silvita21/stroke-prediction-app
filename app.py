import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image

# --------------------------------------------
# 🧠 Konfigurasi Halaman
# --------------------------------------------
st.set_page_config(page_title="Prediksi Penyakit Stroke", layout="centered")
st.title("🧬 Prediksi Penyakit Stroke")
st.write("""
Aplikasi ini dibuat untuk memprediksi kemungkinan seseorang **terkena penyakit stroke** 
berdasarkan beberapa faktor risiko medis.  
Dikembangkan oleh **Silvita** untuk submission project DQLab 2025.
""")

# --------------------------------------------
# 📦 Load Model
# --------------------------------------------
model_path = "model_best_randomforest_stroke.pkl"

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    st.success("✅ Model berhasil dimuat!")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
# --------------------------------------------
# ⚙️ Threshold Sensitivitas
# --------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Pengaturan Threshold")
threshold = st.sidebar.slider(
    "Atur Sensitivitas Model (Threshold)",
    min_value=0.0,
    max_value=1.0,
    value=0.49,  # default hasil Youden’s Index
    step=0.01
)

st.sidebar.info(
    "🔹 Threshold rendah (mis. 0.3): lebih sensitif, lebih banyak deteksi stroke\n"
    "🔹 Threshold tinggi (mis. 0.7): lebih spesifik, tapi bisa lewatkan kasus positif"
)

# --------------------------------------------

# --------------------------------------------
# 🧾 Sidebar Input Data
# --------------------------------------------
st.sidebar.header("🩺 Input Data Pasien")

age = st.sidebar.number_input("Umur", min_value=1, max_value=120, value=45)
heart_disease = st.sidebar.selectbox("Penyakit Jantung", ["Tidak", "Ya"])
hypertension = st.sidebar.selectbox("Hipertensi", ["Tidak", "Ya"])
avg_glucose_level = st.sidebar.number_input("Rata-rata Glukosa (mg/dL)", min_value=50.0, max_value=300.0, value=100.0)

# Konversi input ke numerik
heart_disease = 1 if heart_disease == "Ya" else 0
hypertension = 1 if hypertension == "Ya" else 0

# Buat DataFrame dari input user
input_data = pd.DataFrame({
    "age": [age],
    "heart_disease": [heart_disease],
    "avg_glucose_level": [avg_glucose_level],
    "hypertension": [hypertension]
})

# 🔍 Prediksi
# --------------------------------------------
if st.sidebar.button("Prediksi Sekarang"):
    with st.spinner("🔎 Model sedang menganalisis..."):
        time.sleep(2)

        # Prediksi probabilitas stroke
        proba = model.predict_proba(input_data)[0][1]

        # Tentukan hasil berdasarkan threshold
        pred = 1 if proba >= threshold else 0

        st.subheader("📊 Hasil Prediksi")
        st.write(f"**Probabilitas terkena stroke:** `{proba:.2f}`")

        if pred == 1:
            st.error("⚠️ Berdasarkan model, Anda **berisiko terkena stroke.**")
            img = Image.open("stroke.jpg") if "stroke.jpg" else None
            if img:
                st.image(img, caption="Ilustrasi Stroke", use_container_width=True)
        else:
            st.success("✅ Berdasarkan model, Anda **tidak menunjukkan indikasi stroke.**")
            st.balloons()

# --------------------------------------------
# 🧾 Penjelasan Tambahan
# --------------------------------------------
st.markdown("---")
st.markdown("""
**ℹ️ Catatan Penting:**  
Model ini dibuat berdasarkan data historis pasien.  
Hasil prediksi **tidak menggantikan diagnosis dokter.**  
Gunakan sebagai alat bantu untuk memahami risiko awal.
""")
