import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# ===============================
# Fungsi Preprocessing Input (disamakan dengan training)
# ===============================
def preprocess_input(df):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['hour'] = df['time'].dt.hour
    df['dayofyear'] = df['time'].dt.dayofyear
    df['time_numeric'] = (df['time'] - df['time'].min()).dt.total_seconds()

    # Buat kolom lag dummy (karena data masa depan tidak punya NO2)
    df['lag_1'] = 0
    df['lag_2'] = 0
    df['lag_3'] = 0

    return df[['time_numeric', 'hour', 'dayofyear', 'lag_1', 'lag_2', 'lag_3']]

# ===============================
# Load model dan scaler
# ===============================
model = joblib.load('model/knn_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# ===============================
# Konfigurasi Halaman
# ===============================
st.set_page_config(
    page_title="Prediksi Kadar NO₂ - KNN",
    page_icon="🌫️",
    layout="wide"
)

# Judul utama
st.markdown("<h1 style='text-align: center;'>🌫️ Prediksi Kadar NO₂ Menggunakan KNN</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Aplikasi ini memprediksi kadar NO₂ berdasarkan waktu pengamatan dan menampilkan evaluasi model.</p>", unsafe_allow_html=True)

st.divider()

# ===============================
# Bagian 1 — Prediksi ke Depan
# ===============================
st.subheader("📅 Prediksi Kadar NO₂ 7 Hari ke Depan")

with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        start_date = st.date_input(
            "Pilih Tanggal Awal Prediksi",
            pd.Timestamp.now().date()
        )
    with col2:
        pred_days = st.slider("Rentang Prediksi (hari ke depan)", 1, 7, 7)

    st.markdown("<div style='text-align: center; margin-top: 15px;'>", unsafe_allow_html=True)
    predict_button = st.button("🔍 Prediksi 7 Hari ke Depan", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_button:
        try:
            # Membuat rentang waktu dari tanggal awal ke beberapa hari ke depan
            time_range = pd.date_range(start=start_date, periods=pred_days, freq='D')

            # Membuat DataFrame untuk diproses
            df_future = pd.DataFrame({'time': time_range})

            # Ubah waktu ke fitur numerik sesuai preprocessing
            df_future_processed = preprocess_input(df_future)

            # Normalisasi data
            df_future_scaled = scaler.transform(df_future_processed)

            # Prediksi kadar NO₂
            predictions = model.predict(df_future_scaled)

            # Tampilkan hasil dalam tabel
            st.markdown("### 📊 Hasil Prediksi")
            results_df = pd.DataFrame({
                'Tanggal': time_range,
                'Perkiraan NO₂ (µg/m³)': predictions
            })
            st.dataframe(results_df.style.format({'Perkiraan NO₂ (µg/m³)': '{:.2f}'}), use_container_width=True)

            # Grafik prediksi ke depan
            st.markdown("### 📈 Grafik Prediksi 7 Hari ke Depan")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(results_df['Tanggal'], results_df['Perkiraan NO₂ (µg/m³)'], marker='o', linestyle='-', color='teal')
            ax.set_title(f"Prediksi Kadar NO₂ untuk {pred_days} Hari ke Depan")
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Kadar NO₂ (µg/m³)")
            ax.grid(True)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

st.divider()

# ===============================
# Bagian 2 — Evaluasi Model
# ===============================
st.subheader("📊 Evaluasi Model KNN")

with st.container():
    col_data, col_eval = st.columns([2, 3])

    with col_data:
        data = pd.read_csv('data/NO2_bangkalan.csv')
        data['time'] = pd.to_datetime(data['time'], errors='coerce')
        data['NO2'] = pd.to_numeric(data['NO2'], errors='coerce')
        data = data.dropna(subset=['time', 'NO2'])

        # Buat fitur waktu dan lag seperti saat training
        data['hour'] = data['time'].dt.hour
        data['dayofyear'] = data['time'].dt.dayofyear
        data['time_numeric'] = (data['time'] - data['time'].min()).dt.total_seconds()

        # Tambahkan lag (sama dengan training)
        for lag in range(1, 4):
            data[f'lag_{lag}'] = data['NO2'].shift(lag)

        # Hapus baris dengan NaN akibat lag
        data = data.dropna()

        # Siapkan fitur dan target
        X = data[['time_numeric', 'hour', 'dayofyear', 'lag_1', 'lag_2', 'lag_3']]
        y = data['NO2']

        # Normalisasi dengan scaler
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        # Hilangkan nilai NaN
        mask = (~pd.isna(y)) & (~pd.isna(y_pred))
        y, y_pred = y[mask], y_pred[mask]

        # Hitung metrik
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred) * 100

    with col_eval:
        st.markdown("### 📈 Hasil Evaluasi Model")
        col1, col2, col3 = st.columns(3)
        col1.metric("📉 MSE", f"{mse:.2f}")
        col2.metric("📈 R²", f"{r2:.3f}")
        col3.metric("📊 MAPE (%)", f"{mape:.2f}%")

st.markdown("---")

# Grafik aktual vs prediksi
st.subheader("📈 Grafik Prediksi vs Aktual")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data['time'], y, label='Aktual', linewidth=2)
ax.plot(data['time'], y_pred, label='Prediksi (KNN)', linestyle='--')
ax.set_xlabel("Waktu")
ax.set_ylabel("Kadar NO₂ (µg/m³)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.divider()
st.caption("Oleh NURIL ANJAR ROYANI — KNN Regressor untuk Prediksi Kadar NO₂")
