import pandas as pd
import joblib

def preprocess_input(df):
    """
    Mengubah kolom 'time' menjadi fitur numerik (detik sejak awal dataset),
    serta menambahkan fitur waktu tambahan agar sesuai dengan model KNN.
    """
    # Load scaler dan dataset referensi
    scaler = joblib.load('model/scaler.pkl')
    data_ref = pd.read_csv('data/NO2_bangkalan.csv')
    data_ref['time'] = pd.to_datetime(data_ref['time'], errors='coerce')

    # Gunakan waktu awal dataset sebagai anchor
    t0 = data_ref['time'].min()

    # Pastikan format waktu valid
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    if df['time'].isna().any():
        raise ValueError("Format waktu tidak valid. Gunakan format YYYY-MM-DD HH:MM:SS")

    # Buat fitur tambahan sesuai model
    df['hour'] = df['time'].dt.hour
    df['dayofyear'] = df['time'].dt.dayofyear
    df['time_numeric'] = (df['time'] - t0).dt.total_seconds()

    # Urutkan kolom sesuai urutan saat training
    X = df[['time_numeric', 'hour', 'dayofyear']]

    # Normalisasi menggunakan scaler yang sama
    processed = scaler.transform(X)

    return processed
