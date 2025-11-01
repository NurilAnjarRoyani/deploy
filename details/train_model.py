import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# === 1. Load dataset ===
data_path = 'data/NO2_bangkalan.csv'
data = pd.read_csv(data_path)

# === 2. Konversi kolom waktu ===
data['time'] = pd.to_datetime(data['time'], errors='coerce')
data = data.dropna(subset=['time'])

# === 3. Tangani nilai kosong dan tipe data ===
data['NO2'] = pd.to_numeric(data['NO2'], errors='coerce')
data = data.dropna(subset=['NO2'])

# === 4. Buat fitur lag untuk memperbaiki autokorelasi ===
for lag in range(1, 4):  # gunakan 3 lag terakhir
    data[f'lag_{lag}'] = data['NO2'].shift(lag)
data = data.dropna()

# === 5. Ekstrak fitur waktu tambahan ===
data['hour'] = data['time'].dt.hour
data['dayofyear'] = data['time'].dt.dayofyear
data['time_numeric'] = (data['time'] - data['time'].min()).dt.total_seconds()

# === 6. Siapkan fitur dan target ===
X = data[['time_numeric', 'hour', 'dayofyear', 'lag_1', 'lag_2', 'lag_3']]
y = data['NO2']

# === 7. Normalisasi data ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === 8. Split data latih dan uji ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=False
)

# === 9. Latih model KNN ===
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# === 10. Evaluasi model ===
y_pred = knn.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(" Model KNN berhasil dilatih tanpa error!")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Jumlah data setelah dibersihkan: {len(data)} baris")

# === 11. Simpan model dan scaler ===
os.makedirs('model', exist_ok=True)
joblib.dump(knn, 'model/knn_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print(" Model dan scaler telah disimpan di folder 'model/'")

# === 12. Fungsi prediksi multi-hari ===
def predict_future(days=1):
    """
    Memprediksi nilai NO2 untuk beberapa hari ke depan.
    """
    future_preds = []
    last_rows = data.iloc[-3:].copy()

    for d in range(days):
        new_time = last_rows['time'].iloc[-1] + pd.Timedelta(hours=24)
        new_features = {
            'time_numeric': (new_time - data['time'].min()).total_seconds(),
            'hour': new_time.hour,
            'dayofyear': new_time.dayofyear,
            'lag_1': last_rows['NO2'].iloc[-1],
            'lag_2': last_rows['NO2'].iloc[-2],
            'lag_3': last_rows['NO2'].iloc[-3],
        }
        X_new = scaler.transform(pd.DataFrame([new_features]))
        pred = knn.predict(X_new)[0]
        future_preds.append((new_time, pred))

        # Update untuk prediksi berikutnya
        new_row = pd.DataFrame({'time': [new_time], 'NO2': [pred]})
        last_rows = pd.concat([last_rows, new_row]).iloc[-3:]

    return pd.DataFrame(future_preds, columns=['Tanggal', 'Prediksi_NO2'])

# Simpan hasil prediksi 7 hari ke depan
future_df = predict_future(days=7)
print("\n Prediksi 7 Hari ke Depan:")
print(future_df)
