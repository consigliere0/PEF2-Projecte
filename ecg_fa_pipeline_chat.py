import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Simulació senyal ECG amb FA
def simulate_ecg_signal(duration_sec=60, fs=250):
    t = np.linspace(0, duration_sec, duration_sec * fs)
    signal = 0.3 * np.sin(2 * np.pi * 1.2 * t)  # component sinusal
    fa_pattern = np.random.randn(len(t)) * 0.1
    signal[(t >= 20) & (t <= 40)] += fa_pattern[(t >= 20) & (t <= 40)] + 0.3 * np.sin(15 * t[(t >= 20) & (t <= 40)])
    labels = np.where((t >= 20) & (t <= 40), 1, 0)
    return pd.DataFrame({'time': t, 'ecg': signal, 'label': labels})

# 2. Filtratge bandpass
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype='band')

def filter_signal(data, lowcut=0.5, highcut=40.0, fs=250):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)

# 3. Segmentació + extracció de features
def segment_and_extract(df, fs=250, window_sec=10):
    win_size = fs * window_sec
    features, labels = [], []
    for i in range(0, len(df) - win_size, win_size):
        segment = df['ecg_filtered'].values[i:i + win_size]
        label = int(df['label'].iloc[i:i + win_size].mean() > 0.5)
        feat = [
            segment.mean(),
            segment.std(),
            np.max(segment),
            np.min(segment),
            np.percentile(segment, 25),
            np.percentile(segment, 75)
        ]
        features.append(feat)
        labels.append(label)
    return np.array(features), np.array(labels)

# 4. Executa el pipeline complet
if __name__ == "__main__":
    df_ecg = simulate_ecg_signal() # ESTA ES LA LINEA A SUSTITUIR
    df_ecg['ecg_filtered'] = filter_signal(df_ecg['ecg'].values)

    X, y = segment_and_extract(df_ecg)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n📊 Informe de classificació:")
    print(classification_report(y_test, y_pred))
    print("\n📉 Matriu de confusió:")
    print(confusion_matrix(y_test, y_pred))

    # Visualització amb àrees de FA
    time_vals = df_ecg['time'].values.astype(float)
    ecg_vals = df_ecg['ecg_filtered'].values.astype(float)
    fa_zone = df_ecg['label'].values.astype(bool)

    plt.figure(figsize=(12, 4))
    plt.plot(time_vals, ecg_vals, label='ECG filtrat', color='blue')
    plt.fill_between(time_vals, -1, 1, where=fa_zone, color='red', alpha=0.2, label='FA simulada')
    plt.xlabel("Temps (s)")
    plt.ylabel("ECG (mV)")
    plt.title("Simulació de senyal ECG amb FA")
    plt.legend()
    plt.tight_layout()
    plt.show()


# CON EL ARCHIVO DEL BITALINO
# sustituir la linea marcada: df_ecg = simulate_ecg_signal()
# por los siguientes bloques
df_ecg = pd.read_csv("el_teu_fitxer.csv")
df_ecg.rename(columns={'ECG_column_name': 'ecg'}, inplace=True)  # adapta el nom de la columna
df_ecg['time'] = np.arange(len(df_ecg)) / 250  # si no tens columna 'time', crea-la
df_ecg['label'] = 0  # si no tens etiquetes reals, pots posar-les totes a 0
