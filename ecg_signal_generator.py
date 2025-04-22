import numpy as np
import pandas as pd

def simulate_ecg_signal(duration_sec=60, fs=250):
    t = np.linspace(0, duration_sec, duration_sec * fs)
    signal = 0.3 * np.sin(2 * np.pi * 1.2 * t)
    fa_pattern = np.random.randn(len(t)) * 0.1
    signal[(t >= 20) & (t <= 40)] += fa_pattern[(t >= 20) & (t <= 40)] + 0.3 * np.sin(15 * t[(t >= 20) & (t <= 40)])
    labels = np.where((t >= 20) & (t <= 40), 1, 0)
    return pd.DataFrame({'time': t, 'ecg': signal, 'label': labels})

df = simulate_ecg_signal()
df.to_csv("senyal_ecg_simulat.csv", index=False)
print("✅ CSV guardat com 'senyal_ecg_simulat.csv'")

