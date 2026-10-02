import pandas as pd
import numpy as np

def audit():
    df = pd.read_csv("siparta_sensor_dataset.csv")
    print("=== AUDIT DATASET SIPARTA ===")
    print(f"Total Baris: {len(df)}")
    
    # Deteksi Dummy (Apakah data ini hasil np.random.uniform?)
    # Kita cek batas-batasnya untuk setiap class
    print("\n--- DISTRIBUSI BERDASARKAN KELAS ---")
    for lbl in [0, 1, 2]:
        subset = df[df['label'] == lbl]
        if not subset.empty:
            mins = subset[['mics5524', 'tgs2600', 'mq2', 'mq135']].min().min()
            maxs = subset[['mics5524', 'tgs2600', 'mq2', 'mq135']].max().max()
            print(f"Kelas {lbl} ({len(subset)} baris): Range {mins:.2f}V - {maxs:.2f}V")

    # Cek duplikat dan NaN
    print("\n--- KUALITAS DATA ---")
    print(f"Duplikat: {df.duplicated().sum()}")
    print(f"Nilai Kosong (NaN): {df.isna().sum().sum()}")
    
    # Identifikasi Anomali / Data Ekstrem
    anomalies = df[(df.drop('label', axis=1) < 0).any(axis=1) | (df.drop('label', axis=1) > 5.0).any(axis=1)]
    print(f"Nilai Sensor < 0V atau > 5V (Di luar kapabilitas ADS1115): {len(anomalies)}")

if __name__ == '__main__':
    audit()
