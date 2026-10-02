import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. GENERATE SYNTHETIC DATASET (Realistic)
# ==========================================
# We generate realistic analog voltage readings (0.0 to 5.0 V)
print("Generating synthetic dataset for physical sensors...")

np.random.seed(42)
num_samples = 3000

# AMAN (Safe): Normal air, voltages generally low (0.1 - 1.2 V)
aman_data = np.random.uniform(low=0.1, high=1.2, size=(1000, 4))
aman_labels = np.array([0] * 1000)

# WASPADA (Warning): Slight elevation in one or more gases (1.2 - 2.5 V)
waspada_data = np.random.uniform(low=1.2, high=2.5, size=(1000, 4))
waspada_labels = np.array([1] * 1000)

# BAHAYA (Danger): High concentration of harmful gases (2.5 - 4.8 V)
bahaya_data = np.random.uniform(low=2.5, high=4.8, size=(1000, 4))
bahaya_labels = np.array([2] * 1000)

# Combine and shuffle
X = np.vstack((aman_data, waspada_data, bahaya_data))
y = np.concatenate((aman_labels, waspada_labels, bahaya_labels))

# Convert to DataFrame just to save as CSV for audit trail
df = pd.DataFrame(X, columns=['mics5524', 'tgs2600', 'mq2', 'mq135'])
df['label'] = y
df.to_csv('siparta_sensor_dataset.csv', index=False)
print(f"Dataset generated: siparta_sensor_dataset.csv (Shape: {df.shape})")

# ==========================================
# 2. PREPROCESSING
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler explicitly so it can be loaded in inference.py
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
scaler_path = os.path.join(model_dir, 'siparta_scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to {scaler_path}")

# ==========================================
# 3. TRAIN JST (ANN) MODEL
# ==========================================
# Model requires one-hot encoding for categorical crossentropy or sparse
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=3)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training ANN model...")
model.fit(
    X_train_scaled, y_train_cat, 
    epochs=20, 
    batch_size=32, 
    validation_split=0.2,
    verbose=1
)

# Evaluate
loss, acc = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {acc*100:.2f}%")

# Save keras model
keras_path = os.path.join(model_dir, 'siparta_ann.keras')
model.save(keras_path)
print(f"Keras model saved to {keras_path}")

# ==========================================
# 4. CONVERT TO TFLITE (For Edge/RPi)
# ==========================================
print("Converting model to TFLite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_path = os.path.join(model_dir, 'siparta_ann.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
    
print(f"TFLite model successfully saved to {tflite_path}")
print("\n[SUCCESS] Pipeline end-to-end completed! Dataset, Scaler, and Models are ready for production.")
