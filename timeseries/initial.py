# Working version

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# Load datasets
train_a = pd.read_csv('/kaggle/input/water-pump-pressure-anomaly-detection/train/TRAIN_A.csv')
train_b = pd.read_csv('/kaggle/input/water-pump-pressure-anomaly-detection/train/TRAIN_B.csv')

# Function to transform timestamp
def transform_timestamp(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d/%m/%y %H:%M')
    data['hour_sin'] = np.sin(2 * np.pi * data['timestamp'].dt.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['timestamp'].dt.hour / 24)
    data['minute_sin'] = np.sin(2 * np.pi * data['timestamp'].dt.minute / 60)
    data['minute_cos'] = np.cos(2 * np.pi * data['timestamp'].dt.minute / 60)
    return data.drop(columns=['timestamp'])

# Apply transformation
train_a = transform_timestamp(train_a)
train_b = transform_timestamp(train_b)

# Define target columns
target_columns_a = ['anomaly'] + [col for col in train_a.columns if '_flag' in col]
target_columns_b = ['anomaly'] + [col for col in train_b.columns if '_flag' in col]

# Preprocess datasets
def preprocess_data(data, target_columns):
    targets = data[target_columns].values
    features = data.drop(columns=target_columns, errors='ignore').values
    return features, targets

X_a, y_a = preprocess_data(train_a, target_columns_a)
X_b, y_b = preprocess_data(train_b, target_columns_b)

# Normalize datasets
scaler_a = MinMaxScaler().fit(X_a)
scaler_b = MinMaxScaler().fit(X_b)
X_a = scaler_a.transform(X_a)
X_b = scaler_b.transform(X_b)

# Determine feature dimensions
feature_dim_a = X_a.shape[1]
feature_dim_b = X_b.shape[1]

# Pad target labels
max_targets = max(y_a.shape[1], y_b.shape[1])
y_a = np.pad(y_a, ((0, 0), (0, max_targets - y_a.shape[1])), mode='constant')
y_b = np.pad(y_b, ((0, 0), (0, max_targets - y_b.shape[1])), mode='constant')

# Define LSTM model
def build_model(input_dim, output_dim):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(None, input_dim)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(output_dim, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train models for A and B
model_a = build_model(feature_dim_a, max_targets)
model_a.fit(np.expand_dims(X_a, axis=1), y_a, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
model_a.save('anomaly_model_a.h5')

model_b = build_model(feature_dim_b, max_targets)
model_b.fit(np.expand_dims(X_b, axis=1), y_b, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
model_b.save('anomaly_model_b.h5')

# Test data processing
def process_test_data(file_path, scaler, feature_dim):
    test_data = pd.read_csv(file_path)
    test_data['timestamp'] = test_data['timestamp'].str.extract(r'T-(\\d+)', expand=False).fillna(0).astype(int) * -1
    test_data['timestamp'] = pd.to_datetime(test_data['timestamp'], unit='m', origin='unix', errors='coerce').fillna(pd.Timestamp(0))
    test_data = transform_timestamp(test_data)
    
    padded_features = np.zeros((test_data.shape[0], feature_dim))
    padded_features[:, :test_data.shape[1]] = test_data.values
    padded_features = scaler.transform(padded_features)
    return np.expand_dims(padded_features, axis=1)

# Prediction function
def predict_test_data(test_dir, model, scaler, feature_dim):
    predictions = []
    ids = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                test_features = process_test_data(file_path, scaler, feature_dim)
                pred = model.predict(test_features)
                predictions.append((file.replace('.csv', ''), pred[0].tolist()))
    return predictions

# Predict for C and D
model_a = tf.keras.models.load_model('anomaly_model_a.h5')
model_b = tf.keras.models.load_model('anomaly_model_b.h5')

predictions_c = predict_test_data('/kaggle/input/water-pump-pressure-anomaly-detection/test/C', model_a, scaler_a, feature_dim_a)
predictions_d = predict_test_data('/kaggle/input/water-pump-pressure-anomaly-detection/test/D', model_a, scaler_a, feature_dim_a)

# Save results
all_predictions = predictions_c + predictions_d
submission = pd.DataFrame({
    'ID': [pred[0] for pred in all_predictions],
    'flag_list': [str(pred) for pred in all_predictions]
})
submission.to_csv('Submission.csv', index=False)
