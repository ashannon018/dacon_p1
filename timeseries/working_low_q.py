import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional
import tensorflow as tf
import logging

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO)

# Load train datasets
train_a = pd.read_csv('/kaggle/input/waterpump-anomaly/train/TRAIN_A.csv')
train_b = pd.read_csv('/kaggle/input/waterpump-anomaly/train/TRAIN_B.csv')


# Define all columns for alignment
all_columns = list(set(train_a.columns).union(set(train_b.columns)))

# Define target columns
target_columns = ['anomaly'] + [col for col in all_columns if '_flag' in col]

# Define feature columns excluding 'timestamp' and target columns
feature_columns = [col for col in all_columns if col not in target_columns + ['timestamp']]

# Add missing columns with default value (0)
def align_columns(data, all_columns):
    for col in all_columns:
        if col not in data.columns:
            data[col] = 0
    return data

train_a = align_columns(train_a, all_columns)
train_b = align_columns(train_b, all_columns)

# Transform timestamps
def transform_timestamp(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d/%m/%y %H:%M')
    data['hour_sin'] = np.sin(2 * np.pi * data['timestamp'].dt.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['timestamp'].dt.hour / 24)
    data['minute_sin'] = np.sin(2 * np.pi * data['timestamp'].dt.minute / 60)
    data['minute_cos'] = np.cos(2 * np.pi * data['timestamp'].dt.minute / 60)
    return data.drop(columns=['timestamp'])

train_a = transform_timestamp(train_a)
train_b = transform_timestamp(train_b)

# Combine datasets
X_a, y_a = train_a[feature_columns].values, train_a[target_columns].values
X_b, y_b = train_b[feature_columns].values, train_b[target_columns].values

# Combine into one dataset
X = np.vstack([X_a, X_b])
y = np.vstack([y_a, y_b])

# Normalize combined data
scaler = MinMaxScaler()
X = pd.DataFrame(X, columns=feature_columns)  # Ensure X has feature names
X_scaled = scaler.fit_transform(X)

# Build LSTM Model with Bidirectional LSTM
model = Sequential([
    Masking(mask_value=0.0, input_shape=(None, X_scaled.shape[1])),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(y.shape[1], activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, min_lr=1e-4)

# Train the model
model.fit(X_scaled.reshape(-1, 1, X_scaled.shape[1]), y, epochs=35, batch_size=32, validation_split=0.2, verbose=1,
          callbacks=[early_stopping]
         )  # Set verbose to 0
logging.info("Model training completed.")

model.save('anomaly_model.h5')

#-----------
# Save predictions to submission file incrementally
def save_predictions(predictions, test_file, submission_file):
    with open(submission_file, 'a') as f:
        for i, pred in enumerate(predictions):
            pred_str = ','.join(map(str, (pred > 0.5).astype(int)))  # Convert predictions to binary flags
            f.write(f"{test_file}_{i},[{pred_str}]\n")

# Process and save predictions for test datasets
submission_file = 'Submission-m1.csv'

# Remove existing submission file if exists
if os.path.exists(submission_file):
    os.remove(submission_file)
    
# Process and save predictions for test C
for test_file in os.listdir('/kaggle/input/waterpump-anomaly/test/C'):
    if test_file.endswith('.csv'):
        test_data = pd.read_csv(f'/kaggle/input/waterpump-anomaly/test/C/{test_file}')
        test_data = transform_test_data(test_data)
        for col in feature_columns:
            if col not in test_data.columns:
                test_data[col] = 0
        test_features = scaler.transform(test_data[feature_columns].values)
        predictions = model.predict(test_features.reshape(-1, 1, test_features.shape[1]))
        save_predictions(predictions, test_file.replace('.csv', ''), submission_file)

# Process and save predictions for test D
for test_file in os.listdir('/kaggle/input/waterpump-anomaly/test/D'):
    if test_file.endswith('.csv'):
        test_data = pd.read_csv(f'/kaggle/input/waterpump-anomaly/test/D/{test_file}')
        test_data = transform_test_data(test_data)
        for col in feature_columns:
            if col not in test_data.columns:
                test_data[col] = 0
        test_features = scaler.transform(test_data[feature_columns].values)
        predictions = model.predict(test_features.reshape(-1, 1, test_features.shape[1]))
        save_predictions(predictions, test_file.replace('.csv', ''), submission_file)
