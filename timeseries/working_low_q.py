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


###Parrarel###
from concurrent.futures import ProcessPoolExecutor
import tensorflow as tf

# Parallel test data processing
def process_test_file_parallel(file_path, scaler, feature_dim):
    test_data = pd.read_csv(file_path)
    test_data['timestamp'] = test_data['timestamp'].str.extract(r'T-(\\d+)', expand=False).fillna(0).astype(int) * -1
    test_data['timestamp'] = pd.to_datetime(test_data['timestamp'], unit='m', origin='unix', errors='coerce').fillna(pd.Timestamp(0))
    test_data = transform_timestamp(test_data)

    padded_features = np.zeros((test_data.shape[0], feature_dim))
    padded_features[:, :test_data.shape[1]] = test_data.values
    padded_features = scaler.transform(padded_features)
    return np.expand_dims(padded_features, axis=1), file_path

def predict_test_batch(file_paths, model, scaler, feature_dim):
    features = []
    ids = []
    for file_path in file_paths:
        test_features, file_id = process_test_file_parallel(file_path, scaler, feature_dim)
        features.append(test_features)
        ids.append(file_id)

    features = np.concatenate(features, axis=0)
    predictions = model.predict(features)
    return [(os.path.basename(file).replace('.csv', ''), pred.tolist()) for file, pred in zip(ids, predictions)]

def predict_test_data_parallel(test_dir, model, scaler, feature_dim, batch_size=16):
    file_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(test_dir)
        for file in files if file.endswith('.csv')
    ]

    predictions = []
    with ProcessPoolExecutor() as executor:
        for i in range(0, len(file_paths), batch_size):
            batch_files = file_paths[i:i + batch_size]
            batch_preds = predict_test_batch(batch_files, model, scaler, feature_dim)
            predictions.extend(batch_preds)

    return predictions

# Predict for C and D in parallel
predictions_c = predict_test_data_parallel('/kaggle/input/water-pump-pressure-anomaly-detection/test/C', model_a, scaler_a, feature_dim_a)
predictions_d = predict_test_data_parallel('/kaggle/input/water-pump-pressure-anomaly-detection/test/D', model_a, scaler_a, feature_dim_a)

# Save results
all_predictions = predictions_c + predictions_d
submission = pd.DataFrame({
    'ID': [pred[0] for pred in all_predictions],
    'flag_list': [str(pred[1]) for pred in all_predictions]
})
submission.to_csv('Submission_model_a.csv', index=False)
