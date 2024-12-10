import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.models import load_model

# Load train datasets
train_a = pd.read_csv('Train_A.csv')
train_b = pd.read_csv('Train_B.csv')

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.models import load_model

# Load train datasets
train_a = pd.read_csv('Train_A.csv')
train_b = pd.read_csv('Train_B.csv')



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

# Transform timestamps
train_a = transform_timestamp(train_a)
train_b = transform_timestamp(train_b)

# Combine datasets
X_a, y_a = train_a[feature_columns].values, train_a[target_columns].values
X_b, y_b = train_b[feature_columns].values, train_b[target_columns].values

# Combine into one dataset
X = np.vstack([X_a, X_b])
y = np.vstack([y_a, y_b])

# Normalize combined data
scaler = MinMaxScaler().fit(X)
X = scaler.transform(X)



# Build LSTM Model
model = Sequential([
    Masking(mask_value=0.0, input_shape=(None, feature_dim)),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(output_dim, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X.reshape(-1, 1, feature_dim), y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
model.save('anomaly_model.h5')


def transform_test_data(data):
    data['timestamp'] = data['timestamp'].str.extract(r'T-(\\d+)', expand=False).fillna(0).astype(int) * -1
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='m', origin='unix', errors='coerce')
    data['hour_sin'] = np.sin(2 * np.pi * data['timestamp'].dt.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['timestamp'].dt.hour / 24)
    data['minute_sin'] = np.sin(2 * np.pi * data['timestamp'].dt.minute / 60)
    data['minute_cos'] = np.cos(2 * np.pi * data['timestamp'].dt.minute / 60)
    return data.drop(columns=['timestamp'])

def process_test_file(file_path, scaler, feature_columns, model):
    test_data = pd.read_csv(file_path)
    test_data = transform_test_data(test_data)
    for col in feature_columns:
        if col not in test_data.columns:
            test_data[col] = 0
    test_data = test_data[feature_columns]
    scaled_data = scaler.transform(test_data)
    pred = model.predict(scaled_data.reshape(scaled_data.shape[0], 1, scaled_data.shape[1]))
    return (os.path.basename(file_path).replace('.csv', ''), (pred > 0.5).astype(int).tolist())

# Predict anomalies for test files
def process_test_directory(test_dir, scaler, feature_columns, model):
    predictions = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.csv'):
                pred = process_test_file(os.path.join(root, file), scaler, feature_columns, model)
                predictions.append(pred)
    return predictions

# Load model and process test data
model = load_model('anomaly_model.h5')
predictions_c = process_test_directory('C', scaler, feature_columns, model)
predictions_d = process_test_directory('D', scaler, feature_columns, model)

# Save submission
submission = pd.DataFrame({
    'ID': [pred[0] for pred in predictions_c + predictions_d],
    'flag_list': [str(pred[1]) for pred in predictions_c + predictions_d]
})
submission.to_csv('Submission.csv', index=False)


