# ThreadPoolExecutor를 대신 ProcessPoolExecutor로 변경하여 CPU 병렬 처리를 활용할 수 있다. 이는 CPU 코어를 더 효율적으로 사용할 수 있다.


# Transform test data
def transform_test_data(data):
    data['timestamp'] = data['timestamp'].str.extract(r'T-(\d+)', expand=False).fillna(0).astype(int) * -1
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='m', origin='unix', errors='coerce')
    data['hour_sin'] = np.sin(2 * np.pi * data['timestamp'].dt.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['timestamp'].dt.hour / 24)
    data['minute_sin'] = np.sin(2 * np.pi * data['timestamp'].dt.minute / 60)
    data['minute_cos'] = np.cos(2 * np.pi * data['timestamp'].dt.minute / 60)
    return data.drop(columns=['timestamp'])

# Process individual test file
def process_test_file(file_path):
    test_data = pd.read_csv(file_path)
    test_data = transform_test_data(test_data)
    for col in feature_columns:
        if col not in test_data.columns:
            test_data[col] = 0
    test_data = test_data[feature_columns]
    test_data_scaled = scaler.transform(pd.DataFrame(test_data, columns=feature_columns))  # Transform with feature names
    pred = model.predict(test_data_scaled.reshape(test_data_scaled.shape[0], 1, test_data_scaled.shape[1]))
    return (os.path.basename(file_path).replace('.csv', ''), (pred > 0.5).astype(int).tolist())

# Process all test files in a directory using parallel execution
def process_test_directory_parallel(test_dir):
    predictions = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for root, _, files in os.walk(test_dir):
            for file in files:
                if file.endswith('.csv'):
                    futures.append(executor.submit(process_test_file, os.path.join(root, file)))
        for future in futures:
            predictions.append(future.result())
    return predictions

# Load model and process test data
model = load_model('anomaly_model.h5')
predictions_c = process_test_directory_parallel('/kaggle/input/water-pump-pressure-anomaly-detection/test/C')
predictions_d = process_test_directory_parallel('/kaggle/input/water-pump-pressure-anomaly-detection/test/D')

# Save submission
submission = pd.DataFrame({
    'ID': [pred[0] for pred in predictions_c + predictions_d],
    'flag_list': [str(pred[1]) for pred in predictions_c + predictions_d]
})
submission.to_csv('Submission.csv', index=False)


#ProcessPoolExecutor로

# Transform test data
def transform_test_data(data):
    data['timestamp'] = data['timestamp'].str.extract(r'T-(\d+)', expand=False).fillna(0).astype(int) * -1
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='m', origin='unix', errors='coerce')
    data['hour_sin'] = np.sin(2 * np.pi * data['timestamp'].dt.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['timestamp'].dt.hour / 24)
    data['minute_sin'] = np.sin(2 * np.pi * data['timestamp'].dt.minute / 60)
    data['minute_cos'] = np.cos(2 * np.pi * data['timestamp'].dt.minute / 60)
    return data.drop(columns=['timestamp'])

# Process individual test file
def process_test_file(file_path):
    test_data = pd.read_csv(file_path)
    test_data = transform_test_data(test_data)
    for col in feature_columns:
        if col not in test_data.columns:
            test_data[col] = 0
    test_data = test_data[feature_columns]
    test_data_scaled = scaler.transform(pd.DataFrame(test_data, columns=feature_columns))  # Transform with feature names
    pred = model.predict(test_data_scaled.reshape(test_data_scaled.shape[0], 1, test_data_scaled.shape[1]), batch_size=128)  # Batch prediction
    return (os.path.basename(file_path).replace('.csv', ''), (pred > 0.5).astype(int).tolist())

# Process all test files in a directory using parallel execution
def process_test_directory_parallel(test_dir):
    predictions = []
    with ProcessPoolExecutor() as executor:  # Use ProcessPoolExecutor for better CPU utilization
        futures = []
        for root, _, files in os.walk(test_dir):
            for file in files:
                if file.endswith('.csv'):
                    futures.append(executor.submit(process_test_file, os.path.join(root, file)))
        for future in futures:
            predictions.append(future.result())
    return predictions

# Load model and process test data
model = load_model('anomaly_model.h5')
predictions_c = process_test_directory_parallel('/kaggle/input/water-pump-pressure-anomaly-detection/test/C')
predictions_d = process_test_directory_parallel('/kaggle/input/water-pump-pressure-anomaly-detection/test/D')

# Save submission
submission = pd.DataFrame({
    'ID': [pred[0] for pred in predictions_c + predictions_d],
    'flag_list': [str(pred[1]) for pred in predictions_c + predictions_d]
})
submission.to_csv('Submission.csv', index=False)
