def process_test_directory_parallel_gpu(directory, scaler, feature_columns, model, max_workers=None):
    file_paths = [os.path.join(root, file)
                  for root, _, files in os.walk(directory) for file in files if file.endswith('.csv')]

    batch_size = 64  # Number of files per batch
    results = []

    # Process in batches
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]
        future_results = executor.map(process_test_files_batch, batches, [scaler]*len(batches),
                                      [feature_columns]*len(batches), [model]*len(batches))
        for batch_results in future_results:
            results.extend(batch_results)

    ids, preds = zip(*results)
    return ids, preds


def process_test_files_batch(file_paths, scaler, feature_columns, model):
    combined_data = []
    ids = []
    
    # Preprocess all files in the batch
    for file_path in file_paths:
        test_data = pd.read_csv(file_path)
        test_data = transform_timestamp_for_test(test_data)
        
        # Ensure consistent features
        missing_columns = [col for col in feature_columns if col not in test_data.columns]
        for col in missing_columns:
            test_data[col] = 0
        test_data = test_data[feature_columns]
        
        scaled_data = scaler.transform(test_data.values)
        combined_data.append(scaled_data)
        ids.append(os.path.basename(file_path).replace('.csv', ''))
    
    # Stack and predict as a batch
    combined_data = np.vstack(combined_data)
    reshaped_data = combined_data.reshape(combined_data.shape[0], 1, combined_data.shape[1])
    preds = model.predict(reshaped_data, batch_size=32)  # Batch size for GPU efficiency
    preds = (preds > 0.5).astype(int)  # Convert probabilities to binary
    
    # Group predictions by file
    results = []
    start = 0
    for file_path, test_data in zip(file_paths, combined_data):
        file_length = len(test_data)  # Number of rows per file
        results.append((os.path.basename(file_path).replace('.csv', ''), preds[start:start+file_length].tolist()))
        start += file_length
    return results

      
