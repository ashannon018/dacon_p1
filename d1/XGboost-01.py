import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Load dataset
train_sample_path = './input/train.csv'
train_df = pd.read_csv(train_sample_path)

# Step 1: Encode the 'SUBCLASS' column (target labels)
le = LabelEncoder()
train_df['SUBCLASS'] = le.fit_transform(train_df['SUBCLASS'])

# Step 2: Convert mutation columns from 'WT' and mutation strings to binary (0 for WT, 1 for mutation)
mutation_cols = train_df.columns[2:]
train_df[mutation_cols] = train_df[mutation_cols].applymap(lambda x: 0 if x == 'WT' else 1)

# Step 3: Standardize the mutation features
scaler = StandardScaler()
train_df[mutation_cols] = scaler.fit_transform(train_df[mutation_cols])

# Step 4: Split the data into training and testing sets
X = train_df[mutation_cols]
y = train_df['SUBCLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train an XGBoost model to find important features
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Get feature importances
feature_importances = xgb_model.feature_importances_
sorted_idx = feature_importances.argsort()

# Select top 2000 features based on importance
top_n = 2000  # Adjust this value to select the top N important features
top_features_idx = sorted_idx[-top_n:]

# Step 6: Reduce training and testing data to the top N features
X_train_reduced = X_train.iloc[:, top_features_idx]
X_test_reduced = X_test.iloc[:, top_features_idx]

# Step 7: Train a new XGBoost model on the reduced feature set
best_xgb = xgb.XGBClassifier(n_estimators=100, random_state=42)
best_xgb.fit(X_train_reduced, y_train)

# Step 8: Evaluate the model on the test set
y_pred = best_xgb.predict(X_test_reduced)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the reduced feature set: {accuracy}")

# Step 9: Load test.csv for prediction
test_path = './input/test.csv'
test_df = pd.read_csv(test_path)

# Preprocess the test data (same as train data preprocessing)
test_df[mutation_cols] = test_df[mutation_cols].applymap(lambda x: 0 if x == 'WT' else 1)
test_df[mutation_cols] = scaler.transform(test_df[mutation_cols])

# Reduce test data to the top N important features
X_test_final = test_df[mutation_cols].iloc[:, top_features_idx]

# Step 10: Predict using the trained model
y_test_pred = best_xgb.predict(X_test_final)

# Step 11: Decode the predicted class labels back to their original subclass names
y_test_pred_labels = le.inverse_transform(y_test_pred)

# Step 12: Store the predictions in the test DataFrame
test_df['Predicted_SUBCLASS'] = y_test_pred_labels

# Display the first few rows of predictions
test_df[['Predicted_SUBCLASS']].value_counts()


test_df[['ID', 'Predicted_SUBCLASS']].to_csv('submit02.csv',index=False)
