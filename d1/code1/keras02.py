import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE  # SMOTE 사용을 위한 라이브러리 추가

# 학습 데이터 읽기
train_file_path = '/content/train.csv'  # 학습 데이터 파일 경로
train_data = pd.read_csv(train_file_path)

# 결측값 처리
train_data.fillna('WT', inplace=True)

# 테스트 데이터 읽기
test_file_path = '/content/test.csv'  # 테스트 데이터 파일 경로
test_data = pd.read_csv(test_file_path)

# 결측값 처리
test_data.fillna('WT', inplace=True)

# 테스트 데이터에 SUBCLASS 열 추가 (NaN으로 설정)
test_data['SUBCLASS'] = None

# 학습 데이터와 테스트 데이터 결합
combined_data = pd.concat([train_data, test_data], ignore_index=True)

# 특징과 타겟 분리
X = combined_data.drop(columns=['ID', 'SUBCLASS'])  # ID와 SUBCLASS 열 제거
y = combined_data['SUBCLASS']

# 특징 인코딩: 'WT'는 0, 그 외는 1로 변환
X_encoded = X.apply(lambda col: col.map(lambda x: 0 if x == 'WT' else 1))

# 레이블 인코딩 (학습 데이터에만 적용)
le = LabelEncoder()
y_encoded = le.fit_transform(y.dropna())  # NaN을 제외하고 인코딩

# 학습 데이터와 테스트 데이터 분리
y_encoded_full = pd.Series([None] * len(combined_data), index=combined_data.index)  # 초기화
y_encoded_full.iloc[:len(y_encoded)] = y_encoded  # 학습 데이터에 대해서만 값 설정

# 학습/검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X_encoded[:len(y_encoded)], y_encoded, test_size=0.2, random_state=42)

# 모델 구성
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),  # 드롭아웃 비율 조정
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),  # 드롭아웃 비율 조정
    layers.Dense(128, activation='relu'),
    layers.Dense(len(le.classes_), activation='softmax')  # 클래스 수에 맞추어 출력 노드 수 설정
])

# 모델 컴파일
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 조기 종료 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 모델 학습
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=16, callbacks=[early_stopping])

# 테스트 데이터 예측
X_test_encoded = X_encoded[len(y_encoded):]  # 테스트 데이터 부분

predictions = model.predict(X_test_encoded)

# 예측 결과 변환
predicted_classes = predictions.argmax(axis=1)

# 레이블 인코딩된 클래스를 원래 클래스 이름으로 변환
predicted_labels = le.inverse_transform(predicted_classes)

# 결과를 DataFrame으로 저장
results = pd.DataFrame({
    'ID': test_data['ID'],
    'SUBCLASS': predicted_labels
})

# 결과를 CSV 파일로 저장
results.to_csv('maybe_predictions_no_scaler.csv', index=False)

print("Predictions saved to 'predictions_no_scaler.csv'.")
