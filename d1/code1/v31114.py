# %%
import pandas as pd 
df = pd.read_csv ('../input/train.csv')

# %%
data = df.copy()

# %%
#data = df.drop(columns = ['ID', 'SUBCLASS'], axis=1)

# %%
df_test = pd.read_csv('../input/test.csv')
#df_test = df_test.drop('ID', axis=1)

# %%
empty_lines =  [ 15, 20, 21, 25, 28, 31, 34, 38, 39, 48, 52, 65, 68,76
,84,96,98,99,105,107,108,109,114,129,157,158,171,172
,197,210,216,230,240,248,258,268,280,290,292,299,300,304
,308,309,312,313,317,330,346,348,349,353,354,356,362,364
,368,385,400,403,414,419,421,423,425,426,427,429,434,436
,453,457,460,464,465,466,468,470,487,488,502,509,510,514
,526,527,530,531,532,543,554,555,567,569,570,575,580,581
,584,587,591,596,597,601,606,615,620,624,628,629,630,631
,633,634,640,651,653,658,663,665,667,671,675,678,682,685
,688,697,704,717,725,734,744,747,751,753,756,760,764,766
,768,773,774,775,777,778,779,792,795,797,799,801,807,808
,816,818,829,839,847,859,861,862,863,867,870,874,882,888
,892,908,916,918,919,928,933,937,939,946,948,949,959,963
,966,969,973,977,980,983,984,1000,1005,1010,1020,1027,1040,1041
,1043,1050,1051,1052,1054,1062,1068,1070,1079,1080,1087,1094,1097,1103
,1108,1109,1119,1124,1132,1133,1142,1148,1151,1153,1156,1161,1164,1166
,1173,1178,1181,1187,1195,1198,1203,1206,1210,1212,1223,1224,1245,1247
,1253,1274,1277,1280,1288,1289,1291,1298,1316,1317,1324,1339,1343,1345
,1349,1351,1356,1357,1368,1373,1389,1393,1407,1409,1417,1427,1431,1439
,1446,1448,1450,1451,1455,1458,1463,1481,1491,1493,1495,1498,1505,1510
,1522,1525,1529,1530,1537,1546,1547,1548,1550,1553,1570,1576,1582,1589
,1591,1593,1605,1607,1608,1619,1629,1636,1637,1643,1652,1672,1688,1689
,1691,1699,1705,1719,1720,1721,1726,1739,1746,1765,1771,1782,1787,1792
,1796,1800,1803,1814,1823,1837,1842,1848,1852,1870,1877,1878,1887,1888
,1897,1907,1916,1917,1923,1928,1938,1944,1956,1964,1966,1967,1991,1996
,2011,2019,2020,2021,2025,2026,2034,2036,2053,2054,2057,2067,2070,2075
,2076,2077,2078,2094,2103,2105,2108,2110,2123,2133,2138,2145,2148,2151
,2154,2160,2172,2180,2194,2195,2209,2228,2231,2237,2259,2266,2269,2272
,2283,2285,2289,2290,2303,2304,2305,2316,2318,2321,2323,2329,2330,2331
,2339,2345,2349,2354,2360,2361,2366,2376,2380,2387,2388,2391,2413,2415
,2416,2417,2420,2433,2435,2436,2441,2444,2445,2450,2454,2456,2461,2462
,2475,2485,2488,2493,2500,2505,2514,2516,2519,2524,2525,2527,2528,2531
,2537,2541 ]


# %%
selected_rows = df_test.iloc[empty_lines]

# %%
selected_rows['SUBCLASS'] = 'COAD'

# %%
selected_rows

# %%
ndf = pd.concat([data, selected_rows], ignore_index=True)
ndf


# %%
data = ndf.drop(columns=['ID', 'SUBCLASS'], axis=1)
data

# %%
total = pd.concat([data,df_test])
total

# %%

def process_cell(value, col_name):
    if value == 'WT':  # 값이 'WT'이면 빈 문자열
        return ''
    elif value == '':
        return ''
    elif isinstance(value, str):  # 문자열일 때는 첫 글자 + 컬럼명
        return  str(value[0]+ col_name)
    elif isinstance(value, (int, float)):  # 숫자일 때는 첫 숫자(소수점 전) + 컬럼명
        return  str(str(value)[0] + col_name)
    return value  # 그 외에는 원래 값 반환

# 각 셀을 처리하는 함수 적용
for col in total.columns:
    total[col] = total[col].apply(lambda x: process_cell(x, col))

# %%
# 데이터프레임을 셀 단위로 고유값으로 변환
unique_values, unique_indices = pd.factorize(total.values.ravel())

# 고유값을 다시 데이터프레임 모양으로 변환
total_transformed = pd.DataFrame(unique_values.reshape(total.shape), columns=total.columns)

print(total_transformed)

# %%


# %%
nRows = data.shape[0]
data = total_transformed[:nRows]
df_test = total_transformed[nRows:]

# %%
data

# %%
from sklearn.preprocessing import LabelEncoder
y = ndf['SUBCLASS']
encoder2 = LabelEncoder()
y = encoder2.fit_transform(y)

# %%
nRows = data.shape[0]
X = total_transformed[:nRows]
df_test = total_transformed[nRows:]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=156)#shuffle=False,stratify = y, 
print(X_train.shape, X_test.shape)

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
#X, y = make_classification(n_samples=1000, n_features=len(total.columns),                            n_informative=2, n_redundant=0,                           random_state=0, shuffle=False)
#clf = RandomForestClassifier(max_depth=2, random_state=0, n_jobs=-1)
clf = RandomForestClassifier(
    n_estimators=500,             # 트리의 개수 (너무 적으면 모델이 불안정하고, 너무 많으면 학습 시간이 오래 걸림)
    max_features='sqrt',          # 각 트리에서 선택할 피처의 수 (sqrt는 약 63개 피처를 사용, 피처가 많을 때 좋은 기본값)
    max_depth=None,               # 트리의 최대 깊이 (None이면 트리가 완전히 성장할 때까지 분할)
    min_samples_split=2,          # 노드를 분할하기 위한 최소 샘플 수 (default값)
    min_samples_leaf=1,           # 리프 노드에 있어야 하는 최소 샘플 수 (default값)
    bootstrap=True,               # 부트스트랩 샘플링을 사용할지 여부 (default는 True, 부트스트랩을 사용하면 트리 간의 다양성을 증가)
    n_jobs=-1,                    # 가능한 모든 CPU 코어를 사용하여 학습 (병렬 처리)
    random_state=42,              # 재현성을 위해 랜덤 시드 설정
    class_weight='balanced',      # 데이터 불균형일 경우 자동으로 가중치 조정
    verbose=1                     # 학습 과정에서 출력 메시지 (진행 상황을 확인할 수 있음)
    )
    
clf.fit(X, y) 

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
# 기본설정 랜덤포레스트
rf = RandomForestClassifier(n_jobs=-1) # 모든 코어 사용

# 피쳐중요도만 확인할 것이기 때문에 fit만 한다.
rf.fit(X_train, y_train)



# %%
# 그냥 확인하기에 피쳐종류가 30개나 되어 한눈에 들어오지 않는다.
print(clf.feature_importances_, '\n')

# 시리즈로 만들어 인덱스를 붙인다
ser = pd.Series(clf.feature_importances_, index=X_train.columns)  

# 내림차순 정렬을 이용한다
top15 = ser.sort_values(ascending=False)[:15]
print(top15)


# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
plt.title('Feature Importances Top 15')
sns.barplot(x=top15, y=top15.index)
plt.show()

# %%
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb 
from xgboost import XGBClassifier, plot_importance
from itertools import combinations


# %%
from sklearn.metrics import accuracy_score
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

# %%
total_transformed2 = total_transformed.iloc[:, top_features_idx]

# %%


# %%


# %%


# %%
tset = set(total_transformed2.columns)


# %%
cgs = pd.read_csv('../input/cancer-genes.csv')
cset =  set(cgs['c.genes'].values)
cset

# %%
len( cset - tset.intersection(cset))

# %%
left = cset - tset.intersection(cset)

# %%
columns_to_add = left
existing_columns = list(set(columns_to_add).intersection(total_transformed.columns))
total_transformed2[existing_columns] = total_transformed[existing_columns].values
total_transformed2

# %%
X

# %%
data

# %%
nRows = df.shape[0]
X = total_transformed2[:nRows]
df_test = total_transformed2[nRows:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=156) #shuffle=False,stratify = y, 
print(X_train.shape, X_test.shape)

# %%
dtrain = xgb.DMatrix(data = X_train, label= y_train)
dtest = xgb.DMatrix(data = X_test, label = y_test)

# %%
df.SUBCLASS.nunique()


# %%
params = {
    'tree_method': 'approx',
    'objective': 'multi:softprob',
    'num_class': df.SUBCLASS.nunique()
}
num_boost_round = 50

dtrain = xgb.DMatrix(data = X_train, label= y_train)
dtest = xgb.DMatrix(data = X_test, label = y_test)
model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round , early_stopping_rounds=5, 
                evals=[(dtrain, 'train'), (dtest, 'valid')],
                verbose_eval=2)

# %%
params = {
    'tree_method': 'approx',
    'objective': 'multi:softprob',
    'learning_rate': 0.05
}
num_boost_round = 1000

clf = xgb.XGBClassifier( n_estimators = 250,        # 부스팅 단계 수
                        max_depth = 8,             # 개별 트리의 최대 깊이
    #                  learning_rate = 0.1,       # 학습률, 낮출수록 느리게 학습하지만 성능 개선 가능
                        subsample=0.7,           # 각 부스팅 단계에서 사용할 샘플의 비율
                        #scale_pos_weight = class_weight_dict,  # 클래스별 가중치
                        colsample_bytree=0.3,    # 각 트리에서 사용할 특성의 비율
                        random_state = 42,         # 랜덤 시드
                        n_jobs = -1,             # 모든 CPU 코어 사용,
                        reg_alpha=0.03,
                        early_stopping_rounds = 50,                        
                        **params)
clf.fit(X_train, y_train, 
        eval_set=[(X_train, y_train), (X_test,y_test)],
        verbose=2);

# %%
y_true = y_test
y_pred = clf.predict(X_test)
y_score = clf.predict_proba(X_test)
y_true.shape, y_pred.shape, y_score.shape

# %%
from sklearn import metrics 
metrics.accuracy_score(y_true, y_pred)

# %%
metrics.roc_auc_score(y_true, y_score, average='weighted', multi_class='ovr')

# %%
res = clf.predict(df_test)
res = encoder2.inverse_transform(res)

df_test1 = pd.read_csv('../input/test.csv')
res_df = pd.DataFrame ({'ID' : df_test1['ID'], 'SUBCLASS' : res})
res_df


# %%
res_df.SUBCLASS.value_counts()

# %%
res_df.to_csv('res-COADplus-01.csv',index = False)

# %%


# %%


# %%


# %%


# %%



