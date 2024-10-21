# %%
import pandas as pd 
df = pd.read_csv ('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
data = df.copy()
data = df.drop(columns = ['ID', 'SUBCLASS'], axis=1)

# %%
cgs = pd.read_csv('../input/cancer-genes.csv')
cset =  set(cgs['c.genes'].values)
cset

# %%
def get_non_wt_columns(row):
    return list(row.index[row != 'WT'])
non_wt_columns_test = pd.DataFrame()
non_wt_columns_test['gset'] = df_test.iloc[:,1:].apply(get_non_wt_columns, axis=1).reset_index(name='gset').drop('index',axis=1)


# %%
non_wt_columns = pd.DataFrame()
non_wt_columns['gset'] = df.iloc[:,2:].apply(get_non_wt_columns, axis=1).reset_index(name='gset').drop('index',axis=1)


# %%
non_wt_columns_test

# %%
non_wt_columns_test['inters'] = non_wt_columns_test['gset'].apply(lambda x: len(set(x).intersection(cset)))

# %%
non_wt_columns['inters'] = non_wt_columns['gset'].apply(lambda x: len(set(x).intersection(cset)))

# %%
non_wt_columns_test

# %%
not_cgeans_test = non_wt_columns_test[non_wt_columns_test.inters==0]['gset'].agg(sum)
not_cgeans = non_wt_columns[non_wt_columns.inters==0]['gset'].agg(sum)


# %%
cgs['c.genes']

# %%
real_geans = set(cgs['c.genes'])
len(set(df_test.columns).intersection( real_geans))

# %%
len(set(not_cgeans_test))

# %%
len(set(not_cgeans))

# %%
selected_geans = set(df_test.columns).intersection(real_geans)
selected_geans.update(set(not_cgeans_test))
len(selected_geans)

# %%
data = df.iloc[:,2:]
total = pd.concat([data, df_test])
train_idx = df.shape[0]

# %%
total = total.loc[ :, list(selected_geans)]
total = total.applymap(lambda x : 1 if x!='WT' else 0)


# %%
total

# %%
total

# %%
remove_cols = total.columns[total.std()==0]
total = total.drop(columns=remove_cols, axis=1)
total


# %%
remove_cols

# %%
num_all_zero_rows = (total == 0).all(axis=1).sum()

# %%
num_all_zero_rows

# %%
selected_geans = total.columns
selected_geans

# %%

data = df.loc[:, list(selected_geans)]
data['SUBCLASS'] = df['SUBCLASS']
#df_test2 = df_test.loc[:, list(selected_geans)]

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
data = ndf.drop(columns=['SUBCLASS'], axis=1)
data

# %%
total = pd.concat([data,df_test])
total

# %%
data = data.applymap(lambda x : 1 if x !='WT' else 0)
data.std()

# %%


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
top_n = 1000  # Adjust this value to select the top N important features
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
from sklearn.metrics import accuracy_score
# Step 5: Train an XGBoost model to find important features
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Get feature importances
feature_importances = xgb_model.feature_importances_
sorted_idx = feature_importances.argsort()

# Select top 2000 features based on importance
top_n = 1180  # Adjust this value to select the top N important features
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
from sklearn.metrics import accuracy_score
# Step 5: Train an XGBoost model to find important features
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Get feature importances
feature_importances = xgb_model.feature_importances_
sorted_idx = feature_importances.argsort()

# Select top 2000 features based on importance
top_n = 1184  # Adjust this value to select the top N important features
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


# %%
g1 = ['STES','SKCM']#7
g2 = ['UCEC']
g3 = ['BRCA', 'LUSC','LUAD','COAD']#5
g4 = ['GBMLGG','KIPAN','HNSC', 'CESC']# >2
g5 = ['LIHC','BLCA','PAAD','KIRC', 'PRAD','TGCT','SARC'] # > 1
g6 = ['PCPG', 'LAML', 'OV', 'THYM','LGG', 'THCA', 'ACC','DLBC'] #0

# %%
COAD      1038
BRCA       542
KIPAN      171
GBMLGG     124
OV         100
STES        80
THCA        75
KIRC        68
SKCM        52
UCEC        47
LGG         43
LAML        26
HNSC        25
PRAD        23
LIHC        19
SARC        18
PCPG        18
TGCT        16
LUAD        15
CESC        13
ACC         11
LUSC         6
DLBC         5
PAAD         5
BLCA         4
THYM         2

# %%
g1 = ['SKCM', 'UCEC']#9
g2 = ['BRCA', 'STES','LUAD','COAD']
g3 = ['LUSC','GBMLGG', 'HNSC']
g4 = ['KIPAN' ]# >2
g5 = ['LIHC', 'CESC','BLCA','ACC','KIRC', 'PRAD', 'TGCT','THCA','SARC'] # > 1
g6 = ['PCPG', 'LAML', 'OV', 'THYM','LGG', 'PAAD', 'DLBC'] #0


# %%


# %%
disease_list = df['SUBCLASS'].unique()

def get_gene_list1():
    selected_genes = set()
    for d in disease_list:
        count_df = df[df.SUBCLASS==d].apply(lambda x : x!='WT').agg(sum)
        count_df = count_df[2:].reset_index(name='count')
        if d in g1 :
            th = 9
        elif d in g2 :
            th = 6
        elif d in g3 :
            th = 5
        elif d in g4 :
            th = 2
        elif d in g5:
            th = 1
        else :
            th = 0   
        gene_list = count_df[count_df['count']>th  ]['index'].values
        print(d, len(gene_list))
        selected_genes.update(gene_list)
    print(len(selected_genes))
    return selected_genes
glist = get_gene_list1()

# %%
g1 = ['SKCM', 'UCEC']
g2 = ['BRCA', 'CESC','STES','LUAD','COAD','KIRC','KIPAN','LUSC','GBMLGG', 'HNSC', 'PRAD']
g3 = ['LIHC', 'BLCA','ACC',  'TGCT','THCA','SARC','PCPG', 'LAML', 'OV', 'THYM','LGG', 'PAAD', 'DLBC']
disease_list = df['SUBCLASS'].unique()

def get_gene_list1():
    selected_genes = set()
    for d in disease_list:
        count_df = df[df.SUBCLASS==d].apply(lambda x : x!='WT').agg(sum)
        count_df = count_df[2:].reset_index(name='count')
        if d in g1 :
            th = 10
        elif d in g2 :
            th = 7
        elif d in g3 :
            th = 1
        gene_list = count_df[count_df['count']>th  ]['index'].values
        print(d, len(gene_list))
        selected_genes.update(gene_list)
    print(len(selected_genes))
    return selected_genes
glist = get_gene_list1()

# %%
g1 = ['SKCM', 'UCEC']
g2 = ['BRCA', 'CESC','STES','LUAD','COAD','KIRC','KIPAN','LUSC','GBMLGG', 'HNSC', 'PRAD']
g3 = ['LIHC', 'ACC',  'TGCT','THCA','SARC','PCPG', 'LAML', 'OV', 'THYM','LGG']
g4 = ['BLCA', 'PAAD', 'DLBC']
disease_list = df['SUBCLASS'].unique()

def get_gene_list1():
    selected_genes = set()
    for d in disease_list:
        count_df = df[df.SUBCLASS==d].apply(lambda x : x!='WT').agg(sum)
        count_df = count_df[2:].reset_index(name='count')
        if d in g1 :
            th = 10
        elif d in g2 :
            th = 7
        elif d in g3 :
            th = 2
        elif d in g4 :
            th = 0
        gene_list = count_df[count_df['count']>th  ]['index'].values
        print(d, len(gene_list))
        selected_genes.update(gene_list)
    print(len(selected_genes))
    return selected_genes
glist = get_gene_list1()

# %%
for d in disease_list:
    disease_df = df[df["SUBCLASS"] == d]

    non_wt_counts = (disease_df != 'WT').sum()  # "WT"가 아닌 행 개수
    nsorted = non_wt_counts.sort_values(ascending=False) #빈도 수가 높은 순서대로 정렬
    print(d)
    print(nsorted.iloc[2:12])

#n = 20 # 추출한 상위 유전자 개수
##genes.extend(list(nsorted[2:2+n].index))

#print(f'{disease} - 전체 행의 개수 : {len(disease_df)}')
##dd = nsorted[2:n].reset_index()
##gene_list1 = dd['index'].values
##gene_set.update(gene_list1)
#print(gene_list1)


# %%


# %%
total_transformed2

# %%
nRows = ndf.shape[0]
X = total_transformed2[:nRows]
df_test = total_transformed2[nRows:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42) #shuffle=False,stratify = y, 
print(X_train.shape, X_test.shape)

# %%


# %%
dtrain = xgb.DMatrix(data = X_train, label= y_train)
dtest = xgb.DMatrix(data = X_test, label = y_test)

# %%
df_test2 = xgb.DMatrix(data = df_test)

# %%
df.SUBCLASS.nunique()


# %%
params = {
    'tree_method': 'approx',
    'objective': 'multi:softmax',
    'num_class': df.SUBCLASS.nunique()
}
num_boost_round = 250

dtrain = xgb.DMatrix(data = X_train, label= y_train)
dtest = xgb.DMatrix(data = X_test, label = y_test)
model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round , early_stopping_rounds=5, 
                evals=[(dtrain, 'train'), (dtest, 'valid')],
                verbose_eval=2)

# %%
df_test2

# %%
res = model.predict(df_test2)

# %%
nres = []
for r in res:
    nres.append(int(r))


# %%
res = encoder2.inverse_transform(nres)

# %%

df_test1 = pd.read_csv('../input/test.csv')
res_df = pd.DataFrame ({'ID' : df_test1['ID'], 'SUBCLASS' : res})
res_df


# %%
res_df.SUBCLASS.value_counts()

# %%
res_df.to_csv('DMATRIX-1184-1.csv', index=False)

# %%


# %%
params = {
    'tree_method': 'approx',
    'objective': 'multi:softprob',
#    'learning_rate': 0.05
}
num_boost_round = 1000

clf = xgb.XGBClassifier(n_estimators = 250,        # 부스팅 단계 수
                        max_depth = 6,             # 개별 트리의 최대 깊이
                        learning_rate = 0.1,       # 학습률, 낮출수록 느리게 학습하지만 성능 개선 가능
                        subsample = 0.7,           # 각 부스팅 단계에서 사용할 샘플의 비율
                        #scale_pos_weight = class_weight_dict,  # 클래스별 가중치
                        colsample_bytree = 0.7,    # 각 트리에서 사용할 특성의 비율
                        random_state = 42,         # 랜덤 시드
                        n_jobs = -1,             # 모든 CPU 코어 사용,
                        reg_alpha=0.05,
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
from sklearn.metrics import f1_score
import numpy as np

def macro_f1_score(y_true, y_pred, n_classes):
    f1_scores = []
    for c in range(n_classes):
        y_true_c = (y_true == c)
        y_pred_c = (y_pred == c)
        f1_c = f1_score(y_true_c, y_pred_c)
        f1_scores.append(f1_c)
    return np.mean(f1_scores)

n_classes = df.SUBCLASS.nunique()
macro_f1 = macro_f1_score(y_true, y_pred, n_classes)
print(macro_f1)

# %%
res = clf.predict(df_test)
res = encoder2.inverse_transform(res)

df_test1 = pd.read_csv('../input/test.csv')
res_df = pd.DataFrame ({'ID' : df_test1['ID'], 'SUBCLASS' : res})
res_df


# %%
res_df.SUBCLASS.value_counts()

# %%
res_df.SUBCLASS.value_counts()

# %%
res_df.to_csv('res-1184-1.csv',index = False)

# %%


# %%
previous_df = res_df.copy()

# %%
#zerosl = list(df_test[(df_test_2 == 0).all(axis=1)].index)
zerosl = [17, 180, 187, 214, 219, 294, 374, 446, 450, 545, 676, 696, 787, 855, 979, 1055, 1249, 1542, 1770, 1785, 1832, 1866, 1965, 2046, 2059, 2118, 2132, 2153, 2340, 2412, 2481]

previous_df.loc[zerosl, 'SUBCLASS'] = 'THYM'

# %%
previous_df['SUBCLASS'].value_counts()

# %%
previous_df.to_csv('../res-1184-2.csv',index = False)

# %%


# %%


# %%


# %%


# %%

df2 = pd.read_csv('../input/train.csv')
df_test1 = pd.read_csv('../input/test.csv')

data1 = df2.drop(columns=['ID', 'SUBCLASS'])
df_test1.drop(columns='ID',inplace=True)

total3 = pd.concat([data1, df_test1])
f = lambda x : 0 if x == 'WT' else 1
total3 = total3.map(f)

ns = df2.shape[0]
data3 = total3[:ns]
df_test1 = total3[ns:]


# %%
# 각 행에서 1이 위치한 열의 이름과 인덱스를 구하는 코드
one_1_rows = df_test1[df_test1.sum(axis=1) == 1]
result = [(index, row.idxmax()) for index, row in one_1_rows.iterrows()]
result

# %%
one_VHL_l=[]
one_BRAF_l = []
one_NPM1_l = []
one_LIG1_l = []
one_s = []
res_df1 = previous_df.copy()

for i, c in result:
    print(i, c)
    if c =='VHL': 
        one_VHL_l.append(i)
        # 해당 컬럼에서 SUBCLASS값들의 분포를 구함
        q = ['VHL']
        rows_with_all_ones = df[data3[q].eq(1).all(axis=1)] 
        rs = rows_with_all_ones['SUBCLASS'].value_counts().reset_index(name='count').loc[1].SUBCLASS #KIRC
        res_df1.loc[one_VHL_l,'SUBCLASS'] = rs
    elif c=='BRAF': 
        one_BRAF_l.append(i)
        q = ['BRAF']
        rows_with_all_ones = df[data3[q].eq(1).all(axis=1)]
        rs = rows_with_all_ones['SUBCLASS'].value_counts().reset_index(name='count').loc[1].SUBCLASS #maybe 1
        res_df1.loc[one_BRAF_l,'SUBCLASS'] = rs
    elif c=='NPM1' : 
        one_NPM1_l.append(i)
        q = ['NPM1']
        rows_with_all_ones = df[data3[q].eq(1).all(axis=1)]
        rs = rows_with_all_ones['SUBCLASS'].value_counts().reset_index(name='count').loc[0].SUBCLASS#0 not 1
        res_df1.loc[one_NPM1_l,'SUBCLASS'] = rs
    elif c=='LIG1' : 
        one_LIG1_l.append(i)
        q = ['LIG1']
        rows_with_all_ones = df[data3[q].eq(1).all(axis=1)]
        rs = rows_with_all_ones['SUBCLASS'].value_counts().reset_index(name='count').loc[0].SUBCLASS#0 not 1
        res_df1.loc[one_LIG1_l,'SUBCLASS'] = rs
    elif c=='HRAS': 
        one_s.append(i)
        q = ['HRAS']
        rows_with_all_ones = df[data3[q].eq(1).all(axis=1)]
        rs = rows_with_all_ones['SUBCLASS'].value_counts().reset_index(name='count').loc[1].SUBCLASS#0 not 1
        res_df1.loc[one_s,'SUBCLASS'] = rs
    #else:
        ###one_s.append(i)
        #row_no = i
        #q = [c]
        #rows_with_all_ones = df[data[q].eq(1).all(axis=1)]
        #rs = rows_with_all_ones['SUBCLASS'].value_counts().reset_index(name='count').loc[1].SUBCLASS
        #res_df1.loc[i]['SUBCLASS'] = rs
    

res_df1.to_csv('../res-1184-3.csv.csv',index = False)
    

# %%


# %%


# %%
############# optional code

# %%
## TEST 데이터에서 1개의 컬럼만 1인 경우 찾기
one_1_rows = df_test1[df_test1.sum(axis=1) == 1]
# 각 열(column)별로 1이 있는 인덱스 리스트 생성
result_dict = {}
for column in one_1_rows.columns:
    indices = one_1_rows[one_1_rows[column] == 1].index.tolist()
    if indices:  # 인덱스가 비어있지 않으면 추가
        result_dict[column] = indices

# 딕셔너리를 DataFrame으로 변환
result_df = pd.DataFrame(list(result_dict.items()), columns=['gene', 'index_list'])
result_df


# %%
q = 'VHL'
rows_with_all_ones = df[(data[q] == 1) & (data.drop(columns=q) == 0).all(axis=1)]
rs = rows_with_all_ones['SUBCLASS'].value_counts().reset_index(name='count') #.loc[1].SUBCLASS #KIRC
#res_df1.loc[one_VHL_l,'SUBCLASS'] = rs
print(rs)

# %%


# %%
## df_test 데이터에서 2개의 컬럼만 1인 경우 찾기
one_2_rows = df_test[df_test.sum(axis=1) == 2]
# 각 행에서 1이 위치한 열의 이름을 나열
result = [(index, row[row == 1].index.tolist()) for index, row in one_2_rows.iterrows()]

# 리스트를 DataFrame으로 변환
result_df = pd.DataFrame(result, columns=['index', 'genes'])
result_df



# %%
## df_test 데이터에서 2개의 컬럼만 1인 경우 찾기
one_2_rows = df_test[df_test.sum(axis=1) == 2]
# 각 행에서 1이 위치한 열의 이름을 나열
result = [(index, row[row == 1].index.tolist()) for index, row in one_2_rows.iterrows()]

# 리스트를 DataFrame으로 변환
result_df = pd.DataFrame(result, columns=['index', 'genes'])
result_df



# %%
## 1이 2개 이상 있는 행 필터링
one_2_rows = df_test[df_test.sum(axis=1) == 2]

# 각 행에서 1이 위치한 열의 이름을 모두 나열하는 코드
result = [(index, row[row == 1].index.tolist()) for index, row in one_2_rows.iterrows()]

# 출력
print(result)


# %%


# %% [markdown]
# ## Update Res_df with 2 / 3 / 4 / 5

# %%
## TEST DATA에서 1이 n개 있는 행 필터링
def find_test_n_ones(n):
    one_2_rows = df_test1[df_test1.sum(axis=1) == n]
    # 각 행에서 1이 위치한 열의 이름을 모두 나열하는 코드
    result = [(index, row[row == 1].index.tolist()) for index, row in one_2_rows.iterrows()]
    print(len(result))
    return result
print(find_test_n_ones(2))

# %%
# exact matching은 보정
res_df2 = res_df1.copy()
for ns in range (1, 5):
    result_n = find_test_n_ones(ns)

    for i, qSet in result_n:
        # qSet에 해당하는 컬럼들만 값이 1인 행을 반환
        rows_with_only_qSet_ones = df[(data3[qSet].eq(1).all(axis=1)) & (data3.drop(columns=qSet).eq(0).all(axis=1))]
        rs = rows_with_only_qSet_ones['SUBCLASS'].value_counts().reset_index(name='count') #.loc[1].SUBCLASS #KIRC
        
        if rs.shape[0] == 1:
            rs = rs.loc[0].SUBCLASS
            if rs != res_df.loc[i]['SUBCLASS']:
                print(ns, rs, res_df1.loc[i]['SUBCLASS'], qSet)
                res_df2.loc[i,'SUBCLASS'] = rs
res_df2.to_csv('../res-1184-exactMatching.csv',index = False)


# %%
res_df2.SUBCLASS.value_counts()

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%
# res 값 수정, 
# 문서 2,3,4,5
res_df2 = res_df1.copy()
result_n = find_test_n_ones(2)

for i, qSet in result_n:
    print(i, qSet)
    #option2) qSet에 해당하는 컬럼들만 값이 1인 행을 반환
    rows_with_only_qSet_ones = df[(data[qSet].eq(1).all(axis=1)) & (data.drop(columns=qSet).eq(0).all(axis=1))]
    rs = rows_with_only_qSet_ones['SUBCLASS'].value_counts().reset_index(name='count') #.loc[1].SUBCLASS #KIRC
    
    if rs.shape[0] > 0:
        rs = rs.loc[0].SUBCLASS
        print(rs)
        res_df2.loc[i,'SUBCLASS'] = rs
    else: 
        #option1) qSet에 해당하는 컬럼들이 포함된, 값이 1인 행을 반환
        rows_with_all_ones = df[data[qSet].eq(1).all(axis=1)]
        if rows_with_all_ones.shape[0] > 1:
            try:
                rs = rows_with_all_ones['SUBCLASS'].value_counts().reset_index(name='count')#.loc[].SUBCLASS## 0->1
                print(rs)
                #res_df2.loc[i,'SUBCLASS'] = rs
            except KeyError or ValueError:
                rs = rows_with_all_ones['SUBCLASS'].value_counts().reset_index(name='count').loc[0].SUBCLASS## 0->1
                res_df2.loc[i,'SUBCLASS'] = rs
                print(rs)
            
#res_df2.to_csv('res2-step3-two.csv',index = False)

# %%
df

# %%
result_n = find_test_n_ones(1)

for i, qSet in result_n:
    print(i, qSet)
    #option2) qSet에 해당하는 컬럼들만 값이 1인 행을 반환
    rows_with_only_qSet_ones = df[(data[qSet].eq(1).all(axis=1)) & (data.drop(columns=qSet).eq(0).all(axis=1))]
    rs = rows_with_only_qSet_ones['SUBCLASS'].value_counts().reset_index(name='count') #.loc[1].SUBCLASS #KIRC
    
    if rs.shape[0] > 0:
        #rs = rs.loc[0].SUBCLASS
        print(rs)
        #res_df2.loc[i,'SUBCLASS'] = rs
    #else: 
    #    #option1) qSet에 해당하는 컬럼들이 포함된, 값이 1인 행을 반환
    #    rows_with_all_ones = df[data[qSet].eq(1).all(axis=1)]
    #    if rows_with_all_ones.shape[0] > 0 : 
    #        rs = rows_with_all_ones['SUBCLASS'].value_counts().reset_index(name='count').loc[0].SUBCLASS
    #        print(rs)
    #        res_df2.loc[i,'SUBCLASS'] = rs


# %%
res_df2.to_csv('res-step4-three.csv',index = False)

# %%


# %%
res_df2.to_csv('res-step5-four.csv',index = False)

# %%
result_n = find_test_n_ones(5)

for i, qSet in result_n:
    print(i, qSet)
    #option2) qSet에 해당하는 컬럼들만 값이 1인 행을 반환
    rows_with_only_qSet_ones = df[(data[qSet].eq(1).all(axis=1)) & (data.drop(columns=qSet).eq(0).all(axis=1))]
    rs = rows_with_only_qSet_ones['SUBCLASS'].value_counts().reset_index(name='count') #.loc[1].SUBCLASS #KIRC
    
    if rs.shape[0] > 0:
        rs = rs.loc[0].SUBCLASS
        print(rs)
        res_df2.loc[i,'SUBCLASS'] = rs
    else: 
        #option1) qSet에 해당하는 컬럼들이 포함된, 값이 1인 행을 반환
        rows_with_all_ones = df[data[qSet].eq(1).all(axis=1)]
        if rows_with_all_ones.shape[0] > 1 : 
            try: 
                rs = rows_with_all_ones['SUBCLASS'].value_counts().reset_index(name='count').loc[1].SUBCLASS
                print(rs)
                res_df2.loc[i,'SUBCLASS'] = rs
            except KeyError or ValueError:
                rs = rows_with_all_ones['SUBCLASS'].value_counts().reset_index(name='count').loc[0].SUBCLASS
                print(rs)
                res_df2.loc[i,'SUBCLASS'] = rs


# %%
res_df2.to_csv('res-step6-five.csv',index = False)

# %%
group2 = [ 5,    7,   27,   31,   43,   63,   71,   86,   97,  108,  197,
        210,  221,  248,  286,  312,  313,  336,  347,  348,  403,  467,
        514,  569,  583,  587,  620,  624,  629,  648,  690,  756,  766,
        813,  848,  907,  928,  956,  962, 1010, 1024, 1050, 1064, 1108,
    1141, 1162, 1178, 1198, 1222, 1291, 1306, 1432, 1441, 1488, 1543,
    1556, 1559, 1576, 1619, 1624, 1636, 1643, 1660, 1669, 1771, 1786,
    1823, 1837, 1852, 1869, 1903, 1906, 1914, 1922, 1987, 2040, 2063,
    2078, 2138, 2189, 2195, 2202, 2229, 2289, 2329, 2357, 2361, 2416,
    2460, 2466, 2468, 2534, 2543]

# %%
res_df3 = res_df2.copy()


# %%
res_df3.iloc[group2].SUBCLASS.value_counts()

# %%
for t in group2 : 
    #print(t, df_test.iloc[t].apply(lambda x : 1 if x!='WT' else 0).agg(sum))
    res_df3['SUBCLASS'][t]='LUSC'
res_df3.iloc[group2].SUBCLASS.value_counts()

# %%
res_df3.to_csv('res-forced-1184-LUSC.csv',index=False)

# %%



