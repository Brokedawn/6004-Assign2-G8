import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import roc_auc_score, average_precision_score

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Load your CSV file
data = pd.read_csv('static.csv')

# Preprocessing steps (you may adjust these based on your dataset)
data['gender'] = data['gender'].replace({'F': 1, 'M': 0}).astype(int)
data = data.drop(['hosp_admittime', 'hosp_dischtime', 'icu_intime', 'icu_outtime', 'race', 'admission_type', 'first_careunit',], axis=1)
data = data.fillna(data.mean())

print(data.columns)

# Read the dynamic-pre-process.csv into another DataFrame
df_dynamic = pd.read_csv('dynamic-pre-process.csv')
df_dynamic = df_dynamic.drop(['charttime'],axis=1)

print(df_dynamic.columns)

# 按'id'列分组，计算每个分组的平均值
df_dynamic = df_dynamic.groupby('id').mean()

# Merge data and df_dynamic based on the 'id' column
data = pd.merge(data, df_dynamic, on='id', how='left')

data = data.drop('id',axis=1)

print(data.columns)

# Extract features (X) and target (y)
X_raw = data.drop(columns=['icu_death'])  # Replace 'target_column_name' with the name of your target column
y_df = data['icu_death']  # Replace 'target_column_name' with the name of your target column

print(y_df.value_counts())

# 提取 'icu_death' 列为 0 和 1 的样本
class_0_indices = data[data['icu_death'] == 0].index
class_1_indices = data[data['icu_death'] == 1].index

# 随机抽取2000个样本
class_0_sampled_indices = np.random.choice(class_0_indices, size=2000, replace=False)
class_1_sampled_indices = np.random.choice(class_1_indices, size=2000, replace=False)

# 合并抽样得到的样本索引
sampled_indices = np.concatenate([class_0_sampled_indices, class_1_sampled_indices])

# 根据抽样得到的索引提取对应的样本
X_resampled = X_raw.loc[sampled_indices]
y_resampled = y_df.loc[sampled_indices]

y_df = y_resampled

print(y_df.value_counts())

# Standardize the features
X_df = X_resampled

# We convert dataframe to PyTorch tensor datatype,
# and then split it into training and testing parts.
X = torch.tensor(X_df.to_numpy(),dtype=torch.float32)
m,n = X.shape
y = torch.tensor(y_df.to_numpy(),dtype=torch.float32).reshape(m,1)

# We use an approx 6:4 train test splitting
cases = ['train','test']
case_list = np.random.choice(cases,size=X.shape[0],replace=True,p=[0.6,0.4])
X_train = X[case_list=='train']
X_test = X[case_list=='test']
y_train = y[case_list=='train']
y_test = y[case_list=='test']

model = LogisticRegression(penalty='l1', solver='liblinear')

# 使用 SequentialFeatureSelector 进行前向特征选择
forward_selection = SFS(model,
                         k_features=6,   # 选择的特征数量
                         forward=True,   # 前向选择
                         scoring='accuracy',  # 评分指标
                         cv=5)           # 交叉验证折数

# 在训练集上执行特征选择
forward_selection.fit(X_train.numpy(), y_train.numpy())

# 获取选中的特征索引
selected_feature_indices = forward_selection.k_feature_idx_

# 根据索引获取选中的特征
selected_features = X_df.columns[list(selected_feature_indices)]
print("FORWARD SELECTION Selected features:", selected_features)#['gender', 'admission_age', 'charlson_score', 'atrial_fibrillation','malignant_cancer', 'ckd', 'cld', 'copd', 'diabetes', 'hypertension','ihd', 'stroke']

