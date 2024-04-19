import pandas as pd
import numpy as np
import torch
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA
import plotly.express as px

# Read the CSV file into a DataFrame
df = pd.read_csv('static-pre-process.csv')


# Convert 'icu_outtime' and 'icu_intime' columns to datetime format if needed
df['icu_outtime'] = pd.to_datetime(df['icu_outtime'])
df['icu_intime'] = pd.to_datetime(df['icu_intime'])

# Calculate ICU stay duration in hours and keep only the total number of hours
df['icu_hours'] = (df['icu_outtime'] - df['icu_intime']).dt.total_seconds() / 3600
df = df[df['icu_hours']<=168]

# Categorize ICU stay duration into three categories
df['icu_duration_category'] = pd.cut(df['icu_hours'], bins=[0,48,float('inf')], labels=[0, 1], include_lowest=True)


# Read the dynamic-pre-process.csv into another DataFrame
df_dynamic = pd.read_csv('dynamic-pre-process.csv')

# 删除 'charttime' 和 'icu_duration_category' 列
df_dynamic = df_dynamic.drop(['charttime'], axis=1)

# 按'id'列分组，计算每个分组的平均值
df_dynamic = df_dynamic.groupby('id').mean() #求平均后再加入

# Merge the 'icu_hours' column from DataFrame df into DataFrame df_dynamic based on the 'id' column
df_dynamic = df_dynamic.merge(df[['id', 'icu_duration_category']], on='id', how='left')

# Now df_dynamic contains the 'icu_hours' column from df merged based on 'id'
print(df_dynamic.columns)

data = df_dynamic.drop(['id'], axis=1)



# Extract features (X) and target (y)
X_raw = data.drop(columns=['icu_duration_category'])  # Replace 'target_column_name' with the name of your target column
y_df = data['icu_duration_category']

print(y_df.value_counts())

# 分别选取每个类别的样本
class_0_samples = data[data['icu_duration_category'] == 0].sample(n=6000, replace=True, random_state=42)
class_1_samples = data[data['icu_duration_category'] == 1].sample(n=6000, replace=True, random_state=42)

# 将抽样的数据帧进行拼接
sampled_data = pd.concat([class_0_samples, class_1_samples], ignore_index=True)

# 对抽样后的数据进行随机打乱
sampled_data = sampled_data.sample(frac=1, random_state=42).reset_index(drop=True)

# 验证抽样后数据中各类别的分布
print(sampled_data['icu_duration_category'].value_counts())

# 现在您可以使用 sampled_data 数据帧进行进一步的分析

X_df = sampled_data.drop(columns=['icu_duration_category'])
y_df = sampled_data['icu_duration_category']

pca = PCA(n_components=20)
pca.fit(X_df)

# Compute cumulative explained variance
cumulative_variance = pca.explained_variance_ratio_.cumsum()

import matplotlib.pyplot as plt

# 绘制累积方差的曲线图
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
plt.title('Cumulative Explained Variance by Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

# is better
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

# 定义随机森林模型
rf_model = RandomForestClassifier()

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# 定义网格搜索，减少交叉验证的折数为3
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3)

# 在训练集上拟合网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数:", grid_search.best_params_)

# 使用最佳参数的模型进行预测
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("调优后随机森林模型准确率:", accuracy)

# 计算AUC
y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print("调优后随机森林模型AUC:", auc)

import pandas as pd
import plotly.express as px

# 将预测结果和真实标签转换为 DataFrame
result_df = pd.DataFrame({'Prediction': y_pred, 'True Label': y_test})

# 使用交叉表比较真实标签和预测标签的分布情况
cross_table = pd.crosstab(index=result_df['True Label'], columns=result_df['Prediction'], margins=True)

# 打印交叉表
print("交叉表:")
print(cross_table)

import pandas as pd
import plotly.express as px

# 使用预测概率和真实标签创建 pred_df DataFrame
pred_df = pd.DataFrame({
    'pred_probability': y_pred_proba,
    'label': y_test
})

# 使用 Plotly Express 绘制散点图
fig = px.scatter(pred_df, y='pred_probability', color='label', title='预测概率与真实标签的关系')
fig.update_traces(marker=dict(size=10, opacity=0.8))
fig.update_layout(xaxis_title='Predicted Probability', yaxis_title='True Label')
fig.show()
