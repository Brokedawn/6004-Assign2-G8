import pandas as pd
import torch
import numpy as np
import plotly.express as px


from imblearn.over_sampling import SMOTE

# Load your CSV file
data = pd.read_csv('static.csv')

# Preprocessing steps (you may adjust these based on your dataset)
data['gender'] = data['gender'].replace({'F': 1, 'M': 0}).astype(int)
data = data.drop(['id', 'hosp_admittime', 'hosp_dischtime', 'icu_intime', 'icu_outtime', 'race', 'admission_type', 'first_careunit',], axis=1)
data = data.fillna(data.mean())

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
X_df = (X_resampled - X_resampled.mean()) / X_resampled.std()

X_df = X_df[['los_icu', 'admission_age', 'charlson_score', 'ckd', 'hypertension']] #forward

#X_df = X_df[['los_icu', 'admission_age', 'charlson_score', 'ckd', 'diabetes','hypertension', 'ihd']] #backward

# Convert dataframe to PyTorch tensor datatype,
# and then split it into training and testing parts.
X = torch.tensor(X_df.to_numpy(), dtype=torch.float32)
m, n = X.shape
y = torch.tensor(y_df.to_numpy(), dtype=torch.float32).reshape(m, 1)

# We use an approx 6:4 train test splitting
cases = ['train', 'test']
case_list = np.random.choice(cases, size=X.shape[0], replace=True, p=[0.6, 0.4])
X_train = X[case_list == 'train']
X_test = X[case_list == 'test']
y_train = y[case_list == 'train']
y_test = y[case_list == 'test']

h = torch.nn.Linear(
    in_features=n,
    out_features=1,
    bias=True
)
sigma = torch.nn.Sigmoid()

# Logistic model is linear+sigmoid
f = torch.nn.Sequential(
    h,
    sigma
)

J_BCE = torch.nn.BCELoss()
GD_optimizer = torch.optim.SGD(params=f.parameters(), lr=0.001)

nIter = 10000
printInterval = 1000

for i in range(nIter):
    GD_optimizer.zero_grad()
    pred = f(X_train)
    loss = J_BCE(pred, y_train)
    loss.backward()
    GD_optimizer.step()
    if i == 0 or ((i + 1) % printInterval) == 0:
        print('Iter {}: average BCE loss is {:.3f}'.format(i + 1, loss.item()))

# Test on test data
threshold = 0.5

with torch.no_grad():
    pred_test = f(X_test)

# 将预测值转换为二进制
binary_pred = (pred_test > threshold).float()
# 将标签转换为二进制
binary_label = (y_test > 0.5).float()

# 计算准确率
acc = (binary_pred == binary_label).float().mean().item()
print('Accuracy on test dataset is {:.2f}%'.format(acc * 100))

pd.crosstab(
    index=binary_label.squeeze(),
    columns=binary_pred.squeeze(),
    rownames=['Label'],
    colnames=['Pred']
)

# Use plotly express (px) to visualize test results.
# px expects DataFrame input

pred_df = pd.DataFrame(
    {
        'pred_probability': pred_test.squeeze(),
        'label': binary_label.squeeze()
    }
)
fig = px.scatter(data_frame=pred_df, y='pred_probability', color='label')
fig.show()


# 12 is better
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

# 定义随机森林模型
rf_model = RandomForestClassifier()

# 定义参数网格
param_grid = {
    'n_estimators': [300],
    'max_depth': [20],
    'min_samples_split': [2],
    'min_samples_leaf': [2]
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


# We use toy datasets in scikit-learn package
from sklearn.datasets import load_breast_cancer

# Tools in sklearn to select best model
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV

# Decision tree classifier in sklearn
from sklearn.tree import DecisionTreeClassifier as DTC, plot_tree

# We use f1 score to test model performance
from sklearn.metrics import f1_score

# Import matplotlib.pyplot to visualize tree models
import matplotlib.pyplot as plt

# We first build a shallow decision tree.
TreeModel = DTC(criterion='entropy',max_depth=1,random_state=15)
TreeModel.fit(X_train,y_train)

# Splitting rules can be visualized by using plot_tree in sklearn
plt.figure(figsize=(14,10))
plot_tree(
    TreeModel,
    filled=True,
    feature_names=X_df.columns,  # Updated to use the actual feature names
    class_names=['not ill','aid']
)
plt.show()

# The `max_depth` parameter is important for decision tree.
# We use `GridSearchCV` to select the best `max_depth`.

parameters = {'max_depth':np.arange(start=1,stop=10,step=1)}
print(parameters)

stratifiedCV = StratifiedKFold(n_splits=8)
TreeModel = DTC(criterion='entropy')
BestTree = GridSearchCV(
    TreeModel,
    param_grid=parameters,
    scoring='f1',
    cv=stratifiedCV
)
BestTree.fit(X_train,y_train)

print(BestTree.best_estimator_)

print(BestTree.best_score_)

y_pred = BestTree.predict(X_test)
print('F1 score on test set: {:.4f}'.format(f1_score(y_test,y_pred)))
pd.crosstab(y_test,y_pred)

from xgboost import XGBClassifier as XGBC

parameters = {
    'n_estimators':np.arange(start=2,stop=20,step=2),
    'max_depth':np.arange(start=2,stop=6,step=1),
    'learning_rate':np.arange(start=0.05,stop=0.4,step=0.05)
}

print(parameters)

stratifiedCV = StratifiedKFold(n_splits=8)
# XGBC: XGBoost classifier
XGBoostModel = XGBC()
BestXGBoost = GridSearchCV(
    XGBoostModel,
    param_grid=parameters,
    scoring='f1',
    cv=stratifiedCV,
    verbose=1,
    n_jobs=-1 # use all cpu cores to speedup grid search
)
BestXGBoost.fit(X_train,y_train)

print(BestXGBoost.best_params_)

print(BestXGBoost.best_score_)

y_pred = BestXGBoost.predict(X_test)
print('F1 score on test set: {:.4f}'.format(f1_score(y_test,y_pred)))
pd.crosstab(y_test,y_pred)

# Support Vector Classifier
from sklearn.svm import SVC

# 'C': strength of L2 regularization on linear SVM. Larger 'C' --> smaller regularization.
parameters = {
    'C':np.arange(start=1,stop=20,step=5)
}
stratifiedCV = StratifiedKFold(n_splits=8)
SVCModel = SVC(kernel='linear')
BestSVC = GridSearchCV(
    SVCModel,
    param_grid=parameters,
    scoring='f1',
    cv=stratifiedCV,
    verbose=1,
    n_jobs=-1
)
BestSVC.fit(X_train,y_train)

print(BestSVC.best_estimator_)

print(BestSVC.best_score_)

y_pred = BestSVC.predict(X_test)
print('F1 score on test set: {:.4f}'.format(f1_score(y_test,y_pred)))
pd.crosstab(y_test,y_pred)
auc_svc = roc_auc_score(y_test, y_pred)
print("SVM(LEANER) AUC:", auc_svc)

nonlinear_models = {
    'DecisionTree':DTC(criterion='entropy'),
    'XGBoost':XGBC(),
    'SVM_rbf':SVC(kernel='rbf')
}

stratifiedCV = StratifiedKFold(n_splits=8)


params = {
    'DecisionTree':{
        'max_depth':np.arange(start=1,stop=10)
    },
    'XGBoost':{
        'n_estimators':np.arange(start=2,stop=20,step=2),
        'max_depth':np.arange(start=2,stop=6),
        'learning_rate':np.arange(start=0.05,stop=0.4,step=0.05)
    },
    'SVM_rbf':{
        'C':np.arange(0.5,5,step=0.5)
    }
}

records = {}

for model in nonlinear_models:
    BestParams = GridSearchCV(
        nonlinear_models[model],
        param_grid = params[model],
        scoring='f1',
        cv=stratifiedCV,
        n_jobs=-1
    )
    BestParams.fit(X_train,y_train)
    records[model] = BestParams
    print('For {} cross validation F1 score is {:.4f}'.format(model,BestParams.best_score_))

from sklearn.metrics import accuracy_score

# Decision Tree
TreeModel.fit(X_train, y_train)
y_pred_tree = TreeModel.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print("Decision Tree Accuracy:", accuracy_tree)

# XGBoost
BestXGBoost.fit(X_train, y_train)
y_pred_xgboost = BestXGBoost.predict(X_test)
accuracy_xgboost = accuracy_score(y_test, y_pred_xgboost)
print("XGBoost Accuracy:", accuracy_xgboost)

# Support Vector Classifier
BestSVC.fit(X_train, y_train)
y_pred_svc = BestSVC.predict(X_test)
accuracy_svc = accuracy_score(y_test, y_pred_svc)
print("SVC Accuracy:", accuracy_svc)

from sklearn.metrics import roc_auc_score

# Decision Tree
y_prob_tree = TreeModel.predict_proba(X_test)[:, 1]
auc_tree = roc_auc_score(y_test, y_prob_tree)
print("Decision Tree AUC:", auc_tree)

# XGBoost
y_prob_xgboost = BestXGBoost.predict_proba(X_test)[:, 1]
auc_xgboost = roc_auc_score(y_test, y_prob_xgboost)
print("XGBoost AUC:", auc_xgboost)

# Support Vector Classifier
y_prob_svc = BestSVC.decision_function(X_test)
auc_svc = roc_auc_score(y_test, y_prob_svc)
print("SVC AUC:", auc_svc)
