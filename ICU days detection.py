import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Read the CSV file into a DataFrame
df = pd.read_csv('static-pre-process.csv')

# Convert 'icu_outtime' and 'icu_intime' columns to datetime format if needed
df['icu_outtime'] = pd.to_datetime(df['icu_outtime'])
df['icu_intime'] = pd.to_datetime(df['icu_intime'])

# Calculate ICU stay duration in hours and keep only the total number of hours
df['icu_hours'] = (df['icu_outtime'] - df['icu_intime']).dt.total_seconds() / 3600

# Now you have the ICU stay duration in hours in the 'icu_hours' column

# Print first 10 rows of DataFrame df
print(df.head(10))

# Read the dynamic-pre-process.csv into another DataFrame
df_dynamic = pd.read_csv('dynamic-pre-process.csv')


# Merge the 'icu_hours' column from DataFrame df into DataFrame df_dynamic based on the 'id' column
df_dynamic = df_dynamic.merge(df[['id', 'icu_hours']], on='id', how='left')


# Now df_dynamic contains the 'icu_hours' column from df merged based on 'id'
print(df_dynamic.columns)

data = df_dynamic.drop(['charttime'], axis=1)

# 按'id'列分组，计算每个分组的平均值
data = data.groupby('id').mean()  #平均后加入（若没这句，则为全部加入）

#data = data.drop(['id'], axis=1)

# Extract features (X) and target (y)
X_raw = data.drop(columns=['icu_hours'])  # Replace 'target_column_name' with the name of your target column
y_df = data['icu_hours']  # Replace 'target_column_name' with the name of your target column



# 标准化数值列
scaler = StandardScaler()


cols = X_raw.columns
arr = scaler.fit_transform(X_raw)
#df_X = pd.DataFrame(data=arr, columns= cols)
df_X = X_raw


# Perform PCA
pca = PCA(n_components=40)
pca.fit(df_X)

# Compute cumulative explained variance
cumulative_variance = pca.explained_variance_ratio_.cumsum()

# Visualize cumulative explained variance
#import matplotlib.pyplot as plt

#plt.figure(figsize=(10, 6))
#plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
#plt.title('Cumulative Explained Variance by PCA Components')
#plt.xlabel('Number of Components')
#plt.ylabel('Cumulative Explained Variance')
#plt.grid(True)
#plt.show()





X_train, X_test, y_train, y_test = train_test_split(df_X, y_df, test_size=0.05)


#model = linear_model.LinearRegression() #159.08116420476426
#model = linear_model.Lasso(alpha=0.1) #153.2606583027587
#model = linear_model.Ridge(alpha=0.1) #164.3379821492509
#model= linear_model.ElasticNet(alpha=0.1) #150.68446002182984
#model = DecisionTreeRegressor() #222.64104607238997
model = RandomForestRegressor() #145.353653975415
#model = GradientBoostingRegressor(max_depth = 200, min_samples_split = 6, n_estimators = 300) #146.88797572289673
#model = LGBMRegressor() #152.19287734728914
#model = XGBRegressor(max_depth = 50,n_estimators = 300) #148.58006861141078
#model = AdaBoostRegressor() #329.31556600875075
#model = MLPRegressor() #147.03330763476612



model.fit(X_train, y_train)
preds = model.predict(X_test)
print(mean_squared_error(y_test, preds, squared=False))

