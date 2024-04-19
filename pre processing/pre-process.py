import pandas as pd
from pandas.tseries.offsets import MonthEnd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

dynamic_df = pd.read_csv('dynamic.csv')
static_df = pd.read_csv('static.csv')

# transfer data-type
def parse_dates(date):
    try:
        return pd.to_datetime(date, format='%m/%d/%y %H:%M')
    except ValueError:
        try:
            return pd.to_datetime(date, format='%Y/%m/%d %H:%M:%S')
        except ValueError:
            return pd.NaT
def safe_change_year(dt, new_year):
    try:
        return dt.replace(year=new_year)
    except ValueError:
        # 如果日期无效设置为当月最后一天
        return dt.replace(year=new_year, day=1) + MonthEnd(1)

for column in static_df.select_dtypes(include=[np.number]).columns:
    median_value = static_df[column].median()
    static_df[column].fillna(median_value, inplace=True)

for column in static_df.select_dtypes(include=['object']).columns:
    mode_value = static_df[column].mode()[0]
    static_df[column].fillna(mode_value, inplace=True)

# 日期处理
def parse_dates(date):
    try:
        return pd.to_datetime(date, format='%m/%d/%y %H:%M')
    except ValueError:
        try:
            return pd.to_datetime(date, format='%Y/%m/%d %H:%M:%S')
        except ValueError:
            return pd.NaT

date_columns = ['hosp_admittime', 'hosp_dischtime', 'icu_intime', 'icu_outtime']
for column in date_columns:
    static_df[column] = parse_dates(static_df[column])

# 字符数据小写和去空格处理
text_columns = ['race']
for column in text_columns:
    static_df[column] = static_df[column].str.lower().str.strip()

# 正态化特定数值特征
normalization_features = [ 'los_icu','admission_age', 'weight_admit', 'height', 'charlson_score']
for feature in normalization_features:
    if feature in static_df.columns:
        static_df[feature] = zscore(static_df[feature].fillna(static_df[feature].mean()))

# 缩放除 'icu_death', 'id' 外的数值特征
numeric_cols = static_df.select_dtypes(include=[np.number]).columns.difference(['los_icu','icu_death', 'id'])
scaler = MinMaxScaler()
static_df[numeric_cols] = scaler.fit_transform(static_df[numeric_cols].fillna(0))

# 编码分类特征，性别的0-1编码
le = LabelEncoder()
static_df['race_encoded'] = le.fit_transform(static_df['race'])
static_df['gender_encoded'] = static_df['gender'].map({'M': 1, 'F': 0})

# 去除原始分类列
static_df.drop(['race', 'gender'], axis=1, inplace=True)

# 保存处理后的数据
static_df.to_csv('static-pre-process.csv', index=False)



for column in dynamic_df.select_dtypes(include=[np.number]).columns:
    median_value = dynamic_df[column].median()
    dynamic_df[column].fillna(median_value, inplace=True)

# 众数填充分类数据
for column in dynamic_df.select_dtypes(include=['object']).columns:
    mode_value = dynamic_df[column].mode()[0]
    dynamic_df[column].fillna(mode_value, inplace=True)

# 时间数据类型转换
date_columns = ['charttime']  # 例如假设 charttime 是动态数据集中的时间戳
for column in date_columns:
    dynamic_df[column] = dynamic_df[column].apply(parse_dates)

# 标准化数值数据
numeric_cols = dynamic_df.select_dtypes(include=[np.number]).columns.difference(['id'])
scaler = MinMaxScaler()
dynamic_df[numeric_cols] = scaler.fit_transform(dynamic_df[numeric_cols].fillna(0))

# 异常值处理
for col in numeric_cols:
    Q1 = dynamic_df[col].quantile(0.25)
    Q3 = dynamic_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    dynamic_df[col] = np.where((dynamic_df[col] < lower_bound) | (dynamic_df[col] > upper_bound),
                               np.nan, dynamic_df[col])
    dynamic_df[col].fillna(median_value, inplace=True)  # 再次填充处理后的异常值

# id 分组并按时间排序
dynamic_df = dynamic_df.sort_values(by=['id', 'charttime'])

# 导出处理后的动态数据
dynamic_df.to_csv('dynamic-pre-process.csv', index=False)
print("Processed dynamic data exported.")
