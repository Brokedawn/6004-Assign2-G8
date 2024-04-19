import pandas as pd


data = pd.read_csv('pre-process.csv')

data['charttime'] = pd.to_datetime(data['charttime'], errors='coerce')

current_year = pd.Timestamp('2024-01-01')

future_dates_count = data[data['charttime'] > current_year].shape[0]
normal_dates_count = data[data['charttime'] <= current_year].shape[0]
total_dates_count = data['charttime'].notna().sum()  # 只统计非空日期

future_dates_ratio = future_dates_count / total_dates_count
normal_dates_ratio = normal_dates_count / total_dates_count

print(f"未来日期数量: {future_dates_count}")
print(f"正常日期数量: {normal_dates_count}")
print(f"未来日期比例: {future_dates_ratio:.2%}")
print(f"正常日期比例: {normal_dates_ratio:.2%}")

race_counts = data['admission_type'].value_counts()
print(race_counts)
