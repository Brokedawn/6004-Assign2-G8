import pandas as pd
import numpy as np
import re


df = pd.read_csv('notes.csv')


def clean_extracted_text(text):
    lines = text.split('\n')

    cleaned_lines = []

    for line in lines:
        # 检查每一行是否存在以大写字母开头的标题（冒号结尾）
        # 去除标题
        new_line = re.sub(r'^[A-Z\s]+:', '', line).strip()
        # 如果行不为空添加到清洗后的文本列表中
        if new_line:
            cleaned_lines.append(new_line)

    # 合并剩余的行为一个文本块
    text = '\n'.join(cleaned_lines)

    # 去除URLs
    text = re.sub(r'http\S+', '', text)
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 去除特殊字符保留换行符以便保持行分隔
    text = re.sub(r'[^\w\s]', '', text)
    # 去除下划线
    text = text.replace('_', '')
    # 转换所有大写字母为小写
    cleaned_text = text.lower()

    return cleaned_text


df['charttime'] = pd.to_datetime(df['charttime'], format='%m/%d/%y %H:%M')
df['text'] = df['text'].apply(clean_extracted_text)


df.to_csv('cleaned_notes.csv', index=False)



