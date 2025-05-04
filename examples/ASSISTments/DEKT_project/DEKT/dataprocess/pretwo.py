

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
import csv


def load_and_preprocess_data(filepath):
    """加载并预处理数据"""
    data = pd.read_csv(
        filepath,
        usecols=['startTime', 'timeTaken', 'studentId', 'skill', 'problemId', 'problemType', 'correct', 'BORED', 'CONCENTRATING', 'CONFUSED', 'FRUSTRATED']
    ).dropna(subset=['skill', 'problemId']).sort_values('startTime')
    data['timeTaken'] = data['timeTaken'].astype(int)
    return data


def discretize_data(data):
    """将数据离散化为100类"""
    bins = [i * (1 / 100) for i in range(101)]
    labels = range(0, 100)
    data['FRU'] = pd.cut(data['FRUSTRATED'], bins=bins, labels=labels, include_lowest=True, right=False)
    data['CONF'] = pd.cut(data['CONFUSED'], bins=bins, labels=labels, include_lowest=True, right=False)
    data['CONC'] = pd.cut(data['CONCENTRATING'], bins=bins, labels=labels, include_lowest=True, right=False)
    data['BOR'] = pd.cut(data['BORED'], bins=bins, labels=labels, include_lowest=True, right=False)
    return data

def apply_mapping(data):
    """应用映射函数到指定列"""
    def map_to_class(value):
        return int(value * 100)
    data['FRU'] = data['FRUSTRATED'].apply(map_to_class)
    data['CONF'] = data['CONFUSED'].apply(map_to_class)
    data['CONC'] = data['CONCENTRATING'].apply(map_to_class)
    data['BOR'] = data['BORED'].apply(map_to_class)
    return data


def round_columns(data, columns, decimals=4):
    """对指定列的值保留指定的小数位数"""
    data[columns] = data[columns].round(decimals)
    return data


def save_data(data, filepath):
    """保存数据到CSV文件"""
    data.to_csv(filepath, index=False)


if __name__ == "__main__":
    # 文件路径
    input_filepath = '../../data/anonymized_full_release_competition_dataset/processed_data.csv'
    output_filepath_1 = '../../data/anonymized_full_release_competition_dataset/test.csv'
    output_filepath_2 = '../../data/anonymized_full_release_competition_dataset/test0.csv'

    # 加载和预处理数据
    data = load_and_preprocess_data(input_filepath)

    # 离散化数据
    data = discretize_data(data)

    # 保存第一次处理后的数据
    save_data(data, output_filepath_1)

    # 应用映射
    data = apply_mapping(data)

    # 保留指定列的小数位数
    columns_to_round = ['BORED', 'CONCENTRATING', 'CONFUSED', 'FRUSTRATED']
    data = round_columns(data, columns_to_round, decimals=4)

    # 保存最终处理后的数据
    save_data(data, output_filepath_2)

