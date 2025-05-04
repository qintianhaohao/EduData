import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 常量化文件路径
INPUT_FILE_PATH = '../../data/anonymized_full_release_competition_dataset/anonymized_full_release_competition_dataset.csv'
OUTPUT_FILE_PATH = '../../data/anonymized_full_release_competition_dataset/processed_data.csv'


def process_data(input_path, output_path):
    """
    处理原始数据集，重命名字段，重新排序列，并保存处理后的数据。

    参数:
        input_path (str): 输入CSV文件的路径
        output_path (str): 输出CSV文件的路径
    """
    # 设置 Pandas 和 Numpy 的显示选项
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    np.set_printoptions(suppress=True)

    # 加载数据
    data = pd.read_csv(input_path, encoding="ISO-8859-1", low_memory=False)

    # 重命名字段以提高可读性
    data['BORED'] = data['confidence(BORED)']
    data['CONCENTRATING'] = data['confidence(CONCENTRATING)']
    data['CONFUSED'] = data['confidence(CONFUSED)']
    data['FRUSTRATED'] = data['confidence(FRUSTRATED)']

    # 重新排序列
    columns_order = [
        'startTime', 'timeTaken', 'studentId', 'skill', 'problemId','problemType', 'correct','BORED','CONCENTRATING','CONFUSED','FRUSTRATED'
    ]
    data = data[columns_order]

    # 保存处理后的数据
    data.to_csv(output_path, index=False)
    print("数据处理完成并已保存至:", output_path)

    return data


def main():
    """
    主函数，负责调用数据处理函数并打印结果。
    """
    # 执行数据处理
    data = process_data(INPUT_FILE_PATH, OUTPUT_FILE_PATH)

    # 打印结果
    print(data.isnull().sum())
    print(data.info())
    print(data)


# 确保代码仅在直接运行时执行
if __name__ == "__main__":
    main()
