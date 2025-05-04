
import pandas as pd

def load_data():
    """加载数据"""
    return pd.read_csv('../../data/anonymized_full_release_competition_dataset/processed_data.csv', encoding="ISO-8859-1", low_memory=False)

def print_data_length(data):
    """打印数据长度"""
    print(len(data))

def clean_data(data):
    """清理数据"""
    columns_to_round = ['BORED','CONCENTRATING','CONFUSED','FRUSTRATED']
    condition_to_delete = (data[columns_to_round] == 1.0).any(axis=1)
    return data[~condition_to_delete]

def save_data(data, file_path):
    """保存数据"""
    data.to_csv(file_path, index=False)

def main():
    """主函数"""
    data = load_data()
    print_data_length(data)
    data_cleaned = clean_data(data)
    print_data_length(data_cleaned)
    file_path = '../../data/anonymized_full_release_competition_dataset/test.csv'
    save_data(data_cleaned, file_path)

if __name__ == "__main__":
    main()




