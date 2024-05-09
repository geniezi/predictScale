import os

import pandas as pd


def read_data(random):
    FILTER_DATA_DIR = 'data_' + random + '/'
    data = pd.read_csv(FILTER_DATA_DIR + 'result_dft.csv')
    return data


def create_folder_with_incrementing_number(base_path, base_folder_name):
    # 检查基础路径是否存在，如不存在则创建
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    folder_number = 0
    folder_path = os.path.join(base_path, f'{base_folder_name}{folder_number}')

    # 如果该文件夹已存在，增加序号直到找到未使用的序号
    while os.path.exists(folder_path):
        folder_number += 1
        folder_path = os.path.join(base_path, f'{base_folder_name}{folder_number}')

    # 创建新的文件夹
    os.makedirs(folder_path)
    return folder_path+'/'


