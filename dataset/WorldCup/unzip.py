import gzip
import os
import struct

import pandas as pd


# 定义函数读取 gzip 文件并按分钟统计请求量
def read_gz_file(file_path):
    # 创建一个空列表用于存储每分钟的请求量和时间戳
    data = []

    # 打开 gzip 文件，使用二进制模式
    with gzip.open(file_path, 'rb') as f:
        # 循环读取文件内容
        while True:
            # 从文件中读取固定长度的数据块，确保每次读取的长度正确
            chunk = f.read(20)

            # 如果读取到了文件末尾，则停止循环
            if not chunk:
                break

            # 解析请求数据
            timestamp, clientID, objectID, size, method, status, file_type, server = struct.unpack('>IIII4B', chunk)

            # 将时间戳转换成分钟级别
            minute_timestamp = timestamp // 60

            # 添加到数据列表中
            data.append((minute_timestamp, 1))  # 每个请求计数为1

    return data


# 遍历路径下的所有.gz文件
def get_file_paths(path):
    file_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.tsv'):
                file_paths.append(os.path.join(root, file))
    return file_paths


# file_paths = get_file_paths('dataset/WorldCup')
# # 指定文件路径
# for file_path in file_paths:
#     print(file_path)
#     data=read_gz_file(file_path)
#     # 将数据转换为 DataFrame
#     df = pd.DataFrame(data, columns=['Time', 'Requests'])
#
#     # 将时间戳转换为可读的时间格式
#     df['Time'] = pd.to_datetime(df['Time'] * 60, unit='s')
#
#     # 按时间进行分组并计算请求量
#     df = df.groupby('Time').sum().reset_index()
#
#     # 将数据保存到 Excel 文件
#     excel_file = file_path.replace('.gz', '.csv')
#     df.to_csv(excel_file, sep='\t',)
#
#     print(f'Data saved to {excel_file}')


# def merge_by_day(file_paths):
#     # 根据日期合并csv数据，文件名为日期_1,2,3,4,5,6,7.csv
#     for day in range(1,93):
#         df = pd.DataFrame()
#         for file_path in file_paths:
#             if file_path.find(f'_day{day}_') != -1:
#                 data = pd.read_csv(file_path, sep='\t')
#                 data=data[['Time','Requests']]
#                 # 合并数据，两个数据time可能有重复，需要求和
#                 df = pd.concat([df, data]).groupby('Time').sum().reset_index()
#         # 截取一个日期并保存
#         df['Time1'] = pd.to_datetime(df['Time']).dt.date
#         # 截取Time中的时间，按照时间排序
#         df['Time'] = pd.to_datetime(df['Time']).dt.time
#         df = df.sort_values(by='Time')
#         # 转换回带日期的时间
#         df['Time'] = df['Time1'].astype(str) + ' ' + df['Time'].astype(str)
#         df = df.drop('Time1', axis=1)
#
#         df.to_csv(f'dataset/WorldCup/{day}.tsv', sep='\t', index=False)
#

def merge_all_in_one(file_paths):
    # 合并所有数据
    df = pd.DataFrame()
    for file_path in file_paths:
        data = pd.read_csv(file_path, sep='\t')
        # 合并数据,将两个对象的Time合并为一列
        df = pd.concat([df, data]).groupby('Time').sum().reset_index()

    # 转换成时间戳
    df['timestamp']=pd.to_datetime(df['Time']).astype(int)/10**9
    df = df.sort_values(by='timestamp')
    df.drop('timestamp', axis=1, inplace=True)


    df.to_csv('dataset/WorldCup/All.tsv', sep='\t', index=False)

file_paths = get_file_paths('dataset/WorldCup')
merge_all_in_one(file_paths)
# merge_by_day(file_paths)
