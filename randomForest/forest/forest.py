import itertools
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import util.osUtil as osUtil


def forest(columns):
    # tuple转list
    columns = list(columns)
    x_train = train_data[columns]
    y_train = train_data['task_type']
    # X_test = test_data.drop('gpu_type', axis=1)
    x_test = test_data[columns]
    y_test = test_data['task_type']
    # 建立随机森林模型
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train)
    # 进行预测
    y_pred = rf_model.predict(x_test)
    # 基础路径至哪里创建文件夹
    base_path = result_dir  # 替换为需要创建文件夹的路径
    base_folder_name = '1'  # 基础文件夹名
    # 创建文件夹并返回新建文件夹的路径
    new_folder_path = osUtil.create_folder_with_incrementing_number(base_path, base_folder_name)
    # 保存性能指标、train_col
    with open(new_folder_path + 'classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))
        f.write('\n')
        f.write(str(columns))
    # 如果需要，可以保存模型和label_encoders以便于后续的使用
    # 这里可以利用joblib或pickle保存
    joblib.dump(rf_model, new_folder_path + 'rf_task_type_model.pkl')
    for col, le in label_encoders.items():
        joblib.dump(le, new_folder_path + f'{col}_encoder.pkl')
    #     打印完成信息
    print(f'已完成：{columns}')


# 加载数据
# random = 'random'
random = 'random_random'
FILTER_DATA_DIR = 'data_' + random + '/'
data = pd.read_csv(FILTER_DATA_DIR + 'result_dft.csv')

# 将分类变量转换为数值型
label_encoders = {}
for column in ['gpu_type', 'task_name']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# # 处理缺失值
# data_random['plan_gpu'].fillna(0, inplace=True)  # 假设没有指定gpu的plan_gpu值可以设置为0
# data_random['gpu_type'].fillna('Unknown', inplace=True)  # 用'Unknown'代表未知的gpu类型
# data_random['task_type'].fillna('CPU', inplace=True)  # 用'CPU'代表CPU型负载# 类型

# # 特征和目标
# X = data_random.drop('gpu_type', axis=1)  # 特征矩阵
# y = data_random['gpu_type']  # 目标变量
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 拆分数据集为训练集和测试集
# 转换为秒数
start = '14'
end = '21'
start_time_start = pd.Timestamp("1970-01-" + start).timestamp()
start_time_end = pd.Timestamp("1970-01-" + end).timestamp()
train_data = data[(data["start_time"] < start_time_end) & (data["start_time"] >= start_time_start)]
test_data = data[data["start_time"] >= start_time_end]

# 转换为X_train， X_test，y_train，y_test
# X_train = train_data.drop('gpu_type', axis=1)
train_columns = ['task_name', 'start_time_day', 'start_weekday', 'gpu_type', 'plan_gpu', 'run_time', 'plan_cpu']
# 排列出train_columns的所有排列组合，然后对每个组合进行随机森林

result_dir = 'randomForest/forest/' + random + '/'+start+'_'+end+'/'
for i in range(1, len(train_columns) + 1):
    for train_col in itertools.combinations(train_columns, i):
        forest(train_col)
    # 每轮进行结果检查，留下最好的几个
    # 读取文件夹下的classification_report.txt文件
    classification_reports = []
    for folder in os.listdir(result_dir):
        # 排除所有非文件夹
        if not os.path.isdir(result_dir + folder):
            continue
        # 排除之前的文件夹
        if folder.__contains__('_parameter'):
            continue
        with open(result_dir + folder + '/classification_report.txt', 'r') as f:
            classification_reports.append((folder, f.read()))
    # # 按照f1-score排序
    # classification_reports.sort(key=lambda x: float(x[1].split()[-2]))
    # # 删除f1-score最低的一半
    # for folder, _ in classification_reports[:len(classification_reports) // 2]:
    #     os.rmdir(result_dir + folder)
    # 把剩下的文件夹放进新的文件夹，名为i_parameter
    os.makedirs(result_dir + str(i) + '_parameter')
    # for folder, _ in classification_reports[len(classification_reports) // 2:]:
    for folder, _ in classification_reports:
        #         移动
        os.rename(result_dir + folder, result_dir + str(i) + '_parameter/' + folder)
    print(f'已完成{i}个参数的组合')
