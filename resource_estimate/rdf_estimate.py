import os
from joblib import load
import util.osUtil as osUtil
import math

modelPath = "resource_estimate/SLOA/random_forest_model.joblib"


def get_response_time(requests, replicas):
    # a = 8.24018
    # b = -0.0003632
    # c = -47.27967

    a = 300
    b = -0.00002632
    c = 0

    response=math.floor(a * math.exp(-b * requests / replicas) + c)
    return response


def get_resource_estimators(serviceName, filePath):
    # 读取模型
    # 使用joblib加载模型
    rf_model_loaded = load(modelPath)

    df = osUtil.read_data('pageviews_by_minute.tsv')

    # 添加一列delay为500
    X_test = df.copy()
    X_test['delay'] = 500
    # 保留requests_scale和delay列
    X_test = X_test[['requests_scale', 'delay']]
    # 重命名列名
    X_test.rename(columns={'requests_scale': 'requests'}, inplace=True)

    # 使用加载的模型进行预测
    y_pred_loaded = rf_model_loaded.predict(X_test)
    return y_pred_loaded


# print(get_resource_estimators(0, 0))
print(get_response_time(10000, 3))
