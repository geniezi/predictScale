import math

import pandas as pd
from joblib import load

modelPath = "SLOA/random_forest_model.joblib"


def get_response_time(requests, replicas):
    # a = 8.24018
    # b = -0.0003632
    # c = -47.27967

    a = 300
    b = -0.00002632
    c = 0

    response=math.floor(a * math.exp(-b * requests / replicas) + c)
    return response


def get_resource_estimators(file_path):
    # 读取模型
    # 使用joblib加载模型
    rf_model_loaded = load(modelPath)

    df = pd.read_csv(file_path)

    # 添加一列delay为500
    X_test = df.copy()
    X_test['delay'] = 500
    # 保留requests_scale和delay列
    X_test = X_test[['predict', 'delay']]
    # # 重命名列名
    X_test.rename(columns={'predict': 'requests'}, inplace=True)

    # 使用加载的模型进行预测
    y_pred_loaded = rf_model_loaded.predict(X_test)
    # 向上取整
    for i in range(len(y_pred_loaded)):
        y_pred_loaded[i] = math.ceil(y_pred_loaded[i])
    y_pred_loaded = y_pred_loaded.astype(int)
    # 添加到df中
    df['replicas'] = y_pred_loaded
    return df


data=get_resource_estimators('predict.csv')
# 保存
data.to_csv('estimate.csv', index=False)

# print(get_response_time(10000, 3))
