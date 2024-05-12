import csv
import importlib
import math
import os
from joblib import dump

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# 计算指标
def rmse_and_mse_mae_compute(actualData, predictData, modelName):
    print(f"模型名称: {modelName}")
    mse = mean_squared_error(actualData, predictData)
    print("MSE:", mse)
    # 计算均方根误差（RMSE）
    rmse = math.sqrt(mse)
    print("RMSE:", rmse)
    # 计算r2
    r2 = r2_score(actualData, predictData)
    print("r2:", r2)
    # 计算校正决定系数R2
    ar2 = 1 - ((1 - r2) * (len(actualData) - 1)) / (len(actualData) - features - 1)
    print("校正r2:", ar2)
    # 计算平均绝对误差（MAE）
    mae = mean_absolute_error(actualData, predictData)
    print("MAE:", mae)
    # 将结果添加到新的csv文件，保存模型的评估结果，文件列名为：模型名称、RMSE、MSE、MAE、R2、校正R2
    with open('resource_estimate/result.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([modelName, rmse, mse, mae, r2, ar2])
    return [rmse, mse, mae, ar2]


# 随机森林
def new_random_forest(X_train, X_test, y_train, y_test):
    rf_model = RandomForestRegressor(n_estimators=60, random_state=42)

    # 使用训练集训练模型
    rf_model.fit(X_train, y_train)

    # 使用训练好的模型进行预测
    # y_pred = np.round(rf_model.predict(X_test)).astype(int)
    y_pred = rf_model.predict(X_test)
    rmse_and_mse_mae_compute(y_test, y_pred, "rdf")

    # 使用joblib保存模型
    dump(rf_model, 'resource_estimate/random_forest_model.joblib')


# 决策树
def new_decision_tree(X_train, X_test, y_train, y_test):
    # 创建决策树回归模型
    tree_reg = DecisionTreeRegressor()

    # 在训练集上训练模型
    tree_reg.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = tree_reg.predict(X_test)

    rmse_and_mse_mae_compute(y_test, y_pred, "dt")


# 线性回归
def new_linear_regression(X_train, X_test, y_train, y_test, scaler_x, scaler_y):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # 创建线性回归模型
    linear_reg = LinearRegression()

    # 在训练集上训练模型
    linear_reg.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = linear_reg.predict(X_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    rmse_and_mse_mae_compute(y_test_inv, y_pred_inv, "lr")


# SVR
def new_svr(X_train, X_test, y_train, y_test, scaler_x, scaler_y):
    # 将数据集分为训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42,shuffle=False)

    # 创建SVR模型
    svr = SVR(kernel='rbf')

    # 在训练集上训练模型
    svr.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = svr.predict(X_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    rmse_and_mse_mae_compute(y_test_inv, y_pred_inv, "SVR")


# 定义梯度提升树模型
def new_gradient_boosting_tree(X_train, X_test, y_train, y_test):
    # 初始化梯度提升树回归模型
    gb_regressor = GradientBoostingRegressor(random_state=42)

    # 训练模型
    gb_regressor.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = gb_regressor.predict(X_test)

    rmse_and_mse_mae_compute(y_test, y_pred, "gbt")


def new_knn(X_train, X_test, y_train, y_test):
    # 定义K近邻回归模型
    k = 3  # 设置K值
    knn = KNeighborsRegressor(n_neighbors=k)

    # 拟合模型
    knn.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = knn.predict(X_test)

    rmse_and_mse_mae_compute(y_test, y_pred, "knn")


def new_naive_bayes(X_train, X_test, y_train, y_test):
    # 创建高斯朴素贝叶斯分类器
    gnb = GaussianNB()

    # 训练模型
    gnb.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = gnb.predict(X_test)

    rmse_and_mse_mae_compute(y_test, y_pred, "naive_bayes")


def new_ridge_regression(X_train, X_test, y_train, y_test, scaler_x, scaler_y):
    # 创建岭回归模型
    ridge = Ridge()

    # 在训练集上训练模型
    ridge.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = ridge.predict(X_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    rmse_and_mse_mae_compute(y_test_inv, y_pred_inv, "ridge")


if __name__ == "__main__":
    # 打印当前路径
    os.system('pwd')
    # 获取当前模块的名称
    current_module = importlib.import_module('__main__')
    # 数据处理
    file_path = "dataset/resource_estimate/SLOA.tsv"
    df = pd.read_csv(file_path, sep='\t', header=0)

    X_train = df.iloc[:, :2]  # 前两列列作为特征
    y_train = df.iloc[:, 2]  # 后1列作为标签

    file_path = "dataset/resource_estimate/SLOValidA.tsv"
    df_val = pd.read_csv(file_path, sep='\t', header=0)
    X_test = df_val.iloc[:, :2]  # 前两列列作为特征
    y_test = df_val.iloc[:, 2]  # 后1列作为标签
    # X_test = df.iloc[:, :2]  # 前两列列作为特征
    # y_test = df.iloc[:, 2]  # 后1列作为标签

    features = 2
    with open('resource_estimate/result.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["模型名称", "RMSE", "MSE", "MAE", "R2", "校正R2"])

    new_random_forest(X_train, X_test, y_train, y_test)
    new_decision_tree(X_train, X_test, y_train, y_test)
    new_gradient_boosting_tree(X_train, X_test, y_train, y_test)
    new_knn(X_train, X_test, y_train, y_test)

    # 创建 MinMaxScaler 对象
    scaler_x = MinMaxScaler()
    # 对训练集特征进行归一化
    X_train_scaled = scaler_x.fit_transform(X_train)
    # 使用相同的缩放器对测试集特征进行归一化
    X_test_scale = scaler_x.transform(X_test)
    scaler_y = MinMaxScaler()
    y_train_scale = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1))
    y_test_scale = scaler_y.transform(np.array(y_test).reshape(-1, 1))

    new_linear_regression(X_train_scaled, X_test_scale, y_train_scale, y_test_scale, scaler_x, scaler_y)
    # 岭回归
    new_ridge_regression(X_train_scaled, X_test_scale, y_train_scale, y_test_scale, scaler_x, scaler_y)
    new_svr(X_train_scaled, X_test_scale, y_train_scale.ravel(), y_test_scale, scaler_x, scaler_y)

    # 朴素贝叶斯
    # new_naive_bayes(X_train, X_test, y_train, y_test)
